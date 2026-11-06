// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! `curiomesh`: CurioMesh's automatic quad retopology and its quality check, without Blender.

use curiomesh::settings::{self, Mirror, Settings};
use curiomesh::{batch, export, io, issues, ladder, retopo};
use serde_json::json;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

const HELP: &str = "CurioMesh standalone: automatic quad retopology that keeps how the model looks, and a check for any retopology.

Commands:
  curiomesh IN -o OUT [settings]               retopologize one model
  curiomesh batch IN... -o FOLDER [--jobs N] [settings]
                                               retopologize every model given (files, or folders of them) into FOLDER,
                                               with a report per file in FOLDER/curiomesh_report.json and .csv.
                                               A file that fails is noted and the rest carry on.
  curiomesh sizes IN [-o FOLDER] [settings]    Quad Size: results at 7 detail values around --detail, from small quads to
                                               big ones, with their faces and view change; -o saves each one
  curiomesh measure SOURCE RESULT [--max-loss PERCENT] [--max-bad FACES] [--report FILE.json] [--cpu]
                                               compare RESULT (from CurioMesh or any other tool) with SOURCE: view change,
                                               bad faces, issue spots. Over a limit, it exits with code 1 (for pipelines).
  curiomesh preset list                        the built-in and saved presets
  curiomesh preset show NAME                   a preset's settings as JSON (a file --settings reads)
  curiomesh preset save NAME [settings]        save the settings given as a preset of your own
  curiomesh --help                             this text

Files: IN, SOURCE and RESULT are .obj, .gltf, .glb or .ply. OUT is .obj or .glb.

Settings (later ones win, so --use game-asset --detail 4 is Game Asset with detail 4):
  --use NAME        a preset: game-asset (6%), scan (2%), low-poly (10%, triangles), sculpt-base (6%, mirrored when
                    symmetric), animation (1.5%, mirrored when symmetric), or one you saved
  --settings FILE   settings from a JSON file (see preset show); keys left out keep their defaults
  --detail P        how much of the view may change, in percent (default 6). Lower keeps more faces
  --reduce-to N     aim for about N faces instead of a view change
  --mirror M        off (default), auto (mirror when the model is symmetric), or x, y, z: solve one half, mirror it,
                    and close the seam, so both sides are the same
  --no-rows         keep every part as reduced: don't offer even rows of quads to smooth parts
  --rows            offer them (the default)
  --baked           judge the result as it will look with a normal map baked from the source: fine bumps don't count,
                    moved outlines and shading do (scans; Scan Cleanup turns it on). --not-baked turns it off
  --pixels N        camera picture size per side (default 192)
  --cpu             make the camera pictures on the processor, not the graphics card
Writing (for game engines):
  --engine E        unreal, unity, godot, roblox or gltf: file type, triangles and triangle limit for that engine
  --format F        obj or glb (batch and sizes; for one model the OUT name decides)
  --triangulate     write triangles (quads are cut along their shorter diagonal)
  --max-tris N      cut meshes into pieces of at most N triangles (0: no limit)
  --scale S         multiply every coordinate by S
  --up z            OBJ only: write Z up instead of Y up
Other:
  --jobs N          batch: models worked on at once (default 1; at most half the processor cores)
  --report FILE     write the numbers and the issue spots as JSON
";

struct Args {
    files: Vec<PathBuf>,
    out: Option<PathBuf>,
    settings: Settings,
    report: Option<PathBuf>,
    jobs: usize,
    max_loss: Option<f64>,
    max_bad: Option<usize>,
}

fn parse(mut words: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut a = Args { files: vec![], out: None, settings: Settings::default(), report: None, jobs: 1, max_loss: None, max_bad: None };
    while let Some(w) = words.next() {
        let mut value = |name: &str| words.next().ok_or(format!("{name} needs a value"));
        let number = |s: String, name: &str| s.parse::<f64>().map_err(|_| format!("{name}: not a number: {s}"));
        let s = &mut a.settings;
        match w.as_str() {
            "-o" | "--out" => a.out = Some(value("-o")?.into()),
            "--detail" => s.detail = number(value("--detail")?, "--detail")?.clamp(ladder::LOWEST, ladder::HIGHEST),
            "--reduce-to" => s.reduce_to = Some((number(value("--reduce-to")?, "--reduce-to")? as usize).max(12)),
            "--report" => a.report = Some(value("--report")?.into()),
            "--pixels" => s.pixels = (number(value("--pixels")?, "--pixels")? as usize).clamp(16, 2048),
            "--max-loss" => a.max_loss = Some(number(value("--max-loss")?, "--max-loss")?),
            "--max-bad" => a.max_bad = Some(number(value("--max-bad")?, "--max-bad")? as usize),
            "--jobs" => a.jobs = (number(value("--jobs")?, "--jobs")? as usize).max(1),
            "--cpu" => s.gpu = false,
            "--no-rows" => s.rows = false,
            "--baked" => s.baked = true,
            "--not-baked" => s.baked = false,
            "--rows" => s.rows = true,
            "--mirror" => s.mirror = Mirror::parse(&value("--mirror")?)?,
            "--use" => *s = settings::preset(&value("--use")?)?,
            "--settings" => *s = Settings::load(&PathBuf::from(value("--settings")?))?,
            "--engine" => s.export = export::engine(&value("--engine")?)?,
            "--format" => s.export.format = export::Format::parse(&value("--format")?)?,
            "--triangulate" => s.export.triangulate = true,
            "--max-tris" => s.export.max_tris = number(value("--max-tris")?, "--max-tris")? as usize,
            "--scale" => s.export.scale = Some(number(value("--scale")?, "--scale")?).filter(|x| *x > 0.0).ok_or("--scale must be above 0")?,
            "--up" => s.export.z_up = match value("--up")?.to_ascii_lowercase().as_str() {
                "z" => true,
                "y" => false,
                other => return Err(format!("--up: {other}? Use y or z")),
            },
            "-h" | "--help" => return Err(String::new()),
            _ if w.starts_with('-') => return Err(format!("unknown option {w}")),
            _ => a.files.push(w.into()),
        }
    }
    Ok(a)
}

fn load(path: &PathBuf) -> Result<io::Mesh, String> {
    Ok(io::weld(&io::read(path)?))
}

fn write_report(path: &Option<PathBuf>, value: serde_json::Value) -> Result<(), String> {
    if let Some(path) = path {
        std::fs::write(path, serde_json::to_string_pretty(&value).unwrap()).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

fn print_check(check: &issues::Check) {
    println!("Bad faces: {} (on broken edges, flipped, duplicate, or bent/concave quads)", check.bad);
    println!("Issue spots: {}", check.spots.len());
    for (rank, s) in check.spots.iter().take(5).enumerate() {
        let kinds: Vec<String> = issues::KINDS.iter().zip(s.kinds).filter(|(_, n)| *n > 0).map(|(k, n)| format!("{n} {k}")).collect();
        println!("  {}. score {:.1} at ({:.3}, {:.3}, {:.3}): {}", rank + 1, s.score, s.center[0], s.center[1], s.center[2], kinds.join(", "));
    }
}

fn measure(a: &Args) -> Result<ExitCode, String> {
    let [_, source, result] = a.files.as_slice() else { return Err("measure needs SOURCE and RESULT".into()) };
    let (source, result) = (load(source)?, load(result)?);
    let change = retopo::measure(&source, &result, a.settings.pixels, a.settings.gpu, a.settings.baked);
    let check = issues::check(&result);
    println!("{} faces ({} quads, {} triangles) from {} source faces", check.faces, check.quads, check.tris, source.faces());
    println!("View change: {change:.2}%");
    print_check(&check);
    write_report(&a.report, json!({"source_faces": source.faces(), "loss": change, "check": check.to_json(200)}))?;
    let over = a.max_loss.is_some_and(|m| change > m) || a.max_bad.is_some_and(|m| check.bad > m);
    if over {
        println!("FAIL: over the limit");
    }
    Ok(if over { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

fn preset(a: &Args) -> Result<ExitCode, String> {
    let word = |i: usize| a.files.get(i).and_then(|p| p.to_str()).unwrap_or("");
    match word(1) {
        "list" | "" => {
            println!("Built in:");
            for p in settings::presets() {
                println!("  {:<12} {}: {}", p.key, p.label, p.words);
            }
            let saved = settings::saved_presets();
            println!("Saved ({}):", settings::presets_dir().display());
            if saved.is_empty() {
                println!("  none yet (curiomesh preset save NAME [settings])");
            }
            for name in saved {
                println!("  {name}");
            }
        }
        "show" => println!("{}", serde_json::to_string_pretty(&settings::preset(word(2))?.to_json()).unwrap()),
        "save" => {
            let path = settings::save_preset(word(2), &a.settings)?;
            println!("Saved {}", path.display());
        }
        other => return Err(format!("preset {other}? Use list, show NAME or save NAME")),
    }
    Ok(ExitCode::SUCCESS)
}

fn batch_run(a: &Args) -> Result<ExitCode, String> {
    let inputs = batch::inputs(&a.files[1..])?;
    if inputs.is_empty() {
        return Err("batch: no .obj, .gltf, .glb or .ply files found".into());
    }
    let out = a.out.clone().ok_or("batch needs -o FOLDER")?;
    let jobs = a.jobs.min(batch::max_jobs());
    println!("{} models, {} at a time, into {}", inputs.len(), jobs, out.display());
    let progress = |p: batch::Progress| match p {
        batch::Progress::Started { index, total, input } => println!("[{}/{}] {}", index + 1, total, input.display()),
        batch::Progress::Line { index, line } if jobs > 1 => println!("  [{}] {}", index + 1, line.trim()),
        batch::Progress::Line { line, .. } => println!("  {line}"),
        batch::Progress::Finished { index, total, report } => match &report.status {
            batch::Status::Ok => println!("[{}/{}] done: {} faces, {:.2}% of the view changed, {} bad faces, {:.1} s -> {}", index + 1, total, report.faces, report.loss, report.bad, report.seconds, report.output.display()),
            batch::Status::Failed(e) => println!("[{}/{}] FAILED {}: {e}", index + 1, total, report.input.display()),
            batch::Status::Cancelled => {}
        },
    };
    let reports = batch::run(&inputs, &a.settings, &out, jobs, &progress, &AtomicBool::new(false))?;
    let (json_path, csv_path) = (out.join("curiomesh_report.json"), out.join("curiomesh_report.csv"));
    batch::write_json(&reports, &a.settings, &json_path)?;
    batch::write_csv(&reports, &csv_path)?;
    let failed = reports.iter().filter(|r| r.status != batch::Status::Ok).count();
    println!("{} of {} done, {} failed. Reports: {} and {}", reports.len() - failed, reports.len(), failed, json_path.display(), csv_path.display());
    Ok(if failed > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS })
}

fn sizes(a: &Args) -> Result<ExitCode, String> {
    let [_, input] = a.files.as_slice() else { return Err("sizes needs one input file".into()) };
    let source = load(input)?;
    println!("{}: {} faces", input.display(), source.faces());
    if let Some(dir) = &a.out {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let rungs = ladder::run(&source, &a.settings, &ladder::details(a.settings.detail), |r| println!("  detail {:>6}%: {} faces, {:.2}% of the view changed", r.detail, r.faces, r.loss), |_| {}, &AtomicBool::new(false))?;
    let keep = ladder::useful(&rungs.iter().map(|r| (r.faces, r.loss)).collect::<Vec<_>>());
    println!("Quad sizes worth keeping, smallest quads first:");
    let mut rows = Vec::new();
    for (i, r) in rungs.iter().enumerate() {
        let useful = keep.contains(&i);
        let mut saved = None;
        if let (Some(dir), true) = (&a.out, useful) {
            let path = dir.join(format!("{stem}_detail{}.{}", r.detail, a.settings.export.format.ext()));
            export::write(&r.outcome.mesh, &path, &a.settings.export)?;
            saved = Some(path.display().to_string());
        }
        if useful {
            println!("  detail {:>6}%: {:>7} faces, {:.2}%{}", r.detail, r.faces, r.loss, saved.as_ref().map(|p| format!(" -> {p}")).unwrap_or_default());
        }
        rows.push(json!({"detail": r.detail, "faces": r.faces, "loss": r.loss, "useful": useful, "file": saved}));
    }
    write_report(&a.report, json!({"source_faces": source.faces(), "settings": a.settings.to_json(), "sizes": rows}))?;
    Ok(ExitCode::SUCCESS)
}

fn single(mut a: Args) -> Result<ExitCode, String> {
    let [input] = a.files.as_slice() else { return Err("give one input file (or a command: batch, sizes, measure, preset)".into()) };
    let out = a.out.clone().ok_or("-o OUT is needed")?;
    let ext = out.extension().and_then(|e| e.to_str()).unwrap_or("");
    a.settings.export.format = export::Format::parse(ext).map_err(|_| format!("{}: OUT must end in .obj or .glb", out.display()))?;
    let source = load(input)?;
    println!("{}: {} faces, {} vertices", input.display(), source.faces(), source.points.len() / 3);
    let done = settings::run(&source, &a.settings, |line| println!("{line}"))?;
    let pieces = export::write(&done.mesh, &out, &a.settings.export)?;
    let check = issues::check(&done.mesh);
    let split = if pieces > 1 { format!(" as {pieces} pieces") } else { String::new() };
    println!("{} faces ({} quads, {} triangles), {:.2}% of the view changed, {} looks, {:.1} s -> {}{split}", check.faces, check.quads, check.tris, done.loss, done.looks, done.seconds, out.display());
    if check.faces as f64 >= 0.9 * source.faces() as f64 {
        println!("Note: this kept nearly every face. A rough surface (a scan) needs most of its faces to change this little; --baked (if you will bake a normal map), --use low-poly, a higher --detail or --reduce-to reduce it.");
    }
    print_check(&check);
    let s = &a.settings;
    write_report(&a.report, json!({"source_faces": source.faces(), "faces": check.faces, "loss": done.loss, "detail": s.detail, "asked": s.reduce_to, "looks": done.looks, "seconds": done.seconds, "settings": s.to_json(), "check": check.to_json(200)}))?;
    Ok(ExitCode::SUCCESS)
}

fn run(a: Args) -> Result<ExitCode, String> {
    match a.files.first().and_then(|f| f.to_str()) {
        Some("measure") => measure(&a),
        Some("preset") | Some("presets") => preset(&a),
        Some("batch") => batch_run(&a),
        Some("sizes") => sizes(&a),
        _ => single(a),
    }
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)).and_then(run) {
        Ok(code) => code,
        Err(e) if e.is_empty() => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("curiomesh: {e}\n\nSee curiomesh --help");
            ExitCode::from(2)
        }
    }
}
