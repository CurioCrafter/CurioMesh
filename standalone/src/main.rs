// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! `curiomesh`: CurioMesh's automatic quad retopology and its quality check, without Blender.

use curiomesh::{io, issues, retopo};
use serde_json::json;
use std::path::PathBuf;
use std::process::ExitCode;

const HELP: &str = "CurioMesh standalone: automatic quad retopology that keeps how the model looks, and a check for any retopology.

Usage:
  curiomesh IN -o OUT [--detail PERCENT] [--use game-asset|scan] [--reduce-to FACES] [--report FILE.json] [--cpu]
  curiomesh measure SOURCE RESULT [--max-loss PERCENT] [--max-bad FACES] [--report FILE.json] [--cpu]

  IN, SOURCE, RESULT   .obj, .gltf, .glb or .ply.   OUT: .obj
  --detail P      how much of the view may change, in percent (default 6, as Game Asset). Lower keeps more faces.
  --use U         a preset: game-asset (detail 6), scan (detail 2)
  --reduce-to N   aim for about N faces instead of a view change
  --report F      write the numbers and the issue spots as JSON
  --pixels N      camera picture size per side (default 192, as the add-on)
  --cpu           make the camera pictures on the processor, not the graphics card
  measure         compare RESULT (from CurioMesh or any other tool) with SOURCE: view change, bad faces, issue spots.
                  With --max-loss / --max-bad it exits with code 1 when RESULT is over, for build pipelines.
";

struct Args {
    files: Vec<PathBuf>,
    out: Option<PathBuf>,
    detail: f64,
    reduce_to: Option<usize>,
    report: Option<PathBuf>,
    pixels: usize,
    gpu: bool,
    max_loss: Option<f64>,
    max_bad: Option<usize>,
}

fn parse(mut words: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut a = Args { files: vec![], out: None, detail: 6.0, reduce_to: None, report: None, pixels: 192, gpu: true, max_loss: None, max_bad: None };
    while let Some(w) = words.next() {
        let mut value = |name: &str| words.next().ok_or(format!("{name} needs a value"));
        let number = |s: String, name: &str| s.parse::<f64>().map_err(|_| format!("{name}: not a number: {s}"));
        match w.as_str() {
            "-o" | "--out" => a.out = Some(value("-o")?.into()),
            "--detail" => a.detail = number(value("--detail")?, "--detail")?,
            "--reduce-to" => a.reduce_to = Some(number(value("--reduce-to")?, "--reduce-to")? as usize),
            "--report" => a.report = Some(value("--report")?.into()),
            "--pixels" => a.pixels = (number(value("--pixels")?, "--pixels")? as usize).clamp(16, 2048),
            "--max-loss" => a.max_loss = Some(number(value("--max-loss")?, "--max-loss")?),
            "--max-bad" => a.max_bad = Some(number(value("--max-bad")?, "--max-bad")? as usize),
            "--cpu" => a.gpu = false,
            "--use" => {
                a.detail = match value("--use")?.to_ascii_lowercase().as_str() {
                    "game-asset" | "game" => 6.0,
                    "scan" | "scan-cleanup" => 2.0,
                    other => return Err(format!("--use: unknown preset {other} (game-asset, scan)")),
                }
            }
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

fn run(a: Args) -> Result<ExitCode, String> {
    if a.files.first().map(|f| f.as_os_str() == "measure").unwrap_or(false) {
        let [_, source, result] = a.files.as_slice() else { return Err("measure needs SOURCE and RESULT".into()) };
        let (source, result) = (load(source)?, load(result)?);
        let change = retopo::measure(&source, &result, a.pixels, a.gpu);
        let check = issues::check(&result);
        println!("{} faces ({} quads, {} triangles) from {} source faces", check.faces, check.quads, check.tris, source.faces());
        println!("View change: {change:.2}%");
        print_check(&check);
        write_report(&a.report, json!({"source_faces": source.faces(), "loss": change, "check": check.to_json(200)}))?;
        let over = a.max_loss.is_some_and(|m| change > m) || a.max_bad.is_some_and(|m| check.bad > m);
        if over {
            println!("FAIL: over the limit");
        }
        return Ok(if over { ExitCode::from(1) } else { ExitCode::SUCCESS });
    }
    let [input] = a.files.as_slice() else { return Err("give one input file".into()) };
    let out = a.out.clone().ok_or("-o OUT is needed")?;
    let source = load(input)?;
    println!("{}: {} faces, {} vertices", input.display(), source.faces(), source.points.len() / 3);
    let settings = retopo::Settings { detail: a.detail, reduce_to: a.reduce_to, pixels: a.pixels, gpu: a.gpu };
    let done = retopo::retopo(&source, &settings, |line| println!("{line}"))?;
    io::write_obj(&done.mesh, &out)?;
    let check = issues::check(&done.mesh);
    println!("{} faces ({} quads, {} triangles), {:.2}% of the view changed, {} looks, {:.1} s -> {}", check.faces, check.quads, check.tris, done.loss, done.looks, done.seconds, out.display());
    print_check(&check);
    write_report(&a.report, json!({"source_faces": source.faces(), "faces": check.faces, "loss": done.loss, "detail": a.detail, "asked": a.reduce_to, "looks": done.looks, "seconds": done.seconds, "check": check.to_json(200)}))?;
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match parse(std::env::args().skip(1)).and_then(run) {
        Ok(code) => code,
        Err(e) if e.is_empty() => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("curiomesh: {e}\n\n{HELP}");
            ExitCode::from(2)
        }
    }
}
