// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Many models with the same settings: each one read, retopologized, written and checked, a failure noted and the rest
//! carried on, with a report per file as JSON and CSV.

use crate::settings::{self, Settings};
use crate::{export, io, issues};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

pub const READABLE: [&str; 4] = ["obj", "gltf", "glb", "ply"];

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Ok,
    Failed(String),
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct FileReport {
    pub input: PathBuf,
    pub output: PathBuf,
    pub status: Status,
    pub source_faces: usize,
    pub faces: usize,
    pub quads: usize,
    pub tris: usize,
    pub loss: f64, // percent of the view changed
    pub bad: usize,
    pub spots: usize,
    pub pieces: usize,
    pub seconds: f64,
}

pub enum Progress<'a> {
    Started { index: usize, total: usize, input: &'a Path },
    Line { index: usize, line: &'a str },
    Finished { index: usize, total: usize, report: &'a FileReport },
}

/// The model files among `paths`: files as given, and the readable files directly inside each folder, sorted.
pub fn inputs(paths: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for path in paths {
        if path.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(path).map_err(|e| format!("{}: {e}", path.display()))?.flatten().map(|e| e.path())
                .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()).is_some_and(|e| READABLE.contains(&e.to_ascii_lowercase().as_str())))
                .collect();
            found.sort();
            out.extend(found);
        } else if path.is_file() {
            out.push(path.clone());
        } else {
            return Err(format!("{}: no such file or folder", path.display()));
        }
    }
    Ok(out)
}

/// The most files worked on at once: half the cores, at least one.
pub fn max_jobs() -> usize {
    (std::thread::available_parallelism().map_or(2, |n| n.get()) / 2).max(1)
}

/// Where each input's result goes: `<stem>_curiomesh.<ext>` in `out_dir`, with the input's extension in the name too
/// when two inputs share a stem.
pub fn outputs(inputs: &[PathBuf], out_dir: &Path, format: export::Format) -> Vec<PathBuf> {
    let stem = |p: &PathBuf| p.file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string();
    inputs.iter().map(|p| {
        let shared = inputs.iter().filter(|q| stem(q) == stem(p)).count() > 1;
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        let name = if shared { format!("{}_{ext}_curiomesh.{}", stem(p), format.ext()) } else { format!("{}_curiomesh.{}", stem(p), format.ext()) };
        out_dir.join(name)
    }).collect()
}

fn one(input: &Path, output: &Path, settings: &Settings, log: &mut dyn FnMut(&str)) -> Result<FileReport, String> {
    let started = std::time::Instant::now();
    let source = io::weld(&io::read(input)?);
    let done = settings::run(&source, settings, &mut *log)?;
    let pieces = settings::write(&source, &done.mesh, output, settings, log)?;
    let check = issues::check(&done.mesh);
    Ok(FileReport {
        input: input.into(), output: output.into(), status: Status::Ok, source_faces: source.faces(), faces: check.faces, quads: check.quads, tris: check.tris,
        loss: done.loss, bad: check.bad, spots: check.spots.len(), pieces, seconds: started.elapsed().as_secs_f64(),
    })
}

/// Retopologize every file in `inputs` into `out_dir` with `settings`, `jobs` files at a time (at most `max_jobs`). A
/// file that fails or panics is reported and the rest go on. Once `cancel` is set no new file starts (files already
/// running finish) and the rest are reported as cancelled. Reports come back in the order of `inputs`.
pub fn run(inputs: &[PathBuf], settings: &Settings, out_dir: &Path, jobs: usize, on_progress: &(dyn Fn(Progress) + Sync), cancel: &AtomicBool) -> Result<Vec<FileReport>, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let outs = outputs(inputs, out_dir, settings.export.format);
    let total = inputs.len();
    let reports: Mutex<Vec<Option<FileReport>>> = Mutex::new(vec![None; total]);
    let next = AtomicUsize::new(0);
    let worker = || loop {
        let index = next.fetch_add(1, Ordering::SeqCst);
        if index >= total {
            break;
        }
        let blank = |status| FileReport { input: inputs[index].clone(), output: outs[index].clone(), status, source_faces: 0, faces: 0, quads: 0, tris: 0, loss: 0.0, bad: 0, spots: 0, pieces: 0, seconds: 0.0 };
        let report = if cancel.load(Ordering::Relaxed) {
            blank(Status::Cancelled)
        } else {
            on_progress(Progress::Started { index, total, input: &inputs[index] });
            let mut log = |line: &str| on_progress(Progress::Line { index, line });
            let tried = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| one(&inputs[index], &outs[index], settings, &mut log)));
            match tried {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => blank(Status::Failed(e)),
                Err(panic) => blank(Status::Failed(format!("stopped by an internal error: {}", panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()))),
            }
        };
        on_progress(Progress::Finished { index, total, report: &report });
        reports.lock().unwrap()[index] = Some(report);
    };
    std::thread::scope(|scope| {
        for _ in 0..jobs.clamp(1, max_jobs()) {
            scope.spawn(worker);
        }
    });
    Ok(reports.into_inner().unwrap().into_iter().flatten().collect())
}

impl Status {
    pub fn word(&self) -> &str {
        match self {
            Status::Ok => "ok",
            Status::Failed(_) => "failed",
            Status::Cancelled => "cancelled",
        }
    }
}

impl FileReport {
    pub fn to_json(&self) -> Value {
        json!({
            "input": self.input.display().to_string(), "output": self.output.display().to_string(), "status": self.status.word(),
            "error": match &self.status { Status::Failed(e) => Some(e.as_str()), _ => None },
            "source_faces": self.source_faces, "faces": self.faces, "quads": self.quads, "tris": self.tris,
            "view_changed_percent": (self.loss * 100.0).round() / 100.0, "bad_faces": self.bad, "issue_spots": self.spots,
            "pieces": self.pieces, "seconds": (self.seconds * 10.0).round() / 10.0,
        })
    }
}

pub fn write_json(reports: &[FileReport], settings: &Settings, path: &Path) -> Result<(), String> {
    let value = json!({"settings": settings.to_json(), "files": reports.iter().map(FileReport::to_json).collect::<Vec<_>>()});
    std::fs::write(path, serde_json::to_string_pretty(&value).unwrap()).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn write_csv(reports: &[FileReport], path: &Path) -> Result<(), String> {
    let quote = |s: &str| format!("\"{}\"", s.replace('"', "\"\""));
    let mut text = String::from("input,output,status,error,source_faces,faces,quads,tris,view_changed_percent,bad_faces,issue_spots,pieces,seconds\n");
    for r in reports {
        let error = match &r.status { Status::Failed(e) => e.as_str(), _ => "" };
        text += &format!("{},{},{},{},{},{},{},{},{:.2},{},{},{},{:.1}\n", quote(&r.input.display().to_string()), quote(&r.output.display().to_string()), r.status.word(), quote(error), r.source_faces, r.faces, r.quads, r.tris, r.loss, r.bad, r.spots, r.pieces, r.seconds);
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bad_file_fails_alone_and_names_stay_apart() {
        let dir = std::env::temp_dir().join("curiomesh_test_batch");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("broken.obj"), "v 0 0 0\nf 1 2 3\n").unwrap();
        std::fs::write(dir.join("notes.txt"), "not a model").unwrap();
        let found = inputs(&[dir.clone()]).unwrap();
        assert_eq!(found.len(), 1);
        let out = dir.join("out");
        let reports = run(&found, &Settings::default(), &out, 1, &|_| {}, &AtomicBool::new(false)).unwrap();
        assert!(matches!(reports[0].status, Status::Failed(_)));
        write_csv(&reports, &out.join("report.csv")).unwrap();
        let names = outputs(&[PathBuf::from("a.obj"), PathBuf::from("a.ply"), PathBuf::from("b.glb")], Path::new("o"), export::Format::Glb);
        assert_eq!(names[2], Path::new("o").join("b_curiomesh.glb"));
        assert_ne!(names[0], names[1]);
    }
}
