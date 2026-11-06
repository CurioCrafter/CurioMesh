// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! The batch screen: a list of models, run one after another on a worker thread, with a results table.

use super::{thousands, GREEN, MUTED, RED, TEXT};
use curiomesh::batch::{self, FileReport, Progress};
use curiomesh::{issues, retopo, settings};
use eframe::egui::{self, RichText};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

pub fn is_model(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| batch::READABLE.contains(&e.to_ascii_lowercase().as_str()))
}

// ------------------------------------------------------------------------------------------------------ ADAPTER
// Where the app calls the engine: `retopo_checked` for one model, `batch::run` in `Batch::start` for many.

/// Retopologize one loaded model (mirroring when asked) and check the result.
pub fn retopo_checked(source: &curiomesh::io::Mesh, settings: &settings::Settings, log: impl FnMut(&str)) -> Result<(retopo::Outcome, issues::Check), String> {
    let out = settings::run(source, settings, log)?;
    let check = issues::check(&out.mesh);
    Ok((out, check))
}

// ------------------------------------------------------------------------------------------------------ state

#[derive(Clone, PartialEq)]
pub enum Status {
    Queued,
    Running,
    Done,
    Failed(String),
    Cancelled,
}

pub struct Item {
    pub path: PathBuf,
    pub status: Status,
    pub steps: usize, // log lines seen: the search gives no percentage, so the bar fills as it reports
    pub line: String,
    pub row: Option<FileReport>,
}

impl Item {
    fn new(path: PathBuf) -> Item {
        Item { path, status: Status::Queued, steps: 0, line: String::new(), row: None }
    }
}

enum Message {
    Started(usize),
    Line(usize, String),
    Finished(usize, FileReport),
    End,
}

#[derive(Default)]
pub struct Batch {
    pub items: Vec<Item>,
    pub out_dir: Option<PathBuf>,
    run: Option<mpsc::Receiver<Message>>,
    cancel: Arc<AtomicBool>,
    started: Option<Instant>,
    pub elapsed: f64,
    pub note: Option<(String, bool)>,
}

/// What the table asks the app to do.
pub enum Action {
    OpenResult(usize),
    ShowInFolder(PathBuf),
}

impl Batch {
    pub fn running(&self) -> bool {
        self.run.is_some()
    }

    /// Adds files, and the models in dropped or chosen folders (one level deep). Returns how many were new.
    pub fn add(&mut self, paths: impl IntoIterator<Item = PathBuf>) -> usize {
        if self.running() {
            return 0;
        }
        let mut found = Vec::new();
        for p in paths {
            if p.is_dir() {
                found.extend(batch::inputs(&[p]).unwrap_or_default());
            } else {
                found.push(p);
            }
        }
        let mut added = 0;
        for p in found.into_iter().filter(|p| is_model(p)) {
            if !self.items.iter().any(|i| i.path == p) {
                self.items.push(Item::new(p));
                added += 1;
            }
        }
        if self.out_dir.is_none() {
            self.out_dir = self.items.first().and_then(|i| i.path.parent()).map(|d| d.join("curiomesh_out"));
        }
        self.note = (added == 0).then(|| ("No new OBJ, glTF, GLB or PLY models there.".to_string(), true));
        added
    }

    pub fn start(&mut self, settings: settings::Settings, ctx: &egui::Context) {
        let Some(out_dir) = self.out_dir.clone() else { return };
        if self.items.is_empty() {
            return;
        }
        for item in &mut self.items {
            *item = Item::new(std::mem::take(&mut item.path)); // every run starts the whole list again
        }
        let paths: Vec<PathBuf> = self.items.iter().map(|i| i.path.clone()).collect();
        self.cancel = Arc::new(AtomicBool::new(false));
        let (cancel, ctx) = (self.cancel.clone(), ctx.clone());
        let (send, receive) = mpsc::channel();
        std::thread::spawn(move || {
            let on_progress = |p: Progress| {
                let _ = send.send(match p {
                    Progress::Started { index, .. } => Message::Started(index),
                    Progress::Line { index, line } => Message::Line(index, line.trim().to_string()),
                    Progress::Finished { index, report, .. } => Message::Finished(index, report.clone()),
                });
                ctx.request_repaint();
            };
            // One file at a time, so the PC stays usable. The engine skips the rest once `cancel` is set.
            if let Err(e) = batch::run(&paths, &settings, &out_dir, 1, &on_progress, &cancel) {
                let _ = send.send(Message::Line(0, e));
            }
            let _ = send.send(Message::End);
            ctx.request_repaint();
        });
        self.run = Some(receive);
        self.started = Some(Instant::now());
        self.elapsed = 0.0;
        self.note = None;
    }

    pub fn cancel(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.note = Some(("Stopping after the model in progress.".into(), false));
    }

    pub fn poll(&mut self) {
        let Some(receive) = &self.run else { return };
        if let Some(t) = self.started {
            self.elapsed = t.elapsed().as_secs_f64();
        }
        let mut finished = false;
        while let Ok(m) = receive.try_recv() {
            match m {
                Message::Started(i) => self.items[i].status = Status::Running,
                Message::Line(i, line) => {
                    self.items[i].steps += 1;
                    self.items[i].line = line;
                }
                Message::Finished(i, report) => {
                    self.items[i].status = match &report.status {
                        batch::Status::Ok => Status::Done,
                        batch::Status::Failed(e) => Status::Failed(e.clone()),
                        batch::Status::Cancelled => Status::Cancelled,
                    };
                    self.items[i].row = (report.status == batch::Status::Ok).then_some(report);
                }
                Message::End => finished = true,
            }
        }
        if finished {
            self.run = None;
            let stopped = self.cancel.load(Ordering::Relaxed);
            let ok = self.items.iter().filter(|i| i.status == Status::Done).count();
            let bad = self.items.iter().filter(|i| matches!(i.status, Status::Failed(_))).count();
            let first = self.items.iter().find_map(|i| if let Status::Failed(e) = &i.status { Some(format!(" First problem: {e}")) } else { None }).unwrap_or_default();
            self.note = Some((format!("{} {ok} done, {bad} failed, in {:.0} s.{first}", if stopped { "Stopped:" } else { "Finished:" }, self.elapsed), bad > 0));
        }
    }

    /// Models that are no longer waiting or working.
    pub fn done_count(&self) -> usize {
        self.items.iter().filter(|i| !matches!(i.status, Status::Queued | Status::Running)).count()
    }

    /// The reports of the finished models, in list order, for the CSV.
    pub fn reports(&self) -> Vec<FileReport> {
        self.items.iter().filter_map(|i| i.row.clone()).collect()
    }
}

// ------------------------------------------------------------------------------------------------------ the screen

impl Batch {
    /// The main area of the batch screen. Returns what the user asked of the app.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        let idle = !self.running();
        let mut action = None;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Batch").heading().strong());
            ui.add_space(12.0);
            ui.add_enabled_ui(idle, |ui| {
                if ui.button("Add files...").on_hover_text("Choose OBJ, glTF, GLB or PLY files to add to the list.").clicked() {
                    if let Some(files) = rfd::FileDialog::new().add_filter("3D models", &batch::READABLE).pick_files() {
                        self.add(files);
                    }
                }
                if ui.button("Add folder...").on_hover_text("Add every model inside a folder (not its subfolders).").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.add([dir]);
                    }
                }
                if ui.button("Clear list").on_hover_text("Remove all models from the list.").clicked() {
                    self.items.clear();
                    self.note = None;
                }
            });
        });
        ui.horizontal(|ui| {
            ui.label(RichText::new("Save results in").color(MUTED));
            let text = self.out_dir.as_ref().map_or("not chosen yet".to_string(), |d| d.display().to_string());
            ui.add_enabled_ui(idle, |ui| {
                if ui.button("Choose...").on_hover_text("Pick the folder the new files go into.").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.out_dir = Some(dir);
                    }
                }
            });
            ui.add(egui::Label::new(RichText::new(text).color(TEXT)).truncate()).on_hover_text("Every result is saved here as NAME_curiomesh.obj (or .glb for a glTF engine).");
        });
        let total = self.items.len();
        if total > 0 {
            let done = self.done_count();
            ui.horizontal(|ui| {
                let part = self.items.iter().find(|i| i.status == Status::Running).map_or(0.0, |i| fraction(i.steps));
                let all = (done as f32 + part) / total as f32;
                let export = ui.add_enabled(done > 0 && idle, egui::Button::new("Export table (CSV)")).on_hover_text("Save the results table as a CSV file for a spreadsheet.");
                if export.clicked() {
                    if let Some(path) = rfd::FileDialog::new().add_filter("CSV", &["csv"]).set_file_name("curiomesh_batch.csv").save_file() {
                        self.note = Some(match batch::write_csv(&self.reports(), &path) {
                            Ok(()) => (format!("Saved {}", path.display()), false),
                            Err(e) => (format!("Couldn't save: {e}"), true),
                        });
                    }
                }
                ui.label(RichText::new(if self.elapsed > 0.0 { format!("{:.0} s", self.elapsed) } else { String::new() }).color(MUTED));
                ui.add(egui::ProgressBar::new(all).desired_width(ui.available_width()).text(format!("{done} of {total} models")).animate(!idle)).on_hover_text("Overall progress.");
            });
        }
        if let Some((text, error)) = &self.note {
            ui.add(egui::Label::new(RichText::new(text).color(if *error { RED } else { GREEN })).wrap());
        }
        ui.separator();
        if total == 0 {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("No models yet").size(20.0).strong());
                ui.label(RichText::new("Add files or a folder, or drop them anywhere in this window.").color(MUTED));
            });
            return None;
        }
        let mut remove = None;
        egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
            egui::Grid::new("batch_table").num_columns(8).spacing([14.0, 8.0]).striped(true).show(ui, |ui| {
                for (head, tip) in [
                    ("Model", "The file you added."),
                    ("Progress", "Waiting, working, done or failed."),
                    ("Faces before", "Faces in the original, after welding duplicate points."),
                    ("Faces after", "Faces in the new mesh."),
                    ("View change", "How much of the view differs from the original. Lower is closer."),
                    ("Bad faces", "Broken, flipped, duplicate or badly bent faces. Should be 0."),
                    ("Time", "Seconds the model took."),
                    ("", ""),
                ] {
                    ui.label(RichText::new(head).color(MUTED).small()).on_hover_text(tip);
                }
                ui.end_row();
                for (n, item) in self.items.iter().enumerate() {
                    let name = item.path.file_name().and_then(|s| s.to_str()).unwrap_or("model");
                    ui.allocate_ui_with_layout(egui::vec2(170.0, 20.0), egui::Layout::left_to_right(egui::Align::Center), |ui| ui.add(egui::Label::new(name).truncate())).inner.on_hover_text(item.path.display().to_string());
                    let (fill, text) = match &item.status {
                        Status::Queued => (0.0, "Waiting"),
                        Status::Running => (fraction(item.steps), "Working"),
                        Status::Done => (1.0, "Done"),
                        Status::Failed(_) => (1.0, "Failed"),
                        Status::Cancelled => (0.0, "Stopped"),
                    };
                    let bar = egui::ProgressBar::new(fill).desired_width(100.0).text(text).animate(item.status == Status::Running);
                    let bar = if matches!(item.status, Status::Failed(_)) { bar.fill(RED) } else { bar };
                    let r = ui.add(bar);
                    match &item.status {
                        Status::Failed(e) => drop(r.on_hover_text(e)),
                        Status::Running => drop(r.on_hover_text(&item.line)),
                        _ => {}
                    }
                    match &item.row {
                        Some(r) => {
                            ui.label(thousands(r.source_faces));
                            ui.label(thousands(r.faces));
                            ui.label(format!("{:.1}%", r.loss));
                            ui.label(RichText::new(r.bad.to_string()).color(if r.bad == 0 { GREEN } else { RED }));
                            ui.label(format!("{:.1} s", r.seconds));
                            ui.horizontal(|ui| {
                                if ui.button("Open result").on_hover_text("Compare the original and the result in the viewer.").clicked() {
                                    action = Some(Action::OpenResult(n));
                                }
                                if ui.button("Show in folder").on_hover_text("Show the saved file in your file browser.").clicked() {
                                    action = Some(Action::ShowInFolder(r.output.clone()));
                                }
                            });
                        }
                        None => {
                            for _ in 0..5 {
                                ui.label("");
                            }
                            if !matches!(item.status, Status::Failed(_)) && idle && ui.small_button("Remove").on_hover_text("Take this model off the list.").clicked() {
                                remove = Some(n);
                            }
                        }
                    }
                    ui.end_row();
                }
            });
        });
        if let Some(n) = remove {
            self.items.remove(n);
        }
        action
    }
}

/// A bar position for a model in progress: fills fast at first, never reaches the end until it is done.
fn fraction(steps: usize) -> f32 {
    0.05 + 0.9 * (1.0 - (-(steps as f32) / 5.0).exp())
}

pub fn show_in_folder(path: &Path) {
    let _ = if cfg!(windows) {
        std::process::Command::new("explorer").arg(format!("/select,{}", path.display())).spawn()
    } else if cfg!(target_vendor = "apple") {
        std::process::Command::new("open").arg("-R").arg(path).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(path.parent().unwrap_or(path)).spawn()
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_model_files_count() {
        assert!(is_model(Path::new("x.GLB")) && is_model(Path::new("x.obj")) && !is_model(Path::new("x.png")));
    }
}
