// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! CurioMesh, the desktop app: drop a model in, pick what it is for, and compare the original with the new quads side
//! by side. The work is the same as `curiomesh` on the command line (`retopo`, `issues`).

#![cfg_attr(windows, windows_subsystem = "windows")] // no console window behind the app

mod gui_batch;
mod gui_guide;
mod gui_prefs;

use curiomesh::settings::Mirror;
use curiomesh::{export, io, issues};
use gui_batch::Batch;
use gui_prefs::{Prefs, CUSTOM};
use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Shape, Stroke, StrokeKind, TextStyle, Vec2};
use std::path::PathBuf;
use std::sync::mpsc;

// The palette, from the project's own look: slate, the amber of its cages, blue for focus, red for its edge stripe.
const SLATE: Color32 = Color32::from_rgb(30, 31, 36);
const PANEL: Color32 = Color32::from_rgb(38, 40, 46);
const LINE: Color32 = Color32::from_rgb(58, 61, 69);
const TEXT: Color32 = Color32::from_rgb(232, 233, 236);
const MUTED: Color32 = Color32::from_rgb(154, 158, 168);
const AMBER: Color32 = Color32::from_rgb(242, 163, 58);
const BLUE: Color32 = Color32::from_rgb(108, 182, 255);
const RED: Color32 = Color32::from_rgb(229, 72, 77);
const GREEN: Color32 = Color32::from_rgb(124, 201, 138);

const MOST_DRAWN: usize = 600_000; // triangles drawn per view: a bigger model is thinned out on screen, not in the work

#[derive(Clone, Copy, PartialEq)]
enum Show {
    Split,
    Original,
    Result,
}

/// What the viewport draws of one mesh: points, and triangles with the face each came from (in face order).
struct Drawable {
    points: Vec<[f32; 3]>,
    tris: Vec<[u32; 3]>,
    face_of: Vec<u32>,
}

impl Drawable {
    fn of(mesh: &io::Mesh) -> Drawable {
        let points = mesh.points.chunks_exact(3).map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        let (mut tris, mut face_of) = (Vec::new(), Vec::new());
        for f in 0..mesh.faces() {
            let face = mesh.face(f);
            for k in 1..face.len().saturating_sub(1) {
                tris.push([face[0] as u32, face[k] as u32, face[k + 1] as u32]);
                face_of.push(f as u32);
            }
        }
        Drawable { points, tris, face_of }
    }
}

struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    focus: [f32; 3],
    radius: f32,
    z_up: bool, // Blender and many CAD exports stand on Z; glTF and most game files on Y
}

impl Camera {
    fn framing(points: &[f64]) -> Camera {
        let (mut low, mut high) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in points.chunks_exact(3) {
            for k in 0..3 {
                low[k] = low[k].min(p[k]);
                high[k] = high[k].max(p[k]);
            }
        }
        let focus = [0, 1, 2].map(|k| ((low[k] + high[k]) / 2.0) as f32);
        let radius = ((0..3).map(|k| (high[k] - low[k]).powi(2)).sum::<f64>().sqrt() / 2.0).max(1e-9) as f32;
        // A guess the Upright button can undo: a model tallest or flattest along Z was most likely made Z-up.
        let extent = [0, 1, 2].map(|k| high[k] - low[k]);
        let z_up = extent[2] >= extent[0].max(extent[1]) || extent[2] < 0.5 * extent[0].min(extent[1]);
        Camera { yaw: 0.6, pitch: 0.35, zoom: 1.0, focus, radius, z_up }
    }

    /// A direction in the model turned so that its up axis points up on screen.
    fn upright(&self, v: [f32; 3]) -> [f32; 3] {
        if self.z_up { [v[0], v[2], -v[1]] } else { v }
    }

    /// Screen position and depth (larger is nearer) of a point, for a view filling `rect`. Y is up in the models.
    fn project(&self, p: [f32; 3], rect: Rect) -> (Pos2, f32) {
        let d = self.upright([p[0] - self.focus[0], p[1] - self.focus[1], p[2] - self.focus[2]]);
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let x = cy * d[0] - sy * d[2];
        let z = sy * d[0] + cy * d[2];
        let y = cp * d[1] - sp * z;
        let depth = sp * d[1] + cp * z;
        let scale = 0.45 * rect.width().min(rect.height()) * self.zoom / self.radius;
        (Pos2::new(rect.center().x + x * scale, rect.center().y - y * scale), depth)
    }
}

enum Message {
    Log(String),
    Done(Result<Finished, String>),
}

struct Finished {
    mesh: io::Mesh,
    loss: f64,
    seconds: f64,
    check: issues::Check,
}

struct Loaded {
    path: PathBuf,
    mesh: io::Mesh,
    drawable: Drawable,
}

#[derive(Clone, Copy, PartialEq)]
enum Screen {
    Home,
    Model,
    Batch,
}

struct App {
    screen: Screen,
    prefs: Prefs,
    saved: String, // the prefs as last written, to save only on a change
    model: Option<Loaded>,
    result: Option<(Finished, Drawable)>,
    running: Option<mpsc::Receiver<Message>>,
    last_line: String,
    note: Option<(String, bool)>, // (message, is an error)
    camera: Option<Camera>,
    show: Show,
    split: f32,
    picked_spot: Option<usize>,
    batch: Batch,
    guide_open: bool,
    guide_topic: usize,
    tour: Option<usize>, // the step of the first-run tour being shown
    tour_forget: bool,
    // For screenshots (CURIOMESH_SHOT=file.png): save the picture once the screen is ready, then close.
    shot: Option<PathBuf>,
    shot_asked: bool,
    frames: u32,
    batch_demo: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext) -> App {
        style(&cc.egui_ctx);
        let prefs = Prefs::load();
        let env = |k: &str| std::env::var(k).unwrap_or_default();
        let mut app = App {
            screen: Screen::Home,
            saved: prefs.to_json().to_string(),
            tour: (!prefs.tour_done).then_some(0),
            tour_forget: true,
            prefs,
            model: None,
            result: None,
            running: None,
            last_line: String::new(),
            note: None,
            camera: None,
            show: Show::Split,
            split: 0.5,
            picked_spot: None,
            batch: Batch::default(),
            guide_open: false,
            guide_topic: 0,
            shot: std::env::var_os("CURIOMESH_SHOT").map(PathBuf::from),
            shot_asked: false,
            frames: 0,
            batch_demo: false,
        };
        let files: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
        if files.len() == 1 && !files[0].is_dir() {
            app.open(files[0].clone(), &cc.egui_ctx); // a model dropped onto the program's icon
            if app.shot.is_some() {
                app.start(&cc.egui_ctx);
            }
        } else if !files.is_empty() {
            app.drop_files(files, &cc.egui_ctx);
        }
        // Switches for screenshots and tests.
        match env("CURIOMESH_SCREEN").as_str() {
            "batch" => app.screen = Screen::Batch,
            "guide" => app.guide_open = true,
            s if s.starts_with("guide:") => (app.guide_open, app.guide_topic) = (true, s[6..].parse().unwrap_or(0)),
            _ => {}
        }
        let demo: Vec<PathBuf> = env("CURIOMESH_BATCH").split(';').filter(|s| !s.is_empty()).map(PathBuf::from).collect();
        if !demo.is_empty() {
            app.screen = Screen::Batch;
            app.batch.add(demo);
            if let Some(out) = std::env::var_os("CURIOMESH_OUT") {
                app.batch.out_dir = Some(out.into());
            }
            if app.shot.is_some() {
                app.batch_demo = true;
                app.batch.start(app.prefs.choices.settings(false), &cc.egui_ctx);
            }
        }
        app
    }

    fn open(&mut self, path: PathBuf, ctx: &egui::Context) {
        if self.running.is_some() {
            return;
        }
        match io::read(&path) {
            Ok(mesh) => {
                let mesh = io::weld(&mesh);
                self.camera = Some(Camera::framing(&mesh.points));
                self.prefs.choices.count = (mesh.faces() / 8).max(100);
                self.prefs.remember(&path);
                self.model = Some(Loaded { drawable: Drawable::of(&mesh), path, mesh });
                self.result = None;
                self.picked_spot = None;
                self.note = None;
                self.show = Show::Split;
                self.screen = Screen::Model;
            }
            Err(e) => self.note = Some((format!("Couldn't open that file. {e}"), true)),
        }
        ctx.request_repaint();
    }

    /// Files dropped on the window or given on the command line: one model opens, anything more goes to the batch.
    fn drop_files(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        if self.screen != Screen::Batch && paths.len() == 1 && !paths[0].is_dir() {
            self.open(paths[0].clone(), ctx);
        } else if !self.batch.running() {
            self.batch.add(paths);
            self.screen = Screen::Batch;
        }
    }

    /// The batch table's "Open result": the original and the saved result, side by side in the viewer.
    fn open_result(&mut self, n: usize, ctx: &egui::Context) {
        let Some((path, Some(row))) = self.batch.items.get(n).map(|i| (i.path.clone(), i.row.clone())) else { return };
        self.open(path, ctx);
        if self.screen != Screen::Model {
            return;
        }
        match io::read(&row.output) {
            Ok(mesh) => {
                let drawable = Drawable::of(&mesh);
                self.result = Some((Finished { check: issues::check(&mesh), mesh, loss: row.loss, seconds: row.seconds }, drawable));
            }
            Err(e) => self.note = Some((format!("Couldn't read the result. {e}"), true)),
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        let Some(model) = &self.model else { return };
        let source = model.mesh.clone();
        let settings = self.prefs.choices.settings(true);
        let (send, receive) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let log = |line: &str| {
                let _ = send.send(Message::Log(line.trim().to_string()));
                ctx.request_repaint();
            };
            let done = gui_batch::retopo_checked(&source, &settings, log).map(|(out, check)| Finished { check, mesh: out.mesh, loss: out.loss, seconds: out.seconds });
            let _ = send.send(Message::Done(done));
            ctx.request_repaint();
        });
        self.running = Some(receive);
        self.last_line = "Reading the model".into();
        self.note = None;
        self.picked_spot = None;
    }

    fn poll(&mut self) {
        let Some(receive) = &self.running else { return };
        while let Ok(message) = receive.try_recv() {
            match message {
                Message::Log(line) => self.last_line = line,
                Message::Done(Ok(finished)) => {
                    let drawable = Drawable::of(&finished.mesh);
                    self.result = Some((finished, drawable));
                    self.running = None;
                    self.show = Show::Split;
                    return;
                }
                Message::Done(Err(e)) => {
                    self.note = Some((format!("Retopology stopped: {e}"), true));
                    self.running = None;
                    return;
                }
            }
        }
    }

    fn save(&mut self) {
        let (Some(model), Some((finished, _))) = (&self.model, &self.result) else { return };
        let stem = model.path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        let target = self.prefs.choices.settings(true).export;
        let ext = target.format.ext();
        let mut dialog = rfd::FileDialog::new().add_filter(ext, &[ext]).set_file_name(format!("{stem}_curiomesh.{ext}"));
        if let Some(folder) = model.path.parent() {
            dialog = dialog.set_directory(folder);
        }
        if let Some(path) = dialog.save_file() {
            self.note = Some(match export::write(&finished.mesh, &path, &target) {
                Ok(1) | Ok(0) => (format!("Saved {}", path.display()), false),
                Ok(n) => (format!("Saved {n} pieces next to {}", path.display()), false),
                Err(e) => (format!("Couldn't save: {e}"), true),
            });
        }
    }

    fn save_report(&mut self) {
        let (Some(model), Some((finished, _))) = (&self.model, &self.result) else { return };
        let stem = model.path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
        if let Some(path) = rfd::FileDialog::new().add_filter("JSON", &["json"]).set_file_name(format!("{stem}_issues.json")).save_file() {
            let report = serde_json::json!({"source_faces": model.mesh.faces(), "faces": finished.check.faces, "loss": finished.loss, "seconds": finished.seconds, "check": finished.check.to_json(200)});
            self.note = Some(match std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()) {
                Ok(()) => (format!("Saved {}", path.display()), false),
                Err(e) => (format!("Couldn't save: {e}"), true),
            });
        }
    }
}

fn style(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.window_fill = PANEL;
    visuals.extreme_bg_color = SLATE;
    visuals.faint_bg_color = Color32::from_rgb(44, 46, 53);
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = Color32::from_rgb(92, 66, 30);
    visuals.selection.stroke = Stroke::new(1.0, AMBER);
    visuals.hyperlink_color = BLUE;
    for w in [&mut visuals.widgets.inactive, &mut visuals.widgets.noninteractive] {
        w.bg_stroke = Stroke::new(1.0, LINE);
        w.corner_radius = CornerRadius::same(6);
    }
    visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(48, 51, 58);
    visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(58, 61, 69);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, BLUE);
    visuals.widgets.hovered.corner_radius = CornerRadius::same(6);
    visuals.widgets.active.corner_radius = CornerRadius::same(6);
    ctx.set_visuals(visuals);
    ctx.style_mut(|s| {
        s.text_styles.insert(TextStyle::Heading, FontId::proportional(22.0));
        s.text_styles.insert(TextStyle::Body, FontId::proportional(15.0));
        s.text_styles.insert(TextStyle::Button, FontId::proportional(15.0));
        s.text_styles.insert(TextStyle::Small, FontId::proportional(12.5));
        s.spacing.item_spacing = Vec2::new(8.0, 8.0);
        s.spacing.button_padding = Vec2::new(12.0, 6.0);
    });
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(14.0);
    ui.label(RichText::new(title).size(13.5).color(MUTED));
}

fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll();
        self.batch.poll();
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() {
            self.drop_files(dropped, ctx);
        }
        if self.screen == Screen::Model && self.model.is_none() {
            self.screen = Screen::Home;
        }
        self.top_bar(ctx);
        match self.screen {
            Screen::Home => self.home(ctx),
            Screen::Model => {
                self.side_panel(ctx);
                self.viewport(ctx);
            }
            Screen::Batch => {
                self.batch_panel(ctx);
                self.batch_view(ctx);
            }
        }
        let mut replay = false;
        gui_guide::guide_window(ctx, &mut self.guide_open, &mut self.guide_topic, &mut replay);
        if replay {
            (self.tour, self.screen, self.guide_open) = (Some(0), Screen::Home, false);
        }
        if let (Some(step), Screen::Home) = (self.tour, self.screen) {
            let mut step = step;
            let over = gui_guide::tour(ctx, &mut step, &mut self.tour_forget);
            self.tour = match over {
                Some(forget) => {
                    self.prefs.tour_done |= forget;
                    None
                }
                None => Some(step),
            };
        }
        let now = self.prefs.to_json().to_string();
        if now != self.saved {
            self.prefs.save();
            self.saved = now;
        }
        self.screenshot(ctx);
    }
}

const SIDE: f32 = 336.0;

fn side_frame() -> egui::Frame {
    egui::Frame::new().fill(PANEL).inner_margin(egui::Margin { left: 24, right: 18, top: 8, bottom: 16 })
}

fn big_button(ui: &mut egui::Ui, text: &str, primary: bool) -> egui::Response {
    let button = if primary {
        egui::Button::new(RichText::new(text).size(17.0).strong().color(SLATE)).fill(AMBER)
    } else {
        egui::Button::new(RichText::new(text).size(17.0).strong()).stroke(Stroke::new(1.5, AMBER))
    };
    ui.add_sized([ui.available_width(), 46.0], button.corner_radius(8.0))
}

impl App {
    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("bar").exact_height(46.0).frame(egui::Frame::new().fill(SLATE).inner_margin(egui::Margin { left: 18, right: 14, top: 0, bottom: 0 })).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                let (bar, _) = ui.allocate_exact_size(Vec2::new(5.0, 24.0), Sense::hover());
                ui.painter().rect_filled(bar, 0.0, RED); // the brand's red edge, as on the project's cover image
                ui.label(RichText::new("CurioMesh").size(19.0).strong());
                ui.add_space(18.0);
                let mut go = self.screen;
                ui.selectable_value(&mut go, Screen::Home, "Home").on_hover_text("Choose what the model is for, and open models.");
                if self.model.is_some() {
                    ui.selectable_value(&mut go, Screen::Model, "Model").on_hover_text("The model you opened: compare it with its result.");
                }
                ui.selectable_value(&mut go, Screen::Batch, "Batch").on_hover_text("Run many models, one after another.");
                self.screen = go;
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(egui::Button::new(RichText::new("Guide").strong()).stroke(Stroke::new(1.0, AMBER))).on_hover_text("A guide to every part of the app.").clicked() {
                        self.guide_open = !self.guide_open;
                    }
                });
            });
        });
    }

    fn pick(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new().add_filter("3D models", &curiomesh::batch::READABLE).pick_file() {
            self.open(path, ctx);
        }
    }

    // ------------------------------------------------------------------------------------------ the start screen

    fn home(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(SLATE).inner_margin(egui::Margin::symmetric(24, 12))).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let width = ui.available_width().min(1040.0);
                ui.horizontal_top(|ui| {
                    ui.add_space(((ui.available_width() - width) / 2.0).max(0.0));
                    ui.vertical(|ui| {
                        ui.set_width(width);
                        ui.label(RichText::new("Clean quads from heavy models").size(30.0).strong());
                        ui.label(RichText::new("Choose what your model is for, then add it. CurioMesh keeps how it looks and drops the faces it doesn't need.").color(MUTED));
                        ui.add_space(14.0);
                        ui.columns(2, |cols| {
                            cols[0].label(RichText::new("1   What is it for?").size(17.0).strong().color(AMBER));
                            cols[0].add_space(4.0);
                            let top_down = egui::Layout::top_down(egui::Align::Min);
                            cols[0].with_layout(top_down, |ui| self.settings_ui(ui, true, true, false));
                            cols[1].with_layout(top_down, |ui| self.add_models(ui, ctx));
                        });
                    });
                });
            });
        });
    }

    fn add_models(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label(RichText::new("2   Add your models").size(17.0).strong().color(AMBER));
        ui.add_space(4.0);
        if big_button(ui, "Open Model...", true).on_hover_text("Choose one OBJ, glTF, GLB or PLY file and retopologize it.").clicked() {
            self.pick(ctx);
        }
        if big_button(ui, "Batch a Folder...", false).on_hover_text("Choose a folder. Every model inside it is retopologized with these settings.").clicked() {
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                self.batch.add([dir]);
                self.screen = Screen::Batch;
            }
        }
        let (rect, response) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 74.0), Sense::hover());
        ui.painter().rect_stroke(rect, 8.0, Stroke::new(1.0, LINE), StrokeKind::Inside);
        ui.painter().text(rect.center() - Vec2::new(0.0, 9.0), Align2::CENTER_CENTER, "or drop files or a folder here", FontId::proportional(15.0), TEXT);
        ui.painter().text(rect.center() + Vec2::new(0.0, 12.0), Align2::CENTER_CENTER, "OBJ, glTF, GLB, PLY.  One file opens. Several go to a batch.", FontId::proportional(12.5), MUTED);
        response.on_hover_text("Drop files anywhere in this window.");
        if let Some((text, error)) = &self.note {
            ui.label(RichText::new(text).color(if *error { RED } else { GREEN }));
        }
        ui.add_space(6.0);
        ui.label(RichText::new("Recent").size(13.5).color(MUTED));
        if self.prefs.recent.is_empty() {
            ui.label(RichText::new("Models you open will show up here.").small().color(MUTED));
        }
        let mut open = None;
        for path in self.prefs.recent.clone() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("model").to_string();
            let folder = path.parent().map(|p| p.display().to_string()).unwrap_or_default();
            let exists = path.exists();
            ui.horizontal(|ui| {
                let r = ui.add_enabled(exists, egui::Button::new(RichText::new(&name).color(if exists { TEXT } else { MUTED })).frame(false));
                if r.on_hover_text(if exists { path.display().to_string() } else { "This file has moved or been deleted.".into() }).on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    open = Some(path.clone());
                }
                ui.add(egui::Label::new(RichText::new(folder).small().color(MUTED)).truncate());
            });
        }
        if let Some(path) = open {
            self.open(path, ctx);
        }
        ui.add_space(10.0);
        if ui.button("New here? Read the guide").on_hover_text("Open the guide to every part of the app.").clicked() {
            self.guide_open = true;
        }
    }

    // ------------------------------------------------------------------------------------------ settings, shared

    /// The settings every run uses, on the start screen, beside a model and beside a batch. `count_ok` is false in a
    /// batch, where one face count can't suit every model.
    fn settings_ui(&mut self, ui: &mut egui::Ui, enabled: bool, count_ok: bool, compact: bool) {
        let c = &mut self.prefs.choices;
        ui.add_enabled_ui(enabled, |ui| {
            if compact {
                // Beside a model the presets fold into a list, to leave room for the result.
                egui::ComboBox::from_id_salt("preset").width(ui.available_width() - 8.0).selected_text(c.label()).show_ui(ui, |ui| {
                    for (key, label, words) in gui_prefs::uses() {
                        if ui.selectable_label(c.preset == key, label).on_hover_text(words).clicked() {
                            c.pick(key);
                        }
                    }
                }).response.on_hover_text("What the model is for. This sets how much of the view may change.");
                let words = gui_prefs::uses().into_iter().find(|u| u.0 == c.preset).map_or("", |u| u.2);
                ui.add(egui::Label::new(RichText::new(words).small().color(MUTED)).wrap());
            }
            for (key, label, words) in gui_prefs::uses().into_iter().filter(|_| !compact) {
                let chosen = c.preset == key;
                let frame = egui::Frame::new()
                    .fill(if chosen { Color32::from_rgb(52, 45, 36) } else { Color32::from_rgb(44, 46, 53) })
                    .stroke(Stroke::new(if chosen { 1.5 } else { 1.0 }, if chosen { AMBER } else { LINE }))
                    .corner_radius(8.0)
                    .inner_margin(egui::Margin::symmetric(10, 6));
                let shown = frame.show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    ui.spacing_mut().item_spacing.y = 2.0;
                    ui.label(RichText::new(label).strong().color(if chosen { AMBER } else { TEXT }));
                    ui.add(egui::Label::new(RichText::new(words).small().color(MUTED)).wrap());
                });
                let response = ui.interact(shown.response.rect, ui.id().with(key), Sense::click());
                if response.on_hover_cursor(egui::CursorIcon::PointingHand).on_hover_text(format!("{words}. Click to use it.")).clicked() {
                    c.pick(key);
                }
                ui.add_space(-2.0);
            }
            if c.preset == CUSTOM {
                ui.add(egui::Slider::new(&mut c.custom_detail, 0.5..=15.0).suffix("% may change").step_by(0.5)).on_hover_text("The most the view may change. Lower keeps more faces and detail.");
            }
            section(ui, "Options");
            if count_ok {
                ui.checkbox(&mut c.aim_for_count, "Aim for a face count instead").on_hover_text("Ask for about this many faces and ignore the view-change limit. Good for a fixed budget.");
                if c.aim_for_count {
                    ui.add(egui::DragValue::new(&mut c.count).range(24..=5_000_000).speed(50.0).suffix(" faces")).on_hover_text("About how many faces the result should have.");
                }
            } else {
                ui.label(RichText::new("A fixed face count suits one model, so a batch uses the limit above.").small().color(MUTED));
            }
            ui.checkbox(&mut c.rows, "Even rows on smooth parts").on_hover_text("Smooth parts are also tried as even rows of quads, and keep them when the view still holds and no bad faces come in.");
            ui.horizontal(|ui| {
                ui.label("Measuring detail");
                egui::ComboBox::from_id_salt("pixels").selected_text(match c.pixels { 0..=160 => "Quick", 161..=224 => "Normal", _ => "Fine" }).show_ui(ui, |ui| {
                    for (name, pixels) in [("Quick", 128), ("Normal", 192), ("Fine", 256)] {
                        ui.selectable_value(&mut c.pixels, pixels, name);
                    }
                }).response.on_hover_text("How big the camera pictures are when the view change is measured. Fine sees smaller differences and takes longer.");
            });
            ui.horizontal(|ui| {
                ui.label("Mirror");
                let name = |m: Mirror| match m { Mirror::Off => "Off", Mirror::Auto => "Auto", Mirror::Axis(0) => "X", Mirror::Axis(1) => "Y", _ => "Z" };
                egui::ComboBox::from_id_salt("mirror").selected_text(name(c.mirror)).show_ui(ui, |ui| {
                    for m in [Mirror::Off, Mirror::Auto, Mirror::Axis(0), Mirror::Axis(1), Mirror::Axis(2)] {
                        ui.selectable_value(&mut c.mirror, m, name(m));
                    }
                }).response.on_hover_text("For a symmetric model: solve one half and mirror it, so both sides match. Auto looks for the plane and skips models that are not symmetric.");
            });
            ui.horizontal(|ui| {
                ui.label("Send to");
                let engines = export::engines();
                let label = engines.iter().find(|e| e.0 == c.engine).map_or("Plain OBJ", |e| e.1);
                egui::ComboBox::from_id_salt("engine").selected_text(label).show_ui(ui, |ui| {
                    ui.selectable_value(&mut c.engine, String::new(), "Plain OBJ").on_hover_text("An OBJ file with the quads as they are.");
                    for (key, label, words, _) in engines {
                        ui.selectable_value(&mut c.engine, key.to_string(), label).on_hover_text(words);
                    }
                }).response.on_hover_text("The game engine the file is for. This sets the file type, triangles and size limits.");
            });
            ui.checkbox(&mut c.gpu, "Use the graphics card").on_hover_text("Draw the camera pictures on the graphics card. Faster. Turn it off if it fails on your computer.");
        });
    }

    // ------------------------------------------------------------------------------------------ one model

    fn side_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls").exact_width(SIDE).resizable(false).frame(side_frame()).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                self.model_section(ui, ctx);
                if self.model.is_some() {
                    section(ui, "What it's for");
                    let idle = self.running.is_none();
                    self.settings_ui(ui, idle, true, true);
                    ui.add_space(12.0);
                    if idle {
                        let button = egui::Button::new(RichText::new("Retopologize").size(16.0).strong().color(SLATE)).fill(AMBER).corner_radius(8.0);
                        if ui.add_sized([ui.available_width(), 40.0], button).on_hover_text("Make the new quad mesh with these settings.").clicked() {
                            self.start(ctx);
                        }
                    } else {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.add(egui::Label::new(RichText::new(&self.last_line).color(MUTED).small()).truncate());
                        });
                    }
                    self.result_section(ui);
                }
                if let Some((text, error)) = &self.note {
                    ui.add_space(10.0);
                    ui.label(RichText::new(text).color(if *error { RED } else { GREEN }));
                }
            });
        });
    }

    fn model_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        section(ui, "Model");
        let Some(model) = &self.model else { return };
        let name = model.path.file_name().and_then(|n| n.to_str()).unwrap_or("model").to_string();
        ui.add(egui::Label::new(RichText::new(name).strong()).truncate()).on_hover_text(model.path.display().to_string());
        ui.label(RichText::new(format!("{} faces, {} points", thousands(model.mesh.faces()), thousands(model.mesh.points.len() / 3))).color(MUTED).small());
        if ui.add_enabled(self.running.is_none(), egui::Button::new("Open another model")).on_hover_text("Choose a different file. You can also drop one on the window.").clicked() {
            self.pick(ctx);
        }
    }

    // ------------------------------------------------------------------------------------------ a batch

    fn batch_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("batch_controls").exact_width(SIDE).resizable(false).frame(side_frame()).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                section(ui, "Settings for every model");
                let idle = !self.batch.running();
                self.settings_ui(ui, idle, false, true);
                ui.add_space(12.0);
                if idle {
                    let n = self.batch.items.len();
                    let ready = n > 0 && self.batch.out_dir.is_some();
                    let button = egui::Button::new(RichText::new(format!("Run batch ({n})")).size(16.0).strong().color(SLATE)).fill(AMBER).corner_radius(8.0);
                    let tip = if ready { "Retopologize every model in the list, one after another." } else { "Add models and choose an output folder first." };
                    let response = ui.add_enabled_ui(ready, |ui| ui.add_sized([ui.available_width(), 40.0], button)).inner.on_hover_text(tip).on_disabled_hover_text(tip);
                    if response.clicked() {
                        let settings = self.prefs.choices.settings(false);
                        self.batch.start(settings, ctx);
                    }
                } else {
                    let button = egui::Button::new(RichText::new("Cancel").size(16.0).strong()).stroke(Stroke::new(1.5, RED)).corner_radius(8.0);
                    if ui.add_sized([ui.available_width(), 40.0], button).on_hover_text("Stop after the model in progress. Finished results stay saved.").clicked() {
                        self.batch.cancel();
                    }
                }
            });
        });
    }

    fn batch_view(&mut self, ctx: &egui::Context) {
        let mut action = None;
        egui::CentralPanel::default().frame(egui::Frame::new().fill(SLATE).inner_margin(egui::Margin::symmetric(20, 14))).show(ctx, |ui| {
            action = self.batch.ui(ui);
        });
        match action {
            Some(gui_batch::Action::OpenResult(n)) => self.open_result(n, ctx),
            Some(gui_batch::Action::ShowInFolder(path)) => gui_batch::show_in_folder(&path),
            None => {}
        }
    }

    fn result_section(&mut self, ui: &mut egui::Ui) {
        let Some((finished, _)) = &self.result else { return };
        let source_faces = self.model.as_ref().map_or(0, |m| m.mesh.faces());
        let limit = self.prefs.choices.detail();
        let (quads, faces, loss, bad, seconds, spots) = (finished.check.quads, finished.check.faces, finished.loss, finished.check.bad, finished.seconds, finished.check.spots.len());
        let rows: Vec<(String, f64)> = finished.check.spots.iter().take(100).map(|s| {
            let kinds: Vec<String> = issues::KINDS.iter().zip(s.kinds).filter(|(_, n)| *n > 0).map(|(k, n)| format!("{n} {}", plural(k, n))).collect();
            (kinds.join(", "), s.score)
        }).collect();
        section(ui, "Result");
        ui.label(RichText::new(format!("{} {}", thousands(faces), if quads == faces { "quads" } else { "faces" })).size(28.0).strong());
        ui.label(RichText::new(format!("from {} faces, {:.0}× fewer", thousands(source_faces), source_faces as f64 / faces.max(1) as f64)).color(MUTED));
        ui.add_space(6.0);
        egui::Grid::new("numbers").num_columns(2).spacing([16.0, 6.0]).show(ui, |ui| {
            ui.label(RichText::new("View change").color(MUTED));
            let fits = self.prefs.choices.aim_for_count || loss <= limit * 1.1;
            ui.label(RichText::new(format!("{loss:.1}%")).color(if fits { GREEN } else { AMBER })).on_hover_text("Share of the model's outline, depth and surface direction that differs, seen from six sides.");
            ui.end_row();
            ui.label(RichText::new("Bad faces").color(MUTED));
            ui.label(RichText::new(bad.to_string()).color(if bad == 0 { GREEN } else { RED })).on_hover_text("Faces on broken edges, flipped, duplicated, or quads bent past 60 degrees or concave.");
            ui.end_row();
            ui.label(RichText::new("Time").color(MUTED));
            ui.label(format!("{seconds:.1} s"));
            ui.end_row();
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new(RichText::new("Save model").strong()).stroke(Stroke::new(1.0, AMBER))).on_hover_text("Save the new mesh as a file for the engine you chose (OBJ, or GLB for Unreal, Godot and glTF).").clicked() {
                self.save();
            }
            if ui.button("Save issue report").on_hover_text("Save the numbers and the list of spots as a JSON file.").clicked() {
                self.save_report();
            }
        });
        section(ui, if spots == 0 { "Nothing to look at".to_string() } else { format!("Spots to look at ({spots})") }.as_str());
        if spots == 0 {
            ui.label(RichText::new("No poles, slivers or folds were found.").small().color(MUTED));
            return;
        }
        ui.label(RichText::new("Worst first. Click one to see it.").small().color(MUTED));
        for (i, (kinds, score)) in rows.iter().enumerate() {
            let picked = self.picked_spot == Some(i);
            let text = RichText::new(format!("{}.  {kinds}", i + 1)).color(if picked { AMBER } else { TEXT });
            if ui.selectable_label(picked, text).on_hover_text(format!("Score {score:.1}: the heavier its kinds and the more faces, the higher")).clicked() {
                self.picked_spot = if picked { None } else { Some(i) };
                self.focus_spot();
            }
        }
    }

    fn focus_spot(&mut self) {
        let (Some(i), Some((finished, _)), Some(camera)) = (self.picked_spot, &self.result, &mut self.camera) else { return };
        let spot = &finished.check.spots[i];
        camera.focus = spot.center.map(|x| x as f32);
        camera.zoom = (camera.radius / (spot.radius as f32 * 4.0).max(camera.radius * 0.02)).clamp(1.0, 40.0);
        self.show = Show::Result;
    }

    fn viewport(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(SLATE)).show(ctx, |ui| {
            let rect = ui.max_rect();
            let response = ui.allocate_rect(rect, Sense::click_and_drag());
            let Some(model) = &self.model else {
                ui.painter().text(rect.center() - Vec2::new(0.0, 14.0), Align2::CENTER_CENTER, "Open a model from Home", FontId::proportional(24.0), TEXT);
                ui.painter().text(rect.center() + Vec2::new(0.0, 18.0), Align2::CENTER_CENTER, "CurioMesh keeps how it looks and gives you clean quads", FontId::proportional(15.0), MUTED);
                return;
            };
            let camera = self.camera.as_mut().unwrap();
            let split_x = rect.left() + self.split * rect.width();
            let handle = Rect::from_center_size(Pos2::new(split_x, rect.center().y), Vec2::new(16.0, rect.height()));
            let on_handle = self.result.is_some() && self.show == Show::Split && response.hover_pos().is_some_and(|p| handle.contains(p));
            if response.dragged() {
                let started_on_handle = self.result.is_some() && self.show == Show::Split && ui.input(|i| i.pointer.press_origin()).is_some_and(|p| handle.contains(p));
                if started_on_handle {
                    self.split = ((response.interact_pointer_pos().map_or(split_x, |p| p.x) - rect.left()) / rect.width()).clamp(0.02, 0.98);
                } else {
                    let d = response.drag_delta();
                    camera.yaw += d.x * 0.008;
                    camera.pitch = (camera.pitch + d.y * 0.008).clamp(-1.5, 1.5);
                }
            }
            if response.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    camera.zoom = (camera.zoom * (scroll * 0.0025).exp()).clamp(0.2, 60.0);
                }
            }
            if response.double_clicked() {
                *camera = Camera::framing(&model.mesh.points);
                self.picked_spot = None;
            }
            if on_handle {
                ctx.set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
            }
            let camera = self.camera.as_ref().unwrap();
            let highlight: Option<&[usize]> = self.picked_spot.and_then(|i| self.result.as_ref().map(|(f, _)| f.check.spots[i].faces.as_slice()));
            let painter = ui.painter_at(rect);
            match (&self.result, self.show) {
                (Some((_, result)), Show::Split) => {
                    painter.with_clip_rect(Rect::from_min_max(rect.min, Pos2::new(split_x, rect.max.y))).add(draw(&model.drawable, camera, rect, None, None));
                    painter.with_clip_rect(Rect::from_min_max(Pos2::new(split_x, rect.min.y), rect.max)).add(draw(result, camera, rect, Some(AMBER), highlight));
                    painter.line_segment([Pos2::new(split_x, rect.top()), Pos2::new(split_x, rect.bottom())], Stroke::new(2.0, TEXT));
                    painter.circle(Pos2::new(split_x, rect.center().y), 13.0, SLATE, Stroke::new(2.0, TEXT));
                    painter.text(Pos2::new(split_x, rect.center().y), Align2::CENTER_CENTER, "↔", FontId::proportional(14.0), TEXT);
                    painter.text(Pos2::new(split_x - 14.0, rect.top() + 18.0), Align2::RIGHT_TOP, "Original", FontId::proportional(14.0), MUTED);
                    painter.text(Pos2::new(split_x + 14.0, rect.top() + 18.0), Align2::LEFT_TOP, "CurioMesh", FontId::proportional(14.0), AMBER);
                }
                (Some((_, result)), Show::Result) => {
                    painter.add(draw(result, camera, rect, Some(AMBER), highlight));
                }
                _ => {
                    painter.add(draw(&model.drawable, camera, rect, None, None));
                }
            }
            painter.text(rect.left_bottom() + Vec2::new(18.0, -16.0), Align2::LEFT_BOTTOM, "Drag to turn, scroll to zoom, double-click to frame the model", FontId::proportional(12.5), MUTED);
            let mut show = self.show;
            let mut turn = false;
            let has_result = self.result.is_some();
            egui::Area::new(egui::Id::new("show")).anchor(Align2::RIGHT_TOP, Vec2::new(-14.0, 60.0)).show(ctx, |ui| {
                egui::Frame::new().fill(PANEL).corner_radius(8.0).inner_margin(4.0).stroke(Stroke::new(1.0, LINE)).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if has_result {
                            ui.selectable_value(&mut show, Show::Split, "Side by side").on_hover_text("Original on the left, new quads on the right. Drag the line to compare.");
                            ui.selectable_value(&mut show, Show::Original, "Original").on_hover_text("Only the original model.");
                            ui.selectable_value(&mut show, Show::Result, "Result").on_hover_text("Only the new quads.");
                            ui.separator();
                        }
                        turn = ui.button("Upright").on_hover_text("Stand the model on its other axis, if it lies on its side").clicked();
                    });
                });
            });
            self.show = show;
            if turn {
                if let Some(camera) = &mut self.camera {
                    camera.z_up = !camera.z_up;
                }
            }
        });
    }

    fn screenshot(&mut self, ctx: &egui::Context) {
        let Some(path) = self.shot.clone() else { return };
        self.frames += 1;
        let batch_done = !self.batch_demo || (!self.batch.running() && self.batch.done_count() == self.batch.items.len());
        let ready = self.frames > 12 && batch_done && (self.screen != Screen::Model || self.result.is_some());
        if !ready {
            ctx.request_repaint();
            return;
        }
        if !self.shot_asked {
            self.shot_asked = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            return;
        }
        let image = ctx.input(|i| i.events.iter().find_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None }));
        if let Some(image) = image {
            let rgba: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
            let _ = image::save_buffer(&path, &rgba, image.size[0] as u32, image.size[1] as u32, image::ColorType::Rgba8);
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint();
    }
}

fn plural(kind: &str, n: usize) -> String {
    let word = match kind {
        "broken" => "broken face",
        "bent" => "bent quad",
        "sliver" => "sliver",
        "pole" => "face at a pole",
        other => other,
    };
    if n == 1 { word.to_string() } else if word == "face at a pole" { "faces at poles".to_string() } else { format!("{word}s") }
}

/// One mesh as flat-shaded triangles, back to front so nearer faces cover farther ones, with its face outlines drawn
/// along with each face when `wire` is given. Faces in `highlight` are filled red.
fn draw(d: &Drawable, camera: &Camera, rect: Rect, wire: Option<Color32>, highlight: Option<&[usize]>) -> Shape {
    let screen: Vec<(Pos2, f32)> = d.points.iter().map(|&p| camera.project(p, rect)).collect();
    let step = d.tris.len().div_ceil(MOST_DRAWN).max(1);
    let marked: std::collections::HashSet<u32> = highlight.map(|h| h.iter().map(|&f| f as u32).collect()).unwrap_or_default();
    // Farthest first. Back faces are kept: an open model shows its inside, and nearer faces cover them anyway.
    let mut order: Vec<(f32, usize)> = (0..d.tris.len()).step_by(step).map(|t| {
        let [a, b, c] = d.tris[t].map(|v| screen[v as usize].1);
        ((a + b + c) / 3.0, t)
    }).collect();
    order.sort_by(|x, y| x.0.total_cmp(&y.0));
    let light = {
        let l = [0.35f32, 0.75, 0.55];
        let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        [l[0] / n, l[1] / n, l[2] / n]
    };
    let (sy, cy) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();
    let to_view = |v: [f32; 3]| {
        let x = cy * v[0] - sy * v[2];
        let z = sy * v[0] + cy * v[2];
        [x, cp * v[1] - sp * z, sp * v[1] + cp * z]
    };
    let base = if wire.is_some() { [118.0f32, 124.0, 138.0] } else { [158.0, 162.0, 170.0] };
    let mut mesh = egui::Mesh::default();
    let wire_width = 1.1f32;
    let mut done_face = u32::MAX;
    for (_, t) in order {
        let [a, b, c] = d.tris[t];
        let (pa, pb, pc) = (d.points[a as usize], d.points[b as usize], d.points[c as usize]);
        let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
        let nv = to_view(camera.upright([n[0] / length, n[1] / length, n[2] / length]));
        let shade = 0.28 + 0.72 * (nv[0] * light[0] + nv[1] * light[1] + nv[2] * light[2]).abs();
        let face = d.face_of[t];
        let color = if marked.contains(&face) {
            Color32::from_rgb(200, 62, 66)
        } else {
            Color32::from_rgb((base[0] * shade) as u8, (base[1] * shade) as u8, (base[2] * shade) as u8)
        };
        let i = mesh.vertices.len() as u32;
        for v in [a, b, c] {
            mesh.colored_vertex(screen[v as usize].0, color);
        }
        mesh.add_triangle(i, i + 1, i + 2);
        // The face's outline right after its fill, so faces in front cover the lines behind them.
        if let Some(color) = wire {
            if face != done_face {
                done_face = face;
                let first = d.face_of.partition_point(|&f| f < face);
                let mut corners: Vec<u32> = Vec::new();
                for k in first..d.tris.len() {
                    if d.face_of[k] != face {
                        break;
                    }
                    if corners.is_empty() {
                        corners.push(d.tris[k][0]);
                        corners.push(d.tris[k][1]);
                    }
                    corners.push(d.tris[k][2]);
                }
                for k in 0..corners.len() {
                    let (p, q) = (screen[corners[k] as usize].0, screen[corners[(k + 1) % corners.len()] as usize].0);
                    let dir = (q - p).normalized().rot90() * wire_width * 0.5;
                    let j = mesh.vertices.len() as u32;
                    for v in [p + dir, p - dir, q - dir, q + dir] {
                        mesh.colored_vertex(v, color);
                    }
                    mesh.add_triangle(j, j + 1, j + 2);
                    mesh.add_triangle(j, j + 2, j + 3);
                }
            }
        }
    }
    Shape::mesh(mesh)
}

fn main() -> eframe::Result {
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../assets/curiomesh.png")).ok();
    // CURIOMESH_SIZE=1280x720 opens at that size, to check the layout.
    let size = std::env::var("CURIOMESH_SIZE").ok().and_then(|s| s.split_once('x').and_then(|(w, h)| Some([w.parse().ok()?, h.parse().ok()?]))).unwrap_or([1360.0, 840.0]);
    let mut viewport = egui::ViewportBuilder::default().with_title("CurioMesh").with_inner_size(size).with_min_inner_size([900.0, 560.0]);
    if let Some(icon) = icon {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions { viewport, ..Default::default() };
    eframe::run_native("CurioMesh", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
