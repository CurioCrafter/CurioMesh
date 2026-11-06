// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! What the desktop app remembers between runs: the chosen settings, recent files and the first-run tour. One small
//! JSON file in the user's config folder (`CURIOMESH_PREFS` points elsewhere, for tests).

use serde_json::{json, Value};
use std::path::PathBuf;


use curiomesh::{export, settings};
use settings::Mirror;

pub const CUSTOM: &str = "custom";

/// What the settings panel offers as a use: the engine's presets, then the user's own limit. (key, label, words).
pub fn uses() -> Vec<(&'static str, &'static str, &'static str)> {
    let mut all: Vec<_> = settings::presets().into_iter().map(|p| (p.key, p.label, p.words)).collect();
    all.push((CUSTOM, "Your own limit", "You choose how much of the view may change."));
    all
}

/// Everything in the settings panel.
#[derive(Clone, PartialEq, Debug)]
pub struct Choices {
    pub preset: String, // a preset's key, or CUSTOM
    pub custom_detail: f64,
    pub aim_for_count: bool,
    pub count: usize,
    pub rows: bool,
    pub triangulate: bool, // from the preset (Low Poly writes triangles)
    pub mirror: Mirror,
    pub engine: String, // an engine's key, or "" for a plain OBJ
    pub gpu: bool,
    pub pixels: usize,
    pub keep_uvs: bool,
    pub bake_normal: bool,
    pub bake_color: bool,
    pub bake_size: usize,
}

impl Default for Choices {
    fn default() -> Self {
        Choices { preset: "game-asset".into(), custom_detail: 4.0, aim_for_count: false, count: 5000, rows: true, triangulate: false, mirror: Mirror::Off, engine: String::new(), gpu: true, pixels: 192, keep_uvs: false, bake_normal: false, bake_color: false, bake_size: 2048 }
    }
}

impl Choices {
    /// Percent of the view that may change.
    pub fn detail(&self) -> f64 {
        self.base().map_or(self.custom_detail, |s| s.detail)
    }

    fn base(&self) -> Option<settings::Settings> {
        settings::presets().into_iter().find(|p| p.key == self.preset).map(|p| p.settings)
    }

    pub fn pick(&mut self, key: &str) {
        self.preset = key.to_string();
        if let Some(s) = self.base() {
            (self.rows, self.mirror, self.triangulate) = (s.rows, s.mirror, s.export.triangulate);
        }
    }

    pub fn label(&self) -> &'static str {
        uses().into_iter().find(|u| u.0 == self.preset).map_or("Your own limit", |u| u.1)
    }

    /// `count_allowed` is false in a batch: one face count can't suit many models.
    pub fn settings(&self, count_allowed: bool) -> settings::Settings {
        let mut export = export::engine(&self.engine).unwrap_or_default();
        export.triangulate |= self.triangulate;
        settings::Settings {
            detail: self.detail(),
            reduce_to: (self.aim_for_count && count_allowed).then_some(self.count),
            pixels: self.pixels,
            gpu: self.gpu,
            rows: self.rows,
            mirror: self.mirror,
            export,
            keep_uvs: self.keep_uvs,
            bake_normal: self.bake_normal,
            bake_color: self.bake_color,
            bake_size: self.bake_size,
        }
    }
}

#[derive(Clone, Default)]
pub struct Prefs {
    pub choices: Choices,
    pub recent: Vec<PathBuf>,
    pub tour_done: bool,
}

fn file() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("CURIOMESH_PREFS") {
        return Some(p.into());
    }
    let var = |k: &str| std::env::var_os(k).map(PathBuf::from);
    let base = if cfg!(windows) {
        var("APPDATA")?
    } else if cfg!(target_vendor = "apple") {
        var("HOME")?.join("Library/Application Support")
    } else {
        var("XDG_CONFIG_HOME").or_else(|| var("HOME").map(|h| h.join(".config")))?
    };
    Some(base.join("CurioMesh").join("gui.json"))
}

const MOST_RECENT: usize = 8;

impl Prefs {
    pub fn load() -> Prefs {
        let text = file().and_then(|f| std::fs::read_to_string(f).ok()).unwrap_or_default();
        Prefs::from_json(&serde_json::from_str(&text).unwrap_or(Value::Null))
    }

    pub fn from_json(v: &Value) -> Prefs {
        let mut p = Prefs::default();
        let c = &mut p.choices;
        if let Some(key) = uses().into_iter().find(|u| v["preset"] == u.0) {
            c.preset = key.0.to_string();
        }
        c.triangulate = v["triangulate"].as_bool().unwrap_or(false);
        c.mirror = v["mirror"].as_str().and_then(|m| Mirror::parse(m).ok()).unwrap_or(Mirror::Off);
        c.engine = v["engine"].as_str().filter(|e| export::engine(e).is_ok()).unwrap_or("").to_string();
        c.custom_detail = v["custom_detail"].as_f64().unwrap_or(c.custom_detail).clamp(0.5, 15.0);
        c.aim_for_count = v["aim_for_count"].as_bool().unwrap_or(false);
        c.count = v["count"].as_u64().map_or(c.count, |n| (n as usize).clamp(24, 5_000_000));
        c.rows = v["rows"].as_bool().unwrap_or(c.rows);
        c.gpu = v["gpu"].as_bool().unwrap_or(true);
        c.pixels = v["pixels"].as_u64().map_or(192, |n| (n as usize).clamp(64, 512));
        c.keep_uvs = v["keep_uvs"].as_bool().unwrap_or(false);
        c.bake_normal = v["bake_normal"].as_bool().unwrap_or(false);
        c.bake_color = v["bake_color"].as_bool().unwrap_or(false);
        c.bake_size = v["bake_size"].as_u64().map_or(2048, |n| (n as usize).clamp(256, 8192));
        p.recent = v["recent"].as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(PathBuf::from)).take(MOST_RECENT).collect()).unwrap_or_default();
        p.tour_done = v["tour_done"].as_bool().unwrap_or(false);
        p
    }

    pub fn to_json(&self) -> Value {
        let c = &self.choices;
        json!({
            "preset": c.preset, "triangulate": c.triangulate, "mirror": c.mirror.word(), "engine": c.engine, "custom_detail": c.custom_detail, "aim_for_count": c.aim_for_count, "count": c.count,
            "rows": c.rows, "gpu": c.gpu, "pixels": c.pixels, "tour_done": self.tour_done,
            "keep_uvs": c.keep_uvs, "bake_normal": c.bake_normal, "bake_color": c.bake_color, "bake_size": c.bake_size,
            "recent": self.recent.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
        })
    }

    pub fn save(&self) {
        if let Some(f) = file() {
            if let Some(dir) = f.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(f, serde_json::to_string_pretty(&self.to_json()).unwrap());
        }
    }

    pub fn remember(&mut self, path: &std::path::Path) {
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.to_path_buf());
        self.recent.truncate(MOST_RECENT);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefs_round_trip() {
        let mut p = Prefs::default();
        p.choices.pick("low-poly");
        p.choices.engine = "godot".into();
        p.choices.pixels = 256;
        (p.choices.keep_uvs, p.choices.bake_normal, p.choices.bake_size) = (true, true, 1024);
        p.remember(std::path::Path::new("a.obj"));
        p.tour_done = true;
        let back = Prefs::from_json(&p.to_json());
        assert_eq!(back.choices, p.choices);
        assert_eq!(back.recent, p.recent);
        assert!(back.tour_done && !back.choices.rows && back.choices.detail() == 10.0 && back.choices.settings(true).export.triangulate);
        assert_eq!(Prefs::from_json(&Value::Null).choices, Choices::default()); // a missing or broken file gives the defaults
    }
}
