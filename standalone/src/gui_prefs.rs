// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! What the desktop app remembers between runs: the chosen settings, recent files and the first-run tour. One small
//! JSON file in the user's config folder (`CURIOMESH_PREFS` points elsewhere, for tests).

use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Purpose {
    Game,
    Scan,
    LowPoly,
    Sculpt,
    Custom,
}

impl Purpose {
    pub const ALL: [Purpose; 5] = [Purpose::Game, Purpose::Scan, Purpose::LowPoly, Purpose::Sculpt, Purpose::Custom];

    pub fn title(self) -> &'static str {
        match self {
            Purpose::Game => "Game asset",
            Purpose::Scan => "Scan cleanup",
            Purpose::LowPoly => "Low poly",
            Purpose::Sculpt => "Sculpt base",
            Purpose::Custom => "Your own limit",
        }
    }

    pub fn blurb(self) -> &'static str {
        match self {
            Purpose::Game => "Few faces for real-time use. Up to 6% of the view may change.",
            Purpose::Scan => "Keeps small detail from a scan. Up to 2% may change.",
            Purpose::LowPoly => "As few big quads as possible. Up to 12% may change.",
            Purpose::Sculpt => "A tidy base to sculpt on. Up to 3% may change, even rows.",
            Purpose::Custom => "You choose how much of the view may change.",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Purpose::Game => "game",
            Purpose::Scan => "scan",
            Purpose::LowPoly => "low_poly",
            Purpose::Sculpt => "sculpt",
            Purpose::Custom => "custom",
        }
    }

    /// Whether choosing this preset turns even rows of quads on.
    pub fn rows(self) -> bool {
        self != Purpose::LowPoly
    }
}

/// Everything in the settings panel.
#[derive(Clone, PartialEq, Debug)]
pub struct Choices {
    pub purpose: Purpose,
    pub custom_detail: f64,
    pub aim_for_count: bool,
    pub count: usize,
    pub rows: bool,
    pub gpu: bool,
    pub pixels: usize,
}

impl Default for Choices {
    fn default() -> Self {
        Choices { purpose: Purpose::Game, custom_detail: 4.0, aim_for_count: false, count: 5000, rows: true, gpu: true, pixels: 192 }
    }
}

impl Choices {
    /// Percent of the view that may change.
    pub fn detail(&self) -> f64 {
        match self.purpose {
            Purpose::Game => 6.0,
            Purpose::Scan => 2.0,
            Purpose::LowPoly => 12.0,
            Purpose::Sculpt => 3.0,
            Purpose::Custom => self.custom_detail,
        }
    }

    pub fn pick(&mut self, purpose: Purpose) {
        self.purpose = purpose;
        if purpose != Purpose::Custom {
            self.rows = purpose.rows();
        }
    }

    /// `count_allowed` is false in a batch: one face count can't suit many models.
    pub fn settings(&self, count_allowed: bool) -> curiomesh::retopo::Settings {
        curiomesh::retopo::Settings {
            detail: self.detail(),
            reduce_to: (self.aim_for_count && count_allowed).then_some(self.count),
            pixels: self.pixels,
            gpu: self.gpu,
            rows: self.rows,
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
        if let Some(purpose) = Purpose::ALL.into_iter().find(|x| v["purpose"] == x.key()) {
            c.purpose = purpose;
        }
        c.custom_detail = v["custom_detail"].as_f64().unwrap_or(c.custom_detail).clamp(0.5, 15.0);
        c.aim_for_count = v["aim_for_count"].as_bool().unwrap_or(false);
        c.count = v["count"].as_u64().map_or(c.count, |n| (n as usize).clamp(24, 5_000_000));
        c.rows = v["rows"].as_bool().unwrap_or(c.rows);
        c.gpu = v["gpu"].as_bool().unwrap_or(true);
        c.pixels = v["pixels"].as_u64().map_or(192, |n| (n as usize).clamp(64, 512));
        p.recent = v["recent"].as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(PathBuf::from)).take(MOST_RECENT).collect()).unwrap_or_default();
        p.tour_done = v["tour_done"].as_bool().unwrap_or(false);
        p
    }

    pub fn to_json(&self) -> Value {
        let c = &self.choices;
        json!({
            "purpose": c.purpose.key(), "custom_detail": c.custom_detail, "aim_for_count": c.aim_for_count, "count": c.count,
            "rows": c.rows, "gpu": c.gpu, "pixels": c.pixels, "tour_done": self.tour_done,
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
        p.choices.pick(Purpose::LowPoly);
        p.choices.pixels = 256;
        p.remember(std::path::Path::new("a.obj"));
        p.tour_done = true;
        let back = Prefs::from_json(&p.to_json());
        assert_eq!(back.choices, p.choices);
        assert_eq!(back.recent, p.recent);
        assert!(back.tour_done && !back.choices.rows && back.choices.detail() == 12.0);
        assert_eq!(Prefs::from_json(&Value::Null).choices, Choices::default()); // a missing or broken file gives the defaults
    }
}
