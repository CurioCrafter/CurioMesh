// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! Everything a run needs, as one value that reads and writes a small JSON file, plus the built-in presets (the add-on's
//! Uses, `profiles.py`, where the standalone can do them) and presets people save themselves.

use crate::export::Target;
use crate::io::Mesh;
use crate::{mirror, retopo};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mirror {
    Off,
    Auto,
    Axis(usize),
}

impl Mirror {
    pub fn parse(word: &str) -> Result<Mirror, String> {
        Ok(match word.to_ascii_lowercase().as_str() {
            "off" | "no" | "none" => Mirror::Off,
            "auto" => Mirror::Auto,
            "x" => Mirror::Axis(0),
            "y" => Mirror::Axis(1),
            "z" => Mirror::Axis(2),
            other => return Err(format!("mirror: {other}? Use off, auto, x, y or z")),
        })
    }
    pub fn word(&self) -> &'static str {
        match self {
            Mirror::Off => "off",
            Mirror::Auto => "auto",
            Mirror::Axis(a) => ["x", "y", "z"][*a],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub detail: f64,              // percent of the view that may change
    pub reduce_to: Option<usize>, // a face count instead
    pub pixels: usize,
    pub gpu: bool,
    pub rows: bool,
    pub baked: bool, // judge as it will look with a normal map baked from the source (scans)
    pub mirror: Mirror,
    pub export: Target,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { detail: 6.0, reduce_to: None, pixels: 192, gpu: true, rows: true, baked: false, mirror: Mirror::Off, export: Target::default() }
    }
}

/// (key, label, what it is for, settings).
pub struct Preset {
    pub key: &'static str,
    pub label: &'static str,
    pub words: &'static str,
    pub settings: Settings,
}

/// The built-in presets: the add-on's Uses that the standalone can do.
pub fn presets() -> Vec<Preset> {
    let with = |detail: f64, rows: bool, mirror: Mirror, triangulate: bool| {
        let mut s = Settings { detail, rows, mirror, ..Settings::default() };
        s.export.triangulate = triangulate;
        s
    };
    vec![
        Preset { key: "game-asset", label: "Game Asset", words: "All quads, few faces, for a game model: up to 6% of the view may change. The default", settings: with(6.0, true, Mirror::Off, false) },
        Preset { key: "scan", label: "Scan Cleanup", words: "Keeps more of the detail of a dirty scan, judged as it will look with its normal map baked: up to 2% of the view may change", settings: Settings { baked: true, ..with(2.0, true, Mirror::Off, false) } },
        Preset { key: "low-poly", label: "Low Poly", words: "Fewest faces for a prop or background, written as triangles: up to 10% may change. Small parts may vanish", settings: with(10.0, false, Mirror::Off, true) },
        Preset { key: "sculpt-base", label: "Sculpt Base", words: "Even quads to sculpt on, mirrored when the model is symmetric: up to 6% may change", settings: with(6.0, true, Mirror::Auto, false) },
        Preset { key: "animation", label: "Animation", words: "More, evener quads so the mesh bends cleanly, mirrored when symmetric: up to 1.5% may change. Check eyes, mouth and joints by hand", settings: with(1.5, true, Mirror::Auto, false) },
    ]
}

/// Where saved presets live: CurioMesh/presets in the user's settings folder.
pub fn presets_dir() -> PathBuf {
    let base = std::env::var_os("APPDATA").map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(if cfg!(target_os = "macos") { "Library/Application Support" } else { ".config" })))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("CurioMesh").join("presets")
}

/// Names of the presets saved in `presets_dir`.
pub fn saved_presets() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(presets_dir()).into_iter().flatten().flatten()
        .filter_map(|e| { let p = e.path(); (p.extension()? == "json").then(|| p.file_stem()?.to_str().map(str::to_string)).flatten() })
        .collect();
    names.sort();
    names
}

/// A built-in preset by key or label, else a saved one by name.
pub fn preset(name: &str) -> Result<Settings, String> {
    let wanted = name.to_ascii_lowercase().replace([' ', '_'], "-");
    let alias = match wanted.as_str() {
        "game" => "game-asset",
        "scan-cleanup" => "scan",
        "sculpt" => "sculpt-base",
        other => other,
    };
    if let Some(p) = presets().into_iter().find(|p| p.key == alias || p.label.to_ascii_lowercase().replace(' ', "-") == alias) {
        return Ok(p.settings);
    }
    let path = presets_dir().join(format!("{name}.json"));
    if path.exists() {
        return Settings::load(&path);
    }
    let keys: Vec<&str> = presets().iter().map(|p| p.key).collect();
    Err(format!("no preset called {name} (built in: {}; saved: {})", keys.join(", "), saved_presets().join(", ")))
}

pub fn save_preset(name: &str, settings: &Settings) -> Result<PathBuf, String> {
    if name.is_empty() || name.contains(['/', '\\', ':', '.']) {
        return Err(format!("{name}: a preset name can't be empty or hold / \\ : or ."));
    }
    let dir = presets_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{name}.json"));
    settings.save(&path)?;
    Ok(path)
}

impl Settings {
    pub fn retopo(&self) -> retopo::Settings {
        retopo::Settings { detail: self.detail, reduce_to: self.reduce_to, pixels: self.pixels, gpu: self.gpu, rows: self.rows, baked: self.baked }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "curiomesh_settings": 1,
            "detail": self.detail,
            "reduce_to": self.reduce_to,
            "pixels": self.pixels,
            "gpu": self.gpu,
            "rows": self.rows,
            "baked": self.baked,
            "mirror": self.mirror.word(),
            "export": self.export.to_json(),
        })
    }

    /// Settings from JSON: any key left out keeps its default, an unknown key is an error (it is likely a typo).
    pub fn from_json(value: &Value) -> Result<Settings, String> {
        let mut s = Settings::default();
        s.apply(value)?;
        Ok(s)
    }

    /// Set the keys `value` holds and leave the rest.
    pub fn apply(&mut self, value: &Value) -> Result<(), String> {
        let empty = Map::new();
        let map = value.as_object().ok_or("settings must be a JSON object")?;
        let number = |k: &str, v: &Value| v.as_f64().ok_or(format!("settings: {k} must be a number"));
        let flag = |k: &str, v: &Value| v.as_bool().ok_or(format!("settings: {k} must be true or false"));
        for (k, v) in map {
            match k.as_str() {
                "curiomesh_settings" => {}
                "detail" => self.detail = number(k, v)?.clamp(0.1, 80.0),
                "reduce_to" => self.reduce_to = if v.is_null() { None } else { Some(number(k, v)?.max(12.0) as usize) },
                "pixels" => self.pixels = (number(k, v)? as usize).clamp(16, 2048),
                "gpu" => self.gpu = flag(k, v)?,
                "rows" => self.rows = flag(k, v)?,
                "baked" => self.baked = flag(k, v)?,
                "mirror" => self.mirror = Mirror::parse(v.as_str().ok_or("settings: mirror must be off, auto, x, y or z")?)?,
                "export" => self.export.apply(v.as_object().unwrap_or(&empty))?,
                other => return Err(format!("settings: unknown key {other}")),
            }
        }
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Settings, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let value: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        Settings::from_json(&value).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, serde_json::to_string_pretty(&self.to_json()).unwrap()).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// The mesh the search works on: the source, or one half of it when mirroring.
pub struct Work {
    pub mesh: Mesh,
    pub plane: Option<mirror::Plane>,
}

pub fn work_on(source: &Mesh, settings: &Settings, log: &mut dyn FnMut(&str)) -> Work {
    let (axes, chosen): (Vec<usize>, bool) = match settings.mirror {
        Mirror::Off => return Work { mesh: source.clone(), plane: None },
        Mirror::Auto => (vec![0, 1, 2], false),
        Mirror::Axis(a) => (vec![a], true),
    };
    let plane = mirror::find_plane(source, &axes).into_iter().next().unwrap();
    if !plane.ok(chosen) {
        log(&format!("Not mirrored: not symmetric enough. {}", plane.text()));
        return Work { mesh: source.clone(), plane: None };
    }
    log(&format!("Mirroring on {}", plane.text()));
    Work { mesh: mirror::cut_half(source, plane.axis, plane.offset, 1.0), plane: Some(plane) }
}

/// A result on the half, mirrored back to the whole and looked at against the whole source.
pub fn finish(source: &Mesh, settings: &Settings, work: &Work, mut out: retopo::Outcome, log: &mut dyn FnMut(&str)) -> retopo::Outcome {
    if let Some(plane) = &work.plane {
        let (whole, seam) = mirror::mirror_result(&out.mesh, plane.axis, plane.offset, 1.0);
        out.loss = retopo::measure(source, &whole, settings.pixels, settings.gpu, settings.baked);
        out.looks += 1;
        log(&format!("  Mirrored: {} faces, {seam} corners on the seam, {:.2}% of the view changed", whole.faces(), out.loss));
        out.mesh = whole;
    }
    out
}

/// One whole run: mirror when asked, retopologize, and mirror back.
pub fn run(source: &Mesh, settings: &Settings, mut log: impl FnMut(&str)) -> Result<retopo::Outcome, String> {
    let started = std::time::Instant::now();
    let work = work_on(source, settings, &mut log);
    let mut half = settings.retopo();
    if work.plane.is_some() {
        half.reduce_to = half.reduce_to.map(|n| (n / 2).max(12));
    }
    let out = retopo::retopo(&work.mesh, &half, &mut log)?;
    let mut out = finish(source, settings, &work, out, &mut log);
    out.seconds = started.elapsed().as_secs_f64();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_round_trip_and_partial_files() {
        let mut s = preset("low-poly").unwrap();
        s.reduce_to = Some(500);
        s.mirror = Mirror::Axis(2);
        let back = Settings::from_json(&s.to_json()).unwrap();
        assert_eq!(back.to_json(), s.to_json());
        let partial = Settings::from_json(&json!({"detail": 3})).unwrap();
        assert_eq!((partial.detail, partial.rows), (3.0, true));
        assert!(Settings::from_json(&json!({"detial": 3})).is_err());
        assert!(preset("Game Asset").is_ok() && preset("nope-not-here").is_err());
    }
}
