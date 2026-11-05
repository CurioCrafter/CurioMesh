// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors
//! The in-app guide and the first-run tour. Plain words, short sentences.

use super::{AMBER, MUTED, PANEL, TEXT};
use eframe::egui::{self, Align2, RichText, Stroke, Vec2};

enum Block {
    P(&'static str),
    H(&'static str),
    Bullets(&'static [&'static str]),
}
use Block::{Bullets, H, P};

const TOPICS: [(&str, &[Block]); 8] = [
    ("Start here", &[
        P("CurioMesh turns a heavy model into a clean mesh made of quads. It keeps how the model looks and drops the faces it doesn't need."),
        H("The short way"),
        Bullets(&[
            "1. Choose what the model is for. This is the Home screen, before you open anything.",
            "2. Open a model, or drop it on the window. For many models, use Batch a Folder.",
            "3. Press Retopologize. A model of 50,000 to 100,000 faces takes a few seconds to a minute.",
            "4. Drag the line across the model to compare. Check the numbers. Save the OBJ.",
        ]),
        P("Your choices are remembered the next time you start the app."),
        H("Files"),
        P("Opens OBJ, glTF, GLB and PLY. Saves OBJ."),
    ]),
    ("What it's for", &[
        P("A preset is a ready-made answer to one question: how much of the view may change? A smaller number keeps more faces and more detail. A bigger number gives fewer, larger quads."),
        Bullets(&[
            "Game asset: up to 6% may change. Few faces, good for real-time use. Flat parts become big quads; edges and outlines stay.",
            "Scan cleanup: up to 2%. Keeps small bumps and wear from a scan or photo capture.",
            "Low poly: up to 12%, and no extra rows. The fewest faces. Small details will be lost. Good for far-away props and blockouts.",
            "Sculpt base: up to 3%, with even rows on smooth parts. A tidy start to sculpt on. Check that the shape is close enough for your idea.",
            "Your own limit: pick any value from 0.5% to 15% with the slider.",
        ]),
        P("If you are not sure, start with Game asset. Look at the result. If detail is missing, try Scan cleanup. If it is still too heavy, try Low poly."),
    ]),
    ("Other settings", &[
        H("Aim for a face count instead"),
        P("Asks for about this many faces, whatever the view change turns out to be. Use it when a budget is fixed, for example 5,000 faces. It works on one model at a time, so it is turned off in a batch."),
        H("Even rows on smooth parts"),
        P("Smooth parts, like a rounded body, are also tried as even rows of quads. CurioMesh keeps the rows only when the view still holds and no bad faces come in. Turn it off for the fewest faces."),
        H("Measuring detail"),
        P("How big the camera pictures are when CurioMesh measures the view change. Quick is faster. Fine sees smaller differences and takes longer. Normal is right for most models."),
        H("Use the graphics card"),
        P("Draws the camera pictures on the graphics card. This is faster. If it fails on your computer, turn it off and the processor does the work."),
    ]),
    ("Reading the numbers", &[
        H("View change"),
        P("CurioMesh looks at the model from six sides, before and after. It counts the pixels where the outline, the depth or the direction of the surface differs. That share is the view change."),
        Bullets(&[
            "Green means it is inside your limit.",
            "Amber means it is a little over. Try a lower limit, or look at the spots.",
            "A face-count request has no limit, so the number is only for your information.",
        ]),
        H("Bad faces"),
        P("Faces on broken edges, flipped or duplicated faces, and quads bent past 60 degrees or folded. A good result has 0. If you see red, open the list of spots and fix those places, or try a different preset."),
        H("Spots to look at"),
        P("The list under the result shows places worth a look, worst first: slivers (very thin faces), faces at poles (where many edges meet) and bent quads. Click one to fly to it. Its faces turn red."),
        H("Faces"),
        P("The big number is how many faces the result has, and how many times fewer than the original."),
    ]),
    ("Compare view", &[
        P("After a run, the model is split by a line. The original is on the left. The new quads are on the right, with their edges drawn."),
        Bullets(&[
            "Drag the line to move the split.",
            "Drag anywhere else to turn the model.",
            "Scroll to zoom. Double-click to frame the whole model again.",
            "Side by side, Original and Result switch what is shown.",
            "Upright stands the model on its other axis, if it lies on its side.",
        ]),
        P("Compare silhouettes first, then the places where the surface curves. If something important is flat in the result, lower the limit."),
    ]),
    ("Batch", &[
        P("Batch runs many models one after another with the same settings."),
        Bullets(&[
            "Open Batch, then add files or a folder. You can also drop them on the window. A folder adds the models directly inside it.",
            "Choose the folder for the results. Each one is saved as NAME_curiomesh.obj.",
            "Press Run batch. Each model has its own bar, and the top bar shows all of them.",
            "Cancel stops after the model that is in progress. Finished results stay saved.",
            "A model that can't be read or fails shows Failed with the reason. The others carry on.",
        ]),
        P("The table lists faces before and after, view change, bad faces and time. Open result compares a model with its result. Show in folder finds the OBJ. Export table saves everything as a CSV file for a spreadsheet."),
        P("The window stays responsive while a batch runs. You can look at the guide, but you can't change the list until it stops."),
    ]),
    ("Tips by kind of model", &[
        H("Scans and photo captures"),
        Bullets(&["Use Scan cleanup.", "Scans are noisy. A low limit keeps the noise as detail, so raise it a little if the result is heavy.", "Check the spots list for slivers."]),
        H("Game assets and CAD"),
        Bullets(&["Use Game asset, or Low poly for far-away props.", "Hard edges and outlines are kept. Flat areas turn into large quads.", "Aim for a face count when you have a fixed budget."]),
        H("Sculpts"),
        Bullets(&["Use Sculpt base for a tidy base mesh to continue sculpting.", "Keep even rows on.", "Compare curved areas closely in the compare view."]),
        H("Many files"),
        Bullets(&["Try your settings on one model first, then batch the folder.", "Look at the bad faces column. Open any row that shows red."]),
    ]),
    ("Mouse and keys", &[
        Bullets(&[
            "Drop a file on the window to open it. Drop several files or a folder to start a batch.",
            "Drag in the viewer to turn the model. Scroll to zoom. Double-click to frame it.",
            "Drag the white line to compare.",
            "Hover over any control for a short tip.",
        ]),
        P("Settings are saved in a small file in your user config folder."),
    ]),
];

/// The guide window. `topic` is the page shown.
pub fn guide_window(ctx: &egui::Context, open: &mut bool, topic: &mut usize, replay_tour: &mut bool) {
    egui::Window::new("Guide").open(open).fixed_size([780.0, 470.0]).collapsible(false).anchor(Align2::CENTER_CENTER, Vec2::ZERO).show(ctx, |ui| {
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(160.0);
                for (i, (title, _)) in TOPICS.iter().enumerate() {
                    if ui.selectable_label(*topic == i, *title).on_hover_text("Show this page of the guide.").clicked() {
                        *topic = i;
                    }
                }
                ui.add_space(10.0);
                if ui.button("Replay the tour").on_hover_text("Show the four-step welcome again.").clicked() {
                    *replay_tour = true;
                }
            });
            ui.separator();
            ui.allocate_ui_with_layout(Vec2::new(580.0, 460.0), egui::Layout::top_down(egui::Align::Min), |ui| egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.label(RichText::new(TOPICS[*topic].0).heading().strong());
                for block in TOPICS[*topic].1 {
                    match block {
                        P(text) => drop(ui.label(*text)),
                        H(text) => {
                            ui.add_space(4.0);
                            ui.label(RichText::new(*text).strong().color(AMBER));
                        }
                        Bullets(lines) => {
                            for line in *lines {
                                ui.label(format!("•  {line}"));
                            }
                        }
                    }
                }
            }));
        });
    });
}

const TOUR: [(&str, &str); 4] = [
    ("Welcome", "CurioMesh turns a heavy model into clean quads. Start by choosing what the model is for. That sets how much detail is kept."),
    ("Add your models", "Click Open Model, or drop a file anywhere on the window. Many files? Use Batch a Folder."),
    ("Read the result", "View change says how much the look moved. Bad faces should be 0. Drag the line across the model to compare."),
    ("Need help?", "The Guide button at the top right explains every setting. Hover over any control for a short tip."),
];

/// The tour card. Returns Some(dont_show_again) when the tour is over (finished or skipped).
pub fn tour(ctx: &egui::Context, step: &mut usize, forget: &mut bool) -> Option<bool> {
    let mut over = None;
    egui::Area::new(egui::Id::new("tour")).anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -28.0)).order(egui::Order::Foreground).show(ctx, |ui| {
        egui::Frame::new().fill(PANEL).stroke(Stroke::new(1.5, AMBER)).corner_radius(10.0).inner_margin(16.0).shadow(egui::Shadow { offset: [0, 4], blur: 16, spread: 2, color: egui::Color32::from_black_alpha(140) }).show(ui, |ui| {
            ui.set_width(440.0);
            let (title, text) = TOUR[*step];
            ui.label(RichText::new(format!("Step {} of {}", *step + 1, TOUR.len())).small().color(MUTED));
            ui.label(RichText::new(title).size(19.0).strong().color(TEXT));
            ui.add(egui::Label::new(text).wrap());
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let last = *step + 1 == TOUR.len();
                if ui.add(egui::Button::new(RichText::new(if last { "Done" } else { "Next" }).strong()).stroke(Stroke::new(1.0, AMBER))).on_hover_text(if last { "Close the tour." } else { "Go to the next step." }).clicked() {
                    if last { over = Some(*forget) } else { *step += 1 }
                }
                if *step > 0 && ui.button("Back").on_hover_text("Go to the step before.").clicked() {
                    *step -= 1;
                }
                if ui.button("Skip").on_hover_text("Close the tour now.").clicked() {
                    over = Some(*forget);
                }
                ui.checkbox(forget, "Don't show again").on_hover_text("Leave this ticked and the tour won't come back. You can replay it from the Guide.");
            });
        });
    });
    over
}
