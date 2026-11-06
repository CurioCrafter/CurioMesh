# Branch `blender`: the extensions.blender.org build

This branch is the Python-only build of CurioMesh for the official Blender Extensions site. Its zip is reviewed by
Blender's moderators, who read the code closely. Follow every rule below on this branch, always. When unsure whether
something is allowed, leave it out and ask the owner. Breaking a rule costs a rejection and annoys a reviewer we want
on our side.

## Release policy
- New features go to the GitHub build (`main`) first. They come to this branch only in a later version, after the
  current site version is approved, and only cut down to follow these rules.
- Don't add features to a version already under review. Fix only what the reviewer asked for, plus rule fixes.
- Merge `main` into this branch, never this branch into `main`.

## What the first review (0.9.7) rejected, never again
1. Editing `sys.path` or `sys.modules`, or passing `PYTHONPATH` / `--python-use-system-env` to child processes.
2. `threading`, `queue`, `concurrent.futures`, `multiprocessing`, `asyncio`. Slow work runs in timer slices
   (`triadq/tasks.py`), a few rows per slice, never one long blocking call.
3. Running any outside program (e.g. `nvidia-smi`, `sysctl`). The only subprocess allowed is Blender itself
   (`bpy.app.binary_path --background --factory-startup`), and only when the user turned on Background Blender
   Processes (off by default).
4. Names of things that are not in this build: no "Lite", "Turbo", native core, Rust, GPU core, GitHub-only speed
   claims, in code, identifiers, UI, comments or docs.
5. `depsgraph_update_*` handlers. Draw callbacks only draw.
6. Anything that only exists in another build.

## Rules from the Terms of Service, moderation guidelines and manifest docs
- GPL-compatible licence (`SPDX:GPL-3.0-or-later`); every bundled file GPL-compatible or CC0 with a note.
- Manifest: tagline short, no ending punctuation; tags from the official list; permissions only for what is used,
  each reason one short sentence without ending punctuation; no `bl_info` in `__init__.py`; no `platforms`/`wheels`
  unless really bundled (we bundle none).
- No binaries, wheels, bytecode, ctypes, `eval`/`exec`, pickle, `__main__` blocks, absolute paths, backslash paths.
- No network access. Opening a URL in the browser only from a button that says so ("(opens browser)"). No telemetry,
  no update checker.
- No promotion: no star or donation prompts, no "Pro" paid-tier wording, no links pushing users to other builds.
- Files: written only under `bpy.utils.extension_path_user(__package__, create=True)` or a path the user picked in a
  file browser. Temp job folders deleted when the job ends. Nothing into the add-on folder, the .blend's scenes,
  config folders or system caches.
- Preferences keyed by `__package__`; relative imports only, no import fallbacks.
- Clean unregister: every class, timer, draw handler, property, icon collection and handler removed; nothing left
  after disable. Registering twice must not error. No keymaps, no changes to user preferences, themes or other
  add-ons. No monkey-patching `bpy`.
- No print spam at import or register. No dead code, unused modules or properties nothing reads.
- Our own work: no names, wording or UI from other add-ons (SoftWrap, Quad Remesher, RetopoFlow, Hard Ops/BoxCutter,
  ZRemesher, Smart Remesh, LoopTools look-alikes).
- The listing text, README (`docs/LISTING_README.md`) and screenshots describe only what this build has, and the
  screenshots are taken with this build.
- Blender must stay responsive: no single step should freeze it for seconds on ordinary meshes; Stop and Apply must
  respond quickly.

## Before every site zip
1. `python scripts/build_release.py --blender-listing` (its `check_listing()` / `verify_archive()` fail on most of the
   above automatically; add a check there whenever a new rule is learned).
2. `blender --command extension validate <zip>`.
3. Install into a clean profile, run `tests/blender_registration.py`, one smoke retopo (see
   `bench/blendersite/install_check.sh` in the CurioMesh-main folder).
4. Update the review notes (`bench/blendersite/CHANGES.md`) with what changed and why.
5. Run a strict mock review (several moderator lenses: code, manifest/listing, hands-on, "what would this reviewer
   pick on next") until a round finds nothing to fix. Only then hand the zip to the owner, who uploads it and posts
   the reply.
