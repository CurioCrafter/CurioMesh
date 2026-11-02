# Security

## Reporting a vulnerability

Please report security problems privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability** ([direct link](https://github.com/CurioCrafter/CurioMesh/security/advisories/new)), which opens a private security advisory. Don't open a public issue for it.

Say what an attacker would need (for example "the victim opens this .blend" or "imports this result.npz"), what
happens, and the Blender, CurioMesh and OS versions. A small file that shows it helps most. You'll get an answer in
the advisory; fixes go out in a new release and are credited there unless you'd rather not be named.

Only the latest release is supported with fixes.

## What CurioMesh does and doesn't do

- It never connects to the network and sends no telemetry.
- It starts other programs only in two places: a second copy of the same Blender (`bpy.app.binary_path`) for the
  QuadriFlow rows, and `nvidia-smi` from the folders on `PATH` to read the graphics card.
- Files it reads from elsewhere (an Import Job Result `result.npz`, an imported fix library, the data saved in a
  `.blend`) are checked before use, and no pickle or `eval` is used.
- An exported job folder holds `run_job.py` and a copy of the add-on's Python code. It is a program: run only job
  folders you exported yourself or got from someone you trust.

## Where releases come from

Release zips are built only by the GitHub Actions release workflow, from the tag, with `SHA256SUMS`. A zip built on someone's own computer is a test build: its native binary holds that computer's local paths, and it should not be published.
