# Running a big job on a stronger computer

Auto Segments can be too much for the computer you model on: a scan with a million
faces can take most of an hour and a few gigabytes of memory. CurioMesh has two ways to
deal with that.

1. **Work in small steps on your own PC.** Set **Effort** to Light (or leave it on Auto).
   Huge pieces are cut into regions, each region is finished and shown on its own, and
   finished regions are saved to disk so a stopped job can resume.
2. **Run the same job somewhere else.** **Export Job** writes a folder. You copy it to any
   computer with Blender, run one command, and copy `result.npz` back. CurioMesh never
   uploads anything and has no accounts or keys: you move the folder yourself.

## The folder

Press **Export Job** in the Auto Segments panel (the "This computer" box) and pick where
to put it. You get `<mesh name>_curiomesh_job/`:

| File | What it is |
| --- | --- |
| `mesh.npz` | your mesh (welded, with its material indices) |
| `job.json` | the settings: Change Allowed or Reduce To, and the effort (Full) |
| `curiomesh_engine/` | a copy of the CurioMesh Python code, so the add-on is not needed there |
| `run_job.py` | runs the same search the button runs, without a window |
| `README.txt` | the same steps as below, for whoever runs it |

An exported job solves the whole mesh. Guide curves, painted density and Mirror are not
included.

## Steps on a generic Linux machine

Any Linux machine you can log into works. Today the job runs on the CPU, so pick
**many CPU cores and plenty of memory** (see the table below), not the biggest graphics
card.

1. Rent or start a Linux machine from the provider you like, and log in with SSH. Note
   what it costs per hour.
2. Install what a windowless Blender needs, and Blender itself. On Ubuntu or Debian:

   ```bash
   sudo apt update
   sudo apt install -y xz-utils libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6
   # Get the link of the Linux .tar.xz from blender.org/download (Blender 3.6 or newer), then:
   wget -O blender.tar.xz "<link>"
   mkdir blender && tar -xf blender.tar.xz -C blender --strip-components=1
   ```
3. Copy the job folder to it (from your PC):

   ```bash
   scp -r MyMesh_curiomesh_job user@host:~/
   ```
4. Run it. Use `tmux` or `nohup` so it keeps going if your connection drops:

   ```bash
   cd ~/MyMesh_curiomesh_job
   nohup ~/blender/blender --background --python run_job.py > log.txt 2>&1 &
   tail -f log.txt
   ```

   It prints the machine it found, then a line every ten seconds with the percent and
   the part it is working on ("Part 3 of 12: upper left"). Add `-- --effort LIGHT` after
   `run_job.py` to make it gentler.
5. When the log says `Done`, copy `result.npz` back:

   ```bash
   scp user@host:~/MyMesh_curiomesh_job/result.npz .
   ```
6. **Stop or delete the machine now.** It bills while it is on, even when idle.
7. In Blender, select the original mesh, press **Import Result**, pick `result.npz`, and
   press **Apply**. It refuses a result made from a different mesh.

If the job stops or the machine crashes, run the same command again. Finished parts are
in the `progress` folder and are not done twice.

## How much machine

CurioMesh measures its own limits so the panel can say "this mesh is large for this PC".
The numbers it uses (Full effort, 12 hardware threads free):

<!-- MEASURED: filled in from timed runs -->

## What a graphics card speeds up today

<!-- GPU: to be filled in by the CUDA acceleration work. Until it lands, the honest answer is below. -->

As of this version **no step of Auto Segments runs on a graphics card**. The looking
(camera pictures), the face reduction and the pairing into quads all run on the CPU, and
most of it uses one core. Only the optional compiled Turbo kernels use several, and the
exported folder carries the Python code only, so a job there runs on the Python paths. A
rented machine helps through its CPU speed and memory, not through its GPU. This section will list the
steps that use CUDA once they exist.

## Price comparison

<!-- PRICE TABLE: a comparison of cloud machines is coming; it is filled in by the pricing research. -->

A comparison of what common providers charge for the machine you need is coming. Until
then, work out the cost yourself: hours the panel estimates times the provider's price
per hour. Add the time to copy the folder both ways.
