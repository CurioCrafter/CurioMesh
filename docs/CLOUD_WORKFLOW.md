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
| `run_job.py` | runs the same search the button runs, without a window |
| `README.txt` | the same steps as below, for whoever runs it |

Mirror (the solved half), guide curves and painted density are carried in the folder
(`guides.npz`, and `drop` inside `mesh.npz`), so the result is the one the button would give.

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
   The job folder does not carry CurioMesh's code. `run_job.py` uses the CurioMesh add-on installed and enabled in the Blender that runs it, so install the release zip for that Blender version there too (the README's Install section), and use the same CurioMesh version as on your PC.
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
in the `curiomesh_resume` folder and are not done twice.

## How much machine

CurioMesh measures its own limits so the panel can say "this mesh is large for this PC".
The numbers it uses (Full effort, 12 hardware threads free):

<!-- MEASURED: filled in from timed runs -->

## What a graphics card speeds up today

Measured 2026-11-02 on a Ryzen 9 5900X and RTX 3080 Ti, Blender 5.2.3.

**One step runs on a graphics card: the camera pictures of a mesh over 150,000 triangles.** It uses wgpu (Vulkan, DirectX 12 or Metal), so an NVIDIA, AMD, Intel or Apple card works with nothing to install; CUDA was not built, on purpose. Against the 24-thread processor path, one set of six pictures is 1.1 to 1.4 times faster at 170,000 triangles, about 3 times at 1.3 million, level at 17,000 and slower below that, which is why smaller meshes stay on the processor. Pictures take 2 to 17% of a run, so even a free card would give 1.02 to 1.2 times. A 170,000-triangle model at Detail 2% took 292 s with the card and 342 s without, of which the pictures were 8.6 s and 12.0 s; the rest of the difference is run-to-run noise from other work on the PC. Most of a run is graph work and Blender's own reduction, which a GPU doesn't help with.

The card works in single precision, so a pixel on a depth tie can pick the neighbouring triangle. Against the processor's pictures the camera counts 0.02 to 0.4% of the pixels as changed on real meshes (up to 1.3% on a very dense scan at 32 pixels a side). The Use Graphics Card setting, or `CURIOMESH_GPU=0`, keeps everything on the processor.

**Measuring the parts uses several cores** (Use Several Cores, from 0.9.1): helper Blenders measure parts at once, as many as the free cores and memory allow. The rest of a job (pairing, the checks and repairs) still uses about one core, apart from the native steps. So a rented machine helps through a fast core, enough memory, and more cores on a job with many parts. The add-on installed on the rented machine decides this: it runs the native core if that add-on zip includes the core for the machine's system, and the Python paths if not.

## Price comparison

Prices read on 2026-11-01. Nothing was rented, started or signed up for, and they change often, so check the live page before you rent. Cost is roughly the hours the panel estimates times the price per hour, plus the time to copy the folder both ways.

**A CPU machine is the right rental today.** A GPU machine pays for a card that sits idle, apart from the pictures step (1 to 2% of a run).

### CPU machines (the ones that fit today's tool)

| Provider | Machine | Price | Kind | Min billing | Link |
|---|---|---|---|---|---|
| Hetzner Cloud | CCX13, 2 dedicated vCPU, 8 GB | €0.0697 | steady | hourly, monthly cap | [page](https://www.hetzner.com/cloud/general-purpose/) |
| Hetzner Cloud | CCX23, 4 vCPU, 16 GB | €0.1386 | steady | hourly | same |
| Hetzner Cloud | CCX33, 8 vCPU, 32 GB | €0.2227 | steady | hourly | same |
| Hetzner Cloud | CCX53, 32 vCPU, 128 GB | €0.8558 | steady | hourly | same |
| Hetzner Cloud | CCX63, 48 vCPU, 192 GB | €1.3686 | steady | hourly | same |
| Hetzner dedicated | AX162, 48-core EPYC 9454P | €0.984, plus **€304 setup** | steady | no minimum term, but the setup fee kills short use | [page](https://www.hetzner.com/dedicated-rootserver/matrix-ax/) |
| Salad CPU groups | any vCPU and RAM mix | $0.005 per vCPU, $0.001 per GB (32 vCPU, 64 GB = $0.224) | steady list, consumer PCs | per second | [page](https://salad.com/pricing) |
| Vast.ai | verified offers with at least 32 threads (a cheap GPU comes with them): RTX 3070 box $0.108, RTX A4000 box with EPYC 7742 $0.137 | live offers | marketplace | per second | [live API query](https://console.vast.ai/api/v0/bundles/) |
| AWS | c7a.8xlarge, 32 vCPU | $0.72 on demand (aggregator; AWS's own list not read), spot about $0.59 to $0.82 | spot moves | per second, 60 s minimum (from memory) | [Holori](https://calculator.holori.com/aws/ec2/c7a.8xlarge/us-east-1) |

RunPod has CPU pods, but its pricing page does not list them and I found no reliable price: it shows them only in its console.

Cost per vCPU-hour: Hetzner CCX about €0.027 to €0.035, Salad about $0.007 (with 2 GB each), Vast about $0.0035 to $0.0043. Vast and Salad are cheap because they are other people's machines, so read the private files section first.

Rough cost of the measured 1.26M-face job (about 7 minutes, one core), if the machine's speed is near this PC's: about €0.02 on a CCX23. A rented vCPU is usually slower than a desktop Ryzen core, and this was not tested.

### GPU machines (for when a GPU step exists, or other work)

| Provider | GPU | On demand | Spot or low tier | Kind | Storage | Min billing | Link |
|---|---|---|---|---|---|---|---|
| RunPod Community | RTX A5000 / 3090 / 4090 / L4 | $0.16 / $0.22 / $0.34 / $0.44 | not listed here | third-party hosts, list price | container disk $0.10 per GB per month, network volume $0.07 | per second | [page](https://www.runpod.io/pricing) |
| RunPod Secure | RTX A5000 / 3090 / 4090 / L4 | $0.27 / $0.50 / $0.89 / $0.59 | not listed here | vetted data centers, list price | same | per second | same |
| Vast.ai, any host | RTX 4090, cheapest 1-GPU offers | $0.30 to $0.34 (one de-verified outlier at $0.13) | bids from $0.21 | marketplace | about $0.13 to $0.40 per GB per month | per second | [live API query](https://console.vast.ai/api/v0/bundles/) |
| Vast.ai, verified | RTX 3090 | $0.16 to $0.19 | bids from $0.12 | marketplace | about $0.37 to $0.87 per GB per month | per second | same |
| Vast.ai, verified data center | RTX 4090 | $0.93 to $1.20 | bids from $0.28 | marketplace | about $0.27 per GB per month | per second | same |
| Salad | RTX 3090 / 4090 | $0.20 / $0.44 at High priority | $0.09 / $0.16 at Lowest (prices new today) | consumer PCs, can be pre-empted | not listed | per second while running | [page](https://salad.com/pricing) |
| TensorDock | RTX 3090 / 4090 | about $0.20 / $0.35 to $0.37 | 4090 about $0.20 | marketplace | not read | not read | [4090 page](https://tensordock.com/gpu-4090), own page via search only, [pricing page is 404](https://www.tensordock.com/pricing) |
| Lambda | Quadro RTX 6000 / A6000 / A10 | $0.69 / $1.09 / $1.29 (A10 with 30 vCPU, 226 GB) | none | own cloud | local SSD included | not stated | [page](https://lambda.ai/pricing) |
| DigitalOcean (owns Paperspace) | RTX 4000 Ada / RTX 6000 Ada | $0.76 / $1.57 | none | own cloud, steady | not read | per second, 5 minute minimum | [page](https://www.digitalocean.com/pricing/gpu-droplets) |
| Paperspace Gradient | A4000 / A5000 | $0.76 / $1.38 on an undated page, may be stale | none | own cloud | not read | not read | [page](https://support.paperspace.com/hc/en-us/articles/360002484474-Gradient-Pricing) |
| Hetzner dedicated | RTX PRO 4000 SFF 24 GB (GEX45) | €0.3429, plus **€209 setup** | none | own data centers | included disk | no use for short jobs | [page](https://www.hetzner.com/dedicated-rootserver/matrix-gpu/) |
| AWS | g4dn.xlarge T4 / g6.xlarge L4 / g5.xlarge A10G | $0.526 / $0.805 / $1.006 | T4 about $0.23, L4 $0.35 to $0.73 by region, A10G about $0.30 to $0.45 (one blog) | spot moves | EBS extra | per second | aggregators: [g6](https://calculator.holori.com/aws/ec2/g6.xlarge/us-east-1), [g4dn](https://calculator.holori.com/aws/ec2/g4dn.xlarge/us-east-1) |
| Google Cloud | g2-standard-4 L4 | $0.707 | sources say $0.28 to $0.62 | spot moves | extra | per second, 1 minute minimum (from memory) | [gcloud-compute](https://gcloud-compute.com/g2-standard-4.html) |
| Google Colab | T4 / L4 / A100 | about $0.12 / $0.17 / $0.54 (one March 2026 table; other sources disagree) | none | compute units, $10 per 100 units | Drive | not guaranteed | [FAQ](https://research.google.com/colaboratory/faq.html) |

## Known limits

- Light on a 1.5 million face mesh still peaks at about 4.6 GB of memory.
- Guide Curves and Paint Density in an exported job were only unit tested, not run end to end.
