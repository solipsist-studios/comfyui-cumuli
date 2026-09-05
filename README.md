<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# comfyui-cumuli

ComfyUI nodes that take **one monocular video** to a **streamable `.sogst` 4D
gaussian splat**, entirely inside ComfyUI.

```
Load Video ──VIDEO──┐        Model Loader ─▶ LoRA loaders (stock) ──MODEL─┐
                    ▼                                                     ▼
  Generate Ring      4DAnyone: 24 synchronized novel views, RCP on  (~25 min)
      └─ Stage Ring         transpose to one directory per frame
          └─ Ring Masks         BiRefNet mattes, via ComfyUI's own model
              └─ Build 4DGS Dataset   visual-hull init cloud + RGBA training frames
                  └─ Train 4DGS           rotor 4D gaussian splatting  (~1 h)   ◀─DATASET── Load 4DGS Dataset
                      └─ Bake SOGST           splat_4d.sogst, in the output gallery
                          └─ Preview SOGST        → SPLAT → stock Render Splat
```

Three opaque types travel between nodes — `CUMULI_RING`, `CUMULI_FLIPBOOK`,
`CUMULI_DATASET` — each a handle to a directory tree the producer already
wrote. There are therefore no Save nodes (the producers are the save nodes) and
one Load node per type (`Load Ring`, `Load Flipbook`, `Load 4DGS Dataset`) for
re-entry and for external data.

Why this exists: reconstruction quality on subject captures is limited by **view
sparsity**, not by the trainer or the container. 4DAnyone synthesizes the dense
ring that a 10-camera rig cannot capture.

## Quickstart

You need ComfyUI, an NVIDIA card with about 32 GB of VRAM, and a CUDA toolkit
new enough to target it.

1. **Extract the zip** into `<ComfyUI>/custom_nodes/`, so you have
   `<ComfyUI>/custom_nodes/comfyui-cumuli/`.
2. **Run the installer** in that folder — double-click `install.bat` on Windows,
   or `./install.sh` on Linux. It clones the three checkouts it drives
   (4DAnyone, OMG4, cumuli), installs the Python dependencies into ComfyUI's own
   environment, downloads the model weights, and writes `config.json` pointing
   at all of it. Expect it to take a while and around 30 GB.
3. **Restart ComfyUI** if it was running — custom nodes load at startup.
4. **Load the workflow** (`workflows/cumuli_video_to_sogst.json`) and drop your
   clip into the Load Video node.
5. **Hit Run.**

### One licence-gated download

The installer fetches everything it legally can, but **SMPL-X body models** are
gated behind registration and cannot be downloaded for you. GVHMR needs them for
the motion solve, so Generate Ring fails without them. The installer prints this
at the end and names the exact path; the short version is: register at
[smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/), download
`models_smplx_v1_1.zip`, and put `SMPLX_NEUTRAL.npz` under
`deps/4DAnyone/models/body_models/smplx/`.

If ComfyUI does not ship `birefnet.safetensors` in
`models/background_removal/`, add it too — Ring Masks needs it.

### About your clip

At least **121 frames after `start_time`** (~5 s at 24 fps) at 720p or better.
Generate Ring checks this up front rather than failing an hour in.

### What to expect

Measured on an RTX 5090 (32 GB), 24 views with RCP on and the default
`enable_turbo`, from one 121-frame clip:

| stage | time | peak VRAM |
|---|---|---|
| Generate Ring | ~23 min | 29.5 GiB reserved |
| Stage + Masks + Dataset | a few minutes | modest |
| Train 4DGS | ~1 h (30000 iterations) | whole card |
| Bake SOGST | ~1 min | — |

`enable_turbo` is on by default and does the ring in 4 denoising steps. Turning
it off runs the base model — substantially slower, and the configuration the
older figures in this README were measured against.

The `.sogst` and its interchange PLY land in ComfyUI's output gallery under
`cumuli/`. Everything heavier lives under `<work_root>/<run_name>/`.

Nothing is thrown away between stages, so you can re-enter anywhere: **Load
Ring** picks up a finished ring, **Load Flipbook** a staged tree, **Load 4DGS
Dataset** a finished dataset. All three are discovery dropdowns with a refresh
button. See [Caching](#caching) for when a stage re-runs.

### If you would rather do it by hand

`./install.sh --help` breaks the run into parts: `--no-fetch` keeps checkouts
you already have, `--no-models` and `--no-configure` skip those stages,
`--deps-dir` moves the clones, `--work-root` sets the scratch drive, and
`--dry-run` prints every command without running any of it. The sections below
document what each stage does and why.

## Requirements

- A 4DAnyone checkout with its models (`~/Dev/github/4DAnyone` by default).
- An OMG4 rotor-4DGS checkout for training, and cumuli's `scripts/bake_sogst.py`
  for the bake (`~/Dev/github/cumuli/deps/OMG4` by default).
- A background-removal model in `models/background_removal/`
  (`birefnet.safetensors` ships with ComfyUI).
- About 32 GB of VRAM for a 24-view ring.

Everything runs in ComfyUI's environment. On top of a stock ComfyUI install it
needs the packages below, all additive — no downgrade of torch, numpy,
transformers, timm or ultralytics.

**`install.sh` / `install.bat` do all of it** — see [Quickstart](#quickstart).
It is idempotent (already-installed packages are skipped), auto-detects the CUDA
toolkit and GPU arch for the extension builds, and records the five pinned
packages before and after: if pip moves one, the run fails loudly instead of
leaving a broken ComfyUI to discover an hour later. `--force` reinstalls and
rebuilds, which is what you want after a torch upgrade. The interpreter is the
one thing it cannot guess reliably: it prefers `--python`, then
`$COMFYUI_PYTHON`, then an activated venv/conda env, and refuses outright if the
interpreter it picked has no torch.

The rest of this section is what the script does, for anyone doing it by hand:

```bash
# 4DAnyone + vendored GVHMR
pip install smplx==0.1.28 hydra-zen hydra_colorlog yacs lapx ftfy \
            sentencepiece fire colorlog ffmpeg-python
# the .sogst bake
pip install dahuffman
# the rig solve (Solve Rig). --no-deps throughout: torch declares *older*
# pinned cudnn/nccl than a working ComfyUI usually carries, so a plain install
# silently downgrades them underneath it.
pip install --no-deps h5py narwhals plotly pycolmap==4.0.4 \
    "lightglue @ git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8"
git clone --recurse-submodules https://github.com/cvg/Hierarchical-Localization.git \
    -b master <somewhere> && git -C <somewhere> checkout c13273bd0ecc2917a35910fd843712a1c6243193
pip install --no-deps -e <somewhere>
# the trainer
pip install cupy-cuda13x cuml-cu13 omegaconf imagesize
CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH \
TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" \
pip install <OMG4>/diff-gaussian-rasterization <OMG4>/simple-knn <OMG4>/pointops2 \
            --no-build-isolation --no-deps
```

`hloc` **must** be an editable install: its SuperPoint extractor reaches up to
`third_party/` relative to the repo root, which a copied install into
`site-packages` loses. The recursive clone matters for the same reason — the
weights live in the `SuperGluePretrainedNetwork` submodule, which a plain
`pip install git+URL` never fetches. `aliked` (BSD) needs neither and is the
node's default; `superpoint`'s weights are non-commercial research use only.

The last line pre-builds OMG4's CUDA extensions into this environment, so the
trainer never JIT-compiles at run time. Three hard-won details: point
`CUDA_HOME` at an nvcc that knows your GPU (`/usr/bin/nvcc` is often a stale
distro CUDA that cannot target Blackwell); pass `--no-deps`, or pip will
"helpfully" upgrade torch out from under ComfyUI while resolving the
extensions' requirements; and the builds are ABI-bound to the torch version,
so redo them if torch changes.

## Configuration

The idiomatic place is **ComfyUI's own Settings dialog** (search "Cumuli"):
work root, discovery roots, the VRAM floor and the checkout paths are all
registered there and stored as `cumuli.*` keys in the per-user
`comfy.settings.json`, exactly like other packs' settings. For headless or
per-checkout overrides, `config.json` next to `config.example.json` (or a file
named by `CUMULI_CONFIG`) beats the UI values, and environment variables
(`CUMULI_FDANYONE_ROOT`, `CUMULI_TRAINER_ROOT`, `CUMULI_PIPELINE_ROOT`,
`CUMULI_WORK_ROOT`, ...) beat everything. All layers are re-read on every node
run, so edits need no restart.

The three checkout paths are **`fdanyone_root`** (ring generation),
**`trainer_root`** (the OMG4 entry point) and **`cumuli_root`** (the scripts
Bake SOGST and Solve Rig drive in place). They are independent: OMG4 does not
have to be cumuli's submodule.

Set **`dataset_roots`** / **`flipbook_roots`** / **`ring_roots`** to the
directories where your external captures live; the loader nodes scan them (plus
`work_root`, and the 4DAnyone data dir for rings) into their dropdowns, model-loader style, and re-scan on the widget's refresh
button and at every queue. This works over remote connections, where a native
server-side file dialog cannot.

Set **`work_root`** to a large drive: heavy per-run intermediates (flipbook
staging, the 4DGS dataset, trainer checkpoints — ~20 GB per run) land under
`<work_root>/<run_name>/` instead of ComfyUI's temp and output directories.

**Everything runs in ComfyUI's own environment** — one interpreter, one torch.
The generation and training stages run as *child processes of that same
interpreter*, not of another env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
only takes effect before a process's first CUDA allocation, and both stages
want the whole device. The subprocess env also pins
`FDANYONE_ATTENTION_BACKEND=sdpa` — 4DAnyone's auto policy would pick
sageattention when ComfyUI has it installed, and at this model's shapes
sageattn peaks at exactly **2x** SDPA's memory, which is the difference between
RCP fitting a 32 GB card and not.

## The nodes

Every node lives in the **Cumuli** category and is named `Cumuli <something>`;
the table below drops the prefix. 4DAnyone is one external model this pack
drives, alongside the OMG4 trainer and cumuli's own `bake_sogst.py` — only
**Generate Ring** is 4DAnyone, which is why it alone carries the model's name
in its label (`Cumuli Generate Ring (4DAnyone)`).

| node | what it does |
|---|---|
| **Generate Ring** | Runs 4DAnyone. Takes a `VIDEO` socket (e.g. Load Video) — a file-backed, untrimmed video is used in place; trimmed or synthesized video is staged under `run_name`. The optional `MODEL` input folds the accumulated LoRA stack into the DiT weights inside the subprocess; `prompt` overrides the fixed prompt so trigger words reach cross-attention. Unloads ComfyUI's models and refuses to start below a free-VRAM floor. |
| **Load Ring** | Opens a finished result directory, so the graph can be re-entered without regenerating. The widget is a discovery combo (the 4DAnyone data dir + config `ring_roots`) with a refresh button, like the other two loaders. |
| **Ring Contact Sheet** | One frame from every view, tiled. The fastest way to spot cross-view identity drift. |
| **Select View** | One view as VIDEO + IMAGE + its camera JSON. |
| **Stage Ring** | Transposes 24 videos x 121 frames into 121 frame directories, with a per-frame `transforms.json`. |
| **Solve Rig (HLOC)** | Solves a **real** multi-camera rig into poses, for footage that ships no calibration. Runs multi-timestamp HLOC/pycolmap over a directory of per-camera videos, sharing one COLMAP camera per physical camera. Emits a typed `RIG` plus a QA readout (registered cameras, reprojection error, per-camera pose repeatability). Reads the capture directory only. |
| **Stage Capture** | The real-capture counterpart of Stage Ring: transposes a solved capture into frame directories with the solved poses. Stages only cameras that registered, and clamps the window to the shortest clip. |
| **Load Flipbook** | Opens a staged tree without re-running Stage Ring — including an **external frame-major capture** (`frame_NNNN/images_flat/<label>.png` + per-frame `transforms.json`), which then gets this pack's own matting and hull. The widget is a discovery combo (work_root + config `flipbook_roots`) with a refresh button. |
| **Ring Masks** | Mattes every staged frame. Takes a `BACKGROUND_REMOVAL` model, so ComfyUI's stock loader (or any substitute) drives it. Reports foreground coverage, because empty masks collapse the visual hull much later and confusingly. |
| **Build 4DGS Dataset** | Carves a time-stamped visual-hull init cloud, bakes the mattes into RGBA alpha, writes `transforms_train/test.json` with per-camera intrinsics, and checks the result against the trainer's contract. Emits a typed `DATASET`. |
| **Load 4DGS Dataset** | Validates and describes an existing dataset directory (D-NeRF layout) — the entry point for **finished external datasets** and real-capture exports. The widget is a discovery combo (work_root + config `dataset_roots`) with a refresh button. |
| **Train 4DGS** | Trains the rotor 4DGS model from a `DATASET` socket (Build or Load — never a raw path), reporting the trainer's own iteration count and PSNR. |
| **Bake SOGST** | Slices the 4D gaussians into the `.sogst` container and writes it to the output folder as a downloadable artifact. Also emits the 4D interchange PLY, which the preview reads. |
| **Preview SOGST** | Evaluates the baked clip at one instant and outputs ComfyUI's native `SPLAT`. |

### Caching

There are **no cache toggles** — no `on_existing`, no `overwrite`, no
skip-if-exists. Nodes follow ComfyUI's own semantics: a node re-runs only when
its inputs change. For the expensive artifacts the same rule extends to disk:
each stage stamps its output with a fingerprint of its inputs (clip bytes →
ring params/seed/LoRA/prompt → chained downstream), so an unchanged re-queue
after a server restart reuses the 90-minute ring and the 1-hour checkpoint
instantly, and a changed input replaces the stale artifact automatically.
Two deliberate exceptions: a directory the bridge did not stamp is never
deleted (foreign results error with guidance instead), and seed-only changes
keep the GVHMR motion cache, which does not depend on the seed.

### LoRA identity

The DiT is a Wan 2.2 TI2V-5B finetune, so ComfyUI's **own** `Load Diffusion
Model` node loads it (detected as Wan 2.2; the 4DAnyone-specific modules are
dropped with a warning, which is fine — ComfyUI never runs this copy, it is a
LoRA attachment point). Symlink the checkpoint into `models/diffusion_models/`
and chain stock LoRA loaders — anything trained against Wan 2.2-5B (dim 3072,
e.g. ai-toolkit's `wan22_5b` arch) applies:

```
Load Diffusion Model ─▶ LoraLoaderModelOnly ─▶ Generate Ring.model
```

At queue time the patch stack is folded into per-weight `up @ down` factors
(strength and alpha folded in; stacked LoRAs concatenate along rank) and
merged into the weights on CPU at subprocess load — bit-identical to comfy's
own `calculate_weight`, zero VRAM cost. Put the LoRA's trigger word in
Generate Ring's `prompt` or the cross-attention half of the LoRA stays inert.
Patch types the fold cannot express (DoRA, offsets) are refused loudly.

### Real captures

Three entry points, by how far your data has been processed:

- **raw rig footage, no calibration** (one video per camera, no poses) →
  **Solve Rig (HLOC)** → **Stage Capture** → Ring Masks → Build 4DGS Dataset;
- frames + cameras, no dataset yet → **Load Flipbook** → Ring Masks → Build
  4DGS Dataset (this pack computes the mattes and hull for you);
- a finished D-NeRF dataset (`transforms_train.json` + `points3d.ply`) →
  **Load 4DGS Dataset** → Train 4DGS.

#### Solving a rig

`Solve Rig` drives cumuli's `scripts/multiframe_sfm.py` in place, in ComfyUI's
own environment (`hloc`, `pycolmap` and `lightglue` are installed into it; see
Install). The rig is static, so each sampled timestamp is an independent
measurement of the same poses — tracks span space *and* time, and people moving
in the scene fall out as cross-time outliers.

Two costs shape every setting:

- **Matching is exhaustive**, with no retrieval step, so pairs grow as
  `(cameras x timestamps)^2`. 200 cameras at one timestamp is 19,900 pairs and
  fine; the same rig at four timestamps is 319,600 and is not. The node refuses
  a combination above 40,000 pairs and tells you what *would* fit.
- **Baseline matters more than camera count.** Thinning a dense rig with
  `camera_stride` cuts the quadratic cost, but widens the angle between
  neighbours; past a point cameras stop registering at all. Measured on a
  200-camera array: `camera_stride 16` (12 cameras) registered 4 of 12, while
  `camera_stride 4` (47 cameras) registered 46 of 47.

Read the report before going further. `mean reprojection error` under ~2 px is
necessary but *not* sufficient — check `mean track length` (expect > 8) and the
per-camera rotation spread (expect < 0.05°). A single-timestamp solve cannot
report a meaningful spread at all, because each camera has only one measurement
of itself; the node says so rather than printing a reassuring `0.0000`.

**Sync is the thing that quietly breaks this.** The solve samples the same
frame *index* on every camera, so if the clips are not frame-aligned, "timestamp
k" is a different real moment per camera and the static-rig premise collapses —
which shows up as cameras disagreeing with themselves across timestamps. A
hardware trigger does not guarantee alignment: on a 200-camera array whose clips
came off the trigger at 300, 301 and 302 frames, a 47-camera / 4-timestamp solve
put **31 of 47 cameras above the rotation-spread threshold** (median 0.31°,
worst 4.88°) while still reporting a healthy 1.218 px reprojection error.
Measure the offsets (`measure_sync.py`) and pass `sync_json` rather than
assuming.

This node implements only the *poses* stage of the wider capture pipeline. It
does not rotate frames (rigs mounted in portrait with no container rotation
metadata need `extract_synced_frames.py --rotate` first), undistort them, or
colour-correct them. For a rig that needs those, pre-process into a frame tree
and come in through **Load Flipbook** instead.

Mixed rigs are the norm — witness and top-down cameras often differ in
resolution or frame rate from the body array. `drop_odd_formats` keeps the
majority format and says what it dropped; turning it off makes a mixed rig an
error instead, because one dataset cannot hold two image sizes.

### Viewing the result

ComfyUI has no `.sogst` viewer, and the format's own player is a browser build,
so rather than ship a viewer this pack meets ComfyUI's existing one. **Preview
SOGST** evaluates every gaussian at a chosen clip time — centre moved along its
velocity, opacity scaled by its temporal window — and hands the result to the
stock `SPLAT` type:

```
Bake SOGST ──ply_path──> Preview SOGST ──splat──> Render Splat ──> Preview Image
```

Scrubbing `time_seconds` (or `frame_index`) plays the clip back. `Render Splat`
with `frames` above 1 orbits the camera, so one node gives a turntable batch to
feed a video node; its `depth`, `normal` and `clay` styles work too, as do
`Splat to File 3D` → `Preview 3D` and the other stock splat nodes.

The three things the format specification warns fail silently are all pinned by
tests: the temporal envelope is **unnormalised**, `t_sigma` is a **standard
deviation** not a variance, and `a` is the **raw `dt²` coefficient** with no
factor of ½.

### Settings that matter

- **`enable_rcp True` (the default) with `views_per_group 4`** on Generate Ring.
  RCP generates four anchor views the whole ring is conditioned on; without it
  each denoising group invents its own far side and the back of the ring will
  not reconstruct (measured: scene swaps, 1.6x saturation swings). With group 4
  and the sdpa pin it peaks at ~28.6 GiB on a 32 GB card; group 6 does not fit.
- **Input length.** 4DAnyone always generates exactly 121 frames. The clip must
  supply that many after `start_time` — about 5 s at 24 fps — at 720p or better.
  The node checks this and says what is missing rather than failing an hour in.
- **`test_cameras`** on Build 4DGS Dataset. Left empty, the dataset duplicates a
  training camera into the test split, so that PSNR is a training-view monitor
  and not a held-out score.
- **`sh_degree`** on Train 4DGS: content-dependent. 2 measured better on one
  real subject capture (30.87 vs 30.04 dB); 3 measured better on a generated
  ring (23.2 vs 21.0 dB held-out). When in doubt, try both — the runs are an
  hour each and the fingerprint cache keeps whichever you keep.
- **`mask_filter_root`** on Bake SOGST enables the lifetime mask-consistency
  filter, which drops silhouette-escaping splats. It is junk removal, not a
  quality regulariser.

## Command line

The dataset half also runs headless, with the same code the nodes use:

```bash
python scripts/cumuli_export.py \
    --result_dir <4DAnyone>/data/fdanyone/<run> \
    --out_dir    /datasets/<run> \
    --masks comfyui --comfyui_root <ComfyUI> \
    --downscale 2 --test_cameras 12
```

`--stop_after {stage,masks,dataset}` stops early; `--dry_run` reports the plan.

## Tests

```bash
pytest tests/                      # 80 unit tests, no GPU, no model loads
ruff check .
```

The tests cover the command lines, every input-validation path, the fps mirror
of 4DAnyone's own `choose_canonical_fps`, subprocess streaming and cancellation,
the OpenCV-to-OpenGL camera conversion against a real published rig, and the
dataset contract.

## Provenance

`cumuli_bridge/dataset4d.py` and `cumuli_bridge/validate.py` are vendored
from the cumuli pipeline (`scripts/build_flipbook_4dgs_dataset.py`,
`scripts/build_4dgs_dataset.py`, `scripts/validate_stage_output.py`), same
copyright holder and licence, adapted from CLI scripts into libraries so the
pack stays in one environment. `bake_sogst.py` is driven in place rather than
copied: it tracks the `.sogst` specification, and a stale copy would silently
emit a non-conforming container.
