# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

## What this is

A ComfyUI custom node pack that takes one monocular video to a streamable
`.sogst` 4D gaussian splat. It is not a self-contained program: it *drives*
three external checkouts (4DAnyone, the OMG4 rotor-4DGS trainer, and cumuli's
`bake_sogst.py`) whose locations come from configuration. Read `README.md`
first — it documents the node chain, the required external checkouts, the
additive pip installs, and the settings that matter empirically.

The pack is developed out of tree and symlinked into `<ComfyUI>/custom_nodes/`.

## Commands

Everything runs in ComfyUI's own conda environment (`comfyenv` on the reference
workstation), so use that interpreter — not the system python:

```bash
~/miniconda3/envs/comfyenv/bin/python -m pytest tests/ -q   # 80 tests, no GPU, no model loads
~/miniconda3/envs/comfyenv/bin/python -m pytest tests/ -q -k progress   # one test / subset
ruff check .
```

`tests/pytest.ini` pins `--import-mode=importlib` and keeps rootdir inside
`tests/`, because the pack root carries an `__init__.py` (ComfyUI imports the
whole directory as a package) that pytest would otherwise try to collect.

Headless run of the dataset half, using the same library code the nodes use:

```bash
python scripts/cumuli_export.py --result_dir <4DAnyone>/data/fdanyone/<run> \
    --out_dir /datasets/<run> --masks comfyui --comfyui_root <ComfyUI> \
    --downscale 2 --test_cameras 12
# --stop_after {stage,masks,dataset} stops early; --dry_run reports the plan
```

## Architecture

`cumuli_bridge/nodes.py` is the only ComfyUI-aware module. It defines
fourteen V3 `IO.ComfyNode` classes registered through
`CumuliExtension.get_node_list()`,
exported from the package root as `comfy_entrypoint`. Every other module is
plain Python over numpy/Pillow/PyAV and is importable without ComfyUI — which
is what lets `scripts/cumuli_export.py` reuse them. Keep it that way: put
ComfyUI imports (`folder_paths`, `comfy.*`, `comfy_api`, `server`) in
`nodes.py`, and guard the one exception (`vram.py`'s `comfy.model_management`)
with try/ImportError as it already is.

Data flows through the graph as three opaque custom types — `CUMULI_RING`,
`CUMULI_FLIPBOOK`, `CUMULI_DATASET` — backed by `ring.RingResult`,
`flipbook.Flipbook` and `dataset4d.DatasetHandle`. Those classes are the only
things that know their respective directory layouts; nothing downstream should
reach into paths directly. Each type has exactly one Load node and no Save
node: the artifacts are disk-native, so the producers *are* the save nodes.

Two standing policies (user directives — do not regress them):

- **Single environment.** Everything runs in comfyenv; subprocesses use
  `sys.executable`. Never add env-selection knobs (a `trainer_python` was
  added once and removed). The trainer's CUDA extensions
  (diff_gaussian_rasterization, simple_knn, pointops2) are pre-built into
  comfyenv — build with an nvcc that knows the GPU arch and `--no-deps`, or
  pip will upgrade torch out from under ComfyUI.
- **ComfyUI-native caching, no toggles.** No `on_existing`/`overwrite`/
  skip-if-exists widgets. Nodes re-run only when inputs change; expensive
  stages extend this to disk via input fingerprints (`runner.write_stamp` /
  `prepare_artifact_dir`). A stale stamped artifact is replaced silently; an
  unstamped (foreign) directory is never deleted — it errors with guidance.
  Seed-only changes must keep the GVHMR motion cache.

Module map, in pipeline order:

| module | role |
|---|---|
| `settings.py` | Locates the external checkouts. Resolution order: defaults → JSON config → env vars, re-read every node run. |
| `vram.py` | Unloads ComfyUI's models and enforces a free-VRAM floor before a 45 min job. |
| `process.py` | The subprocess primitive: merged stdout/stderr on one pipe, child in its own process group so Cancel kills the tree. |
| `progress.py` | Parses 4DAnyone's mixed logging + tqdm output into one monotonic 0..1 fraction. |
| `runner.py` | Builds and validates the 4DAnyone argv; all pre-flight input checks live here. |
| `sfm.py` | Real-capture entry: solves a physical rig into poses by driving cumuli's `multiframe_sfm.py` in place, then stages the capture as a flipbook. |
| `ring.py` | Read model of a finished result directory (`cameras.json`, `videos/dense/NN.mp4`, …). |
| `flipbook.py` | Transposes N view-major videos into frame-major staging dirs. |
| `masks.py` | BiRefNet mattes in-process, through the `BACKGROUND_REMOVAL` socket. |
| `dataset4d.py` | Visual-hull init cloud + RGBA training frames + `transforms_train/test.json`. |
| `validate.py` | Acceptance check of that dataset against the trainer's contract. |
| `train.py` | Drives the OMG4 trainer and `bake_sogst.py` as child processes; parses their progress. |
| `sogst.py` | Evaluates a baked 4D PLY at one instant into ComfyUI's native `SPLAT`. |
| `videoio.py` | PyAV probe/decode helpers shared by the preview and export paths. |

## Invariants that fail silently

These are the things the code and tests exist to protect. Do not "simplify"
them away.

- **Subprocesses are same-interpreter by design.** Ring generation and training
  run as children of `sys.executable`, not of another environment. The reason
  is CUDA, not dependencies: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  only takes effect before a process's first CUDA allocation, and both stages
  want the whole device. `conda_env` restores launching a different env.
- **`.sogst` temporal math.** The envelope is *unnormalised* (no
  `1/sqrt(2πσ²)`), `t_sigma` is a standard deviation not a variance, and `a` is
  the raw `dt²` coefficient with no factor of ½. All three are pinned by tests
  in `tests/test_bridge.py`.
- **The rig solve never touches the capture.** `multiframe_sfm.py --videos_dir`
  *globs the directory* and has no camera-subset flag, so `sfm.link_cameras`
  stages symlinks for the selected cameras and the solve is pointed at those.
  Pointing it at the capture would silently solve every camera in it, including
  the odd-format ones `check_uniform` just rejected. Never call cumuli's
  `run_hloc.py` either: it restructures a flat image directory in place with
  `Path.replace`, moving the caller's files.
- **Seed the solve as PINHOLE, not OPENCV.** There is no distortion estimate to
  seed, and the dataset builder rejects nonzero distortion; a distortion term
  fitted from a fiction would reach training as images that disagree with their
  own intrinsics. `build_flipbook_transforms` re-checks and refuses rather than
  quietly dropping a coefficient.
- **Never create a `sparse/` directory** beside a built dataset: the trainer's
  scene loader checks for one first and silently switches to the COLMAP reader.
- **Masks are never dilated** — dilated alpha supervision teaches a bright
  silhouette fringe.
- **`bake_sogst.py` and `sogst_ply.py` are driven in place, never copied.** They
  track the `.sogst` specification; a stale copy emits a non-conforming
  container without erroring.
- **Vendored files** — `dataset4d.py` and `validate.py`, from the cumuli
  pipeline — keep their upstream behaviour. The only intended deltas are
  argparse→function, `print`→`logging`, and `sys.exit`→typed exceptions.
- **Camera labels are fixed-width zero-padded decimal**, width derived from the
  camera count, so `sorted(labels)` orders correctly past 100 views.
- **Fail before the GPU job, not during it.** `runner.py` validates clip length,
  resolution and fps up front precisely so an hour is not wasted.

## Conventions

- Every source file carries the two-line SPDX + Required Notice header
  (PolyForm-Noncommercial-1.0.0, Solipsist Studios Inc.).
- The ruff config mirrors the ComfyUI checkout so `ruff check .` agrees at both
  roots. `T` is selected: **`print` is banned, use `LOGGER`** (`logging.getLogger("comfyui-cumuli")`).
- Each module raises its own typed error (`SettingsError`, `RingError`,
  `FlipbookError`, `MaskError`, `DatasetError`, `ValidationError`,
  `TrainingError`, `SogstError`, `SubprocessError`/`SubprocessCancelled`);
  `nodes.py` catches them at the boundary. Error messages are expected to say
  what is missing and what to do, not just what failed.
- Long-running nodes report through `_progress_reporter` and poll
  `_cancelled()`; new long stages should do both.
- `config.json` is gitignored local machine state; `config.example.json` is the
  template and must stay in sync with `DEFAULTS` in `settings.py`.
- **Cumuli is the pack's nomenclature; 4DAnyone is one model it drives.**
  Node ids and classes are `Cumuli<Verb><Noun>` and match each other, display
  names are `Cumuli <words>`, the category is `Cumuli`, socket types are
  `CUMULI_*`, HTTP routes are `/cumuli/...`, the ComfyUI settings keys are
  `cumuli.*`, this pack's env vars are `CUMULI_*`, and the logger is
  `comfyui-cumuli`. The name 4DAnyone stays only where it names the external
  project: the `fdanyone_root` setting and `CUMULI_FDANYONE_ROOT`, the
  `<data_dir>/fdanyone/<run>` result layout and the `fdanyone.*` python modules
  inside that checkout, the `FDANYONE_*` env vars that checkout itself reads
  (`FDANYONE_PERSISTENT_PARAMS`, `FDANYONE_ATTENTION_BACKEND`), and the
  `Cumuli Generate Ring (4DAnyone)` label. Never rename those to `CUMULI_*`:
  they are another program's interface.
