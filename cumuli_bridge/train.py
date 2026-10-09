# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Drive the cumuli 4DGS trainer and the ``.sogst`` bake.

Both run as child processes of *this* interpreter (``sys.executable``), not of
another environment. The reasons are the same ones that keep ring generation
out-of-process:

* the trainer's renderer JIT-compiles a CUDA extension on first use and then
  owns the device for the length of the run, and
* a segfault in a CUDA extension should not take the ComfyUI server with it.

The trainer is the installed ``cumuli_trainer`` package, started as
``python -m cumuli_trainer --config <run>/gs4d_config.yaml``. The config is
JSON (valid YAML) and carries only keys the trainer knows: it rejects unknown
ones, so a typo cannot be silently ignored. Caching decisions belong to the
node's input fingerprints.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import logging
import os
import re

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .process import CommandResult, SubprocessCancelled, run_streaming
from .settings import BridgeSettings, SettingsError
from .vram import free_bytes, require_free_vram

LOGGER = logging.getLogger("comfyui-cumuli")

CONFIG_NAME = "gs4d_config.yaml"
MODEL_DIRNAME = "train4d_output"
SOGST_NAME = "splat_4d.sogst"
#: The trainer's output: the 4D interchange PLY the bake reads.
PLY_NAME = "splat_4d.ply"

#: ``Training progress:  33%|###   | 10000/30000 [12:34<25:08, 13.3it/s, Loss=…, PSNR=…]``
_TRAIN_BAR = re.compile(r"Training progress:\s*\d{1,3}%\|.*?\|\s*(\d+)\s*/\s*(\d+)")
_PSNR = re.compile(r"PSNR=([\d.]+)")
_LOSS = re.compile(r"Loss=([\d.]+)")
_SAVING = re.compile(r"\[ITER (\d+)\] Saving Gaussians")

#: Bake milestones, in the order ``bake_sogst.py`` prints them.
_BAKE_STEPS = (
    ("Loading checkpoint", "loading checkpoint"),
    ("Slicing 4D covariance", "slicing 4D covariance"),
    ("Folding temporal SH", "folding temporal SH"),
    ("Mask-consistency filter", "mask-consistency filter"),
    ("SH overshoot clamp", "clamping SH overshoot"),
    ("Pruning:", "pruning by peak alpha"),
    ("Interchange PLY", "writing interchange PLY"),
)


class TrainingError(RuntimeError):
    """Raised when training or baking cannot start or does not finish."""


# --------------------------------------------------------------------------
# progress
# --------------------------------------------------------------------------
@dataclass
class TrainProgress:
    """Monotonic 0..1 across the training child, with the trainer's own metrics."""

    total_iterations: int = 30000
    fraction: float = 0.0
    iteration: int = 0
    psnr: float | None = None
    loss: float | None = None
    message: str = "starting the trainer"

    def update(self, line: str) -> bool:
        text = line.strip()
        if not text:
            return False
        match = _TRAIN_BAR.search(text)
        if match:
            self.iteration = int(match.group(1))
            self.total_iterations = max(1, int(match.group(2)))
            psnr = _PSNR.search(text)
            loss = _LOSS.search(text)
            if psnr:
                self.psnr = float(psnr.group(1))
            if loss:
                self.loss = float(loss.group(1))
            value = min(1.0, self.iteration / self.total_iterations)
            if value > self.fraction:
                self.fraction = value
            detail = f" psnr {self.psnr:.2f}" if self.psnr is not None else ""
            self.message = f"iteration {self.iteration}/{self.total_iterations}{detail}"
            return True
        if _SAVING.search(text):
            self.message = "saving the checkpoint"
            return True
        for marker, message in (
            ("Loading Training Cameras", "loading training cameras"),
            ("Number of points at initialisation", "initialising from points3d.ply"),
            ("Found transforms_train.json", "reading the dataset"),
            ("Building extension", "compiling the CUDA rasterizer (first run only)"),
        ):
            if marker in text:
                self.message = message
                return True
        return False


@dataclass
class BakeProgress:
    """Milestone progress for the bake, which prints steps but has no bar."""

    fraction: float = 0.0
    message: str = "starting the bake"
    _index: int = field(default=0, repr=False)

    def update(self, line: str) -> bool:
        text = line.strip()
        if not text:
            return False
        for index, (marker, message) in enumerate(_BAKE_STEPS):
            if marker in text:
                if index >= self._index:
                    self._index = index
                    self.fraction = max(self.fraction, (index + 1) / (len(_BAKE_STEPS) + 1))
                    self.message = message
                    return True
        return False


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass
class TrainOptions:
    """The trainer's knobs, named as the Train 4DGS widgets name them, with the
    trainer's own defaults. ``write_config`` says which config key each becomes."""

    out_dir: Path
    dataset_dir: Path
    duration_seconds: float
    fps: float = 24.0
    iterations: int = 30000
    #: ``representation.max_init_points``: a random subset of points3d.ply to
    #: start from. 0 starts from all of it.
    num_pts: int = 0
    #: ``cameras_per_step``: cameras rendered per step, all at one instant.
    batch_size: int = 2
    #: ``strategy.refine_stop``: the step after which Gaussians are no longer
    #: relocated or added.
    densify_until_iter: int = 25000
    #: ``strategy.cap_max``: the most Gaussians the strategy will grow to.
    densify_until_num_points: int = 300000
    t_init_div: int = 100
    sh_degree: int = 2
    #: ``background``. It must match what ``DatasetOptions.background``
    #: composited the held-out GT against, or PSNR/LPIPS score against the
    #: wrong backdrop. False (black) is the trainer's default.
    white_background: bool = False
    #: ``losses.opacity_mask``: charges rendered opacity wherever the per-view
    #: silhouette says background, regardless of colour, so it prunes floaters
    #: of any shade -- unlike matching the training background to the embed
    #: destination, which only substitutes one floater colour for another.
    #: 0 disables.
    lambda_opa_mask: float = 0.005
    #: ``export_min_contribution``: drop Gaussians whose largest contribution to
    #: any training view is below this many pixels. 0 keeps them all.
    export_min_contribution: float = 2.0
    trainer_config: Path | None = None

    @property
    def model_dir(self) -> Path:
        return self.out_dir / MODEL_DIRNAME

    @property
    def config_path(self) -> Path:
        return self.out_dir / CONFIG_NAME

    @property
    def checkpoint(self) -> Path:
        """What a finished run leaves behind: the PLY the bake reads."""
        return self.model_dir / PLY_NAME

    def validate(self) -> None:
        if self.export_min_contribution < 0:
            raise TrainingError(f"export_min_contribution must not be negative, got {self.export_min_contribution}.")
        if self.iterations < 1:
            raise TrainingError(f"iterations must be positive, got {self.iterations}.")
        if self.duration_seconds <= 0:
            raise TrainingError(f"duration_seconds must be positive, got {self.duration_seconds}.")
        if self.batch_size < 1:
            raise TrainingError(f"batch_size must be at least 1, got {self.batch_size}.")
        if self.num_pts < 0 or self.densify_until_num_points < 1 or self.densify_until_iter < 0:
            raise TrainingError(
                "num_pts (0 = all), densify_until_num_points (at least 1) and densify_until_iter must not be negative."
            )
        if not (self.dataset_dir / "transforms_train.json").is_file():
            raise TrainingError(
                f"{self.dataset_dir} has no transforms_train.json; build the dataset before training."
            )
        if (self.dataset_dir / "sparse").exists():
            raise TrainingError(
                f"{self.dataset_dir}/sparse exists, which makes the trainer use its COLMAP reader "
                "instead of the D-NeRF one. Remove it."
            )


def check_trainer(settings: BridgeSettings) -> None:
    """Fail before the GPU job if the trainer cannot be imported.

    Only checkable when the child will run in this interpreter; with
    ``conda_env``/``python_exe`` set, the child's own import error is reported.
    """

    try:
        settings.launcher()  # validates python_exe
    except SettingsError as exc:
        raise TrainingError(str(exc)) from None
    if settings.runs_in_this_environment:
        missing = [name for name in ("cumuli_trainer", "gsplat") if importlib.util.find_spec(name) is None]
        if missing:
            raise TrainingError(
                f"The 4DGS trainer needs {' and '.join(missing)} in ComfyUI's environment, and "
                f"{'it is' if len(missing) == 1 else 'they are'} not installed. Run ./install.sh "
                "(its 'trainer' group installs them without touching torch)."
            )


def trainer_config(options: TrainOptions) -> dict:
    """The trainer's config for these options. Each Train 4DGS widget becomes
    one key: num_pts -> ``representation.max_init_points``, batch_size ->
    ``cameras_per_step``, densify_until_iter -> ``strategy.refine_stop``,
    densify_until_num_points -> ``strategy.cap_max``, white_background ->
    ``background``, lambda_opa_mask -> ``losses.opacity_mask``."""

    representation: dict = {"name": "spacetime"}
    if options.num_pts > 0:
        representation["max_init_points"] = options.num_pts
    config = {
        "source_path": str(options.dataset_dir.resolve()),
        "model_path": str(options.model_dir.resolve()),
        "time_duration": [0.0, round(options.duration_seconds, 6)],
        "fps": options.fps,
        "iterations": options.iterations,
        "sh_degree": options.sh_degree,
        "test_iterations": [options.iterations],
        "save_iterations": [options.iterations],
        "cameras_per_step": options.batch_size,
        "background": "white" if options.white_background else "black",
        "representation": representation,
        "strategy": {
            "name": "mcmc",
            "refine_stop": options.densify_until_iter,
            "cap_max": options.densify_until_num_points,
        },
        "losses": {"opacity_mask": options.lambda_opa_mask},
        "export_min_contribution": options.export_min_contribution,
    }
    if options.t_init_div:
        config["t_init_div"] = options.t_init_div
    return config


def write_config(settings: BridgeSettings, options: TrainOptions) -> Path:
    """Write the trainer's config as JSON, which is valid YAML and cannot
    misquote a path. A caller-supplied config is used as it is."""

    if options.trainer_config is not None:
        LOGGER.info("Using caller-supplied trainer config %s", options.trainer_config)
        return Path(options.trainer_config)
    options.out_dir.mkdir(parents=True, exist_ok=True)
    options.config_path.write_text(json.dumps(trainer_config(options), indent=2) + "\n")
    LOGGER.info("Wrote trainer config %s", options.config_path)
    return options.config_path


def build_train_argv(settings: BridgeSettings, options: TrainOptions, config: Path) -> list[str]:
    return list(settings.launcher()) + ["-m", "cumuli_trainer", "--config", str(config)]


def cuda_toolchain_env(base: Path = Path("/usr/local"), environ: dict | None = None) -> dict[str, str]:
    """CUDA_HOME/PATH for torch's JIT extension builds.

    gsplat compiles its CUDA kernels on first use. torch falls back to
    ``which nvcc`` when CUDA_HOME is unset, and on many distros that is a
    stale apt CUDA that cannot target current GPUs. Prefer an explicit
    CUDA_HOME; otherwise pick the newest ``<base>/cuda-*`` that has nvcc.
    """

    environ = os.environ if environ is None else environ
    if environ.get("CUDA_HOME"):
        return {}

    def version_key(path: Path) -> list[int]:
        return [int(part) for part in re.findall(r"\d+", path.name)]

    for candidate in sorted(base.glob("cuda-*"), key=version_key, reverse=True):
        if (candidate / "bin" / "nvcc").is_file():
            return {
                "CUDA_HOME": str(candidate),
                "PATH": f"{candidate / 'bin'}:{environ.get('PATH', '')}",
            }
    return {}


def build_train_env(settings: BridgeSettings, options: TrainOptions) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(cuda_toolchain_env())
    return env


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------
@dataclass
class TrainOutcome:
    checkpoint: Path
    command: list[str]
    result: CommandResult | None = None
    final_psnr: float | None = None


def train(
    settings: BridgeSettings,
    options: TrainOptions,
    *,
    on_progress: Callable[[TrainProgress], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TrainOutcome:
    """Train to ``options.iterations``. Whether a finished run can be reused is
    the node's call, from its input fingerprints."""

    options.validate()
    check_trainer(settings)

    config = write_config(settings, options)
    argv = build_train_argv(settings, options, config)
    state = TrainProgress(total_iterations=options.iterations)

    def handle(line: str) -> None:
        LOGGER.debug("trainer| %s", line)
        if state.update(line) and on_progress is not None:
            on_progress(state)

    result = run_streaming(
        argv,
        cwd=options.out_dir,  # an installed module: it needs no particular working directory
        env=build_train_env(settings, options),
        on_line=handle,
        should_cancel=should_cancel,
    )
    if not options.checkpoint.is_file():
        raise TrainingError(
            f"The trainer exited cleanly but {options.checkpoint} does not exist. Last output:\n"
            + "\n".join(result.lines[-15:])
        )
    return TrainOutcome(
        checkpoint=options.checkpoint,
        command=argv,
        result=result,
        final_psnr=state.psnr,
    )


@dataclass
class BakeOptions:
    #: The trainer's ``splat_4d.ply``. It carries its own clip times and frame
    #: rate, so none are passed to the bake.
    checkpoint: Path
    output: Path
    mask_filter_root: Path | None = None
    shn_count: int = 65536
    segment_duration: float = 0.1
    webp_method: int = 4
    #: Also write the 4D interchange PLY. It comes from the same post-filter
    #: arrays as the container, so the two cannot disagree, and it is what the
    #: preview node reads.
    emit_ply: Path | None = None
    #: Bounds the coloured part of the view-dependent SH: removes the coloured
    #: glints a sparse rig leaves on opaque splats.
    sh_chroma_clamp: float = 0.02
    #: Softening: blur every splat by this many pixels at the training cameras
    #: (needs ``mask_filter_root``), and widen every splat's lifetime by this
    #: many frames. 0 turns either off.
    smooth_px: float = 0.5
    smooth_frames: int = 2


def find_bake_script(settings: BridgeSettings) -> Path:
    """cumuli's ``bake_sogst.py``. Driven in place rather than vendored: it is
    a thousand lines that track the ``.sogst`` spec, and a stale copy here would
    silently emit a non-conforming container."""

    try:
        return settings.pipeline_script("bake_sogst.py")
    except SettingsError as exc:
        raise TrainingError(str(exc)) from None


def build_bake_argv(settings: BridgeSettings, options: BakeOptions) -> list[str]:
    script = find_bake_script(settings)
    if options.smooth_px > 0 and options.mask_filter_root is None:
        raise TrainingError(
            "smooth_px needs the mask filter's dataset (the training cameras). Turn mask_filter on, "
            "or set smooth_px to 0."
        )
    argv = list(settings.launcher()) + [
        str(script),
        "--input",
        str(options.checkpoint),
        "--output",
        str(options.output),
        "--sh_chroma_clamp",
        str(options.sh_chroma_clamp),
        "--shn_count",
        str(options.shn_count),
        "--segment_duration",
        str(options.segment_duration),
        "--webp_method",
        str(options.webp_method),
    ]
    if options.smooth_px > 0:
        argv += ["--smooth_px", str(options.smooth_px)]
    if options.smooth_frames > 0:
        argv += ["--smooth_frames", str(options.smooth_frames)]
    if options.mask_filter_root is not None:
        argv += ["--mask_filter_root", str(options.mask_filter_root)]
    if options.emit_ply is not None:
        argv += ["--emit_ply", str(options.emit_ply)]
    return argv


def unpack_sogst_to_ply(settings: BridgeSettings, archive: Path, destination: Path | None = None) -> Path:
    """Turn a ``.sogst`` archive back into the 4D interchange PLY.

    Uses the pipeline's own ``sogst_ply.py`` rather than a reimplementation, so
    the unpack always matches the container revision that wrote the file. The
    result is cached beside the archive.
    """

    archive = Path(archive).expanduser()
    if not archive.is_file():
        raise TrainingError(f"Not found: {archive}")
    destination = Path(destination) if destination is not None else archive.with_suffix(".ply")
    if destination.is_file() and destination.stat().st_mtime >= archive.stat().st_mtime:
        LOGGER.info("reusing unpacked interchange PLY %s", destination)
        return destination

    script = find_bake_script(settings).parent / "sogst_ply.py"
    if not script.is_file():
        raise TrainingError(f"Could not find sogst_ply.py next to the bake script ({script}).")
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_streaming(
        list(settings.launcher()) + [str(script), "--input", str(archive), "--output", str(destination)],
        cwd=script.parent,
        env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")},
        on_line=lambda line: LOGGER.info("sogst_ply| %s", line),
    )
    if not destination.is_file():
        raise TrainingError(f"sogst_ply.py exited cleanly but {destination} was not written.")
    return destination


def bake(
    settings: BridgeSettings,
    options: BakeOptions,
    *,
    on_progress: Callable[[BakeProgress], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Bake a trained PLY into a ``.sogst`` container."""

    if not options.checkpoint.is_file():
        raise TrainingError(f"Trained PLY not found: {options.checkpoint}")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    argv = build_bake_argv(settings, options)
    state = BakeProgress()

    def handle(line: str) -> None:
        LOGGER.info("bake| %s", line)
        if state.update(line) and on_progress is not None:
            on_progress(state)

    script = find_bake_script(settings)
    run_streaming(
        argv,
        cwd=script.parent,  # bake_sogst.py adds its own directory to sys.path
        env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")},
        on_line=handle,
        should_cancel=should_cancel,
    )
    if not options.output.is_file():
        raise TrainingError(f"The bake exited cleanly but {options.output} was not written.")
    return options.output


# --------------------------------------------------------------------------
# windowed training: train several short clips, bounded by real VRAM capacity
# --------------------------------------------------------------------------
@dataclass
class WindowTrainSpec:
    """One window's training job: its own dataset slice and output directory."""

    index: int
    options: TrainOptions
    offset_seconds: float
    dataset_dir: Path


@dataclass
class WindowTrainOutcome:
    index: int
    checkpoint: Path
    duration_seconds: float
    fps: float
    offset_seconds: float
    dataset_dir: Path
    final_psnr: float | None = None


def clamp_parallel_windows(device: str, minimum_gb: float, requested: int) -> tuple[int, str]:
    """How many windows can safely train at once on this card.

    ``minimum_gb`` is a fixed per-window VRAM floor (30 GB on the reference
    32 GB card, where one training already claims the "whole card" per
    ``README.md``), so total device memory sets a hard ceiling of
    ``floor(total / minimum)`` on real concurrency regardless of what the
    node asks for. Computing that once, up front, and sizing the worker pool
    to it is what lets concurrency stay honest instead of racing every
    window's own VRAM check against its siblings: on the reference single-GPU
    workstation this clamps to 1 and windows train back-to-back exactly as a
    single wide fit would; it only rises above 1 on hardware (or a
    ``min_free_vram_gb``/``num_pts`` combination) that genuinely has room for
    more than one "whole card" job at a time.
    """

    requested = max(1, int(requested))
    if minimum_gb <= 0:
        return requested, "min_free_vram_gb is disabled (<=0); not clamping to VRAM capacity"
    _, total = free_bytes(device)
    total_gb = total / (1024**3)
    if total_gb <= 0:
        return requested, f"could not query {device}'s memory; not clamping to VRAM capacity"
    capacity = max(1, int(total_gb // minimum_gb))
    if capacity < requested:
        return capacity, (
            f"{device} has {total_gb:.1f} GB total; at a {minimum_gb:.1f} GB floor per window only "
            f"{capacity} can safely train at once, capped down from the requested {requested}"
        )
    return requested, (
        f"{device} has {total_gb:.1f} GB total; the requested {requested} concurrent window(s) fit "
        f"within the {minimum_gb:.1f} GB floor each"
    )


def train_windows(
    settings: BridgeSettings,
    specs: list[WindowTrainSpec],
    *,
    min_free_vram_gb: float,
    max_parallel: int = 1,
    on_progress: Callable[[int, TrainProgress], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[list[WindowTrainOutcome], str]:
    """Train every window's dataset slice with ``train()``, unchanged, bounded
    to a hardware-safe number running at once. Returns the outcomes in window
    order plus a note on how concurrency was decided (surfaced in the node's
    report so a clamp-down is visible, not silent)."""

    if not specs:
        raise TrainingError("train_windows called with no windows.")
    effective, note = clamp_parallel_windows(settings.device, min_free_vram_gb, max_parallel)

    def run_one(spec: WindowTrainSpec) -> WindowTrainOutcome:
        if should_cancel is not None and should_cancel():
            raise SubprocessCancelled(f"Cancelled before window {spec.index} started.")
        require_free_vram(settings.device, min_free_vram_gb)

        def progress(state: TrainProgress) -> None:
            if on_progress is not None:
                on_progress(spec.index, state)

        outcome = train(settings, spec.options, on_progress=progress, should_cancel=should_cancel)
        return WindowTrainOutcome(
            index=spec.index,
            checkpoint=outcome.checkpoint,
            duration_seconds=spec.options.duration_seconds,
            fps=spec.options.fps,
            offset_seconds=spec.offset_seconds,
            dataset_dir=spec.dataset_dir,
            final_psnr=outcome.final_psnr,
        )

    if effective <= 1 or len(specs) == 1:
        return [run_one(spec) for spec in specs], note

    pending = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective) as pool:
        for spec in specs:
            pending[pool.submit(run_one, spec)] = spec.index
        ordered: dict[int, WindowTrainOutcome] = {}
        for future in concurrent.futures.as_completed(pending):
            ordered[pending[future]] = future.result()
    return [ordered[spec.index] for spec in specs], note


# --------------------------------------------------------------------------
# merging windows: driven in place, exactly like bake_sogst.py
# --------------------------------------------------------------------------
@dataclass
class MergeOptions:
    #: (sogst path, offset in global seconds), in window order.
    segments: list[tuple[Path, float]]
    output: Path
    #: "fade" measured better on both PSNR and LPIPS than a hard cut, per
    #: merge_sogst_segments.py's own docstring; 0.35s is where it measured
    #: LPIPS was best (PSNR keeps improving out to 1.00s, which is a wide
    #: fade blurring across the seam -- 0.35s takes the LPIPS side of that).
    mode: str = "fade"
    fade_seconds: float = 0.35


def find_merge_script(settings: BridgeSettings) -> Path:
    """cumuli's ``merge_sogst_segments.py``, driven in place for the same
    reason ``bake_sogst.py`` is: it tracks the stitching math (temporal
    gating, time-codebook re-snapping) precisely, and its own docstring
    documents a past bug (dropping higher-order SH during the merge) that a
    stale local copy could silently reintroduce."""

    try:
        return settings.pipeline_script("merge_sogst_segments.py")
    except SettingsError as exc:
        raise TrainingError(str(exc)) from None


def build_merge_argv(settings: BridgeSettings, options: MergeOptions) -> list[str]:
    script = find_merge_script(settings)
    argv = list(settings.launcher()) + [
        str(script),
        "--out",
        str(options.output),
        "--mode",
        options.mode,
    ]
    if options.mode == "fade":
        argv += ["--fade", str(options.fade_seconds)]
    for path, offset in options.segments:
        argv += ["--segment", str(path), f"{offset:.6f}"]
    return argv


def merge_windows(
    settings: BridgeSettings,
    options: MergeOptions,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Stitch each window's baked ``.sogst`` into one archive over global time.

    Seam placement is left to the script's own default (the midpoint between
    each pair of adjacent windows) -- the same rule the research this feature
    is based on always used, and what a plan-driven run would compute anyway.
    """

    if len(options.segments) < 2:
        raise TrainingError("merge_windows needs at least two window .sogst files.")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    script = find_merge_script(settings)
    argv = build_merge_argv(settings, options)
    lines: list[str] = []

    def handle(line: str) -> None:
        LOGGER.info("merge| %s", line)
        lines.append(line)

    run_streaming(
        argv,
        cwd=script.parent,  # merge_sogst_segments.py imports sibling modules by relative path
        env={k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")},
        on_line=handle,
        should_cancel=should_cancel,
    )
    if not options.output.is_file():
        raise TrainingError(
            f"merge_sogst_segments.py exited cleanly but {options.output} was not written. Last output:\n"
            + "\n".join(lines[-15:])
        )
    return options.output
