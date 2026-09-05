# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Drive the rotor 4DGS trainer and the ``.sogst`` bake.

Both run as child processes of *this* interpreter (``sys.executable``), not of
another environment. The reasons are the same ones that keep ring generation
out-of-process:

* the trainer's renderer JIT-compiles a CUDA extension on first import and then
  owns the device for tens of minutes, and
* a segfault in a CUDA extension should not take the ComfyUI server with it.

The trainer config is generated from cumuli's own
``configs/gs4d_pretrain_template.yaml`` by ``str.format_map``, exactly as
``run_unified_pipeline.py:stage_train4d`` does, so a run started here and a run
started by the orchestrator produce the same config. Resume semantics are
preserved too; caching decisions belong to the node's input fingerprints.

Two upstream details this module works around, both verified by reading the
trainer:

* ``train_scratch.py`` imports ``scene``/``gaussian_renderer``/``utils`` as
  top-level packages, so the child must run with ``cwd`` at the trainer repo.
* ``recursive_merge`` asserts every YAML key exists as a parser attribute, so
  the generated config must stay a strict subset of the template's keys.
"""

from __future__ import annotations

import logging
import os
import re

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .process import CommandResult, run_streaming
from .settings import BridgeSettings, SettingsError

LOGGER = logging.getLogger("comfyui-cumuli")

CONFIG_NAME = "gs4d_config.yaml"
MODEL_DIRNAME = "train4d_output"
SOGST_NAME = "splat_4d.sogst"

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
    """The knobs cumuli's own train4d stage exposes, with its defaults."""

    out_dir: Path
    dataset_dir: Path
    duration_seconds: float
    fps: float = 24.0
    iterations: int = 30000
    num_pts: int = 100000
    batch_size: int = 2
    densify_until_iter: int = 25000
    densify_until_num_points: int = 3000000
    t_init_div: int = 100
    sh_degree: int = 3
    trainer_config: Path | None = None

    @property
    def model_dir(self) -> Path:
        return self.out_dir / MODEL_DIRNAME

    @property
    def config_path(self) -> Path:
        return self.out_dir / CONFIG_NAME

    @property
    def checkpoint(self) -> Path:
        return self.model_dir / f"chkpnt{self.iterations}.pth"

    def validate(self) -> None:
        if self.iterations < 1:
            raise TrainingError(f"iterations must be positive, got {self.iterations}.")
        if self.duration_seconds <= 0:
            raise TrainingError(f"duration_seconds must be positive, got {self.duration_seconds}.")
        if self.batch_size < 1:
            raise TrainingError(f"batch_size must be at least 1, got {self.batch_size}.")
        if not (self.dataset_dir / "transforms_train.json").is_file():
            raise TrainingError(
                f"{self.dataset_dir} has no transforms_train.json; build the dataset before training."
            )
        if (self.dataset_dir / "sparse").exists():
            raise TrainingError(
                f"{self.dataset_dir}/sparse exists, which makes the trainer use its COLMAP reader "
                "instead of the D-NeRF one. Remove it."
            )


def find_template(settings: BridgeSettings) -> Path:
    """cumuli's trainer config template, next to its OMG4 submodule."""

    candidates = [
        settings.trainer_root.parent.parent / "configs" / "gs4d_pretrain_template.yaml",
        settings.trainer_root / "configs" / "gs4d_pretrain_template.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise TrainingError(
        "Could not find gs4d_pretrain_template.yaml. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Point trainer_root at a checkout whose parent holds cumuli's configs/, or pass a "
        "trainer_config explicitly."
    )


def write_config(settings: BridgeSettings, options: TrainOptions) -> Path:
    """Fill cumuli's template. Kept as text substitution, as upstream does."""

    if options.trainer_config is not None:
        LOGGER.info("Using caller-supplied trainer config %s", options.trainer_config)
        return Path(options.trainer_config)
    template = find_template(settings).read_text()
    options.out_dir.mkdir(parents=True, exist_ok=True)
    options.config_path.write_text(
        template.format_map(
            {
                "time_max": f"{options.duration_seconds:.6f}",
                "num_pts": options.num_pts,
                "batch_size": options.batch_size,
                "source_path": str(options.dataset_dir.resolve()),
                "model_path": str(options.model_dir.resolve()),
                "iterations": options.iterations,
                "densify_until_iter": options.densify_until_iter,
                "densify_until_num_points": options.densify_until_num_points,
            }
        )
    )
    if options.sh_degree != 3:
        # sh_degree 2 measured better than 3 on subject captures; the template
        # hard-codes 3, so rewrite that one line rather than fork the template.
        text = options.config_path.read_text().replace("sh_degree: 3", f"sh_degree: {options.sh_degree}")
        options.config_path.write_text(text)
    LOGGER.info("Wrote trainer config %s", options.config_path)
    return options.config_path


def build_train_argv(settings: BridgeSettings, options: TrainOptions, config: Path) -> list[str]:
    return list(settings.launcher()) + [
        str(settings.trainer_entrypoint),
        "--config",
        str(config),
        # Explicit iteration lists make the checkpoint filename deterministic.
        "--test_iterations",
        str(options.iterations),
        "--save_iterations",
        str(options.iterations),
    ]


def cuda_toolchain_env(base: Path = Path("/usr/local"), environ: dict | None = None) -> dict[str, str]:
    """CUDA_HOME/PATH for torch's JIT extension builds.

    The trainer compiles OMG4's *modified* rotor-4D rasterizer on first use
    (``gaussian_renderer/diff_gaussian_rasterization.py`` calls
    ``cpp_extension.load`` -- the pip-installable package is a different,
    unmodified rasterizer, so the JIT is mandatory). torch falls back to
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
    if options.t_init_div:
        # Initial temporal sigma is sqrt(duration / div). Upstream's default of 5
        # bakes several frames of motion smear into the starting sigma.
        env["GS4D_T_INIT_DIV"] = str(options.t_init_div)
    env.update(settings.trainer_env)
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
    """Train to ``options.iterations``, or skip if that checkpoint already exists."""

    options.validate()
    try:
        settings.validate_trainer()
    except SettingsError as exc:
        raise TrainingError(str(exc)) from None

    config = write_config(settings, options)
    argv = build_train_argv(settings, options, config)
    state = TrainProgress(total_iterations=options.iterations)

    def handle(line: str) -> None:
        LOGGER.debug("trainer| %s", line)
        if state.update(line) and on_progress is not None:
            on_progress(state)

    result = run_streaming(
        argv,
        cwd=settings.trainer_root,  # the trainer imports scene/ utils/ as top-level packages
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
    checkpoint: Path
    output: Path
    duration_seconds: float
    fps: float = 24.0
    mask_filter_root: Path | None = None
    sh_clamp: float = 3.0
    filter_corrupted: bool = False
    shn_count: int = 65536
    segment_duration: float = 0.1
    webp_method: int = 4
    #: Also write the 4D interchange PLY. It comes from the same post-filter
    #: arrays as the container, so the two cannot disagree, and it is what the
    #: preview node reads.
    emit_ply: Path | None = None


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
    argv = list(settings.launcher()) + [
        str(script),
        "--input",
        str(options.checkpoint),
        "--output",
        str(options.output),
        "--time_min",
        "0",
        "--time_max",
        f"{options.duration_seconds:.6f}",
        "--fps",
        str(options.fps),
        "--sh_clamp",
        str(options.sh_clamp),
        "--shn_count",
        str(options.shn_count),
        "--segment_duration",
        str(options.segment_duration),
        "--webp_method",
        str(options.webp_method),
    ]
    if not options.filter_corrupted:
        # The bad_color/garbage filters were calibrated for the legacy SVQ
        # pipeline and delete healthy splats from an explicit-SH checkpoint.
        argv.append("--no_filter_corrupted")
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
    """Bake a trained checkpoint into a ``.sogst`` container."""

    if not options.checkpoint.is_file():
        raise TrainingError(f"Checkpoint not found: {options.checkpoint}")
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


