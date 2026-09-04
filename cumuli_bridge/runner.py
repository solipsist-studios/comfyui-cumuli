# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Build, validate and execute one 4DAnyone inference run."""

from __future__ import annotations

import logging
import os
import re
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .process import CommandResult, run_streaming
from .progress import ProgressState
from .settings import BridgeSettings
from .videoio import VideoProbe, probe

LOGGER = logging.getLogger("comfyui-cumuli")

#: Fallback canonical clip length; the real value is read from the checkout.
DEFAULT_NUM_FRAMES = 121

#: Rates 4DAnyone will downsample to when ``target_fps`` is ``auto``.
AUTO_DOWNSAMPLE_FPS = (
    Fraction(24, 1),
    Fraction(24000, 1001),
    Fraction(25, 1),
    Fraction(30, 1),
    Fraction(30000, 1001),
)
_INTEGER_RATIO_TOLERANCE = 1e-3

#: 4DAnyone always generates at this resolution; inputs below it get upscaled.
MIN_INPUT_SHORT_SIDE = 704
RECOMMENDED_INPUT_SHORT_SIDE = 720

VALID_VIEWS_PER_GROUP = (4, 6)
MIN_PITCH = -15
MAX_PITCH = 45

_NUM_FRAMES_RE = re.compile(r"^\s*num_frames\s*:\s*int\s*=\s*(\d+)", re.MULTILINE)
_SAFE_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ValidationError(RuntimeError):
    """Raised for an input the pipeline would reject, before any GPU work."""


def canonical_num_frames(settings: BridgeSettings) -> int:
    """Read the fixed clip length out of the checkout, falling back to 121."""

    config = settings.fdanyone_root / "fdanyone" / "config.py"
    try:
        match = _NUM_FRAMES_RE.search(config.read_text())
    except OSError:
        return DEFAULT_NUM_FRAMES
    return int(match.group(1)) if match else DEFAULT_NUM_FRAMES


def choose_canonical_fps(input_rate: Fraction) -> Fraction:
    """Mirror of ``fdanyone.video.choose_canonical_fps``.

    Duplicated rather than imported because the 4DAnyone package cannot be
    imported into this interpreter; kept side by side with a test that pins the
    known cases (48 -> 24, 50 -> 25, 60 -> 30, 120 -> 30, 40 -> 40).
    """

    if input_rate <= 0:
        raise ValidationError(f"Input frame rate must be positive, got {input_rate}.")
    divisible: list[tuple[Fraction, int]] = []
    for candidate in AUTO_DOWNSAMPLE_FPS:
        ratio = float(input_rate / candidate)
        multiple = round(ratio)
        if multiple >= 2 and abs(ratio - multiple) <= _INTEGER_RATIO_TOLERANCE:
            divisible.append((candidate, multiple))
    if divisible:
        _, multiple = max(divisible, key=lambda match: float(match[0]))
        return input_rate / multiple
    return input_rate


def parse_layer_pitches(value: str | Sequence[int]) -> tuple[int, ...]:
    """Accept ``"15"``, ``"-10,15,35"`` or ``"[-10, 15]"`` and return a tuple."""

    if not isinstance(value, str):
        pitches = tuple(int(item) for item in value)
    else:
        cleaned = value.strip().strip("[]()")
        if not cleaned:
            raise ValidationError("layer_pitches must list at least one pitch in degrees, for example 15.")
        parts = [part.strip() for part in cleaned.replace(";", ",").split(",") if part.strip()]
        try:
            pitches = tuple(int(part) for part in parts)
        except ValueError:
            raise ValidationError(f"layer_pitches must be whole degrees, got {value!r}.") from None
    if not pitches:
        raise ValidationError("layer_pitches must list at least one pitch in degrees, for example 15.")
    if len(set(pitches)) != len(pitches):
        raise ValidationError(f"layer_pitches must not repeat a pitch, got {list(pitches)}.")
    bad = [pitch for pitch in pitches if not MIN_PITCH <= pitch <= MAX_PITCH]
    if bad:
        raise ValidationError(f"Each layer pitch must be between {MIN_PITCH} and {MAX_PITCH} degrees, got {bad}.")
    return pitches


def resolve_views_per_group(value: str | int, views_per_layer: int) -> int | str:
    """Validate ``views_per_group`` the way ``fdanyone.views`` will."""

    if isinstance(value, str) and value.strip().lower() == "auto":
        divisors = [size for size in VALID_VIEWS_PER_GROUP if views_per_layer % size == 0]
        if not divisors:
            raise ValidationError(f"views_per_layer ({views_per_layer}) must be divisible by 4 or 6.")
        return "auto"
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"views_per_group must be 'auto', 4 or 6, got {value!r}.") from None
    if size not in VALID_VIEWS_PER_GROUP:
        raise ValidationError(f"views_per_group must be 'auto', 4 or 6, got {value!r}.")
    if views_per_layer % size:
        raise ValidationError(f"views_per_layer ({views_per_layer}) must be divisible by views_per_group ({size}).")
    return size


def stage_source_video(settings: BridgeSettings, video_path: str | Path, run_name: str) -> Path:
    """Return a path whose stem is ``run_name``, linking the input if needed.

    4DAnyone derives both the result directory and the GVHMR motion cache key
    from ``Path(video_path).stem``, so renaming a run means presenting the same
    bytes under a different filename. A symlink under ``<data_dir>/source`` does
    that without copying gigabytes, and keeps the cache reuse explicit: the same
    ``run_name`` reuses the same motion solve.
    """

    source = Path(video_path).expanduser().resolve()
    run_name = run_name.strip()
    if not run_name or run_name == source.stem:
        return source
    if not _SAFE_RUN_NAME.match(run_name):
        raise ValidationError(
            f"run_name must start with a letter or digit and contain only letters, digits, '.', '_' or '-', "
            f"got {run_name!r}."
        )
    staged = settings.source_root / f"{run_name}{source.suffix}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.is_symlink():
        if staged.resolve() == source:
            return staged
        staged.unlink()
    elif staged.exists():
        if staged.samefile(source):
            return staged
        raise ValidationError(f"{staged} already exists and is not the requested input video.")
    staged.symlink_to(source)
    return staged


def materialize_video_source(video: object, run_name: str, source_root: Path) -> Path:
    """Turn a ComfyUI VIDEO input into a file 4DAnyone can read.

    Duck-typed against comfy_api's VideoInput so this module stays importable
    without ComfyUI. Two paths:

    * A file-backed video with no active trim window is used in place -- the
      bytes on disk are the source of truth, so the GVHMR motion cache (keyed
      by stem, validated by mtime and size) behaves exactly as if the user had
      typed video_path.
    * Anything else (a trim window, an in-memory stream, frames synthesized by
      upstream nodes) must be written out. That file lands at
      ``<source_root>/<run_name>.mp4``, so ``run_name`` is required: a random
      temp name would silently defeat both the result-directory policy and the
      motion cache. When the freshly written bytes are identical to what is
      already staged, the existing file is kept so its mtime -- and with it the
      motion-cache validation -- survives re-queues.
    """

    trim = (0.0, 0.0)
    get_trim = getattr(video, "get_active_trim_window", None)
    if callable(get_trim):
        trim = tuple(get_trim())
    get_source = getattr(video, "get_stream_source", None)
    source = get_source() if callable(get_source) else None

    if not any(trim) and isinstance(source, (str, os.PathLike)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise ValidationError(f"The connected video points at a missing file: {path}")
        return path.resolve()

    run_name = (run_name or "").strip()
    if not run_name:
        raise ValidationError(
            "The connected video is trimmed or not backed by a file on disk, so it must be written "
            "out under a stable name. Set run_name; it keys the staged file, the result directory "
            "and the GVHMR motion cache."
        )
    if not _SAFE_RUN_NAME.match(run_name):
        raise ValidationError(
            f"run_name must start with a letter or digit and contain only letters, digits, '.', '_' or '-', "
            f"got {run_name!r}."
        )

    target = source_root / f"{run_name}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{run_name}.tmp-{os.getpid()}.mp4")
    try:
        if hasattr(source, "read"):  # BytesIO-backed: keep the original container bytes
            source.seek(0)
            with open(tmp, "wb") as handle:
                shutil.copyfileobj(source, handle)
        else:
            save_to = getattr(video, "save_to", None)
            if not callable(save_to):
                raise ValidationError(
                    f"The connected video ({type(video).__name__}) offers neither a file source nor save_to()."
                )
            save_to(str(tmp))
        if target.is_file() and _files_identical(tmp, target):
            tmp.unlink()
        else:
            os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _files_identical(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ca = fa.read(1 << 20)
            cb = fb.read(1 << 20)
            if ca != cb:
                return False
            if not ca:
                return True


@dataclass(frozen=True)
class RunRequest:
    """One fully specified inference job."""

    video_path: Path
    views_per_layer: int = 24
    layer_pitches: tuple[int, ...] = (15,)
    start_yaw: int = 0
    yaw_span: int = 360
    views_per_group: int | str = 4
    enable_rcp: bool = True
    enable_tcr: bool = True
    start_time: float = 0.0
    target_fps: str = "auto"
    lora_path: str = ""
    prompt: str = ""
    seed: int = 42
    device: str = "cuda:0"

    @property
    def run_name(self) -> str:
        return self.video_path.stem

    @property
    def num_target_views(self) -> int:
        return self.views_per_layer * len(self.layer_pitches)

    def to_dict(self) -> dict:
        return {
            "video_path": str(self.video_path),
            "views_per_layer": self.views_per_layer,
            "layer_pitches": list(self.layer_pitches),
            "start_yaw": self.start_yaw,
            "yaw_span": self.yaw_span,
            "views_per_group": self.views_per_group,
            "enable_rcp": self.enable_rcp,
            "enable_tcr": self.enable_tcr,
            "start_time": self.start_time,
            "target_fps": self.target_fps,
            "seed": self.seed,
            "device": self.device,
        }


def build_request(
    *,
    video_path: str | Path,
    views_per_layer: int,
    layer_pitches: str | Sequence[int],
    start_yaw: int,
    yaw_span: int,
    views_per_group: str | int,
    enable_rcp: bool,
    enable_tcr: bool,
    start_time: float,
    target_fps: str,
    seed: int,
    device: str,
) -> RunRequest:
    """Validate every knob and produce a :class:`RunRequest`."""

    views_per_layer = int(views_per_layer)
    if views_per_layer <= 0:
        raise ValidationError(f"views_per_layer must be positive, got {views_per_layer}.")
    pitches = parse_layer_pitches(layer_pitches)
    group = resolve_views_per_group(views_per_group, views_per_layer)
    yaw_span = int(yaw_span)
    if not 0 < yaw_span <= 360:
        raise ValidationError(f"yaw_span must be between 1 and 360 degrees, got {yaw_span}.")
    start_yaw = (int(start_yaw) + 180) % 360 - 180
    if int(seed) < 0:
        raise ValidationError(f"seed must be non-negative, got {seed}.")
    if float(start_time) < 0:
        raise ValidationError(f"start_time must be non-negative, got {start_time}.")
    fps = str(target_fps).strip() or "auto"
    if fps.lower() != "auto":
        try:
            if Fraction(fps) <= 0:
                raise ValueError
        except (ValueError, ZeroDivisionError):
            raise ValidationError(f"target_fps must be 'auto' or a positive number, got {target_fps!r}.") from None
        fps = fps.lower()
    else:
        fps = "auto"

    return RunRequest(
        # Absolute but NOT resolved: ``stage_source_video`` renames a run by
        # presenting the same bytes through a symlink whose stem is the run
        # name, and 4DAnyone derives the result directory and the GVHMR cache
        # key from that stem. Calling ``resolve()`` here would follow the link
        # back to the original filename and silently write over the original
        # run's results.
        video_path=Path(os.path.abspath(os.path.expanduser(str(video_path)))),
        views_per_layer=views_per_layer,
        layer_pitches=pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        views_per_group=group,
        enable_rcp=bool(enable_rcp),
        enable_tcr=bool(enable_tcr),
        start_time=float(start_time),
        target_fps=fps,
        seed=int(seed),
        device=str(device),
    )


def build_argv(settings: BridgeSettings, request: RunRequest) -> list[str]:
    """Assemble the exact ``inference.py`` command line.

    ``fire`` literal-evaluates ``--flag=value``, so lists and booleans are
    written in Python syntax and passed as single argv entries. Nothing goes
    through a shell.
    """

    pitches = "[" + ",".join(str(pitch) for pitch in request.layer_pitches) + "]"
    argv = list(settings.launcher())
    argv += [
        str(settings.inference_script),
        f"--video_path={request.video_path}",
        f"--views_per_layer={request.views_per_layer}",
        f"--layer_pitches={pitches}",
        f"--start_yaw={request.start_yaw}",
        f"--yaw_span={request.yaw_span}",
        f"--views_per_group={request.views_per_group}",
        f"--enable_rcp={bool(request.enable_rcp)}",
        f"--enable_tcr={bool(request.enable_tcr)}",
        f"--data_dir={settings.data_dir}",
        f"--model_dir={settings.model_dir}",
        f"--gvhmr_root={settings.gvhmr_root}",
        f"--device={request.device}",
        f"--target_fps={request.target_fps}",
        f"--start_time={request.start_time}",
        f"--seed={request.seed}",
    ]
    return argv


def build_env(settings: BridgeSettings, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    # ComfyUI's own torch/lib paths must not leak into the other environment.
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    env.update(settings.subprocess_env)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items() if v})
    return env


@dataclass
class VideoRequirements:
    """What the input clip has versus what the fixed 121-frame contract needs."""

    probe: VideoProbe
    canonical_fps: Fraction
    required_source_frames: int
    available_source_frames: int
    num_frames: int

    @property
    def ok(self) -> bool:
        return self.available_source_frames >= self.required_source_frames

    @property
    def required_seconds(self) -> float:
        return float(self.num_frames / self.canonical_fps)


def check_video(settings: BridgeSettings, request: RunRequest) -> VideoRequirements:
    """Confirm the input can supply one canonical clip from ``start_time``."""

    info = probe(request.video_path)
    if info.width <= 0 or info.height <= 0:
        raise ValidationError(f"Could not read the frame size of {request.video_path}.")
    num_frames = canonical_num_frames(settings)

    if request.target_fps == "auto":
        canonical = choose_canonical_fps(info.fps)
    else:
        canonical = Fraction(request.target_fps)

    # The decoder picks source frames nearest to a canonical clock, so the run
    # needs num_frames canonical periods of source material after start_time.
    span_seconds = float((num_frames - 1) / canonical)
    required = int(round(span_seconds * float(info.fps))) + 1
    available = info.frames_from(request.start_time)

    requirements = VideoRequirements(
        probe=info,
        canonical_fps=canonical,
        required_source_frames=required,
        available_source_frames=available,
        num_frames=num_frames,
    )
    if not requirements.ok:
        raise ValidationError(
            f"{request.video_path.name} cannot supply a {num_frames}-frame clip starting at "
            f"{request.start_time:.2f}s. 4DAnyone always generates exactly {num_frames} frames at "
            f"{float(canonical):.3f} fps, which needs {requirements.required_seconds:.2f}s "
            f"({required} source frames at {float(info.fps):.3f} fps) but only {available} remain. "
            "Use a longer clip, or lower start_time."
        )
    short_side = min(info.width, info.height)
    if short_side < MIN_INPUT_SHORT_SIDE:
        raise ValidationError(
            f"{request.video_path.name} is {info.width}x{info.height}. Generation is fixed at "
            f"704x1280, so the short side must be at least {MIN_INPUT_SHORT_SIDE}px "
            f"({RECOMMENDED_INPUT_SHORT_SIDE}px or more is recommended)."
        )
    if short_side < RECOMMENDED_INPUT_SHORT_SIDE:
        LOGGER.warning(
            "4DAnyone input %s is %dx%d; 720p or better is recommended.",
            request.video_path.name,
            info.width,
            info.height,
        )
    return requirements



def motion_is_cached(settings: BridgeSettings, request: RunRequest) -> bool:
    """GVHMR motion is keyed by the video stem and reused across runs."""

    return (settings.motion_dir(request.run_name) / "motion.json").is_file()


@dataclass
class RunOutcome:
    result_dir: Path
    command: list[str]
    result: CommandResult | None = None
    log_tail: list[str] = field(default_factory=list)


def execute(
    settings: BridgeSettings,
    request: RunRequest,
    *,
    on_progress: Callable[[ProgressState], None] | None = None,
    on_line: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> RunOutcome:
    """Validate, then run 4DAnyone to completion.

    The caller (the node) has already applied fingerprint caching: when this
    function runs, any stale result directory has been cleared."""

    settings.validate()
    check_video(settings, request)
    result_dir = settings.result_dir(request.run_name)
    argv = build_argv(settings, request)
    state = ProgressState(expected_views=request.num_target_views)

    def handle(line: str) -> None:
        LOGGER.debug("4danyone| %s", line)
        if on_line is not None:
            on_line(line)
        if state.update(line) and on_progress is not None:
            on_progress(state)

    result = run_streaming(
        argv,
        cwd=settings.fdanyone_root,
        env=build_env(settings, {
            "FDANYONE_LORA_PATH": request.lora_path,
            "FDANYONE_PROMPT": request.prompt,
        }),
        on_line=handle,
        should_cancel=should_cancel,
    )
    if not (result_dir / "metadata.json").is_file():
        raise RuntimeError(
            f"4DAnyone finished with status 0 but published no result at {result_dir}. "
            "The last lines of its output were:\n" + "\n".join(result.lines[-15:])
        )
    return RunOutcome(result_dir=result_dir, command=argv, result=result, log_tail=result.lines)


def extract_lora_factors(patches: dict) -> tuple[dict, int, list[str]]:
    """Fold a ComfyUI ModelPatcher patch stack into per-weight LoRA factors.

    ``patches`` maps ``diffusion_model.<key>.weight`` to a list of
    ``(strength, patch, strength_model, offset, function)`` entries, where
    ``patch`` is either a modern weight-adapter object (``.name == "lora"``,
    ``.weights == (up, down, alpha, mid, dora, reshape)``) or the legacy
    ``("lora", (...))`` tuple. Strength and alpha are folded into the up
    factor; stacked LoRAs concatenate along the rank axis, so each key ends
    up with exactly one ``up @ down`` product. Returns ``(factors, merged,
    unsupported)`` where factors maps the bare 4DAnyone state-dict key (no
    ``diffusion_model.`` prefix) to ``(up, down)`` fp16 tensors, and
    unsupported lists patch descriptions this fold cannot express (the caller
    should refuse loudly rather than half-apply a model).
    """

    import torch

    factors: dict = {}
    unsupported: list[str] = []
    for key, entries in patches.items():
        if not key.endswith(".weight"):
            unsupported.append(f"{key}: only .weight patches are supported")
            continue
        base = key[: -len(".weight")]
        if base.startswith("diffusion_model."):
            base = base[len("diffusion_model."):]
        ups, downs = [], []
        for entry in entries:
            strength, patch = float(entry[0]), entry[1]
            offset, function = entry[3], entry[4]
            if offset is not None or function is not None:
                unsupported.append(f"{key}: offset/function patches")
                continue
            if hasattr(patch, "weights") and getattr(patch, "name", "") == "lora":
                weights = patch.weights
            elif isinstance(patch, (tuple, list)) and len(patch) == 2 and patch[0] == "lora":
                weights = patch[1]
            else:
                unsupported.append(f"{key}: {type(patch).__name__}")
                continue
            up, down, alpha = weights[0], weights[1], weights[2]
            mid = weights[3] if len(weights) > 3 else None
            dora = weights[4] if len(weights) > 4 else None
            if mid is not None or dora is not None:
                unsupported.append(f"{key}: mid/dora weights")
                continue
            up = up.to(torch.float32).reshape(up.shape[0], -1)
            down = down.to(torch.float32).reshape(down.shape[0], -1)
            rank = down.shape[0]
            scale = strength * ((float(alpha) / rank) if alpha is not None else 1.0)
            ups.append(up * scale)
            downs.append(down)
        if ups:
            factors[base] = (
                torch.cat(ups, dim=1).to(torch.float16),
                torch.cat(downs, dim=0).to(torch.float16),
            )
    return factors, len(factors), unsupported


def save_lora_factors(factors: dict, path: Path) -> Path:
    """Write folded factors as ``<key>.up`` / ``<key>.down`` safetensors."""

    from safetensors.torch import save_file

    flat = {}
    for base, (up, down) in factors.items():
        flat[f"{base}.up"] = up.contiguous()
        flat[f"{base}.down"] = down.contiguous()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(path))
    return path


# --- ComfyUI-native caching: input fingerprints stamped onto disk artifacts --------

FINGERPRINT_FILE = ".bridge_fingerprint.json"
MASKS_FINGERPRINT_FILE = ".bridge_fingerprint.masks.json"


def compute_fingerprint(parts: dict) -> str:
    """Stable digest of a dict of JSON-able parts (Paths become strings)."""

    import hashlib
    import json as _json

    def norm(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): norm(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [norm(v) for v in value]
        return value

    payload = _json.dumps(norm(parts), sort_keys=True, separators=(",", ":"))
    return hashlib.md5(payload.encode()).hexdigest()


def file_identity(path: str | Path) -> dict:
    stat = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_stamp(root: Path, filename: str = FINGERPRINT_FILE) -> dict | None:
    import json as _json

    try:
        return _json.loads((Path(root) / filename).read_text())
    except (OSError, ValueError):
        return None


def write_stamp(root: Path, fingerprint: str, filename: str = FINGERPRINT_FILE, **extras) -> None:
    import json as _json

    payload = {"fingerprint": fingerprint, **extras}
    (Path(root) / filename).write_text(_json.dumps(payload, indent=1))


def prepare_artifact_dir(root: Path, fingerprint: str) -> str:
    """Return ``"reuse"`` when the on-disk artifact matches the inputs, else
    clear the way and return ``"build"``.

    Mirrors ComfyUI's own cache semantics on disk: same inputs -> same output,
    reused without asking; changed inputs -> the stale artifact is replaced.
    A directory that exists but carries no stamp was not produced by this
    bridge, and is never deleted -- that is the one case that still errors.
    """

    root = Path(root)
    if not root.exists():
        return "build"
    stamp = read_stamp(root)
    if stamp is None:
        if not any(root.iterdir()):
            return "build"
        raise ValidationError(
            f"{root} exists but was not produced by this bridge (no {FINGERPRINT_FILE}). "
            "Move or delete it, or load it explicitly with the Cumuli Load Ring node."
        )
    if stamp.get("fingerprint") == fingerprint:
        return "reuse"
    LOGGER.info("Inputs changed; replacing stale artifact %s", root)
    shutil.rmtree(root)
    return "build"


def ring_fingerprint(request: RunRequest) -> tuple[str, str]:
    """Return ``(fingerprint, motion_key)`` for one ring request.

    ``motion_key`` covers only what the GVHMR motion solve depends on (the
    clip bytes and how it is windowed), so seed- or view-layout-only changes
    keep the motion cache while still regenerating the ring.
    """

    lora = {}
    if request.lora_path:
        import hashlib

        lora = {"md5": hashlib.md5(Path(request.lora_path).read_bytes()).hexdigest()}
    clip = {
        "video": file_identity(request.video_path),
        "start_time": float(request.start_time),
        "target_fps": str(request.target_fps),
    }
    motion_key = compute_fingerprint(clip)
    full = compute_fingerprint({
        "clip": clip,
        "views_per_layer": request.views_per_layer,
        "layer_pitches": list(request.layer_pitches),
        "start_yaw": request.start_yaw,
        "yaw_span": request.yaw_span,
        "views_per_group": str(request.views_per_group),
        "enable_rcp": request.enable_rcp,
        "enable_tcr": request.enable_tcr,
        "seed": request.seed,
        "prompt": request.prompt,
        "lora": lora,
    })
    return full, motion_key


def clear_stale_motion(settings: BridgeSettings, request: RunRequest, motion_key: str,
                       previous_stamp: dict | None) -> bool:
    """Drop the GVHMR motion cache when the clip it solved no longer matches.

    The pipeline hard-errors on a stale motion cache, so clearing it here turns
    a confusing failure 45 minutes in into a clean re-solve. Returns True when
    the cache was removed.
    """

    motion_dir = settings.motion_dir(request.run_name)
    if not motion_dir.exists():
        return False
    if previous_stamp is not None:
        if previous_stamp.get("motion_key") == motion_key:
            return False
        shutil.rmtree(motion_dir)
        return True
    # No stamp to compare against: fall back to the identity GVHMR itself records.
    import json as _json

    try:
        meta = _json.loads((motion_dir / "motion.json").read_text())
        stat = Path(request.video_path).stat()
        if meta.get("source_size_bytes") == stat.st_size and meta.get("source_mtime_ns") == stat.st_mtime_ns:
            return False
    except (OSError, ValueError, KeyError):
        pass
    shutil.rmtree(motion_dir)
    return True



def discover_flipbooks(settings: BridgeSettings) -> list[str]:
    """Existing staged flipbook trees under work_root and flipbook_roots."""

    roots = ([settings.work_root] if settings.work_root else []) + list(settings.flipbook_roots)
    found = [
        str(path) for path in _candidate_dirs(roots)
        if (path / "frame_0000" / "transforms.json").is_file()
    ]
    return sorted(set(found))


def _candidate_dirs(roots) -> list[Path]:
    seen: list[Path] = []
    for raw in roots:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        seen.append(root)
        try:
            children = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
        except OSError:
            continue
        for child in children:
            seen.append(child)
            for name in ("dataset_4dgs", "flipbook_src"):
                nested = child / name
                if nested.is_dir():
                    seen.append(nested)
    return seen


def discover_datasets(settings: BridgeSettings) -> list[str]:
    """Existing 4DGS datasets under work_root and the configured dataset_roots."""

    roots = ([settings.work_root] if settings.work_root else []) + list(settings.dataset_roots)
    found = [
        str(path) for path in _candidate_dirs(roots)
        if (path / "transforms_train.json").is_file() and (path / "points3d.ply").is_file()
    ]
    return sorted(set(found))


def discover_flipbooks(settings: BridgeSettings) -> list[str]:
    """Existing staged flipbook trees under work_root and flipbook_roots."""

    roots = ([settings.work_root] if settings.work_root else []) + list(settings.flipbook_roots)
    found = [
        str(path) for path in _candidate_dirs(roots)
        if (path / "frame_0000" / "transforms.json").is_file()
    ]
    return sorted(set(found))
