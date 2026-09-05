# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Solve a static multi-camera rig into camera poses with HLOC + pycolmap.

This is the entry point for *real captures*. The 4DAnyone path synthesises a
ring whose cameras are known by construction; footage off a physical rig
arrives with no poses at all, and every downstream stage here needs them.

The solve itself is cumuli's ``scripts/multiframe_sfm.py``, driven **in place**
by absolute path for the same reason ``train.py`` drives ``bake_sogst.py`` in
place: it tracks pycolmap's API and its rotation-averaging has already been
corrected once, so a vendored copy would rot silently.

Why multi-timestamp rather than one instant: the rig is static, so every video
timestamp is an independent measurement of the same camera poses. Tracks then
span space *and* time, all images from one physical camera share one COLMAP
camera, and people moving in the scene fall out as cross-time outliers. The
cost is quadratic -- see :func:`pair_count` -- because the upstream script
pairs exhaustively with no retrieval step.

Two things this module deliberately does *not* do:

* It never calls cumuli's ``run_hloc.py``. That wrapper restructures a flat
  image directory in place with ``Path.replace`` -- it **moves the caller's
  files**, which is indefensible for a node fed a path the user typed.
* It never writes into the capture directory. Everything lands under
  ``outputs_dir``; the source videos are opened read-only.

Everything runs in ComfyUI's own interpreter. ``hloc``, ``pycolmap`` and
``lightglue`` are installed into it directly, so the child process is
``sys.executable`` -- a new process, the same environment.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from .process import run_streaming
from .settings import SettingsError
from .videoio import VideoError, probe

LOGGER = logging.getLogger("comfyui-cumuli")

#: Video containers a rig capture plausibly ships as. Matches the upstream
#: script's own set, so a file we accept is a file it can read.
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi")

TRANSFORMS_NAME = "transforms_multiframe.json"
REPORT_NAME = "report.json"
POINTS_NAME = "background_points.ply"
INIT_TRANSFORMS_NAME = "init_transforms.json"

#: Focal length as a fraction of image width, used when the rig ships no
#: calibration. 1.2 is cumuli's own guess for the Canon array
#: (``configs/ericpare_array.json``); COLMAP refines it from there.
DEFAULT_FOCAL_GUESS = 1.2

#: Above this many image pairs the exhaustive matcher stops being a job and
#: becomes a wall. 200 cameras at one timestamp is 19,900 pairs and fine; the
#: same rig at four timestamps is 319,600 and is not. Checked up front because
#: this pack fails before the expensive job, not during it.
MAX_PAIRS = 40_000

_FEATURE_TYPES = ("superpoint", "aliked")

#: Per-camera rotation disagreement across timestamps, in degrees, above which
#: the static-rig premise has visibly failed. From the capture guide.
MAX_ROT_SPREAD_DEG = 0.05

#: Mean track length below which the static scene did not really triangulate.
MIN_TRACK_LENGTH = 8.0


class SfmError(RuntimeError):
    """Raised when a rig cannot be solved, or the inputs would not survive it."""


@dataclass(frozen=True)
class CameraSource:
    """One physical camera, as one video file.

    ``label`` is the file stem, which is what the upstream script uses as the
    COLMAP camera name. Rig exports are normally already zero padded
    (``0002.mov`` -> ``"0002"``), so ``sorted(labels)`` orders correctly.
    """

    label: str
    path: Path
    width: int
    height: int
    num_frames: int
    fps: Fraction

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass(frozen=True)
class SolveOptions:
    """Everything the solve needs that is not derived from the videos."""

    videos_dir: Path
    outputs_dir: Path
    num_timestamps: int = 1
    feature_type: str = "aliked"
    resize_max: int = 1920
    max_keypoints: int = 8192
    focal_guess: float = DEFAULT_FOCAL_GUESS
    refine_intrinsics: bool = True
    refine_principal_point: bool = False
    filter_max_reproj_error: float = 4.0
    max_rot_spread_deg: float = 10.0
    sync_json: Path | None = None

    def validated(self) -> SolveOptions:
        if self.feature_type not in _FEATURE_TYPES:
            raise SfmError(f"feature_type must be one of {_FEATURE_TYPES}, got {self.feature_type!r}.")
        if self.num_timestamps < 1:
            raise SfmError(f"num_timestamps must be at least 1, got {self.num_timestamps}.")
        if self.resize_max < 256:
            raise SfmError(f"resize_max is far too small to match on, got {self.resize_max}.")
        if self.max_keypoints < 256:
            raise SfmError(f"max_keypoints is far too small to reconstruct from, got {self.max_keypoints}.")
        if not 0.1 <= self.focal_guess <= 10.0:
            raise SfmError(
                f"focal_guess is a multiple of image width and should sit near 1.2, got {self.focal_guess}."
            )
        return self


@dataclass(frozen=True)
class CameraQuality:
    """Per-camera pose repeatability across the sampled timestamps.

    These are the numbers the capture guide treats as the health metric: a
    camera whose pose disagrees with itself between timestamps did not solve,
    whatever the global reprojection error says.
    """

    label: str
    num_views: int
    num_inliers: int
    center_spread: float
    rot_spread_deg: float

    @property
    def solved(self) -> bool:
        return self.num_inliers > 0


@dataclass(frozen=True)
class RigSolve:
    """A finished solve on disk, described well enough to stage a flipbook."""

    root: Path
    transforms_path: Path
    report_path: Path
    points_path: Path | None
    labels: tuple[str, ...]
    num_points: int
    mean_reprojection_error: float
    mean_track_length: float
    num_registered_images: int
    cameras: tuple[CameraQuality, ...]

    @property
    def num_cameras(self) -> int:
        return len(self.labels)

    def unsolved(self) -> tuple[CameraQuality, ...]:
        return tuple(camera for camera in self.cameras if not camera.solved)

    def worst(self, count: int = 5) -> tuple[CameraQuality, ...]:
        ranked = sorted(self.cameras, key=lambda camera: camera.rot_spread_deg, reverse=True)
        return tuple(ranked[:count])

    def unstable(self, threshold: float = MAX_ROT_SPREAD_DEG) -> tuple[CameraQuality, ...]:
        """Cameras whose pose disagrees with itself across timestamps.

        The rig is static, so a camera should solve to the same pose at every
        sampled instant. Disagreement means the static-rig premise did not
        hold: most often the clips are not actually frame-aligned, so the same
        frame index is a different real moment on different cameras.
        """

        return tuple(c for c in self.cameras if c.rot_spread_deg > threshold and c.num_views > 1)

    def summary(self) -> str:
        """A short QA readout, in the terms the capture guide uses.

        Reprojection error alone is not a verdict: a solve can look sharp and
        still be wrong. Track length and per-camera pose repeatability are what
        actually say whether the rig reconstructed.
        """

        lines = [
            f"{self.num_cameras} cameras, {self.num_registered_images} images registered, "
            f"{self.num_points} points",
            f"mean reprojection error {self.mean_reprojection_error:.3f} px "
            f"(expect < 2), mean track length {self.mean_track_length:.2f} (expect > 8)",
        ]
        if self.mean_track_length and self.mean_track_length < MIN_TRACK_LENGTH:
            lines.append(
                f"WARNING: mean track length {self.mean_track_length:.2f} is below {MIN_TRACK_LENGTH}. "
                "The static scene was barely triangulated -- add timestamps, raise resize_max, "
                "or expect too few init points to train from."
            )
        unsolved = self.unsolved()
        if unsolved:
            listed = ", ".join(camera.label for camera in unsolved[:8])
            more = "" if len(unsolved) <= 8 else f" (+{len(unsolved) - 8} more)"
            lines.append(f"UNSOLVED cameras ({len(unsolved)}): {listed}{more}")

        measured = [c for c in self.cameras if c.num_views > 1]
        if not measured:
            lines.append(
                "NOTE: one timestamp per camera, so pose repeatability could not be measured. "
                "A single-instant solve cannot tell you whether it is right."
            )
        else:
            spread = max(c.rot_spread_deg for c in measured)
            unstable = self.unstable()
            lines.append(f"worst rotation spread {spread:.4f} deg (expect < {MAX_ROT_SPREAD_DEG})")
            if unstable:
                listed = ", ".join(c.label for c in unstable[:8])
                more = "" if len(unstable) <= 8 else f" (+{len(unstable) - 8} more)"
                lines.append(
                    f"WARNING: {len(unstable)} of {len(measured)} cameras disagree with themselves "
                    f"across timestamps ({listed}{more}). The rig is static, so this means the "
                    "clips are not frame-aligned: measure the offsets and pass sync_json, rather "
                    "than assuming the trigger synced them."
                )
        return "\n".join(lines)


def pair_count(num_cameras: int, num_timestamps: int) -> int:
    """Image pairs the exhaustive matcher will build.

    The upstream script uses ``pairs_from_exhaustive`` with no retrieval, so
    this is the whole cost model: every image against every other.
    """

    images = max(0, num_cameras) * max(0, num_timestamps)
    return images * (images - 1) // 2


def find_sfm_script(settings) -> Path:
    """cumuli's ``multiframe_sfm.py``, driven in place.

    Located by ``cumuli_root``; see ``BridgeSettings.pipeline_script`` for the
    legacy fallbacks it still accepts.
    """

    try:
        return settings.pipeline_script("multiframe_sfm.py")
    except SettingsError as exc:
        raise SfmError(str(exc)) from None


def discover_cameras(
    videos_dir: str | Path,
    *,
    include: Sequence[str] | None = None,
    stride: int = 1,
    limit: int | None = None,
) -> tuple[CameraSource, ...]:
    """Probe every video in ``videos_dir`` and return them ordered by label.

    ``include`` selects specific labels; ``stride`` then thins the rig evenly
    (``stride=8`` on a 200 camera array keeps 25 cameras spread all the way
    around it, which is the shape that actually reconstructs). ``limit`` caps
    the count after thinning.
    """

    root = Path(videos_dir).expanduser()
    if not root.is_dir():
        raise SfmError(f"Capture directory not found: {root}")
    if stride < 1:
        raise SfmError(f"stride must be at least 1, got {stride}.")

    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise SfmError(
            f"{root} holds no video files ({', '.join(VIDEO_EXTS)}). "
            "Point this at the rig's movies directory, one file per camera."
        )
    if include is not None:
        wanted = {str(label) for label in include}
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            raise SfmError(f"{root} has no video for camera {sorted(missing)[0]!r}.")
    files = files[::stride]
    if limit is not None:
        files = files[:limit]

    cameras: list[CameraSource] = []
    for path in files:
        try:
            info = probe(path)
        except VideoError as exc:
            raise SfmError(f"Could not probe {path.name}: {exc}") from None
        if not info.width or not info.height:
            raise SfmError(f"{path.name} reports no frame size; it is not a usable camera clip.")
        cameras.append(
            CameraSource(
                label=path.stem,
                path=path.resolve(),
                width=info.width,
                height=info.height,
                num_frames=info.num_frames,
                fps=info.fps,
            )
        )
    if len(cameras) < 2:
        raise SfmError(f"A rig solve needs at least two cameras; {root} yielded {len(cameras)}.")
    return tuple(cameras)


def group_cameras(cameras: Iterable[CameraSource]) -> dict[tuple[int, int, str], list[str]]:
    """Bucket cameras by (width, height, fps). One bucket means a uniform rig."""

    groups: dict[tuple[int, int, str], list[str]] = {}
    for camera in cameras:
        groups.setdefault((camera.width, camera.height, str(camera.fps)), []).append(camera.label)
    return groups


def check_uniform(cameras: Sequence[CameraSource]) -> None:
    """Reject a rig whose cameras disagree on format.

    The dataset builder downstream needs one image size per dataset, and a
    camera shooting at a different frame rate does not share a timeline with
    the rest. Mixed rigs are common -- witness and top cameras often differ --
    so the error names the odd ones out and tells the caller to drop them
    rather than silently solving a rig that cannot become a dataset.
    """

    groups = group_cameras(cameras)
    if len(groups) <= 1:
        return
    ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    (majority_key, majority) = ordered[0]
    detail = "; ".join(
        f"{width}x{height}@{fps}: {len(labels)} cameras ({', '.join(labels[:4])}"
        + ("..." if len(labels) > 4 else "")
        + ")"
        for (width, height, fps), labels in ordered
    )
    odd = [label for _, labels in ordered[1:] for label in labels]
    raise SfmError(
        "This rig is not format-uniform, so it cannot become one 4DGS dataset. "
        f"Found {len(groups)} formats -- {detail}. "
        f"The majority is {majority_key[0]}x{majority_key[1]}@{majority_key[2]} ({len(majority)} cameras); "
        f"exclude the other {len(odd)} (e.g. {', '.join(odd[:6])}) and solve again."
    )


def select_majority_format(cameras: Sequence[CameraSource]) -> tuple[tuple[CameraSource, ...], tuple[str, ...]]:
    """Keep the largest format group, and report what was dropped.

    Rigs mix formats more often than not -- witness and top-down cameras run at
    a different size or frame rate than the body array. Returns
    ``(kept, dropped_labels)`` so the caller can say so out loud rather than
    quietly solving a different rig than the one that was asked for.
    """

    groups = group_cameras(cameras)
    if len(groups) <= 1:
        return tuple(cameras), ()
    majority_key = max(groups.items(), key=lambda item: len(item[1]))[0]
    kept = tuple(c for c in cameras if (c.width, c.height, str(c.fps)) == majority_key)
    dropped = tuple(c.label for c in cameras if (c.width, c.height, str(c.fps)) != majority_key)
    return kept, dropped


def check_scale(num_cameras: int, num_timestamps: int) -> None:
    """Refuse a solve whose exhaustive matching would never finish."""

    pairs = pair_count(num_cameras, num_timestamps)
    if pairs <= MAX_PAIRS:
        return
    affordable = 1
    while pair_count(num_cameras, affordable + 1) <= MAX_PAIRS:
        affordable += 1
    raise SfmError(
        f"{num_cameras} cameras at {num_timestamps} timestamps is {pairs:,} image pairs, "
        f"over the {MAX_PAIRS:,} this matcher can pair exhaustively (it has no retrieval step). "
        f"Use num_timestamps={affordable} at this camera count, or thin the rig with a larger stride."
    )


def common_frame_count(cameras: Sequence[CameraSource]) -> int:
    """Frames every camera has. Hardware sync still leaves ragged trims."""

    return min(camera.num_frames for camera in cameras)


def build_init_transforms(cameras: Sequence[CameraSource], focal_guess: float) -> dict:
    """Seed intrinsics for the solve: a centred pinhole guess per camera.

    Only the intrinsics are read upstream -- poses are what we are solving for
    -- so ``transform_matrix`` is identity and is never consulted. The model is
    PINHOLE rather than OPENCV precisely because there is no distortion
    estimate to seed: claiming zero distortion in an OPENCV model would let the
    refinement wander from a fiction, and the dataset builder downstream
    rejects nonzero distortion anyway.
    """

    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    frames = []
    for camera in cameras:
        focal = float(focal_guess) * camera.width
        frames.append(
            {
                "camera_label": camera.label,
                "file_path": f"{camera.label}/0000.png",
                "camera_model": "PINHOLE",
                "transform_matrix": identity,
                "fl_x": focal,
                "fl_y": focal,
                "cx": camera.width / 2.0,
                "cy": camera.height / 2.0,
                "w": camera.width,
                "h": camera.height,
            }
        )
    return {"camera_model": "PINHOLE", "frames": frames}


def write_init_transforms(cameras: Sequence[CameraSource], focal_guess: float, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_init_transforms(cameras, focal_guess), indent=1))
    return path


def link_cameras(cameras: Sequence[CameraSource], staging_dir: str | Path) -> Path:
    """Symlink the selected cameras into their own directory, and return it.

    The upstream script takes a *directory* and globs every video in it: it has
    no camera-subset flag. Pointing it straight at the capture would therefore
    solve the whole rig no matter which cameras were selected -- including the
    odd-format ones :func:`check_uniform` just rejected. Staging links gives
    exact control over the camera set while keeping the capture read-only, and
    costs nothing on disk.
    """

    staging = Path(staging_dir)
    if staging.exists():
        for stale in staging.iterdir():
            if stale.is_symlink() or stale.is_file():
                stale.unlink()
    staging.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        link = staging / f"{camera.label}{camera.path.suffix}"
        link.symlink_to(camera.path)
    return staging


def build_solve_argv(
    settings, script: Path, options: SolveOptions, init_transforms: Path, videos_dir: Path
) -> list[str]:
    """The child command line.

    ``settings.launcher()`` is ``[sys.executable]`` by default, so this is a
    new process in the same environment. ``videos_dir`` is the staged link
    directory, not the capture -- see :func:`link_cameras`.
    """

    argv = list(settings.launcher()) + [
        str(script),
        "--videos_dir",
        str(videos_dir),
        "--init_transforms",
        str(init_transforms),
        "--outputs_dir",
        str(options.outputs_dir),
        "--num_timestamps",
        str(options.num_timestamps),
        "--feature_type",
        options.feature_type,
        "--resize_max",
        str(options.resize_max),
        "--max_keypoints",
        str(options.max_keypoints),
        "--filter_max_reproj_error",
        str(options.filter_max_reproj_error),
        "--max_rot_spread_deg",
        str(options.max_rot_spread_deg),
    ]
    if options.refine_intrinsics:
        argv.append("--refine_intrinsics")
    if options.refine_principal_point:
        argv.append("--refine_principal_point")
    if options.sync_json is not None:
        argv += ["--sync_json", str(options.sync_json)]
    return argv


def solve_fingerprint(options: SolveOptions, cameras: Sequence[CameraSource]) -> str:
    """Digest of everything that changes the solve.

    Camera identity is (label, size, frame count) rather than file bytes: a
    200 file rig is ~12 GB and hashing it would cost more than the solve.
    """

    from .runner import compute_fingerprint

    return compute_fingerprint(
        {
            "cameras": [
                {
                    "label": camera.label,
                    "w": camera.width,
                    "h": camera.height,
                    "frames": camera.num_frames,
                    "fps": str(camera.fps),
                }
                for camera in cameras
            ],
            "num_timestamps": options.num_timestamps,
            "feature_type": options.feature_type,
            "resize_max": options.resize_max,
            "max_keypoints": options.max_keypoints,
            "focal_guess": round(float(options.focal_guess), 6),
            "refine_intrinsics": options.refine_intrinsics,
            "refine_principal_point": options.refine_principal_point,
            "filter_max_reproj_error": round(float(options.filter_max_reproj_error), 6),
            "max_rot_spread_deg": round(float(options.max_rot_spread_deg), 6),
            "sync_json": str(options.sync_json) if options.sync_json else "",
        }
    )


def read_solve(outputs_dir: str | Path) -> RigSolve:
    """Read a finished solve back off disk, validating it as we go."""

    root = Path(outputs_dir).expanduser().resolve()
    transforms_path = root / TRANSFORMS_NAME
    report_path = root / REPORT_NAME
    if not transforms_path.is_file():
        raise SfmError(f"{root} holds no {TRANSFORMS_NAME}; the solve did not finish.")
    try:
        transforms = json.loads(transforms_path.read_text())
    except (OSError, ValueError) as exc:
        raise SfmError(f"Could not read {transforms_path}: {exc}") from None
    frames = transforms.get("frames") or []
    if len(frames) < 2:
        raise SfmError(f"{transforms_path} describes fewer than two cameras; the rig did not solve.")
    labels = tuple(str(frame.get("camera_label", "")) for frame in frames)

    report: dict = {}
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            report = {}
    reconstruction = report.get("reconstruction") or {}
    stats = report.get("per_camera_pose_stats") or {}
    cameras = tuple(
        CameraQuality(
            label=str(label),
            num_views=int(entry.get("num_views", 0) or 0),
            num_inliers=int(entry.get("num_inliers", 0) or 0),
            center_spread=float(entry.get("center_spread", 0.0) or 0.0),
            rot_spread_deg=float(entry.get("rot_spread_deg", 0.0) or 0.0),
        )
        for label, entry in sorted(stats.items())
    )
    points_path = root / POINTS_NAME
    return RigSolve(
        root=root,
        transforms_path=transforms_path,
        report_path=report_path if report_path.is_file() else root / REPORT_NAME,
        points_path=points_path if points_path.is_file() else None,
        labels=labels,
        num_points=int(reconstruction.get("num_points3D", 0) or 0),
        mean_reprojection_error=float(reconstruction.get("mean_reprojection_error", 0.0) or 0.0),
        mean_track_length=float(reconstruction.get("mean_track_length", 0.0) or 0.0),
        num_registered_images=int(reconstruction.get("num_reg_images", 0) or 0),
        cameras=cameras,
    )


_TQDM = re.compile(r"^(?P<desc>[^:|]*?)\s*:\s*(?P<pct>\d{1,3})%\|.*?\|\s*(?P<n>\d+)\s*/\s*(?P<total>\d+)")
_EXTRACTED = re.compile(r"^\s*extracted\s+(?P<label>\S+)")
_FOUND = re.compile(r"^Found\s+(\d+)\s+videos")
_MATCHING = re.compile(r"^Matching\s+([\d,]+)\s+pairs")

#: (start, end) of each stage in the overall 0..1 range. Weights come from the
#: shape of the job: matching dominates because it is quadratic in cameras.
STAGE_BANDS = {
    "probe": (0.00, 0.02),
    "extract": (0.02, 0.12),
    "features": (0.12, 0.32),
    "match": (0.32, 0.78),
    "map": (0.78, 0.94),
    "refine": (0.94, 0.99),
}


@dataclass
class SfmProgress:
    """Fold the child's output into one monotonic fraction plus a status line."""

    expected_cameras: int = 0
    fraction: float = 0.0
    stage: str = "probe"
    message: str = "starting"
    _extracted: int = field(default=0, repr=False)

    def _advance(self, stage: str, ratio: float, message: str) -> bool:
        low, high = STAGE_BANDS[stage]
        ratio = min(max(ratio, 0.0), 1.0)
        value = low + (high - low) * ratio
        changed = False
        if value > self.fraction + 1e-9:
            self.fraction = value
            changed = True
        if message != self.message or stage != self.stage:
            changed = True
        self.stage = stage
        self.message = message
        return changed

    def update(self, line: str) -> bool:
        text = line.strip()
        if not text:
            return False

        found = _FOUND.match(text)
        if found:
            self.expected_cameras = int(found.group(1))
            return self._advance("probe", 1.0, f"found {self.expected_cameras} cameras")

        extracted = _EXTRACTED.match(text)
        if extracted:
            self._extracted += 1
            ratio = self._extracted / max(1, self.expected_cameras)
            return self._advance("extract", ratio, f"extracting frames {self._extracted}/{self.expected_cameras or '?'}")

        matching = _MATCHING.match(text)
        if matching:
            return self._advance("match", 0.0, f"matching {matching.group(1)} pairs")

        if text.startswith("Extracting features"):
            return self._advance("features", 0.0, "extracting features")
        if text.startswith("Running incremental mapping"):
            return self._advance("map", 0.1, "incremental mapping (intrinsics locked)")
        if text.startswith("Global bundle adjustment"):
            return self._advance("refine", 0.2, "global bundle adjustment")
        if text.startswith("After refinement"):
            return self._advance("refine", 0.8, text)
        if text.startswith("Exported") or text.startswith("Wrote"):
            return self._advance("refine", 1.0, "writing results")

        match = _TQDM.search(text)
        if match:
            desc = match.group("desc").rsplit("|", 1)[-1].strip().lower()
            done = int(match.group("n"))
            total = max(1, int(match.group("total")))
            ratio = done / total
            if "match" in desc:
                return self._advance("match", ratio, f"matching {done}/{total}")
            if "featur" in desc or "extract" in desc:
                return self._advance("features", ratio, f"features {done}/{total}")
        return False


def solve_rig(
    settings,
    options: SolveOptions,
    cameras: Sequence[CameraSource],
    *,
    on_progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> RigSolve:
    """Solve ``cameras`` into poses under ``options.outputs_dir``.

    Reuses an existing solve whose stamped fingerprint matches, replaces a
    stale one, and refuses to touch a directory this bridge did not write --
    the same disk-cache semantics the rest of the pack uses.
    """

    from .runner import prepare_artifact_dir, write_stamp

    options = options.validated()
    check_uniform(cameras)
    check_scale(len(cameras), options.num_timestamps)
    script = find_sfm_script(settings)

    outputs_dir = Path(options.outputs_dir).expanduser().resolve()
    fingerprint = solve_fingerprint(options, cameras)
    if prepare_artifact_dir(outputs_dir, fingerprint) == "reuse":
        LOGGER.info("Reusing the rig solve in %s", outputs_dir)
        return read_solve(outputs_dir)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    init_transforms = write_init_transforms(
        cameras, options.focal_guess, outputs_dir / INIT_TRANSFORMS_NAME
    )
    staged = link_cameras(cameras, outputs_dir / "cameras_selected")
    argv = build_solve_argv(settings, script, options, init_transforms, staged)

    state = SfmProgress(expected_cameras=len(cameras))

    def on_line(line: str) -> None:
        LOGGER.info("sfm: %s", line)
        if on_progress is not None and state.update(line):
            on_progress(state.fraction, state.message)

    LOGGER.info(
        "Solving %d cameras at %d timestamps (%s pairs) with %s",
        len(cameras),
        options.num_timestamps,
        f"{pair_count(len(cameras), options.num_timestamps):,}",
        options.feature_type,
    )
    run_streaming(
        argv,
        cwd=script.parent,
        env=_solve_env(settings),
        on_line=on_line,
        should_cancel=should_cancel,
    )

    solve = read_solve(outputs_dir)
    # One stamp, under the name prepare_artifact_dir reads, carrying the
    # human-readable report the rest of the pack's stamps also carry.
    write_stamp(
        outputs_dir,
        fingerprint,
        cameras=len(cameras),
        num_timestamps=options.num_timestamps,
        feature_type=options.feature_type,
        report=solve.summary(),
    )
    return solve


#: Intrinsics keys the flipbook contract carries. Anything else the solve
#: emitted (notably a distortion term) is a reason to stop, not to drop.
_INTRINSICS = ("fl_x", "fl_y", "cx", "cy", "w", "h")

#: Distortion coefficients that must not survive into a dataset. The builder
#: rejects nonzero distortion, and a silently dropped coefficient would mean
#: training against images that do not match their own camera model.
_DISTORTION = ("k1", "k2", "k3", "k4", "p1", "p2")


def build_flipbook_transforms(solve: RigSolve) -> dict:
    """Convert a solve into the per-frame ``transforms.json`` the builder reads.

    The two formats are nearly the same object -- both are nerfstudio-style
    with an OpenGL camera-to-world matrix -- so this is mostly a projection
    onto the keys the builder actually reads, plus one hard check: a camera
    that came back with distortion cannot go downstream, because the dataset
    builder assumes pinhole and would train against images that disagree with
    their own intrinsics.
    """

    try:
        payload = json.loads(Path(solve.transforms_path).read_text())
    except (OSError, ValueError) as exc:
        raise SfmError(f"Could not read {solve.transforms_path}: {exc}") from None

    frames = []
    for frame in payload.get("frames", []):
        label = str(frame.get("camera_label", ""))
        distorted = [key for key in _DISTORTION if float(frame.get(key, 0.0) or 0.0) != 0.0]
        if distorted:
            raise SfmError(
                f"Camera {label} solved with nonzero distortion ({', '.join(distorted)}), which the "
                "4DGS dataset builder cannot represent. Undistort the frames first, or re-solve "
                "seeded as PINHOLE so no distortion term is fitted."
            )
        missing = [key for key in _INTRINSICS if key not in frame]
        if missing:
            raise SfmError(f"Camera {label} is missing intrinsics {missing} in {solve.transforms_path}.")
        frames.append(
            {
                "camera_label": label,
                "transform_matrix": frame["transform_matrix"],
                "fl_x": float(frame["fl_x"]),
                "fl_y": float(frame["fl_y"]),
                "cx": float(frame["cx"]),
                "cy": float(frame["cy"]),
                "w": int(frame["w"]),
                "h": int(frame["h"]),
            }
        )
    if len(frames) < 2:
        raise SfmError(f"{solve.transforms_path} yielded fewer than two usable cameras.")
    return {"camera_model": "OPENCV", "frames": frames}


def stage_capture(
    solve: RigSolve,
    cameras: Sequence[CameraSource],
    root: str | Path,
    *,
    start_frame: int = 0,
    num_frames: int | None = None,
    frame_stride: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
):
    """Transpose the capture into the frame-major tree the builder reads.

    Only cameras that actually registered are staged: an unsolved camera has
    no pose, and a frame directory holding an image with no matching entry in
    ``transforms.json`` is exactly the silent inconsistency the builder cannot
    detect. The window is clamped to the shortest clip, because hardware sync
    still leaves ragged trims and every camera must appear in every frame.
    """

    from PIL import Image

    from .flipbook import IMAGES_SUBDIR, TRANSFORMS_NAME, Flipbook, frame_dir_name

    if frame_stride < 1:
        raise SfmError(f"frame_stride must be at least 1, got {frame_stride}.")
    if start_frame < 0:
        raise SfmError(f"start_frame must be non-negative, got {start_frame}.")

    transforms = build_flipbook_transforms(solve)
    labels = [frame["camera_label"] for frame in transforms["frames"]]
    by_label = {camera.label: camera for camera in cameras}
    missing = [label for label in labels if label not in by_label]
    if missing:
        raise SfmError(
            f"The solve names camera {missing[0]!r}, which is not in the selected capture. "
            "Stage the same camera selection that was solved."
        )
    staged_cameras = [by_label[label] for label in labels]

    available = min(camera.num_frames for camera in staged_cameras) - start_frame
    if available <= 0:
        raise SfmError(
            f"start_frame {start_frame} is past the end of the shortest clip "
            f"({min(c.num_frames for c in staged_cameras)} frames)."
        )
    wanted = available if num_frames is None else min(int(num_frames) * frame_stride, available)
    count = (wanted + frame_stride - 1) // frame_stride
    if count < 1:
        raise SfmError("The requested window contains no frames.")

    root = Path(root).expanduser().resolve()
    encoded = json.dumps(transforms, indent=1)
    total = len(staged_cameras)
    for position, camera in enumerate(staged_cameras):
        if should_cancel is not None and should_cancel():
            raise SfmError("Cancelled while staging the capture.")
        written = 0
        for index, frame in enumerate(iter_frames_of(camera.path)):
            if index < start_frame:
                continue
            if (index - start_frame) % frame_stride:
                continue
            if written >= count:
                break
            destination = root / frame_dir_name(written) / IMAGES_SUBDIR / f"{camera.label}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            # compress_level 1 matches the rest of the pack: staging bytes are
            # read once, so encode time dominates size.
            Image.fromarray(frame).save(destination, optimize=False, compress_level=1)
            written += 1
        if written < count:
            raise SfmError(
                f"Camera {camera.label} yielded {written} frames of the {count} requested; "
                "the clip is shorter than the window."
            )
        LOGGER.info("staged %s: %d frames", camera.label, written)
        if on_progress is not None:
            on_progress(position + 1, total)

    for index in range(count):
        (root / frame_dir_name(index) / TRANSFORMS_NAME).write_text(encoded)

    reference = staged_cameras[0]
    try:
        camera_ids = tuple(int(label) for label in labels)
    except ValueError:
        camera_ids = tuple(range(len(labels)))
    return Flipbook(
        root=root,
        labels=tuple(labels),
        camera_ids=camera_ids,
        num_frames=count,
        fps=reference.fps / frame_stride,
        width=reference.width,
        height=reference.height,
    )


def iter_frames_of(path: Path):
    """Indirection so the staging loop can be exercised without a real video."""

    from .videoio import iter_frames

    return iter_frames(path)


def _solve_env(settings) -> dict[str, str]:
    """Child environment. ``build_env`` already strips the interpreter leakage
    (``PYTHONPATH``/``PYTHONHOME``) that would confuse a sibling process."""

    from .runner import build_env

    # The script imports its sibling ``image_formats`` module, which works
    # because the child's cwd is the scripts directory; nothing else is needed.
    return build_env(settings)
