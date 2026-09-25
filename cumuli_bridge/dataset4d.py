# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Assemble a D-NeRF-style 4D Gaussian Splatting dataset from a staged ring.

Vendored from the cumuli pipeline (same copyright holder, same licence):
``scripts/build_flipbook_4dgs_dataset.py`` plus the four helpers it imports
from ``scripts/build_4dgs_dataset.py`` (``project``, ``in_mask``,
``write_ply_with_time``, ``convert_image``). Kept as a library rather than a
shelled-out CLI so the whole node pack runs in one environment: it needs only
numpy, Pillow and the standard library, all present in ``comfyenv``.

Deliberate differences from the upstream script, and nothing else:

* it is a function over a :class:`~cumuli_bridge.flipbook.Flipbook` instead of
  an ``argparse`` main, so it reports progress and honours cancellation;
* ``print`` calls became ``logging`` calls (ComfyUI's ruff config bans print);
* ``sys.exit`` calls became :class:`DatasetError`, with the collapsed-hull case
  explaining what a 4DAnyone ring failing that check actually means.

Output layout, unchanged::

    <out>/realcams/cam<label>/frame_NNNNN.png   RGBA, mask in alpha, downscaled
    <out>/transforms_train.json                 per-view entries, per-camera
    <out>/transforms_test.json                  intrinsics, no global block
    <out>/points3d.ply                          visual-hull points with `time`
    <out>/eval_gt_flat/frame_NNNNN.png          only when a test camera is given

Never create a ``sparse/`` directory beside these: the trainer's scene loader
checks for one first and would switch to the COLMAP reader.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .flipbook import IMAGES_SUBDIR, Flipbook, MASKS_SUBDIR

LOGGER = logging.getLogger("comfyui-cumuli")

#: Binary layout written by ``write_ply_with_time`` -- kept as one constant so
#: the window slicer's reader and the writer can never silently drift apart.
_PLY_RECORD_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("time", "<f4")]
)


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be assembled."""


# --------------------------------------------------------------------------
# Geometry helpers (vendored verbatim in behaviour)
# --------------------------------------------------------------------------
def project(w2c, fl_x, fl_y, cx, cy, pts):
    """World points ``[N,3]`` -> ``(u, v, in-front mask)`` in full-res pixels."""

    cam = (w2c[:3, :3] @ pts.T + w2c[:3, 3:4]).T
    z = cam[:, 2]
    front = z > 0.05
    zs = np.where(front, z, 1.0)
    return fl_x * cam[:, 0] / zs + cx, fl_y * cam[:, 1] / zs + cy, front


def in_mask(mask, u, v, front):
    h, w = mask.shape
    ui = np.clip(u, 0, w - 1).astype(np.int32)
    vi = np.clip(v, 0, h - 1).astype(np.int32)
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return inside & (mask[vi, ui] > 127)


def write_ply_with_time(path, pts, rgb, times):
    """Binary little-endian PLY carrying the per-point ``time`` the 4D trainer
    buckets its init points by. A static cloud is not usable."""

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float time\nend_header\n"
    )
    rec = struct.Struct("<fffBBBf")
    with open(path, "wb") as fp:
        fp.write(header.encode("ascii"))
        for p, c, t in zip(pts, rgb, times):
            fp.write(rec.pack(p[0], p[1], p[2], c[0], c[1], c[2], t))


def convert_image(src_img, src_mask, dst, downscale):
    """RGB + mask -> RGBA with the mask in alpha, optionally downscaled."""

    from PIL import Image

    with Image.open(src_img) as im:
        rgb = im.convert("RGB")
    with Image.open(src_mask) as mm:
        alpha = mm.convert("L")
    if downscale != 1:
        size = (rgb.width // downscale, rgb.height // downscale)
        rgb = rgb.resize(size, Image.LANCZOS)
        alpha = alpha.resize(size, Image.LANCZOS)
    rgb.putalpha(alpha)
    # compress_level 1: much faster than PIL's default 6 for ~25% more bytes;
    # the trainer decodes each image once into its cache, so encode dominates.
    rgb.save(dst, optimize=False, compress_level=1)
    return dst


def flatten_gt(rgba_path, dst, background="black"):
    """RGBA -> RGB composited over ``background``, matching what the trainer
    renders: OMG4's own ``white_background`` flag composites onto pure black
    or pure white and nothing else (see ``TrainOptions.white_background``), so
    this must agree with whichever the model actually trained against or the
    held-out PSNR/LPIPS score reverts to comparing against the wrong backdrop.
    """

    from PIL import Image

    with Image.open(rgba_path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32)
    bg = 255.0 if background == "white" else 0.0
    alpha = rgba[..., 3:4] / 255.0
    rgb = rgba[..., :3] * alpha + bg * (1.0 - alpha)
    Image.fromarray(rgb.astype(np.uint8)).save(dst, optimize=False, compress_level=1)


# --------------------------------------------------------------------------
# Rig + hull
# --------------------------------------------------------------------------
def load_rig(frame_dirs: Sequence[Path]) -> dict:
    """Static rig from the first frame's transforms.json, verified against the last."""

    def read(frame_dir: Path) -> dict:
        payload = json.loads((frame_dir / "transforms.json").read_text())
        cams = {}
        for entry in payload["frames"]:
            for key in ("k1", "k2", "p1", "p2"):
                if abs(entry.get(key, 0.0)) > 1e-9:
                    raise DatasetError(
                        f"Camera {entry['camera_label']} declares distortion {key}={entry[key]}; "
                        "this dataset builder expects undistorted pinhole views."
                    )
            c2w = np.array(entry["transform_matrix"], dtype=np.float64)
            colmap = c2w.copy()
            colmap[:3, 1:3] *= -1  # OpenGL/Blender -> COLMAP
            cams[entry["camera_label"]] = {
                "c2w_gl": c2w,
                "w2c": np.linalg.inv(colmap),
                "intr": (entry["fl_x"], entry["fl_y"], entry["cx"], entry["cy"]),
                "w": entry["w"],
                "h": entry["h"],
            }
        return cams

    rig = read(frame_dirs[0])
    check = read(frame_dirs[-1])
    if sorted(rig) != sorted(check):
        raise DatasetError("The camera set differs between the first and last frame.")
    for label in rig:
        if not np.allclose(rig[label]["c2w_gl"], check[label]["c2w_gl"]) or not np.allclose(
            rig[label]["intr"], check[label]["intr"]
        ):
            raise DatasetError(f"Camera {label} moves between frames; the rig must be static.")
    return rig


def _load_frame_masks(frame_dir: Path, labels: Sequence[str], masks_dir: str) -> dict:
    from PIL import Image

    return {
        label: np.array(Image.open(frame_dir / masks_dir / f"{label}.png").convert("L")) for label in labels
    }


def hull_votes(rig: dict, masks: dict, pts) -> np.ndarray:
    votes = np.zeros(len(pts), dtype=np.int32)
    for label, cam in rig.items():
        u, v, front = project(cam["w2c"], *cam["intr"], pts)
        votes += in_mask(masks[label], u, v, front)
    return votes


def carve_frame(frame_dir, rig, masks_dir, bbox, n_target, min_views, color_cams, rng):
    """Visual-hull points and colours for one frame: ``[K,3]`` pts, ``[K,3]`` rgb."""

    from PIL import Image

    labels = sorted(rig)
    masks = _load_frame_masks(frame_dir, labels, masks_dir)
    lo, hi = bbox
    kept = []
    for _ in range(8):  # rejection-sample in batches
        cand = rng.uniform(lo, hi, size=(n_target * 4, 3))
        good = hull_votes(rig, masks, cand) >= min_views
        kept.append(cand[good])
        if sum(len(k) for k in kept) >= n_target:
            break
    pts = np.concatenate(kept, axis=0)[:n_target]
    if len(pts) == 0:
        return pts.astype(np.float32), np.zeros((0, 3), dtype=np.uint8)

    rgb = np.zeros((len(pts), 3), dtype=np.float64)
    n_seen = np.zeros(len(pts), dtype=np.int32)
    for label in color_cams:
        img = np.asarray(Image.open(frame_dir / IMAGES_SUBDIR / f"{label}.png").convert("RGB"))
        u, v, front = project(rig[label]["w2c"], *rig[label]["intr"], pts)
        ok = in_mask(masks[label], u, v, front)
        ui = np.clip(u, 0, img.shape[1] - 1).astype(np.int32)
        vi = np.clip(v, 0, img.shape[0] - 1).astype(np.int32)
        rgb[ok] += img[vi[ok], ui[ok]]
        n_seen[ok] += 1
    rgb /= np.maximum(n_seen, 1)[:, None]
    rgb[n_seen == 0] = 127.0
    return pts.astype(np.float32), rgb.astype(np.uint8)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
@dataclass
class DatasetOptions:
    out_dir: Path
    fps: float = 24.0
    downscale: int = 1
    test_cameras: tuple[str, ...] = ()
    holdout_cameras: tuple[str, ...] = ()
    masks_dir: str = MASKS_SUBDIR
    hull_points: int = 300_000
    hull_min_views: int = 9
    jobs: int = 8
    #: OMG4's own ``white_background`` flag is a hard binary (see
    #: ``TrainOptions.white_background``); this must match whatever the
    #: model trained against, or held-out scoring composites onto the wrong
    #: backdrop. "black" matches every dataset built before this option
    #: existed.
    background: str = "black"

    def validate(self) -> None:
        if self.downscale < 1:
            raise DatasetError(f"downscale must be at least 1, got {self.downscale}.")
        if self.fps <= 0:
            raise DatasetError(f"fps must be positive, got {self.fps}.")
        if self.hull_points < 1:
            raise DatasetError(f"hull_points must be positive, got {self.hull_points}.")
        if self.hull_min_views < 1:
            raise DatasetError(f"hull_min_views must be at least 1, got {self.hull_min_views}.")
        if self.jobs < 1:
            raise DatasetError(f"jobs must be at least 1, got {self.jobs}.")
        if self.background not in ("black", "white"):
            raise DatasetError(f"background must be 'black' or 'white', got {self.background!r}.")


@dataclass
class DatasetSummary:
    out_dir: Path
    train_cameras: list[str] = field(default_factory=list)
    test_cameras: list[str] = field(default_factory=list)
    holdout_cameras: list[str] = field(default_factory=list)
    num_frames: int = 0
    duration_seconds: float = 0.0
    init_points: int = 0
    images_written: int = 0
    hull_bbox: list[list[float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_dir": str(self.out_dir),
            "train_cameras": self.train_cameras,
            "test_cameras": self.test_cameras,
            "holdout_cameras": self.holdout_cameras,
            "num_frames": self.num_frames,
            "duration_seconds": round(self.duration_seconds, 4),
            "init_points": self.init_points,
            "images_written": self.images_written,
            "hull_bbox": self.hull_bbox,
            "notes": self.notes,
        }


def build_dataset(
    flipbook: Flipbook,
    options: DatasetOptions,
    *,
    on_progress: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> DatasetSummary:
    """Carve the hull, bake RGBA frames, and write both transforms files."""

    options.validate()
    out = Path(options.out_dir).expanduser().resolve()
    frame_dirs = flipbook.frame_dirs
    if not frame_dirs:
        raise DatasetError(f"No frame_* directories under {flipbook.root}.")
    rig = load_rig(frame_dirs)
    labels = sorted(rig)
    frame_idx = list(range(len(frame_dirs)))

    unknown = [c for c in tuple(options.test_cameras) + tuple(options.holdout_cameras) if c not in rig]
    if unknown:
        raise DatasetError(f"Unknown test/holdout camera label(s): {unknown}. Known labels: {labels}.")

    summary = DatasetSummary(
        out_dir=out,
        test_cameras=list(options.test_cameras),
        holdout_cameras=list(options.holdout_cameras),
        num_frames=len(frame_dirs),
        duration_seconds=frame_idx[-1] / options.fps if frame_idx else 0.0,
    )
    if len(options.test_cameras) > 1:
        summary.notes.append(
            "eval_gt_flat basenames collide across multiple test cameras; only the last written survives. "
            "Use one test camera for scoring."
        )

    def report(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(fraction, message)

    def cancelled() -> bool:
        return should_cancel is not None and should_cancel()

    # -- visual-hull bbox discovery on the middle frame ---------------------
    report(0.02, "discovering the hull bounding box")
    rng = np.random.default_rng(0)
    mid_dir = frame_dirs[len(frame_dirs) // 2]
    masks = _load_frame_masks(mid_dir, labels, options.masks_dir)
    positions = [np.linalg.inv(rig[c]["w2c"])[:3, 3] for c in labels]
    centroid = np.mean(positions, axis=0)
    span = max(np.ptp(positions, axis=0)) or 4.0
    cand = rng.uniform(centroid - span, centroid + span, size=(500_000, 3))
    good = cand[hull_votes(rig, masks, cand) >= options.hull_min_views]
    if len(good) < 100:
        raise DatasetError(
            f"Visual-hull discovery found only {len(good)} points with hull_min_views="
            f"{options.hull_min_views} across {len(labels)} cameras. For a generated ring this usually means "
            "the views disagree about where the subject is, or the mattes are empty. Check the mask preview, "
            "then lower hull_min_views."
        )
    pad = 0.15 * (good.max(0) - good.min(0)) + 0.05
    bbox = (good.min(0) - pad, good.max(0) + pad)
    summary.hull_bbox = [bbox[0].tolist(), bbox[1].tolist()]
    LOGGER.info("hull bbox (%s): %s .. %s (%d seed points)", mid_dir.name, np.round(bbox[0], 2), np.round(bbox[1], 2), len(good))

    # -- per-frame hull carving --------------------------------------------
    per_frame = max(options.hull_points // len(frame_dirs), 200)
    color_cams = labels[:: max(len(labels) // 4, 1)][:4]
    LOGGER.info("carving %d points/frame, colours from %s", per_frame, color_cams)
    hull: list = [None] * len(frame_dirs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.jobs) as pool:
        futures = {
            pool.submit(
                carve_frame,
                d,
                rig,
                options.masks_dir,
                bbox,
                per_frame,
                options.hull_min_views,
                color_cams,
                np.random.default_rng(1000 + i),
            ): i
            for i, d in enumerate(frame_dirs)
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            hull[futures[future]] = future.result()
            report(0.02 + 0.38 * done / len(frame_dirs), f"carving hull {done}/{len(frame_dirs)}")
            if cancelled():
                for pending in futures:
                    pending.cancel()
                raise DatasetError("Cancelled while carving the visual hull.")

    times = np.concatenate(
        [np.full(len(p), frame_idx[i] / options.fps, dtype=np.float32) for i, (p, _) in enumerate(hull)]
    )
    pts = np.concatenate([p for p, _ in hull])
    rgb = np.concatenate([c for _, c in hull])
    if len(pts) == 0:
        raise DatasetError("The visual hull carved zero points; the trainer cannot initialise from an empty cloud.")
    out.mkdir(parents=True, exist_ok=True)
    write_ply_with_time(out / "points3d.ply", pts, rgb, times)
    summary.init_points = int(len(pts))
    LOGGER.info("points3d.ply: %d points over [%.3f, %.3f]s", len(pts), times.min(), times.max())

    # -- RGBA images --------------------------------------------------------
    jobs: list[tuple[Path, Path, Path]] = []
    for label in labels:
        (out / "realcams" / f"cam{label}").mkdir(parents=True, exist_ok=True)
        for i, d in enumerate(frame_dirs):
            jobs.append(
                (
                    d / IMAGES_SUBDIR / f"{label}.png",
                    d / options.masks_dir / f"{label}.png",
                    out / "realcams" / f"cam{label}" / f"frame_{frame_idx[i] + 1:05d}.png",
                )
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=options.jobs) as pool:
        futures = [pool.submit(convert_image, s, m, d, options.downscale) for s, m, d in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            report(0.40 + 0.55 * done / len(jobs), f"baking RGBA {done}/{len(jobs)}")
            if cancelled():
                for pending in futures:
                    pending.cancel()
                raise DatasetError("Cancelled while baking RGBA frames.")
    summary.images_written = len(jobs)

    # -- transforms (per-frame intrinsics; the cameras differ) --------------
    ds = options.downscale

    def entries(cam_labels: Sequence[str]) -> list[dict]:
        rows = []
        for label in cam_labels:
            cam = rig[label]
            fl_x, fl_y, cx, cy = cam["intr"]
            for i in frame_idx:
                rows.append(
                    {
                        "file_path": f"realcams/cam{label}/frame_{i + 1:05d}",
                        "camera_label": label,
                        "time": i / options.fps,
                        "fl_x": fl_x / ds,
                        "fl_y": fl_y / ds,
                        "cx": cx / ds,
                        "cy": cy / ds,
                        "w": cam["w"] // ds,
                        "h": cam["h"] // ds,
                        "transform_matrix": cam["c2w_gl"].tolist(),
                    }
                )
        return rows

    excluded = set(options.test_cameras) | set(options.holdout_cameras)
    train_labels = [c for c in labels if c not in excluded]
    if not train_labels:
        raise DatasetError("Every camera was held out; nothing would train.")
    summary.train_cameras = train_labels
    (out / "transforms_train.json").write_text(
        json.dumps({"camera_model": "OPENCV", "frames": entries(train_labels)}, indent=1)
    )
    (out / "transforms_test.json").write_text(
        json.dumps(
            {"camera_model": "OPENCV", "frames": entries(options.test_cameras or train_labels[:1])},
            indent=1,
        )
    )
    if not options.test_cameras:
        summary.notes.append(
            "No test camera was held out, so transforms_test.json duplicates a training camera. "
            "Its PSNR is a training-view monitor, not a held-out score."
        )

    # -- flat GT for scoring (held-out cameras only) ------------------------
    if options.test_cameras:
        (out / "eval_gt_flat").mkdir(exist_ok=True)
        gt_jobs = [
            (
                out / "realcams" / f"cam{label}" / f"frame_{i + 1:05d}.png",
                out / "eval_gt_flat" / f"frame_{i + 1:05d}.png",
            )
            for label in options.test_cameras
            for i in frame_idx
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=options.jobs) as pool:
            for future in [pool.submit(flatten_gt, s, d, options.background) for s, d in gt_jobs]:
                future.result()
        LOGGER.info("eval_gt_flat: %d %s-composited GT frames", len(frame_dirs), options.background)

    report(1.0, "dataset complete")
    LOGGER.info(
        "transforms: %d train cams, %d test cams -> %s",
        len(train_labels),
        len(options.test_cameras) or 1,
        out,
    )
    return summary


# --------------------------------------------------------------------------
# Windowing: slice an already-built dataset, not rebuild it
# --------------------------------------------------------------------------
def _filter_transforms(src: Path, dst: Path, frame_start: int, frame_count: int, fps: float) -> int:
    """Copy one transforms JSON, keeping only frames in the window and
    rewriting their ``time`` to be local (0-based) to the window.

    ``file_path`` entries are left untouched: they stay relative to
    ``realcams/...``, which the caller symlinks in unchanged, so no image
    bytes are copied or renumbered for a window.
    """

    payload = json.loads(src.read_text())
    kept = []
    for frame in payload.get("frames", []):
        index = round(float(frame["time"]) * fps)
        if frame_start <= index < frame_start + frame_count:
            frame = dict(frame)
            frame["time"] = round((index - frame_start) / fps, 7)
            kept.append(frame)
    dst.write_text(json.dumps({**payload, "frames": kept}, indent=1))
    return len(kept)


def _slice_points3d(src: Path, dst: Path, frame_start: int, frame_count: int, fps: float) -> int:
    """Filter ``points3d.ply`` by the same per-point ``time`` window, rebasing
    it to local time exactly as ``_filter_transforms`` does for the frames."""

    data = src.read_bytes()
    marker = b"end_header\n"
    header_end = data.index(marker) + len(marker)
    header = data[:header_end].decode("ascii")
    match = re.search(r"element vertex (\d+)", header)
    if not match:
        raise DatasetError(f"{src} has no 'element vertex' line; not a recognised points3d.ply.")
    count = int(match.group(1))
    records = np.frombuffer(data, dtype=_PLY_RECORD_DTYPE, count=count, offset=header_end)

    index = np.rint(records["time"] * fps).astype(np.int64)
    mask = (index >= frame_start) & (index < frame_start + frame_count)
    kept = records[mask]
    if len(kept) == 0:
        raise DatasetError(
            f"Window frames [{frame_start}, {frame_start + frame_count}) carve zero init points "
            f"from {src}. The window is too short relative to hull_points, or misaligned with the "
            "dataset's own frame timestamps."
        )
    local_time = ((index[mask] - frame_start).astype(np.float32) / np.float32(fps))
    pts = np.stack([kept["x"], kept["y"], kept["z"]], axis=1)
    rgb = np.stack([kept["r"], kept["g"], kept["b"]], axis=1)
    write_ply_with_time(dst, pts, rgb, local_time)
    return len(kept)


def slice_dataset_window(dataset_dir: Path, out_dir: Path, frame_start: int, frame_count: int, fps: float) -> None:
    """Point a training window at its slice of an already-built dataset.

    Nothing expensive is redone: the visual hull and RGBA frames are the
    costly part of ``build_dataset`` and a window reuses them verbatim via one
    directory symlink, filtering only the two small artifacts that carry a
    ``time`` field (``transforms_{train,test}.json``, ``points3d.ply``) and
    rebasing that field to be local to the window -- exactly the local time
    origin ``train.TrainOptions.duration_seconds`` expects for a standalone
    training run, and what ``bake.BakeOptions``/``merge_windows`` need to shift
    back into global time afterwards via each window's own offset.
    """

    dataset_dir = Path(dataset_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    realcams_link = out_dir / "realcams"
    if realcams_link.is_symlink():
        realcams_link.unlink()
    elif realcams_link.exists():
        raise DatasetError(f"{realcams_link} exists and is not a symlink; refusing to overwrite it.")
    realcams_link.symlink_to(dataset_dir / "realcams")

    for name in ("transforms_train.json", "transforms_test.json"):
        src = dataset_dir / name
        if not src.is_file():
            continue
        kept = _filter_transforms(src, out_dir / name, frame_start, frame_count, fps)
        if kept == 0:
            raise DatasetError(
                f"Window frames [{frame_start}, {frame_start + frame_count}) keep zero entries from "
                f"{src}."
            )

    _slice_points3d(dataset_dir / "points3d.ply", out_dir / "points3d.ply", frame_start, frame_count, fps)
    LOGGER.info(
        "sliced window [%d, %d) of %s -> %s", frame_start, frame_start + frame_count, dataset_dir, out_dir
    )


@dataclass(frozen=True)
class DatasetHandle:
    """A validated 4DGS dataset, as passed between nodes.

    ``fingerprint`` is the bridge's input fingerprint when the dataset was
    built (or previously stamped) by this pack; ``None`` means an external
    dataset, identified downstream by its files' mtime/size instead.
    """

    root: Path
    fingerprint: str | None = None
    source: str = "external"
