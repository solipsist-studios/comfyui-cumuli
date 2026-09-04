# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Transpose a 4DAnyone ring into the frame-major staging tree the 4DGS
dataset builder reads.

4DAnyone publishes one video per *camera*. The builder reads one directory per
*frame*, so this module decodes all N view clips once and writes::

    <root>/frame_0000/transforms.json
    <root>/frame_0000/images_flat/00.png .. 23.png
    <root>/frame_0001/...

`transforms.json` is written once and copied into every frame directory: the
generated ring is static by construction, and the builder verifies that by
comparing the first and last frame.

Camera labels are fixed-width zero-padded decimal (24 views -> ``"00".."23"``),
the convention every downstream consumer looks cameras up by. The width comes
from the camera count so that ``sorted(labels)`` orders cameras correctly past
100 views.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .ring import RingResult
from .videoio import iter_frames

LOGGER = logging.getLogger("comfyui-cumuli")

IMAGES_SUBDIR = "images_flat"
MASKS_SUBDIR = "fmasks_clean"
TRANSFORMS_NAME = "transforms.json"


class FlipbookError(RuntimeError):
    """Raised when the staging tree cannot be written or is inconsistent."""


def label_width(num_views: int) -> int:
    """Fixed label width: 2 digits up to 100 cameras, 3 beyond, and so on."""

    return max(2, len(str(max(0, num_views - 1))))


def camera_label(camera_id: int, width: int) -> str:
    return f"{camera_id:0{width}d}"


def frame_dir_name(frame_index: int) -> str:
    return f"frame_{frame_index:04d}"


@dataclass(frozen=True)
class Flipbook:
    """A written staging tree, described well enough to build a dataset from."""

    root: Path
    labels: tuple[str, ...]
    camera_ids: tuple[int, ...]
    num_frames: int
    fps: Fraction
    width: int
    height: int

    @property
    def frame_dirs(self) -> list[Path]:
        return sorted(d for d in self.root.iterdir() if d.is_dir() and d.name.startswith("frame_"))

    def label_of(self, camera_id: int) -> str:
        return self.labels[self.camera_ids.index(camera_id)]

    def duration(self) -> float:
        return float((self.num_frames - 1) / self.fps) if self.num_frames > 1 else 0.0


def build_transforms(ring: RingResult, camera_ids: Sequence[int], width: int) -> dict:
    """The per-frame ``transforms.json`` payload.

    Only ``frames`` is read downstream, and each entry needs ``camera_label``,
    an OpenGL/nerfstudio camera-to-world ``transform_matrix``, per-camera
    pinhole intrinsics and the image size. Distortion keys are deliberately
    absent: the builder rejects any nonzero distortion, and the generated views
    are pinhole by construction.
    """

    frames = []
    for camera_id in camera_ids:
        camera = ring.camera(camera_id)
        focal_x, focal_y = camera.focal
        cx, cy = camera.principal_point
        frames.append(
            {
                "camera_label": camera_label(camera_id, width),
                "transform_matrix": camera.nerf_transform(),
                "fl_x": focal_x,
                "fl_y": focal_y,
                "cx": cx,
                "cy": cy,
                "w": camera.width or ring.width,
                "h": camera.height or ring.height,
            }
        )
    return {"camera_model": "OPENCV", "frames": frames}


def write_flipbook(
    ring: RingResult,
    root: str | Path,
    *,
    camera_ids: Sequence[int] | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    overwrite: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Flipbook:
    """Decode the ring into ``root`` and return the resulting :class:`Flipbook`."""

    from PIL import Image

    if frame_stride < 1:
        raise FlipbookError(f"frame_stride must be at least 1, got {frame_stride}.")
    root = Path(root).expanduser().resolve()
    available = [camera.camera_id for camera in ring.cameras]
    ids = list(camera_ids) if camera_ids else available
    unknown = [view for view in ids if view not in available]
    if unknown:
        raise FlipbookError(f"{ring.root} has no view {unknown[0]} (it has {available[0]}..{available[-1]}).")
    if len(ids) < 2:
        raise FlipbookError("A 4DGS dataset needs at least two views; select more of the ring.")

    width = label_width(len(ids))
    labels = tuple(camera_label(view, width) for view in ids)
    transforms = json.dumps(build_transforms(ring, ids, width), indent=1)

    written_per_view: list[int] = []
    total = len(ids)
    for position, camera_id in enumerate(ids):
        if should_cancel is not None and should_cancel():
            raise FlipbookError("Cancelled while decoding the ring.")
        label = labels[position]
        count = 0
        for index, frame in enumerate(iter_frames(ring.video_path(camera_id))):
            if index % frame_stride:
                continue
            if max_frames is not None and count >= max_frames:
                break
            destination = root / frame_dir_name(count) / IMAGES_SUBDIR / f"{label}.png"
            count += 1
            if destination.is_file() and not overwrite:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            # compress_level 1 matches the dataset builder's own choice: these
            # are staging bytes, read once, so encode time dominates size.
            Image.fromarray(frame).save(destination, optimize=False, compress_level=1)
        if count == 0:
            raise FlipbookError(f"View {camera_id:02d} of {ring.root} decoded to zero frames.")
        written_per_view.append(count)
        LOGGER.info("flipbook %s: %d frames", label, count)
        if on_progress is not None:
            on_progress(position + 1, total)

    num_frames = min(written_per_view)
    if num_frames != max(written_per_view):
        raise FlipbookError(
            f"Views decoded to different lengths ({min(written_per_view)}..{max(written_per_view)} frames). "
            "The dataset builder needs every camera present in every frame."
        )
    for index in range(num_frames):
        (root / frame_dir_name(index) / TRANSFORMS_NAME).write_text(transforms)

    fps = ring.fps / frame_stride
    return Flipbook(
        root=root,
        labels=labels,
        camera_ids=tuple(ids),
        num_frames=num_frames,
        fps=fps,
        width=ring.width,
        height=ring.height,
    )


def check_complete(flipbook: Flipbook, subdir: str) -> None:
    """Fail loudly if ``subdir`` is not present for every label in every frame.

    The builder assumes completeness and would otherwise die on a bare
    ``FileNotFoundError`` deep inside a worker thread.
    """

    missing: list[str] = []
    for index in range(flipbook.num_frames):
        directory = flipbook.root / frame_dir_name(index) / subdir
        for label in flipbook.labels:
            if not (directory / f"{label}.png").is_file():
                missing.append(f"{frame_dir_name(index)}/{subdir}/{label}.png")
                if len(missing) > 5:
                    break
        if len(missing) > 5:
            break
    if missing:
        raise FlipbookError(
            f"{len(missing)}+ files missing from the staging tree, first: {missing[0]}. "
            f"Every camera must have a {subdir} entry in every frame."
        )


def load_flipbook(root: str | Path, fps: Fraction) -> Flipbook:
    """Reconstruct a :class:`Flipbook` from a staging tree already on disk.

    The tree is self-describing except for the frame rate, which only the
    source clip knew: pass ``fps`` explicitly (the Stage Ring stamp records it
    for trees this bridge wrote). Accepts external trees too -- real captures
    laid out frame-major with a per-frame ``transforms.json`` -- and raises
    :class:`FlipbookError` on anything malformed rather than guessing.
    """

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FlipbookError(f"Flipbook directory not found: {root}")
    frame_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("frame_"))
    if not frame_dirs:
        raise FlipbookError(f"{root} holds no frame_NNNN directories; not a staged flipbook.")
    transforms_path = frame_dirs[0] / TRANSFORMS_NAME
    try:
        frames = json.loads(transforms_path.read_text())["frames"]
    except (OSError, ValueError, KeyError) as exc:
        raise FlipbookError(f"Could not read {transforms_path}: {exc}") from None
    if len(frames) < 2:
        raise FlipbookError(f"{transforms_path} describes fewer than two cameras.")
    labels = tuple(str(frame["camera_label"]) for frame in frames)
    if len(set(labels)) != len(labels):
        raise FlipbookError(f"{transforms_path} repeats a camera_label.")
    try:
        camera_ids = tuple(int(label) for label in labels)
    except ValueError:
        # External captures may use non-numeric labels; synthesize stable ids.
        camera_ids = tuple(range(len(labels)))
    missing = [
        f"{directory.name}/{IMAGES_SUBDIR}/{label}.png"
        for directory in (frame_dirs[0], frame_dirs[-1])
        for label in labels
        if not (directory / IMAGES_SUBDIR / f"{label}.png").is_file()
    ]
    if missing:
        raise FlipbookError(f"{root} is incomplete; missing e.g. {missing[0]}.")
    if fps <= 0:
        raise FlipbookError(
            "The frame rate cannot be recovered from the tree alone; set fps on the loader."
        )
    return Flipbook(
        root=root,
        labels=labels,
        camera_ids=camera_ids,
        num_frames=len(frame_dirs),
        fps=Fraction(fps),
        width=int(frames[0]["w"]),
        height=int(frames[0]["h"]),
    )
