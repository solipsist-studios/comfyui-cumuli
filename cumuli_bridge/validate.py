# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Check that an assembled dataset really is what the 4D trainer expects.

Vendored from the cumuli pipeline's ``scripts/validate_stage_output.py`` (the
``dataset4d`` case and its two PLY header helpers), same copyright holder and
licence. Carried here rather than shelled out so the pack stays in one
environment, and because a builder can exit cleanly while still having written
garbage -- an RGB image where alpha was meant to be baked in, or a static init
cloud the 4D trainer cannot bucket by time.

Both failures are silent at this stage and expensive an hour into training,
which is why this runs as the pack's own acceptance check.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Enough of the file to hold any plausible PLY header.
PLY_HEADER_READ_BYTES = 4096

REQUIRED_FRAME_KEYS = ("file_path", "time", "fl_x", "fl_y", "cx", "cy")


class ValidationError(RuntimeError):
    """Raised when the dataset does not satisfy the trainer's contract."""


def ply_vertex_count(path: Path) -> int:
    """Vertex count from a PLY header, or -1 when it is missing or unparseable."""

    with path.open("rb") as handle:
        header = handle.read(PLY_HEADER_READ_BYTES).decode("ascii", errors="replace")
    if not header.startswith("ply"):
        return -1
    for line in header.splitlines():
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) == 3 and parts[2].isdigit():
            return int(parts[2])
    return -1


def ply_header_has_property(path: Path, name: str) -> bool:
    """True when the PLY header declares a property with this name."""

    with path.open("rb") as handle:
        header = handle.read(PLY_HEADER_READ_BYTES).decode("ascii", errors="replace")
    if not header.startswith("ply"):
        return False
    for line in header.splitlines():
        parts = line.split()
        if parts[:1] == ["property"] and parts[-1:] == [name]:
            return True
    return False


def validate_dataset(dataset_dir: str | Path) -> dict:
    """Raise :class:`ValidationError` unless the dataset is trainable.

    Returns a small dict of what was checked, so a node can show it.
    """

    from PIL import Image

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValidationError(f"Dataset directory not found: {root}")

    frames = None
    for name in ("transforms_train.json", "transforms_test.json"):
        path = root / name
        if not path.is_file():
            raise ValidationError(f"{path} was not produced")
        try:
            data = json.loads(path.read_text())
        except ValueError as exc:
            raise ValidationError(f"{path} is not valid JSON: {exc}") from None
        if name == "transforms_train.json":
            frames = data.get("frames", [])

    if not frames:
        raise ValidationError(f"{root / 'transforms_train.json'} has zero frames")

    # Per-view intrinsics are the contract: every frame entry carries its own
    # fl/c/w/h/time, and there is deliberately no global intrinsics block.
    for key in REQUIRED_FRAME_KEYS:
        missing = [i for i, frame in enumerate(frames) if key not in frame]
        if missing:
            raise ValidationError(
                f"transforms_train.json: {len(missing)} frames lack {key!r} (first at index {missing[0]})"
            )

    # file_path is extensionless. The image must exist and must be RGBA: the
    # mask is baked into alpha, and an RGB image means the bake silently failed
    # while every file still exists.
    probe = root / (frames[0]["file_path"] + ".png")
    if not probe.is_file():
        raise ValidationError(f"{probe} referenced by transforms_train.json is missing")
    with Image.open(probe) as image:
        if image.mode != "RGBA":
            raise ValidationError(f"{probe} is mode {image.mode}, expected RGBA (mask in alpha)")

    ply = root / "points3d.ply"
    if not ply.is_file():
        raise ValidationError(f"{ply} was not produced")
    count = ply_vertex_count(ply)
    if count < 1:
        raise ValidationError(f"{ply} declares zero points (a collapsed visual hull?)")
    if not ply_header_has_property(ply, "time"):
        raise ValidationError(
            f"{ply} header lacks a per-point 'time' property -- the 4D trainer buckets its init "
            "points by time and cannot use a static cloud"
        )

    return {
        "dataset_dir": str(root),
        "train_entries": len(frames),
        "probe_image": str(probe),
        "probe_mode": "RGBA",
        "init_points": count,
        "init_cloud_has_time": True,
    }
