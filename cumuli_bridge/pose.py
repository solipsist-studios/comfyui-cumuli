# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Body pose from SAM 3D Body, estimated inside ComfyUI.

The model loader and the estimate live in 4DAnyone (``fdanyone.pose``) so that standalone
inference and these nodes share one code path. This module finds that file in the configured
4DAnyone checkout, imports it in this interpreter (it needs only numpy, av and ComfyUI), and
adds what is specific to ComfyUI: resolving the weights from its ``detection`` model folders.
"""

from __future__ import annotations

import sys
from pathlib import Path

DETECTION_FOLDER = "detection"


def _fdanyone_pose(fdanyone_root: Path):
    """``fdanyone.pose`` from the 4DAnyone checkout. The rest of the package is not imported."""

    root = str(Path(fdanyone_root).expanduser())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from fdanyone import pose as loader
    except ImportError as exc:
        raise PoseError(
            f"4DAnyone at {root} has no fdanyone/pose.py; update the checkout to a version with SAM 3D Body pose."
        ) from exc
    return loader


class PoseError(RuntimeError):
    """Raised when the body pose cannot be estimated, with what to do about it."""


def resolve_weights(setting: str, *, downloaded_dir: Path | None = None) -> Path:
    """The SAM 3D Body weights file named by the ``sam3d_weights`` setting.

    An absolute path is used as it is. A bare name is looked up in ComfyUI's
    ``detection`` model folders, then in ``downloaded_dir``, where 4DAnyone's
    ``download_model.py`` puts them (``<model_dir>/sam3d/detection``).
    """

    text = (setting or "").strip()
    if not text:
        raise PoseError("The 'sam3d_weights' setting is empty; name the SAM 3D Body weights file.")
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        if not candidate.is_file():
            raise PoseError(f"SAM 3D Body weights not found at {candidate}.")
        return candidate
    try:
        import folder_paths

        found = folder_paths.get_full_path(DETECTION_FOLDER, text)
    except ImportError:  # running from the CLI, outside ComfyUI
        found = None
    if found is None and downloaded_dir is not None and (Path(downloaded_dir) / text).is_file():
        found = str(Path(downloaded_dir) / text)
    if found is None:
        raise PoseError(
            f"SAM 3D Body weights '{text}' were not found in ComfyUI's '{DETECTION_FOLDER}' model folder. "
            "Run 4DAnyone's scripts/download_model.py (or the installer) to fetch them, or set "
            "'sam3d_weights' to a full path."
        )
    return Path(found)


def host_problem(fdanyone_root: Path) -> str | None:
    """Why this ComfyUI cannot estimate the pose, or ``None`` when it can."""

    return _fdanyone_pose(fdanyone_root).host_problem()


def estimate_pose(video: Path, out_npz: Path, *, fdanyone_root: Path, weights: Path, batch_size: int = 8) -> dict:
    """Write the generator's pose npz for the canonical clip, in this process."""

    loader = _fdanyone_pose(fdanyone_root)
    try:
        return loader.estimate_pose(video, out_npz, weights=weights, batch_size=batch_size)
    except loader.PoseError as exc:
        raise PoseError(str(exc)) from exc
