# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Foreground mattes for a staged ring, using ComfyUI's own BiRefNet.

4DAnyone hallucinates a grey studio background. The dataset builder needs a
subject mask per camera per frame: it carves the visual hull from them, and it
bakes them into the training images' alpha. Without masks the background trains
as scene content -- one unmasked ring splat measured over 130 m of extent.

ComfyUI ships BiRefNet natively (``comfy.bg_removal_model``, the
``background_removal`` model folder, and the stock ``Load Background Removal
Model`` node), so this runs in ``comfyenv`` in-process with no extra dependency
and no second environment. Any model that satisfies the ``BACKGROUND_REMOVAL``
socket works, so a different matting model can be substituted from the graph.

Masks are never dilated. Dilated alpha supervision teaches a bright silhouette
fringe.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .flipbook import Flipbook, frame_dir_name

LOGGER = logging.getLogger("comfyui-cumuli")

#: Images pushed through the matting model at once. BiRefNet runs at a fixed
#: 1024x1024 internally, so this bounds peak VRAM independently of view size.
DEFAULT_BATCH = 8


class MaskError(RuntimeError):
    """Raised when mattes cannot be produced."""


def _to_batch(paths: list[Path]):
    """Load images as the float32 0..1 ``(B, H, W, 3)`` tensor ComfyUI uses."""

    import torch
    from PIL import Image

    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
    stacked = np.stack(frames)
    return torch.from_numpy(stacked).float().div_(255.0)


def matte_flipbook(
    flipbook: Flipbook,
    bg_removal_model,
    *,
    subdir: str = "fmasks_clean",
    batch_size: int = DEFAULT_BATCH,
    overwrite: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> int:
    """Write ``<frame>/<subdir>/<label>.png`` for every staged image.

    Returns the number of mattes written. Existing files are kept unless
    ``overwrite`` is set, so a cancelled run resumes cheaply.
    """

    from PIL import Image

    if bg_removal_model is None:
        raise MaskError(
            "No background removal model connected. Add ComfyUI's 'Load Background Removal Model' "
            "node (models/background_removal/, for example birefnet.safetensors) and wire it in."
        )
    if batch_size < 1:
        raise MaskError(f"batch_size must be at least 1, got {batch_size}.")

    jobs: list[tuple[Path, Path]] = []
    for index in range(flipbook.num_frames):
        frame_root = flipbook.root / frame_dir_name(index)
        destination = frame_root / subdir
        for label in flipbook.labels:
            source = frame_root / "images_flat" / f"{label}.png"
            target = destination / f"{label}.png"
            if not source.is_file():
                raise MaskError(f"Staged image is missing: {source}")
            if target.is_file() and not overwrite:
                continue
            jobs.append((source, target))

    total = len(jobs)
    if total == 0:
        LOGGER.info("All %d mattes already present under %s", flipbook.num_frames * len(flipbook.labels), subdir)
        return 0

    import torch

    written = 0
    for start in range(0, total, batch_size):
        if should_cancel is not None and should_cancel():
            raise MaskError("Cancelled while matting the ring.")
        batch = jobs[start : start + batch_size]
        images = _to_batch([source for source, _ in batch])
        # ComfyUI wraps node execution in inference_mode, but the CLI does not.
        # Without it BiRefNet builds a full autograd graph and exhausts a 32 GB
        # card on a handful of 1024px images. Nesting is harmless.
        with torch.inference_mode():
            masks = bg_removal_model.encode_image(images)
        raster = (masks.clamp(0.0, 1.0).mul(255.0).round().to("cpu").numpy()).astype(np.uint8)
        for (_, target), plane in zip(batch, raster):
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(plane, mode="L").save(target, optimize=False, compress_level=1)
            written += 1
        if on_progress is not None:
            on_progress(written, total)
    LOGGER.info("wrote %d mattes to %s/*/%s", written, flipbook.root, subdir)
    return written


def mask_coverage(flipbook: Flipbook, subdir: str = "fmasks_clean") -> float:
    """Mean foreground fraction of the middle frame, as a sanity signal.

    A value near 0 means the matting model found no subject, which the dataset
    builder would only report much later as a collapsed visual hull.
    """

    from PIL import Image

    frame_root = flipbook.root / frame_dir_name(flipbook.num_frames // 2) / subdir
    fractions = []
    for label in flipbook.labels:
        path = frame_root / f"{label}.png"
        if not path.is_file():
            continue
        with Image.open(path) as image:
            plane = np.asarray(image.convert("L"))
        fractions.append(float((plane > 127).mean()))
    return float(np.mean(fractions)) if fractions else 0.0
