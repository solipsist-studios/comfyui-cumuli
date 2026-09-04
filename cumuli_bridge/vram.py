# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Hand the whole GPU to a long child stage (ring generation, training).

The known-good 24-view ring peaks near 31.9 GB on a 32 GB card, so any model
ComfyUI is still holding will push the child into an out-of-memory error
roughly 40 minutes into the run. Freeing first, and refusing to start when the
card is still busy, turns that into an immediate, readable failure.
"""

from __future__ import annotations

import gc
import logging

LOGGER = logging.getLogger("comfyui-cumuli")

_BYTES_PER_GB = 1024 ** 3


class InsufficientVRAM(RuntimeError):
    """Raised when the GPU cannot host the run even after unloading."""


def _device_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.rsplit(":", 1)[1])
        except ValueError:
            return 0
    return 0


def release_comfy_vram() -> None:
    """Unload every ComfyUI model and return the allocator's cache to the driver."""

    try:
        import comfy.model_management as mm
    except ImportError:  # running from the CLI, outside ComfyUI
        LOGGER.debug("comfy.model_management unavailable; skipping VRAM release.")
        return
    LOGGER.info("Cumuli: unloading ComfyUI models before launching the child stage")
    mm.unload_all_models()
    try:
        mm.cleanup_models()
    except Exception:  # pragma: no cover - depends on ComfyUI version
        LOGGER.debug("cleanup_models() unavailable", exc_info=True)
    gc.collect()
    mm.soft_empty_cache(force=True)


def free_bytes(device: str = "cuda:0") -> tuple[int, int]:
    """Return ``(free, total)`` device memory in bytes, or ``(0, 0)`` if unknown."""

    try:
        import torch
    except ImportError:
        return (0, 0)
    if not torch.cuda.is_available():
        return (0, 0)
    try:
        return torch.cuda.mem_get_info(_device_index(device))
    except Exception:  # pragma: no cover - driver dependent
        LOGGER.debug("torch.cuda.mem_get_info failed", exc_info=True)
        return (0, 0)


def require_free_vram(device: str, minimum_gb: float) -> tuple[float, float]:
    """Free ComfyUI's VRAM, then check the card has ``minimum_gb`` available.

    Returns ``(free_gb, total_gb)``. A minimum of ``0`` disables the check,
    which is what a CPU-only test run or a multi-GPU box may want.
    """

    release_comfy_vram()
    free, total = free_bytes(device)
    free_gb = free / _BYTES_PER_GB
    total_gb = total / _BYTES_PER_GB
    if minimum_gb <= 0 or total == 0:
        return (free_gb, total_gb)
    if total_gb + 0.5 < minimum_gb:
        raise InsufficientVRAM(
            f"{device} has {total_gb:.1f} GB of memory in total but this configuration needs about "
            f"{minimum_gb:.1f} GB. Lower views_per_group, disable RCP, or lower min_free_vram_gb "
            "in the bridge config if you know the run fits."
        )
    if free_gb < minimum_gb:
        raise InsufficientVRAM(
            f"Only {free_gb:.1f} GB of {total_gb:.1f} GB is free on {device}; this stage needs about "
            f"{minimum_gb:.1f} GB. Another process is holding the GPU -- close it, or lower "
            "min_free_vram_gb in the bridge config."
        )
    LOGGER.info("Cumuli: %.1f GB of %.1f GB free on %s", free_gb, total_gb, device)
    return (free_gb, total_gb)
