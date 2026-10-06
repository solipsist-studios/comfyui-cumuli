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


class NoCudaDevice(RuntimeError):
    """Raised when the GPU a stage was asked to use is not there."""


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


def check_device(device: str) -> None:
    """Stop now, with what to do, if the requested GPU does not exist.

    Without this a machine with no CUDA device sails through ``require_free_vram`` (an unknown size
    counts as "nothing to check") and fails much later: the pose stage would try the model on the
    CPU and only then would the generator refuse to start. The pack's rule is to fail before the
    expensive stage, so it is checked first.
    """

    try:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        count = 0
    if count == 0:
        raise NoCudaDevice(
            "No CUDA device is available. This stage runs a multi-billion-parameter model and needs an NVIDIA "
            "GPU with about 30 GB of memory; ComfyUI reported none. Check the NVIDIA driver, or generate the "
            "ring on a machine with a GPU and bring it here with Load Ring."
        )
    text = (device or "").strip()
    if ":" in text:
        try:
            index = int(text.rsplit(":", 1)[1])
        except ValueError:
            return  # a malformed name is reported by the caller that parses it
        if not 0 <= index < count:
            raise NoCudaDevice(
                f"{device} does not exist: this machine has {count} CUDA device(s), cuda:0 to cuda:{count - 1}. "
                "Pick one from the device dropdown."
            )


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
            f"{minimum_gb:.1f} GB. Disable RCP, or lower min_free_vram_gb "
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
