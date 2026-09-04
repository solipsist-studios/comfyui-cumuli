# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Thin PyAV helpers shared by the preview and export paths.

ComfyUI already depends on PyAV, so decoding here never pulls a new dependency
and never needs one of the external checkouts.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


class VideoError(RuntimeError):
    """Raised when a video cannot be probed or decoded."""


@dataclass(frozen=True)
class VideoProbe:
    """The handful of facts the runner needs before it commits to a 47 min job."""

    path: Path
    width: int
    height: int
    fps: Fraction
    num_frames: int
    duration: float
    frames_are_exact: bool

    def frames_from(self, start_time: float) -> int:
        """Frames still available after seeking ``start_time`` seconds in."""

        if start_time <= 0:
            return self.num_frames
        skipped = int(round(start_time * float(self.fps)))
        return max(0, self.num_frames - skipped)


def _open(path: Path):
    import av

    try:
        return av.open(str(path))
    except Exception as exc:  # av raises a family of unrelated errors
        raise VideoError(f"Could not open video {path}: {exc}") from None


def probe(path: str | Path) -> VideoProbe:
    path = Path(path).expanduser()
    if not path.is_file():
        raise VideoError(f"Video file not found: {path}")
    container = _open(path)
    try:
        if not container.streams.video:
            raise VideoError(f"{path} contains no video stream.")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = Fraction(rate) if rate else Fraction(24, 1)
        width = int(stream.codec_context.width or 0)
        height = int(stream.codec_context.height or 0)

        duration = 0.0
        if stream.duration is not None and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = float(container.duration) / 1_000_000.0

        exact = bool(stream.frames)
        if exact:
            count = int(stream.frames)
        else:
            count = int(duration * float(fps)) if duration > 0 else 0
        return VideoProbe(
            path=path.resolve(),
            width=width,
            height=height,
            fps=fps,
            num_frames=count,
            duration=duration,
            frames_are_exact=exact,
        )
    finally:
        container.close()


def iter_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield every frame of ``path`` as an ``uint8`` RGB array."""

    path = Path(path).expanduser()
    container = _open(path)
    try:
        if not container.streams.video:
            raise VideoError(f"{path} contains no video stream.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")
    finally:
        container.close()


def read_frames(path: str | Path, indices: Sequence[int] | None = None) -> np.ndarray:
    """Decode selected frames, returned as ``(N, H, W, 3)`` ``uint8``.

    Decoding is sequential rather than seek based: the generated clips are only
    121 frames long, and sequential decode is both exact and fast enough.
    """

    if indices is not None:
        wanted = sorted({int(i) for i in indices})
        if not wanted:
            raise VideoError("No frame indices requested.")
        if wanted[0] < 0:
            raise VideoError(f"Frame indices must be non-negative, got {wanted[0]}.")
        limit = wanted[-1]
    else:
        wanted = None
        limit = None

    collected: dict[int, np.ndarray] = {}
    frames: list[np.ndarray] = []
    for index, frame in enumerate(iter_frames(path)):
        if wanted is None:
            frames.append(frame)
            continue
        if index in collected or index in set(wanted):
            collected[index] = frame
        if limit is not None and index >= limit:
            break

    if wanted is None:
        if not frames:
            raise VideoError(f"{path} decoded to zero frames.")
        return np.stack(frames)

    missing = [i for i in wanted if i not in collected]
    if missing:
        raise VideoError(f"{path} has no frame {missing[0]} (decoded {len(collected)} of the requested frames).")
    return np.stack([collected[i] for i in wanted])
