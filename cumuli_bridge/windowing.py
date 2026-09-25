# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Split a clip's frame count into evenly-sized training windows.

A long clip reconstructs measurably better as several short, independently
trained 4DGS models stitched together than as one wide fit: on a 5 s/121-frame
take, a uniform 4-window split (31/30/30/30 frames) scored LPIPS 0.00778
against 0.00899 for a single wide model. A dynamic-program planner that also
chooses non-uniform boundaries nudged that to 0.00774 -- a further 0.5%, for
a lot of extra machinery (a motion-signal extractor, refit regression
coefficients, a rate-distortion search over window counts). This module is
the simple half of that result: an even split capped at a maximum window
length, which captures nearly all of the gain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class WindowingError(RuntimeError):
    """Raised when a clip's frame count cannot be split into valid windows."""


@dataclass(frozen=True)
class Window:
    """One training window's slice of the parent clip's frames, 0-based."""

    index: int
    frame_start: int
    frame_count: int

    @property
    def frame_end(self) -> int:
        return self.frame_start + self.frame_count

    def offset_seconds(self, fps: float) -> float:
        return self.frame_start / fps


def even_windows(total_frames: int, max_frames: int) -> list[Window]:
    """As-even-as-possible split of ``total_frames`` into windows of at most
    ``max_frames`` each.

    ``total_frames <= max_frames`` returns a single window spanning the whole
    clip -- today's behaviour, unchanged. Otherwise the frame count is split
    into ``ceil(total_frames / max_frames)`` windows of nearly equal length
    (sizes differ by at most one frame, with the extra frame going to the
    earliest windows), rather than repeatedly slicing off ``max_frames`` at a
    time, which would leave a short straggler window at the end. 121 frames
    at ``max_frames=31`` gives ``[31, 30, 30, 30]``, matching the uniform
    4-window split measured above.
    """

    if total_frames < 1:
        raise WindowingError(f"total_frames must be positive, got {total_frames}.")
    if max_frames < 1:
        raise WindowingError(f"max_frames must be positive, got {max_frames}.")
    if total_frames <= max_frames:
        return [Window(index=0, frame_start=0, frame_count=total_frames)]

    n = math.ceil(total_frames / max_frames)
    base, remainder = divmod(total_frames, n)
    windows = []
    start = 0
    for i in range(n):
        count = base + 1 if i < remainder else base
        windows.append(Window(index=i, frame_start=start, frame_count=count))
        start += count
    return windows
