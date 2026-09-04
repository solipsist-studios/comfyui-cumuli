# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Turn 4DAnyone's log and tqdm output into a single monotonic 0..1 fraction.

The child process reports progress in four different shapes:

* ``logging`` lines such as ``... | INFO | Estimating source foreground masks``
* tqdm bars ``RCP 1-to-4:  50%|#####     | 12/24 [...]``
* tqdm bars ``Generate 24 target views:  50%|#####     | 12/24 [...]``
* ``Decoding target camera 07`` once per view during VAE decode

The stage weights below come from a real 24-view run on this hardware:
denoising took 3280 s of a 3901 s pipeline and decoding took 495 s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: (start, end) of each stage inside the overall 0..1 progress range.
STAGE_BANDS = {
    "setup": (0.00, 0.02),
    "motion": (0.02, 0.03),
    "skeleton": (0.03, 0.05),
    "rcp": (0.05, 0.07),
    "denoise": (0.07, 0.86),
    "decode": (0.86, 0.99),
}

_TQDM = re.compile(
    r"^(?P<desc>[^:|]*?)\s*:\s*(?P<pct>\d{1,3})%\|.*?\|\s*(?P<n>\d+)\s*/\s*(?P<total>\d+)"
)
_DECODE_TARGET = re.compile(r"Decoding target camera\s+(\d+)")
_DECODE_RCP = re.compile(r"Decoding RCP camera\s+(\d+)")
_RCP_DESC = re.compile(r"^RCP 1-to-\d+$")
_TARGET_DESC = re.compile(r"^Generate\s+(\d+)\s+target views$")

_SETUP_MARKERS = (
    ("Downloading", "downloading model files"),
    ("Reusing validated GVHMR result", "reusing cached GVHMR motion"),
    ("Using cuda", "gpu selected"),
)


@dataclass
class ProgressState:
    """Monotonic progress plus the most recent human-readable status."""

    expected_views: int = 24
    fraction: float = 0.0
    stage: str = "setup"
    message: str = "starting"
    _decoded: int = field(default=0, repr=False)

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
        """Fold one output line in. Returns ``True`` when the UI should refresh."""

        text = line.strip()
        if not text:
            return False

        match = _TQDM.search(text)
        if match:
            desc = match.group("desc").strip()
            # A merged log line can prefix the bar; keep only its tail.
            desc = desc.rsplit("|", 1)[-1].strip()
            done = int(match.group("n"))
            total = max(1, int(match.group("total")))
            ratio = done / total
            if _RCP_DESC.match(desc):
                return self._advance("rcp", ratio, f"proposal views {done}/{total}")
            target = _TARGET_DESC.match(desc)
            if target:
                self.expected_views = int(target.group(1)) or self.expected_views
                return self._advance("denoise", ratio, f"denoising step {done}/{total}")
            return False

        decode = _DECODE_TARGET.search(text)
        if decode:
            self._decoded = max(self._decoded, int(decode.group(1)) + 1)
            ratio = self._decoded / max(1, self.expected_views)
            return self._advance("decode", ratio, f"decoding view {self._decoded}/{self.expected_views}")

        if _DECODE_RCP.search(text):
            return self._advance("rcp", 1.0, "decoding proposal views")

        if "foreground masks" in text:
            return self._advance("skeleton", 0.2, "estimating foreground masks")
        if "skeleton" in text.lower() and "Loading skeleton conditioning" in text:
            return self._advance("skeleton", 0.9, "loading skeleton conditioning")
        if "gvhmr" in text.lower() or "GVHMR" in text:
            return self._advance("motion", 0.5, "solving body motion (GVHMR)")
        for marker, message in _SETUP_MARKERS:
            if marker in text:
                return self._advance("setup", 0.5, message)
        return False
