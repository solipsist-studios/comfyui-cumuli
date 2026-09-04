# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Evaluate a baked 4D asset at one instant, as ComfyUI's native SPLAT type.

A ``.sogst`` stores each gaussian **at its own temporal centre**, not at t = 0,
plus a linear velocity and a temporal window. Per the format specification, a
player evaluates a splat at clip time ``t`` (seconds, absolute) as::

    dt       = t - t_center
    mean(t)  = xyz + v*dt                    (motion.degree == 1)
    mean(t)  = xyz + v*dt + a*dt*dt          (motion.degree == 2)
    alpha(t) = sigmoid(opacity) * exp(-0.5 * (dt / t_sigma)^2)

Rotation, scale and colour are constant in t.

Three things fail silently if you get them wrong, and the spec calls out all
three, so they are stated here too:

1. **The temporal factor is unnormalised** -- no ``1/sqrt(2*pi*sigma^2)``.
   Adding it darkens every splat by a sigma-dependent factor that reads as a
   global exposure bug.
2. **``t_sigma`` is a standard deviation in seconds, not a variance.**
3. **``a`` is the raw ``dt^2`` coefficient, not half-acceleration** -- no
   factor of 1/2.

The input is the 4D interchange PLY (specification section 7): flat float32
columns, all 19 required fields, optionally 45 ``f_rest_*`` and 3 accelerations.
Only reading is done here, and only of columns this module names, so there is no
copy of the container writer to drift out of date. A ``.sogst`` archive is
unpacked to that PLY first, by the pipeline's own ``sogst_ply.py``.

The output is ComfyUI's ``SPLAT``: linear scales, wxyz quaternions, opacity in
0..1 and SH as ``(N, K, 3)``. That plugs straight into the stock **Render
Splat** node, so scrubbing ``time_seconds`` gives a real 4D preview, and Render
Splat's ``frames > 1`` turntable gives an orbit video, with no viewer of our own.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("comfyui-cumuli")

#: The 19 columns every interchange PLY carries, in specification order.
BASE_COLUMNS = (
    "x", "y", "z",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "scale_0", "scale_1", "scale_2",
    "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "vx", "vy", "vz",
    "t_center",
    "t_sigma",
)
ACCEL_COLUMNS = ("ax", "ay", "az")
COMMENT_PREFIX = "sogst."

#: Clip scalars may arrive in PLY comments or in this sidecar. The sidecar wins
#: when both are present, per the format specification.
SIDECAR_SUFFIX = ".sogst.json"
REQUIRED_SCALARS = ("time_min", "time_max", "fps")

#: SH degree 3 is 15 higher-order coefficients per channel.
_F_REST_COUNT = 45


class SogstError(RuntimeError):
    """Raised when a baked asset cannot be read or evaluated."""


@dataclass
class Sogst4D:
    """One baked clip: per-splat spacetime arrays plus the clip's own clock."""

    xyz: np.ndarray  # (N, 3) position at t_center
    velocity: np.ndarray  # (N, 3) scene units per second
    accel: np.ndarray | None  # (N, 3) raw dt^2 coefficient, or None
    rotation: np.ndarray  # (N, 4) wxyz
    log_scale: np.ndarray  # (N, 3) natural log
    logit_opacity: np.ndarray  # (N,)
    t_center: np.ndarray  # (N,) seconds
    t_sigma: np.ndarray  # (N,) seconds, standard deviation
    f_dc: np.ndarray  # (N, 3)
    f_rest: np.ndarray | None  # (N, 45) channel-major
    time_min: float
    time_max: float
    fps: float
    source: Path

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def motion_degree(self) -> int:
        return 2 if self.accel is not None else 1

    @property
    def sh_degree(self) -> int:
        return 3 if self.f_rest is not None else 0

    @property
    def duration(self) -> float:
        return float(self.time_max - self.time_min)

    def summary(self) -> str:
        return (
            f"{self.count:,} splats over [{self.time_min:.3f}, {self.time_max:.3f}]s "
            f"@ {self.fps:g} fps, motion degree {self.motion_degree}, SH degree {self.sh_degree}"
        )

    # -- evaluation -------------------------------------------------------
    def alpha_at(self, time_seconds: float) -> np.ndarray:
        """Per-splat opacity at ``time_seconds``, in 0..1."""

        dt = float(time_seconds) - self.t_center
        # Unnormalised on purpose: see the module docstring.
        envelope = np.exp(-0.5 * np.square(dt / np.maximum(self.t_sigma, 1e-8)))
        peak = 1.0 / (1.0 + np.exp(-self.logit_opacity))
        return (peak * envelope).astype(np.float32)

    def mean_at(self, time_seconds: float) -> np.ndarray:
        """Per-splat centre at ``time_seconds``."""

        dt = (float(time_seconds) - self.t_center)[:, None]
        mean = self.xyz + self.velocity * dt
        if self.accel is not None:
            mean = mean + self.accel * np.square(dt)  # raw dt^2 coefficient, no 1/2
        return mean.astype(np.float32)

    def sh_at(self) -> np.ndarray:
        """SH coefficients as ``(N, K, 3)`` with the DC term first.

        The PLY stores ``f_rest`` channel-major, ``(N, 3*(K-1))``, which is the
        same layout ComfyUI's own PLY reader expects, so the two agree.
        """

        dc = self.f_dc[:, None, :]  # (N, 1, 3)
        if self.f_rest is None:
            return dc.astype(np.float32)
        n = self.count
        k_minus_one = self.f_rest.shape[1] // 3
        rest = self.f_rest.reshape(n, 3, k_minus_one).transpose(0, 2, 1)  # (N, K-1, 3)
        return np.concatenate([dc[:, :, :], rest], axis=1).astype(np.float32)


def _read_ply_columns(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read a flat float32 binary PLY into named columns plus its comments."""

    with path.open("rb") as handle:
        header = b""
        while not header.endswith(b"end_header\n"):
            line = handle.readline()
            if not line:
                raise SogstError(f"{path}: truncated PLY header")
            header += line
        text = header.decode("ascii", "replace")
        if "binary_little_endian" not in text:
            raise SogstError(f"{path}: expected a binary_little_endian PLY")

        count = None
        names: list[str] = []
        comments: dict[str, str] = {}
        for line in text.splitlines():
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float "):
                names.append(line.split()[-1])
            elif line.startswith("property"):
                raise SogstError(f"{path}: non-float property {line!r}; the interchange PLY is all float32")
            elif line.startswith("comment "):
                body = line[len("comment ") :].strip()
                if body.startswith(COMMENT_PREFIX):
                    key, _, value = body[len(COMMENT_PREFIX) :].partition(" ")
                    comments[key] = value.strip()
        if count is None:
            raise SogstError(f"{path}: no 'element vertex' in the header")
        data = np.fromfile(handle, dtype=np.float32, count=count * len(names))

    if data.size != count * len(names):
        raise SogstError(f"{path}: truncated body ({data.size:,} of {count * len(names):,} float32)")
    data = data.reshape(count, len(names))
    return {name: data[:, index] for index, name in enumerate(names)}, comments


def load_interchange_ply(path: str | Path) -> Sogst4D:
    """Read a 4D interchange PLY into :class:`Sogst4D`."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise SogstError(f"Interchange PLY not found: {path}")
    columns, comments = _read_ply_columns(path)

    missing = [name for name in BASE_COLUMNS if name not in columns]
    if missing:
        raise SogstError(
            f"{path} is missing the required column(s) {missing}. This is a 4D interchange PLY "
            "reader; a plain 3DGS PLY has no spacetime fields."
        )

    def stack(names) -> np.ndarray:
        return np.stack([columns[name] for name in names], axis=1).astype(np.float32)

    rest_names = sorted(
        (name for name in columns if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if rest_names and len(rest_names) != _F_REST_COUNT:
        raise SogstError(f"{path}: {len(rest_names)} of {_F_REST_COUNT} f_rest_* columns; all or none.")

    # The sidecar wins over the comments when both carry a key.
    sidecar_path = path.with_name(path.stem + SIDECAR_SUFFIX)
    sidecar: dict = {}
    if sidecar_path.is_file():
        try:
            loaded = json.loads(sidecar_path.read_text())
            if isinstance(loaded, dict):
                sidecar = loaded
        except (OSError, ValueError):
            LOGGER.warning("Ignoring unreadable sidecar %s", sidecar_path)

    def scalar(key: str) -> float:
        for source in (sidecar, comments):
            if key in source:
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    continue
        raise SogstError(
            f"{path} carries no '{key}'. The clip scalars {REQUIRED_SCALARS} place the splats on a "
            "clock, and guessing them would silently map the time slider to the wrong instants. "
            f"They belong in the PLY's 'sogst.' comments or in {sidecar_path.name}; re-emit the PLY "
            "from the bake step rather than editing it."
        )

    t_center = columns["t_center"].astype(np.float32)
    return Sogst4D(
        xyz=stack(("x", "y", "z")),
        velocity=stack(("vx", "vy", "vz")),
        accel=stack(ACCEL_COLUMNS) if all(name in columns for name in ACCEL_COLUMNS) else None,
        rotation=stack(("rot_0", "rot_1", "rot_2", "rot_3")),
        log_scale=stack(("scale_0", "scale_1", "scale_2")),
        logit_opacity=columns["opacity"].astype(np.float32),
        t_center=t_center,
        t_sigma=columns["t_sigma"].astype(np.float32),
        f_dc=stack(("f_dc_0", "f_dc_1", "f_dc_2")),
        f_rest=stack(rest_names) if rest_names else None,
        time_min=scalar("time_min"),
        time_max=scalar("time_max"),
        fps=scalar("fps"),
        source=path,
    )


def to_splat(asset: Sogst4D, time_seconds: float, *, alpha_threshold: float = 1.0 / 255.0):
    """Evaluate at ``time_seconds`` and return ComfyUI's ``SPLAT``.

    Splats whose temporal envelope has faded below ``alpha_threshold`` are
    dropped rather than rendered transparent: at any one instant most of a
    clip's gaussians are outside their window, and culling them is what makes
    the preview fast.
    """

    import torch
    from comfy_api.latest import Types

    alpha = asset.alpha_at(time_seconds)
    keep = alpha > alpha_threshold
    kept = int(keep.sum())
    if kept == 0:
        raise SogstError(
            f"No splat is active at t={time_seconds:.3f}s. The clip covers "
            f"[{asset.time_min:.3f}, {asset.time_max:.3f}]s."
        )

    positions = asset.mean_at(time_seconds)[keep]
    scales = np.exp(asset.log_scale[keep])  # the PLY stores natural log
    rotation = asset.rotation[keep]
    norm = np.linalg.norm(rotation, axis=1, keepdims=True)
    rotation = rotation / np.maximum(norm, 1e-8)  # wxyz, as ComfyUI expects
    sh = asset.sh_at()[keep]

    def tensor(array: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(array)).float()

    return (
        Types.SPLAT(
            tensor(positions)[None],
            tensor(scales)[None],
            tensor(rotation)[None],
            tensor(alpha[keep]).reshape(1, -1, 1),
            tensor(sh)[None],
        ),
        kept,
    )


def frame_time(asset: Sogst4D, frame_index: int) -> float:
    """Clip time of ``frame_index`` on the asset's own clock."""

    fps = asset.fps if asset.fps > 0 else 24.0
    return float(asset.time_min + frame_index / fps)


def frame_count(asset: Sogst4D) -> int:
    fps = asset.fps if asset.fps > 0 else 24.0
    return max(1, int(math.floor(asset.duration * fps)) + 1)
