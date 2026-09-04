# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Read-side model of a published 4DAnyone result directory.

A result directory (``<data_dir>/fdanyone/<video_stem>/``) contains::

    cameras.json        OpenCV rig for every generated view
    metadata.json       input/output/generation provenance
    skeletons/NN.mp4    conditioning skeleton render per view
    videos/dense/NN.mp4 generated target view (704x1280, 121 frames)
    videos/sparse/NN.mp4 optional RCP proposal views

Everything downstream of the runner talks to this class rather than to the
directory layout directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

CAMERAS_JSON = "cameras.json"
METADATA_JSON = "metadata.json"
DENSE_DIR = "videos/dense"
SPARSE_DIR = "videos/sparse"
SKELETON_DIR = "skeletons"


class RingError(RuntimeError):
    """Raised when a result directory is absent or does not match its manifest."""


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise RingError(f"Missing {path.name}: {path}") from None
    except ValueError as exc:
        raise RingError(f"{path} is not valid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise RingError(f"{path} must contain a JSON object.")
    return payload


def _fraction(value: object, default: Fraction = Fraction(24, 1)) -> Fraction:
    if isinstance(value, str) and "/" in value:
        num, _, den = value.partition("/")
        try:
            return Fraction(int(num), int(den))
        except (ValueError, ZeroDivisionError):
            return default
    try:
        return Fraction(str(value)).limit_denominator(1000000)
    except (ValueError, ZeroDivisionError, TypeError):
        return default


@dataclass(frozen=True)
class RingCamera:
    """One generated view: intrinsics, extrinsics and the files that hold it."""

    camera_id: int
    layer_index: int
    pitch: float
    yaw: float
    k: list[list[float]]
    camera_to_world: list[list[float]]
    width: int
    height: int
    video: str
    skeleton_video: str

    @property
    def label(self) -> str:
        return f"gen{self.camera_id:02d}"

    @property
    def focal(self) -> tuple[float, float]:
        return float(self.k[0][0]), float(self.k[1][1])

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.k[0][2]), float(self.k[1][2])

    def nerf_transform(self) -> list[list[float]]:
        """Convert the OpenCV camera-to-world matrix to the NeRF/OpenGL one.

        4DAnyone publishes ``x right, y down, z forward``. ``transforms.json``
        consumers (nerfstudio, 3DGS, 4DGS) expect ``x right, y up, z back``, so
        the second and third basis columns flip sign. Translation is untouched.
        """

        matrix = [list(map(float, row)) for row in self.camera_to_world]
        for row in matrix[:3]:
            row[1] = -row[1]
            row[2] = -row[2]
        return matrix


@dataclass(frozen=True)
class RingResult:
    """A completed 4DAnyone run on disk."""

    root: Path
    cameras: tuple[RingCamera, ...]
    metadata: dict
    rig: dict

    # -- construction ------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path) -> RingResult:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise RingError(f"4DAnyone result directory not found: {root}")
        rig = _load_json(root / CAMERAS_JSON)
        metadata = _load_json(root / METADATA_JSON)
        records = rig.get("cameras")
        if not isinstance(records, list) or not records:
            raise RingError(f"{root / CAMERAS_JSON} does not list any cameras.")
        cameras = []
        for record in records:
            if not isinstance(record, dict):
                raise RingError(f"{root / CAMERAS_JSON} contains a malformed camera record.")
            camera_id = int(record["camera_id"])
            cameras.append(
                RingCamera(
                    camera_id=camera_id,
                    layer_index=int(record.get("layer_index", 0)),
                    pitch=float(record.get("pitch", 0.0)),
                    yaw=float(record.get("yaw", 0.0)),
                    k=record["K"],
                    camera_to_world=record["camera_to_world"],
                    width=int(record.get("image_width", 0)),
                    height=int(record.get("image_height", 0)),
                    video=str(record.get("video", f"{DENSE_DIR}/{camera_id:02d}.mp4")),
                    skeleton_video=str(record.get("skeleton_video", f"{SKELETON_DIR}/{camera_id:02d}.mp4")),
                )
            )
        return cls(root=root, cameras=tuple(cameras), metadata=metadata, rig=rig)

    # -- convenience -------------------------------------------------------
    @property
    def run_name(self) -> str:
        return self.root.name

    @property
    def num_views(self) -> int:
        return len(self.cameras)

    @property
    def num_frames(self) -> int:
        return int(self.metadata.get("output", {}).get("frames_per_video", 121))

    @property
    def width(self) -> int:
        return int(self.metadata.get("output", {}).get("width", self.cameras[0].width))

    @property
    def height(self) -> int:
        return int(self.metadata.get("output", {}).get("height", self.cameras[0].height))

    @property
    def fps(self) -> Fraction:
        return _fraction(self.metadata.get("output", {}).get("fps", "24/1"))

    @property
    def seed(self) -> int:
        return int(self.metadata.get("generation", {}).get("seed", -1))

    @property
    def view_plan(self) -> dict:
        plan = self.metadata.get("generation", {}).get("view_plan", {})
        return plan if isinstance(plan, dict) else {}

    def frame_time(self, frame_index: int) -> float:
        """Presentation time of ``frame_index`` in seconds from clip start."""

        fps = self.fps
        return float(frame_index * fps.denominator / fps.numerator)

    def camera(self, camera_id: int) -> RingCamera:
        for camera in self.cameras:
            if camera.camera_id == camera_id:
                return camera
        raise RingError(f"View {camera_id} is not part of {self.root} ({self.num_views} views).")

    def video_path(self, camera_id: int) -> Path:
        path = self.root / self.camera(camera_id).video
        if not path.is_file():
            raise RingError(f"Generated video is missing: {path}")
        return path

    def skeleton_path(self, camera_id: int) -> Path:
        path = self.root / self.camera(camera_id).skeleton_video
        if not path.is_file():
            raise RingError(f"Skeleton video is missing: {path}")
        return path

    def sparse_paths(self) -> tuple[Path, ...]:
        sparse = self.root / SPARSE_DIR
        if not sparse.is_dir():
            return ()
        return tuple(sorted(sparse.glob("*.mp4")))

    def summary(self) -> str:
        return (
            f"{self.run_name}: {self.num_views} views x {self.num_frames} frames "
            f"@ {self.width}x{self.height} {float(self.fps):.3f}fps"
        )

    def to_dict(self) -> dict:
        """A JSON-safe view used for the node's text/summary outputs."""

        return {
            "result_dir": str(self.root),
            "run_name": self.run_name,
            "num_views": self.num_views,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "seed": self.seed,
            "view_plan": self.view_plan,
            "camera_ids": [camera.camera_id for camera in self.cameras],
        }
