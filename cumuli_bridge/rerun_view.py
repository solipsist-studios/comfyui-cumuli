# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""A finished ring as a Rerun recording: every camera as a frustum with its generated video
playing on the image plane, on one shared timeline, in the ring's own world frame.

The logging follows 4DAnyone's own viewer (``fdanyone/space/scene.py``): a pinhole per
camera, the camera's pose as a transform, the video attached to the image plane and
indexed by frame time. The ring's frame is right-handed and Y-up with OpenCV cameras
(``cameras.json`` says so), which is exactly Rerun's ``RIGHT_HAND_Y_UP`` world and
``RDF`` pinhole convention, so nothing is converted.

``rerun`` is imported inside the functions and its version is checked against the pinned
viewer's (see ``viewer_assets``): a recording from a newer SDK may not load in an older
viewer. Importable without ComfyUI or the SDK.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from pathlib import Path

import numpy as np

from .ring import RingResult
from .viewer_assets import VIEWER_VERSION

LOGGER = logging.getLogger("comfyui-cumuli")

RERUN_VERSION = VIEWER_VERSION

#: How far in front of each camera its video is drawn, in scene units (metres).
IMAGE_PLANE_DISTANCE = 0.45
#: Where the subject stands (the ring's world origin is the ground under the subject).
SUBJECT_HEIGHT = 0.9


class RerunError(RuntimeError):
    """Raised when the recording cannot be written, with what to do about it."""


def sdk_problem(expected_version: str = RERUN_VERSION) -> str | None:
    """Why this environment cannot write a recording the pinned viewer plays, or ``None``."""

    try:
        import rerun
    except ImportError:
        return (
            f"rerun-sdk is not installed. Install the version that matches the viewer: "
            f"pip install rerun-sdk=={expected_version}"
        )
    if rerun.__version__ != expected_version:
        return (
            f"rerun-sdk {rerun.__version__} is installed but the bundled viewer is {expected_version}, and a "
            f"recording from a different version may not load. Install the matching SDK: "
            f"pip install rerun-sdk=={expected_version} (the installer can do this: ./install.sh --groups viewer)"
        )
    return None


def recording_key(ring: RingResult) -> str:
    """What a ring's recording depends on: the camera rig, every video file (name, size, mtime)
    and the viewer version. An unchanged ring maps to the same key, so its recording is reused."""

    digest = hashlib.sha256(RERUN_VERSION.encode())
    digest.update((ring.root / "cameras.json").read_bytes())
    for camera in ring.cameras:
        path = ring.video_path(camera.camera_id)
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def write_ring_recording(ring: RingResult, destination: Path, *, expected_version: str = RERUN_VERSION) -> dict:
    """Write ``ring`` as an ``.rrd`` at ``destination`` (replaced atomically); return a summary."""

    problem = sdk_problem(expected_version)
    if problem:
        raise RerunError(problem)
    import rerun as rr
    import rerun.blueprint as rrb

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid.uuid4().hex}.rrd")
    fps = float(ring.fps)

    positions = np.array([np.asarray(camera.camera_to_world, dtype=np.float64)[:3, 3] for camera in ring.cameras])
    radius = max(float(np.linalg.norm(positions[:, [0, 2]], axis=1).max()), 1.0)
    target = np.array([0.0, SUBJECT_HEIGHT, 0.0])
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="world",
            name=ring.run_name,
            eye_controls=rrb.EyeControls3D(
                position=target + np.array([0.9, 0.6, 1.2]) * radius,
                look_target=target,
                eye_up=[0, 1, 0],
            ),
        ),
        rrb.TimePanel(timeline="time", play_state="playing", loop_mode="all"),
        rrb.BlueprintPanel(state="hidden"),
        rrb.SelectionPanel(state="hidden"),
    )

    recording = rr.RecordingStream("cumuli_ring", recording_id=f"ring-{uuid.uuid4().hex[:12]}")
    frames = ring.num_frames
    try:
        recording.save(str(temporary), default_blueprint=blueprint)
        recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        recording.log(
            "world/subject",
            rr.Points3D([[0.0, 0.0, 0.0]], radii=0.03, colors=[[255, 120, 0]], labels=["origin"]),
            static=True,
        )
        for camera in ring.cameras:
            entity = f"world/cameras/{camera.camera_id:02d}"
            c2w = np.asarray(camera.camera_to_world, dtype=np.float64)
            recording.log(entity, rr.Transform3D(mat3x3=c2w[:3, :3], translation=c2w[:3, 3]), static=True)
            recording.log(
                entity,
                rr.Pinhole(
                    image_from_camera=np.asarray(camera.k, dtype=np.float64),
                    resolution=[camera.width, camera.height],
                    camera_xyz=rr.ViewCoordinates.RDF,
                    image_plane_distance=IMAGE_PLANE_DISTANCE,
                ),
                static=True,
            )
            asset = rr.AssetVideo(path=ring.video_path(camera.camera_id))
            stamps = asset.read_frame_timestamps_nanos()
            if len(stamps) != frames:
                LOGGER.warning(
                    "view %d has %d frames but the ring says %d; using the video's own count",
                    camera.camera_id, len(stamps), frames,
                )
            recording.log(f"{entity}/image", asset, static=True)
            recording.send_columns(
                f"{entity}/image",
                indexes=[rr.TimeColumn("time", duration=np.arange(len(stamps)) / fps)],
                columns=rr.VideoFrameReference.columns_nanos(stamps),
            )
        recording.flush()
        recording.disconnect()
        os.replace(temporary, destination)
    finally:
        try:
            recording.disconnect()
        except Exception:  # noqa: BLE001 - already closed on the success path
            pass
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "cameras": len(ring.cameras),
        "frames": frames,
        "bytes": destination.stat().st_size,
    }
