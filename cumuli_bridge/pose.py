# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Body pose from SAM 3D Body, estimated inside ComfyUI.

4DAnyone conditions its generator on a 70-keypoint MHR body pose, and that is exactly
what SAM 3D Body emits (the same 70 names in the same order). ComfyUI ships SAM 3D
Body, so the pose is estimated here, in the interpreter that already has the model
machinery, and handed to the generator as a small npz (``--sam3d_npz``). That replaces
GVHMR and SMPL-X, whose licences do not allow commercial use.

Adapted from ``core/four_d_anyone/pose.py`` in ComfyUI-SplatKit by mickmumpitz, which is
MIT-licensed (copyright 2026 mickmumpitz; the licence is reproduced in
``docs/THIRD_PARTY_NOTICES.md``). Changes: our logging and error types, weights
resolved from a setting, and no dependency on SplatKit's package layout.

The loader is ours for one reason that is not a preference: ComfyUI's own loader picks
float16 on any card that supports it and installs no manual cast, and with hand
refinement on the decoder feeds a float32 intermediate into those half weights and dies
on ``mat1 and mat2 must have the same dtype``. Building in float32 avoids that, at 5.6 GB
of weights instead of 2.8 GB. Everything else is the core loader, line for line.

``comfy`` and ``folder_paths`` are imported inside the functions that need them, so this
module imports cleanly outside ComfyUI (the way ``vram.py`` does).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("comfyui-cumuli")

#: The npz contract the generator validates (``fdanyone.pipeline.SAM3D_KEYS``).
NPZ_KEYS = ("keypoints_incam", "vertices", "cam_t", "keypoints_2d", "intrinsics", "image_size")

DETECTION_FOLDER = "detection"


class PoseError(RuntimeError):
    """Raised when the body pose cannot be estimated, with what to do about it."""


def core_has_sam3d() -> bool:
    """Whether this ComfyUI ships the SAM 3D Body nodes (it arrived in 0.34)."""

    try:
        import comfy_extras.nodes_sam3d_body  # noqa: F401
    except Exception:
        return False
    return True


def resolve_weights(setting: str) -> Path:
    """The SAM 3D Body weights file named by the ``sam3d_weights`` setting.

    An absolute path is used as it is. A bare name is looked up in ComfyUI's
    ``detection`` model folders, which is where the weights are installed.
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
    if found is None:
        raise PoseError(
            f"SAM 3D Body weights '{text}' were not found in ComfyUI's '{DETECTION_FOLDER}' model folder. "
            "Download sam_3d_body_dinov3_bf16.safetensors (Comfy-Org/sam-3d-body on Hugging Face, "
            "under Meta's SAM License) into ComfyUI/models/detection/, or set 'sam3d_weights' to a full path."
        )
    return Path(found)


def host_problem() -> str | None:
    """Why this ComfyUI cannot estimate the pose, or ``None`` when it can."""

    if not core_has_sam3d():
        return (
            "this ComfyUI does not ship the SAM 3D Body nodes (they arrived in 0.34). Update ComfyUI; "
            "the body pose cannot be estimated without them."
        )
    return None


def _read_frames(video: Path) -> np.ndarray:
    import av

    frames = []
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise PoseError(f"No frames could be decoded from {video}.")
    return np.stack(frames)


def _load_model(model_file: str):
    """The core's SAM3DBody_Loader with the dtype pinned to float32 (see the module docstring)."""

    import comfy.model_management
    import comfy.model_patcher
    import comfy.ops
    import comfy.utils
    import torch
    from comfy.ldm.sam3d_body.model.model import SAM3DBody

    state_dict = comfy.utils.load_torch_file(model_file, safe_load=True)
    state_dict = {key.replace(".layers.0.0.", ".layers.0."): value for key, value in state_dict.items()}

    load_device = comfy.model_management.get_torch_device()
    dtype = torch.float32
    quant_config = comfy.utils.detect_layer_quantization(state_dict, "")
    if quant_config is not None:
        operations = comfy.ops.mixed_precision_ops(quant_config, dtype)
    else:
        operations = comfy.ops.pick_operations(dtype, None, load_device=load_device, disable_fast_fp8=True)

    model = SAM3DBody(dtype=dtype, operations=operations)
    state_dict.pop("hand_cls_embed.weight", None)
    state_dict.pop("hand_cls_embed.bias", None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise PoseError(
            f"SAM 3D Body checkpoint key mismatch: missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}. "
            f"{model_file} is not the weights this ComfyUI's SAM 3D Body expects."
        )
    model.backbone_dtype = dtype
    return comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
        size=comfy.model_management.module_size(model),
    )


def _default_intrinsics(height: int, width: int) -> np.ndarray:
    """What the model assumes when no field of view is given: a diagonal focal length and a
    centred principal point. Recorded explicitly, because everything downstream projects with it."""

    focal = float(np.hypot(width, height))
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def estimate_pose(
    video: Path,
    out_npz: Path,
    *,
    weights: Path,
    fov: float = 0.0,
    batch_size: int = 16,
    hands: bool = True,
) -> dict:
    """Run SAM 3D Body over the canonical clip and write the generator's pose npz.

    No person detector is run: the core node estimates one body from the whole frame,
    which is the pipeline's contract (exactly one person). Two people in frame give one
    blended estimate and a poor generation, not an error here.
    """

    problem = host_problem()
    if problem:
        raise PoseError(problem)
    import torch
    from comfy_extras.nodes_sam3d_body import SAM3DBody_Predict

    frames = _read_frames(Path(video))
    count, height, width, _ = frames.shape
    LOGGER.info("Cumuli: estimating body pose with SAM 3D Body (%d frames at %dx%d)", count, width, height)

    patcher = _load_model(str(weights))
    image = torch.from_numpy(frames).float().div_(255.0)  # ComfyUI's IMAGE convention
    pose = SAM3DBody_Predict.execute(
        sam3d_body_model=patcher,
        image=image,
        run_hand_refinement=hands,
        fov=float(fov),
        batch_size=int(batch_size),
    ).result[0]

    keypoints, vertices, cam_t, keypoints_2d = [], [], [], []
    for index, people in enumerate(pose["frames"]):
        if not people:
            raise PoseError(
                f"SAM 3D Body found no body in frame {index} of the clip. The subject must stay in frame "
                "for the whole 121-frame window; pick a start_time where they do."
            )
        person = people[0]
        keypoints.append(np.asarray(person["pred_keypoints_3d"], dtype=np.float32))
        vertices.append(np.asarray(person["pred_vertices"], dtype=np.float32))
        cam_t.append(np.asarray(person["pred_cam_t"], dtype=np.float32))
        keypoints_2d.append(np.asarray(person["pred_keypoints_2d"], dtype=np.float32))

    if fov:
        from comfy_extras.sam3d_body.utils import cam_int_from_fov

        cam_int = cam_int_from_fov(int(height), int(width), float(fov))
        intrinsics = np.asarray(cam_int[0] if cam_int.ndim == 3 else cam_int, dtype=np.float64)
    else:
        intrinsics = _default_intrinsics(height, width)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    # Written aside and moved into place: this file is a cache, and a half-written npz from an
    # interrupted run would be taken for a finished pose by the next one.
    staged = out_npz.with_name(out_npz.name + ".partial.npz")
    np.savez_compressed(
        staged,
        keypoints_incam=np.stack(keypoints),
        vertices=np.stack(vertices),
        cam_t=np.stack(cam_t),
        keypoints_2d=np.stack(keypoints_2d),
        intrinsics=intrinsics,
        image_size=np.asarray((int(height), int(width)), dtype=np.int64),
    )
    os.replace(staged, out_npz)
    # Give the card back before the generator starts; it needs nearly all of it.
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:  # noqa: BLE001 - freeing memory is best effort
        LOGGER.debug("Could not release ComfyUI models after pose estimation", exc_info=True)
    return {
        "path": str(out_npz),
        "frames": count,
        "weights": str(weights),
        "keypoints": int(np.stack(keypoints).shape[1]),
    }
