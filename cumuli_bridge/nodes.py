# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""ComfyUI V3 nodes for the cumuli pipeline: one clip or capture to ``.sogst``.

The only ComfyUI-aware module in the pack. Every node is registered in the
``Cumuli`` category with a ``Cumuli*`` id, and user-facing errors carry a
``Cumuli:`` prefix -- 4DAnyone names only the external checkout the ring
generator drives, one of three (with the OMG4 trainer and cumuli's
``bake_sogst.py``).

Everything here runs inside ``comfyenv``. Ring generation and training shell
out to *this same* interpreter, so each can configure the CUDA allocator before
its first allocation; see ``settings.py``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from typing_extensions import override

import folder_paths
from comfy.model_management import InterruptProcessingException, processing_interrupted
from comfy.utils import ProgressBar
from comfy_api.latest import ComfyExtension, InputImpl, IO

from . import runner
from .dataset4d import DatasetError, DatasetHandle, DatasetOptions, build_dataset
from .flipbook import MASKS_SUBDIR, FlipbookError, check_complete, load_flipbook, write_flipbook
from .masks import MaskError, mask_coverage, matte_flipbook
from .process import SubprocessCancelled, SubprocessError
from .ring import RingError, RingResult
from . import sfm
from .settings import SettingsError, load_settings
from .sogst import SogstError, frame_count, frame_time, load_interchange_ply, to_splat
from .train import BakeOptions, TrainOptions, TrainingError, bake, train, unpack_sogst_to_ply
from .validate import ValidationError, validate_dataset
from .videoio import read_frames
from .vram import InsufficientVRAM, require_free_vram

LOGGER = logging.getLogger("comfyui-cumuli")

CATEGORY = "Cumuli"

#: Opaque handles passed between the nodes.
Ring = IO.Custom("CUMULI_RING")
FlipbookIO = IO.Custom("CUMULI_FLIPBOOK")
DatasetIO = IO.Custom("CUMULI_DATASET")
#: A solved real-capture rig: the poses a physical rig does not ship with.
RigIO = IO.Custom("CUMULI_RIG")

_PROGRESS_STEPS = 1000


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _cancelled() -> bool:
    return bool(processing_interrupted())


def _send_text(node_id: str | None, text: str) -> None:
    if not node_id:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_progress_text(text, node_id)
    except Exception:  # pragma: no cover - the UI is optional
        LOGGER.debug("Could not send progress text", exc_info=True)


def _images_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """``(N, H, W, 3)`` uint8 -> the float32 0..1 IMAGE tensor ComfyUI expects."""

    return torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)


def _progress_reporter(node_id: str | None):
    bar = ProgressBar(_PROGRESS_STEPS, node_id=node_id)

    def report(fraction: float, message: str) -> None:
        bar.update_absolute(int(max(0.0, min(1.0, fraction)) * _PROGRESS_STEPS), _PROGRESS_STEPS)
        _send_text(node_id, f"{fraction * 100:.0f}% {message}")

    return bar, report


def _tile(tiles: list[np.ndarray], columns: int, tile_height: int) -> np.ndarray:
    columns = max(1, min(int(columns), len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    tile_w = max(tile.shape[1] for tile in tiles)
    sheet = np.zeros((rows * tile_height, columns * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * tile_height : row * tile_height + tile.shape[0],
            column * tile_w : column * tile_w + tile.shape[1],
        ] = tile
    return sheet


def _label_tile(image, text: str):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 8 + 7 * len(text), 16), fill=(0, 0, 0))
    draw.text((4, 3), text, fill=(255, 255, 255))
    return image




def parse_view_selection(text: str, available: list[int]) -> tuple[int, ...] | None:
    """Parse ``"0,6,12"`` or ``"0-11"`` into view ids; empty means every view."""

    text = (text or "").strip()
    if not text:
        return None
    selected: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            low, _, high = part.partition("-")
            try:
                start, end = int(low), int(high)
            except ValueError:
                raise RuntimeError(f"Cumuli: cannot read view range {part!r}.") from None
            selected.extend(range(min(start, end), max(start, end) + 1))
        else:
            try:
                selected.append(int(part))
            except ValueError:
                raise RuntimeError(f"Cumuli: cannot read view id {part!r}.") from None
    unknown = [view for view in selected if view not in available]
    if unknown:
        raise RuntimeError(f"Cumuli: this ring has no view {unknown[0]} (it has {available[0]}..{available[-1]}).")
    return tuple(dict.fromkeys(selected))


def parse_labels(text: str, known: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a camera-label list (``"00,12"`` or ``"12"``); empty means none."""

    text = (text or "").strip()
    if not text:
        return ()
    labels = tuple(part.strip() for part in text.replace(";", ",").split(",") if part.strip())
    unknown = [label for label in labels if label not in known]
    if unknown:
        raise RuntimeError(f"Cumuli: unknown camera label {unknown[0]!r}. Known labels: {', '.join(known)}.")
    return labels


def _staging_root(explicit: str, run_name: str, settings=None) -> Path:
    text = (explicit or "").strip()
    if text:
        return Path(text).expanduser()
    work = settings.work_dir(run_name) if settings is not None else None
    if work is not None:
        return work / "flipbook_src"
    return Path(folder_paths.get_temp_directory()) / "cumuli" / run_name / "flipbook_src"


# --------------------------------------------------------------------------
# 1. generate
# --------------------------------------------------------------------------
_NO_DATASETS = "(none found -- add dataset_roots to the bridge config)"
_NO_FLIPBOOKS = "(none found -- add flipbook_roots to the bridge config)"
_NO_RINGS = "(none found -- generate a ring, or add ring_roots to the bridge config)"


def _discovered(kind: str) -> list[str]:
    try:
        settings = load_settings()
    except SettingsError:
        return []
    if kind == "datasets":
        return runner.discover_datasets(settings)
    if kind == "rings":
        return runner.discover_rings(settings)
    return runner.discover_flipbooks(settings)


def _register_option_routes() -> None:
    """Serve the loader combos' remote option lists (GET /cumuli/options/*).

    The same discovery also runs in define_schema, so the server-side combo
    validation always re-scans at queue time; these routes only feed the
    widget's refresh button between /object_info reloads. Registered lazily so
    the module stays importable without a running server."""

    try:
        from aiohttp import web
        from server import PromptServer

        routes = PromptServer.instance.routes
    except Exception:  # standalone import (tests, CLI)
        return

    @routes.get("/cumuli/options/datasets")
    async def _datasets(request):
        return web.json_response(_discovered("datasets") or [_NO_DATASETS])

    @routes.get("/cumuli/options/flipbooks")
    async def _flipbooks(request):
        return web.json_response(_discovered("flipbooks") or [_NO_FLIPBOOKS])

    @routes.get("/cumuli/options/rings")
    async def _rings(request):
        return web.json_response(_discovered("rings") or [_NO_RINGS])


_register_option_routes()


class CumuliGenerateRing(IO.ComfyNode):
    """Generate a synchronized ring of novel views from one monocular clip.

    The input is a **file on disk**, not an IMAGE batch, for three reasons:
    4DAnyone re-decodes the clip itself against a canonical presentation clock;
    the file stem is the cache key for the reusable GVHMR motion solve and the
    name of the published result directory; and the job runs in a separate
    interpreter, so any in-memory batch would have to be re-encoded anyway.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliGenerateRing",
            display_name="Cumuli Generate Ring (4DAnyone)",
            category=CATEGORY,
            description=(
                "Runs the 4DAnyone human 4D reconstruction pipeline in its own conda environment and "
                "returns the generated ring of target views. A 24-view ring takes roughly 45 minutes "
                "and needs the whole GPU, so ComfyUI's models are unloaded first."
            ),
            is_experimental=True,
            inputs=[
                IO.Model.Input(
                    "model",
                    optional=True,
                    tooltip=(
                        "Optional: the 4DAnyone checkpoint via ComfyUI's own Load Diffusion "
                        "Model node (it detects the DiT as Wan 2.2), with standard LoRA "
                        "loaders on top. The accumulated LoRA stack is folded and merged "
                        "into the DiT weights inside the subprocess -- zero extra VRAM. Put "
                        "trigger words in 'prompt'."
                    ),
                ),
                IO.Video.Input(
                    "video",
                    tooltip=(
                        "Source video, e.g. from a Load Video node. Needs at least 121 usable frames "
                        "at 720p or better. A file-backed, untrimmed video is used in place; a trimmed "
                        "or synthesized one is written out and then requires run_name."
                    ),
                ),
                IO.String.Input(
                    "run_name",
                    default="",
                    tooltip=(
                        "Optional name for this run. Empty uses the video's filename stem. "
                        "The name keys both the result directory and the reusable GVHMR motion cache."
                    ),
                    optional=True,
                ),
                IO.Int.Input("views_per_layer", default=24, min=4, max=96, step=1,
                             tooltip="Evenly spaced yaw views per pitch layer. Must divide by 4 or 6."),
                IO.Combo.Input("views_per_group", options=["4", "6", "auto"], default="4",
                               tooltip="Target views denoised together. 6 overruns 32 GB; 4 is the safe value."),
                IO.String.Input("layer_pitches", default="15",
                                tooltip="Camera pitch per layer in degrees, comma separated, each between -15 and 45."),
                IO.Int.Input("start_yaw", default=0, min=-180, max=180, step=1,
                             tooltip="First yaw in every layer. 0 faces the person."),
                IO.Int.Input("yaw_span", default=360, min=1, max=360, step=1,
                             tooltip="Angular range each layer sweeps. The end angle is excluded."),
                IO.Boolean.Input("enable_rcp", default=True,
                                 tooltip="Reference-view proposals: anchors at yaw 60/135/210/285 that every "
                                         "dense group is then conditioned on. Without it the far side of the "
                                         "ring is invented per group and will not reconstruct. Costs one extra "
                                         "packed view (about 2.9 GiB) and ~10 min; fits 32 GB at group 4."),
                IO.Boolean.Input("enable_tcr", default=True,
                                 tooltip="Shift view groups between denoising steps for cross-view consistency."),
                IO.Float.Input("start_time", default=0.0, min=0.0, max=86400.0, step=0.1,
                               tooltip="Where the fixed 121-frame window starts on the input timeline, in seconds."),
                IO.String.Input("target_fps", default="auto",
                                tooltip="'auto' keeps the source clock unless it divides cleanly to 24/25/30."),
                IO.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=True),
                IO.String.Input("device", default="cuda:0",
                                tooltip="CUDA device for the subprocess.", advanced=True),
                IO.String.Input("prompt", default="",
                                tooltip="Override the model's fixed prompt -- put LoRA trigger words "
                                        "here (mixing with the stock Chinese prompt is fine, e.g. "
                                        "'rin_karasuba, \u89c6\u9891\u4e2d\u7684\u4eba\u5728\u505a\u52a8\u4f5c'). Empty keeps the stock prompt."),
                IO.Float.Input("min_free_vram_gb", default=-1.0, min=-1.0, max=200.0, step=0.5,
                               tooltip="Refuse to start below this much free VRAM. -1 uses the bridge config value.",
                               advanced=True),
                IO.Boolean.Input("dry_run", default=False,
                                 tooltip="Validate everything and report the command line without running it.",
                                 advanced=True),
                # Appended last on purpose: a saved workflow stores widget values
                # positionally, so inserting anywhere earlier shifts every value
                # after it.
                IO.Boolean.Input("enable_turbo", default=True,
                                 tooltip="Use 4DAnyone-Turbo (a distilled LoRA over the same base "
                                         "checkpoint) for accelerated denoising. Off runs the base "
                                         "model: slower, and the configuration the pack's empirical "
                                         "settings were measured against. Part of the ring "
                                         "fingerprint, so switching regenerates.",
                                 advanced=True),
                # Also appended last, for the same reason as enable_turbo above.
                IO.String.Input("motion_source", default="", optional=True,
                                tooltip="A directory holding motion.safetensors + motion.json "
                                        "(fit_rig_motion.py's output) to condition this run on, "
                                        "in place of running GVHMR on 'video'. For the rig-gap-fill "
                                        "pipeline: every generation run for one capture shares the "
                                        "same fit, since it comes from the real rig's own pose, not "
                                        "from any one camera's footage. Empty runs GVHMR as normal.",
                                advanced=True),
            ],
            outputs=[
                Ring.Output(display_name="ring"),
                IO.String.Output(display_name="result_dir"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        video,
        model=None,
        run_name="",
        views_per_layer=24,
        views_per_group="4",
        layer_pitches="15",
        start_yaw=0,
        yaw_span=360,
        enable_rcp=True,
        enable_tcr=True,
        start_time=0.0,
        target_fps="auto",
        seed=42,
        device="cuda:0",
        prompt="",
        min_free_vram_gb=-1.0,
        dry_run=False,
        enable_turbo=True,
        motion_source="",
    ) -> IO.NodeOutput:
        node_id = cls.hidden.unique_id
        try:
            settings = load_settings()
            settings.validate()
            source = runner.materialize_video_source(video, run_name, settings.source_root)
            staged = runner.stage_source_video(settings, source, run_name)
            request = runner.build_request(
                video_path=staged,
                views_per_layer=views_per_layer,
                layer_pitches=layer_pitches,
                start_yaw=start_yaw,
                yaw_span=yaw_span,
                views_per_group=views_per_group,
                enable_rcp=enable_rcp,
                enable_tcr=enable_tcr,
                start_time=start_time,
                target_fps=target_fps,
                seed=seed,
                device=device,
                enable_turbo=enable_turbo,
                motion_source=(motion_source or "").strip() or None,
            )
            lora_note = None
            if model is not None and getattr(model, "patches", None):
                factors, merged, unsupported = runner.extract_lora_factors(model.patches)
                if unsupported:
                    raise runner.ValidationError(
                        "The connected model carries patches this bridge cannot fold into "
                        "plain LoRA factors: " + "; ".join(sorted(set(unsupported))[:4])
                    )
                if factors:
                    work = settings.work_dir(request.run_name)
                    root = work if work is not None else (
                        Path(folder_paths.get_temp_directory()) / "cumuli" / request.run_name
                    )
                    lora_file = runner.save_lora_factors(factors, root / "lora_merge.safetensors")
                    request = dataclasses.replace(request, lora_path=str(lora_file))
                    lora_note = f"lora: {merged} weights folded -> {lora_file}"
            elif model is not None:
                lora_note = "lora: model connected but carries no patches; base weights unchanged"
            if (prompt or "").strip():
                request = dataclasses.replace(request, prompt=prompt.strip())
            requirements = runner.check_video(settings, request)
            result_dir = settings.result_dir(request.run_name)
            motion_injected = runner.inject_motion(settings, request)
            fingerprint, motion_key = runner.ring_fingerprint(request)
            previous_stamp = runner.read_stamp(result_dir)
            decision = runner.prepare_artifact_dir(result_dir, fingerprint)
            argv = runner.build_argv(settings, request)
        except (SettingsError, runner.ValidationError) as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        cached_motion = runner.motion_is_cached(settings, request)
        header = [
            f"run_name: {request.run_name}",
            f"input: {requirements.probe.width}x{requirements.probe.height} "
            f"@ {float(requirements.probe.fps):.3f} fps, {requirements.probe.num_frames} frames",
            f"clip: {requirements.num_frames} frames @ {float(requirements.canonical_fps):.3f} fps "
            f"from {request.start_time:.2f}s",
            f"views: {request.num_target_views} ({request.views_per_layer} per layer x "
            f"{len(request.layer_pitches)} layer(s), group {request.views_per_group}, rcp={request.enable_rcp})",
            f"result_dir: {result_dir}",
            f"gvhmr motion cache: {'reused' if cached_motion else 'will be solved'} "
            f"({settings.motion_dir(request.run_name)})",
            "command: " + " ".join(argv),
        ]
        if motion_injected:
            header.append(f"motion: injected from {request.motion_source} (GVHMR skipped)")
        if lora_note:
            header.append(lora_note)
        if request.prompt:
            header.append(f"prompt override: {request.prompt!r}")

        if dry_run:
            _send_text(node_id, "dry run")
            return IO.NodeOutput(None, str(result_dir), "DRY RUN\n" + "\n".join(header))

        if decision == "reuse" and (result_dir / "metadata.json").is_file():
            ring = RingResult.load(result_dir)
            _send_text(node_id, "cached result (inputs unchanged)")
            return IO.NodeOutput(ring, str(result_dir), "CACHED RESULT\n" + "\n".join(header))
        if runner.clear_stale_motion(settings, request, motion_key, previous_stamp):
            header.append("gvhmr motion cache: cleared (clip changed)")

        minimum = settings.min_free_vram_gb if min_free_vram_gb < 0 else float(min_free_vram_gb)
        try:
            free_gb, total_gb = require_free_vram(request.device, minimum)
        except InsufficientVRAM as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None
        header.append(f"vram: {free_gb:.1f} GB free of {total_gb:.1f} GB (need {minimum:.1f} GB)")

        progress = ProgressBar(_PROGRESS_STEPS, node_id=node_id)

        def on_progress(state) -> None:
            progress.update_absolute(int(state.fraction * _PROGRESS_STEPS), _PROGRESS_STEPS)
            _send_text(node_id, f"{state.fraction * 100:.0f}% {state.message}")

        try:
            outcome = runner.execute(
                settings,
                request,
                on_progress=on_progress,
                should_cancel=_cancelled,
            )
        except SubprocessCancelled:
            raise InterruptProcessingException() from None
        except (SubprocessError, runner.ValidationError) as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        progress.update_absolute(_PROGRESS_STEPS, _PROGRESS_STEPS)
        runner.write_stamp(outcome.result_dir, fingerprint, motion_key=motion_key)
        ring = RingResult.load(outcome.result_dir)
        elapsed = outcome.result.elapsed if outcome.result else 0.0
        report = "\n".join(header + [f"elapsed: {elapsed / 60.0:.1f} min", ring.summary()])
        _send_text(node_id, ring.summary())
        return IO.NodeOutput(ring, str(outcome.result_dir), report)


# --------------------------------------------------------------------------
# 2. load / inspect
# --------------------------------------------------------------------------
class CumuliLoadRing(IO.ComfyNode):
    """Open an already generated result directory without re-running anything."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliLoadRing",
            display_name="Cumuli Load Ring",
            category=CATEGORY,
            description="Load a published 4DAnyone result directory (the one holding cameras.json).",
            inputs=[
                IO.Combo.Input(
                    "result_dir",
                    options=_discovered("rings") or [_NO_RINGS],
                    remote=IO.RemoteOptions(route="/cumuli/options/rings", refresh_button=True),
                    tooltip="Finished rings discovered under <data_dir>/fdanyone and the bridge "
                            "config's ring_roots (directories holding cameras.json and videos/). "
                            "The config is re-read on refresh.",
                ),
            ],
            outputs=[
                Ring.Output(display_name="ring"),
                IO.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, result_dir="") -> IO.NodeOutput:
        text = (result_dir or "").strip()
        if not text or text.startswith("(none found"):
            raise RuntimeError(
                "Cumuli: no ring selected. Generate one, or add its parent directory to "
                "ring_roots in the bridge config and refresh the widget."
            )
        path = Path(text).expanduser()
        # A bare run name still resolves, so a typed value or a workflow saved
        # before this was a combo keeps working.
        if not path.is_absolute() and not path.exists():
            try:
                path = load_settings().result_dir(text)
            except SettingsError as exc:
                raise RuntimeError(f"Cumuli: {exc}") from None
        try:
            ring = RingResult.load(path)
        except RingError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None
        return IO.NodeOutput(ring, json.dumps(ring.to_dict(), indent=1))

    @classmethod
    def fingerprint_inputs(cls, result_dir):
        text = (result_dir or "").strip()
        path = Path(text).expanduser()
        if not path.is_absolute() and not path.exists():
            try:
                path = load_settings().result_dir(text)
            except SettingsError:
                return float("nan")
        try:
            return os.path.getmtime(path / "metadata.json")
        except OSError:
            return float("nan")


class CumuliRingContactSheet(IO.ComfyNode):
    """Tile one frame from every view into a single contact-sheet IMAGE."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliRingContactSheet",
            display_name="Cumuli Ring Contact Sheet",
            category=CATEGORY,
            description=(
                "Decodes the same frame from every generated view and tiles it, so a 24-view ring can be "
                "checked for identity drift and view consistency at a glance."
            ),
            inputs=[
                Ring.Input("ring"),
                IO.Int.Input("frame_index", default=0, min=0, max=4096, step=1,
                             tooltip="Frame to sample from every view, 0 based."),
                IO.Int.Input("columns", default=6, min=1, max=24, step=1),
                IO.Int.Input("tile_height", default=320, min=64, max=1280, step=16,
                             tooltip="Height each view is resized to before tiling."),
                IO.Combo.Input("source", options=["generated", "skeleton"], default="generated"),
                IO.Boolean.Input("label_views", default=True,
                                 tooltip="Burn the view id and yaw into each tile."),
            ],
            outputs=[IO.Image.Output(display_name="contact_sheet")],
        )

    @classmethod
    def execute(cls, ring, frame_index=0, columns=6, tile_height=320, source="generated", label_views=True):
        from PIL import Image

        if ring is None:
            raise RuntimeError("Cumuli: no ring connected. Run the generator or load a result directory.")
        frame_index = int(frame_index)
        if frame_index >= ring.num_frames:
            raise RuntimeError(
                f"Cumuli: frame_index {frame_index} is past the end of this ring ({ring.num_frames} frames)."
            )
        tiles = []
        for camera in ring.cameras:
            if _cancelled():
                raise InterruptProcessingException()
            path = ring.skeleton_path(camera.camera_id) if source == "skeleton" else ring.video_path(camera.camera_id)
            image = Image.fromarray(read_frames(path, [frame_index])[0])
            scale = tile_height / image.height
            image = image.resize((max(1, int(round(image.width * scale))), tile_height), Image.LANCZOS)
            if label_views:
                image = _label_tile(image, f"{camera.label}  {camera.yaw:.0f}deg")
            tiles.append(np.asarray(image, dtype=np.uint8))
        return IO.NodeOutput(_images_to_tensor(_tile(tiles, columns, tile_height)[None]))


class CumuliSelectView(IO.ComfyNode):
    """Pull one view out of the ring as a VIDEO plus an optional IMAGE batch."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliSelectView",
            display_name="Cumuli Select View",
            category=CATEGORY,
            description="Return one generated view for downstream preview, saving or image processing.",
            inputs=[
                Ring.Input("ring"),
                IO.Int.Input("camera_id", default=0, min=0, max=255, step=1),
                IO.Combo.Input("source", options=["generated", "skeleton"], default="generated"),
                IO.Boolean.Input("decode_frames", default=False,
                                 tooltip="Also decode the clip to an IMAGE batch. 121 frames at 704x1280 is ~1.3 GB."),
                IO.Int.Input("max_frames", default=121, min=1, max=4096, step=1,
                             tooltip="Cap on decoded frames when decode_frames is on."),
            ],
            outputs=[
                IO.Video.Output(display_name="video"),
                IO.Image.Output(display_name="frames"),
                IO.String.Output(display_name="camera_json"),
            ],
        )

    @classmethod
    def execute(cls, ring, camera_id=0, source="generated", decode_frames=False, max_frames=121):
        if ring is None:
            raise RuntimeError("Cumuli: no ring connected.")
        try:
            camera = ring.camera(int(camera_id))
            path = ring.skeleton_path(camera.camera_id) if source == "skeleton" else ring.video_path(camera.camera_id)
        except RingError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        if decode_frames:
            indices = list(range(min(int(max_frames), ring.num_frames)))
            frames = _images_to_tensor(read_frames(path, indices))
        else:
            # One frame keeps the socket typed without materializing the clip.
            frames = _images_to_tensor(read_frames(path, [0]))

        info = {
            "camera_id": camera.camera_id,
            "label": camera.label,
            "yaw": camera.yaw,
            "pitch": camera.pitch,
            "K": camera.k,
            "camera_to_world_opencv": camera.camera_to_world,
            "transform_matrix_opengl": camera.nerf_transform(),
            "video": str(path),
        }
        return IO.NodeOutput(InputImpl.VideoFromFile(str(path)), frames, json.dumps(info, indent=1))


# --------------------------------------------------------------------------
# 3. stage -> matte -> build
# --------------------------------------------------------------------------
class CumuliStageRing(IO.ComfyNode):
    """Decode the ring into the frame-major staging tree the builder reads."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliStageRing",
            display_name="Cumuli Stage Ring",
            category=CATEGORY,
            description=(
                "Transposes the ring (one video per camera) into one directory per frame, with an "
                "images_flat/<label>.png per camera and a per-frame transforms.json carrying "
                "OpenGL camera-to-world poses. This is the input the 4DGS dataset builder reads."
            ),
            inputs=[
                Ring.Input("ring"),
                IO.String.Input("staging_dir", default="",
                                tooltip="Where to stage. Empty uses <ComfyUI temp>/cumuli/<run_name>/flipbook_src.",
                                optional=True),
                IO.String.Input("views", default="",
                                tooltip="Views to stage, e.g. '0,6,12,18' or '0-11'. Empty stages all.",
                                optional=True),
                IO.Int.Input("frame_stride", default=1, min=1, max=16, step=1,
                             tooltip="Keep every Nth frame. The effective fps is divided to match."),
                IO.Int.Input("max_frames", default=0, min=0, max=4096, step=1,
                             tooltip="Cap on staged frames per view. 0 stages the whole clip.", advanced=True),
            ],
            outputs=[
                FlipbookIO.Output(display_name="flipbook"),
                IO.String.Output(display_name="staging_dir"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, ring, staging_dir="", views="", frame_stride=1, max_frames=0) -> IO.NodeOutput:
        if ring is None:
            raise RuntimeError("Cumuli: no ring connected.")
        node_id = cls.hidden.unique_id
        bar, report = _progress_reporter(node_id)
        try:
            settings = load_settings()
        except SettingsError:
            settings = None
        root = _staging_root(staging_dir, ring.run_name, settings)
        selection = parse_view_selection(views, [camera.camera_id for camera in ring.cameras])

        try:
            flipbook = write_flipbook(
                ring,
                root,
                camera_ids=selection,
                frame_stride=int(frame_stride),
                max_frames=int(max_frames) or None,
                # ComfyUI's cache decides whether this node runs at all; when it
                # does run, its inputs changed, so staged frames are re-encoded.
                overwrite=True,
                on_progress=lambda done, total: report(done / total, f"staging view {done}/{total}"),
                should_cancel=_cancelled,
            )
        except FlipbookError as exc:
            if _cancelled():
                raise InterruptProcessingException() from None
            raise RuntimeError(f"Cumuli: {exc}") from None

        ring_stamp = runner.read_stamp(ring.root)
        runner.write_stamp(flipbook.root, runner.compute_fingerprint({
            "ring": ring_stamp["fingerprint"] if ring_stamp else runner.file_identity(ring.root / "metadata.json"),
            "views": views or "", "frame_stride": int(frame_stride), "max_frames": int(max_frames),
        }), fps=str(flipbook.fps))
        bar.update_absolute(_PROGRESS_STEPS, _PROGRESS_STEPS)
        text = "\n".join(
            [
                f"staging_dir: {flipbook.root}",
                f"cameras: {len(flipbook.labels)} ({flipbook.labels[0]}..{flipbook.labels[-1]})",
                f"frames: {flipbook.num_frames} @ {float(flipbook.fps):.3f} fps "
                f"({flipbook.duration():.3f}s)",
                f"resolution: {flipbook.width}x{flipbook.height}",
            ]
        )
        _send_text(node_id, f"{flipbook.num_frames} frames x {len(flipbook.labels)} views")
        return IO.NodeOutput(flipbook, str(flipbook.root), text)


class CumuliSolveRig(IO.ComfyNode):
    """Solve a real multi-camera rig into poses, so a capture can enter the pipeline.

    The 4DAnyone path produces a ring whose cameras are known by construction.
    Footage off a physical rig has no poses at all, and every stage downstream
    needs them, so this is where a real capture starts.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliSolveRig",
            display_name="Cumuli Solve Rig (HLOC)",
            category=CATEGORY,
            description=(
                "Runs multi-timestamp HLOC/pycolmap structure-from-motion over a directory of "
                "per-camera videos and solves one shared pose per physical camera. The rig is "
                "static, so every sampled timestamp is an independent measurement of the same "
                "poses: tracks span space and time, and people moving in the scene fall out as "
                "outliers. Matching is exhaustive, so pairs grow with (cameras x timestamps)^2 -- "
                "keep timestamps at 1 on a large rig. The capture directory is only ever read."
            ),
            inputs=[
                IO.String.Input("capture_dir", default="",
                                tooltip="Directory of per-camera videos, one file per camera. The file stem becomes the camera label."),
                IO.String.Input("outputs_dir", default="",
                                tooltip="Where the solve goes. Empty uses <work_root>/<run_name>/sfm.",
                                optional=True),
                IO.String.Input("run_name", default="rig",
                                tooltip="Names the output directory when outputs_dir is empty.",
                                optional=True),
                IO.Int.Input("camera_stride", default=1, min=1, max=64, step=1,
                             tooltip="Keep every Nth camera. Thinning a dense rig cuts the quadratic matching cost, "
                                     "but widens the baseline between neighbours -- too large and cameras stop registering."),
                IO.Int.Input("max_cameras", default=0, min=0, max=512, step=1,
                             tooltip="Cap the camera count after striding. 0 keeps them all.", optional=True),
                IO.Boolean.Input("drop_odd_formats", default=True,
                                 tooltip="Drop cameras whose resolution or fps differs from the majority. "
                                         "Off makes a mixed rig an error instead."),
                IO.Int.Input("num_timestamps", default=1, min=1, max=24, step=1,
                             tooltip="Time instants sampled per camera. More is better conditioned but costs "
                                     "quadratically: 200 cameras only afford 1. Below 2 the solve cannot "
                                     "measure whether its own poses are repeatable."),
                IO.String.Input("sync_json", default="",
                                tooltip="Per-camera frame offsets from measure_sync.py. Empty assumes the clips "
                                        "are already frame-aligned -- which a hardware trigger does not "
                                        "guarantee when the clips have different lengths.",
                                optional=True),
                IO.Combo.Input("feature_type", options=["aliked", "superpoint"], default="aliked",
                               tooltip="Local feature. aliked is BSD-licensed; superpoint's weights are "
                                       "non-commercial research use only."),
                IO.Int.Input("resize_max", default=1920, min=512, max=4096, step=64,
                             tooltip="Longest image side fed to the detector."),
                IO.Int.Input("max_keypoints", default=8192, min=512, max=32768, step=512,
                             tooltip="Keypoints per image.", advanced=True),
                IO.Float.Input("focal_guess", default=1.2, min=0.2, max=5.0, step=0.05,
                               tooltip="Seed focal length as a multiple of image width, used when the rig ships "
                                       "no calibration. The solve refines it.", advanced=True),
                IO.Boolean.Input("refine_intrinsics", default=True,
                                 tooltip="Refine focal length in the final global bundle adjustment. "
                                         "Intrinsics stay locked during incremental mapping either way.",
                                 advanced=True),
            ],
            outputs=[
                RigIO.Output(display_name="rig"),
                IO.String.Output(display_name="transforms_path"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        capture_dir="",
        outputs_dir="",
        run_name="rig",
        camera_stride=1,
        max_cameras=0,
        drop_odd_formats=True,
        num_timestamps=1,
        sync_json="",
        feature_type="aliked",
        resize_max=1920,
        max_keypoints=8192,
        focal_guess=1.2,
        refine_intrinsics=True,
    ) -> IO.NodeOutput:
        node_id = cls.hidden.unique_id
        bar, report = _progress_reporter(node_id)
        try:
            settings = load_settings()
        except SettingsError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        capture = (capture_dir or "").strip()
        if not capture:
            raise RuntimeError("Cumuli: set capture_dir to the rig's directory of per-camera videos.")

        root = (outputs_dir or "").strip()
        if root:
            solve_root = Path(root).expanduser()
        else:
            work = settings.work_dir(run_name or "rig")
            solve_root = (work / "sfm") if work is not None else (
                Path(folder_paths.get_temp_directory()) / "cumuli" / (run_name or "rig") / "sfm"
            )

        notes: list[str] = []
        try:
            cameras = sfm.discover_cameras(
                capture, stride=int(camera_stride), limit=int(max_cameras) or None
            )
            if drop_odd_formats:
                cameras, dropped = sfm.select_majority_format(cameras)
                if dropped:
                    listed = ", ".join(dropped[:8]) + ("..." if len(dropped) > 8 else "")
                    notes.append(f"dropped {len(dropped)} odd-format cameras: {listed}")
                    LOGGER.info("Solve Rig dropped %d odd-format cameras", len(dropped))
            if len(cameras) < 2:
                raise sfm.SfmError(
                    f"Only {len(cameras)} camera survived selection; a rig solve needs at least two."
                )

            sync = (sync_json or "").strip()
            if sync and not Path(sync).expanduser().is_file():
                raise sfm.SfmError(f"sync_json not found: {sync}")
            options = sfm.SolveOptions(
                videos_dir=Path(capture).expanduser(),
                outputs_dir=solve_root,
                sync_json=Path(sync).expanduser() if sync else None,
                num_timestamps=int(num_timestamps),
                feature_type=str(feature_type),
                resize_max=int(resize_max),
                max_keypoints=int(max_keypoints),
                focal_guess=float(focal_guess),
                refine_intrinsics=bool(refine_intrinsics),
            )
            _send_text(node_id, f"solving {len(cameras)} cameras")
            solve = sfm.solve_rig(
                settings,
                options,
                cameras,
                on_progress=report,
                should_cancel=_cancelled,
            )
        except SubprocessCancelled:
            raise InterruptProcessingException() from None
        except (sfm.SfmError, SubprocessError, ValidationError) as exc:
            if _cancelled():
                raise InterruptProcessingException() from None
            raise RuntimeError(f"Cumuli: {exc}") from None

        bar.update_absolute(_PROGRESS_STEPS, _PROGRESS_STEPS)
        text = "\n".join(
            notes
            + [
                f"outputs_dir: {solve.root}",
                f"cameras solved: {len(solve.labels)} of {len(cameras)} submitted",
                solve.summary(),
            ]
        )
        unsolved = solve.unsolved()
        _send_text(
            node_id,
            f"{len(solve.labels) - len(unsolved)}/{len(cameras)} cameras solved",
        )
        return IO.NodeOutput(solve, str(solve.transforms_path), text)


class CumuliStageCapture(IO.ComfyNode):
    """Stage a solved real capture into the frame-major tree the builder reads.

    The counterpart of Stage Ring for footage that came off a physical rig:
    same output contract, poses from the solve instead of from construction.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliStageCapture",
            display_name="Cumuli Stage Capture",
            category=CATEGORY,
            description=(
                "Transposes a solved capture into one directory per frame, with an "
                "images_flat/<label>.png per camera and a per-frame transforms.json carrying the "
                "solved OpenGL camera-to-world poses. Only cameras that registered are staged, and "
                "the window is clamped to the shortest clip so every camera appears in every frame. "
                "Feed the result straight into Ring Masks."
            ),
            inputs=[
                RigIO.Input("rig"),
                IO.String.Input("capture_dir", default="",
                                tooltip="The same directory of per-camera videos that was solved."),
                IO.String.Input("staging_dir", default="",
                                tooltip="Where to stage. Empty uses <work_root>/<run_name>/flipbook_src.",
                                optional=True),
                IO.String.Input("run_name", default="rig",
                                tooltip="Names the staging directory when staging_dir is empty.",
                                optional=True),
                IO.Int.Input("start_frame", default=0, min=0, max=100000, step=1,
                             tooltip="First frame of the window, in source frames."),
                IO.Int.Input("num_frames", default=121, min=0, max=4096, step=1,
                             tooltip="Frames to stage after striding. 0 stages to the end of the shortest clip."),
                IO.Int.Input("frame_stride", default=1, min=1, max=16, step=1,
                             tooltip="Keep every Nth frame. The effective fps is divided to match."),
            ],
            outputs=[
                FlipbookIO.Output(display_name="flipbook"),
                IO.String.Output(display_name="staging_dir"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        rig=None,
        capture_dir="",
        staging_dir="",
        run_name="rig",
        start_frame=0,
        num_frames=121,
        frame_stride=1,
    ) -> IO.NodeOutput:
        if rig is None:
            raise RuntimeError("Cumuli: no rig connected; run Solve Rig first.")
        capture = (capture_dir or "").strip()
        if not capture:
            raise RuntimeError("Cumuli: set capture_dir to the same videos that were solved.")

        node_id = cls.hidden.unique_id
        bar, report = _progress_reporter(node_id)
        try:
            settings = load_settings()
        except SettingsError:
            settings = None
        root = _staging_root(staging_dir, run_name or "rig", settings)

        try:
            # Discover with no striding: the solve already fixed the camera set,
            # and stage_capture stages exactly the cameras it names.
            cameras = sfm.discover_cameras(capture)
            flipbook = sfm.stage_capture(
                rig,
                cameras,
                root,
                start_frame=int(start_frame),
                num_frames=int(num_frames) or None,
                frame_stride=int(frame_stride),
                on_progress=lambda done, total: report(done / total, f"staging camera {done}/{total}"),
                should_cancel=_cancelled,
            )
        except sfm.SfmError as exc:
            if _cancelled():
                raise InterruptProcessingException() from None
            raise RuntimeError(f"Cumuli: {exc}") from None

        runner.write_stamp(
            flipbook.root,
            runner.compute_fingerprint({
                "solve": str(rig.transforms_path),
                "cameras": list(flipbook.labels),
                "start_frame": int(start_frame),
                "num_frames": int(num_frames),
                "frame_stride": int(frame_stride),
            }),
            fps=str(flipbook.fps),
        )
        bar.update_absolute(_PROGRESS_STEPS, _PROGRESS_STEPS)
        text = "\n".join(
            [
                f"staging_dir: {flipbook.root}",
                f"cameras: {len(flipbook.labels)} ({flipbook.labels[0]}..{flipbook.labels[-1]})",
                f"frames: {flipbook.num_frames} @ {float(flipbook.fps):.3f} fps "
                f"({flipbook.duration():.3f}s)",
                f"resolution: {flipbook.width}x{flipbook.height}",
            ]
        )
        _send_text(node_id, f"{flipbook.num_frames} frames x {len(flipbook.labels)} cameras")
        return IO.NodeOutput(flipbook, str(flipbook.root), text)


class CumuliLoadFlipbook(IO.ComfyNode):
    """Open a staged flipbook tree without re-running Stage Ring -- one node, one job.

    Re-enters the pipeline at the masking/dataset stage, for a tree this
    bridge staged earlier or for an external frame-major capture
    (frame_NNNN/images_flat/<label>.png plus a per-frame transforms.json).
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliLoadFlipbook",
            display_name="Cumuli Load Flipbook",
            category=CATEGORY,
            description=(
                "Loads an existing staged flipbook directory so Ring Masks and Build "
                "4DGS Dataset can run on it. Accepts external frame-major captures too."
            ),
            is_experimental=True,
            inputs=[
                IO.Combo.Input(
                    "flipbook_dir",
                    options=_discovered("flipbooks") or [_NO_FLIPBOOKS],
                    remote=IO.RemoteOptions(route="/cumuli/options/flipbooks", refresh_button=True),
                    tooltip="Staged trees discovered under work_root and the bridge config's "
                            "flipbook_roots (frame_NNNN dirs with images_flat/ and "
                            "transforms.json). The config is re-read on refresh.",
                ),
                IO.Float.Input("fps", default=0.0, min=0.0, max=240.0, step=0.001,
                               tooltip="Source frame rate. 0 reads it from the Stage Ring stamp; external "
                                       "trees carry no rate, so set it explicitly for those."),
            ],
            outputs=[
                FlipbookIO.Output(display_name="flipbook"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, flipbook_dir, fps):
        root = Path((flipbook_dir or "").strip()).expanduser()
        try:
            frames = sorted(d.name for d in root.iterdir() if d.name.startswith("frame_"))
            stat = (root / frames[0] / "transforms.json").stat() if frames else None
            tail = f"{stat.st_size}:{stat.st_mtime_ns}" if stat else "none"
            return f"{len(frames)}|{tail}|{fps}"
        except OSError:
            return f"missing|{fps}"

    @classmethod
    def execute(cls, flipbook_dir="", fps=0.0) -> IO.NodeOutput:
        from fractions import Fraction

        root = Path((flipbook_dir or "").strip()).expanduser()
        if not str(flipbook_dir).strip() or str(flipbook_dir).startswith("(none found"):
            raise RuntimeError(
                "Cumuli: no flipbook selected. Stage a ring, or add the tree's parent "
                "directory to flipbook_roots in the bridge config and refresh the widget."
            )
        rate = Fraction(str(fps)) if float(fps) > 0 else Fraction(0)
        stamp = runner.read_stamp(root)
        if rate <= 0 and stamp and stamp.get("fps"):
            rate = Fraction(stamp["fps"])
        try:
            flipbook = load_flipbook(root, rate)
        except FlipbookError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None
        lines = [
            f"staging_dir: {flipbook.root}",
            f"cameras: {len(flipbook.labels)} ({flipbook.labels[0]}..{flipbook.labels[-1]})",
            f"frames: {flipbook.num_frames} @ {float(flipbook.fps):.3f} fps ({flipbook.duration():.3f}s)",
            f"resolution: {flipbook.width}x{flipbook.height}",
            "provenance: staged by this bridge (fingerprinted)" if stamp
            else "provenance: external tree (set fps manually; masks may still be needed)",
        ]
        _send_text(cls.hidden.unique_id, f"{flipbook.num_frames} frames x {len(flipbook.labels)} views")
        return IO.NodeOutput(flipbook, "\n".join(lines))


class CumuliRingMasks(IO.ComfyNode):
    """Matte the staged ring with ComfyUI's own background-removal model."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliRingMasks",
            display_name="Cumuli Ring Masks",
            category=CATEGORY,
            description=(
                "Writes a foreground matte per camera per frame beside the staged images. The dataset "
                "builder carves its visual hull from these and bakes them into the training alpha, so "
                "they are required, not optional: an unmasked ring trains the hallucinated background "
                "as scene content. Feed it ComfyUI's 'Load Background Removal Model' node."
            ),
            inputs=[
                FlipbookIO.Input("flipbook"),
                IO.BackgroundRemoval.Input("bg_removal_model",
                                           tooltip="For example birefnet.safetensors from models/background_removal."),
                IO.Int.Input("batch_size", default=8, min=1, max=64, step=1,
                             tooltip="Images matted per forward pass."),
                IO.Int.Input("preview_columns", default=6, min=1, max=24, step=1, advanced=True),
                IO.Int.Input("preview_height", default=240, min=64, max=1280, step=16, advanced=True),
            ],
            outputs=[
                FlipbookIO.Output(display_name="flipbook"),
                IO.Image.Output(display_name="mask_preview"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, flipbook, bg_removal_model, batch_size=8,
                preview_columns=6, preview_height=240) -> IO.NodeOutput:
        from PIL import Image

        if flipbook is None:
            raise RuntimeError("Cumuli: no staged flipbook connected. Add a 'Stage Ring' node first.")
        node_id = cls.hidden.unique_id
        bar, report = _progress_reporter(node_id)

        try:
            written = matte_flipbook(
                flipbook,
                bg_removal_model,
                subdir=MASKS_SUBDIR,
                batch_size=int(batch_size),
                overwrite=True,
                on_progress=lambda done, total: report(done / total, f"matting {done}/{total}"),
                should_cancel=_cancelled,
            )
        except MaskError as exc:
            if _cancelled():
                raise InterruptProcessingException() from None
            raise RuntimeError(f"Cumuli: {exc}") from None

        coverage = mask_coverage(flipbook, MASKS_SUBDIR)
        notes = []
        if coverage < 0.005:
            notes.append(
                f"WARNING: the middle frame's masks cover only {coverage * 100:.2f}% of the image. "
                "The visual hull will collapse. Check the preview and the matting model."
            )

        middle = flipbook.root / f"frame_{flipbook.num_frames // 2:04d}" / MASKS_SUBDIR
        tiles = []
        for label in flipbook.labels:
            path = middle / f"{label}.png"
            if not path.is_file():
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                scale = preview_height / image.height
                image = image.resize((max(1, int(round(image.width * scale))), preview_height), Image.LANCZOS)
                tiles.append(np.asarray(_label_tile(image, label), dtype=np.uint8))
        preview = _tile(tiles, preview_columns, preview_height)[None] if tiles else np.zeros((1, 8, 8, 3), np.uint8)

        stage_stamp = runner.read_stamp(flipbook.root)
        runner.write_stamp(flipbook.root, runner.compute_fingerprint({
            "stage": stage_stamp["fingerprint"] if stage_stamp else str(flipbook.root),
            "bg_model": type(bg_removal_model).__name__,
        }), filename=runner.MASKS_FINGERPRINT_FILE)
        bar.update_absolute(_PROGRESS_STEPS, _PROGRESS_STEPS)
        text = "\n".join(
            [
                f"masks: {written} written to {flipbook.root}/frame_*/{MASKS_SUBDIR}",
                f"total expected: {flipbook.num_frames * len(flipbook.labels)}",
                f"middle-frame foreground coverage: {coverage * 100:.2f}%",
                *notes,
            ]
        )
        _send_text(node_id, f"{written} mattes, {coverage * 100:.1f}% coverage")
        return IO.NodeOutput(flipbook, _images_to_tensor(preview), text)


class CumuliBuildDataset(IO.ComfyNode):
    """Assemble the D-NeRF style 4DGS training dataset from the staged ring."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliBuildDataset",
            display_name="Cumuli Build 4DGS Dataset",
            category=CATEGORY,
            description=(
                "Carves a time-stamped visual-hull init cloud, bakes RGBA frames with the mattes in "
                "alpha, and writes transforms_train.json / transforms_test.json with per-camera "
                "intrinsics. The result trains directly in a rotor 4DGS trainer."
            ),
            inputs=[
                FlipbookIO.Input("flipbook"),
                IO.String.Input("out_dir", default="",
                                tooltip="Destination. Empty writes <ComfyUI output>/cumuli/<run>/dataset_4dgs.",
                                optional=True),
                IO.Float.Input("fps", default=0.0, min=0.0, max=240.0, step=0.001,
                               tooltip="Timeline fps. 0 uses the staged clip's own rate."),
                IO.Int.Input("downscale", default=1, min=1, max=8, step=1,
                             tooltip="Integer image downscale. 1 keeps full 704x1280 resolution."),
                IO.String.Input("test_cameras", default="",
                                tooltip="Camera labels held out and scored, e.g. '12'. Empty means all cameras train.",
                                optional=True),
                IO.String.Input("holdout_cameras", default="",
                                tooltip="Extra labels excluded from training but not scored.",
                                optional=True, advanced=True),
                IO.Int.Input("hull_points", default=300000, min=1000, max=3000000, step=1000, advanced=True),
                IO.Int.Input("hull_min_views", default=9, min=1, max=96, step=1,
                             tooltip="How many camera masks a hull point must fall inside. Lower it if the "
                                     "hull collapses on an inconsistent ring.", advanced=True),
                IO.Int.Input("jobs", default=8, min=1, max=32, step=1, advanced=True),
                IO.Boolean.Input("validate", default=True,
                                 tooltip="Check the finished dataset against the trainer's contract."),
            ],
            outputs=[
                IO.String.Output(display_name="dataset_dir"),
                IO.String.Output(display_name="report"),
                DatasetIO.Output(display_name="dataset"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        flipbook,
        out_dir="",
        fps=0.0,
        downscale=1,
        test_cameras="",
        holdout_cameras="",
        hull_points=300000,
        hull_min_views=9,
        jobs=8,
        validate=True,
    ) -> IO.NodeOutput:
        if flipbook is None:
            raise RuntimeError("Cumuli: no staged flipbook connected. Add 'Stage Ring' and 'Ring Masks' first.")
        node_id = cls.hidden.unique_id
        bar, report = _progress_reporter(node_id)

        destination = (out_dir or "").strip()
        if destination:
            target = Path(destination).expanduser()
        else:
            run_name = flipbook.root.parent.name
            try:
                work = load_settings().work_dir(run_name)
            except SettingsError:
                work = None
            target = (
                work / "dataset_4dgs"
                if work is not None
                else Path(folder_paths.get_output_directory()) / "cumuli" / run_name / "dataset_4dgs"
            )

        try:
            check_complete(flipbook, MASKS_SUBDIR)
        except FlipbookError as exc:
            raise RuntimeError(
                f"Cumuli: {exc} Run the 'Cumuli Ring Masks' node on this flipbook before building."
            ) from None

        masks_stamp = runner.read_stamp(flipbook.root, filename=runner.MASKS_FINGERPRINT_FILE)
        ds_fingerprint = runner.compute_fingerprint({
            "masks": masks_stamp["fingerprint"] if masks_stamp else str(flipbook.root),
            "fps": float(fps) or float(flipbook.fps),
            "downscale": int(downscale),
            "test_cameras": test_cameras or "",
            "holdout_cameras": holdout_cameras or "",
            "hull_points": int(hull_points),
            "hull_min_views": int(hull_min_views),
            "validate": bool(validate),
        })
        try:
            decision = runner.prepare_artifact_dir(target, ds_fingerprint)
        except runner.ValidationError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None
        if decision == "reuse":
            stamp = runner.read_stamp(target) or {}
            if "report" in stamp:
                _send_text(node_id, "cached dataset (inputs unchanged)")
                handle = DatasetHandle(root=target, fingerprint=ds_fingerprint, source="built")
                return IO.NodeOutput(str(target), "CACHED DATASET\n" + stamp.get("report", str(target)), handle)
            # Our stamp but no success report: a crashed build. Rebuild it.
            shutil.rmtree(target, ignore_errors=True)

        target.mkdir(parents=True, exist_ok=True)
        runner.write_stamp(target, ds_fingerprint)  # pre-stamp: ours even if we crash

        options = DatasetOptions(
            out_dir=target,
            fps=float(fps) or float(flipbook.fps),
            downscale=int(downscale),
            test_cameras=parse_labels(test_cameras, flipbook.labels),
            holdout_cameras=parse_labels(holdout_cameras, flipbook.labels),
            masks_dir=MASKS_SUBDIR,
            hull_points=int(hull_points),
            hull_min_views=int(hull_min_views),
            jobs=int(jobs),
        )
        try:
            summary = build_dataset(flipbook, options, on_progress=report, should_cancel=_cancelled)
        except DatasetError as exc:
            if _cancelled():
                raise InterruptProcessingException() from None
            raise RuntimeError(f"Cumuli: {exc}") from None

        checked = None
        if validate:
            try:
                checked = validate_dataset(summary.out_dir)
            except ValidationError as exc:
                raise RuntimeError(f"Cumuli: the dataset failed its own contract check: {exc}") from None

        provenance = {
            "source_flipbook": str(flipbook.root),
            "camera_ids": list(flipbook.camera_ids),
            "camera_labels": list(flipbook.labels),
            "fps": options.fps,
            "downscale": options.downscale,
            "world": (
                "4DAnyone canonical human world: normalized scale (the subject is roughly 1.2 units tall, "
                "not metres) and yawed so the subject faces +Z. Self-consistent for standalone training; "
                "mixing these views with a real camera rig needs a similarity alignment first."
            ),
            "camera_convention": "transform_matrix is OpenGL/Blender camera-to-world (OpenCV y/z columns negated)",
            "summary": summary.to_dict(),
            "validation": checked,
        }
        (summary.out_dir / "cumuli_export.json").write_text(json.dumps(provenance, indent=1))

        bar.update_absolute(_PROGRESS_STEPS, _PROGRESS_STEPS)
        report_text = _dataset_report(summary, options, checked)
        runner.write_stamp(summary.out_dir, ds_fingerprint, report=report_text)
        _send_text(node_id, f"{summary.init_points:,} init points")
        handle = DatasetHandle(root=summary.out_dir, fingerprint=ds_fingerprint, source="built")
        return IO.NodeOutput(str(summary.out_dir), report_text, handle)


def _dataset_report(summary, options, checked) -> str:
    duration = summary.duration_seconds
    lines = [
        f"dataset_dir: {summary.out_dir}",
        f"train cameras: {len(summary.train_cameras)} ({', '.join(summary.train_cameras)})",
        f"test cameras: {', '.join(summary.test_cameras) or 'none (see note)'}",
        f"frames: {summary.num_frames} @ {options.fps:.3f} fps ({duration:.3f}s)",
        f"images: {summary.images_written} RGBA, downscale {options.downscale}",
        f"init cloud: {summary.init_points:,} points with per-point time",
    ]
    if checked:
        lines.append(f"validated: {checked['train_entries']} train entries, probe image is RGBA")
    lines.extend(f"note: {note}" for note in summary.notes)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 4. train -> bake
# --------------------------------------------------------------------------
class CumuliLoadDataset(IO.ComfyNode):
    """Open an existing 4DGS dataset directory -- one node, one job.

    Validates the trainer's contract (transforms, init cloud, image files) and
    describes what it found, so a real multi-view capture can flow into the
    training section exactly like a generated ring's dataset. Produces no
    files and never writes into the directory.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliLoadDataset",
            display_name="Cumuli Load 4DGS Dataset",
            category=CATEGORY,
            description=(
                "Validates and describes an existing 4DGS dataset (D-NeRF layout: "
                "transforms_train.json + points3d.ply) so it can feed Train 4DGS. "
                "Works for real multi-view captures as well as generated rings."
            ),
            is_experimental=True,
            inputs=[
                IO.Combo.Input(
                    "dataset_dir",
                    options=_discovered("datasets") or [_NO_DATASETS],
                    remote=IO.RemoteOptions(route="/cumuli/options/datasets", refresh_button=True),
                    tooltip="Datasets discovered under work_root and the bridge config's "
                            "dataset_roots (transforms_train.json + points3d.ply). Add roots "
                            "to the config to surface captures stored elsewhere; the config is "
                            "re-read on refresh, no restart needed.",
                ),
            ],
            outputs=[
                DatasetIO.Output(display_name="dataset"),
                IO.String.Output(display_name="dataset_dir"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, dataset_dir):
        # Cache like LoadVideo: identical path re-validates only when the
        # dataset's own files change.
        root = Path((dataset_dir or "").strip()).expanduser()
        parts = []
        for name in ("transforms_train.json", "transforms_test.json", "points3d.ply"):
            try:
                stat = (root / name).stat()
                parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
            except OSError:
                parts.append(f"{name}:missing")
        return "|".join(parts)

    @classmethod
    def execute(cls, dataset_dir="") -> IO.NodeOutput:
        root = Path((dataset_dir or "").strip()).expanduser()
        if not str(dataset_dir).strip() or str(dataset_dir).startswith("(none found"):
            raise RuntimeError(
                "Cumuli: no dataset selected. Build one, or add its parent directory "
                "to dataset_roots in the bridge config and refresh the widget."
            )
        try:
            checked = validate_dataset(root)
        except ValidationError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None
        duration, rate, timestamps = _dataset_timeline(root)
        stamp = runner.read_stamp(root)
        lines = [
            f"dataset_dir: {root.resolve()}",
            f"timeline: {timestamps} timestamps, {duration:.3f}s @ {rate:.3f} fps",
            *(f"{key}: {value}" for key, value in sorted(checked.items())),
            "provenance: built by this bridge (fingerprinted)" if stamp
            else "provenance: external dataset (identified by file mtime/size)",
        ]
        _send_text(cls.hidden.unique_id, f"{timestamps} timestamps, {duration:.2f}s")
        handle = DatasetHandle(
            root=root.resolve(),
            fingerprint=stamp.get("fingerprint") if stamp else None,
            source="built" if stamp else "external",
        )
        return IO.NodeOutput(handle, str(root.resolve()), "\n".join(lines))


class CumuliTrain4DGS(IO.ComfyNode):
    """Train a rotor 4D Gaussian Splatting model on the generated ring.

    This is a long node on purpose: 30k iterations is roughly an hour on a
    5090. It reports the trainer's own iteration count and PSNR to the progress
    bar, and it keeps the upstream resume rule -- a finished checkpoint for the
    requested iteration count is reused instead of retrained.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliTrain4DGS",
            display_name="Cumuli Train 4DGS",
            category=CATEGORY,
            description=(
                "Trains the rotor 4DGS model on a dataset built from the ring. Runs as a child "
                "process of ComfyUI's own interpreter and needs the whole GPU, so ComfyUI's models "
                "are unloaded first. Expect roughly an hour for the default 30000 iterations."
            ),
            is_experimental=True,
            inputs=[
                DatasetIO.Input("dataset",
                                tooltip="From Build 4DGS Dataset, or Cumuli Load 4DGS Dataset for "
                                        "an existing directory (including real multi-view captures)."),
                IO.Float.Input("duration_seconds", default=0.0, min=0.0, max=600.0, step=0.001,
                               tooltip="Clip length. 0 reads it from the dataset's own timestamps."),
                IO.Float.Input("fps", default=0.0, min=0.0, max=240.0, step=0.001,
                               tooltip="Advisory frame rate. 0 reads it from the dataset."),
                IO.String.Input("out_dir", default="",
                                tooltip="Run directory for the config and checkpoints. "
                                        "Empty uses the dataset's parent.", optional=True),
                IO.Int.Input("iterations", default=30000, min=100, max=200000, step=100),
                IO.Int.Input("num_pts", default=100000, min=1000, max=2000000, step=1000),
                IO.Int.Input("batch_size", default=2, min=1, max=16, step=1),
                IO.Combo.Input("sh_degree", options=["3", "2", "1"], default="2",
                               tooltip="Spherical-harmonic degree. 2 measured better than 3 on "
                                       "subject captures (30.87 vs 30.04 dB)."),
                IO.Int.Input("densify_until_iter", default=25000, min=100, max=200000, step=100,
                             advanced=True),
                IO.Int.Input("densify_until_num_points", default=3000000, min=10000, max=20000000,
                             step=10000, advanced=True),
                IO.Int.Input("t_init_div", default=100, min=0, max=1000, step=1,
                             tooltip="Initial temporal sigma is sqrt(duration/div). 0 keeps the "
                                     "trainer's own default of 5, which smears short clips.",
                             advanced=True),
                IO.Float.Input("min_free_vram_gb", default=-1.0, min=-1.0, max=200.0, step=0.5,
                               advanced=True),
                IO.Boolean.Input("dry_run", default=False,
                                 tooltip="Write the config and report the command without training.",
                                 advanced=True),
            ],
            outputs=[
                IO.String.Output(display_name="checkpoint"),
                IO.Float.Output(display_name="duration_seconds"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        dataset,
        duration_seconds=0.0,
        fps=0.0,
        out_dir="",
        iterations=30000,
        num_pts=100000,
        batch_size=2,
        sh_degree="2",
        densify_until_iter=25000,
        densify_until_num_points=3000000,
        t_init_div=100,
        min_free_vram_gb=-1.0,
        dry_run=False,
    ) -> IO.NodeOutput:
        node_id = cls.hidden.unique_id
        if dataset is None:
            raise RuntimeError(
                "Cumuli: no dataset connected. Link Build 4DGS Dataset, or "
                "Cumuli Load 4DGS Dataset for an existing directory."
            )
        handle = dataset
        dataset = Path(handle.root)
        if not dataset.is_dir():
            raise RuntimeError(f"Cumuli: dataset directory does not exist: {dataset}")
        duration, dataset_fps, frames = _dataset_timeline(dataset)
        duration = float(duration_seconds) or duration
        rate = float(fps) or dataset_fps

        try:
            settings = load_settings()
            settings.validate_trainer()
        except SettingsError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        options = TrainOptions(
            out_dir=Path((out_dir or "").strip()).expanduser() if out_dir.strip() else dataset.parent,
            dataset_dir=dataset,
            duration_seconds=duration,
            fps=rate,
            iterations=int(iterations),
            num_pts=int(num_pts),
            batch_size=int(batch_size),
            densify_until_iter=int(densify_until_iter),
            densify_until_num_points=int(densify_until_num_points),
            t_init_div=int(t_init_div),
            sh_degree=int(sh_degree),
        )
        train_fingerprint = runner.compute_fingerprint({
            "dataset": handle.fingerprint
            if handle.fingerprint
            else runner.file_identity(dataset / "transforms_train.json"),
            "duration": duration, "fps": rate,
            "iterations": int(iterations), "num_pts": int(num_pts),
            "batch_size": int(batch_size), "sh_degree": int(sh_degree),
            "densify_until_iter": int(densify_until_iter),
            "densify_until_num_points": int(densify_until_num_points),
            "t_init_div": int(t_init_div),
        })
        header = [
            f"dataset: {dataset} ({frames} timestamps, {duration:.3f}s @ {rate:.3f} fps)",
            f"model_dir: {options.model_dir}",
            f"iterations: {options.iterations}  num_pts: {options.num_pts}  "
            f"batch_size: {options.batch_size}  sh_degree: {options.sh_degree}",
            f"checkpoint: {options.checkpoint}",
        ]

        if dry_run:
            try:
                from .train import build_train_argv, write_config

                config = write_config(settings, options)
                argv = build_train_argv(settings, options, config)
            except TrainingError as exc:
                raise RuntimeError(f"Cumuli: {exc}") from None
            return IO.NodeOutput(str(options.checkpoint), duration,
                                 "DRY RUN\n" + "\n".join(header + ["command: " + " ".join(argv)]))

        try:
            decision = runner.prepare_artifact_dir(options.model_dir, train_fingerprint)
        except runner.ValidationError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None
        if decision == "reuse" and options.checkpoint.is_file():
            header.append("cached checkpoint (inputs unchanged)")
            _send_text(node_id, "cached checkpoint")
            return IO.NodeOutput(str(options.checkpoint), duration, "\n".join(header))
        if decision == "reuse":
            # Our stamp, no checkpoint: a crashed run. Rebuilding our own
            # incomplete artifact is safe; foreign dirs still error above.
            shutil.rmtree(options.model_dir, ignore_errors=True)
            header.append("previous training crashed before finishing; retraining")

        minimum = settings.min_free_vram_gb if min_free_vram_gb < 0 else float(min_free_vram_gb)
        try:
            free_gb, total_gb = require_free_vram(settings.device, minimum)
        except InsufficientVRAM as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None
        header.append(f"vram: {free_gb:.1f} GB free of {total_gb:.1f} GB")

        options.model_dir.mkdir(parents=True, exist_ok=True)
        runner.write_stamp(options.model_dir, train_fingerprint)  # pre-stamp: ours even if we crash

        progress = ProgressBar(_PROGRESS_STEPS, node_id=node_id)

        def on_progress(state) -> None:
            progress.update_absolute(int(state.fraction * _PROGRESS_STEPS), _PROGRESS_STEPS)
            _send_text(node_id, f"{state.fraction * 100:.0f}% {state.message}")

        try:
            outcome = train(settings, options, on_progress=on_progress, should_cancel=_cancelled)
        except SubprocessCancelled:
            raise InterruptProcessingException() from None
        except (TrainingError, SubprocessError) as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        progress.update_absolute(_PROGRESS_STEPS, _PROGRESS_STEPS)
        elapsed = outcome.result.elapsed if outcome.result else 0.0
        header.append(f"elapsed: {elapsed / 60.0:.1f} min")
        if outcome.final_psnr is not None:
            header.append(f"final training PSNR: {outcome.final_psnr:.2f} dB")
        runner.write_stamp(options.model_dir, train_fingerprint)
        _send_text(node_id, "training complete")
        return IO.NodeOutput(str(outcome.checkpoint), duration, "\n".join(header))


class CumuliBakeSogst(IO.ComfyNode):
    """Bake a trained checkpoint into a streamable ``.sogst`` asset."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliBakeSogst",
            display_name="Cumuli Bake SOGST",
            category=CATEGORY,
            description=(
                "Slices the trained 4D Gaussians into the .sogst container: positions at each "
                "splat's temporal centre, linear motion, temporal sigma and spherical harmonics, "
                "packed as lossless WebP texture planes. The result lands in ComfyUI's output "
                "folder and appears as a downloadable artifact."
            ),
            is_experimental=True,
            inputs=[
                IO.String.Input("checkpoint", default="", tooltip="A chkpntNNNNN.pth from the trainer."),
                IO.Float.Input("duration_seconds", default=0.0, min=0.0, max=600.0, step=0.001,
                               tooltip="Must match the time_duration the model trained with."),
                IO.Float.Input("fps", default=24.0, min=1.0, max=240.0, step=0.001,
                               tooltip="Advisory playback rate recorded in the container."),
                IO.String.Input("filename_prefix", default="cumuli/splat_4d",
                                tooltip="Output name under ComfyUI's output folder.", optional=True),
                IO.String.Input("mask_filter_root", default="",
                                tooltip="Dataset directory. Enables the lifetime mask-consistency "
                                        "filter, which drops silhouette-escaping splats. Slow but "
                                        "worth it; empty disables it.", optional=True),
                IO.Float.Input("sh_clamp", default=3.0, min=0.0, max=10.0, step=0.1,
                               tooltip="Attenuate higher SH bands above this bare DC. 1.5 is the "
                                       "upstream default and is too aggressive for explicit-SH "
                                       "checkpoints.", advanced=True),
                IO.Boolean.Input("filter_corrupted", default=False,
                                 tooltip="Upstream's bad-colour filters. Calibrated for the legacy "
                                         "SVQ path; they delete healthy splats here.", advanced=True),
                IO.Int.Input("shn_count", default=65536, min=256, max=262144, step=256, advanced=True),
                IO.Float.Input("segment_duration", default=0.1, min=0.0, max=5.0, step=0.01,
                               tooltip="Temporal segment length for streaming. 0 disables segmentation.",
                               advanced=True),
                IO.Boolean.Input("emit_ply", default=True,
                                 tooltip="Also write the 4D interchange PLY beside the .sogst. It comes "
                                         "from the same arrays, and the Preview node reads it."),
            ],
            outputs=[
                IO.String.Output(display_name="sogst_path"),
                IO.String.Output(display_name="ply_path"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        checkpoint,
        duration_seconds=0.0,
        fps=24.0,
        filename_prefix="cumuli/splat_4d",
        mask_filter_root="",
        sh_clamp=3.0,
        filter_corrupted=False,
        shn_count=65536,
        segment_duration=0.1,
        emit_ply=True,
    ) -> IO.NodeOutput:
        node_id = cls.hidden.unique_id
        source = Path((checkpoint or "").strip()).expanduser()
        if not source.is_file():
            raise RuntimeError(f"Cumuli: checkpoint not found: {source}")
        if float(duration_seconds) <= 0:
            raise RuntimeError(
                "Cumuli: duration_seconds must match the clip length the model trained with; "
                "wire it from the Train node's duration output."
            )
        try:
            settings = load_settings()
        except SettingsError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix or "cumuli/splat_4d", folder_paths.get_output_directory()
        )
        name = f"{filename}_{counter:05}_.sogst"
        target = Path(full_output_folder) / name

        options = BakeOptions(
            checkpoint=source,
            output=target,
            duration_seconds=float(duration_seconds),
            fps=float(fps),
            mask_filter_root=Path(mask_filter_root.strip()).expanduser() if mask_filter_root.strip() else None,
            sh_clamp=float(sh_clamp),
            filter_corrupted=bool(filter_corrupted),
            shn_count=int(shn_count),
            segment_duration=float(segment_duration),
            emit_ply=target.with_suffix(".ply") if emit_ply else None,
        )
        progress = ProgressBar(_PROGRESS_STEPS, node_id=node_id)

        def on_progress(state) -> None:
            progress.update_absolute(int(state.fraction * _PROGRESS_STEPS), _PROGRESS_STEPS)
            _send_text(node_id, f"{state.fraction * 100:.0f}% {state.message}")

        try:
            written = bake(settings, options, on_progress=on_progress, should_cancel=_cancelled)
        except SubprocessCancelled:
            raise InterruptProcessingException() from None
        except (TrainingError, SubprocessError) as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        progress.update_absolute(_PROGRESS_STEPS, _PROGRESS_STEPS)
        megabytes = written.stat().st_size / (1024 * 1024)
        report = "\n".join(
            [
                f"sogst: {written}",
                f"size: {megabytes:.2f} MB",
                f"clip: 0.000 .. {options.duration_seconds:.3f}s @ {options.fps:g} fps "
                f"({megabytes / max(options.duration_seconds, 1e-6):.2f} MB/s)",
                f"mask filter: {options.mask_filter_root or 'disabled'}",
            ]
        )
        _send_text(node_id, f"{megabytes:.1f} MB sogst")
        results = [{"filename": name, "subfolder": subfolder, "type": "output"}]
        ply = str(options.emit_ply) if options.emit_ply and options.emit_ply.is_file() else ""
        if ply:
            report += f"\ninterchange ply: {ply}"
        return IO.NodeOutput(str(written), ply, report, ui={"3d": results})


class CumuliPreviewSogst(IO.ComfyNode):
    """Evaluate a baked 4D asset at one instant, as ComfyUI's native SPLAT.

    There is no `.sogst` viewer in ComfyUI, and the format's own player is a
    browser build. Rather than ship a viewer, this evaluates the clip at a
    chosen time and hands the result to ComfyUI's own gaussian-splat stack: wire
    `splat` into the stock **Render Splat** node for a still, or set its
    `frames` above 1 for an orbit. Scrubbing `time_seconds` gives 4D playback.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CumuliPreviewSogst",
            display_name="Cumuli Preview SOGST",
            category=CATEGORY,
            description=(
                "Reads a baked .sogst (or its 4D interchange PLY) and evaluates every gaussian at one "
                "clip time: centre moved along its velocity, opacity scaled by its temporal window. "
                "Outputs ComfyUI's SPLAT type, so the stock Render Splat node draws it."
            ),
            is_experimental=True,
            inputs=[
                IO.String.Input("path", default="",
                                tooltip="A .sogst archive or a 4D interchange .ply. The Bake node's "
                                        "ply_path output is the direct wire."),
                IO.Float.Input("time_seconds", default=0.0, min=0.0, max=600.0, step=0.001,
                               tooltip="Clip time to evaluate, on the asset's own clock."),
                IO.Int.Input("frame_index", default=-1, min=-1, max=100000, step=1,
                             tooltip="Evaluate this frame instead, using the asset's fps. -1 uses "
                                     "time_seconds."),
                IO.Float.Input("alpha_threshold", default=0.004, min=0.0, max=1.0, step=0.001,
                               tooltip="Drop gaussians whose temporal window has faded below this. Most "
                                       "of a clip's splats are inactive at any instant; culling them is "
                                       "what makes the preview fast.", advanced=True),
            ],
            outputs=[
                IO.Splat.Output(display_name="splat"),
                IO.Float.Output(display_name="time_seconds"),
                IO.String.Output(display_name="report"),
            ],
            hidden=[IO.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, path, time_seconds=0.0, frame_index=-1, alpha_threshold=0.004) -> IO.NodeOutput:
        node_id = cls.hidden.unique_id
        source = Path((path or "").strip()).expanduser()
        if not source.is_file():
            raise RuntimeError(f"Cumuli: not found: {source}")

        if source.suffix.lower() == ".sogst":
            try:
                source = unpack_sogst_to_ply(load_settings(), source)
            except (SettingsError, TrainingError, SubprocessError) as exc:
                raise RuntimeError(f"Cumuli: could not unpack the .sogst archive: {exc}") from None

        try:
            asset = load_interchange_ply(source)
            instant = frame_time(asset, int(frame_index)) if int(frame_index) >= 0 else float(time_seconds)
            splat, kept = to_splat(asset, instant, alpha_threshold=float(alpha_threshold))
        except SogstError as exc:
            raise RuntimeError(f"Cumuli: {exc}") from None

        report = "\n".join(
            [
                f"asset: {asset.source}",
                asset.summary(),
                f"t = {instant:.4f}s  (frame {int(round((instant - asset.time_min) * asset.fps))} "
                f"of {frame_count(asset)})",
                f"active splats: {kept:,} of {asset.count:,} ({100.0 * kept / max(asset.count, 1):.1f}%)",
                "Wire 'splat' into Render Splat; set its frames above 1 for an orbit.",
            ]
        )
        _send_text(node_id, f"{kept:,} splats at t={instant:.2f}s")
        return IO.NodeOutput(splat, instant, report)


def _dataset_timeline(dataset: Path) -> tuple[float, float, int]:
    """Clip duration, frame rate and timestamp count, read from the dataset."""

    path = dataset / "transforms_train.json"
    if not path.is_file():
        raise RuntimeError(f"Cumuli: {path} is missing; this is not a 4DGS dataset directory.")
    try:
        frames = json.loads(path.read_text()).get("frames", [])
    except ValueError as exc:
        raise RuntimeError(f"Cumuli: {path} is not valid JSON: {exc}") from None
    times = sorted({float(frame["time"]) for frame in frames if "time" in frame})
    if len(times) < 2:
        raise RuntimeError(f"Cumuli: {path} carries fewer than two distinct timestamps.")
    duration = times[-1]
    step = times[1] - times[0]
    return duration, (1.0 / step if step > 0 else 24.0), len(times)


class CumuliExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            CumuliGenerateRing,
            CumuliLoadRing,
            CumuliRingContactSheet,
            CumuliSelectView,
            CumuliStageRing,
            CumuliSolveRig,
            CumuliStageCapture,
            CumuliLoadFlipbook,
            CumuliRingMasks,
            CumuliBuildDataset,
            CumuliLoadDataset,
            CumuliTrain4DGS,
            CumuliBakeSogst,
            CumuliPreviewSogst,
        ]


async def comfy_entrypoint() -> CumuliExtension:
    return CumuliExtension()
