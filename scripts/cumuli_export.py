#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Turn a 4DAnyone result into a 4D Gaussian Splatting training dataset.

The same code path the ComfyUI nodes use, without ComfyUI. Run it with the
ComfyUI environment's interpreter, which already has numpy, Pillow, PyAV and
torch:

    ~/miniconda3/envs/comfyenv/bin/python scripts/cumuli_export.py --help

Masks are required: the builder carves its visual hull from them and bakes them
into the training alpha. Two sources are available here:

  --masks comfyui   load a background-removal model from a ComfyUI checkout
                    (needs --comfyui_root, or COMFYUI_ROOT in the environment)
  --masks existing  reuse mattes already staged under frame_*/fmasks_clean/

Examples
--------
    # Full dataset from a finished ring, full resolution
    scripts/cumuli_export.py \\
        --result_dir ~/Dev/github/4DAnyone/data/fdanyone/cam01x24b \\
        --out_dir    /media/IronWolf/Datasets/cam01x24b_4dgs \\
        --masks comfyui --comfyui_root /media/IronWolf/ComfyUI

    # Stage and mask only, so the build can be inspected first
    scripts/cumuli_export.py --result_dir ... --out_dir ... --stop_after masks

    # Report the plan and exit
    scripts/cumuli_export.py --result_dir ... --out_dir ... --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cumuli_bridge.dataset4d import DatasetError, DatasetOptions, build_dataset  # noqa: E402
from cumuli_bridge.flipbook import MASKS_SUBDIR, FlipbookError, check_complete, write_flipbook  # noqa: E402
from cumuli_bridge.masks import MaskError, mask_coverage, matte_flipbook  # noqa: E402
from cumuli_bridge.ring import RingError, RingResult  # noqa: E402
from cumuli_bridge.validate import ValidationError, validate_dataset  # noqa: E402

LOGGER = logging.getLogger("cumuli-export")

STOP_STAGES = ("stage", "masks", "dataset")


def parse_views(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    selected: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            low, _, high = part.partition("-")
            selected.extend(range(int(low), int(high) + 1))
        else:
            selected.append(int(part))
    return tuple(dict.fromkeys(selected))


def parse_labels(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(part.strip() for part in text.replace(";", ",").split(",") if part.strip())


def load_bg_model(comfyui_root: str | None, model_name: str):
    """Import ComfyUI far enough to load a background-removal model."""

    root = comfyui_root or os.environ.get("COMFYUI_ROOT")
    if not root:
        raise MaskError(
            "--masks comfyui needs --comfyui_root (or COMFYUI_ROOT) pointing at a ComfyUI checkout, "
            "so the background-removal model can be loaded."
        )
    root_path = Path(root).expanduser().resolve()
    if not (root_path / "comfy" / "bg_removal_model.py").is_file():
        raise MaskError(f"{root_path} does not look like a ComfyUI checkout.")
    sys.path.insert(0, str(root_path))
    from comfy.bg_removal_model import load as load_bg  # noqa: PLC0415

    candidate = root_path / "models" / "background_removal" / model_name
    if not candidate.is_file():
        available = sorted(p.name for p in (root_path / "models" / "background_removal").glob("*"))
        raise MaskError(f"{candidate} not found. Available: {available or 'none'}")
    model = load_bg(str(candidate))
    if model is None:
        raise MaskError(f"{candidate} is not a valid background removal model.")
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--result_dir", required=True, type=Path,
                        help="A published 4DAnyone result directory (the one with cameras.json).")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Run directory. The dataset lands in <out_dir>/dataset_4dgs.")
    parser.add_argument("--staging_dir", type=Path, default=None,
                        help="Staging tree. Default: <out_dir>/flipbook_src.")
    parser.add_argument("--views", default=None, help="Views to export, e.g. '0,6,12' or '0-11'. Default: all.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--max_frames", type=int, default=0, help="Cap staged frames per view. 0 means all.")
    parser.add_argument("--fps", type=float, default=0.0, help="Timeline fps. 0 uses the clip's own rate.")
    parser.add_argument("--downscale", type=int, default=1, help="Integer image downscale (default: 1).")
    parser.add_argument("--masks", choices=["comfyui", "existing"], default="comfyui")
    parser.add_argument("--comfyui_root", default=None, help="ComfyUI checkout, for --masks comfyui.")
    parser.add_argument("--bg_model", default="birefnet.safetensors",
                        help="File in <comfyui_root>/models/background_removal.")
    parser.add_argument("--mask_batch", type=int, default=8)
    parser.add_argument("--test_cameras", default=None, help="Labels held out and scored, e.g. '12'.")
    parser.add_argument("--holdout_cameras", default=None, help="Labels excluded from training but not scored.")
    parser.add_argument("--hull_points", type=int, default=300_000)
    parser.add_argument("--hull_min_views", type=int, default=9)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--stop_after", choices=STOP_STAGES, default="dataset")
    parser.add_argument("--overwrite", action="store_true", help="Redo staging and matting that already exist.")
    parser.add_argument("--no_validate", action="store_true", help="Skip the dataset contract check.")
    parser.add_argument("--dry_run", action="store_true", help="Report the plan and exit.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    try:
        ring = RingResult.load(args.result_dir)
    except RingError as exc:
        LOGGER.error("%s", exc)
        return 1

    out_dir = args.out_dir.expanduser().resolve()
    staging = (args.staging_dir or (out_dir / "flipbook_src")).expanduser().resolve()
    dataset_dir = out_dir / "dataset_4dgs"
    views = parse_views(args.views)

    LOGGER.info("%s", ring.summary())
    LOGGER.info("staging  -> %s", staging)
    LOGGER.info("dataset  -> %s", dataset_dir)
    LOGGER.info("views=%s stride=%d downscale=%d masks=%s stop_after=%s",
                len(views) if views else ring.num_views, args.frame_stride, args.downscale,
                args.masks, args.stop_after)
    if args.dry_run:
        LOGGER.info("dry run; nothing written")
        return 0

    try:
        flipbook = write_flipbook(
            ring,
            staging,
            camera_ids=views,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames or None,
            overwrite=args.overwrite,
            on_progress=lambda done, total: LOGGER.info("staged view %d/%d", done, total),
        )
    except FlipbookError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("staged %d frames x %d views @ %.3f fps",
                flipbook.num_frames, len(flipbook.labels), float(flipbook.fps))
    if args.stop_after == "stage":
        return 0

    try:
        if args.masks == "comfyui":
            model = load_bg_model(args.comfyui_root, args.bg_model)
            written = matte_flipbook(
                flipbook,
                model,
                subdir=MASKS_SUBDIR,
                batch_size=args.mask_batch,
                overwrite=args.overwrite,
                on_progress=lambda done, total: (
                    LOGGER.info("matted %d/%d", done, total) if done % 200 == 0 or done == total else None
                ),
            )
            LOGGER.info("%d mattes written", written)
        check_complete(flipbook, MASKS_SUBDIR)
    except (MaskError, FlipbookError) as exc:
        LOGGER.error("%s", exc)
        return 1
    coverage = mask_coverage(flipbook, MASKS_SUBDIR)
    LOGGER.info("middle-frame foreground coverage: %.2f%%", coverage * 100)
    if coverage < 0.005:
        LOGGER.warning("masks look empty; the visual hull will collapse")
    if args.stop_after == "masks":
        return 0

    options = DatasetOptions(
        out_dir=dataset_dir,
        fps=args.fps or float(flipbook.fps),
        downscale=args.downscale,
        test_cameras=parse_labels(args.test_cameras),
        holdout_cameras=parse_labels(args.holdout_cameras),
        masks_dir=MASKS_SUBDIR,
        hull_points=args.hull_points,
        hull_min_views=args.hull_min_views,
        jobs=args.jobs,
    )
    try:
        summary = build_dataset(
            flipbook,
            options,
            on_progress=lambda fraction, message: LOGGER.debug("%3.0f%% %s", fraction * 100, message),
        )
    except DatasetError as exc:
        LOGGER.error("%s", exc)
        return 1

    if not args.no_validate:
        try:
            LOGGER.info("validation: %s", json.dumps(validate_dataset(summary.out_dir)))
        except ValidationError as exc:
            LOGGER.error("dataset failed its contract check: %s", exc)
            return 1

    LOGGER.info("%s", json.dumps(summary.to_dict(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
