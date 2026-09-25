# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Rig-gap-fill: drive a real rig's own coverage gaps with 4DAnyone.

A real rig covers only part of the azimuth circle. This module computes
exactly which azimuth arcs a target camera_rig_spec.py rig-spec needs that
the real rig does not already provide (``plan_ring_gaps.py``), and merges
one or more 4DAnyone generation runs filling those gaps with the real
cameras into one training dataset (``build_hybrid_dataset.py``). Both are
cumuli scripts, driven in place by the same ``settings.launcher()`` /
``settings.pipeline_script()`` convention ``sfm.py`` and ``train.py`` use --
this module owns no pipeline logic of its own, only the subprocess plumbing
and the typed objects the nodes pass around.

conda env: whatever ``settings.launcher()`` resolves to (ComfyUI's own
interpreter by default). plan_ring_gaps.py needs only numpy, already
ubiquitous; build_hybrid_dataset.py additionally needs PIL and plyfile.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .process import run_streaming
from .settings import SettingsError

LOGGER = logging.getLogger("comfyui-cumuli")


class RigGapError(RuntimeError):
    """Raised for a malformed rig-spec, a failed plan, or a failed merge."""


# ---------------------------------------------------------------------------
# 1. RigSpec: a target camera_rig_spec.py rig, defined or loaded
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RigSpec:
    """A camera_rig_spec.py rig-spec dict, plus where it came from.

    Deliberately thin: this module does not import camera_rig_spec.py itself
    (that would need cumuli's scripts/ on sys.path from inside ComfyUI's own
    interpreter, an extra coupling this pack avoids elsewhere too -- see
    sfm.py's own note on driving cumuli's scripts in place rather than
    importing them). Real validation happens where it already lives: inside
    plan_ring_gaps.py's own call to camera_rig_spec.validate_spec, the first
    time this spec is actually used.
    """

    spec: dict
    source_path: Path | None = None

    @property
    def name(self) -> str:
        return str(self.spec.get("name") or (self.source_path.stem if self.source_path else "rig"))

    def to_dict(self) -> dict:
        return {"name": self.name, "source_path": str(self.source_path) if self.source_path else None,
                "layout": self.spec.get("layout"), "spec": self.spec}


def load_rig_spec(spec_path: str | Path | None = None, spec_json: str | None = None) -> RigSpec:
    """Load a rig-spec from a file path, or parse one given inline.

    Exactly one of the two must be given -- a node offering both as optional
    string widgets needs to decide which one the person meant, and silently
    preferring one over a mistakenly-also-filled-in other invites a spec
    change that does nothing.
    """

    path_text = (str(spec_path).strip() if spec_path else "")
    json_text = (spec_json or "").strip()
    if path_text and json_text:
        raise RigGapError("Give spec_path or spec_json, not both -- it's ambiguous which one you meant.")
    if not path_text and not json_text:
        raise RigGapError("Give either spec_path (e.g. configs/rigs/ring16.json) or spec_json.")

    if path_text:
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise RigGapError(f"spec_path not found: {path}")
        try:
            spec = json.loads(path.read_text())
        except ValueError as exc:
            raise RigGapError(f"{path} is not valid JSON: {exc}") from None
        return RigSpec(spec=spec, source_path=path.resolve())

    try:
        spec = json.loads(json_text)
    except ValueError as exc:
        raise RigGapError(f"spec_json is not valid JSON: {exc}") from None
    return RigSpec(spec=spec, source_path=None)


def materialize_rig_spec(rig_spec: RigSpec, dest: Path) -> Path:
    """A real file plan_ring_gaps.py's --rig_spec can read.

    Returns rig_spec.source_path directly when it has one (no copy -- stays
    the file a person can re-edit and rerun against); otherwise writes
    rig_spec.spec out to `dest`, for a spec that was only ever inline JSON.
    """

    if rig_spec.source_path is not None:
        return rig_spec.source_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rig_spec.spec, indent=1))
    return dest


# ---------------------------------------------------------------------------
# 2. plan_ring_gaps.py
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunPlanEntry:
    """One planned 4DAnyone generation run, read back from plan_ring_gaps.py.

    Field names and meaning match plan_ring_gaps.RunPlan.to_dict() exactly;
    kept as a separate class rather than importing that one, for the same
    reason RigSpec doesn't import camera_rig_spec.py.
    """

    anchor_camera: str
    views_per_layer: int
    layer_pitches: tuple[float, ...]
    start_yaw: float
    yaw_span: float
    target_azimuths_deg: tuple[float, ...]

    @classmethod
    def from_dict(cls, d: dict) -> RunPlanEntry:
        return cls(
            anchor_camera=str(d["anchor_camera"]),
            views_per_layer=int(d["views_per_layer"]),
            layer_pitches=tuple(float(p) for p in d["layer_pitches"]),
            start_yaw=float(d["start_yaw"]),
            yaw_span=float(d["yaw_span"]),
            target_azimuths_deg=tuple(float(a) for a in d["target_azimuths_deg"]),
        )

    def run_name(self, capture_name: str) -> str:
        """A stable, distinct run_name per plan entry -- required, since two
        entries anchored on the same real camera but covering different gaps
        must not collide in CumuliGenerateRing's own run_name-keyed cache."""
        azimuth_tag = f"{self.start_yaw:+.0f}".replace("+", "p").replace("-", "m")
        return f"{capture_name}_{self.anchor_camera}_{azimuth_tag}"


@dataclass(frozen=True)
class CoveragePlan:
    runs: tuple[RunPlanEntry, ...]
    rig_spec: RigSpec
    real_transforms: Path
    front_azimuth_deg: float
    min_separation_deg: float

    def summary(self) -> str:
        if not self.runs:
            return "no gaps: the real rig already covers the target spec within min_separation_deg"
        total_views = sum(r.views_per_layer for r in self.runs)
        anchors = ", ".join(sorted({r.anchor_camera for r in self.runs}))
        return (f"{len(self.runs)} run(s), {total_views} target view(s) total, "
                f"anchored on: {anchors}")


def find_plan_script(settings) -> Path:
    try:
        return settings.pipeline_script("plan_ring_gaps.py")
    except SettingsError as exc:
        raise RigGapError(str(exc)) from None


def build_plan_argv(settings, script: Path, real_transforms: Path, rig_spec_path: Path,
                    front_azimuth_deg: float, min_separation_deg: float, out_json: Path) -> list[str]:
    return list(settings.launcher()) + [
        str(script),
        "--real_transforms", str(real_transforms),
        "--rig_spec", str(rig_spec_path),
        "--front_azimuth_deg", str(front_azimuth_deg),
        "--min_separation_deg", str(min_separation_deg),
        "--out_json", str(out_json),
    ]


def plan_coverage(settings, real_transforms: str | Path, rig_spec: RigSpec, *,
                  front_azimuth_deg: float = 0.0, min_separation_deg: float = 20.0,
                  work_dir: Path, on_line: Callable[[str], None] | None = None,
                  should_cancel: Callable[[], bool] | None = None) -> CoveragePlan:
    """Run plan_ring_gaps.py and read back its plan."""

    from .runner import build_env

    real_transforms = Path(real_transforms).expanduser()
    if not real_transforms.is_file():
        raise RigGapError(f"real_transforms not found: {real_transforms}")

    script = find_plan_script(settings)
    work_dir.mkdir(parents=True, exist_ok=True)
    rig_spec_path = materialize_rig_spec(rig_spec, work_dir / "rig_spec.json")
    out_json = work_dir / "coverage_plan.json"

    argv = build_plan_argv(settings, script, real_transforms, rig_spec_path,
                           front_azimuth_deg, min_separation_deg, out_json)
    try:
        run_streaming(argv, cwd=script.parent, env=build_env(settings),
                      on_line=on_line, should_cancel=should_cancel)
    except Exception as exc:  # noqa: BLE001 - surfaced as RigGapError below
        raise RigGapError(f"plan_ring_gaps.py failed: {exc}") from exc

    try:
        payload = json.loads(out_json.read_text())
    except (OSError, ValueError) as exc:
        raise RigGapError(f"plan_ring_gaps.py did not produce a readable plan: {exc}") from None

    runs = tuple(RunPlanEntry.from_dict(d) for d in payload)
    return CoveragePlan(runs=runs, rig_spec=rig_spec, real_transforms=real_transforms,
                        front_azimuth_deg=front_azimuth_deg, min_separation_deg=min_separation_deg)


# ---------------------------------------------------------------------------
# 3. build_hybrid_dataset.py
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HybridDatasetOptions:
    real_transforms: Path
    real_root: Path
    masks_root: Path
    undistorted_transforms: Path
    motion_dir: Path
    transform: Path
    out_dir: Path
    gen_runs: tuple[tuple[str, Path], ...]  # (name, dataset_dir), one per RunPlanEntry
    gen_weight: float = 0.5
    holdout: str = ""
    min_separation_deg: float = 20.0
    color_match: bool = False


def find_hybrid_script(settings) -> Path:
    try:
        return settings.pipeline_script("build_hybrid_dataset.py")
    except SettingsError as exc:
        raise RigGapError(str(exc)) from None


def build_hybrid_argv(settings, script: Path, options: HybridDatasetOptions) -> list[str]:
    argv = list(settings.launcher()) + [
        str(script),
        "--real_transforms", str(options.real_transforms),
        "--real_root", str(options.real_root),
        "--masks_root", str(options.masks_root),
        "--undistorted_transforms", str(options.undistorted_transforms),
        "--motion_dir", str(options.motion_dir),
        "--transform", str(options.transform),
        "--out_dir", str(options.out_dir),
        "--gen_weight", str(options.gen_weight),
        "--min_separation_deg", str(options.min_separation_deg),
    ]
    if options.holdout:
        argv += ["--holdout", options.holdout]
    if options.color_match:
        argv.append("--color_match")
    for name, dataset_dir in options.gen_runs:
        argv += ["--gen_run", name, str(dataset_dir)]
    return argv


def build_hybrid_dataset(settings, options: HybridDatasetOptions, *,
                         on_line: Callable[[str], None] | None = None,
                         should_cancel: Callable[[], bool] | None = None) -> Path:
    """Run build_hybrid_dataset.py. Returns options.out_dir on success."""

    from .runner import build_env

    if not options.gen_runs:
        raise RigGapError("build_hybrid_dataset needs at least one --gen_run.")
    script = find_hybrid_script(settings)
    argv = build_hybrid_argv(settings, script, options)
    try:
        run_streaming(argv, cwd=script.parent, env=build_env(settings),
                      on_line=on_line, should_cancel=should_cancel)
    except Exception as exc:  # noqa: BLE001 - surfaced as RigGapError below
        raise RigGapError(f"build_hybrid_dataset.py failed: {exc}") from exc
    if not (options.out_dir / "transforms_train.json").is_file():
        raise RigGapError(
            f"build_hybrid_dataset.py exited without writing {options.out_dir}/transforms_train.json"
        )
    return options.out_dir


# ---------------------------------------------------------------------------
# 4. fdanyone_to_omg4.py: one generation run -> one --gen_run dataset
# ---------------------------------------------------------------------------
def find_fdanyone_to_omg4_script(settings) -> Path:
    try:
        return settings.pipeline_script("fdanyone_to_omg4.py")
    except SettingsError as exc:
        raise RigGapError(str(exc)) from None


def build_fdanyone_to_omg4_argv(settings, script: Path, result_dir: Path, motion_dir: Path,
                                smplx_model: Path, out_dir: Path) -> list[str]:
    return list(settings.launcher()) + [
        str(script),
        "--result_dir", str(result_dir),
        "--motion_dir", str(motion_dir),
        "--smplx_model", str(smplx_model),
        "--out_dir", str(out_dir),
    ]


def convert_fdanyone_to_omg4(settings, result_dir: Path, motion_dir: Path, smplx_model: Path,
                             out_dir: Path, *, on_line: Callable[[str], None] | None = None,
                             should_cancel: Callable[[], bool] | None = None) -> Path:
    """Convert one finished 4DAnyone result into a --gen_run dataset for
    build_hybrid_dataset.py. Returns out_dir on success."""

    from .runner import build_env

    script = find_fdanyone_to_omg4_script(settings)
    argv = build_fdanyone_to_omg4_argv(settings, script, result_dir, motion_dir, smplx_model, out_dir)
    try:
        run_streaming(argv, cwd=script.parent, env=build_env(settings),
                      on_line=on_line, should_cancel=should_cancel)
    except Exception as exc:  # noqa: BLE001 - surfaced as RigGapError below
        raise RigGapError(f"fdanyone_to_omg4.py failed: {exc}") from exc
    if not (out_dir / "transforms_train.json").is_file():
        raise RigGapError(
            f"fdanyone_to_omg4.py exited without writing {out_dir}/transforms_train.json"
        )
    return out_dir
