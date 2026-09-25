# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)

"""Unit tests for the rig-gap-fill plumbing: RigSpec loading, the
plan_ring_gaps.py / build_hybrid_dataset.py argv builders, and run-name
uniqueness. Nothing here runs a real subprocess or touches the GPU --
that's exactly the property test_bridge.py's own docstring asks of tests
that must be right before a multi-hour job starts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cumuli_bridge import riggap  # noqa: E402
from cumuli_bridge.settings import BridgeSettings  # noqa: E402


def bridge_settings(root: Path) -> BridgeSettings:
    return BridgeSettings(
        fdanyone_root=root,
        conda_env="",
        conda_exe="",
        python_exe=sys.executable,
        data_dir=root / "data",
        model_dir=root / "models",
        gvhmr_root=root / "third_party" / "GVHMR",
        cumuli_root=root / "cumuli",
        device="cuda:0",
        min_free_vram_gb=30.0,
    )


RING_SPEC = {
    "name": "t", "layout": "rings", "target": [0.0, 0.0, 0.0],
    "rings": [{"count": 8, "radius": 3.0, "height": 1.0}],
    "resolution": [640, 480],
    "intrinsics": {"lens_mm": 35, "sensor_width_mm": 36},
    "eval": {"count": 0},
}


# --------------------------------------------------------------------- RigSpec
def test_load_rig_spec_from_path(tmp_path):
    path = tmp_path / "ring16.json"
    path.write_text(json.dumps(RING_SPEC))
    rig = riggap.load_rig_spec(spec_path=path)
    assert rig.spec["layout"] == "rings"
    assert rig.source_path == path.resolve()
    assert rig.name == "t"


def test_load_rig_spec_from_inline_json():
    rig = riggap.load_rig_spec(spec_json=json.dumps(RING_SPEC))
    assert rig.spec["layout"] == "rings"
    assert rig.source_path is None


def test_load_rig_spec_rejects_both():
    with pytest.raises(riggap.RigGapError, match="not both"):
        riggap.load_rig_spec(spec_path="x.json", spec_json="{}")


def test_load_rig_spec_rejects_neither():
    with pytest.raises(riggap.RigGapError, match="Give either"):
        riggap.load_rig_spec()


def test_load_rig_spec_rejects_missing_file(tmp_path):
    with pytest.raises(riggap.RigGapError, match="not found"):
        riggap.load_rig_spec(spec_path=tmp_path / "missing.json")


def test_load_rig_spec_rejects_malformed_json():
    with pytest.raises(riggap.RigGapError, match="not valid JSON"):
        riggap.load_rig_spec(spec_json="{not json")


def test_materialize_rig_spec_uses_the_source_file_directly(tmp_path):
    path = tmp_path / "ring16.json"
    path.write_text(json.dumps(RING_SPEC))
    rig = riggap.load_rig_spec(spec_path=path)
    materialized = riggap.materialize_rig_spec(rig, tmp_path / "unused.json")
    assert materialized == path.resolve()
    assert not (tmp_path / "unused.json").exists()


def test_materialize_rig_spec_writes_inline_specs_out(tmp_path):
    rig = riggap.load_rig_spec(spec_json=json.dumps(RING_SPEC))
    dest = tmp_path / "written.json"
    materialized = riggap.materialize_rig_spec(rig, dest)
    assert materialized == dest
    assert json.loads(dest.read_text())["layout"] == "rings"


# --------------------------------------------------------------- RunPlanEntry
def test_run_plan_entry_from_dict_round_trips():
    d = {"anchor_camera": "07", "views_per_layer": 8, "layer_pitches": [12.5],
         "start_yaw": -30.0, "yaw_span": 60.0, "target_azimuths_deg": [-60.0, -30.0, 0.0]}
    entry = riggap.RunPlanEntry.from_dict(d)
    assert entry.anchor_camera == "07"
    assert entry.layer_pitches == (12.5,)
    assert entry.target_azimuths_deg == (-60.0, -30.0, 0.0)


def test_run_name_is_distinct_for_different_gaps_on_the_same_anchor():
    """Two RunPlanEntry objects anchored on the same real camera but
    covering different gaps must not collide -- CumuliGenerateRing caches
    strictly by run_name."""
    a = riggap.RunPlanEntry("07", 8, (12.5,), -30.0, 60.0, (-60.0, -30.0, 0.0))
    b = riggap.RunPlanEntry("07", 8, (12.5,), 90.0, 60.0, (60.0, 90.0, 120.0))
    assert a.run_name("grandprix") != b.run_name("grandprix")


def test_run_name_is_a_safe_identifier():
    entry = riggap.RunPlanEntry("07", 8, (12.5,), -30.5, 60.0, (0.0,))
    name = entry.run_name("grandprix")
    assert " " not in name
    assert name.startswith("grandprix_07_")


# ----------------------------------------------------------------- CoveragePlan
def test_coverage_plan_summary_reports_no_gaps():
    rig = riggap.load_rig_spec(spec_json=json.dumps(RING_SPEC))
    plan = riggap.CoveragePlan(runs=(), rig_spec=rig, real_transforms=Path("t.json"),
                               front_azimuth_deg=0.0, min_separation_deg=20.0)
    assert "no gaps" in plan.summary()


def test_coverage_plan_summary_counts_runs_and_views():
    rig = riggap.load_rig_spec(spec_json=json.dumps(RING_SPEC))
    runs = (
        riggap.RunPlanEntry("07", 8, (12.5,), -30.0, 60.0, (-60.0, -30.0, 0.0)),
        riggap.RunPlanEntry("11", 4, (12.5,), 90.0, 30.0, (90.0, 120.0)),
    )
    plan = riggap.CoveragePlan(runs=runs, rig_spec=rig, real_transforms=Path("t.json"),
                               front_azimuth_deg=0.0, min_separation_deg=20.0)
    summary = plan.summary()
    assert "2 run(s)" in summary
    assert "12 target view(s)" in summary
    assert "07" in summary and "11" in summary


# ------------------------------------------------------------------- plan argv
def test_build_plan_argv_is_the_known_good_command_line(tmp_path):
    settings = bridge_settings(tmp_path)
    argv = riggap.build_plan_argv(
        settings, tmp_path / "plan_ring_gaps.py", tmp_path / "real.json",
        tmp_path / "spec.json", front_azimuth_deg=15.0, min_separation_deg=20.0,
        out_json=tmp_path / "plan.json")
    assert argv == [
        sys.executable, str(tmp_path / "plan_ring_gaps.py"),
        "--real_transforms", str(tmp_path / "real.json"),
        "--rig_spec", str(tmp_path / "spec.json"),
        "--front_azimuth_deg", "15.0",
        "--min_separation_deg", "20.0",
        "--out_json", str(tmp_path / "plan.json"),
    ]


def test_plan_coverage_rejects_a_missing_real_transforms(tmp_path):
    settings = bridge_settings(tmp_path)
    rig = riggap.load_rig_spec(spec_json=json.dumps(RING_SPEC))
    with pytest.raises(riggap.RigGapError, match="real_transforms not found"):
        riggap.plan_coverage(settings, tmp_path / "missing.json", rig, work_dir=tmp_path / "work")


# ---------------------------------------------------------------- hybrid argv
def test_build_hybrid_argv_covers_every_gen_run(tmp_path):
    settings = bridge_settings(tmp_path)
    options = riggap.HybridDatasetOptions(
        real_transforms=tmp_path / "real.json", real_root=tmp_path / "real",
        masks_root=tmp_path / "masks", undistorted_transforms=tmp_path / "und.json",
        motion_dir=tmp_path / "motion", transform=tmp_path / "T.json",
        out_dir=tmp_path / "out",
        gen_runs=(("gap_07", tmp_path / "gen_a"), ("gap_11", tmp_path / "gen_b")),
        gen_weight=0.5, holdout="cam04", min_separation_deg=20.0, color_match=False,
    )
    argv = riggap.build_hybrid_argv(settings, tmp_path / "build_hybrid_dataset.py", options)
    assert argv.count("--gen_run") == 2
    i = argv.index("--gen_run")
    assert argv[i:i + 3] == ["--gen_run", "gap_07", str(tmp_path / "gen_a")]
    j = argv.index("--gen_run", i + 1)
    assert argv[j:j + 3] == ["--gen_run", "gap_11", str(tmp_path / "gen_b")]
    assert "--holdout" in argv and "cam04" in argv
    assert "--color_match" not in argv


def test_build_hybrid_argv_omits_holdout_when_unset(tmp_path):
    settings = bridge_settings(tmp_path)
    options = riggap.HybridDatasetOptions(
        real_transforms=tmp_path / "real.json", real_root=tmp_path / "real",
        masks_root=tmp_path / "masks", undistorted_transforms=tmp_path / "und.json",
        motion_dir=tmp_path / "motion", transform=tmp_path / "T.json",
        out_dir=tmp_path / "out", gen_runs=(("gap_07", tmp_path / "gen_a"),),
        holdout="",
    )
    argv = riggap.build_hybrid_argv(settings, tmp_path / "build_hybrid_dataset.py", options)
    assert "--holdout" not in argv


def test_build_fdanyone_to_omg4_argv_is_the_known_good_command_line(tmp_path):
    settings = bridge_settings(tmp_path)
    argv = riggap.build_fdanyone_to_omg4_argv(
        settings, tmp_path / "fdanyone_to_omg4.py", tmp_path / "result",
        tmp_path / "motion", tmp_path / "smplx.npz", tmp_path / "gen_out")
    assert argv == [
        sys.executable, str(tmp_path / "fdanyone_to_omg4.py"),
        "--result_dir", str(tmp_path / "result"),
        "--motion_dir", str(tmp_path / "motion"),
        "--smplx_model", str(tmp_path / "smplx.npz"),
        "--out_dir", str(tmp_path / "gen_out"),
    ]


def test_build_hybrid_dataset_rejects_zero_gen_runs(tmp_path):
    settings = bridge_settings(tmp_path)
    options = riggap.HybridDatasetOptions(
        real_transforms=tmp_path / "real.json", real_root=tmp_path / "real",
        masks_root=tmp_path / "masks", undistorted_transforms=tmp_path / "und.json",
        motion_dir=tmp_path / "motion", transform=tmp_path / "T.json",
        out_dir=tmp_path / "out", gen_runs=(),
    )
    with pytest.raises(riggap.RigGapError, match="at least one"):
        riggap.build_hybrid_dataset(settings, options)
