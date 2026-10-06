# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Unit tests for the trained-model read model (``CUMULI_MODEL``).

The on-disk format is cumuli's window plan plus each window's trainer config,
so these tests build both the way the trainer and plan_temporal_windows.py
write them, and pin the numbers against cumuli's own documented example.

    ~/miniconda3/envs/comfyenv/bin/python -m pytest tests/ -q -k model
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cumuli_bridge import runner  # noqa: E402
from cumuli_bridge.model import (  # noqa: E402
    PLAN_NAME,
    ModelError,
    TrainedModel,
    write_plan,
)
from cumuli_bridge.settings import BridgeSettings  # noqa: E402
from cumuli_bridge.windowing import even_windows  # noqa: E402

# The shape the trainer template renders to: OptimizationParams is nested
# under ModelParams by indentation, and the paths are absolute.
CONFIG = """\
gaussian_dim: 4
time_duration: [0.0, {duration:.6f}]
num_pts: 100000
rot_4d: True

ModelParams:
  sh_degree: {sh_degree}
  source_path: "{source}"
  model_path: "{model}"
  white_background: False

  OptimizationParams:
    iterations: {iterations}
    lambda_dssim: 0.2
"""


def make_run(out_dir: Path, *, frames: int = 31, fps: float = 24.0, iterations: int = 100,
             sh_degree: int = 3, dataset: Path | None = None, checkpoint: bool = True) -> Path:
    """A finished run as the trainer leaves it: config, dataset, checkpoint."""

    dataset = dataset or out_dir / "dataset_4dgs"
    dataset.mkdir(parents=True, exist_ok=True)
    times = [i / fps for i in range(frames)]
    (dataset / "transforms_train.json").write_text(json.dumps({"frames": [{"time": t} for t in times]}))
    model_dir = out_dir / "train4d_output"
    model_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        (model_dir / f"chkpnt{iterations}.pth").write_bytes(b"")
    (out_dir / "gs4d_config.yaml").write_text(CONFIG.format(
        duration=(frames - 1) / fps, sh_degree=sh_degree, source=dataset, model=model_dir,
        iterations=iterations,
    ))
    return out_dir


def make_windowed(root: Path, total: int = 121, max_frames: int = 31, fps: float = 24.0) -> Path:
    windows = even_windows(total, max_frames)
    entries = []
    for w in windows:
        out_dir = make_run(root / "windows" / f"win_{w.index:02d}", frames=w.frame_count, fps=fps)
        entries.append((w.index, w.frame_start, w.frame_count, out_dir))
    return write_plan(root, run=root / "dataset_4dgs", fps=fps, windows=entries)


def test_written_plan_matches_cumulis_documented_offsets_and_seams(tmp_path):
    # merge_sogst_segments.py's own usage example: 121 frames at 24 fps cut
    # 31/30/30/30 starts its windows at 0, 1.291667, 2.541667, 3.791667.
    plan = json.loads(make_windowed(tmp_path).read_text())
    assert [w["offset_seconds"] for w in plan["windows"]] == [0.0, 1.2916667, 2.5416667, 3.7916667]
    assert [w["frame_count"] for w in plan["windows"]] == [31, 30, 30, 30]
    # Midway between the last frame of one window and the first of the next.
    assert plan["seams_seconds"] == [1.2708333, 2.5208333, 3.7708333]
    assert plan["cut"] == "uniform"


def test_windowed_run_loads_from_its_directory(tmp_path):
    make_windowed(tmp_path)
    model = TrainedModel.load(tmp_path)
    assert model.fps == 24.0
    assert [w.index for w in model.windows] == [0, 1, 2, 3]
    assert model.windows[0].duration_seconds == pytest.approx(30 / 24)
    assert model.windows[1].duration_seconds == pytest.approx(29 / 24)
    assert model.duration_seconds == pytest.approx(5.0)
    assert model.windows[2].checkpoint == tmp_path / "windows" / "win_02" / "train4d_output" / "chkpnt100.pth"
    assert model.windows[2].dataset_dir == tmp_path / "windows" / "win_02" / "dataset_4dgs"


def test_a_plan_from_cumulis_planner_loads_unchanged(tmp_path):
    # plan_temporal_windows.py writes fields this reader ignores, and an
    # out_dir relative to the plan is resolved against it.
    make_run(tmp_path / "uniform" / "win_0", frames=31)
    make_run(tmp_path / "uniform" / "win_1", frames=30)
    plan = {
        "run": "/runs/ring12_5s", "fps": 24.0, "cut": "uniform", "signal": "none",
        "coefficients": {"quality_intercept": 0.007}, "predicted": {"stitched_lpips": 0.0078},
        "seam_motion": [1.0],
        "windows": [
            {"index": 1, "frame_start": 131, "frame_count": 30, "local_frame_start": 31,
             "offset_seconds": 1.2916667, "out_dir": "uniform/win_1"},
            {"index": 0, "frame_start": 100, "frame_count": 31, "local_frame_start": 0,
             "offset_seconds": 0.0, "out_dir": str(tmp_path / "uniform" / "win_0")},
        ],
        "seams_seconds": [1.2708333],
    }
    (tmp_path / "window_plan.json").write_text(json.dumps(plan))
    model = TrainedModel.load(tmp_path / "window_plan.json")
    assert [w.index for w in model.windows] == [0, 1]
    assert model.windows[1].out_dir == tmp_path / "uniform" / "win_1"
    assert model.windows[1].offset_seconds == pytest.approx(1.2916667)


def test_a_plan_whose_windows_were_never_placed_is_refused(tmp_path):
    (tmp_path / PLAN_NAME).write_text(json.dumps({
        "fps": 24.0, "windows": [{"index": 0, "offset_seconds": 0.0, "out_dir": None}],
    }))
    with pytest.raises(ModelError, match="carry no out_dir"):
        TrainedModel.load(tmp_path)


def test_a_single_run_reads_its_frame_rate_from_the_dataset(tmp_path):
    make_run(tmp_path, frames=48, fps=30.0, sh_degree=2)
    model = TrainedModel.load(tmp_path)
    assert model.fps == pytest.approx(30.0)
    assert len(model.windows) == 1
    assert model.windows[0].duration_seconds == pytest.approx(47 / 30)
    assert model.windows[0].sh_degree == 2


def test_a_bare_checkpoint_loads_its_run_and_keeps_that_checkpoint(tmp_path):
    make_run(tmp_path, iterations=30000)
    earlier = tmp_path / "train4d_output" / "chkpnt7000.pth"
    earlier.write_bytes(b"")
    model = TrainedModel.load(earlier)
    assert model.windows[0].checkpoint == earlier


def test_a_windows_checkpoint_is_refused_on_its_own(tmp_path):
    # The failure this type exists for: one window baked as if it were the
    # whole clip, silently writing 1.25 s of a 5 s take.
    make_windowed(tmp_path)
    checkpoint = tmp_path / "windows" / "win_00" / "train4d_output" / "chkpnt100.pth"
    with pytest.raises(ModelError, match="one window of the windowed run"):
        TrainedModel.load(checkpoint)


def test_a_moved_run_falls_back_to_the_standard_names(tmp_path):
    original = make_run(tmp_path / "before")
    moved = original.rename(tmp_path / "after")
    model = TrainedModel.load(moved)
    assert model.windows[0].checkpoint == moved / "train4d_output" / "chkpnt100.pth"
    assert model.windows[0].dataset_dir == moved / "dataset_4dgs"


def test_an_unfinished_run_says_training_did_not_finish(tmp_path):
    make_run(tmp_path, checkpoint=False)
    with pytest.raises(ModelError, match="training did not finish"):
        TrainedModel.load(tmp_path)


def test_a_directory_that_is_not_a_run_is_refused(tmp_path):
    with pytest.raises(ModelError, match="not a trained run"):
        TrainedModel.load(tmp_path)


def test_window_selection(tmp_path):
    make_windowed(tmp_path)
    model = TrainedModel.load(tmp_path)
    assert model.select("") is model
    assert [w.index for w in model.select("1-2").windows] == [1, 2]
    assert [w.index for w in model.select("3").windows] == [3]
    assert model.select("1-2").duration_seconds == pytest.approx(59 / 24)
    with pytest.raises(ModelError, match="not contiguous"):
        model.select("0,2")
    with pytest.raises(ModelError, match="no window 7"):
        model.select("7")
    with pytest.raises(ModelError, match="cannot read"):
        model.select("x")


def test_discover_models_finds_windowed_and_single_runs(tmp_path):
    make_windowed(tmp_path / "work" / "windowed_run")
    make_run(tmp_path / "work" / "single_run")
    (tmp_path / "work" / "not_a_run").mkdir()
    scoped = BridgeSettings(
        fdanyone_root=tmp_path, conda_env="", conda_exe="", python_exe="", data_dir=tmp_path,
        model_dir=tmp_path, device="cpu", min_free_vram_gb=0.0,
        work_root=str(tmp_path / "work"),
    )
    found = runner.discover_models(scoped)
    assert str(tmp_path / "work" / "windowed_run") in found
    assert str(tmp_path / "work" / "single_run") in found
    assert str(tmp_path / "work" / "not_a_run") not in found
