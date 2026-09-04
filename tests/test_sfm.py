# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Unit tests for the real-capture rig solve.

Nothing here runs HLOC, opens a video or touches the GPU: the expensive parts
are the argv, the seed intrinsics and the pre-flight checks, and those are
exactly the parts that must be right *before* a multi-hour solve starts.

    ~/miniconda3/envs/comfyenv/bin/python -m pytest tests/ -q -k sfm
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cumuli_bridge import sfm  # noqa: E402
from cumuli_bridge.settings import BridgeSettings  # noqa: E402


def camera(label: str, width: int = 1920, height: int = 1080, fps: str = "30000/1001", frames: int = 300):
    return sfm.CameraSource(
        label=label,
        path=Path(f"/capture/{label}.mov"),
        width=width,
        height=height,
        num_frames=frames,
        fps=Fraction(fps),
    )


def uniform_rig(count: int = 8) -> tuple[sfm.CameraSource, ...]:
    return tuple(camera(f"{i:04d}") for i in range(2, 2 + count))


def bridge_settings(root: Path) -> BridgeSettings:
    """Settings that stay in this interpreter -- ``python_exe`` set skips conda
    discovery, which is what the solve relies on."""

    return BridgeSettings(
        fdanyone_root=root,
        conda_env="",
        conda_exe="",
        python_exe=sys.executable,
        data_dir=root / "data",
        model_dir=root / "models",
        gvhmr_root=root / "third_party" / "GVHMR",
        device="cuda:0",
        min_free_vram_gb=30.0,
    )


# --------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------
def test_pair_count_is_exhaustive():
    # The upstream matcher pairs every image with every other, with no
    # retrieval step, so this is the whole cost model.
    assert sfm.pair_count(200, 1) == 19_900
    assert sfm.pair_count(200, 4) == 319_600
    assert sfm.pair_count(1, 1) == 0


def test_check_scale_allows_a_big_rig_at_one_timestamp():
    sfm.check_scale(200, 1)


def test_check_scale_rejects_multi_timestamp_on_a_big_rig():
    with pytest.raises(sfm.SfmError) as excinfo:
        sfm.check_scale(200, 4)
    message = str(excinfo.value)
    assert "319,600" in message
    # The error must say what *would* work, not merely that this does not.
    assert "num_timestamps=1" in message


# --------------------------------------------------------------------------
# rig uniformity
# --------------------------------------------------------------------------
def test_check_uniform_accepts_one_format():
    sfm.check_uniform(uniform_rig())


def test_check_uniform_names_the_odd_cameras():
    cameras = uniform_rig(6) + (
        camera("0100", width=1280, height=720, fps="60000/1001", frames=610),
    )
    with pytest.raises(sfm.SfmError) as excinfo:
        sfm.check_uniform(cameras)
    message = str(excinfo.value)
    assert "0100" in message
    assert "1280x720" in message


def test_select_majority_format_keeps_the_body_array():
    cameras = uniform_rig(6) + (
        camera("0100", width=1280, height=720, fps="60000/1001"),
        camera("0101", width=1280, height=720, fps="60000/1001"),
    )
    kept, dropped = sfm.select_majority_format(cameras)
    assert len(kept) == 6
    assert dropped == ("0100", "0101")


def test_select_majority_format_is_a_no_op_on_a_clean_rig():
    cameras = uniform_rig(4)
    kept, dropped = sfm.select_majority_format(cameras)
    assert kept == cameras
    assert dropped == ()


def test_common_frame_count_takes_the_shortest_clip():
    # Hardware sync still leaves ragged trims; the dataset builder needs every
    # camera present in every frame.
    cameras = (camera("0002", frames=302), camera("0003", frames=300), camera("0004", frames=301))
    assert sfm.common_frame_count(cameras) == 300


# --------------------------------------------------------------------------
# seed intrinsics
# --------------------------------------------------------------------------
def test_init_transforms_are_centred_pinhole():
    payload = sfm.build_init_transforms(uniform_rig(2), focal_guess=1.2)
    assert payload["camera_model"] == "PINHOLE"
    frame = payload["frames"][0]
    assert frame["fl_x"] == pytest.approx(1.2 * 1920)
    assert frame["fl_y"] == pytest.approx(1.2 * 1920)
    assert frame["cx"] == pytest.approx(960.0)
    assert frame["cy"] == pytest.approx(540.0)
    assert frame["w"] == 1920 and frame["h"] == 1080


def test_init_transforms_declare_no_distortion():
    # PINHOLE rather than OPENCV is deliberate: there is no distortion estimate
    # to seed, and the dataset builder downstream rejects nonzero distortion.
    frame = sfm.build_init_transforms(uniform_rig(1), focal_guess=1.2)["frames"][0]
    for key in ("k1", "k2", "k3", "k4", "p1", "p2"):
        assert key not in frame


def test_init_transforms_label_matches_upstream_lookup():
    # Upstream matches a camera by camera_label equality first; keep the label
    # exactly the video stem so that lookup cannot fall through to the fuzzy
    # file_path branch.
    payload = sfm.build_init_transforms((camera("0042"),), focal_guess=1.2)
    assert payload["frames"][0]["camera_label"] == "0042"


def test_write_init_transforms_round_trips(tmp_path: Path):
    path = sfm.write_init_transforms(uniform_rig(3), 1.2, tmp_path / "init.json")
    payload = json.loads(path.read_text())
    assert len(payload["frames"]) == 3


# --------------------------------------------------------------------------
# options validation
# --------------------------------------------------------------------------
def options(tmp_path: Path, **overrides) -> sfm.SolveOptions:
    base = {
        "videos_dir": tmp_path / "movies",
        "outputs_dir": tmp_path / "sfm",
    }
    base.update(overrides)
    return sfm.SolveOptions(**base)


def test_options_reject_an_unknown_feature(tmp_path: Path):
    with pytest.raises(sfm.SfmError):
        options(tmp_path, feature_type="sift").validated()


def test_options_reject_a_nonsense_focal_guess(tmp_path: Path):
    # focal_guess is a multiple of image width, so 1200 is a pixel value that
    # someone typed into the wrong box.
    with pytest.raises(sfm.SfmError):
        options(tmp_path, focal_guess=1200.0).validated()


def test_options_accept_the_defaults(tmp_path: Path):
    assert options(tmp_path).validated().num_timestamps == 1


# --------------------------------------------------------------------------
# argv
# --------------------------------------------------------------------------
def test_solve_argv_targets_the_staged_links_not_the_capture(tmp_path: Path):
    settings = bridge_settings(tmp_path)
    opts = options(tmp_path)
    staged = tmp_path / "cameras_selected"
    argv = sfm.build_solve_argv(settings, tmp_path / "multiframe_sfm.py", opts, tmp_path / "init.json", staged)
    assert argv[0] == sys.executable
    assert str(staged) == argv[argv.index("--videos_dir") + 1]
    # The capture directory must never be handed to a script that globs it.
    assert str(opts.videos_dir) not in argv


def test_solve_argv_omits_refinement_flags_when_off(tmp_path: Path):
    argv = sfm.build_solve_argv(
        bridge_settings(tmp_path),
        tmp_path / "s.py",
        options(tmp_path, refine_intrinsics=False),
        tmp_path / "init.json",
        tmp_path / "links",
    )
    assert "--refine_intrinsics" not in argv
    assert "--refine_principal_point" not in argv


def test_solve_argv_includes_refinement_when_on(tmp_path: Path):
    argv = sfm.build_solve_argv(
        bridge_settings(tmp_path),
        tmp_path / "s.py",
        options(tmp_path, refine_intrinsics=True, refine_principal_point=True),
        tmp_path / "init.json",
        tmp_path / "links",
    )
    assert "--refine_intrinsics" in argv
    assert "--refine_principal_point" in argv


# --------------------------------------------------------------------------
# camera staging
# --------------------------------------------------------------------------
def test_link_cameras_links_only_the_selection(tmp_path: Path):
    capture = tmp_path / "movies"
    capture.mkdir()
    for index in range(4):
        (capture / f"000{index}.mov").write_bytes(b"x")
    cameras = tuple(
        sfm.CameraSource(f"000{i}", capture / f"000{i}.mov", 1920, 1080, 300, Fraction(30))
        for i in (0, 2)
    )
    staged = sfm.link_cameras(cameras, tmp_path / "links")
    assert sorted(p.name for p in staged.iterdir()) == ["0000.mov", "0002.mov"]
    assert all(p.is_symlink() for p in staged.iterdir())
    # The capture itself is untouched.
    assert len(list(capture.iterdir())) == 4


def test_link_cameras_clears_a_previous_selection(tmp_path: Path):
    capture = tmp_path / "movies"
    capture.mkdir()
    for index in range(3):
        (capture / f"000{index}.mov").write_bytes(b"x")
    first = tuple(
        sfm.CameraSource(f"000{i}", capture / f"000{i}.mov", 1920, 1080, 300, Fraction(30))
        for i in (0, 1, 2)
    )
    sfm.link_cameras(first, tmp_path / "links")
    staged = sfm.link_cameras(first[:1], tmp_path / "links")
    assert sorted(p.name for p in staged.iterdir()) == ["0000.mov"]


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def test_discover_rejects_a_directory_with_no_videos(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(sfm.SfmError) as excinfo:
        sfm.discover_cameras(tmp_path / "empty")
    assert "no video files" in str(excinfo.value)


def test_discover_rejects_a_missing_directory(tmp_path: Path):
    with pytest.raises(sfm.SfmError):
        sfm.discover_cameras(tmp_path / "nope")


def test_discover_rejects_a_zero_stride(tmp_path: Path):
    (tmp_path / "movies").mkdir()
    (tmp_path / "movies" / "0002.mov").write_bytes(b"x")
    with pytest.raises(sfm.SfmError):
        sfm.discover_cameras(tmp_path / "movies", stride=0)


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------
def test_fingerprint_changes_with_timestamps(tmp_path: Path):
    cameras = uniform_rig(4)
    one = sfm.solve_fingerprint(options(tmp_path, num_timestamps=1), cameras)
    four = sfm.solve_fingerprint(options(tmp_path, num_timestamps=4), cameras)
    assert one != four


def test_fingerprint_changes_with_the_camera_set(tmp_path: Path):
    opts = options(tmp_path)
    assert sfm.solve_fingerprint(opts, uniform_rig(4)) != sfm.solve_fingerprint(opts, uniform_rig(5))


def test_fingerprint_is_stable_for_identical_inputs(tmp_path: Path):
    opts = options(tmp_path)
    cameras = uniform_rig(4)
    assert sfm.solve_fingerprint(opts, cameras) == sfm.solve_fingerprint(opts, cameras)


# --------------------------------------------------------------------------
# reading a solve back
# --------------------------------------------------------------------------
def write_solve(root: Path, *, labels=("0002", "0003"), inliers=1) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / sfm.TRANSFORMS_NAME).write_text(
        json.dumps(
            {
                "frames": [
                    {"camera_label": label, "fl_x": 2304.0, "fl_y": 2304.0, "cx": 960.0, "cy": 540.0,
                     "w": 1920, "h": 1080, "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
                    for label in labels
                ]
            }
        )
    )
    (root / sfm.REPORT_NAME).write_text(
        json.dumps(
            {
                "reconstruction": {
                    "num_points3D": 619,
                    "mean_track_length": 2.04,
                    "mean_reprojection_error": 0.65,
                    "num_reg_images": len(labels),
                },
                "per_camera_pose_stats": {
                    label: {"num_views": 1, "num_inliers": inliers, "center_spread": 0.0004,
                            "rot_spread_deg": 0.005}
                    for label in labels
                },
            }
        )
    )
    return root


def test_read_solve_parses_the_report(tmp_path: Path):
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    assert solve.num_cameras == 2
    assert solve.num_points == 619
    assert solve.mean_reprojection_error == pytest.approx(0.65)
    assert solve.num_registered_images == 2


def test_read_solve_flags_unsolved_cameras(tmp_path: Path):
    solve = sfm.read_solve(write_solve(tmp_path / "sfm", inliers=0))
    assert len(solve.unsolved()) == 2
    assert "UNSOLVED" in solve.summary()


def test_read_solve_summary_is_clean_when_every_camera_solved(tmp_path: Path):
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    assert "UNSOLVED" not in solve.summary()


def test_summary_notes_that_one_timestamp_cannot_self_check(tmp_path: Path):
    # num_views == 1 means the camera has a single measurement of itself, so a
    # rotation spread of 0.0 is an absence of evidence, not a clean bill.
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    assert "could not be measured" in solve.summary()
    assert solve.unstable() == ()


def test_unstable_flags_cameras_that_disagree_with_themselves(tmp_path: Path):
    root = write_solve(tmp_path / "sfm")
    payload = json.loads((root / sfm.REPORT_NAME).read_text())
    payload["per_camera_pose_stats"]["0002"] = {
        "num_views": 4, "num_inliers": 3, "center_spread": 0.7, "rot_spread_deg": 4.88,
    }
    payload["per_camera_pose_stats"]["0003"] = {
        "num_views": 4, "num_inliers": 4, "center_spread": 0.0005, "rot_spread_deg": 0.007,
    }
    (root / sfm.REPORT_NAME).write_text(json.dumps(payload))
    solve = sfm.read_solve(root)
    unstable = solve.unstable()
    assert [c.label for c in unstable] == ["0002"]
    summary = solve.summary()
    assert "1 of 2 cameras disagree" in summary
    # The warning must point at the actual remedy, not just report a number.
    assert "sync_json" in summary


def test_summary_warns_on_a_short_mean_track(tmp_path: Path):
    root = write_solve(tmp_path / "sfm")  # mean_track_length 2.04
    assert "below 8" in sfm.read_solve(root).summary()


def test_read_solve_rejects_a_missing_transforms(tmp_path: Path):
    (tmp_path / "sfm").mkdir()
    with pytest.raises(sfm.SfmError):
        sfm.read_solve(tmp_path / "sfm")


def test_read_solve_rejects_a_one_camera_rig(tmp_path: Path):
    root = write_solve(tmp_path / "sfm", labels=("0002",))
    with pytest.raises(sfm.SfmError) as excinfo:
        sfm.read_solve(root)
    assert "fewer than two" in str(excinfo.value)


# --------------------------------------------------------------------------
# flipbook bridge
# --------------------------------------------------------------------------
def test_flipbook_transforms_keep_only_what_the_builder_reads(tmp_path: Path):
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    payload = sfm.build_flipbook_transforms(solve)
    assert payload["camera_model"] == "OPENCV"
    frame = payload["frames"][0]
    assert set(frame) == {"camera_label", "transform_matrix", "fl_x", "fl_y", "cx", "cy", "w", "h"}
    # file_path is a solve-side artefact and must not leak downstream.
    assert "file_path" not in frame


def test_flipbook_transforms_reject_distortion(tmp_path: Path):
    # A dropped coefficient would mean training against images that disagree
    # with their own camera model, so this must stop rather than sanitise.
    root = write_solve(tmp_path / "sfm")
    payload = json.loads((root / sfm.TRANSFORMS_NAME).read_text())
    payload["frames"][0]["k1"] = -0.031
    (root / sfm.TRANSFORMS_NAME).write_text(json.dumps(payload))
    solve = sfm.read_solve(root)
    with pytest.raises(sfm.SfmError) as excinfo:
        sfm.build_flipbook_transforms(solve)
    assert "k1" in str(excinfo.value)


def test_flipbook_transforms_tolerate_explicit_zero_distortion(tmp_path: Path):
    root = write_solve(tmp_path / "sfm")
    payload = json.loads((root / sfm.TRANSFORMS_NAME).read_text())
    for frame in payload["frames"]:
        frame["k1"] = 0.0
        frame["p1"] = 0.0
    (root / sfm.TRANSFORMS_NAME).write_text(json.dumps(payload))
    assert len(sfm.build_flipbook_transforms(sfm.read_solve(root))["frames"]) == 2


def test_flipbook_transforms_reject_missing_intrinsics(tmp_path: Path):
    root = write_solve(tmp_path / "sfm")
    payload = json.loads((root / sfm.TRANSFORMS_NAME).read_text())
    del payload["frames"][0]["fl_x"]
    (root / sfm.TRANSFORMS_NAME).write_text(json.dumps(payload))
    with pytest.raises(sfm.SfmError):
        sfm.build_flipbook_transforms(sfm.read_solve(root))


def fake_video_frames(monkeypatch, height: int = 4, width: int = 6, count: int = 10):
    """Replace the decoder so staging can be tested without a real container."""

    import numpy as np

    def frames(path):
        for index in range(count):
            yield np.full((height, width, 3), index, dtype=np.uint8)

    monkeypatch.setattr(sfm, "iter_frames_of", frames)


def test_stage_capture_writes_the_builder_contract(tmp_path: Path, monkeypatch):
    fake_video_frames(monkeypatch, count=10)
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    cameras = (camera("0002", 6, 4, frames=10), camera("0003", 6, 4, frames=10))
    flipbook = sfm.stage_capture(solve, cameras, tmp_path / "flip", num_frames=3)

    assert flipbook.num_frames == 3
    assert flipbook.labels == ("0002", "0003")
    for index in range(3):
        frame_dir = tmp_path / "flip" / f"frame_{index:04d}"
        assert (frame_dir / "transforms.json").is_file()
        for label in ("0002", "0003"):
            assert (frame_dir / "images_flat" / f"{label}.png").is_file()


def test_stage_capture_is_readable_by_load_flipbook(tmp_path: Path, monkeypatch):
    # The whole point of the bridge: what it writes must open as a flipbook.
    from cumuli_bridge.flipbook import load_flipbook

    fake_video_frames(monkeypatch, count=8)
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    cameras = (camera("0002", 6, 4, frames=8), camera("0003", 6, 4, frames=8))
    sfm.stage_capture(solve, cameras, tmp_path / "flip", num_frames=2)

    reopened = load_flipbook(tmp_path / "flip", Fraction(30))
    assert reopened.num_frames == 2
    assert reopened.labels == ("0002", "0003")


def test_stage_capture_stages_only_solved_cameras(tmp_path: Path, monkeypatch):
    # An extra camera in the capture that never registered has no pose; staging
    # it would leave an image with no transforms entry.
    fake_video_frames(monkeypatch, count=6)
    solve = sfm.read_solve(write_solve(tmp_path / "sfm", labels=("0002", "0003")))
    cameras = (camera("0002", 6, 4, frames=6), camera("0003", 6, 4, frames=6), camera("0009", 6, 4, frames=6))
    flipbook = sfm.stage_capture(solve, cameras, tmp_path / "flip", num_frames=2)
    assert "0009" not in flipbook.labels
    assert not (tmp_path / "flip" / "frame_0000" / "images_flat" / "0009.png").exists()


def test_stage_capture_rejects_a_camera_the_solve_names_but_capture_lacks(tmp_path: Path, monkeypatch):
    fake_video_frames(monkeypatch, count=6)
    solve = sfm.read_solve(write_solve(tmp_path / "sfm", labels=("0002", "0003")))
    with pytest.raises(sfm.SfmError):
        sfm.stage_capture(solve, (camera("0002", 6, 4, frames=6),), tmp_path / "flip", num_frames=2)


def test_stage_capture_clamps_to_the_shortest_clip(tmp_path: Path, monkeypatch):
    # Hardware sync still leaves ragged trims. Asking for more frames than the
    # shortest clip can supply stages what exists rather than writing a ragged
    # tree, in the same "up to N" sense as the ring stager's max_frames.
    fake_video_frames(monkeypatch, count=5)
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    cameras = (camera("0002", 6, 4, frames=5), camera("0003", 6, 4, frames=5))
    flipbook = sfm.stage_capture(solve, cameras, tmp_path / "flip", start_frame=4, num_frames=4)
    assert flipbook.num_frames == 1
    assert (tmp_path / "flip" / "frame_0000" / "images_flat" / "0002.png").is_file()
    assert not (tmp_path / "flip" / "frame_0001").exists()


def test_stage_capture_rejects_a_start_past_the_end(tmp_path: Path, monkeypatch):
    fake_video_frames(monkeypatch, count=5)
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    cameras = (camera("0002", 6, 4, frames=5), camera("0003", 6, 4, frames=5))
    with pytest.raises(sfm.SfmError) as excinfo:
        sfm.stage_capture(solve, cameras, tmp_path / "flip", start_frame=99)
    assert "past the end" in str(excinfo.value)


def test_stage_capture_stride_divides_the_fps(tmp_path: Path, monkeypatch):
    fake_video_frames(monkeypatch, count=12)
    solve = sfm.read_solve(write_solve(tmp_path / "sfm"))
    cameras = (camera("0002", 6, 4, frames=12), camera("0003", 6, 4, frames=12))
    flipbook = sfm.stage_capture(solve, cameras, tmp_path / "flip", num_frames=3, frame_stride=2)
    assert flipbook.num_frames == 3
    assert flipbook.fps == Fraction("30000/1001") / 2


# --------------------------------------------------------------------------
# progress
# --------------------------------------------------------------------------
def test_progress_is_monotonic_over_a_real_transcript():
    state = sfm.SfmProgress()
    lines = [
        "Found 47 videos.",
        "Sampling timestamps: [10, 100, 190, 280]",
        "  extracted 0002 (shift +0)",
        "  extracted 0006 (shift +0)",
        "47 cameras, 188 images total.",
        "Extracting features...",
        "  50%|#####     | 94/188 [00:10<00:10]",
        "Matching 17578 pairs...",
        "  50%|#####     | 8789/17578 [01:00<01:00]",
        "Running incremental mapping (intrinsics locked)...",
        "Global bundle adjustment with shared intrinsics refinement...",
        "After refinement: mean reprojection error 0.650px, 619 points",
        "Wrote transforms_multiframe.json and report.json",
    ]
    seen = []
    for line in lines:
        state.update(line)
        seen.append(state.fraction)
    assert seen == sorted(seen)
    assert 0.0 < seen[-1] <= 1.0


def test_progress_records_the_camera_count():
    state = sfm.SfmProgress()
    state.update("Found 47 videos.")
    assert state.expected_cameras == 47


def test_progress_ignores_noise():
    state = sfm.SfmProgress()
    assert state.update("") is False
    assert state.update("some unrelated chatter") is False
    assert state.fraction == 0.0
