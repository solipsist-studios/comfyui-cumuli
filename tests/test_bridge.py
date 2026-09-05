# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Unit tests for the parts that must be right before a 45-minute GPU job starts.

Run with the ComfyUI environment's interpreter::

    ~/miniconda3/envs/comfyenv/bin/python -m pytest tests/ -q

Nothing here launches 4DAnyone, loads a model, or touches the GPU. The
subprocess tests drive a fake ``python -c`` child that replays real tqdm output.
"""

from __future__ import annotations

import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cumuli_bridge import runner  # noqa: E402
from cumuli_bridge.flipbook import camera_label, label_width  # noqa: E402
from cumuli_bridge.process import SubprocessCancelled, SubprocessError, run_streaming  # noqa: E402
from cumuli_bridge.progress import ProgressState  # noqa: E402
from cumuli_bridge.ring import RingCamera  # noqa: E402
from cumuli_bridge.settings import BridgeSettings  # noqa: E402
from cumuli_bridge.train import (  # noqa: E402
    BakeOptions,
    BakeProgress,
    TrainingError,
    TrainOptions,
    TrainProgress,
    build_bake_argv,
    build_train_argv,
    build_train_env,
)
from cumuli_bridge.validate import ValidationError, validate_dataset  # noqa: E402


# --------------------------------------------------------------------------
# settings / argv
# --------------------------------------------------------------------------
@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "4DAnyone"
    (repo / "fdanyone").mkdir(parents=True)
    (repo / "inference.py").write_text("# stub\n")
    (repo / "fdanyone" / "config.py").write_text("class C:\n    num_frames: int = 121\n")
    return repo


@pytest.fixture
def settings(fake_repo: Path) -> BridgeSettings:
    return BridgeSettings(
        fdanyone_root=fake_repo,
        conda_env="4danyone",
        conda_exe="",
        python_exe=sys.executable,  # skips conda discovery
        data_dir=fake_repo / "data",
        model_dir=fake_repo / "models",
        gvhmr_root=fake_repo / "third_party" / "GVHMR",
        device="cuda:0",
        min_free_vram_gb=30.0,
        subprocess_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )


def known_good_request(video: Path) -> runner.RunRequest:
    return runner.build_request(
        video_path=video,
        views_per_layer=24,
        layer_pitches="15",
        start_yaw=0,
        yaw_span=360,
        views_per_group="4",
        enable_rcp=False,
        enable_tcr=True,
        start_time=0.0,
        target_fps="auto",
        seed=42,
        device="cuda:0",
    )


def test_argv_is_the_known_good_command_line(settings, tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    argv = runner.build_argv(settings, known_good_request(video))
    assert argv[0] == sys.executable
    assert argv[1] == str(settings.inference_script)
    flags = dict(part.split("=", 1) for part in argv[2:])
    # The empirically safe configuration for a 32 GB card.
    assert flags["--views_per_layer"] == "24"
    assert flags["--views_per_group"] == "4"
    assert flags["--enable_rcp"] == "False"
    assert flags["--enable_tcr"] == "True"
    assert flags["--layer_pitches"] == "[15]"
    assert flags["--seed"] == "42"
    assert flags["--data_dir"] == str(settings.data_dir)
    # inference() takes gpu_ids, not device: it dropped the latter when it
    # gained multi-GPU view stages. fire binds what it knows, runs the job,
    # and only then chokes on the leftover -- so a stale flag costs 90 minutes.
    assert flags["--gpu_ids"] == "[0]"
    assert "--device" not in flags
    assert flags["--enable_turbo"] == "True"


def test_turbo_changes_the_fingerprint(tmp_path):
    """Turbo is a distilled LoRA over the same base, so it changes the ring."""

    video = tmp_path / "clip.mp4"
    video.touch()
    common = dict(video_path=video, views_per_layer=24, layer_pitches="15", start_yaw=0,
                  yaw_span=360, views_per_group=4, enable_rcp=True, enable_tcr=True,
                  start_time=0.0, target_fps="auto", seed=42, device="cuda:0")
    turbo = runner.build_request(**common, enable_turbo=True)
    base = runner.build_request(**common, enable_turbo=False)
    # The ring fingerprint is what gates the disk cache, not to_dict().
    assert runner.ring_fingerprint(turbo)[0] != runner.ring_fingerprint(base)[0]
    # ...but the motion cache does not depend on the denoiser, so it survives.
    assert runner.ring_fingerprint(turbo)[1] == runner.ring_fingerprint(base)[1]



@pytest.mark.parametrize("device,expected", [
    ("cuda:0", [0]),
    ("cuda:1", [1]),
    ("cuda", None),
    ("cpu", None),
])
def test_gpu_ids_translate_the_device_string(device, expected):
    assert runner.gpu_ids_for(device) == expected


def test_gpu_ids_rejects_a_malformed_device():
    with pytest.raises(runner.ValidationError):
        runner.gpu_ids_for("cuda:x")


def _fake_checkout(tmp_path, params):
    """A stand-in inference.py carrying just the signature we want to test."""

    script = tmp_path / "inference.py"
    script.write_text(f"def inference({', '.join(params)}):\n    return {{}}\n")
    return script


def test_argv_is_checked_against_the_checkouts_signature(settings, tmp_path):
    """A checkout that dropped a parameter fails before the GPU job, not after."""

    script = _fake_checkout(tmp_path, ["video_path", "seed"])
    stale = BridgeSettings(**{**settings.__dict__, "fdanyone_root": tmp_path})
    with pytest.raises(runner.ValidationError) as exc:
        runner.check_argv_against_checkout(
            stale, [sys.executable, str(script), "--video_path=x", "--gpu_ids=[0]"]
        )
    assert "--gpu_ids" in str(exc.value)


def test_argv_check_passes_when_every_flag_is_declared(settings, tmp_path):
    _fake_checkout(tmp_path, ["video_path", "gpu_ids"])
    ok = BridgeSettings(**{**settings.__dict__, "fdanyone_root": tmp_path})
    runner.check_argv_against_checkout(
        ok, [sys.executable, "inference.py", "--video_path=x", "--gpu_ids=[0]"]
    )


def test_argv_check_is_silent_when_the_signature_cannot_be_read(settings, tmp_path):
    """An unreadable checkout degrades to the old behaviour, it does not block."""

    (tmp_path / "inference.py").write_text("def inference(:\n")  # syntax error
    broken = BridgeSettings(**{**settings.__dict__, "fdanyone_root": tmp_path})
    runner.check_argv_against_checkout(broken, ["--anything=1"])


def test_argv_uses_conda_run_when_no_python_is_pinned(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(BridgeSettings, "find_conda", lambda self: "/opt/conda/bin/conda")
    settings = BridgeSettings(**{**settings.__dict__, "python_exe": ""})
    video = tmp_path / "clip.mp4"
    video.touch()
    argv = runner.build_argv(settings, known_good_request(video))
    assert argv[:6] == ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "4danyone", "python"]


def test_subprocess_env_drops_comfyui_python_paths(settings, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/media/IronWolf/ComfyUI")
    env = runner.build_env(settings)
    assert "PYTHONPATH" not in env
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"views_per_group": "5"}, "must be 'auto', 4 or 6"),
        ({"views_per_layer": 10, "views_per_group": "4"}, "divisible"),
        ({"layer_pitches": "60"}, "between -15 and 45"),
        ({"layer_pitches": "15,15"}, "must not repeat"),
        ({"layer_pitches": ""}, "at least one pitch"),
        ({"yaw_span": 0}, "between 1 and 360"),
        ({"seed": -1}, "non-negative"),
        ({"start_time": -2.0}, "non-negative"),
        ({"target_fps": "fast"}, "positive number"),
    ],
)
def test_bad_settings_are_refused_before_any_gpu_work(tmp_path, kwargs, fragment):
    video = tmp_path / "clip.mp4"
    video.touch()
    base = dict(
        video_path=video, views_per_layer=24, layer_pitches="15", start_yaw=0, yaw_span=360,
        views_per_group="4", enable_rcp=False, enable_tcr=True, start_time=0.0,
        target_fps="auto", seed=42, device="cuda:0",
    )
    with pytest.raises(runner.ValidationError, match=fragment):
        runner.build_request(**{**base, **kwargs})


def test_start_yaw_wraps_into_minus180_180(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    request = runner.build_request(
        video_path=video, views_per_layer=24, layer_pitches="15", start_yaw=270, yaw_span=360,
        views_per_group="auto", enable_rcp=False, enable_tcr=True, start_time=0.0,
        target_fps="auto", seed=0, device="cuda:0",
    )
    assert request.start_yaw == -90
    assert request.views_per_group == "auto"


@pytest.mark.parametrize(
    "source, canonical",
    [(48, 24), (50, 25), (60, 30), (120, 30), (40, 40), (24, 24), (30, 30)],
)
def test_canonical_fps_mirrors_the_checkout(source, canonical):
    """Pinned against fdanyone.video.choose_canonical_fps, which is duplicated
    here because the 4DAnyone package cannot be imported into this env."""

    assert runner.choose_canonical_fps(Fraction(source)) == Fraction(canonical)


def test_ntsc_rates_are_preserved_exactly():
    assert runner.choose_canonical_fps(Fraction(30000, 1001)) == Fraction(30000, 1001)
    assert runner.choose_canonical_fps(Fraction(60000, 1001)) == Fraction(30000, 1001)


def test_short_clip_is_rejected_with_a_useful_message(settings, tmp_path, monkeypatch):
    from cumuli_bridge.videoio import VideoProbe

    video = tmp_path / "clip.mp4"
    video.touch()
    probe = VideoProbe(path=video, width=1080, height=1920, fps=Fraction(30),
                       num_frames=60, duration=2.0, frames_are_exact=True)
    monkeypatch.setattr(runner, "probe", lambda path: probe)
    with pytest.raises(runner.ValidationError) as excinfo:
        runner.check_video(settings, known_good_request(video))
    message = str(excinfo.value)
    assert "121-frame clip" in message and "only 60 remain" in message


def test_low_resolution_input_is_rejected(settings, tmp_path, monkeypatch):
    from cumuli_bridge.videoio import VideoProbe

    video = tmp_path / "clip.mp4"
    video.touch()
    probe = VideoProbe(path=video, width=640, height=480, fps=Fraction(30),
                       num_frames=300, duration=10.0, frames_are_exact=True)
    monkeypatch.setattr(runner, "probe", lambda path: probe)
    with pytest.raises(runner.ValidationError, match="704"):
        runner.check_video(settings, known_good_request(video))


def test_artifact_dir_reuses_on_matching_fingerprint(tmp_path):
    root = tmp_path / "result"
    root.mkdir()
    runner.write_stamp(root, "abc")
    assert runner.prepare_artifact_dir(root, "abc") == "reuse"
    assert root.exists()


def test_artifact_dir_rebuilds_on_changed_fingerprint(tmp_path):
    root = tmp_path / "result"
    root.mkdir()
    (root / "old.txt").write_text("stale")
    runner.write_stamp(root, "abc")
    assert runner.prepare_artifact_dir(root, "def") == "build"
    assert not root.exists()  # stale artifact replaced, comfy-cache style


def test_artifact_dir_never_deletes_foreign_directories(tmp_path):
    root = tmp_path / "result"
    root.mkdir()
    (root / "somebody-elses.data").write_text("keep me")
    with pytest.raises(runner.ValidationError, match="not produced by this bridge"):
        runner.prepare_artifact_dir(root, "abc")
    assert (root / "somebody-elses.data").exists()


def test_ring_fingerprint_tracks_seed_but_motion_key_does_not(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    a = runner.build_request(video_path=video, views_per_layer=24, layer_pitches="15",
                             start_yaw=0, yaw_span=360, views_per_group="4", enable_rcp=True,
                             enable_tcr=True, start_time=0.0, target_fps="auto", seed=42, device="cuda:0")
    import dataclasses
    b = dataclasses.replace(a, seed=43)
    fa, ma = runner.ring_fingerprint(a)
    fb, mb = runner.ring_fingerprint(b)
    assert fa != fb, "seed must change the ring fingerprint"
    assert ma == mb, "seed must not invalidate the motion cache"


def test_run_name_symlinks_the_source_so_the_motion_cache_key_changes(settings, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"data")
    staged = runner.stage_source_video(settings, video, "take07")
    assert staged.stem == "take07"
    assert staged.is_symlink() and staged.resolve() == video.resolve()
    # Idempotent: the same name resolves to the same link.
    assert runner.stage_source_video(settings, video, "take07") == staged


def test_run_name_rejects_path_traversal(settings, tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    with pytest.raises(runner.ValidationError, match="run_name must"):
        runner.stage_source_video(settings, video, "../escape")


def test_a_staged_run_name_survives_into_the_request(settings, tmp_path):
    """Regression: build_request used to call Path.resolve(), which followed the
    staging symlink back to the original filename. The run then wrote into the
    ORIGINAL run's result directory -- and the overwrite path of the day
    destroyed it. The request must keep the link's own name."""

    video = tmp_path / "cam01x24b.mp4"
    video.write_bytes(b"data")
    staged = runner.stage_source_video(settings, video, "take07")
    request = runner.build_request(
        video_path=staged, views_per_layer=24, layer_pitches="15", start_yaw=0, yaw_span=360,
        views_per_group="4", enable_rcp=False, enable_tcr=True, start_time=0.0,
        target_fps="auto", seed=42, device="cuda:0",
    )
    assert request.run_name == "take07", "the staging symlink was resolved away"
    assert request.video_path.is_absolute()
    assert settings.result_dir(request.run_name).name == "take07"
    assert "cam01x24b" not in str(settings.result_dir(request.run_name))


# --------------------------------------------------------------------------
# camera convention
# --------------------------------------------------------------------------
def test_opencv_to_opengl_negates_the_y_and_z_columns():
    """The published matrix for camera 00 of a real run, and the OpenGL matrix
    a previously verified conversion of the same run produced."""

    camera = RingCamera(
        camera_id=0, layer_index=0, pitch=15, yaw=0.0,
        k=[[1664.0, 0.0, 352.0], [0.0, 1664.0, 640.0], [0.0, 0.0, 1.0]],
        camera_to_world=[
            [1.0, 1.5714814722902105e-17, -5.864848697740211e-17, -0.022216394543647603],
            [-4.6610313051207004e-33, -0.9659258262890683, -0.2588190451025207, 1.719610559293209],
            [-6.071738158479536e-17, 0.2588190451025207, -0.9659258262890684, 2.821328461170197],
            [0.0, 0.0, 0.0, 1.0],
        ],
        width=704, height=1280, video="videos/dense/00.mp4", skeleton_video="skeletons/00.mp4",
    )
    expected = [
        [1.0, -1.5714814722902105e-17, 5.864848697740211e-17, -0.022216394543647603],
        [-4.6610313051207004e-33, 0.9659258262890683, 0.2588190451025207, 1.719610559293209],
        [-6.071738158479536e-17, -0.2588190451025207, 0.9659258262890684, 2.821328461170197],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert camera.nerf_transform() == expected
    # Translation is untouched, and the flip is its own inverse.
    assert camera.label == "gen00"


def test_camera_labels_are_fixed_width_and_sort_in_camera_order():
    assert label_width(24) == 2
    assert label_width(200) == 3
    labels = [camera_label(i, label_width(200)) for i in range(200)]
    assert labels == sorted(labels)
    assert labels[0] == "000" and labels[-1] == "199"


# --------------------------------------------------------------------------
# progress parsing
# --------------------------------------------------------------------------
def test_progress_is_monotonic_across_a_whole_run():
    state = ProgressState(expected_views=24)
    lines = [
        "2026-08-29 12:00:00 | INFO | Using cuda:0 (NVIDIA GeForce RTX 5090)",
        "2026-08-29 12:00:01 | INFO | Reusing validated GVHMR result at data/gvhmr/results/x",
        "2026-08-29 12:00:02 | INFO | Estimating source foreground masks with BiRefNet",
        "RCP 1-to-4:  50%|#####     | 12/24 [00:30<00:30,  2.5s/it]",
        "Generate 24 target views:   0%|          | 0/24 [00:00<?, ?it/s]",
        "Generate 24 target views:  50%|#####     | 12/24 [27:20<27:20, 136.7s/it]",
        "Generate 24 target views: 100%|##########| 24/24 [54:40<00:00, 136.7s/it]",
        "2026-08-29 13:00:00 | INFO | Decoding target camera 00",
        "2026-08-29 13:00:20 | INFO | Decoding target camera 23",
    ]
    seen = []
    for line in lines:
        state.update(line)
        seen.append(state.fraction)
    assert seen == sorted(seen), "progress went backwards"
    assert 0.0 < seen[0] < seen[-1] < 1.0


def test_denoise_bar_dominates_the_progress_range():
    state = ProgressState(expected_views=24)
    state.update("Generate 24 target views:  50%|#####     | 12/24 [00:00<00:00]")
    midpoint = state.fraction
    assert 0.4 < midpoint < 0.55, f"denoise midpoint should sit near half, got {midpoint}"
    assert state.message == "denoising step 12/24"


def test_unrelated_output_does_not_move_the_bar():
    state = ProgressState()
    assert not state.update("some incidental library warning")
    assert state.fraction == 0.0


# --------------------------------------------------------------------------
# subprocess plumbing
# --------------------------------------------------------------------------
TQDM_CHILD = (
    "import sys, time\n"
    "sys.stderr.write('Generate 24 target views:   0%|          | 0/24 [00:00<?, ?it/s]\\r')\n"
    "sys.stderr.flush()\n"
    "for i in range(1, 25):\n"
    "    sys.stderr.write('Generate 24 target views: %3d%%|##| %d/24 [00:01<00:01]\\r' % (i*100//24, i))\n"
    "    sys.stderr.flush()\n"
    "sys.stdout.write('done\\n')\n"
)


def test_tqdm_carriage_returns_arrive_as_separate_lines(tmp_path):
    lines: list[str] = []
    result = run_streaming([sys.executable, "-c", TQDM_CHILD], cwd=tmp_path, on_line=lines.append)
    assert result.returncode == 0
    bars = [line for line in lines if "target views" in line]
    assert len(bars) == 25, f"expected one line per bar repaint, got {len(bars)}"
    state = ProgressState(expected_views=24)
    for line in lines:
        state.update(line)
    assert state.fraction > 0.8


def test_stderr_and_stdout_share_one_pipe_so_nothing_deadlocks(tmp_path):
    """A child that writes far more than a pipe buffer to both streams."""

    child = (
        "import sys\n"
        "for i in range(4000):\n"
        "    sys.stdout.write('out %d\\n' % i)\n"
        "    sys.stderr.write('err %d\\n' % i)\n"
    )
    result = run_streaming([sys.executable, "-c", child], cwd=tmp_path, tail_lines=10)
    assert result.returncode == 0
    assert result.elapsed < 30


def test_nonzero_exit_raises_with_the_tail_of_the_output(tmp_path):
    child = "import sys\nsys.stderr.write('error: boom\\n')\nraise SystemExit(3)\n"
    with pytest.raises(SubprocessError) as excinfo:
        run_streaming([sys.executable, "-c", child], cwd=tmp_path)
    assert excinfo.value.returncode == 3
    assert "boom" in str(excinfo.value)


def test_cancellation_kills_the_child(tmp_path):
    child = "import sys, time\nwhile True:\n    sys.stdout.write('tick\\n')\n    sys.stdout.flush()\n    time.sleep(0.05)\n"
    started = time.monotonic()
    seen = {"n": 0}

    def cancel() -> bool:
        return seen["n"] > 3

    def count(_line: str) -> None:
        seen["n"] += 1

    with pytest.raises(SubprocessCancelled):
        run_streaming([sys.executable, "-c", child], cwd=tmp_path, on_line=count, should_cancel=cancel)
    assert time.monotonic() - started < 30


# --------------------------------------------------------------------------
# dataset contract
# --------------------------------------------------------------------------
def _write_dataset(root: Path, *, mode: str = "RGBA", ply_time: bool = True, keys: bool = True) -> Path:
    from PIL import Image

    (root / "realcams" / "cam00").mkdir(parents=True)
    Image.new(mode, (8, 8)).save(root / "realcams" / "cam00" / "frame_00001.png")
    frame = {
        "file_path": "realcams/cam00/frame_00001",
        "camera_label": "00",
        "time": 0.0,
        "fl_x": 832.0, "fl_y": 832.0, "cx": 4.0, "cy": 4.0,
        "w": 8, "h": 8,
        "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    }
    if not keys:
        del frame["fl_x"]
    for name in ("transforms_train.json", "transforms_test.json"):
        (root / name).write_text(json.dumps({"camera_model": "OPENCV", "frames": [frame]}))
    time_prop = "property float time\n" if ply_time else ""
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"{time_prop}end_header\n"
    )
    (root / "points3d.ply").write_bytes(header.encode() + b"\x00" * 32)
    return root


def test_a_well_formed_dataset_validates(tmp_path):
    report = validate_dataset(_write_dataset(tmp_path / "ok"))
    assert report["init_points"] == 1
    assert report["probe_mode"] == "RGBA"


def test_rgb_images_are_rejected_because_the_mask_bake_failed(tmp_path):
    with pytest.raises(ValidationError, match="expected RGBA"):
        validate_dataset(_write_dataset(tmp_path / "rgb", mode="RGB"))


def test_a_static_init_cloud_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="per-point 'time'"):
        validate_dataset(_write_dataset(tmp_path / "static", ply_time=False))


def test_missing_per_view_intrinsics_are_rejected(tmp_path):
    with pytest.raises(ValidationError, match="lack 'fl_x'"):
        validate_dataset(_write_dataset(tmp_path / "nointr", keys=False))


# --------------------------------------------------------------------------
# training and bake
# --------------------------------------------------------------------------
def test_trainer_progress_tracks_iterations_and_psnr():
    """The real bar the trainer prints, updated every 10 iterations on stderr."""

    state = TrainProgress(total_iterations=30000)
    assert state.update(
        "Training progress:  33%|###3      | 10000/30000 [12:34<25:08, 13.26it/s, "
        "Loss=0.0123456, PSNR=32.15, Ll1=0.0091, Lssim=0.0512]"
    )
    assert state.iteration == 10000
    assert state.psnr == pytest.approx(32.15)
    assert state.loss == pytest.approx(0.0123456)
    assert state.fraction == pytest.approx(1 / 3, abs=1e-6)


def test_trainer_progress_never_goes_backwards():
    state = TrainProgress(total_iterations=200)
    for iteration in (20, 100, 100, 110, 200):
        state.update(f"Training progress:  50%|##  | {iteration}/200 [00:01<00:01, 1it/s, PSNR=30.99]")
    seen = state.fraction
    state.update("Training progress:  10%|#   | 20/200 [00:01<00:01, 1it/s, PSNR=20.00]")
    assert state.fraction == seen


def test_trainer_first_run_compile_is_reported_not_mistaken_for_progress():
    state = TrainProgress()
    assert state.update("Building extension module diff_gaussian_rasterization...")
    assert state.fraction == 0.0
    assert "compiling" in state.message


def test_bake_progress_walks_the_real_milestones():
    state = BakeProgress()
    lines = [
        "Loading checkpoint /x/chkpnt200.pth ...",
        "  Slicing 4D covariance ...",
        "  Folding temporal SH at each splat's own t_center ...",
        "  SH overshoot clamp: attenuated 12 / 19,676 splats",
        "  Pruning: dropped 3 / 19,676 Gaussians below alpha 0.00098 inside time range",
    ]
    seen = []
    for line in lines:
        state.update(line)
        seen.append(state.fraction)
    assert seen == sorted(seen)
    assert 0.0 < seen[0] < seen[-1] < 1.0


def _train_options(tmp_path: Path, dataset: Path) -> TrainOptions:
    return TrainOptions(
        out_dir=tmp_path / "run",
        dataset_dir=dataset,
        duration_seconds=4.004,
        fps=29.97,
        iterations=30000,
        num_pts=100000,
        batch_size=2,
    )


def test_train_argv_matches_the_orchestrator(settings, tmp_path):
    dataset = _write_dataset(tmp_path / "ds")
    options = _train_options(tmp_path, dataset)
    argv = build_train_argv(settings, options, tmp_path / "gs4d_config.yaml")
    assert argv[0] == sys.executable
    assert argv[1].endswith("train_scratch.py")
    # Explicit iteration lists are what make the checkpoint filename deterministic.
    assert argv[2:] == [
        "--config", str(tmp_path / "gs4d_config.yaml"),
        "--test_iterations", "30000",
        "--save_iterations", "30000",
    ]
    assert options.checkpoint.name == "chkpnt30000.pth"


def test_t_init_div_reaches_the_trainer_as_an_env_var(settings, tmp_path):
    dataset = _write_dataset(tmp_path / "ds")
    options = _train_options(tmp_path, dataset)
    options.t_init_div = 100
    assert build_train_env(settings, options)["GS4D_T_INIT_DIV"] == "100"
    options.t_init_div = 0  # 0 means "leave upstream's default of 5 alone"
    assert "GS4D_T_INIT_DIV" not in build_train_env(settings, options)


def test_a_sparse_directory_is_refused_because_it_switches_the_loader(settings, tmp_path):
    dataset = _write_dataset(tmp_path / "ds")
    (dataset / "sparse").mkdir()
    with pytest.raises(TrainingError, match="COLMAP"):
        _train_options(tmp_path, dataset).validate()


def test_training_needs_a_built_dataset(tmp_path):
    options = TrainOptions(out_dir=tmp_path, dataset_dir=tmp_path / "nothing", duration_seconds=1.0)
    with pytest.raises(TrainingError, match="transforms_train.json"):
        options.validate()


# --------------------------------------------------------------------------
# the 4D evaluation, whose three traps all fail silently
# --------------------------------------------------------------------------
def _write_interchange_ply(path: Path, *, rows: list[dict], accel: bool = False, sh: bool = False,
                           comments: dict | None = None) -> Path:
    import numpy as np

    from cumuli_bridge.sogst import ACCEL_COLUMNS, BASE_COLUMNS

    names = list(BASE_COLUMNS)
    if sh:
        names += [f"f_rest_{i}" for i in range(45)]
    if accel:
        names += list(ACCEL_COLUMNS)
    header = "ply\nformat binary_little_endian 1.0\n"
    for key, value in (comments or {"time_min": "0.0", "time_max": "4.0", "fps": "30"}).items():
        header += f"comment sogst.{key} {value}\n"
    header += f"element vertex {len(rows)}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    body = np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=np.float32)
    path.write_bytes(header.encode("ascii") + body.tobytes())
    return path


def _row(**overrides) -> dict:
    row = {name: 0.0 for name in
           ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2",
            "opacity", "f_dc_0", "f_dc_1", "f_dc_2", "vx", "vy", "vz", "t_center", "t_sigma")}
    row["rot_0"] = 1.0        # identity quaternion, w first
    row["t_sigma"] = 1.0
    row["opacity"] = 0.0      # logit 0 -> peak alpha 0.5
    row.update(overrides)
    return row


def test_position_moves_along_the_velocity_from_t_center(tmp_path):
    from cumuli_bridge.sogst import load_interchange_ply

    path = _write_interchange_ply(tmp_path / "a.ply", rows=[_row(x=1.0, vx=2.0, t_center=1.0)])
    asset = load_interchange_ply(path)
    # Attributes are stored AT t_center, not at t = 0.
    assert asset.mean_at(1.0)[0, 0] == pytest.approx(1.0)
    assert asset.mean_at(3.0)[0, 0] == pytest.approx(5.0)
    assert asset.mean_at(0.0)[0, 0] == pytest.approx(-1.0)


def test_acceleration_is_the_raw_dt_squared_coefficient(tmp_path):
    """Trap 3: there is no factor of 1/2."""

    from cumuli_bridge.sogst import load_interchange_ply

    path = _write_interchange_ply(
        tmp_path / "a.ply", rows=[_row(x=0.0, vx=0.0, ax=3.0, t_center=0.0)], accel=True
    )
    asset = load_interchange_ply(path)
    assert asset.motion_degree == 2
    assert asset.mean_at(2.0)[0, 0] == pytest.approx(12.0)  # 3 * 2^2, not 0.5 * 3 * 2^2


def test_the_temporal_envelope_is_unnormalised(tmp_path):
    """Trap 1: no 1/sqrt(2*pi*sigma^2). Peak alpha is sigmoid(opacity) exactly."""

    from cumuli_bridge.sogst import load_interchange_ply

    path = _write_interchange_ply(tmp_path / "a.ply", rows=[_row(opacity=0.0, t_center=1.0, t_sigma=0.5)])
    asset = load_interchange_ply(path)
    assert asset.alpha_at(1.0)[0] == pytest.approx(0.5)  # sigmoid(0), undiminished


def test_t_sigma_is_a_standard_deviation_not_a_variance(tmp_path):
    """Trap 2: at one sigma away the envelope is exp(-0.5)."""

    from cumuli_bridge.sogst import load_interchange_ply

    path = _write_interchange_ply(tmp_path / "a.ply", rows=[_row(opacity=0.0, t_center=0.0, t_sigma=2.0)])
    asset = load_interchange_ply(path)
    assert asset.alpha_at(2.0)[0] == pytest.approx(0.5 * math.exp(-0.5), rel=1e-5)


def test_scales_are_delogged_and_sh_is_channel_major(tmp_path):
    import numpy as np

    from cumuli_bridge.sogst import load_interchange_ply

    rest = {f"f_rest_{i}": float(i) for i in range(45)}
    path = _write_interchange_ply(
        tmp_path / "a.ply",
        rows=[_row(scale_0=math.log(2.0), scale_1=math.log(3.0), scale_2=math.log(4.0),
                   f_dc_0=1.0, f_dc_1=2.0, f_dc_2=3.0, **rest)],
        sh=True,
    )
    asset = load_interchange_ply(path)
    assert np.allclose(np.exp(asset.log_scale[0]), [2.0, 3.0, 4.0])
    sh = asset.sh_at()
    assert sh.shape == (1, 16, 3)                       # degree 3 -> K = 16
    assert np.allclose(sh[0, 0], [1.0, 2.0, 3.0])       # DC first
    # f_rest is channel-major: the first 15 values are channel R of coeffs 1..15.
    assert sh[0, 1, 0] == pytest.approx(0.0)
    assert sh[0, 1, 1] == pytest.approx(15.0)
    assert sh[0, 1, 2] == pytest.approx(30.0)


def test_a_plain_3dgs_ply_is_refused_with_a_useful_message(tmp_path):
    from cumuli_bridge.sogst import SogstError, load_interchange_ply

    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
              "property float x\nproperty float y\nproperty float z\nend_header\n")
    path = tmp_path / "plain.ply"
    path.write_bytes(header.encode() + b"\x00" * 12)
    with pytest.raises(SogstError, match="spacetime fields"):
        load_interchange_ply(path)


def test_missing_clip_scalars_are_an_error_not_a_guess(tmp_path):
    from cumuli_bridge.sogst import SogstError, load_interchange_ply

    path = _write_interchange_ply(tmp_path / "a.ply", rows=[_row()], comments={"time_min": "0.0"})
    with pytest.raises(SogstError, match="no 'time_max'"):
        load_interchange_ply(path)


def test_the_sidecar_supplies_clip_scalars_when_comments_are_stripped(tmp_path):
    """Specification 7.3: comments or sidecar, either alone is conforming."""

    from cumuli_bridge.sogst import load_interchange_ply

    path = _write_interchange_ply(tmp_path / "a.ply", rows=[_row()], comments={})
    (tmp_path / "a.sogst.json").write_text(json.dumps({"time_min": 0.0, "time_max": 4.0, "fps": 30}))
    asset = load_interchange_ply(path)
    assert (asset.time_min, asset.time_max, asset.fps) == (0.0, 4.0, 30.0)


def test_the_sidecar_wins_over_the_comments(tmp_path):
    from cumuli_bridge.sogst import load_interchange_ply

    path = _write_interchange_ply(tmp_path / "a.ply", rows=[_row()],
                                  comments={"time_min": "0.0", "time_max": "9.0", "fps": "24"})
    (tmp_path / "a.sogst.json").write_text(json.dumps({"time_min": 0.0, "time_max": 4.0, "fps": 30}))
    asset = load_interchange_ply(path)
    assert asset.fps == 30.0 and asset.time_max == 4.0


def test_inactive_splats_are_culled_at_the_chosen_instant(tmp_path):
    from cumuli_bridge.sogst import load_interchange_ply

    rows = [_row(t_center=0.0, t_sigma=0.1), _row(t_center=4.0, t_sigma=0.1)]
    asset = load_interchange_ply(_write_interchange_ply(tmp_path / "a.ply", rows=rows))
    alpha = asset.alpha_at(0.0)
    assert alpha[0] > 0.4 and alpha[1] < 1e-6


def test_bake_argv_disables_the_legacy_corruption_filters_by_default(settings, tmp_path, monkeypatch):
    script = tmp_path / "bake_sogst.py"
    script.touch()
    monkeypatch.setattr("cumuli_bridge.train.find_bake_script", lambda _s: script)
    argv = build_bake_argv(
        settings,
        BakeOptions(
            checkpoint=tmp_path / "chkpnt200.pth",
            output=tmp_path / "out.sogst",
            duration_seconds=4.004,
            fps=29.97,
            mask_filter_root=tmp_path / "ds",
        ),
    )
    flags = argv[argv.index(str(script)) + 1 :]
    assert "--no_filter_corrupted" in flags
    assert flags[flags.index("--time_max") + 1] == "4.004000"
    assert flags[flags.index("--mask_filter_root") + 1] == str(tmp_path / "ds")
    # sh_clamp 1.5 is upstream's default and is too aggressive for explicit SH.
    assert float(flags[flags.index("--sh_clamp") + 1]) == 3.0


# --- VIDEO-socket materialization -------------------------------------------------

class _FakeVideo:
    """Duck-typed stand-in for comfy_api's VideoInput."""

    def __init__(self, source=None, trim=(0.0, 0.0), frames=b""):
        self._source = source
        self._trim = trim
        self._frames = frames

    def get_stream_source(self):
        return self._source

    def get_active_trim_window(self):
        return self._trim

    def save_to(self, path, **kwargs):
        Path(path).write_bytes(self._frames)


def test_file_backed_untrimmed_video_is_used_in_place(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"data")
    out = runner.materialize_video_source(_FakeVideo(source=str(clip)), "", tmp_path / "src")
    assert out == clip.resolve()
    assert not (tmp_path / "src").exists()  # nothing written


def test_trimmed_video_requires_a_run_name(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"data")
    with pytest.raises(runner.ValidationError, match="run_name"):
        runner.materialize_video_source(
            _FakeVideo(source=str(clip), trim=(1.5, 3.0)), "", tmp_path / "src"
        )


def test_trimmed_video_is_written_under_the_run_name(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"data")
    out = runner.materialize_video_source(
        _FakeVideo(source=str(clip), trim=(1.5, 3.0), frames=b"trimmed"), "take01", tmp_path / "src"
    )
    assert out == tmp_path / "src" / "take01.mp4"
    assert out.read_bytes() == b"trimmed"


def test_synthesized_video_round_trips_through_save_to(tmp_path):
    out = runner.materialize_video_source(
        _FakeVideo(source=None, frames=b"encoded"), "gen01", tmp_path / "src"
    )
    assert out.read_bytes() == b"encoded"


def test_bytesio_video_keeps_its_container_bytes(tmp_path):
    import io as _io

    out = runner.materialize_video_source(
        _FakeVideo(source=_io.BytesIO(b"mp4-bytes")), "mem01", tmp_path / "src"
    )
    assert out.read_bytes() == b"mp4-bytes"


def test_identical_requeue_keeps_the_staged_file_mtime(tmp_path):
    video = _FakeVideo(source=None, frames=b"stable")
    first = runner.materialize_video_source(video, "take02", tmp_path / "src")
    before = first.stat().st_mtime_ns
    time.sleep(0.01)
    second = runner.materialize_video_source(video, "take02", tmp_path / "src")
    assert second == first
    assert first.stat().st_mtime_ns == before  # motion-cache validation survives


def test_changed_bytes_replace_the_staged_file(tmp_path):
    first = runner.materialize_video_source(_FakeVideo(frames=b"v1"), "take03", tmp_path / "src")
    runner.materialize_video_source(_FakeVideo(frames=b"v2-longer"), "take03", tmp_path / "src")
    assert first.read_bytes() == b"v2-longer"
    assert not list(first.parent.glob(".*.tmp-*"))  # no temp debris


# --- work_root ---------------------------------------------------------------------

def test_work_dir_is_none_when_unset(settings):
    assert settings.work_root == ""
    assert settings.work_dir("clip01") is None


def test_work_dir_nests_the_run_name(fake_repo):
    s = BridgeSettings(
        fdanyone_root=fake_repo,
        conda_env="", conda_exe="", python_exe=sys.executable,
        data_dir=fake_repo / "data", model_dir=fake_repo / "models",
        gvhmr_root=fake_repo / "third_party" / "GVHMR",
        device="cuda:0", min_free_vram_gb=0.0,
        work_root="/big/drive/comfy",
    )
    assert s.work_dir("clip01") == Path("/big/drive/comfy/clip01")


def test_work_root_loads_from_env(monkeypatch, fake_repo):
    monkeypatch.setenv("CUMULI_WORK_ROOT", "/big/drive/comfy")
    monkeypatch.setenv("CUMULI_FDANYONE_ROOT", str(fake_repo))
    monkeypatch.delenv("CUMULI_CONFIG", raising=False)
    monkeypatch.setenv("CUMULI_CONFIG", str(fake_repo / "no-such-config.json"))
    loaded = BridgeSettings.load()
    assert loaded.work_root == "/big/drive/comfy"
    assert loaded.work_dir("r1") == Path("/big/drive/comfy/r1")


# --- LoRA factor folding -----------------------------------------------------------

class _Adapter:
    name = "lora"

    def __init__(self, up, down, alpha, mid=None, dora=None):
        self.weights = (up, down, alpha, mid, dora, None)


def _entry(patch, strength=1.0):
    return (strength, patch, 1.0, None, None)


def test_lora_fold_matches_manual_math():
    import torch
    up = torch.randn(8, 2); down = torch.randn(2, 6)
    patches = {"diffusion_model.blocks.0.q.weight": [_entry(_Adapter(up, down, 4.0), strength=0.5)]}
    factors, merged, bad = runner.extract_lora_factors(patches)
    assert merged == 1 and not bad
    fup, fdown = factors["blocks.0.q"]
    expect = 0.5 * (4.0 / 2) * (up @ down)
    assert torch.allclose(fup.float() @ fdown.float(), expect, atol=1e-2)


def test_lora_fold_concatenates_stacked_entries():
    import torch
    a = _Adapter(torch.randn(8, 2), torch.randn(2, 6), 2.0)
    b = _Adapter(torch.randn(8, 3), torch.randn(3, 6), 3.0)
    patches = {"diffusion_model.x.weight": [_entry(a, 1.0), _entry(b, -0.5)]}
    factors, merged, bad = runner.extract_lora_factors(patches)
    fup, fdown = factors["x"]
    assert fdown.shape[0] == 5  # ranks concatenated
    expect = (2.0 / 2) * (a.weights[0] @ a.weights[1]) - 0.5 * (3.0 / 3) * (b.weights[0] @ b.weights[1])
    assert torch.allclose(fup.float() @ fdown.float(), expect, atol=1e-2)


def test_lora_fold_supports_legacy_tuples():
    import torch
    up = torch.randn(4, 2); down = torch.randn(2, 4)
    patches = {"diffusion_model.y.weight": [_entry(("lora", (up, down, None, None, None, None)))]}
    factors, merged, bad = runner.extract_lora_factors(patches)
    assert merged == 1 and not bad  # alpha None -> scale 1
    fup, fdown = factors["y"]
    import torch as t
    assert t.allclose(fup.float() @ fdown.float(), up @ down, atol=1e-2)


def test_lora_fold_refuses_what_it_cannot_express():
    import torch
    patches = {"diffusion_model.z.weight": [(1.0, object(), 1.0, None, None)]}
    factors, merged, bad = runner.extract_lora_factors(patches)
    assert not factors and bad and "object" in bad[0]


def test_lora_factors_round_trip_through_safetensors(tmp_path):
    import torch
    from safetensors.torch import load_file
    factors = {"blocks.0.q": (torch.randn(4, 2).half(), torch.randn(2, 4).half())}
    out = runner.save_lora_factors(factors, tmp_path / "f.safetensors")
    back = load_file(str(out))
    assert set(back) == {"blocks.0.q.up", "blocks.0.q.down"}


# --- flipbook loading --------------------------------------------------------------

def _fake_flipbook_tree(root, labels=("00", "01"), frames=3, w=64, h=48):
    import json as _json
    transforms = {"camera_model": "OPENCV", "frames": [
        {"camera_label": label, "transform_matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
         "fl_x": 50.0, "fl_y": 50.0, "cx": w/2, "cy": h/2, "w": w, "h": h} for label in labels]}
    for index in range(frames):
        d = root / f"frame_{index:04d}"
        (d / "images_flat").mkdir(parents=True)
        (d / "transforms.json").write_text(_json.dumps(transforms))
        for label in labels:
            (d / "images_flat" / f"{label}.png").write_bytes(b"png")
    return root


def test_load_flipbook_reconstructs_from_tree(tmp_path):
    from cumuli_bridge.flipbook import load_flipbook
    root = _fake_flipbook_tree(tmp_path / "fb")
    fb = load_flipbook(root, Fraction(24))
    assert fb.labels == ("00", "01") and fb.num_frames == 3
    assert (fb.width, fb.height) == (64, 48) and fb.fps == Fraction(24)


def test_load_flipbook_requires_a_frame_rate(tmp_path):
    from cumuli_bridge.flipbook import FlipbookError, load_flipbook
    root = _fake_flipbook_tree(tmp_path / "fb")
    with pytest.raises(FlipbookError, match="frame rate"):
        load_flipbook(root, Fraction(0))


def test_load_flipbook_rejects_incomplete_trees(tmp_path):
    from cumuli_bridge.flipbook import FlipbookError, load_flipbook
    root = _fake_flipbook_tree(tmp_path / "fb")
    (root / "frame_0002" / "images_flat" / "01.png").unlink()
    with pytest.raises(FlipbookError, match="incomplete"):
        load_flipbook(root, Fraction(24))



# --- artifact discovery ------------------------------------------------------------

def _settings_with_roots(fake_repo, **kw):
    return BridgeSettings(
        fdanyone_root=fake_repo, conda_env="", conda_exe="", python_exe=sys.executable,
        data_dir=fake_repo / "data", model_dir=fake_repo / "models",
        gvhmr_root=fake_repo / "third_party" / "GVHMR",
        device="cuda:0", min_free_vram_gb=0.0, **kw,
    )


def test_discovery_finds_datasets_at_all_depths(fake_repo, tmp_path):
    flat = tmp_path / "captures" / "heidi"
    nested = tmp_path / "work" / "run01" / "dataset_4dgs"
    for d in (flat, nested):
        d.mkdir(parents=True)
        (d / "transforms_train.json").write_text("{}")
        (d / "points3d.ply").write_text("ply")
    (tmp_path / "captures" / "not_a_dataset").mkdir()
    s = _settings_with_roots(fake_repo, work_root=str(tmp_path / "work"),
                             dataset_roots=(str(tmp_path / "captures"),))
    assert runner.discover_datasets(s) == sorted([str(flat), str(nested)])


def test_discovery_finds_flipbooks(fake_repo, tmp_path):
    tree = tmp_path / "work" / "run01" / "flipbook_src"
    (tree / "frame_0000").mkdir(parents=True)
    (tree / "frame_0000" / "transforms.json").write_text("{}")
    s = _settings_with_roots(fake_repo, work_root=str(tmp_path / "work"))
    assert runner.discover_flipbooks(s) == [str(tree)]


def test_discovery_survives_missing_roots(fake_repo, tmp_path):
    s = _settings_with_roots(fake_repo, work_root=str(tmp_path / "nope"),
                             dataset_roots=("/does/not/exist",))
    assert runner.discover_datasets(s) == []


# --- ComfyUI settings-store layer --------------------------------------------------

def test_comfy_settings_layer_applies(monkeypatch, tmp_path, fake_repo):
    from cumuli_bridge import settings as settings_mod
    store = tmp_path / "comfy.settings.json"
    store.write_text(json.dumps({
        "cumuli.work_root": "/big/drive",
        "cumuli.dataset_roots": "/a, /b",
        "Comfy.SomethingElse": True,
    }))
    monkeypatch.setattr(settings_mod, "_comfy_settings_file", lambda: store)
    monkeypatch.setenv("CUMULI_CONFIG", str(tmp_path / "missing.json"))
    monkeypatch.setenv("CUMULI_FDANYONE_ROOT", str(fake_repo))
    loaded = BridgeSettings.load()
    assert loaded.work_root == "/big/drive"
    assert loaded.dataset_roots == ("/a", "/b")


def test_config_file_beats_ui_settings(monkeypatch, tmp_path, fake_repo):
    from cumuli_bridge import settings as settings_mod
    store = tmp_path / "comfy.settings.json"
    store.write_text(json.dumps({"cumuli.work_root": "/from/ui"}))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"work_root": "/from/file"}))
    monkeypatch.setattr(settings_mod, "_comfy_settings_file", lambda: store)
    monkeypatch.setenv("CUMULI_CONFIG", str(cfg))
    monkeypatch.setenv("CUMULI_FDANYONE_ROOT", str(fake_repo))
    assert BridgeSettings.load().work_root == "/from/file"


def test_roots_accept_lists_and_comma_strings():
    from cumuli_bridge.settings import _roots
    assert _roots("/a, /b ,") == ("/a", "/b")
    assert _roots(["/x", "/y"]) == ("/x", "/y")
    assert _roots(None) == ()


# --- CUDA toolchain selection ------------------------------------------------------

def test_toolchain_prefers_newest_cuda(tmp_path):
    from cumuli_bridge.train import cuda_toolchain_env
    for version in ("cuda-12.0", "cuda-13.2"):
        (tmp_path / version / "bin").mkdir(parents=True)
        (tmp_path / version / "bin" / "nvcc").write_text("")
    env = cuda_toolchain_env(base=tmp_path, environ={"PATH": "/usr/bin"})
    assert env["CUDA_HOME"].endswith("cuda-13.2")
    assert env["PATH"].startswith(str(tmp_path / "cuda-13.2" / "bin"))


def test_toolchain_respects_existing_cuda_home(tmp_path):
    from cumuli_bridge.train import cuda_toolchain_env
    assert cuda_toolchain_env(base=tmp_path, environ={"CUDA_HOME": "/opt/cuda"}) == {}


def test_toolchain_empty_when_nothing_found(tmp_path):
    from cumuli_bridge.train import cuda_toolchain_env
    assert cuda_toolchain_env(base=tmp_path, environ={}) == {}


def test_discover_rings_finds_published_results(settings, tmp_path):
    """A ring is a directory with cameras.json and videos/ -- nothing else counts."""

    root = tmp_path / "fdanyone"
    good = root / "run_a"
    (good / "videos").mkdir(parents=True)
    (good / "cameras.json").write_text("{}")
    half = root / "run_b"          # cameras.json but no videos/
    half.mkdir(parents=True)
    (half / "cameras.json").write_text("{}")
    (root / "run_c").mkdir()       # neither

    scoped = BridgeSettings(**{**settings.__dict__, "data_dir": tmp_path})
    found = runner.discover_rings(scoped)
    assert found == [str(good)]


def test_discover_rings_includes_configured_ring_roots(settings, tmp_path):
    extra = tmp_path / "elsewhere" / "copied_run"
    (extra / "videos").mkdir(parents=True)
    (extra / "cameras.json").write_text("{}")
    scoped = BridgeSettings(**{**settings.__dict__, "data_dir": tmp_path,
                               "ring_roots": (str(tmp_path / "elsewhere"),)})
    assert str(extra) in runner.discover_rings(scoped)
