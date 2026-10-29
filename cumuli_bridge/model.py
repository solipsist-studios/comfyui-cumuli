# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Read model of a trained 4DGS run: one or more windows over one clip.

A clip trains either as one model or as several short windows that Bake SOGST
stitches back together. Both are the same thing here -- a list of windows,
where a whole-clip model is a list of one -- so Bake has a single code path
and cannot be handed a window without knowing that it is one.

On disk this is cumuli's own format, not one private to this pack, so a run
trained by cumuli's command line (``plan_temporal_windows.py`` then
``run_window_plan.py``) loads here unchanged, and a run trained here can be
stitched by ``merge_sogst_segments.py --plan``:

* ``window_plan.json`` names each window's ``out_dir`` and ``offset_seconds``
  and the clip's ``fps``.
* Each ``out_dir`` holds the trainer config, ``gs4d_config.yaml``, which the
  trainer itself reads and which therefore already records everything else a
  bake needs: the window's duration (``time_duration``), its dataset
  (``source_path``), where its checkpoints are (``model_path``), their name
  (``chkpnt<iterations>.pth``) and ``sh_degree``.

This module is the only thing that knows that layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .train import CONFIG_NAME, MODEL_DIRNAME

#: The plan file, named as plan_temporal_windows.py's documentation names it.
PLAN_NAME = "window_plan.json"
#: A window's dataset directory, named as cumuli's orchestrators name it.
DATASET_DIRNAME = "dataset_4dgs"


class ModelError(RuntimeError):
    """Raised when a path does not describe a usable trained model."""


@dataclass(frozen=True)
class ModelWindow:
    """One trained window: a checkpoint over ``duration_seconds`` of the clip,
    starting ``offset_seconds`` into it."""

    index: int
    out_dir: Path
    checkpoint: Path
    dataset_dir: Path | None
    offset_seconds: float
    duration_seconds: float
    sh_degree: int


@dataclass(frozen=True)
class TrainedModel:
    """What ``CUMULI_MODEL`` carries between Train/Load Model and Bake SOGST."""

    root: Path
    fps: float
    windows: tuple[ModelWindow, ...]

    @property
    def duration_seconds(self) -> float:
        return max(w.offset_seconds + w.duration_seconds for w in self.windows) - self.windows[0].offset_seconds

    @classmethod
    def load(cls, path: Path | str) -> TrainedModel:
        """Open a window plan, a directory holding one, a single run directory
        (one with ``gs4d_config.yaml``), or a checkpoint inside such a run."""

        path = Path(path).expanduser()
        if path.is_file() and path.suffix == ".json":
            return cls._from_plan(path)
        if path.is_file() and path.suffix == ".pth":
            return cls._from_checkpoint(path)
        if path.is_dir():
            if (path / PLAN_NAME).is_file():
                return cls._from_plan(path / PLAN_NAME)
            if (path / CONFIG_NAME).is_file():
                return cls._from_run(path)
            raise ModelError(
                f"{path} holds neither {PLAN_NAME} nor {CONFIG_NAME}, so it is not a trained run. "
                "Point at the directory Train 4DGS wrote (its out_dir), a cumuli window plan, or a "
                "single cumuli run directory."
            )
        raise ModelError(f"{path} does not exist.")

    @classmethod
    def _from_plan(cls, plan_path: Path) -> TrainedModel:
        try:
            plan = json.loads(plan_path.read_text())
        except (OSError, ValueError) as exc:
            raise ModelError(f"{plan_path} is not a readable window plan: {exc}") from None
        try:
            fps = float(plan["fps"])
            entries = sorted(plan["windows"], key=lambda w: int(w["index"]))
        except (KeyError, TypeError, ValueError):
            raise ModelError(f"{plan_path} has no fps or windows list, so it is not a window plan.") from None
        missing = [w["index"] for w in entries if not w.get("out_dir")]
        if missing:
            raise ModelError(
                f"windows {missing} in {plan_path} carry no out_dir, so the plan does not say where "
                "their models are. A plan from plan_temporal_windows.py needs --out_dir_template, "
                "and its windows need training by run_window_plan.py."
            )
        if not entries:
            raise ModelError(f"{plan_path} lists no windows.")
        windows = []
        for entry in entries:
            out_dir = Path(entry["out_dir"]).expanduser()
            if not out_dir.is_absolute():
                out_dir = plan_path.parent / out_dir
            windows.append(_read_window(int(entry["index"]), out_dir, float(entry["offset_seconds"])))
        return cls(root=plan_path, fps=fps, windows=tuple(windows))

    @classmethod
    def _from_run(cls, run_dir: Path, checkpoint: Path | None = None) -> TrainedModel:
        window = _read_window(0, run_dir, 0.0, checkpoint=checkpoint)
        if window.dataset_dir is None:
            raise ModelError(
                f"{run_dir} has no window plan and its dataset is gone, so there is nothing to read "
                "the clip's frame rate from. Restore the dataset its gs4d_config.yaml names."
            )
        _, fps, _ = dataset_timeline(window.dataset_dir)
        return cls(root=run_dir, fps=fps, windows=(window,))

    @classmethod
    def _from_checkpoint(cls, checkpoint: Path) -> TrainedModel:
        # A window's own directory carries a trainer config just like a
        # whole-clip run does, so ask the enclosing plan first: loading one
        # window as if it were the clip is the silent slice this type exists
        # to prevent.
        for ancestor in list(checkpoint.parents)[:4]:
            plan = ancestor / PLAN_NAME
            if plan.is_file() and _plan_owns(plan, checkpoint):
                raise ModelError(
                    f"{checkpoint} is one window of the windowed run planned in {plan}. Load that "
                    "plan instead, and pick windows with Bake SOGST's windows widget."
                )
        # chkpnt<N>.pth sits in <run>/train4d_output/; the config is beside that.
        for run_dir in (checkpoint.parent.parent, checkpoint.parent):
            if (run_dir / CONFIG_NAME).is_file():
                return cls._from_run(run_dir, checkpoint=checkpoint)
        raise ModelError(
            f"no {CONFIG_NAME} beside {checkpoint} or its parent, so its duration and dataset are "
            "unknown. Load the run directory the trainer wrote."
        )

    def select(self, text: str) -> TrainedModel:
        """Keep the windows ``text`` names (``"0"``, ``"1-2"``, ``"0,1"``);
        empty keeps every window. The selection must be contiguous, since
        stitching across a missing window would leave a hole in the clip."""

        indices = [w.index for w in self.windows]
        chosen = parse_selection(text, indices)
        if chosen is None:
            return self
        positions = sorted(indices.index(i) for i in chosen)
        if positions != list(range(positions[0], positions[-1] + 1)):
            raise ModelError(
                f"windows {sorted(chosen)} are not contiguous; stitching them would leave a gap "
                "in the clip. Select a run of adjacent windows, e.g. '1-2'."
            )
        return replace(self, windows=tuple(self.windows[p] for p in positions))


def parse_selection(text: str, available: list[int]) -> tuple[int, ...] | None:
    """``"0,2"`` or ``"1-3"`` into window indices, the syntax Stage Ring's
    ``views`` widget uses; empty means every window."""

    text = (text or "").strip()
    if not text:
        return None
    selected: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part[1:]:
                low, _, high = part.partition("-")
                start, end = int(low), int(high)
                selected.extend(range(min(start, end), max(start, end) + 1))
            else:
                selected.append(int(part))
        except ValueError:
            raise ModelError(f"cannot read window selection {part!r}.") from None
    unknown = [i for i in selected if i not in available]
    if unknown:
        raise ModelError(f"this model has no window {unknown[0]} (it has {available[0]}..{available[-1]}).")
    return tuple(dict.fromkeys(selected))


def write_plan(
    root: Path, *, run: Path, fps: float, windows: list[tuple[int, int, int, Path]]
) -> Path:
    """Write ``root/window_plan.json`` for ``(index, frame_start, frame_count,
    out_dir)`` windows, in plan_temporal_windows.py's format for a uniform
    cut. The planner's own predictions (``coefficients``, ``predicted``) are
    left out: they describe its motion model, and nothing reads them back."""

    boundaries = [start for _, start, _, _ in windows] + [windows[-1][1] + windows[-1][2]]
    document = {
        "run": str(run),
        "fps": fps,
        "cut": "uniform",
        "signal": "none",
        "windows": [
            {
                "index": index,
                "frame_start": start,
                "frame_count": count,
                "local_frame_start": start,
                "offset_seconds": round(start / fps, 7),
                "out_dir": str(out_dir),
            }
            for index, start, count, out_dir in windows
        ],
        # Midway between the last frame of one window and the first of the
        # next, exactly as the planner and merge_sogst_segments.py place it.
        "seams_seconds": [round((b - 0.5) / fps, 7) for b in boundaries[1:-1]],
    }
    root.mkdir(parents=True, exist_ok=True)
    path = root / PLAN_NAME
    path.write_text(json.dumps(document, indent=2))
    return path


def dataset_timeline(dataset: Path) -> tuple[float, float, int]:
    """Clip duration, frame rate and timestamp count, read from the dataset."""

    path = dataset / "transforms_train.json"
    if not path.is_file():
        raise ModelError(f"{path} is missing; this is not a 4DGS dataset directory.")
    try:
        frames = json.loads(path.read_text()).get("frames", [])
    except ValueError as exc:
        raise ModelError(f"{path} is not valid JSON: {exc}") from None
    times = sorted({float(frame["time"]) for frame in frames if "time" in frame})
    if len(times) < 2:
        raise ModelError(f"{path} carries fewer than two distinct timestamps.")
    duration = times[-1]
    step = times[1] - times[0]
    return duration, (1.0 / step if step > 0 else 24.0), len(times)


def _plan_owns(plan_path: Path, checkpoint: Path) -> bool:
    """Whether ``checkpoint`` lies inside a window of a multi-window plan."""

    try:
        entries = json.loads(plan_path.read_text()).get("windows", [])
    except (OSError, ValueError, AttributeError):
        return False
    if len(entries) < 2:
        return False
    inside = set(checkpoint.resolve().parents)
    for entry in entries:
        out_dir = Path(entry.get("out_dir") or "").expanduser()
        if not out_dir.is_absolute():
            out_dir = plan_path.parent / out_dir
        if entry.get("out_dir") and out_dir.resolve() in inside:
            return True
    return False


def _read_window(index: int, out_dir: Path, offset: float, checkpoint: Path | None = None) -> ModelWindow:
    config_path = out_dir / CONFIG_NAME
    if not config_path.is_file():
        raise ModelError(
            f"window {index}: {config_path} is missing. It records the window's duration, dataset "
            "and checkpoint name; a run trained with a caller-supplied --trainer_config keeps it "
            "elsewhere, so copy that file into the window's directory."
        )
    try:
        config = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ModelError(f"window {index}: {config_path} is not readable YAML: {exc}") from None

    try:
        low, high = (float(t) for t in _lookup(config, "time_duration"))
        iterations = int(_lookup(config, "iterations"))
    except (KeyError, TypeError, ValueError):
        raise ModelError(
            f"window {index}: {config_path} has no time_duration or iterations, so it is not a "
            "4DGS trainer config."
        ) from None
    sh_degree = int(_lookup(config, "sh_degree", 3))

    # The config's paths are absolute; a run directory moved since training
    # keeps its contents under the standard names, so fall back to those.
    model_dir = _existing_dir(_lookup(config, "model_path", None)) or out_dir / MODEL_DIRNAME
    dataset_dir = _existing_dir(_lookup(config, "source_path", None)) or _existing_dir(out_dir / DATASET_DIRNAME)

    checkpoint = checkpoint or model_dir / f"chkpnt{iterations}.pth"
    if not checkpoint.is_file():
        raise ModelError(
            f"window {index}: {checkpoint} does not exist, so training did not finish there. "
            "Re-run Train 4DGS; it resumes from what is cached."
        )
    return ModelWindow(
        index=index,
        out_dir=out_dir,
        checkpoint=checkpoint,
        dataset_dir=dataset_dir,
        offset_seconds=offset,
        duration_seconds=high - low,
        sh_degree=sh_degree,
    )


def _lookup(node, key: str, default=KeyError):
    """``key``'s value anywhere in a parsed config. The trainer's template
    nests OptimizationParams under ModelParams by indentation, and which level
    a key sits at is the template's business, not this reader's."""

    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _lookup(value, key, None)
            if found is not None:
                return found
    if default is KeyError:
        raise KeyError(key)
    return default


def _existing_dir(value) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_dir() else None
