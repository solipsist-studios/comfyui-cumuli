# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Locate the external checkouts this pack drives: 4DAnyone, OMG4, cumuli.

Everything runs in ComfyUI's own environment. Two stages still run as **child
processes of the same interpreter**, for reasons that are about CUDA rather than
about dependencies:

* Ring generation needs ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``,
  which only takes effect before a process makes its first CUDA allocation --
  long done by the time a ComfyUI node runs -- and it peaks near 31.9 GB of a
  32 GB card, so it needs an allocator that starts clean.
* Training is a multi-hour job whose CUDA extensions are happier owning the
  device, and running it out-of-process keeps a crash from taking the server
  with it.

``python_exe`` therefore defaults to ``sys.executable``: same environment, new
process. Setting ``conda_env`` and clearing ``python_exe`` restores the old
behaviour of launching a *different* environment through ``conda run``, which is
still supported for anyone who has not merged the two.

Nothing here is machine specific by construction: every value has a default that
matches the reference workstation, and every value can be overridden by an
environment variable or by a JSON config file.

Resolution order (last wins):

1. Built-in defaults below.
2. ComfyUI's per-user settings store (``user/<user>/comfy.settings.json``,
   keys ``cumuli.<name>``) -- the values the Settings dialog edits.
3. JSON config file -- ``$CUMULI_CONFIG`` if set, otherwise
   ``<package>/config.json``.
4. Environment variables (``CUMULI_FDANYONE_ROOT``, ``CUMULI_TRAINER_ROOT``, ...).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("comfyui-cumuli")

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

#: Environment given to the inference subprocess on top of the inherited one.
#: These three entries are the empirically required settings for a 32 GB card.
#:
#: ``FDANYONE_ATTENTION_BACKEND`` matters specifically because we run inside
#: ComfyUI's environment, where ``sageattention`` is installed for ComfyUI's
#: own use. 4DAnyone's auto policy ranks backends by speed and would pick it,
#: but this pipeline is memory-bound: measured on an RTX 5090 at the shapes
#: this model uses, sageattn peaks at exactly 2x SDPA's memory (it holds INT8
#: copies of q and k plus a smoothed k), while torch SDPA already dispatches
#: to the flash kernel. Pinning SDPA gives back ~1.9 GiB -- most of what RCP
#: costs -- and is bit-identical to the reference path, where sageattn's INT8
#: quantisation carries a ~1.3% relative error.
DEFAULT_SUBPROCESS_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "FDANYONE_PERSISTENT_PARAMS": "1e9",
    "FDANYONE_ATTENTION_BACKEND": "sdpa",
}

DEFAULTS: dict[str, object] = {
    "fdanyone_root": "~/Dev/github/4DAnyone",
    "conda_env": "",  # empty -> stay in this interpreter's environment
    "conda_exe": "",  # empty -> auto-discover on PATH / common install roots
    "python_exe": "",  # empty -> sys.executable, unless conda_env is set
    "data_dir": "data",  # relative paths are resolved against fdanyone_root
    "model_dir": "models",
    "gvhmr_root": "third_party/GVHMR",
    "device": "cuda:0",
    "min_free_vram_gb": 30.0,
    # The rotor 4DGS trainer (OMG4), vendored by the cumuli pipeline.
    "trainer_root": "~/Dev/github/cumuli/deps/OMG4",
    # Where the heavy per-run intermediates (flipbook staging, 4DGS dataset,
    # trainer checkpoints) go when the node's dir widget is left empty. Empty
    # keeps the old behaviour: ComfyUI's temp (staging) and output (dataset).
    # Point it at a large drive; each run gets its own subdirectory.
    "work_root": "",
    # Extra directories the loader nodes scan for existing artifacts, on top
    # of work_root. Each root is checked itself, one and two levels deep
    # (root/<x> and root/<x>/dataset_4dgs), so both flat exports and per-run
    # trees are found.
    "dataset_roots": [],
    "flipbook_roots": [],
    "ring_roots": [],
    "trainer_script": "train_scratch.py",
    "subprocess_env": dict(DEFAULT_SUBPROCESS_ENV),
    "trainer_env": {},
}

_ENV_KEYS = {
    "fdanyone_root": ("CUMULI_FDANYONE_ROOT", "CUMULI_FDANYONE_REPO"),
    "conda_env": ("CUMULI_CONDA_ENV",),
    "conda_exe": ("CUMULI_CONDA_EXE",),
    "python_exe": ("CUMULI_PYTHON",),
    "data_dir": ("CUMULI_DATA_DIR",),
    "model_dir": ("CUMULI_MODEL_DIR",),
    "gvhmr_root": ("CUMULI_GVHMR_ROOT",),
    "device": ("CUMULI_DEVICE",),
    "min_free_vram_gb": ("CUMULI_MIN_FREE_VRAM_GB",),
    "trainer_root": ("CUMULI_TRAINER_ROOT",),
    "trainer_script": ("CUMULI_TRAINER_SCRIPT",),
    "work_root": ("CUMULI_WORK_ROOT",),
}

_CONDA_CANDIDATES = (
    "~/miniconda3/bin/conda",
    "~/miniforge3/bin/conda",
    "~/anaconda3/bin/conda",
    "~/mambaforge/bin/conda",
    "/opt/conda/bin/conda",
)


class SettingsError(RuntimeError):
    """Raised when the bridge cannot be configured well enough to run."""


def _config_path() -> Path | None:
    explicit = os.environ.get("CUMULI_CONFIG")
    if explicit:
        return Path(explicit).expanduser()
    candidate = PACKAGE_ROOT / "config.json"
    return candidate if candidate.is_file() else None


def _comfy_settings_file() -> Path | None:
    """ComfyUI's per-user settings store, when running inside ComfyUI."""

    try:
        import folder_paths  # the one intended comfy import here, like vram.py

        return Path(folder_paths.get_user_directory()) / "default" / "comfy.settings.json"
    except Exception:
        return None


def _load_comfy_settings_overrides() -> dict:
    """Values the user edits in ComfyUI's own Settings dialog (cumuli.*)."""

    path = _comfy_settings_file()
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read %s (%s); ignoring UI settings.", path, exc)
        return {}
    out = {}
    for key in DEFAULTS:
        value = payload.get(f"cumuli.{key}")
        if value not in (None, ""):
            out[key] = value
    return out


def _load_file_overrides() -> dict:
    path = _config_path()
    if path is None:
        return {}
    if not path.is_file():
        LOGGER.warning("CUMULI_CONFIG points at %s which does not exist; using defaults.", path)
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        LOGGER.warning("Could not read the cumuli bridge config %s (%s); using defaults.", path, exc)
        return {}
    if not isinstance(payload, dict):
        LOGGER.warning("The cumuli bridge config %s is not a JSON object; using defaults.", path)
        return {}
    unknown = set(payload) - set(DEFAULTS)
    if unknown:
        LOGGER.warning("Ignoring unknown keys in %s: %s", path, ", ".join(sorted(unknown)))
    return {key: value for key, value in payload.items() if key in DEFAULTS}


def _roots(value) -> tuple[str, ...]:
    """Roots come as a JSON list (config file) or a comma-separated string
    (the Settings dialog's text field)."""

    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(str(root) for root in (value or []))


def _resolve_under(root: Path, value: str) -> Path:
    candidate = Path(str(value)).expanduser()
    return candidate if candidate.is_absolute() else (root / candidate)


@dataclass(frozen=True)
class BridgeSettings:
    """Everything needed to spawn this pack's stages as child processes."""

    fdanyone_root: Path
    conda_env: str
    conda_exe: str
    python_exe: str
    data_dir: Path
    model_dir: Path
    gvhmr_root: Path
    device: str
    min_free_vram_gb: float
    work_root: str = ""
    dataset_roots: tuple[str, ...] = ()
    flipbook_roots: tuple[str, ...] = ()
    ring_roots: tuple[str, ...] = ()
    trainer_root: Path = Path("~/Dev/github/cumuli/deps/OMG4").expanduser()
    trainer_script: str = "train_scratch.py"
    subprocess_env: dict[str, str] = field(default_factory=dict)
    trainer_env: dict[str, str] = field(default_factory=dict)

    # -- discovery ---------------------------------------------------------
    @classmethod
    def load(cls) -> BridgeSettings:
        values = dict(DEFAULTS)
        values["subprocess_env"] = dict(DEFAULT_SUBPROCESS_ENV)
        values.update(_load_comfy_settings_overrides())
        values.update(_load_file_overrides())

        for key, names in _ENV_KEYS.items():
            for name in names:
                raw = os.environ.get(name)
                if raw:
                    values[key] = raw
                    break

        extra = os.environ.get("CUMULI_SUBPROCESS_ENV")
        if extra:
            try:
                parsed = json.loads(extra)
            except ValueError as exc:
                raise SettingsError(f"CUMULI_SUBPROCESS_ENV is not valid JSON: {exc}") from None
            if not isinstance(parsed, dict):
                raise SettingsError("CUMULI_SUBPROCESS_ENV must be a JSON object.")
            merged = dict(values.get("subprocess_env") or {})
            merged.update({str(k): str(v) for k, v in parsed.items()})
            values["subprocess_env"] = merged

        fdanyone_root = Path(str(values["fdanyone_root"])).expanduser()
        try:
            min_free = float(values["min_free_vram_gb"])
        except (TypeError, ValueError):
            raise SettingsError(f"min_free_vram_gb must be a number, got {values['min_free_vram_gb']!r}.") from None

        return cls(
            fdanyone_root=fdanyone_root,
            conda_env=str(values["conda_env"]),
            conda_exe=str(values["conda_exe"]),
            python_exe=str(values["python_exe"]),
            data_dir=_resolve_under(fdanyone_root, str(values["data_dir"])),
            model_dir=_resolve_under(fdanyone_root, str(values["model_dir"])),
            gvhmr_root=_resolve_under(fdanyone_root, str(values["gvhmr_root"])),
            device=str(values["device"]),
            min_free_vram_gb=min_free,
            work_root=str(values.get("work_root") or ""),
            dataset_roots=_roots(values.get("dataset_roots")),
            flipbook_roots=_roots(values.get("flipbook_roots")),
            ring_roots=_roots(values.get("ring_roots")),
            trainer_root=Path(str(values["trainer_root"])).expanduser(),
            trainer_script=str(values["trainer_script"]),
            subprocess_env={str(k): str(v) for k, v in (values.get("subprocess_env") or {}).items()},
            trainer_env={str(k): str(v) for k, v in (values.get("trainer_env") or {}).items()},
        )

    # -- derived paths -----------------------------------------------------
    def work_dir(self, run_name: str) -> Path | None:
        """Per-run directory for heavy intermediates, or None when unset."""

        text = (self.work_root or "").strip()
        if not text:
            return None
        return Path(text).expanduser() / run_name

    @property
    def inference_script(self) -> Path:
        return self.fdanyone_root / "inference.py"

    @property
    def results_root(self) -> Path:
        return self.data_dir / "fdanyone"

    @property
    def motion_root(self) -> Path:
        return self.data_dir / "gvhmr" / "results"

    @property
    def source_root(self) -> Path:
        return self.data_dir / "source"

    def result_dir(self, run_name: str) -> Path:
        return self.results_root / run_name

    def motion_dir(self, run_name: str) -> Path:
        return self.motion_root / run_name

    # -- process launching -------------------------------------------------
    def find_conda(self) -> str:
        if self.conda_exe:
            resolved = Path(self.conda_exe).expanduser()
            if not resolved.is_file():
                raise SettingsError(f"Configured conda executable does not exist: {resolved}")
            return str(resolved)
        found = shutil.which("conda") or shutil.which("mamba") or shutil.which("micromamba")
        if found:
            return found
        for candidate in _CONDA_CANDIDATES:
            path = Path(candidate).expanduser()
            if path.is_file():
                return str(path)
        raise SettingsError(
            "Could not find a conda executable. Set CUMULI_CONDA_EXE or add conda to PATH, "
            "or set CUMULI_PYTHON to the interpreter of the 4DAnyone environment."
        )

    def launcher(self) -> list[str]:
        """Return the argv prefix that starts a child python process.

        By default that is *this* interpreter: 4DAnyone and the trainer both run
        in ComfyUI's own environment, and the child exists only so the CUDA
        allocator can be configured before its first allocation. Setting
        ``conda_env`` (and leaving ``python_exe`` empty) launches a different
        environment through ``conda run`` instead.
        """

        if self.python_exe:
            python = Path(self.python_exe).expanduser()
            if not python.is_file():
                raise SettingsError(f"Configured CUMULI_PYTHON does not exist: {python}")
            return [str(python)]
        if self.conda_env:
            return [self.find_conda(), "run", "--no-capture-output", "-n", self.conda_env, "python"]
        return [sys.executable]


    @property
    def runs_in_this_environment(self) -> bool:
        return not self.conda_env and not self.python_exe

    @property
    def trainer_entrypoint(self) -> Path:
        return self.trainer_root / self.trainer_script

    def validate(self) -> None:
        """Fail early, with a message that names the knob that fixes the problem."""

        if not self.fdanyone_root.is_dir():
            raise SettingsError(
                f"4DAnyone checkout not found at {self.fdanyone_root}. "
                "Set CUMULI_FDANYONE_ROOT or the 'fdanyone_root' key of the bridge config file."
            )
        if not self.inference_script.is_file():
            raise SettingsError(f"{self.inference_script} is missing; {self.fdanyone_root} is not a 4DAnyone checkout.")
        # Touching launcher() here surfaces a missing conda before any GPU work.
        self.launcher()

    def validate_trainer(self) -> None:
        if not self.trainer_entrypoint.is_file():
            raise SettingsError(
                f"Trainer entry point not found at {self.trainer_entrypoint}. "
                "Set CUMULI_TRAINER_ROOT or the 'trainer_root' key of the bridge config file "
                "(it should point at an OMG4 checkout, for example cumuli's deps/OMG4 submodule)."
            )

    def describe(self) -> str:
        environment = self.python_exe or self.conda_env or "this interpreter"
        return (
            f"fdanyone_root={self.fdanyone_root} env={environment} "
            f"data_dir={self.data_dir} device={self.device}"
        )


def load_settings() -> BridgeSettings:
    """Read settings fresh on every call so edits take effect without a restart."""

    return BridgeSettings.load()
