# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""The compiler environment gsplat's CUDA kernels are built in.

gsplat compiles its kernels the first time they are used. That needs two things on
the path of the process doing it: an ``nvcc`` that knows the installed GPU, and, on
Windows, Microsoft's C++ compiler (``cl.exe``). Neither should be something the
user has to arrange by starting ComfyUI from a particular prompt, so this module
finds them:

* the installer builds the kernels once, with ``toolchain_env()`` applied, and
* Train 4DGS applies the same environment to the trainer's process, so a build
  that was skipped or has gone stale still happens, just later.

Plain Python, no ComfyUI imports, so the installer can use it.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path

LOGGER = logging.getLogger("comfyui-cumuli")

#: Where a Windows CUDA toolkit installer puts itself.
_WINDOWS_CUDA_ROOT = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"


def _nvcc_name(system: str) -> str:
    return "nvcc.exe" if system == "Windows" else "nvcc"


def _version_key(path: Path) -> list[int]:
    return [int(part) for part in re.findall(r"\d+", path.name)]


def find_cuda_home(
    environ: Mapping[str, str] | None = None,
    *,
    system: str | None = None,
    roots: list[Path] | None = None,
) -> Path | None:
    """A CUDA toolkit with an nvcc, newest first.

    ``CUDA_HOME``/``CUDA_PATH`` win when they point at a real toolkit. After that
    the newest installed one (``/usr/local/cuda-*``, or the Windows installer's
    folder). A bare ``nvcc`` on PATH is the last resort, because a distro
    ``/usr/bin/nvcc`` is frequently too old to target a current GPU.
    """

    environ = os.environ if environ is None else environ
    system = system or platform.system()
    nvcc = _nvcc_name(system)

    for name in ("CUDA_HOME", "CUDA_PATH"):
        explicit = environ.get(name)
        if explicit and (Path(explicit) / "bin" / nvcc).is_file():
            return Path(explicit)

    if roots is None:
        roots = [Path("/usr/local")] if system != "Windows" else [_WINDOWS_CUDA_ROOT]
    pattern = "cuda-*" if system != "Windows" else "v*"
    for base in roots:
        for candidate in sorted(base.glob(pattern), key=_version_key, reverse=True):
            if (candidate / "bin" / nvcc).is_file():
                return candidate

    found = shutil.which("nvcc", path=environ.get("PATH"))
    return Path(found).resolve().parent.parent if found else None


def find_vcvars(*, vswhere: Path | None = None, run: Callable | None = None) -> Path | None:
    """``vcvars64.bat`` of the newest Visual Studio (or Build Tools) install that has the C++ tools.

    Found through ``vswhere``, which the Visual Studio installer puts at a fixed
    path for exactly this purpose.
    """

    run = run or subprocess.run
    if vswhere is None:
        base = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles") or r"C:\Program Files (x86)"
        vswhere = Path(base) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not Path(vswhere).is_file():
        return None
    try:
        result = run(
            [str(vswhere), "-latest", "-products", "*", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    root = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not root:
        return None
    vcvars = Path(root[0].strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    return vcvars if vcvars.is_file() else None


def parse_set_output(text: str) -> dict[str, str]:
    """``NAME=value`` lines, as ``cmd``'s ``set`` prints them."""

    env: dict[str, str] = {}
    for line in text.splitlines():
        name, sep, value = line.partition("=")
        if sep and name and not name.startswith("="):
            env[name] = value
    return env


def msvc_env(vcvars: Path, environ: Mapping[str, str], *, run: Callable | None = None) -> dict[str, str]:
    """The variables ``vcvars64.bat`` would add or change, as a dict to merge into a child's environment.

    Only the differences from ``environ`` are returned, so the child still inherits
    everything else.
    """

    run = run or subprocess.run
    result = run(
        f'"{vcvars}" >nul 2>&1 && set',
        shell=True, capture_output=True, text=True, timeout=120, env=dict(environ),
    )
    if result.returncode != 0:
        raise OSError(f"{vcvars} failed (status {result.returncode}).")
    after = parse_set_output(result.stdout or "")
    # Windows variable names are case-insensitive; compare the way it does.
    before = {name.upper(): value for name, value in environ.items()}
    return {name: value for name, value in after.items() if before.get(name.upper()) != value}


def toolchain_env(
    environ: Mapping[str, str] | None = None,
    *,
    system: str | None = None,
    cuda_roots: list[Path] | None = None,
    vcvars: Path | None = None,
    run: Callable | None = None,
) -> dict[str, str]:
    """What to add to a child's environment so gsplat can build its kernels.

    Empty when the process is already set up (an explicit ``CUDA_HOME`` and, on
    Windows, ``cl.exe`` on PATH), so a user who prepared the environment by hand is
    never overridden.
    """

    environ = os.environ if environ is None else environ
    system = system or platform.system()
    add: dict[str, str] = {}

    cuda = find_cuda_home(environ, system=system, roots=cuda_roots)
    if cuda is not None and not environ.get("CUDA_HOME"):
        sep = ";" if system == "Windows" else ":"
        add["CUDA_HOME"] = str(cuda)
        add["PATH"] = f"{cuda / 'bin'}{sep}{environ.get('PATH', '')}"

    if system == "Windows" and shutil.which("cl", path=environ.get("PATH")) is None:
        vcvars = vcvars or find_vcvars(run=run)
        if vcvars is None:
            LOGGER.warning(
                "No Microsoft C++ compiler found (cl.exe is not on PATH and vswhere sees no Visual Studio "
                "with the C++ tools). Install 'Build Tools for Visual Studio' with the 'Desktop development "
                "with C++' workload; gsplat needs it to build its CUDA kernels."
            )
        else:
            try:
                msvc = msvc_env(vcvars, {**environ, **add}, run=run)
            except (OSError, subprocess.SubprocessError) as exc:
                LOGGER.warning("Could not load the C++ compiler environment from %s: %s", vcvars, exc)
            else:
                add.update(msvc)
    return add


#: What triggers gsplat's build. Importing the backend compiles the kernels (or loads the cached ones).
BUILD_SNIPPET = "from gsplat.cuda._backend import _C"
