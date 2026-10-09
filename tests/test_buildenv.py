# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Finding the compiler gsplat's kernels are built with, without a special shell.

Windows is exercised through fakes of ``vswhere`` and ``vcvars64.bat`` (a real Visual Studio is
not needed to check that their output is read and merged correctly).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cumuli_bridge import buildenv  # noqa: E402


def _toolkit(root: Path, name: str, nvcc: str = "nvcc") -> Path:
    (root / name / "bin").mkdir(parents=True)
    (root / name / "bin" / nvcc).write_text("")
    return root / name


def test_the_newest_installed_cuda_wins(tmp_path):
    _toolkit(tmp_path, "cuda-12.0")
    _toolkit(tmp_path, "cuda-13.2")
    env = buildenv.toolchain_env({"PATH": "/usr/bin"}, system="Linux", cuda_roots=[tmp_path])
    assert env["CUDA_HOME"].endswith("cuda-13.2")
    assert env["PATH"].startswith(str(tmp_path / "cuda-13.2" / "bin") + ":")


def test_versions_sort_numerically_not_as_text(tmp_path):
    _toolkit(tmp_path, "cuda-9.2")
    _toolkit(tmp_path, "cuda-13.0")
    assert buildenv.find_cuda_home({}, system="Linux", roots=[tmp_path]).name == "cuda-13.0"


def test_an_explicit_cuda_home_is_never_overridden(tmp_path):
    home = _toolkit(tmp_path, "mine")
    _toolkit(tmp_path, "cuda-13.2")
    assert buildenv.toolchain_env({"CUDA_HOME": str(home)}, system="Linux", cuda_roots=[tmp_path]) == {}


def test_cuda_path_is_honoured_too(tmp_path):
    home = _toolkit(tmp_path, "mine")
    assert buildenv.find_cuda_home({"CUDA_PATH": str(home)}, system="Linux", roots=[]) == home


def test_nothing_found_adds_nothing(tmp_path):
    assert buildenv.toolchain_env({"PATH": str(tmp_path)}, system="Linux", cuda_roots=[tmp_path]) == {}


def test_the_windows_toolkit_folder_and_exe_name_are_used(tmp_path):
    _toolkit(tmp_path, "v12.4", nvcc="nvcc.exe")
    _toolkit(tmp_path, "v13.0", nvcc="nvcc.exe")
    nothing_changed = SimpleNamespace(returncode=0, stdout="")
    env = buildenv.toolchain_env(
        {"PATH": "C:/x"}, system="Windows", cuda_roots=[tmp_path], vcvars=Path("vcvars64.bat"),
        run=lambda *a, **k: nothing_changed,
    )
    assert env["CUDA_HOME"].endswith("v13.0")
    assert env["PATH"].endswith(";C:/x")  # the Windows path separator


# --- Windows: Microsoft C++ ---------------------------------------------------------

def test_set_output_is_parsed_the_way_cmd_prints_it():
    text = "PATH=C:\\a;C:\\b\nINCLUDE=C:\\inc\n=C:=C:\\hidden\nNOTAVARIABLE\nEMPTY=\n"
    assert buildenv.parse_set_output(text) == {"PATH": "C:\\a;C:\\b", "INCLUDE": "C:\\inc", "EMPTY": ""}


def test_only_what_vcvars_changed_is_returned():
    fake = SimpleNamespace(returncode=0, stdout="Path=C:\\vc;C:\\a\nINCLUDE=C:\\inc\nHOME=C:\\h\n")
    before = {"PATH": "C:\\a", "HOME": "C:\\h"}
    env = buildenv.msvc_env(Path("vcvars64.bat"), before, run=lambda *a, **k: fake)
    # Case-insensitively the same variable as PATH; HOME is unchanged and left out.
    assert env == {"Path": "C:\\vc;C:\\a", "INCLUDE": "C:\\inc"}


def test_vcvars_is_run_against_the_environment_it_is_asked_to_extend():
    seen = {}

    def run(cmd, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="")

    buildenv.msvc_env(Path("vcvars64.bat"), {"PATH": "C:\\cuda\\bin"}, run=run)
    assert seen["env"] == {"PATH": "C:\\cuda\\bin"}


def test_a_failing_vcvars_is_an_error_not_an_empty_environment():
    import pytest

    with pytest.raises(OSError):
        buildenv.msvc_env(Path("vcvars64.bat"), {}, run=lambda *a, **k: SimpleNamespace(returncode=1, stdout=""))


def test_vswhere_is_asked_for_the_cpp_tools_and_its_answer_becomes_vcvars(tmp_path):
    vswhere = tmp_path / "vswhere.exe"
    vswhere.write_text("")
    install = tmp_path / "VS"
    vcvars = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    vcvars.parent.mkdir(parents=True)
    vcvars.write_text("")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=f"{install}\n")

    assert buildenv.find_vcvars(vswhere=vswhere, run=run) == vcvars
    assert "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" in calls[0]


def test_no_vswhere_or_no_cpp_tools_means_no_vcvars(tmp_path):
    assert buildenv.find_vcvars(vswhere=tmp_path / "missing.exe") is None
    vswhere = tmp_path / "vswhere.exe"
    vswhere.write_text("")
    empty = SimpleNamespace(returncode=0, stdout="")
    assert buildenv.find_vcvars(vswhere=vswhere, run=lambda *a, **k: empty) is None

    def broken(*a, **k):
        raise subprocess.SubprocessError("boom")

    assert buildenv.find_vcvars(vswhere=vswhere, run=broken) is None


def test_windows_gets_the_compiler_environment_when_cl_is_not_on_path(tmp_path):
    cuda = _toolkit(tmp_path, "v13.0", nvcc="nvcc.exe")
    fake = SimpleNamespace(returncode=0, stdout="PATH=C:\\vc;C:\\cuda\\bin\nINCLUDE=C:\\inc\n")
    env = buildenv.toolchain_env(
        {"PATH": "C:\\bin"}, system="Windows", cuda_roots=[tmp_path], vcvars=Path("vcvars64.bat"),
        run=lambda *a, **k: fake,
    )
    assert env["CUDA_HOME"] == str(cuda)
    assert env["INCLUDE"] == "C:\\inc"
    assert env["PATH"] == "C:\\vc;C:\\cuda\\bin"  # the compiler's PATH, which already carries CUDA's


def test_windows_with_cl_already_on_path_is_left_alone(tmp_path):
    cl = tmp_path / "cl.exe"
    cl.write_text("")
    cl.chmod(0o755)
    env = buildenv.toolchain_env({"PATH": str(tmp_path)}, system="Windows", cuda_roots=[tmp_path / "none"])
    assert env == {}


def test_windows_without_visual_studio_warns_and_carries_on(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(buildenv, "find_vcvars", lambda **k: None)
    with caplog.at_level("WARNING", logger="comfyui-cumuli"):
        env = buildenv.toolchain_env({"PATH": str(tmp_path)}, system="Windows", cuda_roots=[tmp_path / "none"])
    assert env == {}
    assert "Build Tools for Visual Studio" in caplog.text


def test_linux_never_looks_for_a_microsoft_compiler(tmp_path, monkeypatch):
    monkeypatch.setattr(buildenv, "find_vcvars", lambda **k: (_ for _ in ()).throw(AssertionError("looked")))
    buildenv.toolchain_env({"PATH": str(tmp_path)}, system="Linux", cuda_roots=[tmp_path / "none"])
