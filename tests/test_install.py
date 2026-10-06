# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""The installer's file-touching and repo-touching paths.

Nothing here reaches the network: the checkout tests clone from throwaway local
repositories over ``file://``, which exercises the same argv the real run uses.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import install  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is required")


def _repo(root: Path, name: str) -> str:
    """A one-commit local repository, standing in for a GitHub remote."""

    path = root / name
    path.mkdir(parents=True)
    (path / "README").write_text(name)
    for argv in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(path), *argv], check=True, capture_output=True)
    return f"file://{path}"


@pytest.fixture
def origins(tmp_path, monkeypatch):
    remotes = tmp_path / "remotes"
    # Local stand-ins keep the same shape, including each repo's pinned ref;
    # the throwaway repos are created on "main".
    monkeypatch.setattr(install, "CHECKOUTS", {
        key: (name, _repo(remotes, name), "main")
        for key, (name, _url, _ref) in install.CHECKOUTS.items()
    })
    return tmp_path / "deps"


def test_fetch_clones_every_checkout(origins):
    paths = install.fetch_checkouts(origins)
    assert set(paths) == {"fdanyone_root", "trainer_root", "cumuli_root"}
    for path in paths.values():
        assert (path / ".git").is_dir()


def test_fetch_is_idempotent(origins):
    first = install.fetch_checkouts(origins)
    before = {k: (v / ".git" / "HEAD").read_text() for k, v in first.items()}
    second = install.fetch_checkouts(origins)
    assert second == first
    assert {k: (v / ".git" / "HEAD").read_text() for k, v in second.items()} == before


def test_fetch_refuses_to_clobber_a_foreign_directory(origins):
    """A non-git directory in the way is never deleted; the user is told."""

    (origins / "OMG4").mkdir(parents=True)
    (origins / "OMG4" / "important.txt").write_text("not ours")
    with pytest.raises(install.InstallError) as exc:
        install.fetch_checkouts(origins)
    assert "--deps-dir" in str(exc.value)
    assert (origins / "OMG4" / "important.txt").is_file()


def test_revisions_report_what_was_fetched(origins):
    paths = install.fetch_checkouts(origins)
    revisions = install.checkout_revisions(paths)
    assert set(revisions) == set(paths)
    assert all(len(sha) == 10 for sha in revisions.values())


def test_write_config_merges_instead_of_replacing(tmp_path, monkeypatch):
    """A re-run must not discard settings the user already tuned."""

    config = tmp_path / "config.json"
    config.write_text(json.dumps({"min_free_vram_gb": 29.0, "dataset_roots": ["/keep/me"]}))
    monkeypatch.setattr(install, "CONFIG_FILE", config)
    install.write_config({"fdanyone_root": tmp_path / "a", "cumuli_root": tmp_path / "b"}, None)
    written = json.loads(config.read_text())
    assert written["min_free_vram_gb"] == 29.0
    assert written["dataset_roots"] == ["/keep/me"]
    assert written["fdanyone_root"] == str(tmp_path / "a")


def test_write_config_records_the_work_root_when_given(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    monkeypatch.setattr(install, "CONFIG_FILE", config)
    install.write_config({"cumuli_root": tmp_path}, "/big/drive")
    assert json.loads(config.read_text())["work_root"] == "/big/drive"


def test_write_config_survives_an_unreadable_existing_file(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text("{ not json")
    monkeypatch.setattr(install, "CONFIG_FILE", config)
    install.write_config({"cumuli_root": tmp_path}, None)
    assert json.loads(config.read_text())["cumuli_root"] == str(tmp_path)


@pytest.mark.parametrize("before,after,expected", [
    # The failure this guard exists for: something moved under ComfyUI.
    ({"torch": "2.13.0"}, {"torch": "2.4.0"}, ["torch: 2.13.0 -> 2.4.0"]),
    ({"torch": "2.13.0"}, {"torch": None}, ["torch: 2.13.0 -> removed"]),
    # Not failures: unchanged, or pulled in fresh as a dependency.
    ({"torch": "2.13.0"}, {"torch": "2.13.0"}, []),
    ({"numpy": None}, {"numpy": "2.5.3"}, []),
    ({"numpy": None}, {"numpy": None}, []),
])
def test_only_real_drift_fails_the_run(before, after, expected):
    assert install.compare_pinned(before, after) == expected


def test_the_sam3d_weights_are_reported_missing_until_they_are_placed(tmp_path):
    """The one asset no installer may fetch (Meta's gated SAM License), so it is named explicitly."""

    comfy = tmp_path / "ComfyUI"
    weights = comfy / "models" / "detection" / install.SAM3D_WEIGHTS
    assert install.missing_manual_assets({"comfyui_root": comfy}) == [str(weights)]
    weights.parent.mkdir(parents=True)
    weights.write_text("")
    assert install.missing_manual_assets({"comfyui_root": comfy}) == []


def test_without_a_comfyui_root_the_weights_are_named_not_checked():
    (message,) = install.missing_manual_assets({})
    assert message == f"ComfyUI/models/detection/{install.SAM3D_WEIGHTS}"


def test_neither_smplx_nor_ultralytics_is_installed_or_guarded():
    """GVHMR's dependencies are gone; reintroducing them would bring back their licences."""

    groups = install.build_groups("13")
    requirements = [req for group in groups.values() for req, _dist in group.requirements]
    assert not any(req.startswith(("smplx", "ultralytics")) for req in requirements)
    assert "ultralytics" not in install.PINNED


def test_each_checkout_is_cloned_at_its_own_pinned_ref(tmp_path, monkeypatch):
    """The pin is per repository, so one can move without dragging the others."""

    remotes = tmp_path / "remotes"
    url = _repo(remotes, "pinned")
    repo = remotes / "pinned"
    # A second commit on main, with the tag left behind on the first.
    subprocess.run(["git", "-C", str(repo), "tag", "v0.0.1"], check=True, capture_output=True)
    (repo / "README").write_text("moved on")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "second"], check=True, capture_output=True)

    monkeypatch.setattr(install, "CHECKOUTS", {"cumuli_root": ("pinned", url, "v0.0.1")})
    paths = install.fetch_checkouts(tmp_path / "deps")
    assert (paths["cumuli_root"] / "README").read_text() == "pinned"


def test_ref_override_beats_the_pin(tmp_path, monkeypatch):
    remotes = tmp_path / "remotes"
    url = _repo(remotes, "pinned")
    repo = remotes / "pinned"
    subprocess.run(["git", "-C", str(repo), "tag", "v0.0.1"], check=True, capture_output=True)
    (repo / "README").write_text("moved on")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "second"], check=True, capture_output=True)

    monkeypatch.setattr(install, "CHECKOUTS", {"cumuli_root": ("pinned", url, "v0.0.1")})
    paths = install.fetch_checkouts(tmp_path / "deps", ref="main")
    assert (paths["cumuli_root"] / "README").read_text() == "moved on"


def _commit_second(repo: Path) -> str:
    """Move the repo's branch on by one commit; return the *first* commit's hash."""

    first = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                           capture_output=True, text=True).stdout.strip()
    (repo / "README").write_text("moved on")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "second"], check=True, capture_output=True)
    return first


def test_a_full_commit_hash_pin_clones_exactly_that_commit(tmp_path, monkeypatch):
    """A pin must keep meaning the same code even after the branch moves on."""

    remotes = tmp_path / "remotes"
    url = _repo(remotes, "pinned")
    first = _commit_second(remotes / "pinned")
    monkeypatch.setattr(install, "CHECKOUTS", {"fdanyone_root": ("pinned", url, first)})
    paths = install.fetch_checkouts(tmp_path / "deps")
    clone = paths["fdanyone_root"]
    assert (clone / "README").read_text() == "pinned"
    head = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == first


def test_only_a_full_forty_character_hash_counts_as_a_commit_pin():
    assert install._is_commit_hash("6d5ec422ba4a4eef48f05c18ca33a9d4e7ca8d33")
    for ref in ("v0.0.1", "main", "f7af869", "f7af8697b282a3106e395b19ca8004dd35c2087z"):
        assert not install._is_commit_hash(ref)


def test_the_4danyone_pin_is_a_full_commit_hash():
    """The fork's SAM 3D Body branch is not tagged yet; a branch name would not stay pinned."""

    _name, url, ref = install.CHECKOUTS["fdanyone_root"]
    assert url.endswith("solipsist-studios/4DAnyone.git")
    assert install._is_commit_hash(ref)


@pytest.mark.parametrize("with_turbo, expected", [(False, "False"), (True, "True")])
def test_the_turbo_lora_is_only_downloaded_when_asked_for(tmp_path, monkeypatch, with_turbo, expected):
    """The Turbo adapter is CC BY-NC-SA 4.0, so a default install must not fetch it."""

    root = tmp_path / "4DAnyone"
    (root / "fdanyone").mkdir(parents=True)
    (root / "fdanyone" / "download.py").write_text("")
    calls = []
    monkeypatch.setattr(install, "_run", lambda argv, **kwargs: calls.append(argv))
    install.fetch_models({"fdanyone_root": root}, with_turbo=with_turbo)
    (argv,) = calls
    assert "enable_turbo=sys.argv[3] == 'True'" in argv[2]    # the -c program reads the flag
    assert argv[-1] == expected


def test_dry_run_clones_nothing(origins):
    paths = install.fetch_checkouts(origins, dry_run=True)
    assert not any(path.exists() for path in paths.values())
