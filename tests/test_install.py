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
    monkeypatch.setattr(install, "CHECKOUTS", {
        key: (name, _repo(remotes, name))
        for key, (name, _) in install.CHECKOUTS.items()
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


def test_smplx_is_reported_missing_until_it_is_placed(tmp_path):
    """The one asset no installer may fetch, so it must be named explicitly."""

    root = tmp_path / "4DAnyone"
    smplx = root / "models" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    assert install.missing_manual_assets({"fdanyone_root": root}) == [str(smplx)]
    smplx.parent.mkdir(parents=True)
    smplx.write_text("")
    assert install.missing_manual_assets({"fdanyone_root": root}) == []


def test_dry_run_clones_nothing(origins):
    paths = install.fetch_checkouts(origins, dry_run=True)
    assert not any(path.exists() for path in paths.values())
