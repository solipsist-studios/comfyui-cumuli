#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""Install this pack's additive dependencies into ComfyUI's own environment.

Everything the README's Requirements section lists, in one idempotent run. Use
the wrappers rather than calling this directly -- they find the interpreter:

    ./install.sh            # Linux / macOS
    install.bat             # Windows

The interpreter matters more than anything else here. Every install goes into
*this* interpreter's environment (``sys.executable``), because the whole pack is
built on one environment: ComfyUI's. Running it with the wrong python installs a
working set of packages somewhere ComfyUI will never look.

Four groups, all installed by default (``--groups core,bake,sfm,trainer``):

  core     4DAnyone's runtime dependencies
  bake     the ``.sogst`` container writer
  sfm      the rig solve behind Solve Rig (hloc, pycolmap, lightglue)
  trainer  the 4DGS trainer: gsplat, cumuli-core and cumuli-trainer

Two hazards this script exists to defuse, both documented in the README and both
silent when they happen by hand:

* **pip upgrading torch out from under ComfyUI.** Several of these packages
  declare pinned dependencies older than a working ComfyUI carries. The affected
  installs pass ``--no-deps``, and the versions of torch, numpy, transformers,
  and timm are recorded before and compared after: if anything moved,
  the run fails loudly instead of leaving a broken ComfyUI to discover later.
* **The trainer's CUDA kernels need a compiler that knows the GPU.** gsplat
  compiles them on first use, which would otherwise be a surprise of several
  minutes in the middle of a graph. The installer builds them once, up front, with
  the toolchain found for it: the newest installed CUDA toolkit (``CUDA_HOME``
  overrides; ``/usr/bin/nvcc`` is often a distro CUDA too old for the GPU) and, on
  Windows, the Visual Studio C++ tools, located with ``vswhere``. No special shell is
  needed. A failed or skipped (``--skip-kernels``) build is not fatal: Train 4DGS
  finds the same toolchain and builds them on its first run instead.

Examples
--------
    scripts/install.py --dry-run           # print every command, change nothing
    scripts/install.py --groups sfm        # just the rig solve
    scripts/install.py --verify-only       # report what is present, install nothing
    scripts/install.py --groups trainer    # just the trainer
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from cumuli_bridge import buildenv, viewer_assets  # noqa: E402 - need PACKAGE_ROOT on the path; plain Python, no ComfyUI

LOGGER = logging.getLogger("cumuli-install")

#: hloc must be an editable clone: its SuperPoint extractor reaches up to
#: ``third_party/`` relative to the repo root, and the weights live in the
#: SuperGluePretrainedNetwork submodule that a plain ``pip install git+URL``
#: never fetches. Both facts are why this is a clone and not a requirement.
HLOC_URL = "https://github.com/cvg/Hierarchical-Localization.git"
HLOC_COMMIT = "c13273bd0ecc2917a35910fd843712a1c6243193"
LIGHTGLUE_PIN = (
    "lightglue @ git+https://github.com/cvg/LightGlue.git"
    "@eb42fee2d71449efb0aa5c10549752b5d75384d8"
)

#: Packages whose version must not move. The additive-install promise is exactly
#: this list holding still across the whole run.
PINNED = ("torch", "numpy", "transformers", "timm")

#: The trainer's release. gsplat is pinned exactly: its kernels are built against the installed
#: torch, and this is the version the trainer was measured with.
GSPLAT_VERSION = "1.5.3"
TRAINER_TAG = "v0.1.0"
CUMULI_CORE_PIN = f"cumuli-core @ git+https://github.com/solipsist-studios/cumuli-core.git@{TRAINER_TAG}"
CUMULI_TRAINER_PIN = f"cumuli-trainer @ git+https://github.com/solipsist-studios/cumuli-trainer.git@{TRAINER_TAG}"


class InstallError(RuntimeError):
    """Raised when a step cannot run, with what to do about it."""


@dataclass
class Group:
    """One installable group: some requirements, optionally some extra work."""

    name: str
    summary: str
    #: ``(requirement, distribution name)`` -- the distribution name is what
    #: decides "already installed", the requirement is what pip is given.
    requirements: list[tuple[str, str]] = field(default_factory=list)
    no_deps: bool = False
    #: A ``name==version`` requirement counts as installed only at *that* version, and installing
    #: it may change what is there. Off for every other group, whose pins are "install if absent".
    exact: bool = False
    #: Not part of the default install: it is asked for by name (``--groups viewer``).
    optional: bool = False


def _run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
         dry_run: bool = False) -> None:
    """Run a command, streaming nothing but reporting it first."""

    # Quoted, so a line copied out of the log runs as-is -- the lightglue
    # requirement carries spaces and an @, and breaks apart unquoted.
    printable = (" ".join(f'"{a}"' if " " in str(a) else str(a) for a in argv)
                 if platform.system() == "Windows"
                 else shlex.join(str(a) for a in argv))
    if cwd:
        printable = f"({cwd}) {printable}"
    LOGGER.info("  $ %s", printable)
    if dry_run:
        return
    merged = dict(os.environ)
    merged.update(env or {})
    result = subprocess.run(  # noqa: S603 - argv is built here, never from input
        [str(a) for a in argv],
        cwd=str(cwd) if cwd else None,
        env=merged,
    )
    if result.returncode != 0:
        raise InstallError(f"Command failed with status {result.returncode}: {printable}")


def installed_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def snapshot_pinned() -> dict[str, str | None]:
    return {name: installed_version(name) for name in PINNED}


def compare_pinned(before: dict[str, str | None], after: dict[str, str | None]) -> list[str]:
    """Describe any guarded package that changed version or disappeared.

    A package that was absent and is now present was pulled in as a dependency
    -- numpy arrives with many packages, for instance -- which is an install, not a
    downgrade. Only a version moving under ComfyUI, or a package vanishing from
    beneath it, is the failure this guard exists to catch.
    """

    moved = []
    for name, was in before.items():
        now = after.get(name)
        if was is None or was == now:
            continue
        moved.append(f"{name}: {was} -> {now or 'removed'}")
    return moved


# -- environment discovery -------------------------------------------------
def default_hloc_dir() -> Path:
    """Beside this checkout, so a clone lands next to the pack it belongs to."""

    return PACKAGE_ROOT.parent / "Hierarchical-Localization"


def build_groups() -> dict[str, Group]:
    """The dependency set."""

    return {
        "core": Group(
            name="core",
            summary="4DAnyone's runtime dependencies",
            requirements=[
                ("hydra-zen", "hydra-zen"),
                ("hydra_colorlog", "hydra_colorlog"),
                ("yacs", "yacs"),
                ("lapx", "lapx"),
                ("ftfy", "ftfy"),
                ("sentencepiece", "sentencepiece"),
                ("fire", "fire"),
                ("colorlog", "colorlog"),
                ("ffmpeg-python", "ffmpeg-python"),
            ],
        ),
        "bake": Group(
            name="bake",
            summary="the .sogst container writer",
            requirements=[("dahuffman", "dahuffman")],
        ),
        "sfm": Group(
            name="sfm",
            summary="the rig solve behind Solve Rig (hloc, pycolmap, lightglue)",
            # --no-deps: torch declares older pinned cudnn/nccl than a working
            # ComfyUI carries, so a plain install silently downgrades them.
            no_deps=True,
            requirements=[
                ("h5py", "h5py"),
                ("narwhals", "narwhals"),
                ("plotly", "plotly"),
                ("pycolmap==4.0.4", "pycolmap"),
                (LIGHTGLUE_PIN, "lightglue"),
            ],
        ),
        "viewer": Group(
            name="viewer",
            summary=f"the Rerun ring viewer node: rerun-sdk {viewer_assets.VIEWER_VERSION}, matched to its bundled viewer files",
            requirements=[(f"rerun-sdk=={viewer_assets.VIEWER_VERSION}", "rerun-sdk")],
            exact=True,
            optional=True,
        ),
        "trainer": Group(
            name="trainer",
            summary=f"the 4DGS trainer: gsplat {GSPLAT_VERSION}, cumuli-core and cumuli-trainer {TRAINER_TAG}",
            # --no-deps: gsplat's and the trainer's requirements are all in ComfyUI's environment
            # already, and a plain install is how torch gets replaced under it.
            no_deps=True,
            requirements=[
                (f"gsplat=={GSPLAT_VERSION}", "gsplat"),
                ("lpips", "lpips"),
                (CUMULI_CORE_PIN, "cumuli-core"),
                (CUMULI_TRAINER_PIN, "cumuli-trainer"),
            ],
            exact=True,
        ),
    }


# -- checkouts -------------------------------------------------------------
#: The two checkouts the pack drives. Cloned rather than bundled, so the user
#: fetches each from its own origin under its own terms. Shallow, and never
#: recursive -- cumuli's submodules belong to the wider pipeline, not to this pack.
#: ``(directory, url, ref)``. The ref is pinned per repository so an archive
#: shipped today installs the same code next year, and so one checkout can move
#: without dragging the other. ``--ref`` overrides both at once. A ref is a
#: tag, a branch, or a full 40-character commit hash.
#:
#: 4DAnyone is pinned to a commit of the fork's ``sam3d-pose`` branch: the first
#: version with SAM 3D Body pose in place of GVHMR and SMPL-X, and upstream's newer
#: command line. Move it to a tag once that branch is merged and released.
CHECKOUTS = {
    "fdanyone_root": (
        "4DAnyone",
        "https://github.com/solipsist-studios/4DAnyone.git",
        "1606c0290c1bd8165d35cbe721df90126eba2781",
    ),
    "cumuli_root": ("cumuli", "https://github.com/solipsist-studios/cumuli.git", TRAINER_TAG),
}

CONFIG_FILE = PACKAGE_ROOT / "config.json"


def fetch_checkouts(deps_dir: Path, *, ref: str | None = None, dry_run: bool = False) -> dict[str, Path]:
    """Clone the checkouts under ``deps_dir``, each at its pinned ref.

    ``ref`` overrides every pin, for testing an unreleased branch.
    """

    git = shutil.which("git")
    if not git:
        raise InstallError(
            "git is required to fetch the checkouts but was not found on PATH. "
            "Install git, or clone them yourself and pass --no-fetch."
        )
    resolved: dict[str, Path] = {}
    for key, (name, url, pinned) in CHECKOUTS.items():
        target = deps_dir / name
        resolved[key] = target
        if (target / ".git").is_dir():
            LOGGER.info("  present  %-24s %s", name, target)
            continue
        if target.exists() and any(target.iterdir()):
            raise InstallError(
                f"{target} exists and is not a git clone. Move it aside, or pass "
                "--deps-dir to put the checkouts somewhere else."
            )
        deps_dir.mkdir(parents=True, exist_ok=True)
        _clone_at(git, url, ref or pinned, target, dry_run=dry_run)
    return resolved


def _is_commit_hash(ref: str) -> bool:
    return len(ref) == 40 and all(char in "0123456789abcdef" for char in ref.lower())


def _clone_at(git: str, url: str, ref: str, target: Path, *, dry_run: bool = False) -> None:
    """Shallow-clone ``url`` at ``ref``: a tag, a branch, or a full commit hash.

    ``--branch`` takes a tag as happily as a branch, and ``--depth 1`` against a tag
    fetches exactly that commit. It cannot take a commit hash, so that case is an
    init plus a fetch of just that commit, which the server allows for any commit
    reachable from one of its branches.
    """

    if not _is_commit_hash(ref):
        _run([git, "clone", "--depth", "1", "--branch", ref, url, str(target)], dry_run=dry_run)
        return
    _run([git, "init", "-q", str(target)], dry_run=dry_run)
    _run([git, "-C", str(target), "remote", "add", "origin", url], dry_run=dry_run)
    _run([git, "-C", str(target), "fetch", "--depth", "1", "origin", ref], dry_run=dry_run)
    _run([git, "-C", str(target), "-c", "advice.detachedHead=false", "checkout", "FETCH_HEAD"], dry_run=dry_run)


def checkout_revisions(paths: dict[str, Path]) -> dict[str, str]:
    """Record what was actually fetched, so a report can name exact commits."""

    git = shutil.which("git")
    out = {}
    for key, path in paths.items():
        if not git or not (path / ".git").is_dir():
            continue
        try:
            result = subprocess.run(  # noqa: S603
                [git, "-C", str(path), "rev-parse", "--short=10", "HEAD"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                out[key] = result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return out


def write_config(paths: dict[str, Path], work_root: str | None, *, dry_run: bool = False) -> None:
    """Write config.json so the nodes find the checkouts with no UI fiddling.

    Merges into an existing file rather than replacing it: someone who has
    already tuned settings should not lose them to a re-run.
    """

    payload: dict[str, object] = {}
    if CONFIG_FILE.is_file():
        try:
            existing = json.loads(CONFIG_FILE.read_text())
            if isinstance(existing, dict):
                payload = existing
        except (OSError, ValueError):
            LOGGER.warning("  %s is not readable JSON; writing a fresh one", CONFIG_FILE)
    payload.update({key: str(path) for key, path in paths.items()})
    if work_root:
        payload["work_root"] = work_root
    LOGGER.info("  writing %s", CONFIG_FILE)
    for key in ("fdanyone_root", "cumuli_root", "work_root"):
        if key in payload:
            LOGGER.info("    %-16s %s", key, payload[key])
    if dry_run:
        return
    CONFIG_FILE.write_text(json.dumps(payload, indent=2) + "\n")


def fetch_models(paths: dict[str, Path], *, with_turbo: bool = False, dry_run: bool = False) -> None:
    """Pull 4DAnyone's published weights, and the SAM 3D Body weights, using its own downloader.

    The Turbo LoRA is left out unless asked for: it is licensed CC BY-NC-SA 4.0, so a
    default install must not put a non-commercial file where commercial use would find it.
    """

    root = paths.get("fdanyone_root")
    if root is None or not (root / "fdanyone" / "download.py").is_file():
        LOGGER.info("  skipped (no 4DAnyone checkout)")
        return
    argv = [sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from fdanyone.download import ensure_models; "
            "ensure_models(model_dir=sys.argv[2], enable_turbo=sys.argv[3] == 'True')",
            str(root), str(root / "models"), str(bool(with_turbo))]
    _run(argv, cwd=root, dry_run=dry_run)


# -- steps -----------------------------------------------------------------
def _wanted_version(req: str) -> str | None:
    """The version of a plain ``name==version`` requirement, or ``None`` for anything else."""

    if "==" not in req or "@" in req:
        return None
    return req.split("==", 1)[1].strip()


def _satisfied(group: Group, req: str, have: str | None) -> bool:
    if have is None:
        return False
    wanted = _wanted_version(req) if group.exact else None
    return wanted is None or have == wanted


def install_requirements(group: Group, *, force: bool, dry_run: bool) -> None:
    pending = []
    for req, dist in group.requirements:
        have = installed_version(dist)
        if force or not _satisfied(group, req, have):
            pending.append(req)
            wanted = _wanted_version(req)
            if group.exact and have and wanted and have != wanted:
                LOGGER.warning("  changing %s %s -> %s (pinned: it must match the files this pack bundles)",
                               dist, have, wanted)
        else:
            LOGGER.info("  present  %-46s %s", dist, have)
    if not pending:
        LOGGER.info("  nothing to install")
        return
    argv = [sys.executable, "-m", "pip", "install"]
    if group.no_deps:
        argv.append("--no-deps")
    argv.extend(pending)
    _run(argv, dry_run=dry_run)


def fetch_viewer_assets(*, dry_run: bool = False) -> None:
    """Fetch the pinned Rerun web viewer files (about 15 MB) and verify them, once, here."""

    if viewer_assets.find_assets() is not None:
        LOGGER.info("  present  %-46s %s", "rerun web viewer files", viewer_assets.VIEWER_VERSION)
        return
    if dry_run:
        LOGGER.info("  $ fetch %s", viewer_assets.TARBALL_URL)
        return
    try:
        directory = viewer_assets.ensure_assets()
    except viewer_assets.ViewerAssetsError as exc:
        raise InstallError(str(exc)) from None
    LOGGER.info("  fetched  %-46s %s", "rerun web viewer files", directory)


def prebuild_kernels(*, dry_run: bool) -> bool:
    """Compile gsplat's CUDA kernels now, so the first Train 4DGS does not.

    Returns whether it built. Failure is reported, not raised: the kernels are cached
    where the trainer will look for them, and if they are missing then Train 4DGS builds
    them itself with the same toolchain, so the install is still usable.
    """

    LOGGER.info("  building gsplat's CUDA kernels (a few minutes the first time; cached afterwards)")
    env = buildenv.toolchain_env()
    if "CUDA_HOME" in env:
        LOGGER.info("    CUDA_HOME=%s", env["CUDA_HOME"])
    try:
        _run([sys.executable, "-c", buildenv.BUILD_SNIPPET], env=env, dry_run=dry_run)
    except InstallError as exc:
        LOGGER.warning("  the kernel build failed: %s", exc)
        LOGGER.warning("  Everything else is installed. Train 4DGS will try again on its first run, with the same toolchain;")
        LOGGER.warning("  if it fails there too, it needs a CUDA toolkit that knows your GPU%s.",
                       " and Visual Studio's C++ build tools" if platform.system() == "Windows" else "")
        return False
    return True


def install_hloc(hloc_dir: Path, *, force: bool, dry_run: bool) -> None:
    """Clone (with submodules), pin the commit, install editable."""

    if installed_version("hloc") and not force:
        try:
            import hloc

            LOGGER.info("  present  %-46s %s", "hloc", Path(hloc.__file__).parent.parent)
            return
        except ImportError:
            LOGGER.warning("  hloc has metadata but does not import; reinstalling")

    git = shutil.which("git")
    if not git:
        raise InstallError("git is required to install hloc but was not found on PATH.")

    if (hloc_dir / ".git").is_dir():
        LOGGER.info("  reusing existing clone at %s", hloc_dir)
        _run([git, "-C", hloc_dir, "submodule", "update", "--init", "--recursive"], dry_run=dry_run)
    else:
        if hloc_dir.exists() and any(hloc_dir.iterdir()):
            raise InstallError(
                f"{hloc_dir} exists and is not a git clone. Move it aside, or pass "
                "--hloc-dir pointing somewhere else."
            )
        # The recursive clone is not optional: SuperPoint's weights live in the
        # SuperGluePretrainedNetwork submodule.
        _run([git, "clone", "--recurse-submodules", HLOC_URL, "-b", "master", str(hloc_dir)],
             dry_run=dry_run)

    _run([git, "-C", str(hloc_dir), "checkout", HLOC_COMMIT], dry_run=dry_run)
    _run([git, "-C", str(hloc_dir), "submodule", "update", "--init", "--recursive"], dry_run=dry_run)
    # Editable, always: a copy into site-packages loses the third_party/ lookup.
    _run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(hloc_dir)], dry_run=dry_run)


# -- verification ----------------------------------------------------------
def verify(groups: dict[str, Group], selected: list[str]) -> list[str]:
    """Report what is present; return the names of anything still missing."""

    missing: list[str] = []
    for name in selected:
        group = groups[name]
        LOGGER.info("%s -- %s", name, group.summary)
        for req, dist in group.requirements:
            have = installed_version(dist)
            good = _satisfied(group, req, have)
            label = "ok" if good else ("WRONG" if have else "MISSING")
            LOGGER.info("  %-8s %-44s %s", label, dist, have or "-")
            if not good:
                missing.append(f"{dist} (installed {have}, need {_wanted_version(req)})" if have else dist)
        if name == "viewer":
            found = viewer_assets.find_assets()
            LOGGER.info("  %-8s %-44s %s", "ok" if found else "MISSING", "rerun web viewer files", found or "-")
            if not found:
                missing.append("rerun web viewer files")
        if name == "sfm":
            have = installed_version("hloc")
            location = ""
            try:
                import hloc

                location = str(Path(hloc.__file__).parent.parent)
            except ImportError:
                have = None
            LOGGER.info("  %-8s %-44s %s", "ok" if have else "MISSING", "hloc (editable)",
                        location or "-")
            if not have:
                missing.append("hloc")
    return missing


def check_interpreter() -> None:
    """Refuse to install into something that is plainly not ComfyUI's environment."""

    try:
        import torch  # noqa: F401
    except ImportError:
        raise InstallError(
            f"{sys.executable} has no torch, so it is not ComfyUI's environment. "
            "Run the wrapper with --python <ComfyUI's interpreter>, or set COMFYUI_PYTHON."
        ) from None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install this pack's dependencies into ComfyUI's environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--groups", default="core,bake,sfm,trainer",
                        help="Comma-separated subset of core,bake,sfm,trainer,viewer. Default: core,bake,sfm,trainer. "
                             "viewer (the Rerun ring viewer node) is opt-in: it pins rerun-sdk to the version of "
                             "the viewer files it bundles, which may change an installed rerun-sdk.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command without running any of them.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Report what is already installed and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Reinstall/rebuild even when already present. Needed after a torch upgrade.")
    parser.add_argument("--hloc-dir", type=Path, default=None,
                        help=f"Where to clone hloc. Default: {default_hloc_dir()}")
    parser.add_argument("--deps-dir", type=Path, default=None,
                        help=f"Where to clone the checkouts. Default: {PACKAGE_ROOT / 'deps'}")
    parser.add_argument("--ref", default=None,
                        help="Override the pinned ref for every checkout (default: each repo's own pin, "
                             + ", ".join(f"{n}@{r[:10] if _is_commit_hash(r) else r}" for n, _, r in CHECKOUTS.values()) + ").")
    parser.add_argument("--work-root", default=None,
                        help="Large drive for per-run intermediates (~20 GB/run). Written to config.json.")
    parser.add_argument("--comfyui-root", type=Path, default=None,
                        help="ComfyUI checkout; lets the installer check whether the SAM 3D Body "
                             "weights are already in its models/detection folder")
    parser.add_argument("--skip-kernels", action="store_true",
                        help="Do not build gsplat's CUDA kernels now. Train 4DGS builds them on its first run instead.")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Do not clone the checkouts; use whatever the config already points at.")
    parser.add_argument("--with-turbo", action="store_true",
                        help="Also download the 4DAnyone-Turbo LoRA. It is CC BY-NC-SA 4.0 "
                             "(non-commercial) and Generate Ring leaves it off by default.")
    parser.add_argument("--no-models", action="store_true",
                        help="Do not download 4DAnyone's published weights.")
    parser.add_argument("--no-configure", action="store_true",
                        help="Do not write config.json.")
    parser.add_argument("--verbose", action="store_true", help="Debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )

    selected = [g.strip() for g in args.groups.split(",") if g.strip()]
    groups = build_groups()
    unknown = [g for g in selected if g not in groups]
    if unknown:
        LOGGER.error("Unknown group(s): %s. Choose from: %s",
                     ", ".join(unknown), ", ".join(groups))
        return 2

    LOGGER.info("interpreter  %s", sys.executable)
    try:
        check_interpreter()
        import torch

        LOGGER.info("torch        %s (cuda %s)", torch.__version__, torch.version.cuda or "cpu")
    except InstallError as exc:
        LOGGER.error("%s", exc)
        return 1
    LOGGER.info("")

    if args.verify_only:
        missing = verify(groups, selected)
        LOGGER.info("")
        if missing:
            LOGGER.info("missing: %s", ", ".join(missing))
            return 1
        LOGGER.info("everything for %s is installed", ", ".join(selected))
        return 0

    before = snapshot_pinned()
    deps_dir = (args.deps_dir or (PACKAGE_ROOT / "deps")).expanduser()
    checkouts: dict[str, Path] = {}

    try:
        if not args.no_fetch:
            LOGGER.info("checkouts -- 4DAnyone, cumuli")
            checkouts = fetch_checkouts(deps_dir, ref=args.ref, dry_run=args.dry_run)
            LOGGER.info("")
        if not args.no_configure and checkouts:
            LOGGER.info("configuration")
            write_config(checkouts, args.work_root, dry_run=args.dry_run)
            LOGGER.info("")
    except InstallError as exc:
        LOGGER.error("%s", exc)
        return 1

    hloc_dir = (args.hloc_dir or default_hloc_dir()).expanduser()

    try:
        for name in selected:
            group = groups[name]
            LOGGER.info("%s -- %s", name, group.summary)
            install_requirements(group, force=args.force, dry_run=args.dry_run)
            if name == "sfm":
                install_hloc(hloc_dir, force=args.force, dry_run=args.dry_run)
            if name == "trainer" and not args.skip_kernels:
                prebuild_kernels(dry_run=args.dry_run)
            if name == "viewer":
                fetch_viewer_assets(dry_run=args.dry_run)
            LOGGER.info("")
    except InstallError as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.dry_run:
        LOGGER.info("dry run: nothing was installed")
        return 0

    # The additive promise, checked rather than trusted.
    moved = compare_pinned(before, snapshot_pinned())
    if moved:
        LOGGER.error("These packages moved during the install, which breaks ComfyUI:")
        for line in moved:
            LOGGER.error("  %s", line)
        LOGGER.error("Reinstall the original versions before starting ComfyUI.")
        return 1

    if not args.no_models and checkouts:
        LOGGER.info("models -- 4DAnyone's published weights")
        try:
            fetch_models(checkouts, with_turbo=args.with_turbo, dry_run=args.dry_run)
        except InstallError as exc:
            LOGGER.error("%s", exc)
            LOGGER.error("The weights can be fetched later; everything else is installed.")
        LOGGER.info("")

    missing = verify(groups, selected)
    revisions = checkout_revisions(checkouts)
    if revisions:
        LOGGER.info("")
        LOGGER.info("checkouts")
        for key, sha in revisions.items():
            LOGGER.info("  %-16s %s @ %s", key, checkouts[key], sha)
    LOGGER.info("")
    if missing:
        LOGGER.error("still missing after the run: %s", ", ".join(missing))
        return 1

    LOGGER.info("done -- %s installed, torch untouched", ", ".join(selected))

    LOGGER.info("")
    LOGGER.info("The SAM 3D Body weights are downloaded with 4DAnyone's models (Meta's SAM License applies).")
    LOGGER.info("Generate Ring needs a ComfyUI that ships the SAM 3D Body nodes (0.34 or newer).")
    LOGGER.info("")
    LOGGER.info("Restart ComfyUI to pick up the nodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
