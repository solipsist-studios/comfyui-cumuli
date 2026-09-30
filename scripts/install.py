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

  core     4DAnyone and its vendored GVHMR
  bake     the ``.sogst`` container writer
  sfm      the rig solve behind Solve Rig (hloc, pycolmap, lightglue)
  trainer  the OMG4 rotor-4DGS trainer, including its CUDA extensions

Two hazards this script exists to defuse, both documented in the README and both
silent when they happen by hand:

* **pip upgrading torch out from under ComfyUI.** Several of these packages
  declare pinned dependencies older than a working ComfyUI carries. The affected
  installs pass ``--no-deps``, and the versions of torch, numpy, transformers,
  timm and ultralytics are recorded before and compared after: if anything moved,
  the run fails loudly instead of leaving a broken ComfyUI to discover later.
* **Building the trainer's CUDA extensions against the wrong nvcc.**
  ``/usr/bin/nvcc`` is often a distro CUDA too old to target the installed GPU.
  The toolkit is auto-detected to match torch's own CUDA major version, and the
  arch list comes from the GPU actually present.

Examples
--------
    scripts/install.py --dry-run           # print every command, change nothing
    scripts/install.py --groups sfm        # just the rig solve
    scripts/install.py --verify-only       # report what is present, install nothing
    scripts/install.py --omg4 ~/src/OMG4 --cuda-home /usr/local/cuda-13.2
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

#: The three CUDA extensions OMG4 ships as source trees, pre-built here so the
#: trainer never JIT-compiles mid-run. ABI-bound to torch: redo after a torch bump.
TRAINER_EXTENSIONS = ("diff-gaussian-rasterization", "simple-knn", "pointops2")

#: Packages whose version must not move. The additive-install promise is exactly
#: this list holding still across the whole run.
PINNED = ("torch", "numpy", "transformers", "timm", "ultralytics")


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
    -- numpy arrives with smplx, for instance -- which is an install, not a
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
def torch_cuda_major() -> str | None:
    """The CUDA major version torch was built against, e.g. ``13``."""

    try:
        import torch
    except ImportError:
        return None
    version = getattr(torch.version, "cuda", None)
    return version.split(".")[0] if version else None


def gpu_arch_list() -> str | None:
    """``TORCH_CUDA_ARCH_LIST`` for the GPU actually installed."""

    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(  # noqa: S603
            [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    caps = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return caps[0] if caps else None


def find_cuda_home(preferred_major: str | None) -> Path | None:
    """An nvcc new enough for this GPU, preferring torch's own CUDA major.

    ``/usr/bin/nvcc`` is deliberately last: a distro CUDA is frequently too old
    to target a current card, and it is the default that silently produces
    extensions the GPU cannot run.
    """

    explicit = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if explicit and (Path(explicit) / "bin" / _nvcc_name()).is_file():
        return Path(explicit)

    roots = sorted(Path("/usr/local").glob("cuda-*"), reverse=True)
    if preferred_major:
        matching = [r for r in roots if r.name.startswith(f"cuda-{preferred_major}")]
        roots = matching + [r for r in roots if r not in matching]
    for root in roots:
        if (root / "bin" / _nvcc_name()).is_file():
            return root

    nvcc = shutil.which("nvcc")
    return Path(nvcc).resolve().parent.parent if nvcc else None


def _nvcc_name() -> str:
    return "nvcc.exe" if platform.system() == "Windows" else "nvcc"


def default_hloc_dir() -> Path:
    """Beside this checkout, so a clone lands next to the pack it belongs to."""

    return PACKAGE_ROOT.parent / "Hierarchical-Localization"


def default_omg4() -> Path | None:
    """Whatever the pack itself is configured to use, so the two agree."""

    try:
        from cumuli_bridge.settings import load_settings

        return load_settings().trainer_root
    except Exception:  # settings are optional here; the flag can supply it
        return None


def build_groups(cuda_major: str | None) -> dict[str, Group]:
    """The dependency set, with the CUDA-versioned wheels matched to torch."""

    # cupy and cuml ship one wheel per CUDA major. Picking the wrong one
    # installs cleanly and fails at import, so follow torch rather than guess.
    suffix = cuda_major or "13"
    return {
        "core": Group(
            name="core",
            summary="4DAnyone and its vendored GVHMR",
            requirements=[
                ("smplx==0.1.28", "smplx"),
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
        "trainer": Group(
            name="trainer",
            summary="the OMG4 rotor-4DGS trainer",
            requirements=[
                (f"cupy-cuda{suffix}x", f"cupy-cuda{suffix}x"),
                (f"cuml-cu{suffix}", f"cuml-cu{suffix}"),
                ("omegaconf", "omegaconf"),
                ("imagesize", "imagesize"),
            ],
        ),
    }


# -- checkouts -------------------------------------------------------------
#: The three checkouts the pack drives. Cloned rather than bundled: OMG4's
#: upstream carries no licence at all, and hloc's SuperGlue submodule is
#: non-commercial research only, so the user fetches each from its own origin
#: under its own terms. Shallow, and never recursive -- cumuli's other
#: submodules belong to the wider pipeline, not to this pack.
#: ``(directory, url, ref)``. The ref is pinned per repository so an archive
#: shipped today installs the same code next year, and so one checkout can move
#: without dragging the others. ``--ref`` overrides all three at once.
CHECKOUTS = {
    "fdanyone_root": ("4DAnyone", "https://github.com/solipsist-studios/4DAnyone.git", "v0.0.1"),
    "trainer_root": ("OMG4", "https://github.com/solipsist-studios/OMG4.git", "v0.0.2"),
    "cumuli_root": ("cumuli", "https://github.com/solipsist-studios/cumuli.git", "v0.0.2"),
}

CONFIG_FILE = PACKAGE_ROOT / "config.json"


def fetch_checkouts(deps_dir: Path, *, ref: str | None = None, dry_run: bool = False) -> dict[str, Path]:
    """Clone the three checkouts under ``deps_dir``, each at its pinned ref.

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
        wanted = ref or pinned
        # --branch takes a tag as happily as a branch, and --depth 1 against a
        # tag fetches exactly that commit.
        _run([git, "clone", "--depth", "1", "--branch", wanted, url, str(target)], dry_run=dry_run)
    return resolved


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
    # OMG4 is cloned standalone here, so trainer_root is the clone itself.
    if work_root:
        payload["work_root"] = work_root
    LOGGER.info("  writing %s", CONFIG_FILE)
    for key in ("fdanyone_root", "trainer_root", "cumuli_root", "work_root"):
        if key in payload:
            LOGGER.info("    %-16s %s", key, payload[key])
    if dry_run:
        return
    CONFIG_FILE.write_text(json.dumps(payload, indent=2) + "\n")


def fetch_models(paths: dict[str, Path], *, dry_run: bool = False) -> None:
    """Pull 4DAnyone's published weights using its own downloader."""

    root = paths.get("fdanyone_root")
    if root is None or not (root / "fdanyone" / "download.py").is_file():
        LOGGER.info("  skipped (no 4DAnyone checkout)")
        return
    argv = [sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from fdanyone.download import ensure_models; "
            "ensure_models(model_dir=sys.argv[2], gvhmr_root=sys.argv[3])",
            str(root), str(root / "models"), str(root / "third_party" / "GVHMR")]
    _run(argv, cwd=root, dry_run=dry_run)


def missing_manual_assets(paths: dict[str, Path]) -> list[str]:
    """What no installer may fetch: licence-gated downloads.

    SMPL-X is not in 4DAnyone's published model list. It is gated behind
    registration at smpl-x.is.tue.mpg.de, and GVHMR needs it for the motion
    solve -- so Generate Ring fails without it, however complete everything
    else looks.
    """

    root = paths.get("fdanyone_root")
    if root is None:
        return []
    smplx = root / "models" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    return [] if smplx.is_file() else [str(smplx)]


# -- steps -----------------------------------------------------------------
def install_requirements(group: Group, *, force: bool, dry_run: bool) -> None:
    pending = [req for req, dist in group.requirements
               if force or installed_version(dist) is None]
    for req, dist in group.requirements:
        have = installed_version(dist)
        if have and not force:
            LOGGER.info("  present  %-46s %s", dist, have)
    if not pending:
        LOGGER.info("  nothing to install")
        return
    argv = [sys.executable, "-m", "pip", "install"]
    if group.no_deps:
        argv.append("--no-deps")
    argv.extend(pending)
    _run(argv, dry_run=dry_run)


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


def build_trainer_extensions(omg4: Path, cuda_home: Path | None, arch: str | None, *,
                             force: bool, dry_run: bool) -> None:
    """Pre-build OMG4's CUDA extensions so the trainer never compiles at run time."""

    present = {
        "diff-gaussian-rasterization": installed_version("diff_gaussian_rasterization"),
        "simple-knn": installed_version("simple_knn"),
        "pointops2": installed_version("pointops2"),
    }
    missing = [name for name in TRAINER_EXTENSIONS if present[name] is None]
    for name in TRAINER_EXTENSIONS:
        if present[name] and not force:
            LOGGER.info("  present  %-46s %s", name, present[name])
    if not missing and not force:
        LOGGER.info("  nothing to build (pass --force to rebuild after a torch upgrade)")
        return

    if not omg4.is_dir():
        raise InstallError(
            f"OMG4 checkout not found at {omg4}. Pass --omg4 <path>, or set the "
            "'trainer_root' key of the bridge config."
        )
    sources = [omg4 / name for name in (missing if not force else list(TRAINER_EXTENSIONS))]
    absent = [s for s in sources if not s.is_dir()]
    if absent:
        raise InstallError(
            f"{omg4} does not look like an OMG4 checkout; missing: "
            + ", ".join(s.name for s in absent)
        )

    if cuda_home is None:
        raise InstallError(
            "No CUDA toolkit found. Install one and pass --cuda-home, or set CUDA_HOME. "
            "It must be an nvcc that knows your GPU -- a distro /usr/bin/nvcc is often "
            "too old to target a current card."
        )
    if arch is None:
        raise InstallError(
            "Could not read the GPU compute capability (nvidia-smi unavailable). "
            "Pass --arch, e.g. --arch 12.0 for an RTX 5090."
        )
    if platform.system() == "Windows":
        LOGGER.warning(
            "  Building CUDA extensions on Windows needs a matching MSVC Build Tools "
            "install and a Developer Command Prompt. If this fails, that is usually why."
        )

    env = {
        "CUDA_HOME": str(cuda_home),
        "PATH": os.pathsep.join([str(cuda_home / "bin"), os.environ.get("PATH", "")]),
        "TORCH_CUDA_ARCH_LIST": arch,
    }
    LOGGER.info("  CUDA_HOME=%s  TORCH_CUDA_ARCH_LIST=%s", cuda_home, arch)
    # --no-deps or pip upgrades torch while resolving the extensions' requirements;
    # --no-build-isolation so they build against the torch already installed.
    _run(
        [sys.executable, "-m", "pip", "install", *[str(s) for s in sources],
         "--no-build-isolation", "--no-deps"],
        env=env,
        dry_run=dry_run,
    )


# -- verification ----------------------------------------------------------
def verify(groups: dict[str, Group], selected: list[str]) -> list[str]:
    """Report what is present; return the names of anything still missing."""

    missing: list[str] = []
    for name in selected:
        group = groups[name]
        LOGGER.info("%s -- %s", name, group.summary)
        for _req, dist in group.requirements:
            have = installed_version(dist)
            LOGGER.info("  %-8s %-44s %s", "ok" if have else "MISSING", dist, have or "-")
            if not have:
                missing.append(dist)
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
        if name == "trainer":
            for dist in ("diff_gaussian_rasterization", "simple_knn", "pointops2"):
                have = installed_version(dist)
                LOGGER.info("  %-8s %-44s %s", "ok" if have else "MISSING", dist, have or "-")
                if not have:
                    missing.append(dist)
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
                        help="Comma-separated subset of core,bake,sfm,trainer. Default: all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command without running any of them.")
    parser.add_argument("--verify-only", action="store_true",
                        help="Report what is already installed and exit.")
    parser.add_argument("--force", action="store_true",
                        help="Reinstall/rebuild even when already present. Needed after a torch upgrade.")
    parser.add_argument("--hloc-dir", type=Path, default=None,
                        help=f"Where to clone hloc. Default: {default_hloc_dir()}")
    parser.add_argument("--omg4", type=Path, default=None,
                        help="OMG4 checkout holding the CUDA extension sources. Default: the bridge config's trainer_root.")
    parser.add_argument("--cuda-home", type=Path, default=None,
                        help="CUDA toolkit for the extension builds. Default: auto-detected to match torch.")
    parser.add_argument("--arch", default=None,
                        help="TORCH_CUDA_ARCH_LIST value. Default: read from nvidia-smi.")
    parser.add_argument("--deps-dir", type=Path, default=None,
                        help=f"Where to clone the three checkouts. Default: {PACKAGE_ROOT / 'deps'}")
    parser.add_argument("--ref", default=None,
                        help="Override the pinned ref for every checkout (default: each repo's own pin, "
                             + ", ".join(f"{n}@{r}" for n, _, r in CHECKOUTS.values()) + ").")
    parser.add_argument("--work-root", default=None,
                        help="Large drive for per-run intermediates (~20 GB/run). Written to config.json.")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Do not clone the checkouts; use whatever the config already points at.")
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
    cuda_major = torch_cuda_major()
    groups = build_groups(cuda_major)
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
            LOGGER.info("checkouts -- 4DAnyone, OMG4, cumuli")
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
    # A freshly fetched OMG4 wins over the config: it is what was just cloned.
    omg4 = (args.omg4 or checkouts.get("trainer_root") or default_omg4() or Path("OMG4")).expanduser()
    cuda_home = args.cuda_home or find_cuda_home(cuda_major)
    arch = args.arch or gpu_arch_list()

    try:
        for name in selected:
            group = groups[name]
            LOGGER.info("%s -- %s", name, group.summary)
            install_requirements(group, force=args.force, dry_run=args.dry_run)
            if name == "sfm":
                install_hloc(hloc_dir, force=args.force, dry_run=args.dry_run)
            if name == "trainer":
                build_trainer_extensions(omg4, cuda_home, arch,
                                         force=args.force, dry_run=args.dry_run)
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
            fetch_models(checkouts, dry_run=args.dry_run)
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

    # The one thing no installer may do for you.
    manual = missing_manual_assets(checkouts)
    if manual:
        LOGGER.info("")
        LOGGER.info("ONE STEP LEFT -- SMPL-X body models are licence-gated and cannot be")
        LOGGER.info("downloaded automatically. Generate Ring needs them for the motion solve.")
        LOGGER.info("  1. register and accept the licence at https://smpl-x.is.tue.mpg.de/")
        LOGGER.info("  2. download models_smplx_v1_1.zip")
        LOGGER.info("  3. place SMPLX_NEUTRAL.npz at:")
        for path in manual:
            LOGGER.info("       %s", path)
    LOGGER.info("")
    LOGGER.info("Restart ComfyUI to pick up the nodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
