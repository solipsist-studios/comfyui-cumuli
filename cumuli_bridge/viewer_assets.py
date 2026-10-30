# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""The Rerun web viewer files, fetched once at install time and verified.

The viewer is the ``@rerun-io/web-viewer`` npm package (MIT / Apache-2.0): two JavaScript
files and a 49 MB WebAssembly module. It is pinned to one version, and that version must
equal the ``rerun-sdk`` that writes the recordings it plays, or a recording written by a
newer SDK may not load. 4DAnyone's own viewer pins the same way (and its notes record the
same 0.37.1 integrity hash).

Nothing here runs when a graph executes. The installer calls :func:`ensure_assets`; a node
only calls :func:`find_assets` and, if the files are absent, says how to get them. A node
that downloaded 50 MB in the middle of a workflow would be a surprise.

Plain Python (urllib, tarfile), importable without ComfyUI.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

LOGGER = logging.getLogger("comfyui-cumuli")

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

#: One version for the viewer files and for ``rerun-sdk``. Change both together.
VIEWER_VERSION = "0.37.1"
TARBALL_URL = f"https://registry.npmjs.org/@rerun-io/web-viewer/-/web-viewer-{VIEWER_VERSION}.tgz"
#: The npm registry's own integrity string for that tarball (sha512, base64).
TARBALL_INTEGRITY = "sha512-po+PoR60eb9O9lexMdFUlgNnfvtohuKeBlPCWQI/6I9XU9tQZpcptQb6R4ZKsXPntnSQEZrFAtGVqlI74+wUPQ=="

#: The only files taken from the package; the rest is TypeScript sources and source maps.
FILES = ("index.js", "re_viewer.js", "re_viewer_bg.wasm")
MANIFEST = "manifest.json"

#: The one change made to the package, for the same reason 4DAnyone makes it: ``index.js``
#: loads its WASM bindings with ``import("./re_viewer")``, a bare specifier that only a
#: bundler resolves. A browser needs the extension. Exactly one occurrence is expected, so a
#: package that is not the pinned one fails loudly instead of being patched blindly.
PATCHES = {"index.js": (b'import("./re_viewer")', b'import("./re_viewer.js")')}


class ViewerAssetsError(RuntimeError):
    """Raised when the viewer files cannot be fetched or verified, with what to do about it."""


def assets_root() -> Path:
    """Where fetched viewer files live (git-ignored; ~50 MB per version)."""

    return PACKAGE_ROOT / "viewer_assets"


def integrity_of(data: bytes) -> str:
    return "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode("ascii")


def find_assets(root: Path | None = None) -> Path | None:
    """The directory holding this version's files, or ``None`` when they are not all there."""

    directory = (root or assets_root()) / VIEWER_VERSION
    try:
        manifest = json.loads((directory / MANIFEST).read_text())
    except (OSError, ValueError):
        return None
    if manifest.get("version") != VIEWER_VERSION:
        return None
    for name in FILES:
        path = directory / name
        if not path.is_file() or path.stat().st_size != manifest.get("files", {}).get(name, {}).get("size"):
            return None
    return directory


def _safe_member(tar: tarfile.TarFile, name: str) -> bytes:
    """One file's bytes from the package, by exact name; never extracts to disk by path."""

    member = tar.getmember(f"package/{name}")
    if not member.isfile():
        raise ViewerAssetsError(f"{name} in the viewer package is not a regular file.")
    handle = tar.extractfile(member)
    if handle is None:
        raise ViewerAssetsError(f"Could not read {name} from the viewer package.")
    return handle.read()


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "comfyui-cumuli installer"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URL
        return response.read()


def ensure_assets(
    *,
    root: Path | None = None,
    fetch: Callable[[str], bytes] = _download,
    integrity: str = TARBALL_INTEGRITY,
    url: str = TARBALL_URL,
) -> Path:
    """Return the directory with this version's viewer files, fetching them if needed.

    The tarball is checked against the pinned sha512 *before* anything is unpacked, and only
    the three named files are read out of it (by exact member name, so a crafted archive
    cannot write elsewhere). The files appear all at once: they are staged in a temporary
    directory and renamed into place, so an interrupted fetch leaves nothing half-installed.
    """

    existing = find_assets(root)
    if existing is not None:
        return existing

    base = root or assets_root()
    target = base / VIEWER_VERSION
    LOGGER.info("Fetching the Rerun web viewer %s (about 15 MB) from %s", VIEWER_VERSION, url)
    try:
        data = fetch(url)
    except OSError as exc:
        raise ViewerAssetsError(
            f"Could not download the Rerun web viewer from {url}: {exc}. Check the network, then run the "
            "installer again."
        ) from None
    actual = integrity_of(data)
    if actual != integrity:
        raise ViewerAssetsError(
            f"The downloaded Rerun web viewer does not match its pinned checksum (expected {integrity}, got "
            f"{actual}). Nothing was installed."
        )

    base.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{VIEWER_VERSION}.", dir=base))
    try:
        files = {}
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for name in FILES:
                try:
                    content = _safe_member(tar, name)
                except KeyError:
                    raise ViewerAssetsError(f"The viewer package has no {name}; it is not the package expected.") from None
                if name in PATCHES:
                    old, new = PATCHES[name]
                    if content.count(old) != 1:
                        raise ViewerAssetsError(
                            f"{name} in the viewer package does not contain {old.decode()!r} exactly once; it is not the "
                            "version this pack was written for."
                        )
                    content = content.replace(old, new)
                (staging / name).write_bytes(content)
                files[name] = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        (staging / MANIFEST).write_text(json.dumps(
            {"version": VIEWER_VERSION, "tarball_integrity": integrity, "source": url, "files": files}, indent=2,
        ) + "\n")
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target
