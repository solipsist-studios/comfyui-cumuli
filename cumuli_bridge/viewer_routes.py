# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""HTTP routes that put the Rerun viewer inside ComfyUI's own port.

Three things are served, all under ``/cumuli/viewer/``:

* ``page`` and ``start.js`` -- a small page and the script that starts the viewer on one recording;
* ``assets/<file>`` -- the pinned viewer files (``viewer_assets``), by exact name;
* ``rrd/<token>.rrd`` -- a recording a node registered, by an opaque token.

Serving them from ComfyUI's port keeps the browser on one origin, so the viewer needs no second
port, no cross-origin headers and no extra firewall exposure -- which matters because the
browser is often not on the machine ComfyUI runs on.

The page is sent with a Content-Security-Policy whose network access is ``'self'`` only. The
viewer by default fetches web fonts, asks GitHub for the latest release and posts usage
telemetry; none of that is wanted from a page embedded in a workflow, and the policy blocks
all of it without relying on a viewer option.

Plain aiohttp: ``setup`` takes any ``RouteTableDef``-like object, so this is testable and
usable without ComfyUI (the node passes ``PromptServer.instance.routes``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import viewer_assets

HERE = Path(__file__).resolve().parent

#: Network access only to this origin. ``wasm-unsafe-eval`` is what compiling the viewer's WASM
#: needs; ``blob:`` is the viewer's own workers and decoded video frames.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; "
    "script-src 'self' 'wasm-unsafe-eval'; "
    "connect-src 'self' blob: data:; "
    "img-src 'self' data: blob:; "
    "media-src 'self' blob: data:; "
    "style-src 'self' 'unsafe-inline'; "
    "font-src 'self' data:; "
    "worker-src 'self' blob:; "
    "frame-ancestors 'self'"
)

CONTENT_TYPES = {".js": "text/javascript", ".wasm": "application/wasm", ".html": "text/html", ".rrd": "application/octet-stream"}

_RECORDINGS: dict[str, Path] = {}


def register_recording(path: str | Path) -> str:
    """An opaque token for one recording file; the route serves only registered files."""

    resolved = Path(path).resolve()
    token = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:20]
    _RECORDINGS[token] = resolved
    return token


def setup(routes, *, find_assets=viewer_assets.find_assets) -> None:
    """Register the viewer routes on ``routes`` (an aiohttp ``RouteTableDef``)."""

    from aiohttp import web

    def respond(path: Path, *, secure: bool = False):
        headers = {"Content-Type": CONTENT_TYPES.get(path.suffix, "application/octet-stream"), "Cache-Control": "no-cache"}
        if secure:
            headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
            headers["X-Content-Type-Options"] = "nosniff"
        return web.FileResponse(path, headers=headers)

    @routes.get("/cumuli/viewer/page")
    async def viewer_page(request):
        return respond(HERE / "viewer_page.html", secure=True)

    @routes.get("/cumuli/viewer/start.js")
    async def viewer_start(request):
        return respond(HERE / "viewer_start.js", secure=True)

    @routes.get("/cumuli/viewer/assets/{name}")
    async def viewer_asset(request):
        name = request.match_info["name"]
        if name not in viewer_assets.FILES:
            return web.Response(status=404, text="unknown viewer file")
        directory = find_assets()
        if directory is None:
            return web.Response(
                status=404,
                text="The Rerun viewer files are not installed. Run ./install.sh --groups viewer, then restart ComfyUI.",
            )
        return respond(directory / name, secure=True)

    @routes.get("/cumuli/viewer/rrd/{token}.rrd")
    async def viewer_recording(request):
        path = _RECORDINGS.get(request.match_info["token"])
        if path is None or not path.is_file():
            return web.Response(status=404, text="unknown or expired recording")
        return respond(path)
