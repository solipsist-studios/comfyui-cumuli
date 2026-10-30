# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
"""The Rerun viewer node's non-GUI parts: asset fetching, recordings, routes."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cumuli_bridge import viewer_assets  # noqa: E402


def _package(members: dict[str, bytes]) -> bytes:
    """A tarball shaped like the npm package: everything under ``package/``."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


GOOD = {f"package/{name}": f"content of {name}".encode() for name in viewer_assets.FILES}
GOOD["package/index.js"] = b'const mod = (await import("./re_viewer")).default;'


def test_assets_are_fetched_verified_and_unpacked_once(tmp_path):
    tarball = _package(GOOD)
    calls = []

    def fetch(url):
        calls.append(url)
        return tarball

    root = tmp_path / "assets"
    directory = viewer_assets.ensure_assets(root=root, fetch=fetch, integrity=viewer_assets.integrity_of(tarball))
    assert directory == root / viewer_assets.VIEWER_VERSION
    for name in viewer_assets.FILES:
        if name != "index.js":
            assert (directory / name).read_bytes() == f"content of {name}".encode()
    # The one patch: a browser needs the extension on the dynamic import.
    assert (directory / "index.js").read_bytes() == b'const mod = (await import("./re_viewer.js")).default;'
    assert viewer_assets.find_assets(root) == directory
    # A second call finds them and downloads nothing.
    viewer_assets.ensure_assets(root=root, fetch=fetch, integrity=viewer_assets.integrity_of(tarball))
    assert len(calls) == 1


def test_a_tarball_that_fails_its_checksum_installs_nothing(tmp_path):
    root = tmp_path / "assets"
    with pytest.raises(viewer_assets.ViewerAssetsError, match="pinned checksum"):
        viewer_assets.ensure_assets(root=root, fetch=lambda url: _package(GOOD), integrity="sha512-wrong")
    assert viewer_assets.find_assets(root) is None
    assert not any(root.iterdir()) if root.exists() else True        # no half-installed staging left behind


def test_a_package_without_the_expected_files_is_refused(tmp_path):
    tarball = _package({"package/index.js": GOOD["package/index.js"]})
    with pytest.raises(viewer_assets.ViewerAssetsError, match="no re_viewer.js"):
        viewer_assets.ensure_assets(root=tmp_path / "a", fetch=lambda url: tarball, integrity=viewer_assets.integrity_of(tarball))
    assert viewer_assets.find_assets(tmp_path / "a") is None


def test_a_package_that_is_not_the_pinned_one_is_not_patched_blindly(tmp_path):
    other = dict(GOOD)
    other["package/index.js"] = b"no dynamic import here"
    tarball = _package(other)
    with pytest.raises(viewer_assets.ViewerAssetsError, match="exactly once"):
        viewer_assets.ensure_assets(root=tmp_path / "a", fetch=lambda url: tarball, integrity=viewer_assets.integrity_of(tarball))
    assert viewer_assets.find_assets(tmp_path / "a") is None


def test_a_network_failure_says_what_to_do(tmp_path):
    def broken(url):
        raise OSError("name resolution failed")

    with pytest.raises(viewer_assets.ViewerAssetsError, match="run the installer again"):
        viewer_assets.ensure_assets(root=tmp_path / "a", fetch=broken)


def test_files_that_no_longer_match_the_manifest_are_not_trusted(tmp_path):
    tarball = _package(GOOD)
    root = tmp_path / "assets"
    directory = viewer_assets.ensure_assets(root=root, fetch=lambda u: tarball, integrity=viewer_assets.integrity_of(tarball))
    (directory / "re_viewer_bg.wasm").write_bytes(b"truncated")
    assert viewer_assets.find_assets(root) is None
    (directory / "re_viewer_bg.wasm").unlink()
    assert viewer_assets.find_assets(root) is None


def test_the_pinned_integrity_is_the_one_upstream_records():
    """4DAnyone's vendored viewer (fdanyone/space/assets/rerun/UPSTREAM.md) pins the same
    package at the same version with this hash; a mismatch here means one of them drifted."""

    assert viewer_assets.VIEWER_VERSION == "0.37.1"
    assert viewer_assets.TARBALL_INTEGRITY == (
        "sha512-po+PoR60eb9O9lexMdFUlgNnfvtohuKeBlPCWQI/6I9XU9tQZpcptQb6R4ZKsXPntnSQEZrFAtGVqlI74+wUPQ=="
    )


# --------------------------------------------------------------------------
# the recording
# --------------------------------------------------------------------------
def _tiny_ring(root: Path, cameras: int = 3, frames: int = 6):
    """A real ring on disk: cameras.json, metadata.json and small real mp4s (Rerun reads the
    videos' frame timestamps, so empty stand-ins will not do)."""

    import json

    import av
    import numpy as np

    from cumuli_bridge.ring import RingResult

    (root / "videos").mkdir(parents=True)
    records = []
    for index in range(cameras):
        yaw = index * 360.0 / cameras
        position = [2.0 * np.sin(np.radians(yaw)), 1.6, 2.0 * np.cos(np.radians(yaw))]
        c2w = np.eye(4)
        c2w[:3, 3] = position
        records.append({
            "camera_id": index, "layer_index": 0, "pitch": 15, "yaw": yaw,
            "K": [[100.0, 0, 32], [0, 100.0, 48], [0, 0, 1]],
            "camera_to_world": c2w.tolist(), "image_width": 64, "image_height": 96,
            "video": f"videos/{index:02d}.mp4",
        })
        with av.open(str(root / "videos" / f"{index:02d}.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=24)
            stream.width, stream.height, stream.pix_fmt = 64, 96, "yuv420p"
            for frame_index in range(frames):
                image = np.full((96, 64, 3), 40 + 30 * index + frame_index, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    (root / "cameras.json").write_text(json.dumps({"cameras": records}))
    (root / "metadata.json").write_text(json.dumps({"output": {"fps": "24/1", "frames_per_video": frames}}))
    return RingResult.load(root)


def test_a_ring_becomes_a_recording_with_every_camera_and_video(tmp_path):
    rr = pytest.importorskip("rerun")
    from cumuli_bridge import rerun_view

    ring = _tiny_ring(tmp_path / "ring")
    destination = tmp_path / "out" / "ring.rrd"
    summary = rerun_view.write_ring_recording(ring, destination, expected_version=rr.__version__)
    assert destination.is_file() and summary["cameras"] == 3 and summary["frames"] == 6
    assert summary["bytes"] == destination.stat().st_size > 0
    assert not list(destination.parent.glob(".*.rrd"))          # no staging file left behind

    # Entity paths and component names are readable in the file, which is a check that does not
    # depend on whichever readback API this SDK version has.
    data = destination.read_bytes()
    assert data[:4] == b"RRF2"
    for index in range(3):
        assert f"world/cameras/{index:02d}/image".encode() in data
    for name in (b"Pinhole", b"Transform3D", b"VideoFrameReference", b"AssetVideo"):
        assert name in data, name


def test_a_second_write_replaces_the_first_atomically(tmp_path):
    rr = pytest.importorskip("rerun")
    from cumuli_bridge import rerun_view

    ring = _tiny_ring(tmp_path / "ring")
    destination = tmp_path / "ring.rrd"
    rerun_view.write_ring_recording(ring, destination, expected_version=rr.__version__)
    first = destination.read_bytes()
    rerun_view.write_ring_recording(ring, destination, expected_version=rr.__version__)
    second = destination.read_bytes()
    assert second[:4] == b"RRF2" and abs(len(second) - len(first)) < 1024
    # Replaced in place, and no staging file is left beside it.
    assert [p.name for p in tmp_path.glob("*.rrd")] == ["ring.rrd"] and not list(tmp_path.glob(".*.rrd"))


def test_the_sdk_must_match_the_pinned_viewer(tmp_path):
    rr = pytest.importorskip("rerun")
    from cumuli_bridge import rerun_view

    ring = _tiny_ring(tmp_path / "ring")
    problem = rerun_view.sdk_problem("0.0.1")
    assert f"rerun-sdk {rr.__version__} is installed but the bundled viewer is 0.0.1" in problem
    assert "pip install rerun-sdk==0.0.1" in problem
    with pytest.raises(rerun_view.RerunError, match="may not load"):
        rerun_view.write_ring_recording(ring, tmp_path / "x.rrd", expected_version="0.0.1")
    assert not (tmp_path / "x.rrd").exists()
    assert rerun_view.sdk_problem(rr.__version__) is None


def test_a_recording_is_reused_until_the_ring_changes(tmp_path):
    from cumuli_bridge import rerun_view
    from cumuli_bridge.ring import RingResult

    root = tmp_path / "ring"
    key = rerun_view.recording_key(_tiny_ring(root))
    assert key == rerun_view.recording_key(RingResult.load(root))             # unchanged ring, same key
    (root / "videos" / "01.mp4").write_bytes(b"regenerated and longer " * 100)  # a view was regenerated
    assert rerun_view.recording_key(RingResult.load(root)) != key
    cameras = root / "cameras.json"
    changed = cameras.read_text().replace('"yaw": 120.0', '"yaw": 121.0')
    cameras.write_text(changed)
    assert rerun_view.recording_key(RingResult.load(root)) not in (key,)


def test_the_sdk_version_is_the_viewer_version():
    from cumuli_bridge import rerun_view

    assert rerun_view.RERUN_VERSION == viewer_assets.VIEWER_VERSION


# --------------------------------------------------------------------------
# the routes
# --------------------------------------------------------------------------
def _serve(tmp_path, assets: bool = True):
    """An aiohttp app with the viewer routes, plus a helper that runs requests against it."""

    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from cumuli_bridge import viewer_routes

    directory = tmp_path / "assets"
    directory.mkdir()
    for name in viewer_assets.FILES:
        (directory / name).write_bytes(f"bytes of {name}".encode())
    def make_app():
        routes = web.RouteTableDef()
        viewer_routes.setup(routes, find_assets=lambda: directory if assets else None)
        app = web.Application()
        app.add_routes(routes)
        return app

    def run(*requests):
        async def go():
            # A fresh app per run: an aiohttp Application binds to the loop it first runs on.
            async with TestClient(TestServer(make_app())) as client:
                out = []
                for path in requests:
                    response = await client.get(path)
                    out.append((response.status, dict(response.headers), await response.read()))
                return out

        return asyncio.run(go())

    return viewer_routes, run


def test_the_page_is_served_with_a_policy_that_allows_only_this_origin(tmp_path):
    _routes, run = _serve(tmp_path)
    ((status, headers, body), (js_status, js_headers, _js)) = run("/cumuli/viewer/page", "/cumuli/viewer/start.js")
    assert status == 200 and headers["Content-Type"].startswith("text/html") and b"start.js" in body
    policy = headers["Content-Security-Policy"]
    # No other origin can be reached: not Google Fonts, not GitHub's release API, not telemetry.
    assert "connect-src 'self' blob: data:" in policy and "default-src 'none'" in policy
    assert "http" not in policy.replace("'self'", "")
    assert "'unsafe-inline'" not in policy.split("script-src")[1].split(";")[0]    # no inline script
    assert "wasm-unsafe-eval" in policy and "frame-ancestors 'self'" in policy
    assert js_status == 200 and js_headers["Content-Type"].startswith("text/javascript")
    assert "Content-Security-Policy" in js_headers


def test_viewer_files_are_served_by_exact_name_with_the_right_types(tmp_path):
    _routes, run = _serve(tmp_path)
    results = run(
        "/cumuli/viewer/assets/index.js", "/cumuli/viewer/assets/re_viewer_bg.wasm",
        "/cumuli/viewer/assets/package.json", "/cumuli/viewer/assets/..%2Fsecret",
    )
    (js, wasm, other, traversal) = results
    assert js[0] == 200 and js[1]["Content-Type"].startswith("text/javascript") and js[2] == b"bytes of index.js"
    assert wasm[0] == 200 and wasm[1]["Content-Type"] == "application/wasm"
    assert other[0] == 404 and traversal[0] == 404       # only the three named files exist as far as the route knows


def test_missing_viewer_files_say_how_to_get_them(tmp_path):
    _routes, run = _serve(tmp_path, assets=False)
    ((status, _headers, body),) = run("/cumuli/viewer/assets/index.js")
    assert status == 404 and b"install.sh --groups viewer" in body


def test_only_a_registered_recording_is_served(tmp_path):
    routes, run = _serve(tmp_path)
    recording = tmp_path / "ring.rrd"
    recording.write_bytes(b"RRF2 data")
    other = tmp_path / "other.rrd"
    other.write_bytes(b"not registered")
    token = routes.register_recording(recording)
    assert token == routes.register_recording(recording)            # stable for the same file
    ((ok_status, ok_headers, ok_body), (unknown, _h, _b)) = run(f"/cumuli/viewer/rrd/{token}.rrd", "/cumuli/viewer/rrd/deadbeef.rrd")
    assert ok_status == 200 and ok_body == b"RRF2 data" and ok_headers["Content-Type"] == "application/octet-stream"
    assert unknown == 404
    recording.unlink()
    ((gone, _h2, _b2),) = run(f"/cumuli/viewer/rrd/{token}.rrd")
    assert gone == 404                                              # a deleted file is not served from a stale path
