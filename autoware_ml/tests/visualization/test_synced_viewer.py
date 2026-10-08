# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the synced comparison page server."""

from __future__ import annotations

import http.server
import json
import socket
import threading
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from autoware_ml.visualization.synced_viewer import (
    HANDLE_EXPOSURE,
    HANDLE_MARKER,
    SyncedViewerManifest,
    SyncedViewerServer,
    load_page,
)

_VIEWER_PAGE = f"<html><body><script>{HANDLE_MARKER}</script></body></html>".encode()
_VIEWER_SCRIPT = b"console.log('re_viewer');"


def _free_port() -> int:
    """Return a TCP port that is free right now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _FakeRerunWebServer:
    """Serve a stand-in for the Rerun web viewer bundle."""

    def __init__(self) -> None:
        self.page = _VIEWER_PAGE
        self.port = _free_port()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    body, content_type = fake.page, "text/html"
                elif self.path == "/re_viewer.js":
                    body, content_type = _VIEWER_SCRIPT, "application/javascript"
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("rerun-final-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream() -> Iterator[_FakeRerunWebServer]:
    fake = _FakeRerunWebServer()
    yield fake
    fake.close()


def _build_server(tmp_path: Path, web_port: int) -> SyncedViewerServer:
    return SyncedViewerServer(
        port=_free_port(),
        web_port=web_port,
        grpc_port=9876,
        timeline="frame",
        blueprint_dir=tmp_path,
    )


def test_page_and_config_are_served(tmp_path: Path, upstream: _FakeRerunWebServer) -> None:
    server = _build_server(tmp_path, upstream.port)

    status, headers, body = server.respond("/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert body == load_page()
    assert body.lower().startswith(b"<!doctype html")
    assert b"rerunHandle" in body

    status, headers, body = server.respond("/config.json?ts=1")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    config = json.loads(body)
    assert config["version"] == 0
    assert config["grpc_port"] == 9876
    assert config["timeline"] == "frame"
    assert config["left"] == {}
    assert config["right"] == {}

    version = server.publish(
        SyncedViewerManifest(
            comparisons=({"key": "gt", "label": "GT · Multi"},),
            camera_states=("off",),
            initial_comparison="gt",
            initial_camera_state="off",
            left={"off": "left-off.rbl"},
            right={"gt": {"off": "right-gt-off.rbl"}},
        )
    )
    assert version == 1
    config = json.loads(server.respond("/config.json")[2])
    assert config["version"] == 1
    assert config["comparisons"] == [{"key": "gt", "label": "GT · Multi"}]
    assert config["left"] == {"off": "left-off.rbl"}
    assert config["right"] == {"gt": {"off": "right-gt-off.rbl"}}


def test_blueprints_are_served_only_from_their_directory(
    tmp_path: Path, upstream: _FakeRerunWebServer
) -> None:
    (tmp_path / "left-off.rbl").write_bytes(b"rbl")
    (tmp_path / "notes.txt").write_bytes(b"not a blueprint")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "left-off.rbl").write_bytes(b"nested")
    server = _build_server(tmp_path, upstream.port)

    status, headers, body = server.respond("/blueprints/left-off.rbl?v=1&n=2")
    assert status == 200
    assert headers["Content-Type"] == "application/octet-stream"
    assert body == b"rbl"
    for name in ("missing.rbl", "notes.txt", "../left-off.rbl", "nested/left-off.rbl", ""):
        assert server.respond(f"/blueprints/{name}")[0] == 404, name


def test_viewer_is_proxied_with_the_handle_exposed(
    tmp_path: Path, upstream: _FakeRerunWebServer
) -> None:
    server = _build_server(tmp_path, upstream.port)

    for path in ("/viewer", "/viewer/", "/viewer/index.html"):
        status, headers, body = server.respond(path)
        assert status == 200, path
        assert HANDLE_EXPOSURE.encode() in body
        assert HANDLE_MARKER.encode() + b"</script>" not in body
        assert headers["Cache-Control"] == "no-store"

    status, headers, body = server.respond("/viewer/re_viewer.js")
    assert status == 200
    assert body == _VIEWER_SCRIPT
    assert headers["Content-Type"] == "application/javascript"
    assert headers["rerun-final-length"] == str(len(_VIEWER_SCRIPT))
    assert headers["Cache-Control"] == "max-age=3600"

    assert server.respond("/viewer/missing.js")[0] == 404
    assert server.respond("/elsewhere")[0] == 404


def test_viewer_page_without_the_handle_line_is_refused(
    tmp_path: Path, upstream: _FakeRerunWebServer
) -> None:
    upstream.page = b"<html><body><script>let handle = somethingElse();</script></body></html>"
    server = _build_server(tmp_path, upstream.port)

    status, _, body = server.respond("/viewer/")
    assert status == 502
    assert b"unsupported" in body


def test_unreachable_viewer_is_reported(tmp_path: Path) -> None:
    server = _build_server(tmp_path, _free_port())

    status, _, body = server.respond("/viewer/re_viewer.js")
    assert status == 502
    assert b"unreachable" in body


def test_server_serves_over_http(tmp_path: Path, upstream: _FakeRerunWebServer) -> None:
    (tmp_path / "left-off.rbl").write_bytes(b"rbl")
    server = _build_server(tmp_path, upstream.port)
    server.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/config.json") as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.load(response)["timeline"] == "frame"
        with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/viewer/") as response:
            assert HANDLE_EXPOSURE.encode() in response.read()
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.port}/blueprints/left-off.rbl"
        ) as response:
            assert response.read() == b"rbl"
        assert server.url == f"http://localhost:{server.port}/"
    finally:
        server.close()
    server.close()
