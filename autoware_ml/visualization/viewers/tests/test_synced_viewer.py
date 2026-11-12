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
from pathlib import Path
import socket
import tempfile
import threading
import unittest
import urllib.request

from autoware_ml.visualization.viewers.synced_viewer import (
    HANDLE_EXPOSURE,
    HANDLE_MARKER,
    SyncedViewerManifest,
    SyncedViewerServer,
    load_page,
)

_VIEWER_PAGE = f"<html><body><script>{HANDLE_MARKER}</script></body></html>".encode()
_VIEWER_SCRIPT = b"console.log('re_viewer');"
_SYNCED_VIEWER_LOGGER = "autoware_ml.visualization.viewers.synced_viewer"


def _free_port() -> int:
    """Return a TCP port that is free right now."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _FakeRerunWebHandler(http.server.BaseHTTPRequestHandler):
    """Answer requests for the stand-in Rerun web viewer page and bundle."""

    server: _FakeRerunWebServer

    def do_GET(self) -> None:
        """Serve the viewer page at ``/``, the bundle at ``/re_viewer.js``, and 404 otherwise."""
        if self.path == "/":
            body, content_type = self.server.page, "text/html"
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

    def log_message(self, format: str, *args: object) -> None:
        """Keep the request log out of the test output."""
        del format, args


class _FakeRerunWebServer(http.server.ThreadingHTTPServer):
    """Serve a stand-in for the Rerun web viewer bundle from a daemon thread."""

    daemon_threads = True

    def __init__(self) -> None:
        """Bind a free local port; the served viewer page can be replaced through ``page``."""
        self.page = _VIEWER_PAGE
        super().__init__(("127.0.0.1", 0), _FakeRerunWebHandler)
        self.port = self.server_address[1]
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    def start(self) -> None:
        """Start serving."""
        self._thread.start()

    def close(self) -> None:
        """Stop serving and release the port."""
        self.shutdown()
        self.server_close()


class _SyncedViewerTestCase(unittest.TestCase):
    """Provide a blueprint directory and a running stand-in for the Rerun web viewer."""

    def setUp(self) -> None:
        """Create the blueprint directory and start the stand-in viewer."""
        self.blueprint_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.upstream = _FakeRerunWebServer()
        self.upstream.start()
        self.addCleanup(self.upstream.close)

    def build_server(self, web_port: int | None = None) -> SyncedViewerServer:
        """Build a synced viewer server that proxies ``web_port`` (the stand-in by default)."""
        return SyncedViewerServer(
            port=_free_port(),
            web_port=self.upstream.port if web_port is None else web_port,
            grpc_port=9876,
            timeline="frame",
            blueprint_dir=self.blueprint_dir,
        )


class TestPageAndConfig(_SyncedViewerTestCase):
    """The page and its configuration are served and follow published manifests."""

    def test_page_and_config_are_served(self) -> None:
        """The page is served as HTML and the config reflects every published manifest."""
        server = self.build_server()

        status, headers, body = server.respond("/")
        self.assertEqual(status, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/html"))
        self.assertEqual(body, load_page())
        self.assertTrue(body.lower().startswith(b"<!doctype html"))
        self.assertIn(b"rerunHandle", body)

        status, headers, body = server.respond("/config.json?ts=1")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/json")
        config = json.loads(body)
        self.assertEqual(config["version"], 0)
        self.assertEqual(config["grpc_port"], 9876)
        self.assertEqual(config["timeline"], "frame")
        self.assertEqual(config["left"], {})
        self.assertEqual(config["right"], {})

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
        self.assertEqual(version, 1)
        config = json.loads(server.respond("/config.json")[2])
        self.assertEqual(config["version"], 1)
        self.assertEqual(config["comparisons"], [{"key": "gt", "label": "GT · Multi"}])
        self.assertEqual(config["left"], {"off": "left-off.rbl"})
        self.assertEqual(config["right"], {"gt": {"off": "right-gt-off.rbl"}})


class TestBlueprints(_SyncedViewerTestCase):
    """Blueprint files are served only from the blueprint directory."""

    def test_blueprints_are_served_only_from_their_directory(self) -> None:
        """Only ``.rbl`` files directly inside the blueprint directory are served."""
        (self.blueprint_dir / "left-off.rbl").write_bytes(b"rbl")
        (self.blueprint_dir / "notes.txt").write_bytes(b"not a blueprint")
        (self.blueprint_dir / "nested").mkdir()
        (self.blueprint_dir / "nested" / "left-off.rbl").write_bytes(b"nested")
        server = self.build_server()

        status, headers, body = server.respond("/blueprints/left-off.rbl?v=1&n=2")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        self.assertEqual(body, b"rbl")
        for name in ("missing.rbl", "notes.txt", "../left-off.rbl", "nested/left-off.rbl", ""):
            with self.subTest(name=name):
                self.assertEqual(server.respond(f"/blueprints/{name}")[0], 404)


class TestViewerProxy(_SyncedViewerTestCase):
    """The Rerun web viewer is proxied with its handle exposed."""

    def test_viewer_is_proxied_with_the_handle_exposed(self) -> None:
        """The viewer page exposes the handle and the bundle keeps its upstream headers."""
        server = self.build_server()

        for path in ("/viewer", "/viewer/", "/viewer/index.html"):
            with self.subTest(path=path):
                status, headers, body = server.respond(path)
                self.assertEqual(status, 200)
                self.assertIn(HANDLE_EXPOSURE.encode(), body)
                self.assertNotIn(HANDLE_MARKER.encode() + b"</script>", body)
                self.assertEqual(headers["Cache-Control"], "no-store")

        status, headers, body = server.respond("/viewer/re_viewer.js")
        self.assertEqual(status, 200)
        self.assertEqual(body, _VIEWER_SCRIPT)
        self.assertEqual(headers["Content-Type"], "application/javascript")
        self.assertEqual(headers["rerun-final-length"], str(len(_VIEWER_SCRIPT)))
        self.assertEqual(headers["Cache-Control"], "max-age=3600")

        self.assertEqual(server.respond("/viewer/missing.js")[0], 404)
        self.assertEqual(server.respond("/elsewhere")[0], 404)

    def test_viewer_page_without_the_handle_line_is_refused(self) -> None:
        """A viewer page that no longer creates the handle is answered with 502."""
        self.upstream.page = (
            b"<html><body><script>let handle = somethingElse();</script></body></html>"
        )
        server = self.build_server()

        with self.assertLogs(_SYNCED_VIEWER_LOGGER, level="ERROR") as logs:
            status, _, body = server.respond("/viewer/")
        self.assertEqual(status, 502)
        self.assertIn(b"unsupported", body)
        self.assertIn("no longer contains", logs.output[0])

    def test_unreachable_viewer_is_reported(self) -> None:
        """A viewer that does not answer is reported with 502."""
        server = self.build_server(web_port=_free_port())

        with self.assertLogs(_SYNCED_VIEWER_LOGGER, level="ERROR") as logs:
            status, _, body = server.respond("/viewer/re_viewer.js")
        self.assertEqual(status, 502)
        self.assertIn(b"unreachable", body)
        self.assertIn("unreachable", logs.output[0])


class TestHttpServing(_SyncedViewerTestCase):
    """The synced viewer server answers real HTTP requests."""

    def test_server_serves_over_http(self) -> None:
        """Config, proxied viewer, and blueprints are reachable over HTTP; close is idempotent."""
        (self.blueprint_dir / "left-off.rbl").write_bytes(b"rbl")
        server = self.build_server()
        server.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/config.json") as response:
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertEqual(json.load(response)["timeline"], "frame")
            with urllib.request.urlopen(f"http://127.0.0.1:{server.port}/viewer/") as response:
                self.assertIn(HANDLE_EXPOSURE.encode(), response.read())
            with urllib.request.urlopen(
                f"http://127.0.0.1:{server.port}/blueprints/left-off.rbl"
            ) as response:
                self.assertEqual(response.read(), b"rbl")
            self.assertEqual(server.url, f"http://localhost:{server.port}/")
        finally:
            server.close()
        server.close()


if __name__ == "__main__":
    unittest.main()
