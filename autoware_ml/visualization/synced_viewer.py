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

"""Serve the synced comparison page beside the Rerun web viewer.

Rerun cannot link the eyes of two 3D views, so the synced page embeds two
complete web viewers side by side, loads a one-view blueprint into each, and
mirrors the pointer, wheel, and keyboard input of one viewer into the other.
The embedded viewers are proxied through this server so they share an origin
with the page, which lets the page reach their canvases and viewer handles.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Line of the Rerun web viewer page that creates the viewer handle. The page
#: keeps the handle in a local variable, so it is exposed for the synced page.
HANDLE_MARKER = "let handle = new wasm_bindgen.WebHandle(options);"
HANDLE_EXPOSURE = HANDLE_MARKER + " window.rerunHandle = handle;"
PAGE_RESOURCE = "synced_viewer.html"
VIEWER_PREFIX = "/viewer/"
BLUEPRINT_PREFIX = "/blueprints/"
_FORWARDED_HEADERS = ("Content-Type", "Content-Encoding", "rerun-final-length")

Response = tuple[int, dict[str, str], bytes]


@dataclass(frozen=True)
class SyncedViewerManifest:
    """Describe the blueprint files the synced page can load."""

    comparisons: tuple[dict[str, str], ...] = ()
    camera_states: tuple[str, ...] = ()
    initial_comparison: str = ""
    initial_camera_state: str = "off"
    #: Camera state to blueprint file name.
    left: dict[str, str] = field(default_factory=dict)
    #: Comparison key to camera state to blueprint file name.
    right: dict[str, dict[str, str]] = field(default_factory=dict)


def load_page() -> bytes:
    """Return the synced comparison page."""
    return resources.files("autoware_ml.visualization").joinpath(PAGE_RESOURCE).read_bytes()


class SyncedViewerServer:
    """Serve the page, its blueprint files, and the proxied Rerun viewer."""

    def __init__(
        self,
        *,
        port: int,
        web_port: int,
        grpc_port: int,
        timeline: str,
        blueprint_dir: Path,
        host: str = "0.0.0.0",
    ) -> None:
        """Prepare the server without binding its port."""
        self.port = port
        self.web_port = web_port
        self.grpc_port = grpc_port
        self.timeline = timeline
        self.blueprint_dir = blueprint_dir
        self.host = host
        self._lock = threading.Lock()
        self._version = 0
        self._manifest = SyncedViewerManifest()
        self._server: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Return the address of the synced page."""
        return f"http://localhost:{self.port}/"

    @property
    def viewer_url(self) -> str:
        """Return the address of the Rerun web viewer that is proxied."""
        return f"http://127.0.0.1:{self.web_port}/"

    def start(self) -> None:
        """Bind the port and serve from a daemon thread."""
        if self._server is not None:
            return
        server = http.server.ThreadingHTTPServer((self.host, self.port), self._handler_class())
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="autoware-ml-synced-viewer", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        """Stop serving."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    def publish(self, manifest: SyncedViewerManifest) -> int:
        """Replace the manifest and return the new version."""
        with self._lock:
            self._version += 1
            self._manifest = manifest
            return self._version

    def config(self) -> dict[str, Any]:
        """Return what the page needs to connect and to load its blueprints."""
        with self._lock:
            manifest = asdict(self._manifest)
            version = self._version
        return {
            **manifest,
            "version": version,
            "grpc_port": self.grpc_port,
            "web_port": self.web_port,
            "timeline": self.timeline,
        }

    def read_blueprint(self, name: str) -> bytes | None:
        """Return one saved blueprint file, or ``None`` for anything else."""
        if name != Path(name).name or not name.endswith(".rbl"):
            return None
        path = self.blueprint_dir / name
        if not path.is_file():
            return None
        return path.read_bytes()

    def respond(self, raw_path: str) -> Response:
        """Route one GET request path to its response."""
        path = urllib.parse.urlsplit(raw_path).path
        if path == "/":
            return 200, {"Content-Type": "text/html; charset=utf-8"}, load_page()
        if path == "/config.json":
            return 200, {"Content-Type": "application/json"}, json.dumps(self.config()).encode()
        if path.startswith(BLUEPRINT_PREFIX):
            body = self.read_blueprint(path[len(BLUEPRINT_PREFIX) :])
            if body is None:
                return 404, {"Content-Type": "text/plain"}, b"unknown blueprint"
            return 200, {"Content-Type": "application/octet-stream"}, body
        if path == VIEWER_PREFIX.rstrip("/") or path.startswith(VIEWER_PREFIX):
            return self.fetch_viewer_asset(path[len(VIEWER_PREFIX) :])
        return 404, {"Content-Type": "text/plain"}, b"not found"

    def fetch_viewer_asset(self, asset: str) -> Response:
        """Proxy one file of the Rerun web viewer, exposing the handle on its page."""
        is_page = asset in ("", "index.html")
        upstream = self.viewer_url + ("" if is_page else asset)
        try:
            with urllib.request.urlopen(upstream, timeout=60) as response:
                body = response.read()
                headers = {
                    name: response.headers[name]
                    for name in _FORWARDED_HEADERS
                    if response.headers.get(name)
                }
        except urllib.error.HTTPError as error:
            return error.code, {"Content-Type": "text/plain"}, error.reason.encode()
        except urllib.error.URLError as error:
            logger.error("Rerun web viewer at %s is unreachable: %s", upstream, error.reason)
            return 502, {"Content-Type": "text/plain"}, b"rerun web viewer unreachable"
        if is_page:
            if HANDLE_MARKER.encode() not in body:
                logger.error(
                    "The Rerun web viewer page no longer contains %r; the synced page "
                    "cannot reach the viewer handle.",
                    HANDLE_MARKER,
                )
                return 502, {"Content-Type": "text/plain"}, b"unsupported rerun web viewer page"
            body = body.replace(HANDLE_MARKER.encode(), HANDLE_EXPOSURE.encode())
            headers["Cache-Control"] = "no-store"
        else:
            # The viewer bundle is large; let the browser keep it for a while.
            headers.setdefault("Cache-Control", "max-age=3600")
        headers.setdefault("Content-Type", "application/octet-stream")
        return 200, headers, body

    def _handler_class(self) -> type[http.server.BaseHTTPRequestHandler]:
        """Build the request handler bound to this server."""
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            """Translate GET requests into :meth:`SyncedViewerServer.respond`."""

            def do_GET(self) -> None:
                status, headers, body = server.respond(self.path)
                self.send_response(status)
                headers = {"Cache-Control": "no-store", **headers}
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("synced viewer: " + format, *args)

        return Handler
