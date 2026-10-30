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

"""Tests for visualization backend selection."""

from __future__ import annotations

import builtins
from collections.abc import Iterator, Mapping
from types import ModuleType
import unittest
from unittest.mock import patch

from autoware_ml.visualization.viewers.backends import (
    NoOpVisualizationBackend,
    create_visualization_backend,
)
from autoware_ml.visualization.viewers.contracts import VisualizationSessionConfig
from autoware_ml.visualization.viewers.events import ScalarEvent

_REAL_IMPORT = builtins.__import__


def _import_without_rerun(name: str, *args: object, **kwargs: object) -> ModuleType:
    """Import a module as usual, except that the Rerun SDK is unavailable."""
    if name == "rerun" or name.startswith("rerun."):
        raise ImportError(f"{name} is unavailable in this environment")
    return _REAL_IMPORT(name, *args, **kwargs)


class TestBackendFactory(unittest.TestCase):
    """The factory builds the configured backend and rejects unknown ones."""

    def test_create_visualization_backend_returns_the_noop_backend(self) -> None:
        """The ``noop`` backend name builds the no-op backend."""
        backend = create_visualization_backend(VisualizationSessionConfig(backend="noop"))

        self.assertIsInstance(backend, NoOpVisualizationBackend)

    def test_create_visualization_backend_rejects_unknown_backends(self) -> None:
        """An unknown backend name is refused."""
        with self.assertRaisesRegex(ValueError, "Unknown visualization backend: sqlite"):
            create_visualization_backend(VisualizationSessionConfig(backend="sqlite"))

    def test_noop_backend_does_not_require_the_rerun_sdk(self) -> None:
        """Smoke tests must exercise the preview pipeline without the Rerun SDK."""
        with patch.object(builtins, "__import__", _import_without_rerun):
            backend = create_visualization_backend(VisualizationSessionConfig(backend="noop"))
            backend.log_events([ScalarEvent("metrics/value", 1.0)])

        self.assertIsInstance(backend, NoOpVisualizationBackend)


class TestNoOpBackend(unittest.TestCase):
    """The no-op backend accepts the backend protocol and drops every event."""

    def test_noop_backend_accepts_the_full_backend_protocol(self) -> None:
        """Every protocol method can be called without a viewer."""
        backend = NoOpVisualizationBackend()

        backend.set_step(1)
        backend.set_timestamp(1.5)
        backend.log_event(ScalarEvent("metrics/value", 1.0))
        backend.log_events([ScalarEvent("metrics/value", 2.0)])
        backend.wait_until_interrupted()

    def test_noop_backend_consumes_lazy_event_iterables(self) -> None:
        """``log_events`` must drain generators so adapter work is not left pending."""
        consumed: list[int] = []

        def events() -> Iterator[ScalarEvent]:
            for index in range(3):
                consumed.append(index)
                yield ScalarEvent("metrics/value", float(index))

        NoOpVisualizationBackend().log_events(events())

        self.assertEqual(consumed, [0, 1, 2])


class TestSessionConfig(unittest.TestCase):
    """The session configuration validates its settings."""

    def test_session_config_rejects_invalid_values(self) -> None:
        """Out-of-range ports, clashing ports, empty names, and unknown modes are refused."""
        invalid: tuple[Mapping[str, object], ...] = (
            {"web_port": 0},
            {"grpc_port": 65536},
            {"web_port": 9090, "grpc_port": 9090},
            {"application_id": ""},
            {"timeline": ""},
            {"server_memory_limit": ""},
            {"point_color_mode": "rainbow"},
            {"sync_views": True, "sync_port": 0},
            {"sync_views": True, "sync_port": 9090},
            {"sync_views": True, "grpc_port": 9091},
        )
        for kwargs in invalid:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                VisualizationSessionConfig(**kwargs)

    def test_session_config_ignores_the_sync_port_while_the_page_is_disabled(self) -> None:
        """A clashing sync port is accepted while the synced page is off."""
        config = VisualizationSessionConfig(sync_port=9090)

        self.assertFalse(config.sync_views)
        self.assertEqual(config.sync_port, 9090)


if __name__ == "__main__":
    unittest.main()
