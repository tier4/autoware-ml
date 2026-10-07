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
from typing import Any

import pytest

from autoware_ml.visualization.backends import (
    NoOpVisualizationBackend,
    create_visualization_backend,
)
from autoware_ml.visualization.contracts import VisualizationSessionConfig
from autoware_ml.visualization.events import ScalarEvent


def test_create_visualization_backend_returns_the_noop_backend() -> None:
    backend = create_visualization_backend(VisualizationSessionConfig(backend="noop"))

    assert isinstance(backend, NoOpVisualizationBackend)


def test_noop_backend_accepts_the_full_backend_protocol() -> None:
    backend = NoOpVisualizationBackend()

    backend.set_step(1)
    backend.log_event(ScalarEvent("metrics/value", 1.0))
    backend.log_events([ScalarEvent("metrics/value", 2.0)])
    backend.wait_until_interrupted()


def test_noop_backend_consumes_lazy_event_iterables() -> None:
    """``log_events`` must drain generators so adapter work is not left pending."""
    consumed: list[int] = []

    def events() -> Any:
        for index in range(3):
            consumed.append(index)
            yield ScalarEvent("metrics/value", float(index))

    NoOpVisualizationBackend().log_events(events())

    assert consumed == [0, 1, 2]


def test_create_visualization_backend_rejects_unknown_backends() -> None:
    with pytest.raises(ValueError, match="Unknown visualization backend: sqlite"):
        create_visualization_backend(VisualizationSessionConfig(backend="sqlite"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"web_port": 0},
        {"grpc_port": 65536},
        {"web_port": 9090, "grpc_port": 9090},
        {"application_id": ""},
        {"timeline": ""},
        {"server_memory_limit": ""},
        {"point_color_mode": "rainbow"},
    ],
)
def test_session_config_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        VisualizationSessionConfig(**kwargs)


def test_noop_backend_does_not_require_the_rerun_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke tests must exercise the preview pipeline without the Rerun SDK."""
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "rerun" or name.startswith("rerun."):
            raise ImportError(f"{name} is unavailable in this environment")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    backend = create_visualization_backend(VisualizationSessionConfig(backend="noop"))
    backend.log_events([ScalarEvent("metrics/value", 1.0)])

    assert isinstance(backend, NoOpVisualizationBackend)
