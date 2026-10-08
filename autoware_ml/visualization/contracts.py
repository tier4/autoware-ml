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

"""Shared visualization backend contracts and configuration types."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from autoware_ml.visualization.events import VisualizationEvent


@dataclass(frozen=True)
class VisualizationSessionConfig:
    """Configure one visualization recording."""

    backend: Literal["rerun", "noop"] = "rerun"
    application_id: str = "autoware-ml"
    recording_id: str | None = None
    web_port: int = 9090
    grpc_port: int = 9876
    wait: bool = True
    server_memory_limit: str = "25%"
    timeline: str = "frame"
    point_color_mode: Literal["semantic", "intensity", "solid"] = "semantic"
    camera_frustums_visible: bool = False
    #: Serve the synced comparison page, which keeps two 3D views on one eye.
    sync_views: bool = False
    sync_port: int = 9091

    def __post_init__(self) -> None:
        """Reject invalid backend settings before any server is started."""
        if self.backend not in {"rerun", "noop"}:
            raise ValueError(f"Unknown visualization backend: {self.backend}")
        ports = [("web_port", self.web_port), ("grpc_port", self.grpc_port)]
        if self.sync_views:
            ports.append(("sync_port", self.sync_port))
        for name, port in ports:
            if not 1 <= port <= 65535:
                raise ValueError(f"{name} must be between 1 and 65535")
        if len({port for _, port in ports}) != len(ports):
            raise ValueError("web_port, grpc_port, and sync_port must be different")
        if not self.application_id.strip():
            raise ValueError("application_id must not be empty")
        if not self.timeline.strip():
            raise ValueError("timeline must not be empty")
        if not self.server_memory_limit.strip():
            raise ValueError("server_memory_limit must not be empty")
        if self.point_color_mode not in {"semantic", "intensity", "solid"}:
            raise ValueError("point color mode must be 'semantic', 'intensity', or 'solid'")


class VisualizationBackend(Protocol):
    """Interface implemented by visualization backends."""

    def set_step(self, step: int) -> None:
        """Advance the backend timeline to one integer step."""

    def set_timestamp(self, timestamp: float) -> None:
        """Set the sensor timestamp for the current frame."""

    def log_event(self, event: VisualizationEvent) -> None:
        """Log one visualization event."""

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Log multiple visualization events."""

    def wait_until_interrupted(self) -> None:
        """Block while an interactive viewer is served, or return immediately."""
