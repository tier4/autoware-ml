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

"""Shared fixtures and doubles for the visualization test suite."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from autoware_ml.visualization.events import VisualizationEvent


class RecordingBackend:
    """Capture every event and step handed to a visualization backend."""

    def __init__(self) -> None:
        """Initialize empty step and event logs."""
        self.steps: list[int] = []
        self.timestamps: list[float] = []
        self.events: list[Any] = []
        self.waited = False

    def set_step(self, step: int) -> None:
        """Record one timeline step."""
        self.steps.append(step)

    def set_timestamp(self, timestamp: float) -> None:
        """Record one sensor timestamp."""
        self.timestamps.append(timestamp)

    def log_event(self, event: VisualizationEvent) -> None:
        """Record one visualization event."""
        self.events.append(event)

    def log_events(self, events: Iterable[VisualizationEvent]) -> None:
        """Record multiple visualization events."""
        self.events.extend(events)

    def wait_until_interrupted(self) -> None:
        """Record that the preview asked the backend to wait."""
        self.waited = True

    def paths_of(self, event_type: type) -> list[str]:
        """Return the entity paths of every recorded event of one type."""
        return [event.path for event in self.events if isinstance(event, event_type)]


@pytest.fixture
def recording_backend() -> RecordingBackend:
    """Return a fresh recording backend."""
    return RecordingBackend()


@pytest.fixture
def preview_calibration() -> dict[str, np.ndarray]:
    """Return the camera matrices of a calibration sample with an identity-rotation extrinsic."""
    return dict(
        image_from_camera=np.array(
            [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        lidar_to_camera=np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
