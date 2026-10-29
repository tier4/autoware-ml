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

"""Pytest fixtures shared by the visualization test suites."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.visualization.tests.doubles import RecordingBackend


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
