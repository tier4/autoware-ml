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

"""Shared scaffolding for the collision metric tests.

The collision provider needs a vehicle, a class list and a lanelet map, and every
collision test needs the same ones. The map stub answers only what the provider
asks of it: the drivable union and the local speed limit.
"""

from __future__ import annotations

import pytest

from autoware_ml.metrics.geometry.reachability import VehicleGeometry

# Ego body in vehicle description terms, measured from the rear axle: 2.0 m wide,
# 3.79 m ahead of the reference point and 1.1 m behind it.
EGO = VehicleGeometry(
    wheel_base=2.79,
    front_overhang=1.0,
    rear_overhang=1.1,
    wheel_tread=1.64,
    left_overhang=0.18,
    right_overhang=0.18,
    max_steer_angle=0.64,
)

CLASS_NAMES = (
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "animal",
    "barrier",
    "traffic_cone",
    "debris",
    "bicycle_rack",
    "vehicle_extension",
)


def collision_box(cx: float, cy: float = 0.0, yaw: float = 0.0) -> list[float]:
    """A 4 x 2 x 1.5 m box row at ``(cx, cy)``, with the velocity columns the suite reads."""
    return [cx, cy, 0.0, 4.0, 2.0, 1.5, yaw, 0.0, 0.0]


class FakeLaneletMap:
    """One drivable polygon for every region, and the caller's own speed fallback."""

    def __init__(self, polygon) -> None:
        self._polygon = polygon

    def region_union(self, tokens):
        """The drivable polygon, whatever tokens are asked for."""
        return self._polygon

    def speed_at(self, x, y, default):
        """The caller's fallback: the stub carries no speed limits."""
        return default


class FakeMapProvider:
    """Serves one :class:`FakeLaneletMap` for every scene token."""

    cache_key = "fake"

    def __init__(self, polygon) -> None:
        self._map = FakeLaneletMap(polygon)

    def get(self, scene_token):
        """The stub map."""
        return self._map

    def available(self, scene_token):
        """Every scene has the stub map."""
        return True


@pytest.fixture
def fake_provider():
    """Factory for a map provider over a caller-supplied drivable polygon."""
    return FakeMapProvider
