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

"""Shared scaffolding for the map-backed metric tests.

The collision provider needs a vehicle, a class list and a lanelet map, and every
collision test needs the same ones. The map stub answers only what the provider
asks of it: the drivable union and the local speed limit.

A frame reaches the metrics with the path of its lanelet map, which the filters and
the collision provider load through ``LaneletMap.from_osm``. The autouse fixture
below stubs that loader with a registry, so a test serves a map under a path and
hands that path over in the frame context, and a frame without a map hands ``None``.
"""

from __future__ import annotations

import pytest

from autoware_ml.metrics.geometry.lanelet import LaneletMap
from autoware_ml.metrics.geometry.reachability import VehicleGeometry

# Map path every test hands over for a frame that has a map.
STUB_MAP_PATH = "stub://map"
_SERVED_MAPS: dict[str, object] = {}


@pytest.fixture(autouse=True)
def stub_lanelet_map_loader(monkeypatch: pytest.MonkeyPatch):
    """Load lanelet maps from the registry instead of OSM files, empty for every test."""
    _SERVED_MAPS.clear()
    load_osm = LaneletMap.from_osm

    def from_osm_or_served(cls, path):
        # A served path returns its stub. Any other path, an OSM file in a test's tmp_path or
        # the None of a map-less frame, still goes through the real loader and fails as loud.
        return _SERVED_MAPS[path] if path in _SERVED_MAPS else load_osm(path)

    monkeypatch.setattr(LaneletMap, "from_osm", classmethod(from_osm_or_served))
    yield
    _SERVED_MAPS.clear()


def serve_lanelet_map(lanelet_map: object, path: str = STUB_MAP_PATH) -> str:
    """Register the map the stubbed loader returns for ``path``.

    Args:
        lanelet_map: Map, real or stub, the loader returns.
        path: Map path a frame context names.

    Returns:
        The path, to hand over as the frame's ``lanelet_map_path``.
    """
    _SERVED_MAPS[path] = lanelet_map
    return path


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


def serve_fake_map(polygon, path: str = STUB_MAP_PATH) -> str:
    """Serve a :class:`FakeLaneletMap` over ``polygon`` under ``path``."""
    return serve_lanelet_map(FakeLaneletMap(polygon), path)
