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

"""Unit tests for reading lanelet2 maps into region polygons."""

from __future__ import annotations

import numpy as np
import pytest

from shapely.geometry import Polygon

from autoware_ml.metrics.geometry.lanelet import (
    LaneletMap,
    LaneletMapProvider,
    load_lanelet_speeds,
    load_region_polygons,
)


def _osm(nodes: dict[int, tuple[float, float]], bodies: str) -> str:
    """Assemble a minimal lanelet2 OSM around the given nodes and elements."""
    node_xml = "\n".join(
        f'  <node id="{node_id}"><tag k="local_x" v="{x}"/><tag k="local_y" v="{y}"/></node>'
        for node_id, (x, y) in nodes.items()
    )
    return f'<?xml version="1.0"?>\n<osm>\n{node_xml}\n{bodies}\n</osm>\n'


_CROSSED_LANELET = _osm(
    {1: (0.0, 0.0), 2: (10.0, 4.0), 3: (0.0, 4.0), 4: (10.0, 0.0)},
    """  <way id="100"><nd ref="1"/><nd ref="2"/></way>
  <way id="101"><nd ref="3"/><nd ref="4"/></way>
  <relation id="200">
    <member type="way" ref="100" role="left"/>
    <member type="way" ref="101" role="right"/>
    <tag k="type" v="lanelet"/><tag k="subtype" v="road"/>
  </relation>""",
)


def test_a_self_crossing_ring_keeps_every_lobe(tmp_path) -> None:
    """Repairing a figure-eight ring must not shrink the region to one lobe."""
    path = tmp_path / "map.osm"
    path.write_text(_CROSSED_LANELET)

    regions = load_region_polygons(str(path))

    road = regions["road"]
    assert len(road) == 2
    assert sum(polygon.area for polygon in road) == pytest.approx(20.0)


def _lanelet_with_speed(speed: str) -> str:
    """A single road lanelet carrying the given raw ``speed_limit`` tag."""
    return _osm(
        {1: (0.0, 0.0), 2: (10.0, 0.0), 3: (0.0, 4.0), 4: (10.0, 4.0)},
        f"""  <way id="100"><nd ref="3"/><nd ref="4"/></way>
  <way id="101"><nd ref="1"/><nd ref="2"/></way>
  <relation id="200">
    <member type="way" ref="100" role="left"/>
    <member type="way" ref="101" role="right"/>
    <tag k="type" v="lanelet"/><tag k="subtype" v="road"/>
    <tag k="speed_limit" v="{speed}"/>
  </relation>""",
    )


@pytest.mark.parametrize(
    ("speed", "message"),
    [("thirty", "unparsable speed_limit"), ("0", "non-positive speed_limit")],
)
def test_a_corrupt_speed_limit_raises(tmp_path, speed: str, message: str) -> None:
    """A speed limit that is not a positive number is map corruption, not a default."""
    path = tmp_path / "map.osm"
    path.write_text(_lanelet_with_speed(speed))

    with pytest.raises(ValueError, match=message):
        load_lanelet_speeds(str(path))


def test_a_valid_speed_limit_converts_to_mps(tmp_path) -> None:
    path = tmp_path / "map.osm"
    path.write_text(_lanelet_with_speed("36"))

    speeds = load_lanelet_speeds(str(path))

    assert len(speeds) == 1
    assert speeds[0][1] == pytest.approx(10.0)  # 36 km/h


def test_speed_at_takes_the_highest_limit_of_overlapping_lanelets() -> None:
    """Worst case: the faster lane bounds how far an agent can reach."""
    slow = Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)])
    fast = Polygon([(5.0, 0.0), (15.0, 0.0), (15.0, 4.0), (5.0, 4.0)])
    lanelet_map = LaneletMap({"road": [slow, fast]}, [(slow, 8.3), (fast, 16.7)])

    assert lanelet_map.speed_at(7.0, 2.0, default=99.0) == pytest.approx(16.7)  # overlap
    assert lanelet_map.speed_at(2.0, 2.0, default=99.0) == pytest.approx(8.3)  # slow lane only
    assert lanelet_map.speed_at(50.0, 50.0, default=99.0) == pytest.approx(99.0)  # off lane


def test_the_margin_sign_picks_the_direction() -> None:
    """Negative erodes the mapped border, positive claims off-map space."""
    lanelet_map = LaneletMap({"road": [Polygon([(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)])]})
    near_border = np.array([[5.0, 0.2]])  # inside, 0.2 m from the outer border
    off_map = np.array([[5.0, -0.3]])  # outside, 0.3 m from the border

    assert lanelet_map.contains(("road",), near_border).tolist() == [True]
    assert lanelet_map.contains(("road",), near_border, margin=-0.5).tolist() == [False]
    assert lanelet_map.contains(("road",), off_map).tolist() == [False]
    assert lanelet_map.contains(("road",), off_map, margin=0.5).tolist() == [True]


def test_the_provider_shares_one_parsed_map_per_path(tmp_path) -> None:
    """Parsing is cached by path, so the provider needs no cache of its own."""
    path = tmp_path / "map.osm"
    path.write_text(_lanelet_with_speed("36"))
    provider = LaneletMapProvider(lambda scene_token: str(path))

    assert provider.get("scene-a") is provider.get("scene-a")
    assert provider.get("scene-b") is provider.get("scene-a")  # same path, same map
