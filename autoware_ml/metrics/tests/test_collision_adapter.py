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

"""Unit tests for the detection-box to reachability TTC adapter (B1/B2)."""

from __future__ import annotations

import unittest
from math import inf
from unittest import mock

import numpy as np
from shapely.geometry import box

from autoware_ml.metrics.detection3d.collision import CollisionTTC
from autoware_ml.metrics.geometry.reachability import EgoReachability, ReachabilityParams
from autoware_ml.metrics.tests.conftest import (
    CLASS_NAMES,
    EGO,
    STUB_MAP_PATH,
    collision_box,
    serve_fake_map,
    stubbed_lanelet_map_loader,
)
from autoware_ml.types.metrics import MotionModel

MOTION_MODELS = {
    "car": "wheeled",
    "truck": "wheeled",
    "bus": "wheeled",
    "train": "wheeled",
    "motorcycle": "wheeled",
    "bicycle": "omnidirectional",
    "pedestrian": "omnidirectional",
    "animal": "omnidirectional",
    "barrier": "static",
    "traffic_cone": "static",
    "debris": "static",
    "bicycle_rack": "static",
    "vehicle_extension": "static",
}
MAX_SPEEDS = {"pedestrian": 3.0, "animal": 4.0, "bicycle": 6.0}


def _adapter():
    serve_fake_map(box(-80.0, -60.0, 500.0, 60.0))
    return CollisionTTC(
        CLASS_NAMES,
        vehicle=EGO,
        motion_models=MOTION_MODELS,
        max_speeds=MAX_SPEEDS,
        params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
        max_speed_mps=10.0,
    )


IDENTITY = np.eye(4)  # ego at map origin, heading 0: base_link == map


def test_adapter_class_dispatch_and_key_cases() -> None:
    adapter = _adapter()
    boxes = np.array(
        [
            collision_box(25.0, 0.0),  # car ahead: a braking lead -> finite
            collision_box(30.0, 0.0),  # barrier ahead: static -> finite
            collision_box(40.0, 0.0),  # oncoming truck (heading pi) -> finite
            collision_box(200.0, 0.0),  # far car: cheap reject -> inf
        ]
    )
    labels = np.array([0, 8, 1, 0])  # car, barrier, truck, car
    boxes[2, 6] = np.pi  # truck heading -x (oncoming)
    ttc = adapter.per_box_ttc(boxes, labels, IDENTITY, STUB_MAP_PATH)

    assert ttc.shape == (4,)
    assert ttc[0] != inf and ttc[0] <= 4.0  # the lead can brake, ego cannot know
    assert ttc[1] != inf and ttc[1] <= 4.0  # static barrier ahead
    assert ttc[2] != inf and ttc[2] <= 4.0  # oncoming
    assert ttc[3] == inf  # far


class TestSharedEgoFrame(unittest.TestCase):
    """Unit tests for the ego frame that providers describing the same ego share."""

    def setUp(self) -> None:
        """Serve stub maps and count the objects the engine actually propagates."""
        self.enterContext(stubbed_lanelet_map_loader())
        self.propagate = self.enterContext(
            mock.patch.object(
                EgoReachability,
                "_propagate",
                autospec=True,
                side_effect=EgoReachability._propagate,
            )
        )

    def test_two_suites_propagate_the_boxes_of_a_frame_once(self) -> None:
        """
        Input: two identically configured providers, as two suites that report the same boxes
        under different prefixes hold them, asked for the same three boxes in the same ego frame.
        Expected: the first provider propagates each box once, the second reads the times of the
        shared ego frame and propagates nothing.
        Check: three propagations after both calls, and equal time arrays.
        """
        first, second = _adapter(), _adapter()
        boxes = np.array(
            [collision_box(25.0, 0.0), collision_box(30.0, 0.0), collision_box(40.0, 3.0)]
        )
        labels = np.array([0, 8, 5])  # car, barrier, bicycle

        ttc = first.per_box_ttc(boxes, labels, IDENTITY, STUB_MAP_PATH)
        self.assertEqual(self.propagate.call_count, 3)
        again = second.per_box_ttc(boxes, labels, IDENTITY, STUB_MAP_PATH)

        self.assertEqual(self.propagate.call_count, 3)
        np.testing.assert_array_equal(again, ttc)

    def test_a_provider_with_another_ego_shares_nothing(self) -> None:
        """
        Input: a barrier 30 m ahead, asked of a provider with the default ego and of one whose
        ego is capped at 2 m/s.
        Expected: the shared frame is keyed by everything that shapes the ego side, so the slow
        provider propagates the box itself instead of reusing the fast provider's time.
        Check: a time within the 4 s horizon for the fast ego, inf for the slow one, and two
        propagations.
        """
        fast = _adapter()
        slow = CollisionTTC(
            CLASS_NAMES,
            vehicle=EGO,
            motion_models=MOTION_MODELS,
            max_speeds=MAX_SPEEDS,
            params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
            max_speed_mps=2.0,
        )
        boxes = np.array([collision_box(30.0, 0.0)])
        labels = np.array([8])  # barrier

        self.assertLessEqual(fast.per_box_ttc(boxes, labels, IDENTITY, STUB_MAP_PATH)[0], 4.0)
        self.assertEqual(slow.per_box_ttc(boxes, labels, IDENTITY, STUB_MAP_PATH)[0], inf)
        self.assertEqual(self.propagate.call_count, 2)


def test_adapter_empty_frame() -> None:
    adapter = _adapter()
    ttc = adapter.per_box_ttc(np.zeros((0, 7)), np.zeros((0,), dtype=int), IDENTITY, STUB_MAP_PATH)
    assert ttc.shape == (0,)


def test_adapter_rejects_unmapped_class() -> None:
    try:
        CollisionTTC(
            ("car", "spaceship"),
            vehicle=EGO,
            motion_models={"car": "wheeled"},
            max_speeds={},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unmapped class")


def test_adapter_rejects_omnidirectional_class_without_max_speed() -> None:
    try:
        CollisionTTC(
            ("car", "wheelchair"),
            vehicle=EGO,
            motion_models={"car": "wheeled", "wheelchair": "omnidirectional"},
            max_speeds={},
        )
    except ValueError as error:
        assert "wheelchair" in str(error)
    else:
        raise AssertionError("expected ValueError for an omnidirectional class without a max speed")


def test_adapter_reads_kind_names_into_the_enum() -> None:
    adapter = CollisionTTC(("car",), vehicle=EGO, motion_models={"car": "wheeled"}, max_speeds={})
    assert adapter.motion_models["car"] is MotionModel.WHEELED


def test_adapter_rejects_unknown_kind_value() -> None:
    try:
        CollisionTTC(
            ("car",),
            vehicle=EGO,
            motion_models={"car": "hovercraft"},
            max_speeds={},
        )
    except ValueError as error:
        assert "hovercraft" in str(error)
    else:
        raise AssertionError("expected ValueError for an unknown motion model")
