"""Unit tests for the detection-box to reachability TTC adapter (B1/B2)."""

from __future__ import annotations

from math import inf

import numpy as np
from shapely.geometry import box

from autoware_ml.metrics.detection3d.collision import CollisionTTC
from autoware_ml.metrics.geometry.reachability import ReachabilityParams, VehicleGeometry
from autoware_ml.types.metrics import AgentKind

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


class _FakeMap:
    def __init__(self, polygon):
        self._polygon = polygon

    def region_union(self, tokens):
        return self._polygon

    def speed_at(self, x, y, default):
        return default


class _FakeProvider:
    """A single wide-open drivable region for every scene (map frame == base_link)."""

    def __init__(self, polygon):
        self._map = _FakeMap(polygon)

    def get(self, scene_token):
        return self._map

    def available(self, context):
        return True


def _adapter():
    road = box(-80.0, -60.0, 500.0, 60.0)
    return CollisionTTC(
        CLASS_NAMES,
        _FakeProvider(road),
        vehicle=EGO,
        params=ReachabilityParams(horizon_s=4.0, dt_s=0.1),
        max_speed_mps=10.0,
    )


IDENTITY = np.eye(4)  # ego at map origin, heading 0: base_link == map


def _box(cx, cy, yaw=0.0):
    return [cx, cy, 0.0, 4.0, 2.0, 1.5, yaw]


def test_adapter_class_dispatch_and_key_cases() -> None:
    adapter = _adapter()
    boxes = np.array(
        [
            _box(25.0, 0.0),  # car ahead: a braking lead -> finite
            _box(30.0, 0.0),  # barrier ahead: static -> finite
            _box(40.0, 0.0),  # oncoming truck (heading pi) -> finite
            _box(200.0, 0.0),  # far car: cheap reject -> inf
        ]
    )
    labels = np.array([0, 8, 1, 0])  # car, barrier, truck, car
    boxes[2, 6] = np.pi  # truck heading -x (oncoming)
    ttc = adapter.per_box_ttc(boxes, labels, IDENTITY, "scene-0")

    assert ttc.shape == (4,)
    assert ttc[0] != inf and ttc[0] <= 4.0  # the lead can brake, ego cannot know
    assert ttc[1] != inf and ttc[1] <= 4.0  # static barrier ahead
    assert ttc[2] != inf and ttc[2] <= 4.0  # oncoming
    assert ttc[3] == inf  # far


def test_adapter_empty_frame() -> None:
    adapter = _adapter()
    ttc = adapter.per_box_ttc(np.zeros((0, 7)), np.zeros((0,), dtype=int), IDENTITY, "scene-0")
    assert ttc.shape == (0,)


def test_adapter_rejects_unmapped_class() -> None:
    road = box(-10.0, -10.0, 10.0, 10.0)
    try:
        CollisionTTC(("car", "spaceship"), _FakeProvider(road), vehicle=EGO)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unmapped class")


def test_adapter_rejects_living_class_without_run_speed() -> None:
    road = box(-10.0, -10.0, 10.0, 10.0)
    try:
        CollisionTTC(
            ("car", "wheelchair"),
            _FakeProvider(road),
            vehicle=EGO,
            kinds={"car": "wheeled", "wheelchair": "living"},
        )
    except ValueError as error:
        assert "wheelchair" in str(error)
    else:
        raise AssertionError("expected ValueError for a living class without a run speed")


def test_adapter_reads_kind_names_into_the_enum() -> None:
    road = box(-10.0, -10.0, 10.0, 10.0)
    adapter = CollisionTTC(("car",), _FakeProvider(road), vehicle=EGO, kinds={"car": "wheeled"})
    assert adapter.kinds["car"] is AgentKind.WHEELED


def test_adapter_rejects_unknown_kind_value() -> None:
    road = box(-10.0, -10.0, 10.0, 10.0)
    try:
        CollisionTTC(
            ("car",),
            _FakeProvider(road),
            vehicle=EGO,
            kinds={"car": "hovercraft"},
        )
    except ValueError as error:
        assert "hovercraft" in str(error)
    else:
        raise AssertionError("expected ValueError for an unknown collision kind")
