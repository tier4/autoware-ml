"""Unit tests for the detection-box to reachability TTC adapter (B1/B2)."""

from __future__ import annotations

from math import inf

import numpy as np
from shapely.geometry import box

from autoware_ml.metrics.detection3d.collision import CollisionTTC
from autoware_ml.metrics.geometry.reachability import ReachabilityParams
from autoware_ml.tests.metrics.conftest import CLASS_NAMES, EGO, FakeMapProvider, collision_box
from autoware_ml.types.metrics import AgentKind


def _adapter():
    road = box(-80.0, -60.0, 500.0, 60.0)
    return CollisionTTC(
        CLASS_NAMES,
        FakeMapProvider(road),
        vehicle=EGO,
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
        CollisionTTC(("car", "spaceship"), FakeMapProvider(road), vehicle=EGO)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unmapped class")


def test_adapter_rejects_living_class_without_run_speed() -> None:
    road = box(-10.0, -10.0, 10.0, 10.0)
    try:
        CollisionTTC(
            ("car", "wheelchair"),
            FakeMapProvider(road),
            vehicle=EGO,
            kinds={"car": "wheeled", "wheelchair": "living"},
        )
    except ValueError as error:
        assert "wheelchair" in str(error)
    else:
        raise AssertionError("expected ValueError for a living class without a run speed")


def test_adapter_reads_kind_names_into_the_enum() -> None:
    road = box(-10.0, -10.0, 10.0, 10.0)
    adapter = CollisionTTC(("car",), FakeMapProvider(road), vehicle=EGO, kinds={"car": "wheeled"})
    assert adapter.kinds["car"] is AgentKind.WHEELED


def test_adapter_rejects_unknown_kind_value() -> None:
    road = box(-10.0, -10.0, 10.0, 10.0)
    try:
        CollisionTTC(
            ("car",),
            FakeMapProvider(road),
            vehicle=EGO,
            kinds={"car": "hovercraft"},
        )
    except ValueError as error:
        assert "hovercraft" in str(error)
    else:
        raise AssertionError("expected ValueError for an unknown collision kind")
