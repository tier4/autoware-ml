"""Tests for 3D box transforms."""

from __future__ import annotations

import numpy as np
import pytest

from autoware_ml.transforms.boxes3d.annotations import normalize_filter_attributes
from autoware_ml.transforms.boxes3d.filters import (
    ObjectMinPointsFilter,
    ObjectNameFilter,
    ObjectRangeFilter,
    ObjectRangeMinPointsFilter,
)
from autoware_ml.transforms.boxes3d.loading import LoadAnnotations3D


def test_object_filters() -> None:
    sample = {
        "gt_boxes": np.array(
            [[0.0, 0.0, 0.0, 1, 1, 1, 0], [10.0, 0.0, 0.0, 1, 1, 1, 0]],
            dtype=np.float32,
        ),
        "gt_names": np.array(["car", "pedestrian"]),
        "coord": np.array(
            [
                [0.1, 0.1, 0.0],
                [-0.2, 0.0, 0.0],
                [0.0, -0.2, 0.0],
                [10.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    named = ObjectNameFilter(classes=["car"])(sample.copy())
    ranged = ObjectRangeFilter(point_cloud_range=[-1.0, -1.0, -1.0, 5.0, 5.0, 5.0])(sample.copy())
    min_points = ObjectMinPointsFilter(min_num_points=3, time_lag_dim=None)(sample.copy())

    assert named["gt_names"].tolist() == ["car"]
    assert ranged["gt_names"].tolist() == ["car"]
    assert min_points["gt_names"].tolist() == ["car"]


def test_object_filters_keep_gt_num_points_aligned() -> None:
    sample = {
        "gt_boxes": np.array(
            [[0.0, 0.0, 0.0, 1, 1, 1, 0], [10.0, 0.0, 0.0, 1, 1, 1, 0]],
            dtype=np.float32,
        ),
        "gt_names": np.array(["car", "pedestrian"]),
        "gt_labels": np.array([0, 1], dtype=np.int64),
        "gt_num_points": np.array([7, 3], dtype=np.int64),
        "coord": np.array([[0.1, 0.1, 0.0]], dtype=np.float32),
    }

    named = ObjectNameFilter(classes=["car"])(sample.copy())
    ranged = ObjectRangeFilter(point_cloud_range=[-1.0, -1.0, -1.0, 5.0, 5.0, 5.0])(sample.copy())
    min_points = ObjectMinPointsFilter(min_num_points=1, time_lag_dim=None)(sample.copy())

    assert named["gt_num_points"].tolist() == [7]
    assert ranged["gt_num_points"].tolist() == [7]
    assert min_points["gt_num_points"].tolist() == [7]


def test_object_min_points_filter_counts_points_inside_rotated_boxes() -> None:
    sample = {
        "gt_boxes": np.array(
            [[0.0, 0.0, 0.0, 2.0, 1.0, 2.0, np.pi / 2]],
            dtype=np.float32,
        ),
        "gt_names": np.array(["car"]),
        "coord": np.array(
            [
                [0.0, 0.4, 0.0],
                [0.0, -0.4, 0.0],
                [0.0, 0.6, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    output = ObjectMinPointsFilter(min_num_points=2, time_lag_dim=None)(sample)

    assert output["gt_boxes"].shape == (1, 7)


def test_object_range_min_points_filter_uses_distance_specific_threshold() -> None:
    sample = {
        "gt_boxes": np.array(
            [
                [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
                [70.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "gt_names": np.array(["car", "car"]),
        "gt_labels": np.array([0, 0], dtype=np.int64),
        "coord": np.array(
            [
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
                [10.2, 0.0, 0.0],
                [10.3, 0.0, 0.0],
                [70.0, 0.0, 0.0],
                [70.1, 0.0, 0.0],
                [70.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    near_filtered = ObjectRangeMinPointsFilter(
        range_radius=[0.0, 60.0],
        min_num_points=5,
        time_lag_dim=None,
    )(sample)
    output = ObjectRangeMinPointsFilter(
        range_radius=[60.0, 130.0],
        min_num_points=3,
        time_lag_dim=None,
    )(near_filtered)

    assert output["gt_boxes"][:, 0].tolist() == [70.0]


# A ground-truth box is annotated on the current frame, so sweep points must never decide whether it
# survives: otherwise the evaluated GT set would depend on how many sweeps the model consumes.
def _densified_sample() -> dict:
    """One box holding 2 current-frame points and 2 sweep points."""
    return {
        "gt_boxes": np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0]], dtype=np.float32),
        "gt_names": np.array(["car"]),
        "gt_labels": np.array([0], dtype=np.int64),
        "points": np.array(
            [
                [0.1, 0.0, 0.0, 1.0, 0.0],
                [0.2, 0.0, 0.0, 1.0, 0.0],
                [0.3, 0.0, 0.0, 1.0, 0.1],
                [0.4, 0.0, 0.0, 1.0, 0.1],
            ],
            dtype=np.float32,
        ),
    }


def test_min_points_filters_ignore_sweep_points_packed_layout() -> None:
    sample = _densified_sample()

    kept = ObjectMinPointsFilter(min_num_points=2, time_lag_dim=4)(dict(sample))
    dropped = ObjectMinPointsFilter(min_num_points=3, time_lag_dim=4)(dict(sample))

    assert kept["gt_names"].tolist() == ["car"]
    assert dropped["gt_names"].tolist() == []


def test_min_points_filters_ignore_sweep_points_split_layout() -> None:
    packed = _densified_sample()
    sample = {
        "gt_boxes": packed["gt_boxes"],
        "gt_names": packed["gt_names"],
        "coord": packed["points"][:, :3],
        "time_lag": packed["points"][:, 4:5],
    }

    dropped = ObjectRangeMinPointsFilter(
        range_radius=[0.0, 60.0], min_num_points=3, time_lag_dim=4
    )(dict(sample))

    assert dropped["gt_names"].tolist() == []


def test_min_points_filters_count_every_point_when_no_time_lag_is_declared() -> None:
    sample = _densified_sample()

    output = ObjectMinPointsFilter(min_num_points=3, time_lag_dim=None)(sample)

    # 4 points inside the box once sweep points count too, so the box survives.
    assert output["gt_names"].tolist() == ["car"]


def test_min_points_filters_reject_a_declared_absence_contradicted_by_the_sample() -> None:
    packed = _densified_sample()
    sample = {
        "gt_boxes": packed["gt_boxes"],
        "gt_names": packed["gt_names"],
        "coord": packed["points"][:, :3],
        "time_lag": packed["points"][:, 4:5],
    }

    with pytest.raises(ValueError, match="time_lag_dim is None"):
        ObjectMinPointsFilter(min_num_points=3, time_lag_dim=None)(sample)


def test_min_points_filters_reject_an_out_of_range_time_lag_column() -> None:
    sample = _densified_sample()
    sample["points"] = sample["points"][:, :4]

    with pytest.raises(ValueError, match="outside the 4 point features"):
        ObjectMinPointsFilter(min_num_points=3, time_lag_dim=4)(sample)


def test_load_annotations3d_builds_detection_targets() -> None:
    sample = {
        "class_names": ["car", "pedestrian"],
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1],
                "velocity": [0.5, -0.1],
                "gt_nusc_name": "car",
                "num_lidar_pts": 12,
                "bbox_3d_isvalid": True,
            },
            {
                "bbox_3d": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                "velocity": [0.0, 0.0],
                "gt_nusc_name": "ignore-me",
                "num_lidar_pts": 3,
                "bbox_3d_isvalid": True,
            },
        ],
    }

    output = LoadAnnotations3D()(sample)

    assert output["gt_boxes"].shape == (1, 9)
    assert output["gt_labels"].tolist() == [0]
    assert output["gt_names"].tolist() == ["car"]
    assert output["gt_num_points"].tolist() == [12]


def test_load_annotations3d_maps_bbox_label_with_label_to_category() -> None:
    sample = {
        "class_names": ["car"],
        "label_to_category": {3: "vehicle.car"},
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1],
                "bbox_label_3d": 3,
                "num_lidar_pts": 12,
                "bbox_3d_isvalid": True,
            }
        ],
    }

    output = LoadAnnotations3D(name_mapping={"vehicle.car": "car"})(sample)

    assert output["gt_labels"].tolist() == [0]
    assert output["gt_names"].tolist() == ["car"]


def test_load_annotations3d_replaces_nonfinite_velocity_with_zero() -> None:
    sample = {
        "class_names": ["car"],
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1],
                "velocity": [np.nan, np.inf],
                "gt_nusc_name": "car",
                "num_lidar_pts": 12,
                "bbox_3d_isvalid": True,
            }
        ],
    }

    output = LoadAnnotations3D()(sample)

    assert output["gt_boxes"].shape == (1, 9)
    assert np.allclose(output["gt_boxes"][0, 7:], np.array([0.0, 0.0], dtype=np.float32))


def test_load_annotations3d_drops_physically_invalid_instances() -> None:
    def make_instance(bbox_3d: list[float], velocity: list[float]) -> dict:
        return {
            "bbox_3d": bbox_3d,
            "velocity": velocity,
            "gt_nusc_name": "car",
            "num_lidar_pts": 12,
            "bbox_3d_isvalid": True,
        }

    sample = {
        "class_names": ["car"],
        "sample_token": "corrupt-frame",
        "instances": [
            make_instance([1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1], [0.5, -0.1]),  # sane
            make_instance([1e39, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1], [0.0, 0.0]),  # f32 overflow
            make_instance([3.0, 1.0, 0.5, 0.5, -0.8, 1.7, 0.0], [0.0, 0.0]),  # negative dim
            make_instance([1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1], [1e6, 0.0]),  # absurd velocity
            make_instance(
                [1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1], [120.0, 120.0]
            ),  # speed norm > bound
            make_instance([4.0, 4.0, 0.5, 4.5, 1.9, 1.4, 0.3], [140.0, 30.0]),  # fast but physical
            make_instance([5.0, 5.0, 0.5, 2.0, 1.0, 1.5, 0.2], [np.nan, np.nan]),  # nan vel: kept
        ],
    }

    output = LoadAnnotations3D()(sample)

    assert output["gt_boxes"].shape == (3, 9)
    assert np.allclose(output["gt_boxes"][0, :3], [1.0, 2.0, 3.0])
    assert np.allclose(output["gt_boxes"][1, 7:], [140.0, 30.0])  # component > norm/sqrt(2), kept
    assert np.allclose(output["gt_boxes"][2, 7:], [0.0, 0.0])  # nan velocity zeroed, box kept


def test_load_annotations3d_prefers_raw_name_over_stale_bbox_label() -> None:
    # A stale bbox_label_3d (here -1, "unclassed" under the file's older taxonomy)
    # must not override the raw gt_nusc_name: the config's name_mapping decides the
    # class, so the box is kept and classified from its raw name.
    sample = {
        "class_names": ["bicycle"],
        "label_to_category": {5: "bicycle"},
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 2.0, 1.0, 1.5, 0.0],
                "bbox_label_3d": -1,
                "gt_nusc_name": "bicycle",
                "num_lidar_pts": 10,
            }
        ],
    }

    output = LoadAnnotations3D(name_mapping={"bicycle": "bicycle"})(sample)

    assert output["gt_boxes"].shape == (1, 9)
    assert output["gt_names"].tolist() == ["bicycle"]


def test_load_annotations3d_ignores_validity_flag() -> None:
    # bbox_3d_isvalid is not a load-time filter: low-point filtering is the
    # point-count filters' job (min_num_points), so an "invalid" (0-lidar-point)
    # box is loaded and left for the point filter to drop.
    sample = {
        "class_names": ["car"],
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 2.0, 1.0, 1.5, 0.0],
                "gt_nusc_name": "car",
                "num_lidar_pts": 0,
                "bbox_3d_isvalid": False,
            }
        ],
    }

    output = LoadAnnotations3D()(sample.copy())

    assert output["gt_boxes"].shape == (1, 9)
    assert output["gt_num_points"].tolist() == [0]


def test_load_annotations3d_filters_raw_class_attributes() -> None:
    sample = {
        "class_names": ["bicycle"],
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 2.0, 1.0, 1.5, 0.0],
                "gt_nusc_name": "motorcycle",
                "gt_attrs": ["vehicle_state.parked"],
                "num_lidar_pts": 10,
            },
            {
                "bbox_3d": [2.0, 2.0, 3.0, 2.0, 1.0, 1.5, 0.0],
                "gt_nusc_name": "motorcycle",
                "gt_attrs": ["two_wheel_vehicle_state.without_rider"],
                "num_lidar_pts": 10,
            },
        ],
    }

    output = LoadAnnotations3D(
        name_mapping={"motorcycle": "bicycle"},
        filter_attributes=[["motorcycle", "vehicle_state.parked"]],
    )(sample)

    assert output["gt_boxes"][:, 0].tolist() == [2.0]
    assert output["gt_names"].tolist() == ["bicycle"]


def test_normalize_filter_attributes_rejects_invalid_entries() -> None:
    with pytest.raises(ValueError, match="filter_attributes entries"):
        normalize_filter_attributes([["bicycle"]])

    with pytest.raises(TypeError, match="filter_attributes entries"):
        normalize_filter_attributes(["bicycle"])


def test_load_annotations3d_prefers_raw_name_over_source_label() -> None:
    # When gt_nusc_name and a pre-baked bbox_label_3d disagree (e.g. the file's
    # class table predates the configured taxonomy), the raw name is authoritative
    # and the pre-baked integer is ignored, no error is raised.
    sample = {
        "class_names": ["car", "pedestrian"],
        "label_to_category": {0: "car"},
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 2.0, 1.0, 1.5, 0.0],
                "bbox_label_3d": 0,
                "gt_nusc_name": "pedestrian",
                "num_lidar_pts": 10,
            }
        ],
    }

    output = LoadAnnotations3D(
        name_mapping={"car": "car", "pedestrian": "pedestrian"},
    )(sample)

    assert output["gt_names"].tolist() == ["pedestrian"]
    assert output["gt_labels"].tolist() == [1]
