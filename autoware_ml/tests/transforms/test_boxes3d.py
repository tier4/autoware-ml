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
    min_points = ObjectMinPointsFilter(min_num_points=3)(sample.copy())

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
    min_points = ObjectMinPointsFilter(min_num_points=1)(sample.copy())

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

    output = ObjectMinPointsFilter(min_num_points=2)(sample)

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
    )(sample)
    output = ObjectRangeMinPointsFilter(
        range_radius=[60.0, 130.0],
        min_num_points=3,
    )(near_filtered)

    assert output["gt_boxes"][:, 0].tolist() == [70.0]


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


def test_load_annotations3d_filters_by_min_lidar_points() -> None:
    sample = {
        "class_names": ["car"],
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1],
                "gt_nusc_name": "car",
                "num_lidar_pts": 4,
                "bbox_3d_isvalid": True,
            },
            {
                "bbox_3d": [2.0, 3.0, 4.0, 4.0, 1.5, 1.7, 0.1],
                "gt_nusc_name": "car",
                "num_lidar_pts": 5,
                "bbox_3d_isvalid": True,
            },
        ],
    }

    output = LoadAnnotations3D(min_num_lidar_points=5)(sample)

    assert output["gt_boxes"].shape == (1, 9)
    assert output["gt_num_points"].tolist() == [5]


def test_load_annotations3d_rejects_negative_min_lidar_points() -> None:
    with pytest.raises(ValueError, match="min_num_lidar_points"):
        LoadAnnotations3D(min_num_lidar_points=-1)


def test_load_annotations3d_preserves_ignored_bbox_label() -> None:
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

    assert output["gt_boxes"].shape == (0, 9)


def test_load_annotations3d_can_match_awml_validity_policy() -> None:
    sample = {
        "class_names": ["car"],
        "instances": [
            {
                "bbox_3d": [1.0, 2.0, 3.0, 2.0, 1.0, 1.5, 0.0],
                "gt_nusc_name": "car",
                "num_lidar_pts": 10,
                "bbox_3d_isvalid": False,
            }
        ],
    }

    default_output = LoadAnnotations3D()(sample.copy())
    awml_output = LoadAnnotations3D(use_valid_flag=False)(sample.copy())

    assert default_output["gt_boxes"].shape == (0, 9)
    assert awml_output["gt_boxes"].shape == (1, 9)


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


def test_load_annotations3d_rejects_disagreeing_source_label() -> None:
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

    with pytest.raises(ValueError, match="Annotation label disagreement"):
        LoadAnnotations3D(
            name_mapping={"car": "car", "pedestrian": "pedestrian"},
        )(sample)
