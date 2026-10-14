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

"""Tests for the boxes the T4 detection task reads out of a record."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytest

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.common.detection3d import Detection3DTask
from autoware_ml.types.geometry import Box3DFieldIndex

_ROOT = "/data/t4dataset"


def box(
    instance: str,
    label_name: str,
    valid: bool = True,
    attributes: list[str] | None = None,
    fields: dict[Box3DFieldIndex, float] | None = None,
) -> dict:
    """
    Record table entry of one box.

    Args:
      instance: Instance id of the box.
      label_name: Trained label name of the box.
      valid: Validity flag of the box.
      attributes: Attribute names of the box.
      fields: Box parameters overriding the unit box at the origin.

    Returns:
      dict: Box as the record table stores it.
    """
    params = np.zeros(len(Box3DFieldIndex), dtype=np.float64)
    params[Box3DFieldIndex.LENGTH : Box3DFieldIndex.YAW] = 1.0
    for index, value in (fields or {}).items():
        params[index] = value
    return {
        "box3d_params": params.tolist(),
        "box3d_instance_id": instance,
        "box3d_dataset_label_name": label_name,
        "box3d_label_name": label_name,
        "box3d_label_index": 0,
        "box3d_num_lidar_points": 8,
        "box3d_num_radar_points": 0,
        "box3d_valid": valid,
        "box3d_attributes": attributes or [],
        "box3d_coordinate": "gravity_center",
    }


def build_task(boxes: list[dict]) -> Detection3DTask:
    """
    Detection task over a single record carrying the given boxes.

    Args:
      boxes: Boxes of the record, as the record table stores them.

    Returns:
      Detection3DTask: Task serving that one record.
    """
    schema = DatasetTableSchema.to_polars_schema()
    records = pl.DataFrame(
        {DatasetTableSchema.BOXES_3D.name: [boxes]},
        schema={DatasetTableSchema.BOXES_3D.name: schema[DatasetTableSchema.BOXES_3D.name]},
    )
    return Detection3DTask(database_root_path=_ROOT, dataset_records_dataframe=records)


def test_every_box_keeps_its_attributes_and_an_unseen_box_stays() -> None:
    # A box with no lidar points is still a valid target. The training filters and the
    # metrics apply the point count, so the dataset serves it
    task = build_task(
        [
            box("a", "bicycle", True, ["vehicle_state.parked"]),
            box("b", "car", False, ["vehicle_state.moving"]),
            box("c", "pedestrian", True, []),
        ]
    )

    boxes = task.get_data_sample(0).detection3d_gt_bboxes_3d

    assert boxes.bbox_label_names == ["bicycle", "car", "pedestrian"]
    assert boxes.bbox_attributes == [["vehicle_state.parked"], ["vehicle_state.moving"], []]


def test_a_record_without_boxes_carries_no_attributes() -> None:
    boxes = build_task([]).get_data_sample(0).detection3d_gt_bboxes_3d

    assert len(boxes) == 0
    assert boxes.bbox_attributes == []


def test_a_box_without_a_velocity_stands_still() -> None:
    task = build_task(
        [
            box(
                "a",
                "car",
                fields={Box3DFieldIndex.VELOCITY_X: np.nan, Box3DFieldIndex.VELOCITY_Y: np.nan},
            ),
            box(
                "b",
                "car",
                fields={Box3DFieldIndex.VELOCITY_X: 3.0, Box3DFieldIndex.VELOCITY_Y: -1.0},
            ),
        ]
    )

    params = task.get_data_sample(0).detection3d_gt_bboxes_3d.bbox_params

    assert params[:, Box3DFieldIndex.VELOCITY_X : Box3DFieldIndex.VELOCITY_Z].tolist() == [
        [0.0, 0.0],
        [3.0, -1.0],
    ]


def test_a_box_without_an_extent_or_a_finite_geometry_is_no_target() -> None:
    task = build_task(
        [
            box("a", "car", fields={Box3DFieldIndex.WIDTH: 0.0}),
            box("b", "car", fields={Box3DFieldIndex.X: np.inf}),
            box("c", "car", fields={Box3DFieldIndex.YAW: np.nan}),
            box("d", "car", fields={Box3DFieldIndex.X: 5.0}),
        ]
    )

    boxes = task.get_data_sample(0).detection3d_gt_bboxes_3d

    assert len(boxes) == 1
    assert boxes.bbox_params[0, Box3DFieldIndex.X] == 5.0


def build_task_with_status(boxes: list[dict], status: bool | None) -> Detection3DTask:
    """
    Detection task over a single record carrying the cone and barrier annotation flag.

    Args:
      boxes: Boxes of the record, as the record table stores them.
      status: Whether cones and barriers are annotated in the record, None when unknown.

    Returns:
      Detection3DTask: Task serving that one record.
    """
    schema = DatasetTableSchema.to_polars_schema()
    columns = [
        DatasetTableSchema.BOXES_3D.name,
        DatasetTableSchema.TRAFFIC_CONE_BARRIER_BBOX_STATUS.name,
    ]
    records = pl.DataFrame(
        {columns[0]: [boxes], columns[1]: [status]},
        schema={column: schema[column] for column in columns},
    )
    return Detection3DTask(database_root_path=_ROOT, dataset_records_dataframe=records)


def test_the_cone_and_barrier_flag_is_none_when_the_records_do_not_carry_it() -> None:
    task = build_task([box("a", "car")])

    assert task.dataset_records_dataframe.columns == [DatasetTableSchema.BOXES_3D.name]
    assert task.get_data_sample(0).detection3d_traffic_cone_barrier_bbox_status is None


@pytest.mark.parametrize("status", [True, False])
def test_the_cone_and_barrier_flag_is_read_from_the_record(status: bool) -> None:
    task = build_task_with_status([box("a", "car")], status)

    assert task.get_data_sample(0).detection3d_traffic_cone_barrier_bbox_status is status


def test_a_record_without_the_cone_and_barrier_flag_serves_none() -> None:
    task = build_task_with_status([box("a", "car")], None)

    assert task.get_data_sample(0).detection3d_traffic_cone_barrier_bbox_status is None


def test_str_names_the_task() -> None:
    assert str(build_task([])) == "Detection3DTask"


def test_log_dataset_info_reports_total_and_lidar_point_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = build_task(
        [
            box("a", "car"),
            {**box("b", "car"), "box3d_num_lidar_points": 0},
            box("c", "pedestrian"),
        ]
    )

    with caplog.at_level(logging.INFO, logger="autoware_ml.datamodule.common.detection3d"):
        task.log_dataset_info()

    assert "Number of bboxes per class in the dataset: " in caplog.text
    assert "Number of bboxes after filtering num_lidar_points > 0 per class: " in caplog.text
    assert "'car': 2" in caplog.text
    assert "'car': 1" in caplog.text
    assert "'pedestrian': 1" in caplog.text
