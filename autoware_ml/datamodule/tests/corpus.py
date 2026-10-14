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

"""Synthetic T4 corpus the pipeline tests read through the real dataset and transforms."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import MappingProxyType

import numpy as np
import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.taxonomy import LabelVocabulary, SegmentationTaxonomy
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.common.detection3d import Detection3DTask
from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DTask
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.boxes3d.filters import BBoxesLabelNameFilter, BBoxesRangeFilter
from autoware_ml.transforms.point_cloud.geometry import (
    GlobalBEVRandomFlip,
    GlobalRotScaleTrans,
    PointsRangeFilter,
    RandomRotateTargetAngle,
)
from autoware_ml.transforms.point_cloud.loading import (
    LoadMultiSweepPointsFromFile,
    LoadPointsFromFile,
    SweepWindow,
)
from autoware_ml.transforms.point_cloud.perturbation import RandomJitter, RandomStrengthJitter
from autoware_ml.transforms.point_cloud.sampling import RandomDropout
from autoware_ml.types.geometry import Box3DFieldIndex

NUM_POINTS = 512
# The record of a sample stores its own frame, then its past sweeps, then its future ones
NUM_PAST_SWEEPS = 1
NUM_FUTURE_SWEEPS = 1
NUM_SWEEPS = 1 + NUM_PAST_SWEEPS + NUM_FUTURE_SWEEPS
POINT_CLOUD_RANGE = (-8.0, -8.0, -2.0, 8.0, 8.0, 2.0)
SCENE = "db_v1/scene_0/dataset_v1"

VOCABULARY = LabelVocabulary({"car": "car", "vehicle.car": "car", "unpainted": None})
SEGMENTATION_TAXONOMY = SegmentationTaxonomy(
    VOCABULARY, ("car",), {"car": "car"}, -1, {"vehicle": ("car",)}
)


def sweep_time_offset(sweep_index: int) -> float:
    """Seconds subtracted from the sample timestamp to date one stored frame.

    The sample owns position 0, the past sweeps follow it and then the future ones, so the
    loader reads the side of a frame off the sign of its time lag.

    Args:
        sweep_index: Position of the frame in the record.

    Returns:
        float: Offset of the frame, positive for a past sweep and negative for a future one.
    """
    if sweep_index == 0:
        return 0.0
    if sweep_index <= NUM_PAST_SWEEPS:
        return 0.05 * sweep_index
    return -0.05 * (sweep_index - NUM_PAST_SWEEPS)


def write_lidar_frame(
    root: Path, record_index: int, frame_index: int, seed: int, masked: bool = True
) -> dict:
    """Write the point cloud and the semantic mask of one lidar frame and return its record.

    Args:
        root: Database root the relative paths resolve against.
        record_index: Position of the record the frame belongs to.
        frame_index: Position of the frame in the record, zero for the frame of the sample,
            then its past sweeps and then its future ones.
        seed: Seed of the points so every frame differs.
        masked: Whether the frame carries a semantic mask.

    Returns:
        dict: The lidar frame as the record table stores it.
    """
    relative = f"{SCENE}/data/LIDAR_TOP/{record_index}_{frame_index}.pcd.bin"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.random.default_rng(seed).uniform(
        low=[-6.0, -6.0, -1.0, 0.0, 0.0],
        high=[6.0, 6.0, 1.0, 1.0, 0.0],
        size=(NUM_POINTS, 5),
    )
    points.astype(np.float32).tofile(path)

    mask_path = None
    if masked:
        mask_path = root / f"{SCENE}/data/LIDARSEG/{record_index}_{frame_index}.bin"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.zeros(NUM_POINTS, dtype=np.uint8).tofile(mask_path)

    return {
        "lidar_frame_id": f"frame-{record_index}-{frame_index}",
        "lidar_keyframe": frame_index == 0,
        "lidar_sensor_id": "lidar",
        "lidar_sensor_channel_name": "LIDAR_TOP",
        "lidar_timestamp_seconds": float(record_index) - sweep_time_offset(frame_index),
        "lidar_pointcloud_path": str(path),
        "lidar_pointcloud_source_path": None,
        "lidar_pointcloud_num_features": 5,
        "lidar_sensor_to_ego_pose_matrix": np.eye(4).tolist(),
        "lidar_frame_ego_pose_to_global_matrix": np.eye(4).tolist(),
        "lidar_sensor_to_lidar_sweep_matrix": np.eye(4).tolist(),
        "lidar_pointcloud_semantic_mask_path": None if mask_path is None else str(mask_path),
    }


def write_corpus(
    root: Path, num_records: int, unmasked_records: frozenset[int] = frozenset()
) -> pl.DataFrame:
    """Write a small synthetic corpus and return its record table.

    Args:
        root: Database root the relative paths resolve against.
        num_records: Number of frames to write.
        unmasked_records: Positions of the records written without a semantic mask.

    Returns:
        pl.DataFrame: The record table of the corpus.
    """
    rows = []
    for record_index in range(num_records):
        lidar_frames = [
            write_lidar_frame(
                root,
                record_index,
                frame_index,
                record_index * NUM_SWEEPS + frame_index,
                masked=record_index not in unmasked_records,
            )
            for frame_index in range(NUM_SWEEPS)
        ]

        box = np.zeros(len(Box3DFieldIndex), dtype=np.float64)
        box[Box3DFieldIndex.X] = 1.0
        box[Box3DFieldIndex.Y] = 1.0
        box[Box3DFieldIndex.Z] = 0.0
        box[Box3DFieldIndex.LENGTH] = 4.0
        box[Box3DFieldIndex.WIDTH] = 2.0
        box[Box3DFieldIndex.HEIGHT] = 1.5
        rows.append(
            {
                DatasetTableSchema.SCENARIO_ID.name: "scenario-0",
                DatasetTableSchema.SAMPLE_ID.name: f"sample-{record_index}",
                DatasetTableSchema.SAMPLE_INDEX.name: record_index,
                DatasetTableSchema.TIMESTAMP_SECONDS.name: float(record_index),
                DatasetTableSchema.LOCATION.name: "odaiba",
                DatasetTableSchema.VEHICLE_TYPE.name: "j6gen2",
                DatasetTableSchema.SCENARIO_NAME.name: "scenario",
                DatasetTableSchema.LIDAR_FRAMES.name: lidar_frames,
                DatasetTableSchema.LIDAR_SOURCES.name: [],
                DatasetTableSchema.IMAGE_FRAMES.name: [],
                DatasetTableSchema.CATEGORY_MAPPING.name: {
                    "category_names": ["car"],
                    "category_indices": [0],
                },
                DatasetTableSchema.BOXES_3D.name: [
                    {
                        "box3d_params": box.tolist(),
                        "box3d_instance_id": f"box-{record_index}",
                        "box3d_dataset_label_name": "vehicle.car",
                        "box3d_label_name": "car",
                        "box3d_label_index": 0,
                        "box3d_num_lidar_points": 64,
                        "box3d_num_radar_points": 0,
                        "box3d_valid": True,
                        "box3d_attributes": [],
                        "box3d_coordinate": "gravity_center",
                    }
                ],
            }
        )
    return pl.DataFrame(rows, schema=DatasetTableSchema.to_polars_schema())


def build_transforms() -> TransformsCompose:
    """Build a lidar training pipeline that loads the sweeps on both sides of a sample."""
    return TransformsCompose(
        pipeline=[
            LoadPointsFromFile(use_dim=(0, 1, 2, 3)),
            LoadMultiSweepPointsFromFile(
                past=SweepWindow(
                    num=NUM_PAST_SWEEPS, time_lag_range=(0.01, 1.0), selection="random"
                ),
                future=SweepWindow(
                    num=NUM_FUTURE_SWEEPS, time_lag_range=(0.01, 1.0), selection="random"
                ),
                use_timestamp_difference=True,
                use_dim=(0, 1, 2, 3),
                bev_remove_radius=0.0,
            ),
            RandomRotateTargetAngle(probability=1.0, yaw_angle_ratios=[0.5, 1.0, 1.5]),
            GlobalRotScaleTrans(
                yaw_rot_range=(-0.2, 0.2),
                scale_ratio_range=(0.95, 1.05),
                translation_std=(0.1, 0.1, 0.05),
            ),
            GlobalBEVRandomFlip(),
            RandomJitter(sigma=0.01, clip=0.05),
            PointsRangeFilter(points_range=POINT_CLOUD_RANGE),
            RandomDropout(dropout_ratio=0.1, probability=1.0),
            RandomStrengthJitter(
                gamma_range=(0.8, 1.25), scale_range=(0.9, 1.1), shift_range=(-0.02, 0.02)
            ),
            BBoxesLabelNameFilter(label_names_to_keep=["car"], class_names=["car"]),
            BBoxesRangeFilter(point_cloud_range=POINT_CLOUD_RANGE),
        ]
    )


def build_dataset(root: Path, records: pl.DataFrame) -> T4Dataset:
    """Build the dataset the datamodule serves, over the synthetic record table."""
    return T4Dataset(
        lidar_intensity_scale=255.0,
        database_root_path=str(root),
        max_num_3d_gt_bboxes=8,
        dataset_records_dataframe=records,
        transforms=build_transforms(),
        dataset_tasks=MappingProxyType(
            {
                "detection3d": Detection3DTask,
                "segmentation3d": partial(T4Segmentation3DTask, taxonomy=SEGMENTATION_TAXONOMY),
            }
        ),
        det3d_supervised=True,
        seg3d_supervised=True,
    )
