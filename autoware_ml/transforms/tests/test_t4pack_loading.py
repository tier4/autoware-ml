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

"""Tests for the multi-task point cloud loaders that read t4pack files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    LiDARPointCloudSample,
    MultiTaskGTSample,
)
from autoware_ml.transforms.multi_task.point_cloud.loading import (
    LoadMultiSweepPointsFromFile,
    LoadMultiSweepPointsFromT4Pack,
    LoadPointsFromFile,
    LoadPointsFromT4Pack,
)
from autoware_ml.utils.point_cloud.t4pack import read_t4pack_index
from autoware_ml.utils.tests.t4pack_fixtures import random_lidar_frame, write_test_pack

NAMES = ("00000.pcd.bin", "00001.pcd.bin", "00002.pcd.bin")


@pytest.fixture
def scene(tmp_path: Path) -> tuple[list[LiDARPointCloudSample], list[LiDARPointCloudSample]]:
    """The same three frames as loose files and as a pack, with their records."""
    rng = np.random.default_rng(0)
    frames = {name: random_lidar_frame(rng, 50 + 10 * i) for i, name in enumerate(NAMES)}
    loose_dir = tmp_path / "loose" / "data" / "LIDAR_CONCAT"
    loose_dir.mkdir(parents=True)
    for name, frame in frames.items():
        frame.tofile(loose_dir / name)
    packed_dir = tmp_path / "packed" / "data"
    packed_dir.mkdir(parents=True)
    write_test_pack(packed_dir / "LIDAR_CONCAT.pack", frames)
    index = read_t4pack_index(str(packed_dir / "LIDAR_CONCAT.pack"))

    def record(path: Path, i: int, t4pack_frame=None) -> LiDARPointCloudSample:
        return LiDARPointCloudSample(
            point_cloud_path=str(path),
            timestamp=10.0 - 0.1 * i,
            sensor_to_ego_pose_matrix=torch.eye(4),
            lidar_to_ego_pose_to_global_matrix=torch.eye(4),
            lidar_sensor_to_lidar_sweep_matrix=torch.eye(4),
            t4pack_frame=t4pack_frame,
        )

    loose = [record(loose_dir / name, i) for i, name in enumerate(NAMES)]
    packed = [
        record(packed_dir / "LIDAR_CONCAT" / name, i, index[name]) for i, name in enumerate(NAMES)
    ]
    return loose, packed


def _sample(records: list[LiDARPointCloudSample]) -> MultiTaskGTSample:
    return MultiTaskGTSample(
        lidar_point_cloud_samples=records,
        point_cloud_data=None,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
    )


def test_t4pack_loader_matches_the_loose_file_loader(scene) -> None:
    """The current frame loads the same points from the pack as from its loose file."""
    loose, packed = scene

    expected = LoadPointsFromFile(bev_remove_radius=1.0)(_sample(loose)).point_cloud_data
    actual = LoadPointsFromT4Pack(bev_remove_radius=1.0)(_sample(packed)).point_cloud_data

    assert torch.equal(actual.points, expected.points)


def test_multi_sweep_t4pack_loader_matches_the_loose_file_loader(scene) -> None:
    """The sweeps load from the pack too, with the same time lag features."""
    loose, packed = scene

    def run(records, load, load_sweeps) -> torch.Tensor:
        sample = load()(_sample(records))
        return load_sweeps(sweeps_num=2, test_mode=True)(sample).point_cloud_data.points

    expected = run(loose, LoadPointsFromFile, LoadMultiSweepPointsFromFile)
    actual = run(packed, LoadPointsFromT4Pack, LoadMultiSweepPointsFromT4Pack)

    assert torch.equal(actual, expected)


def test_t4pack_loader_rejects_a_record_without_a_pack_location(scene) -> None:
    """Records of scenes without a pack name the loader to use instead."""
    loose, _ = scene

    with pytest.raises(ValueError, match="has no t4pack location"):
        LoadPointsFromT4Pack()(_sample(loose))
