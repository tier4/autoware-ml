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

"""Tests for the point cloud file formats of the multi-task point cloud loaders."""

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
    LoadPointsFromFile,
)
from autoware_ml.types.dataset import PCDFileFormat
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


def _load(records, pcd_file_format, sweeps_num: int = 0) -> torch.Tensor:
    sample = LoadPointsFromFile(bev_remove_radius=1.0, pcd_file_format=pcd_file_format)(
        _sample(records)
    )
    if sweeps_num:
        sample = LoadMultiSweepPointsFromFile(
            sweeps_num=sweeps_num, test_mode=True, pcd_file_format=pcd_file_format
        )(sample)
    return sample.point_cloud_data.points


@pytest.mark.parametrize("sweeps_num", [0, 2])
@pytest.mark.parametrize("pcd_file_format", [PCDFileFormat.T4PACK, PCDFileFormat.AUTO])
def test_pack_loads_the_same_points_as_the_bin_files(scene, pcd_file_format, sweeps_num) -> None:
    """The current frame and its sweeps load the same points from the pack as from ``.pcd.bin``."""
    loose, packed = scene

    expected = _load(loose, PCDFileFormat.BIN, sweeps_num)

    assert torch.equal(_load(packed, pcd_file_format, sweeps_num), expected)


def test_auto_reads_the_pack_when_the_record_has_a_location(scene) -> None:
    """With a pack location, auto does not touch the ``.pcd.bin`` path."""
    _, packed = scene
    # The packed records point at loose paths that do not exist
    assert not Path(packed[0].point_cloud_path).exists()

    assert _load(packed, "auto").shape[0] > 0


def test_auto_reads_the_bin_file_without_a_pack_location(scene) -> None:
    """Records of scenes without a pack load from their ``.pcd.bin`` files."""
    loose, _ = scene

    assert torch.equal(_load(loose, "auto"), _load(loose, "bin"))


def test_bin_ignores_the_pack_location(scene) -> None:
    """``bin`` reads the ``.pcd.bin`` path even when the record has a pack location."""
    _, packed = scene

    with pytest.raises(FileNotFoundError):
        _load(packed, "bin")


def test_t4pack_rejects_a_record_without_a_pack_location(scene) -> None:
    """Records of scenes without a pack name the formats to use instead."""
    loose, _ = scene

    with pytest.raises(ValueError, match="has no t4pack location"):
        _load(loose, "t4pack")


def test_an_unknown_format_is_rejected() -> None:
    """A typo in the config fails when the transform is built."""
    with pytest.raises(ValueError, match="'pack' is not a valid PCDFileFormat"):
        LoadPointsFromFile(pcd_file_format="pack")
