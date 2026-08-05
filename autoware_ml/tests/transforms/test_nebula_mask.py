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

"""Tests for the Nebula downsample mask filter."""

import numpy as np
import pytest

from autoware_ml.transforms.point_cloud.nebula_mask import NebulaDownsampleMaskFilter


class TestNebulaDownsampleMaskFilter:
    def test_nebula_downsample_mask_filter_keeps_aligned_point_arrays(self, tmp_path):
        mask_root = tmp_path / "masks"
        calibration_root = tmp_path / "calibrations"
        (mask_root / "front_upper").mkdir(parents=True)
        calibration_root.mkdir()
        mask = np.zeros((2, 10), dtype=np.uint8)
        mask[0, :] = 255
        cv2 = pytest.importorskip("cv2")
        cv2.imwrite(str(mask_root / "front_upper" / "mask.png"), mask)
        (calibration_root / "model.csv").write_text(
            "Channel,Elevation,Azimuth\n1,0.0,0.0\n2,45.0,0.0\n"
        )

        sample = {
            "points": np.array(
                [
                    [10.0, 0.0, 0.0, 1.0],
                    [10.0, 0.0, 10.0, 2.0],
                ],
                dtype=np.float32,
            ),
            "pts_semantic_mask": np.array([7, 9], dtype=np.int64),
            "source_name": "LIDAR_FRONT_UPPER",
        }

        output = NebulaDownsampleMaskFilter(
            mask_root=str(mask_root),
            calibration_root=str(calibration_root),
            lidar_name_to_mask={"front_upper": "front_upper/mask.png"},
            lidar_name_to_model={"front_upper": "test_model"},
            lidar_name_to_calibration={"front_upper": "model.csv"},
            use_calibration_azimuth_offsets=False,
            return_stats=True,
        )(sample)

        assert output["points"].shape == (1, 4)
        np.testing.assert_array_equal(output["pts_semantic_mask"], np.array([7], dtype=np.int64))
        assert output["nebula_downsample_stats"] == [
            {"source_name": "LIDAR_FRONT_UPPER", "num_input_points": 2, "num_kept_points": 1}
        ]

    def test_nebula_downsample_mask_filter_uses_concat_source_slices(self, tmp_path):
        mask_root = tmp_path / "masks"
        calibration_root = tmp_path / "calibrations"
        for name, value in {"front_upper": 255, "rear_upper": 0}.items():
            (mask_root / name).mkdir(parents=True)
            cv2 = pytest.importorskip("cv2")
            cv2.imwrite(str(mask_root / name / "mask.png"), np.full((1, 10), value, np.uint8))
        calibration_root.mkdir()
        (calibration_root / "model.csv").write_text("Channel,Elevation,Azimuth\n1,0.0,0.0\n")

        sample = {
            "points": np.array([[10.0, 0.0, 0.0, 1.0], [20.0, 0.0, 0.0, 2.0]], dtype=np.float32),
            "labels": np.array([1, 2], dtype=np.int64),
            "lidar_sources": {
                "LIDAR_FRONT_UPPER": {
                    "sensor_token": "front",
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": np.eye(3).tolist(),
                },
                "LIDAR_REAR_UPPER": {
                    "sensor_token": "rear",
                    "translation": [0.0, 0.0, 0.0],
                    "rotation": np.eye(3).tolist(),
                },
            },
            "lidar_sources_info": {
                "sources": [
                    {"sensor_token": "front", "idx_begin": 0, "length": 1},
                    {"sensor_token": "rear", "idx_begin": 1, "length": 1},
                ]
            },
        }

        output = NebulaDownsampleMaskFilter(
            mask_root=str(mask_root),
            calibration_root=str(calibration_root),
            lidar_name_to_mask={
                "front_upper": "front_upper/mask.png",
                "rear_upper": "rear_upper/mask.png",
            },
            lidar_name_to_model={"front_upper": "test_model", "rear_upper": "test_model"},
            lidar_name_to_calibration={"front_upper": "model.csv", "rear_upper": "model.csv"},
            use_calibration_azimuth_offsets=False,
        )(sample)

        np.testing.assert_allclose(output["points"], np.array([[10.0, 0.0, 0.0, 1.0]]))
        np.testing.assert_array_equal(output["labels"], np.array([1], dtype=np.int64))

    def test_nebula_downsample_mask_filter_uses_nebula_azimuth_convention(self, tmp_path):
        mask_root = tmp_path / "masks"
        calibration_root = tmp_path / "calibrations"
        (mask_root / "front_upper").mkdir(parents=True)
        calibration_root.mkdir()
        cv2 = pytest.importorskip("cv2")
        mask = np.zeros((1, 4), dtype=np.uint8)
        mask[0, 0] = 255
        cv2.imwrite(str(mask_root / "front_upper" / "mask.png"), mask)
        (calibration_root / "model.csv").write_text("Channel,Elevation,Azimuth\n1,0.0,0.0\n")

        sample = {
            "points": np.array(
                [
                    [0.0, 10.0, 0.0, 1.0],
                    [10.0, 0.0, 0.0, 2.0],
                ],
                dtype=np.float32,
            ),
            "source_name": "LIDAR_FRONT_UPPER",
        }

        output = NebulaDownsampleMaskFilter(
            mask_root=str(mask_root),
            calibration_root=str(calibration_root),
            lidar_name_to_mask={"front_upper": "front_upper/mask.png"},
            lidar_name_to_model={"front_upper": "test_model"},
            lidar_name_to_calibration={"front_upper": "model.csv"},
            use_calibration_azimuth_offsets=False,
        )(sample)

        np.testing.assert_allclose(output["points"], np.array([[0.0, 10.0, 0.0, 1.0]]))

    def test_nebula_downsample_mask_filter_prefers_the_stored_ring_index(self, tmp_path):
        mask_root = tmp_path / "masks"
        calibration_root = tmp_path / "calibrations"
        (mask_root / "front_upper").mkdir(parents=True)
        calibration_root.mkdir()
        cv2 = pytest.importorskip("cv2")
        # Only ring 1 is kept. Both points sit at elevation 0, so estimating the ring from
        # elevation would put both on ring 0 and drop them; the stored index must win.
        mask = np.zeros((2, 4), dtype=np.uint8)
        mask[1, 0] = 255
        cv2.imwrite(str(mask_root / "front_upper" / "mask.png"), mask)
        (calibration_root / "model.csv").write_text(
            "Channel,Elevation,Azimuth\n1,0.0,0.0\n2,45.0,0.0\n"
        )

        # Columns are [x, y, z, intensity, channel].
        sample = {
            "points": np.array(
                [[0.0, 10.0, 0.0, 1.0, 1.0], [0.0, 10.0, 0.0, 2.0, 0.0]],
                dtype=np.float32,
            ),
            "source_name": "LIDAR_FRONT_UPPER",
        }

        kwargs = {
            "mask_root": str(mask_root),
            "calibration_root": str(calibration_root),
            "lidar_name_to_mask": {"front_upper": "front_upper/mask.png"},
            "lidar_name_to_model": {"front_upper": "test_model"},
            "lidar_name_to_calibration": {"front_upper": "model.csv"},
        }
        output = NebulaDownsampleMaskFilter(channel_dim=4, **kwargs)(dict(sample))
        np.testing.assert_allclose(output["points"], sample["points"][:1])

        # Opting out falls back to the elevation estimate, which drops both points.
        fallback = NebulaDownsampleMaskFilter(channel_dim=None, **kwargs)(dict(sample))
        assert fallback["points"].shape[0] == 0

    def test_nebula_downsample_mask_filter_does_not_apply_azimuth_offsets_by_default(self):
        # The driver indexes the mask by raw azimuth; subtracting the per-channel calibration
        # offsets shifts points into the wrong column.
        assert (
            NebulaDownsampleMaskFilter(mask_root="aip_x2_gen2").use_calibration_azimuth_offsets
            is False
        )
