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

"""Tests for inverting the ego-motion correction."""

import json

import numpy as np
import pytest

from autoware_ml.transforms.point_cloud.ego_motion import (
    PRE_CORRECTION_POINTS_KEY,
    InvertEgoMotionCorrection,
    PoseTable,
    invert_ego_motion,
    normalize_quaternions,
)


class TestEgoMotion:
    def test_invert_ego_motion_correction_round_trips_a_stationary_vehicle(self):
        # With the ego pose constant across the sweep, undoing the correction is the identity.
        poses = [
            {
                "timestamp": int((10.0 + i * 0.01) * 1e6),
                "translation": [5.0, -2.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }
            for i in range(20)
        ]
        table = PoseTable(
            times=np.array([p["timestamp"] * 1e-6 for p in poses], dtype=np.float64),
            translations=np.array([p["translation"] for p in poses], dtype=np.float64),
            quaternions=normalize_quaternions(
                np.array([p["rotation"] for p in poses], dtype=np.float64)
            ),
        )
        points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]], dtype=np.float32)
        times = np.array([10.05, 10.09], dtype=np.float64)

        out = invert_ego_motion(points, 10.02, times, table)

        np.testing.assert_allclose(out, points, atol=1e-6)

    def test_invert_ego_motion_correction_attaches_aligned_points(self, tmp_path):
        scene = tmp_path / "scene"
        (scene / "annotation").mkdir(parents=True)
        (scene / "data").mkdir()
        # Vehicle drives +x at 10 m/s; the sweep reference is t=10.0.
        (scene / "annotation" / "ego_pose.json").write_text(
            json.dumps(
                [
                    {
                        "timestamp": int((10.0 + i * 0.01) * 1e6),
                        "translation": [10.0 * (i * 0.01), 0.0, 0.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                    }
                    for i in range(20)
                ]
            )
        )
        # Columns [x, y, z, intensity, channel, return_type, timestamp_ns].
        points = np.zeros((2, 7), dtype=np.float32)
        points[:, 0] = [20.0, 20.0]
        points[:, 6] = [0.0, 50_000_000.0]  # 0 ms and 50 ms into the sweep
        sample = {
            "points": points,
            "lidar_path": str(scene / "data" / "00000.pcd.bin"),
            "lidar_sources_info": {
                "stamp": {"sec": 10, "nanosec": 0},
                "sources": [{"idx_begin": 0, "length": 2, "stamp": {"sec": 10, "nanosec": 0}}],
            },
        }

        output = InvertEgoMotionCorrection(timestamp_dim=6)(sample)

        pre = output[PRE_CORRECTION_POINTS_KEY]
        assert pre.shape == (2, 3)
        # The first point was captured at the reference time, so it does not move. The second was
        # captured 50 ms later, by which time the vehicle had advanced 0.5 m, so relative to the
        # ego frame at that instant the point sits 0.5 m nearer.
        np.testing.assert_allclose(pre[0], [20.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(pre[1], [19.5, 0.0, 0.0], atol=1e-4)

    def test_invert_ego_motion_correction_requires_metadata_when_strict(self):
        sample = {"points": np.zeros((2, 7), dtype=np.float32)}

        with pytest.raises(KeyError):
            InvertEgoMotionCorrection()(sample)

        passthrough = InvertEgoMotionCorrection(strict=False)(dict(sample))
        assert PRE_CORRECTION_POINTS_KEY not in passthrough
