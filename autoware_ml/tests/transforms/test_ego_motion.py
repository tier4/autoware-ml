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
    interpolate_poses,
    invert_ego_motion,
    normalize_quaternions,
    quat_to_rotmat,
    relative_axis_angle,
    rotate_axis_angle,
    slerp,
)


def invert_ego_motion_per_point(
    corrected_ego: np.ndarray,
    reference_time: float,
    point_times: np.ndarray,
    pose_table: PoseTable,
) -> np.ndarray:
    """Interpolate a pose for every point, in global coordinates.

    The straightforward reading of the correction, kept here as the reference that
    :func:`invert_ego_motion` -- which groups points by pose interval to avoid the per-point
    quaternion maths -- is checked against.
    """
    ref_t, ref_q = interpolate_poses(np.array([reference_time], dtype=np.float64), pose_table)
    global_points = corrected_ego.astype(np.float64) @ quat_to_rotmat(ref_q)[0].T + ref_t[0]
    point_t, point_q = interpolate_poses(point_times, pose_table)
    return np.einsum("nj,njk->nk", global_points - point_t, quat_to_rotmat(point_q))


def yawing_drive_pose_table(
    *, rate_hz: float = 75.0, duration_s: float = 0.4, yaw_rate: float = 0.6, speed: float = 12.0
) -> PoseTable:
    """A vehicle driving a curve, sampled the way T4's ``ego_pose.json`` is."""
    times = 10.0 + np.arange(0.0, duration_s, 1.0 / rate_hz)
    yaw = yaw_rate * (times - times[0])
    # Map-scale absolute translations, so that any arithmetic done at that magnitude shows up.
    translations = np.stack(
        [
            81_234.5 + speed * np.cos(yaw) / max(yaw_rate, 1e-9),
            7_233_101.25 + speed * np.sin(yaw) / max(yaw_rate, 1e-9),
            np.full_like(times, 41.5),
        ],
        axis=1,
    )
    quaternions = np.stack(
        [np.cos(yaw / 2.0), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2.0)], axis=1
    )
    return PoseTable(times, translations, normalize_quaternions(quaternions))


class TestPoseMaths:
    def test_relative_axis_angle_recovers_a_known_rotation(self):
        angle = 0.37
        axis = np.array([1.0, -2.0, 0.5])
        axis /= np.linalg.norm(axis)
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])

        recovered_axis, recovered_angle = relative_axis_angle(q0, q1)

        assert recovered_angle == pytest.approx(angle)
        np.testing.assert_allclose(recovered_axis, axis, atol=1e-12)

    def test_relative_axis_angle_takes_the_short_way_round(self):
        # A quaternion and its negation are the same rotation; the relative angle must be 0.
        q0 = normalize_quaternions(np.array([[0.6, 0.8, 0.0, 0.0]]))[0]

        axis, angle = relative_axis_angle(q0, -q0)

        assert axis is None
        assert angle == 0.0

    def test_slerp_survives_identical_quaternions(self):
        # The nearly-parallel fallback is there to keep the division by sin(theta_0) finite; that is
        # the only thing it owes, and it has to hold at exact equality.
        quaternion = normalize_quaternions(np.array([[0.6, 0.8, 0.0, 0.0]]))

        interpolated = slerp(quaternion, quaternion.copy(), np.array([0.37]))

        np.testing.assert_allclose(interpolated, quaternion, atol=1e-15)

    def test_rotate_axis_angle_matches_the_rotation_matrix(self):
        axis = np.array([0.0, 0.0, 1.0])
        angles = np.array([0.0, 0.25, -1.5])
        points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 4.0, 5.0]])

        rotated = rotate_axis_angle(points, axis, angles)

        for index, angle in enumerate(angles):
            cos, sin = np.cos(angle), np.sin(angle)
            matrix = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
            np.testing.assert_allclose(rotated[index], matrix @ points[index], atol=1e-12)


class TestEgoMotion:
    def test_invert_ego_motion_matches_interpolating_a_pose_per_point(self):
        # The grouped implementation is an exact rearrangement, not an approximation, so it has to
        # agree with the per-point reading to far under any distance the filters can resolve. What
        # is left, ~2 nm, is the reference's own float64 noise from working at map scale.
        #
        # The tolerance also guards slerp's nearly-parallel threshold: widen that and lerp stands in
        # for the geodesic across the separations ego poses really have, which shows up here as a
        # 0.25 um disagreement.
        table = yawing_drive_pose_table()
        rng = np.random.default_rng(0)
        points = rng.uniform(-120.0, 120.0, size=(20_000, 3)).astype(np.float32)
        # Ring-major acquisition order: times sweep repeatedly, so points do not arrive sorted.
        sweep = np.tile(np.linspace(0.0, 0.1, 500), 40)
        point_times = table.times[0] + 0.15 + sweep

        grouped = invert_ego_motion(points, table.times[0] + 0.2, point_times, table)
        reference = invert_ego_motion_per_point(points, table.times[0] + 0.2, point_times, table)

        assert np.abs(grouped - reference).max() < 1e-7

    def test_invert_ego_motion_clamps_outside_the_pose_table(self):
        # Points acquired before the first pose or after the last must use the end poses, not
        # extrapolate off them.
        table = yawing_drive_pose_table(duration_s=0.1)
        points = np.array([[10.0, 1.0, 0.5], [10.0, 1.0, 0.5]], dtype=np.float32)
        outside = np.array([table.times[0] - 5.0, table.times[-1] + 5.0], dtype=np.float64)

        grouped = invert_ego_motion(points, table.times[0], outside, table)
        reference = invert_ego_motion_per_point(points, table.times[0], outside, table)

        np.testing.assert_allclose(grouped, reference, atol=1e-6)

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
