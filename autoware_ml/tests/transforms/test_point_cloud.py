import numpy as np
import pytest

from autoware_ml.transforms.point_cloud.crop import (
    CenterShift,
    CropBoxInner,
    CropBoxOuter,
    PointsRangeFilter,
    SphereCrop,
)
from autoware_ml.transforms.point_cloud.geometry import (
    GlobalRotScaleTrans,
    RandomFlip3D,
    RandomRotateTargetAngle,
)
from autoware_ml.transforms.point_cloud.loading import LoadPointsFromFile
from autoware_ml.transforms.point_cloud.perturbation import RandomShift, RandomStrengthJitter
from autoware_ml.transforms.point_cloud.sampling import (
    ElasticDistortion,
    PointShuffle,
    RandomDropout,
)
from autoware_ml.transforms.point_cloud.sweeps import LoadPointsFromMultiSweeps


class TestPointCloudTransforms:
    @pytest.fixture
    def point_cloud(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [-2.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        return {"points": points}

    def test_crop_box_inner(self, point_cloud):
        transform = CropBoxInner(crop_box=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        output = transform(point_cloud)

        expected = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [-2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        assert len(output["points"]) == 4
        assert np.allclose(output["points"], expected)

    def test_crop_box_outer(self, point_cloud):
        transform = CropBoxOuter(crop_box=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
        output = transform(point_cloud)

        expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32)
        assert len(output["points"]) == 3
        assert np.allclose(output["points"], expected)

    def test_point_shuffle_keeps_aligned_arrays(self):
        sample = {
            "points": np.arange(12, dtype=np.float32).reshape(4, 3),
            "labels": np.arange(4, dtype=np.int64),
        }

        output = PointShuffle()(sample)

        assert sorted(output["labels"].tolist()) == [0, 1, 2, 3]
        assert output["points"].shape == (4, 3)

    def test_points_range_filter(self):
        sample = {
            "points": np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                    [5.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "intensity": np.arange(7, dtype=np.float32),
        }

        output = PointsRangeFilter(point_cloud_range=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0])(sample)

        expected_points = np.array(
            [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
        )
        assert np.allclose(output["points"], expected_points)
        assert output["intensity"].tolist() == [0.0, 1.0, 6.0]

    def test_points_range_filter_prevents_max_bound_grid_coords(self):
        point_cloud_range = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
        sample = {
            "coord": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.999, 1.999, 1.999],
                    [2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0],
                ],
                dtype=np.float32,
            )
        }

        ranged = PointsRangeFilter(point_cloud_range=point_cloud_range)(sample)

        # points on the upper bound are excluded so voxel indices stay inside the grid
        assert ranged["coord"].shape == (2, 3)
        grid_coord = np.floor(ranged["coord"] / 1.0).astype(np.int64)
        assert np.all(grid_coord >= 0)
        assert np.all(grid_coord < 2)

    def test_random_flip3d_updates_detection_boxes(self):
        sample = {
            "points": np.array([[1.0, 2.0, 0.0, 1.0]], dtype=np.float32),
            "gt_boxes": np.array(
                [[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.25, 1.5, -0.5]],
                dtype=np.float32,
            ),
        }

        output = RandomFlip3D(flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=0.0)(sample)

        assert np.allclose(output["points"][0, :2], np.array([1.0, -2.0], dtype=np.float32))
        assert np.allclose(
            output["gt_boxes"][0, [1, 6, 8]],
            np.array([-2.0, -0.25, 0.5], dtype=np.float32),
        )

    def test_global_rot_scale_trans_updates_detection_boxes(self):
        sample = {
            "points": np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            "gt_boxes": np.array(
                [[1.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        }

        np.random.seed(0)
        output = GlobalRotScaleTrans(
            rot_range=[0.1, 0.1],
            scale_ratio_range=[2.0, 2.0],
            translation_std=[0.0, 0.0, 0.0],
        )(sample)

        assert np.allclose(
            output["gt_boxes"][0, 3:6],
            np.array([8.0, 4.0, 2.0], dtype=np.float32),
        )
        assert np.allclose(output["gt_boxes"][0, 6], 0.1)

    def test_load_points_from_file_omits_unset_slice_metadata(self, tmp_path):
        points_path = tmp_path / "points.bin"
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32).tofile(points_path)

        output = LoadPointsFromFile(load_dim=4, use_dim=[0, 1, 2, 3])(
            {"lidar_path": str(points_path)}
        )

        assert output["points"].shape == (1, 4)
        assert output["num_current_points"] == 1
        assert "idx_begin" not in output
        assert "length" not in output

    def test_multi_sweeps_time_dim_overwrites_raw_column_with_time_lag(self, tmp_path):
        key_points = np.zeros((2, 5), dtype=np.float32)
        key_points[:, 4] = -1.0
        sweep_points = np.full((3, 5), -1.0, dtype=np.float32)
        sweep_path = tmp_path / "sweep.bin"
        sweep_points.tofile(sweep_path)

        transform = LoadPointsFromMultiSweeps(
            sweeps_num=2,
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
            time_dim=4,
            sweep_selection="nearest",
            time_lag_range=[0.01, 1.0],
        )
        output = transform(
            {
                "points": key_points,
                "timestamp": 10.0,
                "sweeps": [{"lidar_path": str(sweep_path), "timestamp": 9.9}],
            }
        )

        points = output["points"]
        assert points.shape == (5, 5)
        assert np.allclose(points[:2, 4], 0.0)
        assert np.allclose(points[2:, 4], 0.1)

    def test_multi_sweeps_remove_close_removes_axis_aligned_box(self):
        """The removed region is the box |x|,|y| < close_radius, not a radial circle:
        (0.9, 0.9) lies outside the r=1.0 circle but inside the box, so only the box
        semantics remove it."""
        key_points = np.array([[5.0, 5.0, 0.0, 0.0]], dtype=np.float32)
        sweep_points = np.array(
            [
                [0.9, 0.9, 0.0, 0.0],  # inside the box, outside the circle -> removed
                [0.5, -0.5, 0.0, 0.0],  # inside the box -> removed
                [1.05, 0.0, 0.0, 0.0],  # |x| >= radius -> kept
                [0.0, -1.2, 0.0, 0.0],  # |y| >= radius -> kept
            ],
            dtype=np.float32,
        )

        transform = LoadPointsFromMultiSweeps(
            sweeps_num=2,
            load_dim=4,
            use_dim=[0, 1, 2, 3],
            remove_close=True,
            close_radius=1.0,
            sweep_selection="nearest",
            time_lag_range=[0.01, 1.0],
        )
        output = transform(
            {
                "points": key_points,
                "timestamp": 10.0,
                "sweeps": [{"points": sweep_points, "timestamp": 9.9}],
            }
        )

        points = output["points"]
        # key point + the two sweep points outside the box
        assert points.shape == (3, 4)
        assert not np.any(np.all(np.isclose(points[:, :2], [0.9, 0.9]), axis=1))
        assert np.any(np.all(np.isclose(points[:, :2], [1.05, 0.0]), axis=1))
        assert np.any(np.all(np.isclose(points[:, :2], [0.0, -1.2]), axis=1))

    @staticmethod
    def _aged_sweeps():
        """Three stored sweeps at 0.1, 0.2 and 0.5 s before the current frame."""
        return [
            {"points": np.full((1, 4), 10.0, dtype=np.float32), "timestamp": 9.9},
            {"points": np.full((1, 4), 20.0, dtype=np.float32), "timestamp": 9.8},
            {"points": np.full((1, 4), 30.0, dtype=np.float32), "timestamp": 9.5},
        ]

    def _load_with(self, selection, window, sweeps_num=2):
        return LoadPointsFromMultiSweeps(
            sweeps_num=sweeps_num,
            load_dim=4,
            use_dim=[0, 1, 2, 3],
            sweep_selection=selection,
            time_lag_range=window,
        )

    def test_nearest_selection_takes_the_most_recent_eligible_sweep(self):
        transform = self._load_with("nearest", [0.05, 0.25])

        output = transform(
            {
                "points": np.zeros((1, 4), dtype=np.float32),
                "timestamp": 10.0,
                "sweeps": self._aged_sweeps(),
            }
        )

        assert output["points"].shape == (2, 4)
        assert np.allclose(output["points"][1], 10.0)

    def test_time_lag_range_makes_sweeps_outside_the_window_unavailable(self):
        # only the 0.2 s sweep is eligible: 0.1 s is too recent and 0.5 s too old
        transform = self._load_with("nearest", [0.15, 0.25])

        output = transform(
            {
                "points": np.zeros((1, 4), dtype=np.float32),
                "timestamp": 10.0,
                "sweeps": self._aged_sweeps(),
            }
        )

        assert np.allclose(output["points"][1], 20.0)

    def test_a_frame_whose_sweeps_are_all_stale_runs_without_them(self):
        transform = self._load_with("nearest", [0.05, 0.25])

        output = transform(
            {
                "points": np.zeros((1, 4), dtype=np.float32),
                "timestamp": 10.0,
                "sweeps": [{"points": np.ones((1, 4), dtype=np.float32), "timestamp": 9.0}],
            }
        )

        assert output["points"].shape == (1, 4)
        assert output["num_current_points"] == 1

    def test_random_selection_samples_only_among_the_eligible_sweeps(self, monkeypatch):
        calls = {}

        def fake_choice(num_entries, size, replace):
            calls["args"] = (num_entries, size, replace)
            return np.array([1])

        monkeypatch.setattr(
            "autoware_ml.transforms.point_cloud.sweeps.np.random.choice", fake_choice
        )
        # window admits the 0.1 s and 0.2 s sweeps but not the 0.5 s one
        transform = self._load_with("random", [0.05, 0.25])

        output = transform(
            {
                "points": np.zeros((1, 4), dtype=np.float32),
                "timestamp": 10.0,
                "sweeps": self._aged_sweeps(),
            }
        )

        assert calls["args"] == (2, 1, False)
        assert np.allclose(output["points"][1], 20.0)

    def test_random_selection_keeps_the_appended_sweeps_ordered_by_recency(self, monkeypatch):
        monkeypatch.setattr(
            "autoware_ml.transforms.point_cloud.sweeps.np.random.choice",
            lambda num_entries, size, replace: np.array([2, 0]),
        )
        transform = self._load_with("random", [0.05, 0.6], sweeps_num=3)

        output = transform(
            {
                "points": np.zeros((1, 4), dtype=np.float32),
                "timestamp": 10.0,
                "sweeps": self._aged_sweeps(),
            }
        )

        # sampled the 0.5 s and 0.1 s sweeps, appended newest first
        assert np.allclose(output["points"][1], 10.0)
        assert np.allclose(output["points"][2], 30.0)

    def test_multi_sweeps_requires_timestamps_to_age_the_sweeps(self):
        transform = self._load_with("nearest", [0.05, 0.25])

        with pytest.raises(KeyError, match="timestamp"):
            transform(
                {
                    "points": np.zeros((1, 4), dtype=np.float32),
                    "sweeps": [{"points": np.ones((1, 4), dtype=np.float32), "timestamp": 9.9}],
                }
            )
        with pytest.raises(KeyError, match="sweep 'timestamp'"):
            transform(
                {
                    "points": np.zeros((1, 4), dtype=np.float32),
                    "timestamp": 10.0,
                    "sweeps": [{"points": np.ones((1, 4), dtype=np.float32)}],
                }
            )

    def test_multi_sweeps_rejects_an_unknown_selection_or_window(self):
        with pytest.raises(ValueError, match="sweep_selection"):
            self._load_with("newest", [0.05, 0.25])
        with pytest.raises(ValueError, match="min time lag < max time lag"):
            self._load_with("nearest", [0.25, 0.05])
        with pytest.raises(ValueError, match=r"\[min, max\]"):
            self._load_with("nearest", [0.25])

    def test_multi_sweeps_exposes_current_frame_as_leading_block(self):
        key_points = np.zeros((2, 5), dtype=np.float32)
        sweep_points = np.ones((3, 5), dtype=np.float32) * 5.0
        transform = LoadPointsFromMultiSweeps(
            sweeps_num=2,
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
            time_dim=4,
            sweep_selection="nearest",
            time_lag_range=[0.01, 1.0],
        )

        output = transform(
            {
                "points": key_points,
                "timestamp": 10.0,
                "sweeps": [{"points": sweep_points, "timestamp": 9.9}],
            }
        )

        assert output["num_current_points"] == 2
        assert np.all(output["points"][:2, 4] == 0.0)
        assert np.allclose(output["points"][2:, 4], 0.1)

    def test_multi_sweeps_pads_empty_sweeps_with_non_current_lag(self):
        """Scene-first padding stands in for sweeps: the copies carry the minimum
        admissible lag so current-frame selections never count them."""
        transform = LoadPointsFromMultiSweeps(
            sweeps_num=3,
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
            time_dim=4,
            pad_empty_sweeps=True,
            sweep_selection="nearest",
            time_lag_range=[0.05, 0.25],
        )

        output = transform({"points": np.zeros((2, 5), dtype=np.float32), "sweeps": []})

        assert output["points"].shape == (6, 5)
        assert output["num_current_points"] == 2
        assert np.all(output["points"][:2, 4] == 0.0)
        assert np.allclose(output["points"][2:, 4], 0.05)

    def test_multi_sweeps_rejects_a_zero_minimum_time_lag(self):
        """The current frame owns lag 0, so a window admitting zero-lag sweeps is a config
        error: such a sweep would be indistinguishable from the current frame downstream."""
        with pytest.raises(ValueError, match="0 < min time lag"):
            LoadPointsFromMultiSweeps(
                sweeps_num=2,
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
                sweep_selection="nearest",
                time_lag_range=[0.0, 1.0],
            )

    def test_multi_sweeps_rejects_per_sensor_slicing(self):
        transform = LoadPointsFromMultiSweeps(
            sweeps_num=2,
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
            sweep_selection="nearest",
            time_lag_range=[0.01, 1.0],
        )

        with pytest.raises(ValueError, match="idx_begin"):
            transform({"lidar_path": "unused.bin", "idx_begin": 0, "length": 3})

    def test_a_sweepless_frame_needs_no_timestamp(self):
        """The current frame carries lag 0 by definition, so ageing information is only needed
        once there is a sweep to age. Scene-first frames therefore load without a timestamp."""
        transform = LoadPointsFromMultiSweeps(
            sweeps_num=2,
            load_dim=5,
            use_dim=[0, 1, 2, 3, 4],
            time_dim=4,
            sweep_selection="nearest",
            time_lag_range=[0.01, 1.0],
        )

        output = transform({"points": np.zeros((1, 5), dtype=np.float32), "sweeps": []})

        assert output["points"].shape == (1, 5)
        assert output["points"][0, 4] == 0.0
        assert output["num_current_points"] == 1

    def test_random_dropout_keeps_point_arrays_aligned(self):
        np.random.seed(0)
        sample = {
            "coord": np.arange(12, dtype=np.float32).reshape(4, 3),
            "strength": np.arange(4, dtype=np.float32).reshape(4, 1),
            "segment": np.arange(4, dtype=np.int64),
        }

        output = RandomDropout(dropout_ratio=0.5, p=1.0)(sample)

        assert output["coord"].shape[0] == 2
        assert output["strength"].shape[0] == 2
        assert output["segment"].shape[0] == 2

    def test_random_dropout_respects_application_probability(self):
        sample = {
            "coord": np.arange(12, dtype=np.float32).reshape(4, 3),
            "strength": np.arange(4, dtype=np.float32).reshape(4, 1),
        }

        output = RandomDropout(dropout_ratio=0.5, p=0.0)(sample)

        assert output["coord"].shape[0] == 4
        assert output["strength"].shape[0] == 4

    def test_random_rotate_target_angle_rotates_boxes_with_points(self):
        sample = {
            "coord": np.array([[2.0, 0.0, 1.0]], dtype=np.float32),
            "gt_boxes": np.array([[2.0, 0.0, 1.0, 4.0, 2.0, 1.5, 0.1, 3.0, 0.0]], dtype=np.float32),
        }

        output = RandomRotateTargetAngle(angle=[0.5], center=[0.0, 0.0, 0.0], p=1.0)(sample)

        box = output["gt_boxes"][0]
        assert np.allclose(output["coord"], [[0.0, 2.0, 1.0]], atol=1e-6)
        assert np.allclose(box[:3], [0.0, 2.0, 1.0], atol=1e-6)
        assert np.allclose(box[3:6], [4.0, 2.0, 1.5])
        assert np.isclose(box[6], 0.1 + 0.5 * np.pi)
        assert np.allclose(box[7:9], [0.0, 3.0], atol=1e-6)

    def test_random_rotate_target_angle_rejects_boxes_off_z_axis(self):
        sample = {
            "coord": np.zeros((1, 3), dtype=np.float32),
            "gt_boxes": np.zeros((1, 9), dtype=np.float32),
        }

        with pytest.raises(ValueError, match="axis='z'"):
            RandomRotateTargetAngle(angle=[0.5], axis="x", p=1.0)(sample)

    def test_random_strength_jitter_stays_normalized_and_monotonic(self):
        np.random.seed(0)
        sample = {"strength": np.linspace(0.0, 1.0, 5, dtype=np.float32).reshape(5, 1)}

        output = RandomStrengthJitter(
            gamma_range=[0.8, 1.25], scale_range=[0.9, 1.1], shift_range=[-0.02, 0.02]
        )(sample)

        strength = output["strength"]
        assert strength.shape == (5, 1)
        assert strength.dtype == np.float32
        assert strength.min() >= 0.0
        assert strength.max() <= 1.0
        assert np.all(np.diff(strength[:, 0]) >= 0.0)

    def test_sphere_crop_crops_all_point_arrays_consistently(self):
        sample = {
            "coord": np.arange(30, dtype=np.float32).reshape(10, 3),
            "strength": np.arange(10, dtype=np.float32).reshape(10, 1),
            "segment": np.arange(10, dtype=np.int64),
            "grid_coord": np.arange(30, dtype=np.int32).reshape(10, 3),
        }

        output = SphereCrop(point_max=4)(sample)

        assert output["coord"].shape[0] == 4
        assert output["strength"].shape[0] == 4
        assert output["segment"].shape[0] == 4
        assert output["grid_coord"].shape[0] == 4

    def test_sphere_crop_center_mode_is_deterministic(self):
        sample = {
            "coord": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [11.0, 0.0, 0.0],
                    [12.0, 0.0, 0.0],
                    [50.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "segment": np.arange(5, dtype=np.int64),
        }

        output = SphereCrop(point_max=3, mode="center")(sample)

        assert np.array_equal(output["segment"], np.array([1, 2, 3], dtype=np.int64))

    def test_random_rotate_target_angle_rotates_by_selected_angle(self):
        sample = {"coord": np.array([[1.0, 0.0, 0.0]], dtype=np.float32)}

        np.random.seed(0)
        output = RandomRotateTargetAngle(angle=(0.5,), center=[0.0, 0.0, 0.0], p=1.0)(sample)

        assert np.allclose(
            output["coord"], np.array([[0.0, 1.0, 0.0]], dtype=np.float32), atol=1e-5
        )

    def test_random_rotate_target_angle_respects_probability(self, monkeypatch):
        sample = {"coord": np.array([[1.0, 0.0, 0.0]], dtype=np.float32)}
        monkeypatch.setattr(np.random, "rand", lambda: 0.75)

        output = RandomRotateTargetAngle(angle=(0.5,), center=[0.0, 0.0, 0.0], p=0.5)(sample)

        assert np.allclose(output["coord"], np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    def test_random_flip_uses_configured_probability_per_axis(self, monkeypatch):
        sample = {
            "coord": np.array([[1.0, 2.0, 0.0]], dtype=np.float32),
            "normal": np.array([[0.5, 0.25, 1.0]], dtype=np.float32),
        }
        calls = iter([0.2, 0.8])
        monkeypatch.setattr(np.random, "rand", lambda: next(calls))

        # flip_ratio_bev_horizontal flips y-axis; flip_ratio_bev_vertical flips x-axis.
        # rand() returns 0.2 < 0.5 so horizontal flip triggers (y flipped),
        # rand() returns 0.8 >= 0.5 so vertical flip does not trigger.
        output = RandomFlip3D(flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5)(sample)

        assert np.allclose(output["coord"], np.array([[1.0, -2.0, 0.0]], dtype=np.float32))
        assert np.allclose(output["normal"], np.array([[0.5, -0.25, 1.0]], dtype=np.float32))

    def test_random_flip_updates_boxes_on_coord(self):
        sample = {
            "coord": np.array([[1.0, 2.0, 0.0]], dtype=np.float32),
            "gt_boxes": np.array(
                [[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.25, 1.5, -0.5]],
                dtype=np.float32,
            ),
        }

        output = RandomFlip3D(flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=0.0)(sample)

        assert np.allclose(output["coord"][0, :2], np.array([1.0, -2.0], dtype=np.float32))
        assert np.allclose(
            output["gt_boxes"][0, [1, 6, 8]],
            np.array([-2.0, -0.25, 0.5], dtype=np.float32),
        )

    def test_global_rot_scale_trans_geometry_updates_boxes(self):
        sample = {
            "coord": np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            "gt_boxes": np.array(
                [[1.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
        }

        np.random.seed(0)
        output = GlobalRotScaleTrans(
            rot_range=[0.1, 0.1],
            scale_ratio_range=[2.0, 2.0],
            translation_std=None,
        )(sample)

        assert np.allclose(
            output["gt_boxes"][0, 3:6],
            np.array([8.0, 4.0, 2.0], dtype=np.float32),
        )
        assert np.allclose(output["gt_boxes"][0, 6], 0.1)

    def test_random_shift_translates_all_points(self):
        sample = {"coord": np.zeros((2, 3), dtype=np.float32)}

        np.random.seed(0)
        output = RandomShift(shift=[0.5, 0.5, 0.5])(sample)

        assert np.allclose(output["coord"][0], output["coord"][1])
        assert not np.allclose(output["coord"][0], np.zeros(3, dtype=np.float32))

    def test_center_shift_can_keep_z_unchanged(self):
        sample = {"coord": np.array([[0.0, 0.0, 1.0], [2.0, 2.0, 3.0]], dtype=np.float32)}

        output = CenterShift(apply_z=False)(sample)

        assert np.allclose(output["coord"][:, 2], np.array([1.0, 3.0], dtype=np.float32))
        assert np.allclose(output["coord"][:, :2].mean(axis=0), np.zeros(2, dtype=np.float32))

    def test_elastic_distortion_preserves_shape(self):
        sample = {"coord": np.random.rand(8, 3).astype(np.float32)}

        output = ElasticDistortion(distortion_params=[[0.2, 0.4]])(sample)

        assert output["coord"].shape == (8, 3)
