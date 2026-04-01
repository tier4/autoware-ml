import numpy as np
import pytest

from autoware_ml.transforms.common import ToTensor
from autoware_ml.transforms.point_cloud import (
    CenterShift,
    CropBoxInner,
    CropBoxOuter,
    ElasticDistortion,
    GlobalRotScaleTrans,
    GridSample,
    LoadPointsFromFile,
    PointShuffle,
    PointsRangeFilter,
    RandomDropout,
    RandomFlip3D,
    RandomRotateTargetAngle,
    RandomShift,
    SphereCrop,
)


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
                [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
            ),
            "intensity": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }

        output = PointsRangeFilter(point_cloud_range=[-1.0, -1.0, -1.0, 2.0, 2.0, 2.0])(sample)

        assert output["points"].shape[0] == 2
        assert output["intensity"].shape[0] == 2

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
        assert "idx_begin" not in output
        assert "length" not in output

    def test_random_dropout_keeps_point_arrays_aligned(self):
        np.random.seed(0)
        sample = {
            "coord": np.arange(12, dtype=np.float32).reshape(4, 3),
            "strength": np.arange(4, dtype=np.float32).reshape(4, 1),
            "segment": np.arange(4, dtype=np.int64),
        }

        output = RandomDropout(dropout_ratio=0.5, dropout_application_ratio=1.0)(sample)

        assert output["coord"].shape[0] == 2
        assert output["strength"].shape[0] == 2
        assert output["segment"].shape[0] == 2

    def test_grid_sample_keeps_arrays_aligned(self):
        sample = {
            "coord": np.array(
                [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            "strength": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "segment": np.array([10, 11, 12], dtype=np.int64),
        }

        output = GridSample(
            grid_size=0.05,
            mode="train",
            keys=("coord", "strength", "segment"),
            return_grid_coord=True,
        )(sample)

        assert output["coord"].shape[0] == output["strength"].shape[0] == output["segment"].shape[0]
        assert output["grid_coord"].shape[0] == output["coord"].shape[0]

    def test_sphere_crop_crops_all_point_arrays_consistently(self):
        sample = {
            "coord": np.arange(30, dtype=np.float32).reshape(10, 3),
            "strength": np.arange(10, dtype=np.float32).reshape(10, 1),
            "segment": np.arange(10, dtype=np.int64),
            "grid_coord": np.arange(30, dtype=np.int32).reshape(10, 3),
        }

        output = SphereCrop(point_max=4)(sample)
        output = ToTensor()(output)

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
