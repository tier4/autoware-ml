import numpy as np
import pytest

from autoware_ml.transforms.point_cloud import (
    CropBoxInner,
    CropBoxOuter,
    LoadPointsFromFile,
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

    def test_load_points_from_file_omits_unset_slice_metadata(self, tmp_path):
        points_path = tmp_path / "points.bin"
        np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32).tofile(points_path)

        output = LoadPointsFromFile(load_dim=4, use_dim=[0, 1, 2, 3])(
            {"lidar_path": str(points_path)}
        )

        assert output["points"].shape == (1, 4)
        assert "idx_begin" not in output
        assert "length" not in output
