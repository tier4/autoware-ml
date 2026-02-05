import numpy as np
import pytest

from autoware_ml.transforms.lidar.lidar import CropBoxInner, CropBoxOuter


class TestLidarTransforms:
    @pytest.fixture
    def point_cloud(self):
        # Create points: inside, outside, and boundary
        # Box: [-1, -1, -1, 1, 1, 1]
        points = np.array(
            [
                [0.0, 0.0, 0.0],  # Inside
                [0.5, 0.5, 0.5],  # Inside
                [2.0, 0.0, 0.0],  # Outside X
                [0.0, 2.0, 0.0],  # Outside Y
                [0.0, 0.0, 2.0],  # Outside Z
                [-2.0, 0.0, 0.0],  # Outside -X
                [1.0, 1.0, 1.0],  # Boundary (Inside due to <=)
            ],
            dtype=np.float32,
        )
        return {"points": points}

    def test_crop_box_inner(self, point_cloud):
        # CropBoxInner REMOVES points INSIDE (Keeps OUTSIDE)
        box = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        transform = CropBoxInner(crop_box=box)
        output = transform(point_cloud)

        points = output["points"]
        # Expected points: (2,0,0), (0,2,0), (0,0,2), (-2,0,0) -> 4 points
        # 1.0 is considered inside (<= 1.0), so removed.
        assert len(points) == 4
        expected = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [-2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        assert np.allclose(points, expected)

    def test_crop_box_outer(self, point_cloud):
        # CropBoxOuter REMOVES points OUTSIDE (Keeps INSIDE)
        box = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        transform = CropBoxOuter(crop_box=box)
        output = transform(point_cloud)

        points = output["points"]
        # Expected: (0,0,0), (0.5,0.5,0.5), (1,1,1)
        assert len(points) == 3
        expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32)
        assert np.allclose(points, expected)
