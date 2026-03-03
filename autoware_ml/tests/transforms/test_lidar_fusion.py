import numpy as np
import pytest
from autoware_ml.transforms.lidar.lidar import LidarLidarFusion

def test_lidar_lidar_fusion():
    # Setup test data
    # Points are (N, 3+) [x, y, z, intensity, ...]
    l1_points = np.random.rand(100, 4).astype(np.float32)
    l2_points = np.random.rand(100, 4).astype(np.float32)
    
    input_dict = {
        "lidar1_points": l1_points,
        "lidar2_points": l2_points,
        "calibration_data": None # Not used currently in project_spherical call inside transform
    }
    
    width = 1024
    height = 512
    fusion = LidarLidarFusion(width=width, height=height)
    
    # Execute transform
    output = fusion.transform(input_dict)
    
    # Assertions
    assert "fused_img" in output
    fused_img = output["fused_img"]
    assert fused_img.shape == (height, width, 3)
    assert fused_img.dtype == np.float32
    
    # Check if channels are populated
    # Channel 0: L1 range
    # Channel 1: L2 range
    # Channel 2: abs diff
    assert np.all(fused_img[:, :, 2] == np.abs(fused_img[:, :, 0] - fused_img[:, :, 1]))
