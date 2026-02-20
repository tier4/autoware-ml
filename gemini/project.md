# Project description

I am implementing a Lidar-Lidar miscalibration detection pipeline using autoware-ml.
There is an already existing pipeline for Lidar-Camera case (resnet18_base.yaml) and we want to reuse as much of if as possible for the Lidar-Lidar case.

## Implementation Steps

### 1. Data Module Implementation
- [x] Complete `T4LidarCalibrationStatusDataset` in `autoware_ml/datamodule/t4dataset/lidar_calibration_status.py`.
    - [x] Correct JSON info pathing for LiDAR concat information.
    - [x] Implement `_split_points` to extract individual LiDAR point clouds from the concatenated cloud.
    - [x] Implement `_load_calibration_data` to calculate the relative transformation between two LiDARs.

### 2. LiDAR-LiDAR Transforms
- [x] Implement `LidarLidarCalibrationMisalignment` transform.
    - This should apply random 6-DOF noise to the relative transformation between LiDARs.
    - Similar to `CalibrationMisalignment` in `camera_lidar.py`.
- [x] Implement `LidarLidarFusion` transform.
    - This should create a 2D representation (e.g., spherical projection or BEV) that combines data from both LiDARs into a format suitable for a 2D CNN backbone (like ResNet).
    - It should produce a multi-channel "image" (e.g., depth and intensity from both sensors).

### 3. Configuration Setup
- [x] Complete `autoware_ml/configs/tasks/calibration_status/lidar_lidar/resnet18_lidarseg.yaml`.
    - Define the transformation pipeline using the new LiDAR-LiDAR transforms.
    - Configure the `CalibrationStatusClassifier` model.
    - Set appropriate hyperparameters for training.

### 4. Model Training and Validation
- [ ] Train the Lidar-Lidar miscalibration detection model.
- [ ] Evaluate the model performance on the test set.
- [ ] Visualize the results using preview transforms.
