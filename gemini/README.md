# LiDAR-LiDAR Miscalibration Visualizer

This tool provides a graphical interface for visualizing and debugging LiDAR-LiDAR miscalibrations using the `autoware-ml` pipeline. It allows users to project 3D pointclouds into various 2D formats (Depth Maps, Spherical Projections) to prepare data for ResNet-based calibration models.

## Features

- **Tkinter Control Panel**: Intuitive UI for selecting datasets, frames, and LiDAR sources.
- **Dynamic 3D Visualization**: Native integration with [Rerun](https://rerun.io/) for high-performance 3D pointcloud rendering.
- **Virtual Camera Control**:
  - **Extrinsics**: Real-time adjustment of X, Y, Z translation and Roll, Pitch, Yaw rotation.
  - **Intrinsics**: Configurable focal length, principal point, and target resolution.
- **Multiple Projection Methods**:
  - **Unified Global Frame**: View aligned pointclouds in 3D world space.
  - **Depth Map Accumulation**: Z-buffer projection to a dense 2D array.
  - **Spherical Panoramic**: Azimuth/Elevation mapping for wide-FOV visualization.
- **Session Persistence**: Automatically remembers the last used dataset and pickle file paths.
- **LiDAR Alignment**: Option to toggle LiDAR-to-Ego transformations to verify extrinsic calibration accuracy.
- **Configuration Export**: Save your tuned camera parameters to JSON for use in training or inference pipelines.

## Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install numpy rerun-sdk
```
*Note: `tkinter` is usually included with standard Python installations.*

## Usage

1. **Launch the Application**:
   ```bash
   python gemini/lidar_visualizer_gui.py
   ```

2. **Setup Data**:
   - **Select Base Dir**: Point to the root of your dataset (where `LIDAR_CONCAT` or `LIDAR_CONCAT_INFO` folders reside).
   - **Load Pickle**: Select the `.pkl` file containing the dataset metadata.
   - **Select Frame**: Choose the specific timestamp/frame to visualize.
   - **Select LiDARs**: Highlight exactly two LiDAR sources from the list.
   - **Launch**: Click "Process Data & Launch Rerun".

3. **Interact & Debug**:
   - Adjust the sliders in the control panel to position the virtual camera.
   - Toggle "Apply LiDAR-to-Ego Transforms" to see the effect of the calibration matrices.
   - Switch projection methods to see the exact 2D tensor format that will be fed into the ResNet model.

## File Structure

- `lidar_visualizer_gui.py`: The main application script.
- `.last_session.json`: (Auto-generated) Stores paths to previously used files.
- `README.md`: This documentation file.

## Technical Details

- **Coordination Frame**: The tool operates in the Ego vehicle frame.
- **Point Extraction**: Points are sliced from `.pcd.bin` files using indices provided in corresponding `.json` metadata files.
- **Logging Cache**: 3D pointclouds are logged as `static=True` in Rerun to minimize overhead, re-logging only when the frame or transform state changes.
