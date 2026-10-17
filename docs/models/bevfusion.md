---
icon: lucide/scan-line
---

# BEVFusion

BEVFusion is a camera-LiDAR 3D object detection model integrated under the `detection3d` task namespace. It combines a sparse-voxel LiDAR branch (hard voxelization + sparse 3D convolution encoder), a multiview image branch, a LiDAR-depth-guided `DepthLSSTransform` view transform, and a convolutional BEV fusion layer before the TransFusion detection head. The bundled experiments are LiDAR-only: they leave `camera_network` unset and skip the image branch entirely.

## Summary

| Property     | Value                                                                                       |
| ------------ | ------------------------------------------------------------------------------------------- |
| Task         | 3D object detection                                                                         |
| Modality     | LiDAR (camera-LiDAR supported by the model, no bundled experiment yet)                      |
| Input        | Point cloud, optionally with synchronized multiview images                                  |
| Output       | 3D bounding boxes and class scores                                                          |
| Architecture | Sparse voxel encoder + multiview camera backbone/FPN + DepthLSS + fusion + TransFusion head |
| Datasets     | T4Dataset, nuScenes                                                                         |

## Available Configurations

Experiment configs live under `autoware_ml/configs/tasks/detection3d/bevfusion/<dataset>/` and are passed to the CLI as the path relative to `tasks/`. Every bundled leaf is LiDAR-only and sits under a `bevfusion_lidar/` directory. Camera-LiDAR experiments will add a sibling directory that configures the camera branch and fuser.

| Leaf                                                        | Database          | Weights                                             |
| ----------------------------------------------------------- | ----------------- | --------------------------------------------------- |
| `t4dataset/bevfusion_lidar/lidar_<variant>_gen1_base`       | T4Dataset gen1    | Trained from scratch                                |
| `t4dataset/bevfusion_lidar/lidar_<variant>_jpntaxi_base`    | T4Dataset jpntaxi | Fine-tuned from a checkpoint given with `--weights` |
| `t4dataset/bevfusion_lidar/lidar_<variant>_gen2_base`       | T4Dataset gen2    | Fine-tuned from a checkpoint given with `--weights` |
| `t4dataset/bevfusion_lidar/lidar_<variant>_j6gen2_base`     | T4Dataset j6gen2  | Fine-tuned from a checkpoint given with `--weights` |
| `nuscenes/bevfusion_lidar/lidar_<variant>_nuscenes_<range>` | nuScenes          | Trained from scratch                                |

The leaf name spells out its variant: the voxel size, the backbone and neck, the batch size, the epoch count and the detection range, so the values live in one place, the config. The configs are layered so that each file owns one concern:

| File                                                | Owns                                                                                         |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `base.yaml`                                         | Model architecture, voxel preprocessor, optimizer, trainer settings, and ONNX export modules |
| `<dataset>/base_<range>.yaml`                       | Point cloud range, evaluation ranges, proposal count, NMS and score thresholds, transforms   |
| `<dataset>/bevfusion_lidar/base_lidar_<voxel>.yaml` | Voxel size, sparse encoder grid shapes, and head output stride                               |
| `<dataset>/bevfusion_lidar/lidar_<variant>.yaml`    | Database selection, batch size, experiment name, epochs, and optional pretrained weights     |

## Training

Train the base model from scratch:

```bash
autoware-ml train --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m_gen1_base
```

The `jpntaxi`, `gen2` and `j6gen2` configs declare `weights: ???`, so they must be started from a pretrained checkpoint. Pass it with `--weights`. The config already lowers the learning rate for fine-tuning:

```bash
autoware-ml train \
    --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/bevfusion/T4Dataset-gen1_base-v1.0.0/gen1_base/lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m/<run_id>/artifacts/checkpoints/best.ckpt
```

For a pipeline validation run:

```bash
autoware-ml train \
    --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_30e_t4dataset_120m_gen1_base \
    +trainer.fast_dev_run=true
```

Checkpoints are written to `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/` as `best.ckpt` and `last.ckpt`. Resume an interrupted run with `--resume-checkpoint` pointing at `last.ckpt`.

## Evaluation

```bash
autoware-ml test \
    --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/bevfusion/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m/<run_id>/artifacts/checkpoints/best.ckpt
```

Evaluation runs on a single device by default so metrics are deterministic. Pass `--use-config-devices` to keep the `trainer.devices` from the config.

## Deployment

```bash
autoware-ml deploy \
    --config-name detection3d/bevfusion/t4dataset/bevfusion_lidar/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m_j6gen2_base \
    --weights mlruns/bevfusion/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base/lidar_voxel0170_second_secfpn_b16_50e_t4dataset_120m/<run_id>/artifacts/checkpoints/best.ckpt \
    --release v1.0.0
```

The export produces the ONNX modules for `autoware_universe/perception/autoware_bevfusion`. LiDAR-only configurations export a single `bevfusion_lidar.onnx` main body restricted to the first three inputs below, and the runtime node consumes it unchanged. Camera-LiDAR models export two modules: `bevfusion_image_backbone.onnx` encodes raw `uint8` multiview images (the training-time `1 / 255` normalization is baked into the graph) into `image_feats`, and `bevfusion_camera_lidar.onnx` is the main body consuming those features together with the LiDAR depth maps and precomputed `bev_pool` metadata. **The camera-LiDAR main body is not consumed by the current runtime node**: its inputs changed and the node has not been updated yet, see [Runtime compatibility](#runtime-compatibility).

| Input Tensor           | Description                                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `voxels`               | Voxelized LiDAR features                                                                                               |
| `coors`                | Voxel coordinates in `(z, y, x)` order, batch-free                                                                     |
| `num_points_per_voxel` | Point count per voxel                                                                                                  |
| `image_feats`          | Image backbone features from the image backbone ONNX                                                                   |
| `depth_maps`           | Sparse LiDAR depth maps `(num_cameras, 1, H, W)` at the network image resolution, built by `LiDARDepthSparseTransform` |
| `geom_feats`           | Precomputed BEV pooling coordinates                                                                                    |
| `kept`                 | Boolean mask for valid projected frustum points                                                                        |
| `ranks`                | Sorted BEV pooling ranks                                                                                               |
| `indices`              | Sorting indices for pooled frustum features                                                                            |

Every main-body module returns the runtime detection interface: `bbox_pred` with the raw regression channels `(center, height, dim, rot, vel)` per proposal, `score` with per-proposal confidences, and `label_pred` with per-proposal class labels. Metric-space decoding happens in the runtime node. The `height` channel is the z of the box gravity center, which is how `autoware_bevfusion` reads it.

Exports land in `mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/exports/` of a dedicated deploy run linked to the training run. Every module is stamped with its provenance and the `--release` version. Omit `--release` only for throwaway exports, which are stamped `unversioned`. TensorRT engine generation is disabled (`deploy.tensorrt.enabled=false`) because the runtime builds engines itself using the custom sparse-convolution and `bev_pool` plugins.

## Runtime compatibility

### Breaking change: camera-LiDAR main body inputs

The `bevfusion_camera_lidar.onnx` main body no longer takes `points`, `lidar2image` and `img_aug_matrix`. It takes `depth_maps` instead. Previously the graph projected the point cloud onto every camera itself. Now the sparse depth maps are built outside the graph by `LiDARDepthSparseTransform` (`autoware_ml/transforms/camera_lidar/depth.py`), both in the training pipeline and in the export sample. The `kept` input is now sized by `num_frustum_points` rather than `num_kept`.

| Main body input                           | Before (`main`) | Now     |
| ----------------------------------------- | --------------- | ------- |
| `voxels`, `coors`, `num_points_per_voxel` | yes             | yes     |
| `image_feats`                             | yes             | yes     |
| `points`                                  | yes             | removed |
| `lidar2image`                             | yes             | removed |
| `img_aug_matrix`                          | yes             | removed |
| `depth_maps`                              | no              | **new** |
| `geom_feats`, `kept`, `ranks`, `indices`  | yes             | yes     |

`autoware_bevfusion` (`lib/bevfusion_trt.cpp`) still binds `points`, `img_aug_matrix` and `lidar2image` in fusion mode and does not build depth maps, so a camera-LiDAR model exported from this version does not load in the current node, and the camera-LiDAR ONNX published for the current node does not match this export. LiDAR-only exports are unaffected. No shipped experiment runs the camera branch yet, so nothing deployed is broken today, but the runtime must change before the first camera-LiDAR model from this repository ships.

A second behaviour change, outside the export ABI: a camera-only model (`lidar_network: null`) now feeds the image BEV straight into the detection head. The BEV backbone and neck (`pts_backbone` / `pts_neck`) live on the LiDAR branch and do not run without it, unlike upstream BEVFusion where every BEV map goes through them. A camera-only head therefore has to take the view-transform channels as `in_channels`. There is no camera-only export, so the runtime is not affected.

### Planned runtime change (`autoware_bevfusion`)

The node has to build the depth maps that the graph used to build, with the same geometry it already uses for `lidar2image` and `img_aug_matrix`.

1. **Depth map kernel.** Add a CUDA kernel in `lib/preprocess/preprocess_kernel.cu` that projects the sweep points (`points_d_`, the same buffer fed to voxelization) onto every camera and writes a `(num_cameras, 1, roi_height, roi_width)` float32 buffer, reproducing `LiDARDepthSparseTransform.build_depth_maps`.
2. **Bindings.** In `addCameraNetworkIO`, `addCameraProfileDims`, `setTensorAddress` and `configureTensorRTInputs`, replace `points`, `img_aug_matrix` and `lidar2image` with `depth_maps` (`Dims{4, {num_cameras, 1, roi_height, roi_width}}`, min = opt = max). Drop `points_d_` from the engine inputs (it stays as the voxelization buffer) and `img_aug_matrix_d_` from device memory, and keep the host-side `img_aug_matrices_` and the `lidar2image` values for the kernel. Bindings are by name, so the new input order does not matter.
3. **Pipeline.** Allocate `depth_maps_d_` (`num_cameras * roi_height * roi_width` floats) in fusion mode, clear it in `clearDeviceMemory`, and launch the kernel in `preProcess` after `generateSweepPoints` and before `configureTensorRTInputs`. No new ROS parameters are needed: `num_cameras`, `roi_height` and `roi_width` already exist in `ml_package_bevfusion_camera_lidar.param.yaml`.
4. **ABI transition.** Either gate the fusion bindings on the input names of the loaded network (`points` present means the old graph, `depth_maps` the new one) for one release, or bump the camera-LiDAR model package version and switch outright. The node README, the parameter schema and `CHANGELOG.rst` must document the new input set.
5. **Validation.** Export a camera-LiDAR model from this repository, dump the `depth_maps` the export sample used and compare them with the kernel output for the same frame and calibration (equal everywhere except duplicate-hit pixels), then compare `bbox_pred` / `score` / `label_pred` between the PyTorch model and the TensorRT engine on a few frames.

## Implementation

| Path                                                                         | Description                                      |
| ---------------------------------------------------------------------------- | ------------------------------------------------ |
| `autoware_ml/models/detection3d/main_modules/bevfusion.py`                   | BEVFusion model wrapper                          |
| `autoware_ml/models/detection3d/main_modules/bevfusions/bevfusion_lidar.py`  | LiDAR BEV network                                |
| `autoware_ml/models/detection3d/main_modules/bevfusions/bevfusion_camera.py` | Multiview image network                          |
| `autoware_ml/models/detection3d/main_modules/bevfusions/export_wrappers.py`  | ONNX export wrappers                             |
| `autoware_ml/models/detection3d/view_transforms/depth_lss.py`                | Multiview image-to-BEV transform                 |
| `autoware_ml/models/detection3d/main_modules/bevfusions/fuser.py`            | Camera-LiDAR BEV fusion layer                    |
| `autoware_ml/models/detection3d/encoders/voxel.py`                           | Hard voxelization feature encoder                |
| `autoware_ml/models/detection3d/encoders/sparse/sparse_encoder.py`           | Sparse 3D convolution encoder                    |
| `autoware_ml/models/detection3d/backbones/second.py`                         | SECOND backbone                                  |
| `autoware_ml/models/detection3d/necks/second_fpn.py`                         | SECONDFPN neck                                   |
| `autoware_ml/models/detection3d/heads/transfusions/transfusion_head.py`      | TransFusion detection head                       |
| `autoware_ml/models/common/backbones/resnet.py`                              | ResNet multiview image backbone                  |
| `autoware_ml/models/common/necks/lss_fpn.py`                                 | Multiview image neck                             |
| `autoware_ml/models/detection3d/task_modules/`                               | Shared assigners, costs, coders                  |
| `autoware_ml/datamodule/t4dataset/dataset.py`                                | T4Dataset records as a dataset                   |
| `autoware_ml/datamodule/nuscenes/dataset.py`                                 | nuScenes dataset                                 |
| `autoware_ml/datamodule/common/detection3d.py`                               | 3D boxes of a record                             |
| `autoware_ml/transforms/camera/`                                             | Multiview image loading and augmentation         |
| `autoware_ml/configs/datamodule/`                                            | Datamodule, dataset and transform configurations |
| `autoware_ml/preprocessing/detection3d/point_pillar_preprocessor.py`         | Pillar preprocessing                             |
| `autoware_ml/configs/tasks/detection3d/bevfusion/`                           | Experiment configurations                        |

## Acknowledgment

The Autoware-ML BEVFusion implementation was ported from the official mmdetection3d
project by OpenMMLab.

<!-- cspell:ignore Zhijian -->
- Repository: <https://github.com/open-mmlab/mmdetection3d>
- License: Apache License 2.0
- Paper: Liu, Zhijian, et al. "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation" ICRA, 2023.
