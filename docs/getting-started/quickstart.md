---
icon: lucide/zap
---

# Quick Start

This guide gets you from zero to a trained model. We'll train a PTv3 3D semantic segmentation model using the NuScenes dataset.

!!! info "Prerequisites"
    Make sure you finished the [Installation](installation.md) guide.

## 1. Setup Dataset

Download the NuScenes full dataset (v1.0) from the [official website](https://www.nuscenes.org/nuscenes) after registration.
After the download, confirm that the dataset is located at `$AUTOWARE_ML_DATA_PATH/nuscenes`.

## 2. Launch the Container

```bash
cd ~/autoware-ml
./docker/container.sh --run
```

## 3. Train the Model

```bash
autoware-ml train --config-name segmentation3d/ptv3/voxel005_51m_nuscenes
```

Training progress appears in your terminal. Checkpoints are saved automatically.

## 4. Monitor with MLflow

```bash
autoware-ml mlflow ui --port 5000
```

Open [http://localhost:5000](http://localhost:5000) to view loss curves, metrics, and hyperparameters.

## 5. Export for Deployment

```bash
autoware-ml deploy \
    --config-name segmentation3d/ptv3/voxel005_51m_nuscenes \
    --weights mlruns/segmentation3d/ptv3/voxel005_51m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

This generates an ONNX file. TensorRT export is disabled because PTv3 requires a runtime with matching sparse convolution plugins.

To evaluate a trained checkpoint before deployment:

```bash
autoware-ml test \
    --config-name segmentation3d/ptv3/voxel005_51m_nuscenes \
    --weights mlruns/segmentation3d/ptv3/voxel005_51m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```
