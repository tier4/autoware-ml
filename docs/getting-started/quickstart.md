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

## 3. Generate Dataset Info Files

Autoware-ML needs preprocessed info files that index the dataset:

```bash
autoware-ml create-dataset \
    --dataset nuscenes \
    --task segmentation3d \
    --root-path data/nuscenes \
    --out-dir data/nuscenes/info \
    --version v1.0-trainval
```

This creates pickle files for train/val splits.

## 4. Train the Model

```bash
autoware-ml train --config-name segmentation3d/ptv3/voxel005_102m_nuscenes
```

Training progress appears in your terminal. Checkpoints are saved automatically.

## 5. Monitor with MLflow

```bash
autoware-ml mlflow ui --port 5000
```

Open [http://localhost:5000](http://localhost:5000) to view loss curves, metrics, and hyperparameters.

## 6. Export for Deployment

```bash
autoware-ml deploy \
    --config-name segmentation3d/ptv3/voxel005_102m_nuscenes \
    --weights mlruns/segmentation3d/ptv3/voxel005_102m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

This generates an ONNX file. TensorRT export is disabled because PTv3 requires a runtime with matching sparse convolution plugins.

To evaluate a trained checkpoint before deployment:

```bash
autoware-ml test \
    --config-name segmentation3d/ptv3/voxel005_102m_nuscenes \
    --weights mlruns/segmentation3d/ptv3/voxel005_102m_nuscenes/<run_id>/artifacts/checkpoints/best.ckpt
```
