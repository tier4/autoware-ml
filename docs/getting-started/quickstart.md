---
icon: lucide/zap
---

# Quick Start

This guide gets you from zero to a trained model. We'll train a calibration status classifier using the NuScenes dataset.

!!! info "Prerequisites"
    Make sure you finished the [Installation](installation.md) guide.

## 1. Setup Dataset

Download the NuScenes full dataset (v1.0) from the [official website](https://www.nuscenes.org/nuscenes) after registration.
After the download, confirm that the dataset is located at `$AUTOWARE_ML_DATA_PATH/nuscenes`.

## 2. Launch the Container

```bash
cd ~/autoware-ml
./docker/run.sh
```

## 3. Generate Dataset Info Files

Autoware-ML needs preprocessed info files that index the dataset:

```bash
autoware-ml create-dataset \
    --dataset nuscenes \
    --task calibration_status \
    --root-path data/nuscenes \
    --out-dir data/nuscenes/info \
    --version v1.0-trainval
```

This creates pickle files for train/val splits.

## 4. Train the Model

```bash
autoware-ml train --config-name calibration_status/resnet18_nuscenes
```

Training progress appears in your terminal. Checkpoints are saved automatically.

## 5. Monitor with MLflow

```bash
autoware-ml mlflow-ui --port 5000
```

Open [http://localhost:5000](http://localhost:5000) to view loss curves, metrics, and hyperparameters.

## 6. Export for Deployment

```bash
autoware-ml deploy \
    --config-name calibration_status/resnet18_nuscenes \
    +checkpoint=mlruns/<date>/<time>/checkpoints/best.ckpt
```

This generates ONNX and TensorRT files.
