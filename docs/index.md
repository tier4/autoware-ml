---
icon: lucide/house
---

# Autoware-ML

**Autoware-ML** is a machine learning framework for autonomous vehicle perception tasks. Built on PyTorch Lightning and Hydra, it provides a streamlined path from training to deployment.

!!! warning "Early Alpha"
    This project (`tier4/autoware-ml`) is in Early Alpha and will replace [tier4/AWML](https://github.com/tier4/AWML). We welcome your feedback!

## Get Started

<div class="grid cards" markdown>

- :lucide-package: **Installation**

    ---

    Choose your preferred installation method

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

- :lucide-zap: **Quick Start**

    ---

    First interaction with Autoware-ML

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

- :lucide-wrench: **Design**

    ---

    Architecture overview

    [:octicons-arrow-right-24: Design](framework/design.md)

- :lucide-rocket: **Contributing**

    ---

    How to contribute

    [:octicons-arrow-right-24: Contributing](contributing/contribution-overview.md)

</div>

## Key Features

- **Hydra Configuration** - Hierarchical YAML configs with runtime overrides
- **PyTorch Lightning Core** - Scalable training with multi-GPU support
- **Optuna Integration** - Automated hyperparameter optimization
- **MLflow Tracking** - Experiment logging and comparison
- **ONNX & TensorRT Export** - Production deployment with dynamic shapes

## Supported Models

| Task               | Modality      | Model              | NuScenes           | T4 Dataset         | Training           | ONNX               | TensorRT           |
| ------------------ | ------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
| Calibration Status | Camera, LiDAR | ResNet18           | :lucide-check:     | :lucide-check:     | :lucide-check:     | :lucide-check:     | :lucide-check:     |
| Detection 3D       | LiDAR         | CenterPoint        | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: |
| Detection 3D       | LiDAR         | TransFusion        | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: |
| Detection 3D       | Camera, LiDAR | BEVFusion          | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: |
| Detection 3D       | Camera, LiDAR | StreamPETR         | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: |
| Segmentation 3D    | LiDAR         | FRNet              | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: |
| Segmentation 3D    | LiDAR         | PointTransformerV3 | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: | :lucide-hourglass: |

:lucide-check: Available | :lucide-hourglass: Coming Soon

## License

Developed by [TIER IV, Inc.](https://tier4.jp) under the Apache 2.0 License.
