# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Calibration status classification model wrappers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.models.base import BaseModel
from autoware_ml.utils.deploy import ExportSpec


class CalibrationStatusClassifier(BaseModel):
    """Predict calibration-status labels from fused image inputs.

    The model combines a backbone, neck, and classification head inside the
    shared Autoware-ML training interface.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the calibration-status classifier.

        Args:
            backbone: Feature extraction backbone for fused camera inputs.
            neck: Intermediate feature aggregation module.
            head: Classification head that computes logits, predictions, and losses.
            optimizer: Optimizer factory forwarded to :class:`BaseModel`.
            scheduler: Scheduler factory forwarded to :class:`BaseModel`.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
        """
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
        )

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, fused_img: torch.Tensor) -> torch.Tensor:
        """Run the classifier on fused image inputs.

        Args:
            fused_img: Batched fused image tensor.

        Returns:
            Classification logits for each sample.
        """
        feats = self.backbone(fused_img)
        feats = self.neck(feats)
        logits = self.head(feats)
        return logits

    def predict_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert logits into class probabilities."""
        return self.head.predict(outputs)

    def compute_metrics(
        self,
        outputs: torch.Tensor | Sequence[torch.Tensor],
        gt_calibration_status: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute training losses and metrics for one batch.

        Args:
            outputs: Model outputs returned by :meth:`forward`.
            gt_calibration_status: Ground-truth calibration-status labels.

        Returns:
            Dictionary of loss terms and logged metrics.
        """
        return self.head.loss(outputs, gt_calibration_status)

    def build_export_spec(self, batch_inputs_dict: Mapping[str, Any]) -> ExportSpec:
        """Build a calibration-status-specific export specification.

        The generic BaseModel prediction wrapper uses a variadic ``forward(*args)``
        interface around a LightningModule. PyTorch's dynamo ONNX path currently
        fails to decompose that wrapper for this model. Exporting through a plain
        ``nn.Module`` with a concrete ``forward(fused_img)`` signature avoids the
        issue while preserving the original probability-only export contract.

        Args:
            batch_inputs_dict: Example preprocessed batch used for export.

        Returns:
            Export specification for deployment.
        """
        return ExportSpec(
            module=_CalibrationStatusExportModule(
                backbone=self.backbone,
                neck=self.neck,
                head=self.head,
            ),
            args=(batch_inputs_dict["fused_img"],),
            input_param_names=["fused_img"],
        )


class _CalibrationStatusExportModule(nn.Module):
    """Plain export wrapper for calibration-status deployment."""

    def __init__(self, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        """Initialize the export wrapper from model submodules."""
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, fused_img: torch.Tensor) -> torch.Tensor:
        """Export the calibration-status model as a single probability tensor."""
        feats = self.backbone(fused_img)
        feats = self.neck(feats)
        logits = self.head(feats)
        return self.head.predict(logits)
