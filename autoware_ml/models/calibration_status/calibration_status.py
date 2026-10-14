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

from typing import Any

import torch
import torch.nn as nn

from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.models.calibration_status.head_outputs import (
    CalibrationStatusHeadOutputs,
)
from autoware_ml.dataclasses.models.calibration_status.predictions import (
    CalibrationStatusPredictions,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.utils.deploy import ExportSpec


def _image_data(batch_inputs: ModelBatchInputs) -> ImageGTBatch:
    """Read the images of the batch.

    Raises:
        ValueError: If the batch carries no images.
    """
    if batch_inputs.image_data is None:
        raise ValueError("The calibration status classifier needs the images of the batch.")
    return batch_inputs.image_data


class CalibrationStatusClassifier(ModuleBaseModel):
    """Predict calibration-status labels from fused image inputs.

    The model combines a backbone, neck, and classification head inside the
    shared Autoware-ML training interface.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        **kwargs: Any,
    ) -> None:
        """Initialize the calibration-status classifier.

        Args:
            backbone: Feature extraction backbone for fused camera inputs.
            neck: Intermediate feature aggregation module.
            head: Classification head that computes logits, predictions, and losses.
            **kwargs: Keyword arguments forwarded to :class:`ModuleBaseModel`, such as the
                data preprocessor, the logging configuration, the optimizer and the metrics.
        """
        super().__init__(**kwargs)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the classifier on the fused images of the batch.

        Args:
            multi_task_batch_inputs: Model inputs holding the images and their depth maps.

        Returns:
            Calibration status head outputs with the classification logits of every image.
        """
        return self.forward_network(**self.forward_inputs(multi_task_batch_inputs))

    def forward_network(self, fused_img: torch.Tensor) -> ModelOutputs:
        """Run the classifier on fused image inputs.

        Args:
            fused_img: Batched fused image tensor.

        Returns:
            Calibration status head outputs with the classification logits of every image.
        """
        feats = self.backbone(fused_img)
        feats = self.neck(feats)
        logits = self.head(feats)
        return ModelOutputs(
            calibration_status_head_outputs=CalibrationStatusHeadOutputs(logits=logits)
        )

    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        """Pick the fused image of every camera of the batch.

        Args:
            batch_inputs: Model inputs holding the images and their depth maps.

        Returns:
            The fused images, one per camera of every sample.
        """
        return {"fused_img": _image_data(batch_inputs).fused_images()}

    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Convert logits into class probabilities."""
        del multi_task_batch_inputs
        probabilities = self.head.predict(outputs.calibration_status().logits)
        return ModelPredictions(
            calibration_status_predictions=CalibrationStatusPredictions(probabilities=probabilities)
        )

    def compute_metrics(
        self,
        batch_inputs: ModelBatchInputs,
        outputs: ModelOutputs,
    ) -> dict[str, torch.Tensor]:
        """Compute training losses and metrics for one batch.

        Args:
            batch_inputs: Model inputs holding the calibration status of every camera.
            outputs: Model outputs returned by :meth:`forward`.

        Returns:
            Dictionary of loss terms and logged metrics.

        Raises:
            ValueError: If the batch carries no calibration status.
        """
        calibration_statuses = _image_data(batch_inputs).calibration_statuses
        if calibration_statuses is None:
            raise ValueError("Calibration status losses need the status of every camera.")
        return self.head.loss(outputs.calibration_status().logits, calibration_statuses.flatten())

    def build_export_spec(self, batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build a calibration-status-specific export specification.

        The generic Lightning prediction wrapper uses a variadic ``forward(*args)``
        interface around a LightningModule. PyTorch's dynamo ONNX path currently
        fails to decompose that wrapper for this model. Exporting through a plain
        ``nn.Module`` with a concrete ``forward(fused_img)`` signature avoids the
        issue while preserving the original probability-only export contract.

        Args:
            batch_inputs: Example preprocessed batch used for export.

        Returns:
            Export specification for deployment.
        """
        return ExportSpec(
            module=_CalibrationStatusExportModule(
                backbone=self.backbone,
                neck=self.neck,
                head=self.head,
            ),
            args=(self.forward_inputs(batch_inputs)["fused_img"],),
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
