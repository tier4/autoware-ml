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

"""CenterPoint 3D Object Detection Model.

Clean PyTorch Lightning implementation following Autoware-ML design patterns.
"""

from typing import Any, Dict, Sequence, Union

import torch
import torch.nn as nn

from autoware_ml.models.base import BaseModel


class CalibrationStatusClassifier(BaseModel):
    """Calibration status classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        **kwargs: Any,
    ):
        """Initialize calibration status classifier.

        Args:
            backbone: Backbone module.
            neck: Neck module.
            head: Head module.
            **kwargs: Keyword arguments.
        """
        super().__init__(**kwargs)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, fused_img: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            fused_img: Fused image tensor.

        Returns:
            Logits from the classification head.
        """
        feats = self.backbone(fused_img)
        feats = self.neck(feats)
        logits = self.head(feats)
        return logits

    @torch.no_grad()
    def predict(self, fused_img: torch.Tensor) -> torch.Tensor:
        """Predict.

        Args:
            fused_img: Fused image tensor.

        Returns:
            Predictions.
        """
        logits = self(fused_img)
        return self.head.predict(logits)

    def compute_metrics(
        self,
        outputs: Union[torch.Tensor, Sequence[torch.Tensor]],
        gt_calibration_status: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss.

        Args:
            outputs: Model logits from forward().
            gt_calibration_status: Ground truth calibration status tensor.

        Returns:
            Dictionary of losses as tensors.
        """
        losses = self.head.loss(outputs, gt_calibration_status)
        return losses
