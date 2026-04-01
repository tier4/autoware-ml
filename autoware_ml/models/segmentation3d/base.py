# Copyright 2026 TIER IV, Inc.
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

"""Base class for 3D semantic segmentation models.

This module provides a shared intermediate base between :class:`BaseModel` and
concrete segmentation architectures like FRNet and PTv3. It centralizes
prediction conversion, export output naming, and common segmentation metric
helpers so subclasses only implement model-specific logic.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from autoware_ml.metrics.segmentation3d import compute_segmentation_metrics
from autoware_ml.models.base import BaseModel


class BaseSegmentationModel(BaseModel):
    """Shared base for point-wise 3D semantic segmentation models.

    Subclasses MUST set ``self.num_classes`` and ``self.ignore_index`` in their
    ``__init__`` and implement a lightweight hook that tells the base class how
    to extract logits from the model outputs.

    In return the base class provides:

    * ``predict_outputs`` - softmax + argmax -> ``{"pred_labels", "pred_probs"}``
    * ``get_export_output_names`` - driven by the ``EXPORT_OUTPUT_NAMES`` class
      attribute (default ``("pred_labels", "pred_probs")``)
    """

    EXPORT_OUTPUT_NAMES: tuple[str, ...] = ("pred_labels", "pred_probs")

    num_classes: int
    ignore_index: int

    @abstractmethod
    def _get_point_logits(self, outputs: Any) -> torch.Tensor:
        """Extract point-wise logits from the value returned by ``run_model``.

        Args:
            outputs: Raw model outputs.

        Returns:
            Point-wise class logits with shape ``(N, C)``.
        """

    def predict_outputs(self, outputs: Any) -> dict[str, torch.Tensor]:
        """Convert segmentation logits into labels and probabilities."""
        point_logits = self._get_point_logits(outputs)
        pred_probs = torch.softmax(point_logits, dim=1)
        pred_labels = pred_probs.argmax(dim=1)
        return {"pred_labels": pred_labels, "pred_probs": pred_probs}

    def get_export_output_names(self) -> list[str]:
        """Return deployment output names from the class attribute."""
        return list(self.EXPORT_OUTPUT_NAMES)

    def _compute_segmentation_metrics(
        self,
        outputs: Any,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute point-wise segmentation metrics from logits and targets."""
        point_logits = self._get_point_logits(outputs)
        predictions = point_logits.argmax(dim=1)
        metrics = compute_segmentation_metrics(
            predictions,
            targets,
            self.num_classes,
            self.ignore_index,
        )
        return metrics or {}
