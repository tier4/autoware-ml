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

"""Base class for 3D detection models."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

from autoware_ml.metrics.detection3d import CenterDistanceMeanAP, metrics_to_tensors
from autoware_ml.models.base import BaseModel


class Detection3DBaseModel(BaseModel):
    """Base class for all 3D detection models.

    Extends BaseModel with epoch-level AP metric accumulation using
    Lightning's post-batch hooks. Subclasses must implement forward,
    compute_metrics, and decode_detection_predictions.

    AP is accumulated across all validation or test batches via
    on_validation_batch_end and on_test_batch_end, then logged at epoch end.
    """

    def __init__(
        self, *args: Any, metric: CenterDistanceMeanAP | None = None, **kwargs: Any
    ) -> None:
        """Initialize the detection base model.

        Args:
            metric: AP metric configuration. Defaults to CenterDistanceMeanAP
                with standard settings.
            *args: Positional arguments forwarded to BaseModel.
            **kwargs: Keyword arguments forwarded to BaseModel.
        """
        super().__init__(*args, **kwargs)
        metric = metric or CenterDistanceMeanAP()
        self._detection_maps = {
            "val": deepcopy(metric),
            "test": deepcopy(metric),
        }

    def decode_detection_predictions(self, outputs: Any) -> list[dict[str, torch.Tensor]]:
        """Decode raw model outputs into detection predictions.

        Args:
            outputs: Raw forward outputs from the model.

        Returns:
            List of per-sample prediction dicts with bboxes_3d, scores_3d, labels_3d.
        """
        head = getattr(self, "bbox_head", None) or getattr(self, "pts_bbox_head", None)
        if head is None:
            raise AttributeError("Detection model requires a bbox_head or pts_bbox_head attribute.")
        det_outputs = (
            outputs["det_outputs"]
            if isinstance(outputs, dict) and "det_outputs" in outputs
            else outputs
        )
        return head.predict(det_outputs)

    def on_validation_epoch_start(self) -> None:
        """Reset AP state at the start of each validation epoch."""
        self._detection_maps["val"].reset()

    def on_test_epoch_start(self) -> None:
        """Reset AP state at the start of each test epoch."""
        self._detection_maps["test"].reset()

    def on_validation_batch_end(
        self,
        outputs: dict[str, Any],
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate detection predictions after each validation batch.

        Args:
            outputs: Return value of validation_step, includes model_outputs.
            batch: Preprocessed batch dict with gt_boxes and gt_labels.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
        """
        self._accumulate_detection_map("val", outputs, batch)

    def on_test_batch_end(
        self,
        outputs: dict[str, Any],
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate detection predictions after each test batch.

        Args:
            outputs: Return value of test_step, includes model_outputs.
            batch: Preprocessed batch dict with gt_boxes and gt_labels.
            batch_idx: Batch index.
            dataloader_idx: Dataloader index.
        """
        self._accumulate_detection_map("test", outputs, batch)

    def on_validation_epoch_end(self) -> None:
        """Log AP metrics at the end of each validation epoch."""
        self._log_detection_map("val")

    def on_test_epoch_end(self) -> None:
        """Log AP metrics at the end of each test epoch."""
        self._log_detection_map("test")

    def _accumulate_detection_map(
        self,
        stage: str,
        outputs: dict[str, Any],
        batch: Mapping[str, Any],
    ) -> None:
        self._sync_detection_class_names()
        predictions = self.decode_detection_predictions(outputs["model_outputs"])
        self._detection_maps[stage].update(
            predictions,
            batch["gt_boxes"],
            batch["gt_labels"],
            gt_num_points=batch.get("gt_num_points"),
            gt_attributes=batch.get("gt_attributes"),
        )

    def _sync_detection_class_names(self) -> None:
        head = getattr(self, "bbox_head", None) or getattr(self, "pts_bbox_head", None)
        if head is None:
            return
        class_names = getattr(head, "class_names", None)
        if class_names is None:
            return
        class_names = tuple(str(name) for name in class_names)
        for detection_map in self._detection_maps.values():
            detection_map.class_names = class_names

    def _log_detection_map(self, stage: str) -> None:
        metrics = self._detection_maps[stage].compute()
        self._detection_maps[stage].reset()
        if not metrics:
            return
        metric_tensors = metrics_to_tensors(metrics, self.device)
        self.log_dict(
            {f"{stage}/{name}": value for name, value in metric_tensors.items()},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
