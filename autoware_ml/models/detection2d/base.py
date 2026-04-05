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

"""Shared Lightning base class for 2D detection models."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed

from autoware_ml.metrics.detection2d import (
    evaluate_class_agnostic_localization,
    evaluate_coco_predictions,
)
from autoware_ml.models.base import BaseModel


def _gather_picklable(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gather Python objects across distributed ranks."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return data

    gathered: list[list[dict[str, Any]] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, data)
    merged: list[dict[str, Any]] = []
    for shard in gathered:
        if shard is not None:
            merged.extend(shard)
    return merged


class BaseDetectionModel(BaseModel):
    """Shared base model for COCO-style 2D object detection tasks."""

    def __init__(
        self,
        *args: Any,
        compute_coco_metrics_on_val: bool = True,
        compute_localization_metrics_on_val: bool = True,
        compute_coco_metrics_on_test: bool = True,
        compute_localization_metrics_on_test: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.compute_coco_metrics_on_val = compute_coco_metrics_on_val
        self.compute_localization_metrics_on_val = compute_localization_metrics_on_val
        self.compute_coco_metrics_on_test = compute_coco_metrics_on_test
        self.compute_localization_metrics_on_test = compute_localization_metrics_on_test
        self._val_predictions: list[dict[str, Any]] = []
        self._test_predictions: list[dict[str, Any]] = []

    @abstractmethod
    def postprocess_predictions(
        self,
        outputs: Any,
        orig_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert raw outputs into postprocessed predictions."""

    def predict_outputs(self, outputs: Any) -> Any:
        """Return raw model outputs when prediction lacks image-size context."""
        return outputs

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, Any]) -> int | None:
        targets = batch_inputs_dict.get("targets")
        if isinstance(targets, Sequence):
            return len(targets)
        return super().get_log_batch_size(batch_inputs_dict)

    def _shared_eval_step(
        self,
        batch_inputs_dict: Mapping[str, Any],
        step_prefix: str,
        prediction_store: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        outputs = self.run_model(batch_inputs_dict)
        metrics = self.compute_step_metrics(batch_inputs_dict, outputs)
        if "loss" not in metrics:
            raise ValueError("compute_metrics() must return a dict containing a 'loss' key.")

        self.log_dict(
            {f"{step_prefix}/{k}": v for k, v in metrics.items()},
            batch_size=self.get_log_batch_size(batch_inputs_dict),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if not getattr(self.trainer, "sanity_checking", False):
            predictions = self.postprocess_predictions(outputs, batch_inputs_dict["orig_sizes"])
            image_ids = batch_inputs_dict["image_ids"]
            for image_id, prediction in zip(image_ids, predictions, strict=False):
                prediction_store.append(
                    {
                        "image_id": int(image_id.item() if isinstance(image_id, torch.Tensor) else image_id),
                        "boxes": prediction["boxes"].detach().cpu(),
                        "scores": prediction["scores"].detach().cpu(),
                        "labels": prediction["labels"].detach().cpu(),
                    }
                )

        return metrics

    def on_validation_epoch_start(self) -> None:
        self._val_predictions = []

    def on_test_epoch_start(self) -> None:
        self._test_predictions = []

    def validation_step(
        self,
        batch_inputs_dict: Mapping[str, Any],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        return self._shared_eval_step(batch_inputs_dict, "val", self._val_predictions)

    def test_step(
        self,
        batch_inputs_dict: Mapping[str, Any],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        return self._shared_eval_step(batch_inputs_dict, "test", self._test_predictions)

    def _log_dataset_metrics(self, stage: str, prediction_store: list[dict[str, Any]]) -> None:
        if not prediction_store:
            return

        if stage == "val":
            compute_coco_metrics = self.compute_coco_metrics_on_val
            compute_localization_metrics = self.compute_localization_metrics_on_val
        elif stage == "test":
            compute_coco_metrics = self.compute_coco_metrics_on_test
            compute_localization_metrics = self.compute_localization_metrics_on_test
        else:
            compute_coco_metrics = True
            compute_localization_metrics = True

        if not compute_coco_metrics and not compute_localization_metrics:
            return

        datamodule = self.trainer.datamodule
        dataset = getattr(datamodule, f"{stage}_dataset")
        if dataset is None or not hasattr(dataset, "get_coco_api"):
            return

        predictions = _gather_picklable(prediction_store)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.global_rank != 0:
            return

        coco_gt = dataset.get_coco_api()
        metrics: dict[str, float] = {}
        if compute_coco_metrics:
            metrics.update(
                evaluate_coco_predictions(
                    coco_gt=coco_gt,
                    predictions=predictions,
                    label_to_category_id=dataset.label_to_category_id,
                )
            )
        if compute_localization_metrics:
            metrics.update(
                evaluate_class_agnostic_localization(
                    coco_gt=coco_gt,
                    predictions=predictions,
                )
            )
        if not metrics:
            return
        self.log_dict(
            {f"{stage}/{key}": value for key, value in metrics.items()},
            sync_dist=False,
            rank_zero_only=True,
        )

    def on_validation_epoch_end(self) -> None:
        self._log_dataset_metrics("val", self._val_predictions)

    def on_test_epoch_end(self) -> None:
        self._log_dataset_metrics("test", self._test_predictions)

    def predict_step(self, batch_inputs_dict: Mapping[str, Any], batch_idx: int) -> Any:
        del batch_idx
        outputs = self.run_model(batch_inputs_dict)
        return self.postprocess_predictions(outputs, batch_inputs_dict["orig_sizes"])
