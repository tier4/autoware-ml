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

"""Lightning wrapper for RT-DETRv4 2D detection models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.models.detection2d.base import BaseDetectionModel
from autoware_ml.models.detection2d.rtdetrv4.rtv4 import RTv4
from autoware_ml.utils.checkpoints import load_model_from_raw_checkpoint
from autoware_ml.utils.deploy import ExportSpec

logger = logging.getLogger(__name__)


def _is_norm_or_bias_parameter(name: str, parameter: torch.nn.Parameter) -> bool:
    return name.endswith(".bias") or parameter.ndim == 1 or ".norm" in name or ".bn" in name


class RTDETRv4DetectionModel(BaseDetectionModel):
    """Hydra/Lightning wrapper around vendored RT-DETRv4 modules."""

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        init_checkpoint_path: str | None = None,
        init_checkpoint_state_key: str | None = None,
        init_checkpoint_filter_mismatched_shapes: bool = False,
        init_checkpoint_strict: bool = False,
        compute_coco_metrics_on_val: bool = True,
        compute_localization_metrics_on_val: bool = True,
        compute_coco_metrics_on_test: bool = True,
        compute_localization_metrics_on_test: bool = True,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_group_overrides=optimizer_group_overrides,
            scheduler_config=scheduler_config,
            compute_coco_metrics_on_val=compute_coco_metrics_on_val,
            compute_localization_metrics_on_val=compute_localization_metrics_on_val,
            compute_coco_metrics_on_test=compute_coco_metrics_on_test,
            compute_localization_metrics_on_test=compute_localization_metrics_on_test,
        )
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.detector = RTv4(backbone=backbone, encoder=encoder, decoder=decoder)
        self.criterion = criterion
        self.postprocessor = postprocessor
        if init_checkpoint_path is not None:
            incompatible = load_model_from_raw_checkpoint(
                self.detector,
                Path(init_checkpoint_path),
                state_key=init_checkpoint_state_key,
                strict=init_checkpoint_strict,
                filter_mismatched_shapes=init_checkpoint_filter_mismatched_shapes,
            )
            logger.info(
                "Initialized RT-DETRv4 detector from %s with %d missing keys and %d unexpected keys",
                init_checkpoint_path,
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )

    def forward(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.detector(images, targets=targets)

    def postprocess_predictions(
        self,
        outputs: Any,
        orig_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        return self.postprocessor(outputs, orig_sizes)

    def compute_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = self.criterion(outputs, targets)
        total_loss = sum(value for key, value in losses.items() if key.startswith("loss"))
        return {"loss": total_loss, **losses}

    @contextmanager
    def _criterion_mode(self):
        """Enable training-only decoder outputs without mutating BatchNorm state."""
        if self.training:
            yield
            return

        was_training = self.detector.training
        batch_norms = [module for module in self.detector.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
        batch_norm_states = [module.training for module in batch_norms]

        self.detector.train()
        for module in batch_norms:
            module.eval()

        try:
            yield
        finally:
            self.detector.train(was_training)
            for module, state in zip(batch_norms, batch_norm_states, strict=False):
                module.train(state)

    def compute_step_metrics(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: Any,
    ) -> dict[str, torch.Tensor]:
        loss_outputs = outputs
        if not self.training and "aux_outputs" not in outputs:
            with self._criterion_mode():
                loss_outputs = self.detector(
                    batch_inputs_dict["images"],
                    targets=batch_inputs_dict["targets"],
                )
        return self.compute_metrics(loss_outputs, batch_inputs_dict["targets"])

    def build_optimizer_groups(self) -> Mapping[str, list[torch.nn.Parameter]]:
        groups: OrderedDict[str, list[torch.nn.Parameter]] = OrderedDict(
            {
                "backbone": [],
                "backbone_norm": [],
                "encoder_decoder_norm_bias": [],
                "default": [],
            }
        )
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone.") or name.startswith("detector.backbone."):
                group_name = "backbone_norm" if _is_norm_or_bias_parameter(name, parameter) else "backbone"
            elif _is_norm_or_bias_parameter(name, parameter):
                group_name = "encoder_decoder_norm_bias"
            else:
                group_name = "default"
            groups[group_name].append(parameter)
        return groups

    def get_export_output_names(self) -> list[str]:
        return ["pred_labels", "pred_boxes", "pred_scores"]

    def build_export_spec(self, batch_inputs_dict: Mapping[str, Any]) -> ExportSpec:
        return ExportSpec(
            module=_RTDETRv4ExportModule(self.detector, self.postprocessor),
            args=(batch_inputs_dict["images"], batch_inputs_dict["orig_sizes"]),
            input_param_names=["images", "orig_sizes"],
            output_names=self.get_export_output_names(),
        )


class _RTDETRv4ExportModule(nn.Module):
    """Export wrapper that emits postprocessed predictions."""

    def __init__(self, detector: RTv4, postprocessor: nn.Module) -> None:
        super().__init__()
        self.detector = deepcopy(detector).eval()
        self.postprocessor = deepcopy(postprocessor).eval().deploy()

    def forward(self, images: torch.Tensor, orig_sizes: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.detector(images)
        labels, boxes, scores = self.postprocessor(outputs, orig_sizes)
        return labels, boxes, scores
