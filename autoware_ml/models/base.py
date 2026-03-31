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

"""Base model classes for Autoware-ML.

This module defines shared Lightning model interfaces and helper abstractions
used by task-specific model wrappers throughout the framework.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.utils.deploy import ExportSpec, infer_export_spec
from autoware_ml.utils.optimizer import build_lightning_optimizer_config


class BaseModel(L.LightningModule, ABC):
    """Base Lightning Module for all Autoware-ML models.

    Provides common functionality for training, validation, and testing with
    built-in support for flexible optimizer and scheduler configuration.
    All parameters are explicitly typed for IDE support and type checking.
    """

    def __init__(
        self,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
    ):
        """Initialize base model.

        Args:
            optimizer: Callable that returns an optimizer when given model parameters.
            scheduler: Callable that returns a scheduler when given the optimizer.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
        """
        super().__init__()
        self.forward_signature = inspect.signature(self.forward)
        self.compute_metrics_signature = inspect.signature(self.compute_metrics)
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.optimizer_group_overrides = (
            dict(optimizer_group_overrides) if optimizer_group_overrides else None
        )
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {}

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Return structural optimizer groups for the model.

        Models that do not need custom grouping use a single ``default`` group.
        Models with optimizer-group-specific tuning can override this hook.
        """
        return {
            "default": [parameter for parameter in self.parameters() if parameter.requires_grad]
        }

    @abstractmethod
    def forward(self, **kwargs: Any) -> torch.Tensor | Sequence[torch.Tensor]:
        """Forward pass of the model.

        Subclasses can define this method with any signature. The base class
        automatically filters batch inputs to match the method signature using
        signature inspection.

        Args:
            **kwargs: Keyword arguments (subclass-specific).

        Returns:
            Model outputs.
        """
        pass

    @abstractmethod
    def compute_metrics(
        self, outputs: torch.Tensor | Sequence[torch.Tensor], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Compute metrics.

        Args:
            outputs: Model outputs from forward().
            **kwargs: Keyword arguments.

        Returns:
            Dictionary of metrics as tensors with, 'loss' key is required.
        """
        pass

    def _filter_batch_keys(
        self, batch_inputs_dict: Mapping[str, Any], signature: inspect.Signature
    ) -> dict[str, Any]:
        """Filter batch keys to match a method signature.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.
            signature: Target method signature whose parameter names are used
                as the key filter.

        Returns:
            Subset of the batch dictionary containing only matching keys.
        """
        return {k: batch_inputs_dict[k] for k in signature.parameters if k in batch_inputs_dict}

    def _shared_step(
        self, batch_inputs_dict: Mapping[str, Any], step_prefix: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Shared step for training, validation, and test steps.

        Args:
            batch_inputs_dict: Dictionary with input data.
            step_prefix: Prefix for the set (train, val, test).
            **kwargs: Keyword arguments forwarded to ``self.log_dict``.

        Returns:
            Dictionary with metrics.
        """
        outputs = self(**self._filter_batch_keys(batch_inputs_dict, self.forward_signature))
        metrics = self.compute_metrics(
            outputs=outputs,
            **self._filter_batch_keys(batch_inputs_dict, self.compute_metrics_signature),
        )
        if "loss" not in metrics:
            raise ValueError("compute_metrics() must return a dict containing a 'loss' key.")
        batch_size = len(next(iter(batch_inputs_dict.values())))
        self.log_dict(
            {f"{step_prefix}/{k}": v for k, v in metrics.items()},
            batch_size=batch_size,
            **kwargs,
        )

        return metrics

    def training_step(self, batch_inputs_dict: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Total loss tensor (required by Lightning for back propagation).
        """
        metrics = self._shared_step(
            batch_inputs_dict,
            "train",
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return metrics["loss"]

    def validation_step(
        self, batch_inputs_dict: Mapping[str, Any], batch_idx: int
    ) -> dict[str, Any]:
        """Validation step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Dictionary with outputs and batch for optional aggregation.
        """
        return self._shared_step(
            batch_inputs_dict,
            "val",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def test_step(self, batch_inputs_dict: Mapping[str, Any], batch_idx: int) -> dict[str, Any]:
        """Test step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Test outputs.
        """
        return self._shared_step(
            batch_inputs_dict,
            "test",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def predict_step(self, batch_inputs_dict: Mapping[str, Any], batch_idx: int) -> Any:
        """Prediction step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Predictions.
        """
        return self(**self._filter_batch_keys(batch_inputs_dict, self.forward_signature))

    def build_export_spec(self, batch_inputs_dict: Mapping[str, Any]) -> ExportSpec:
        """Build the default deployment export specification for the model.

        Models with tensor-only forwards can rely on this generic
        forward-signature-based implementation. Models that need deployment
        wrappers or export-specific input flattening should override it.

        Args:
            batch_inputs_dict: Example preprocessed batch used for export.

        Returns:
            Export specification for deployment.
        """
        return infer_export_spec(self, batch_inputs_dict)

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        """Configure optimizers and schedulers.

        Scheduler behavior such as ``interval``, ``frequency``, and ``monitor``
        is configured explicitly through ``scheduler_config``. The framework
        only auto-fills ``total_steps`` when the configured scheduler declares
        that argument and it was not already bound in the scheduler factory.

        Returns:
            Optimizer instance or Lightning optimizer configuration dictionary.
        """
        if self.optimizer_partial is None:
            raise ValueError("Optimizer must be provided.")
        return build_lightning_optimizer_config(
            self,
            self.optimizer_partial,
            self.scheduler_partial,
            optimizer_group_overrides=self.optimizer_group_overrides,
            scheduler_config=self.scheduler_config,
            estimated_stepping_batches=self.trainer.estimated_stepping_batches
            if self._trainer is not None
            else None,
        )
