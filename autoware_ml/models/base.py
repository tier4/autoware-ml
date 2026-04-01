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

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import lightning as L
from lightning.pytorch.utilities.data import extract_batch_size
import torch
import torch.nn as nn
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

    def predict_outputs(self, outputs: Any) -> Any:
        """Convert raw model outputs into task-level predictions.

        The default implementation returns the model outputs unchanged. Task
        wrappers should override this when prediction-time outputs differ from
        training-time outputs, for example to convert logits into probabilities
        and labels.

        Args:
            outputs: Raw outputs returned by :meth:`run_model`.

        Returns:
            Task-level predictions.
        """
        return outputs

    @torch.no_grad()
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference and return task-level predictions.

        Args:
            *args: Positional arguments forwarded to :meth:`forward`.
            **kwargs: Keyword arguments forwarded to :meth:`forward`.

        Returns:
            Task-level predictions produced by :meth:`predict_outputs`.
        """
        return self.predict_outputs(self(*args, **kwargs))

    def get_export_output_names(self) -> list[str] | None:
        """Return output names used by the generic export wrapper.

        Models that export structured prediction dictionaries should override
        this hook and return the tensor names in the exported output order.

        Returns:
            Export output names or ``None`` when the generic wrapper should keep
            the model outputs unnamed.
        """
        return None

    def prepare_export_outputs(self, predictions: Any) -> Any:
        """Convert prediction outputs into an ONNX-exportable structure.

        Args:
            predictions: Task-level predictions produced by
                :meth:`predict_outputs`.

        Returns:
            Tensor, tuple of tensors, or another ONNX-exportable structure.

        Raises:
            ValueError: Raised when prediction outputs are a mapping but the
                model does not define explicit export output names.
        """
        if isinstance(predictions, Mapping):
            output_names = self.get_export_output_names()
            if output_names is None:
                raise ValueError(
                    "Structured prediction outputs require explicit export output names."
                )
            return tuple(predictions[name] for name in output_names)
        return predictions

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Return structural optimizer groups for the model.

        Models that do not need custom grouping use a single ``default`` group.
        Models with optimizer-group-specific tuning can override this hook.
        """
        return {
            "default": [parameter for parameter in self.parameters() if parameter.requires_grad]
        }

    @abstractmethod
    def forward(self, **kwargs: Any) -> Any:
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
    def compute_metrics(self, outputs: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
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

    def prepare_forward_inputs(self, batch_inputs_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Build the keyword arguments passed to ``forward``.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.

        Returns:
            Keyword arguments forwarded to :meth:`forward`.
        """
        return self._filter_batch_keys(batch_inputs_dict, self.forward_signature)

    def run_model(self, batch_inputs_dict: Mapping[str, Any]) -> Any:
        """Run the model on one batch.

        Models with standard tensor-only forwards can rely on the default
        implementation. Models that need custom batch unpacking can override
        this hook without changing the public ``forward`` signature.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.

        Returns:
            Model outputs.
        """
        return self(**self.prepare_forward_inputs(batch_inputs_dict))

    def prepare_metric_inputs(self, batch_inputs_dict: Mapping[str, Any]) -> dict[str, Any]:
        """Build the keyword arguments passed to ``compute_metrics``.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.

        Returns:
            Keyword arguments forwarded to :meth:`compute_metrics`.
        """
        return self._filter_batch_keys(batch_inputs_dict, self.compute_metrics_signature)

    def compute_step_metrics(
        self,
        batch_inputs_dict: Mapping[str, Any],
        outputs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute step metrics from model outputs and batch data.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.
            outputs: Outputs returned from :meth:`run_model`.

        Returns:
            Metric dictionary produced by :meth:`compute_metrics`.
        """
        return self.compute_metrics(
            outputs=outputs,
            **self.prepare_metric_inputs(batch_inputs_dict),
        )

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, Any]) -> int | None:
        """Infer the effective sample batch size for logging.

        The default implementation tries Lightning's recursive batch-size
        inference on the actual model inputs. Models with ragged point-cloud
        batches should override this hook to provide an explicit sample count.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.

        Returns:
            Sample batch size when it can be inferred, otherwise ``None``.
        """
        try:
            return extract_batch_size(self.prepare_forward_inputs(batch_inputs_dict))
        except Exception:
            return None

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
        outputs = self.run_model(batch_inputs_dict)
        metrics = self.compute_step_metrics(batch_inputs_dict, outputs)
        if "loss" not in metrics:
            raise ValueError("compute_metrics() must return a dict containing a 'loss' key.")
        batch_size = self.get_log_batch_size(batch_inputs_dict)
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
            Metric dictionary logged for the validation step.
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
            Metric dictionary logged for the test step.
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
        del batch_idx
        return self.predict_outputs(self.run_model(batch_inputs_dict))

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
        raw_spec = infer_export_spec(self, batch_inputs_dict)
        return ExportSpec(
            module=_PredictionExportWrapper(self),
            args=raw_spec.args,
            input_param_names=raw_spec.input_param_names,
            output_names=self.get_export_output_names(),
            supported_stages=raw_spec.supported_stages,
        )

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


class _PredictionExportWrapper(nn.Module):
    """Wrap a model so generic export emits task-level predictions."""

    def __init__(self, model: BaseModel) -> None:
        """Initialize the generic export wrapper.

        Args:
            model: Model instance whose forward and prediction hooks are used
                during export.
        """
        super().__init__()
        self.model = model

    def forward(self, *args: Any) -> Any:
        """Run the wrapped model and convert raw outputs into export outputs."""
        outputs = self.model(*args)
        predictions = self.model.predict_outputs(outputs)
        return self.model.prepare_export_outputs(predictions)
