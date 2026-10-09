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

"""Base model classes for Autoware-ML.

This module defines shared Lightning model interfaces and helper abstractions
used by task-specific model wrappers throughout the framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, final

import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.eval_mixin import MetricEvalMixin
from autoware_ml.preprocessing.base import DataPreprocessing
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.optimizer import build_lightning_optimizer_config


class BaseModel(MetricEvalMixin, L.LightningModule, ABC):
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
        metrics: Sequence[MetricSuite] | None = None,
    ):
        """Initialize base model.

        Args:
            optimizer: Callable that returns an optimizer when given model parameters.
            scheduler: Callable that returns a scheduler when given the optimizer.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
            metrics: Task metrics accumulated during validation and test. Empty
                or ``None`` logs only losses.
        """
        super().__init__(metrics=metrics)
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.optimizer_group_overrides = (
            dict(optimizer_group_overrides) if optimizer_group_overrides else None
        )
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {}
        self._data_preprocessing = DataPreprocessing()

    def set_data_preprocessing(self, data_preprocessing: DataPreprocessing) -> None:
        """Install the runtime preprocessing pipeline.

        Runtime entrypoints attach the preprocessing pipeline from top-level
        config after model construction. The neural network remains
        implemented by ``forward``; this Lightning wrapper owns the batch
        execution lifecycle.

        Args:
            data_preprocessing: Pipeline applied after batch transfer and
                before model forward.
        """
        self._data_preprocessing = data_preprocessing

    def transfer_batch_to_device(
        self, batch: ModelGTBatch, device: torch.device, dataloader_idx: int
    ) -> ModelGTBatch:
        """Move the typed batch to the device Lightning runs the step on.

        Args:
            batch: Collated typed batch from the dataloader.
            device: Target device.
            dataloader_idx: Lightning dataloader index.

        Returns:
            The batch on the target device.
        """
        del dataloader_idx
        return batch.to_device(device)

    def on_after_batch_transfer(self, batch: ModelGTBatch, dataloader_idx: int) -> ModelBatchInputs:
        """Apply runtime preprocessing after Lightning moves a batch to device.

        Args:
            batch: Collated typed batch on the target device.
            dataloader_idx: Lightning dataloader index.

        Returns:
            Model inputs after runtime preprocessing.
        """
        del dataloader_idx
        return self._data_preprocessing(batch, is_training=self.training)

    def predict_outputs(self, batch_inputs: ModelBatchInputs, outputs: Any) -> Any:
        """Convert raw model outputs into task-level predictions.

        The default implementation returns the model outputs unchanged. Task
        wrappers should override this when prediction-time outputs differ from
        training-time outputs, for example to convert logits into probabilities
        and labels.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            Task-level predictions.
        """
        del batch_inputs
        return outputs

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Return structural optimizer groups for the model.

        Models that do not need custom grouping use a single ``default`` group.
        Models with optimizer-group-specific tuning can override this hook.

        Returns:
            Mapping from optimizer group names to parameter sequences.
        """
        return {
            "default": [parameter for parameter in self.parameters() if parameter.requires_grad]
        }

    @abstractmethod
    def forward(self, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Subclasses define this method with the tensor arguments the network reads, the same
        arguments the deployment export traces.

        Args:
            **kwargs: Keyword arguments (subclass-specific).

        Returns:
            Model outputs.
        """
        pass

    @abstractmethod
    def forward_inputs(self, batch_inputs: ModelBatchInputs) -> dict[str, Any]:
        """Pick the arguments of :meth:`forward` from the model inputs.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.

        Returns:
            The keyword arguments of :meth:`forward`.
        """
        pass

    @abstractmethod
    def compute_metrics(
        self, batch_inputs: ModelBatchInputs, outputs: Any
    ) -> dict[str, torch.Tensor]:
        """Compute metrics.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.
            outputs: Model outputs from forward().

        Returns:
            Dictionary of metric tensors. A ``"loss"`` key is required.
        """
        pass

    def get_log_batch_size(self, batch_inputs: ModelBatchInputs) -> int:
        """Give the sample batch size the step metrics are logged with.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.

        Returns:
            Number of samples of the batch.
        """
        return batch_inputs.batch_size()

    def _shared_step(
        self, batch_inputs: ModelBatchInputs, step_prefix: str, **kwargs: Any
    ) -> tuple[dict[str, torch.Tensor], Any]:
        """Run one forward pass, compute metrics, and log them.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.
            step_prefix: Prefix for logging (train, val, test).
            **kwargs: Keyword arguments forwarded to ``self.log_dict``.

        Returns:
            Tuple of the metric dictionary and the raw model outputs.
            The metric dictionary contains at least a ``"loss"`` key.
        """
        outputs = self(**self.forward_inputs(batch_inputs))
        metrics = self.compute_metrics(batch_inputs, outputs)
        if "loss" not in metrics:
            raise ValueError("compute_metrics() must return a dict containing a 'loss' key.")
        batch_size = self.get_log_batch_size(batch_inputs)
        self.log_dict(
            {f"{step_prefix}/{k}": v for k, v in metrics.items()},
            batch_size=batch_size,
            **kwargs,
        )
        return metrics, outputs

    @final
    def training_step(self, batch_inputs: ModelBatchInputs, batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.
            batch_idx: Batch index.

        Returns:
            Total loss tensor required by Lightning for backpropagation.
        """
        metrics, _ = self._shared_step(
            batch_inputs,
            "train",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return metrics["loss"]

    @final
    def validation_step(self, batch_inputs: ModelBatchInputs, batch_idx: int) -> dict[str, Any]:
        """Validation step.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.
            batch_idx: Batch index.

        Returns:
            Dictionary with at least a ``"loss"`` key and a ``"model_outputs"``
            key containing the raw forward outputs. The raw outputs are available
            to ``on_validation_batch_end`` for epoch-level metric accumulation.
        """
        metrics, outputs = self._shared_step(
            batch_inputs,
            "val",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {**metrics, "model_outputs": outputs}

    @final
    def test_step(self, batch_inputs: ModelBatchInputs, batch_idx: int) -> dict[str, Any]:
        """Test step.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.
            batch_idx: Batch index.

        Returns:
            Dictionary with at least a ``"loss"`` key and a ``"model_outputs"``
            key containing the raw forward outputs.
        """
        metrics, outputs = self._shared_step(
            batch_inputs,
            "test",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {**metrics, "model_outputs": outputs}

    @final
    def predict_step(self, batch_inputs: ModelBatchInputs, batch_idx: int) -> Any:
        """Prediction step.

        Args:
            batch_inputs: Model inputs after runtime preprocessing.
            batch_idx: Batch index.

        Returns:
            Predictions.
        """
        del batch_idx
        outputs = self(**self.forward_inputs(batch_inputs))
        return self.predict_outputs(batch_inputs, outputs)

    def build_export_spec(self, batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build the deployment export specification of the model.

        Args:
            batch_inputs: Example model inputs used for export.

        Returns:
            Export specification for deployment.

        Raises:
            NotImplementedError: If the model defines no deployment export.
        """
        raise NotImplementedError(f"{type(self).__name__} defines no deployment export.")

    def build_export_specs(self, batch_inputs: ModelBatchInputs) -> dict[str, ExportSpec]:
        """Build per-module deployment export specifications.

        The default implementation wraps :meth:`build_export_spec` as a single
        ``end_to_end`` module. Models with separate exportable sub-graphs
        override this to return one spec per architectural component.

        Args:
            batch_inputs: Example model inputs used for export.

        Returns:
            Ordered mapping of module name to export specification.
        """
        return {"end_to_end": self.build_export_spec(batch_inputs)}

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
