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
from typing import Any, NamedTuple, final
from types import MappingProxyType

from jaxtyping import Float32
import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.eval_mixin import MetricEvalMixin
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.types.dataset import SplitType
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.optimizer import build_lightning_optimizer_config


class LogDictConfigs(NamedTuple):
    """Configuration for logging metrics in Lightning.

    The batch size is not configurable: the base model infers it from the ground-truth batch so
    the epoch means are weighted by the real number of samples.

    Attributes:
        prog_bar: Whether to log to the progress bar.
        logger: Whether to log to the logger.
        on_step: Whether to also log on each training step. Validation and test always log per
            epoch only, so callbacks that monitor ``val/<key>`` keep their key. With ``True``
            Lightning logs ``train/<key>_step`` and ``train/<key>_epoch`` instead of
            ``train/<key>``, and with ``sync_dist`` it reduces every key across ranks on every
            step.
        on_epoch: Whether to log on each epoch.
        reduce_fx: Reduction function to apply to the logged metric.
        sync_dist: Whether to synchronize across distributed processes.
        rank_zero_only: Whether to log only on rank zero in a distributed environment.
    """

    prog_bar: bool
    logger: bool | None = None
    on_step: bool | None = None
    on_epoch: bool | None = None
    reduce_fx: str = "mean"
    sync_dist: bool = False
    rank_zero_only: bool = False


class ModuleBaseModel(MetricEvalMixin, L.LightningModule, ABC):
    """Base Lightning Module for all Autoware-ML models.

    Provides common functionality for training, validation, and testing with
    built-in support for flexible optimizer and scheduler configuration.
    All parameters are explicitly typed for IDE support and type checking.
    """

    def __init__(
        self,
        log_dict_configs: LogDictConfigs,
        data_preprocessor: DataPreprocessor | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ):
        """Initialize base model.

        Args:
            log_dict_configs: Configuration for logging metrics in Lightning.
            data_preprocessor: Data preprocessor to preprocess input batches, an empty one when
                omitted.
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
        self._data_preprocessor = (
            data_preprocessor
            if data_preprocessor is not None
            else DataPreprocessor(preprocessor_modules=[])
        )
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.optimizer_group_overrides = (
            dict(optimizer_group_overrides) if optimizer_group_overrides else None
        )
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {}
        self.log_dict_configs = log_dict_configs

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
            batch: Collated batch of type :class:`ModelGTBatch` on the target device.
            dataloader_idx: Lightning dataloader index.

        Returns:
            Batch of type :class:`ModelBatchInputs` after runtime preprocessing.
        """
        return self._data_preprocessor(batch, is_training=self.training)

    @abstractmethod
    def decode_outputs(
        self, multi_task_batch_inputs: ModelBatchInputs, outputs: ModelOutputs
    ) -> ModelPredictions:
        """Convert raw model outputs into task-level predictions.

        Prediction-time outputs differ from training-time outputs, for example logits are
        converted into probabilities and labels.

        Args:
            multi_task_batch_inputs: Batch the outputs were computed from, for predictions
                that map back onto the inputs such as per-point labels.
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            Task-level predictions.
        """

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
    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Forward pass of the model.

        Subclasses must follow the signature of this method and return a dataclass containing
        all model outputs. Users must add new implementations to ModelBatchInputs and ModelOutputs
        when supporting new types of tasks to the multi-task model.

        Args:
            multi_task_batch_inputs: Batch of type :class:`ModelBatchInputs` containing
                multi-task inputs.

        Returns:
            Model outputs.
        """

    @abstractmethod
    def compute_metrics(
        self, multi_task_batch_inputs: ModelBatchInputs, multi_task_outputs: ModelOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " num_losses"]]:
        """Compute metrics.

        Args:
            multi_task_batch_inputs: Full batch dictionary after runtime preprocessing.
            multi_task_outputs: Model outputs from forward().

        Returns:
            MappingProxyType[str, Float32[torch.Tensor, " num_losses"]]: Dictionary of metrics.
            The ``loss`` field is required. Note that users must register metrics in
            the dictionary for proper logging and checkpointing.
        """

    def get_log_batch_size(self, multi_task_batch_inputs: ModelBatchInputs) -> int | None:
        """Infer the effective sample batch size for logging.

        The ground-truth batch infers the sample count from the inputs it carries.
        Models with non-standard input structures should override this hook to provide an
        explicit sample count.

        Args:
            multi_task_batch_inputs: Full batch dictionary from the dataloader.

        Returns:
            Sample batch size when it can be inferred, otherwise ``None``.
        """
        return multi_task_batch_inputs.multi_task_gt_batch.infer_batch_size()

    def _core_step(
        self,
        multi_task_batch_inputs: ModelBatchInputs,
        step_prefix: str,
    ) -> tuple[MappingProxyType[str, Float32[torch.Tensor, " 1"]], ModelOutputs]:
        """
        Core step shared by training, validation, and test. It runs one forward pass,
        computes metrics, and logs them.

        Args:
            multi_task_batch_inputs: Full batch dictionary from the dataloader.
            step_prefix: Prefix for logging (train, val, test).

        Returns:
            Tuple of the metric dictionary (as a ``MappingProxyType``) and the raw model outputs.
            The metric dictionary contains at least a ``"loss"`` key.
        """
        outputs = self(multi_task_batch_inputs)
        metrics = self.compute_metrics(multi_task_batch_inputs, outputs)
        if "loss" not in metrics:
            raise ValueError("compute_metrics() must return a dict containing a 'loss' key.")
        batch_size = self.get_log_batch_size(multi_task_batch_inputs)
        # Step-level logging is training-only, see LogDictConfigs.on_step. Enabling it for
        # val/test would make Lightning suffix the keys (``val/loss_step``/``val/loss_epoch``)
        # and break callbacks that monitor ``val/loss``.
        on_step = self.log_dict_configs.on_step if step_prefix == SplitType.TRAIN else False
        logged_values: dict[str, Any] = {f"{step_prefix}/{k}": v for k, v in metrics.items()}
        self.log_dict(
            logged_values,
            prog_bar=self.log_dict_configs.prog_bar,
            logger=self.log_dict_configs.logger,
            on_step=on_step,
            on_epoch=self.log_dict_configs.on_epoch,
            reduce_fx=self.log_dict_configs.reduce_fx,
            sync_dist=self.log_dict_configs.sync_dist,
            batch_size=batch_size,
            rank_zero_only=self.log_dict_configs.rank_zero_only,
        )
        return metrics, outputs

    @final
    def training_step(
        self, multi_task_batch_inputs: ModelBatchInputs, batch_idx: int
    ) -> Float32[torch.Tensor, " 1"]:
        """
        Training step.

        Args:
            multi_task_batch_inputs: Inputs/features from multiple samples in the current batch
                required for the model.
            batch_idx: Current batch index.

        Returns:
            Total loss tensor required for backpropagation.
        """
        metrics, _ = self._core_step(
            multi_task_batch_inputs,
            step_prefix=SplitType.TRAIN.value,
        )
        return metrics["loss"]

    @final
    def validation_step(
        self, multi_task_batch_inputs: ModelBatchInputs, batch_idx: int
    ) -> dict[str, Any]:
        """
        Validation step.

        Args:
            multi_task_batch_inputs: Inputs/features from multiple samples in the current batch
                required for the model.
            batch_idx: Current batch index.

        Returns:
            Dictionary with at least a ``"loss"`` key and a ``"model_outputs"``
            key containing the raw forward outputs. The raw outputs are available
            to ``on_validation_batch_end`` for epoch-level metric accumulation.
        """
        metrics, outputs = self._core_step(
            multi_task_batch_inputs,
            step_prefix=SplitType.VAL.value,
        )
        # TODO (KokSeang): Return a dataclass instead of a dict for better type safety and clarity.
        return {**metrics, "model_outputs": outputs}

    @final
    def test_step(
        self, multi_task_batch_inputs: ModelBatchInputs, batch_idx: int
    ) -> dict[str, Any]:
        """Test step.

        Args:
            multi_task_batch_inputs: Inputs/features from multiple samples in the current batch
                required for the model.
            batch_idx: Batch index.

        Returns:
            Dictionary with at least a ``"loss"`` key and a ``"model_outputs"``
            key containing the raw forward outputs.
        """
        metrics, outputs = self._core_step(
            multi_task_batch_inputs,
            step_prefix=SplitType.TEST.value,
        )
        # TODO (KokSeang): Return a dataclass instead of a dict for better type safety and clarity.
        return {**metrics, "model_outputs": outputs}

    @final
    def predict_step(
        self, multi_task_batch_inputs: ModelBatchInputs, batch_idx: int
    ) -> ModelPredictions:
        """Prediction step.

        Args:
            multi_task_batch_inputs: Inputs/features from multiple samples in the current batch
                required for the model.
            batch_idx: Batch index.

        Returns:
            Predictions after decoding the raw model outputs.
        """
        outputs = self(multi_task_batch_inputs)
        return self.decode_outputs(multi_task_batch_inputs, outputs)

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

    def build_export_spec(self, multi_task_batch_inputs: ModelBatchInputs) -> ExportSpec:
        """Build the single-module deployment export specification.

        :meth:`forward` consumes a :class:`ModelBatchInputs` and returns a
        :class:`ModelOutputs`, neither of which the ONNX exporter can trace, so
        there is no generic signature-based default. Models that support deployment
        must override this hook and flatten the batch into the tensor arguments of
        the exported graph.

        Args:
            multi_task_batch_inputs: Example preprocessed batch used for export.

        Returns:
            Export specification for deployment.
        """
        raise NotImplementedError("Model must implement build_export_spec()")

    def build_export_specs(
        self, multi_task_batch_inputs: ModelBatchInputs
    ) -> dict[str, ExportSpec]:
        """Build per-module deployment export specifications.

        The default implementation wraps :meth:`build_export_spec` as a single
        ``end_to_end`` module. Models with separate exportable sub-graphs
        override this to return one spec per architectural component.

        Args:
            multi_task_batch_inputs: Example preprocessed batch used for export.

        Returns:
            Ordered mapping of module name to export specification.
        """
        return {"end_to_end": self.build_export_spec(multi_task_batch_inputs)}
