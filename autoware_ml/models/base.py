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

"""Base model classes for Autoware-ML framework."""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Sequence, Union

import lightning as L
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class BaseModel(L.LightningModule, ABC):
    """Base Lightning Module for all Autoware-ML models.

    Provides common functionality for training, validation, and testing with
    built-in support for flexible optimizer and scheduler configuration.
    All parameters are explicitly typed for IDE support and type checking.
    """

    def __init__(
        self,
        optimizer: Optional[Callable[..., Optimizer]] = None,
        scheduler: Optional[Callable[[Optimizer], LRScheduler]] = None,
    ):
        """Initialize base model.

        Args:
            optimizer: Callable that returns an optimizer when given model parameters.
            scheduler: Callable that returns a scheduler when given the optimizer.
        """
        super().__init__()
        self.forward_signature = inspect.signature(self.forward)
        self.compute_metrics_signature = inspect.signature(self.compute_metrics)
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler

    @abstractmethod
    def forward(self, **kwargs: Any) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
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
        self, outputs: Union[torch.Tensor, Sequence[torch.Tensor]], **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """Compute metrics.

        Args:
            outputs: Model outputs from forward().
            **kwargs: Keyword arguments.

        Returns:
            Dictionary of metrics as tensors with, 'loss' key is required.
        """
        pass

    def _shared_step(
        self, batch_inputs_dict: Dict[str, Any], step_prefix: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Shared step for training, validation, and test steps.

        Args:
            batch_inputs_dict: Dictionary with input data.
            step_prefix: Prefix for the set (train, val, test).
            **kwargs: Keyword arguments.

        Returns:
            Dictionary with metrics.
        """
        outputs = self(
            **{
                k: batch_inputs_dict[k]
                for k in self.forward_signature.parameters
                if k in batch_inputs_dict
            }
        )
        metrics = self.compute_metrics(
            outputs=outputs,
            **{
                k: batch_inputs_dict[k]
                for k in self.compute_metrics_signature.parameters
                if k in batch_inputs_dict
            },
        )
        assert "loss" in metrics, "'loss' key must be in metrics dictionary."
        batch_size = len(batch_inputs_dict[list(batch_inputs_dict.keys())[0]])
        self.log_dict(
            {f"{step_prefix}/{k}": v for k, v in metrics.items()},
            batch_size=batch_size,
            **kwargs,
        )

        return metrics

    def training_step(self, batch_inputs_dict: Dict[str, Any], batch_idx: int) -> torch.Tensor:
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

    def validation_step(self, batch_inputs_dict: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Validation step.

        Args:
            batch: Validation batch.
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

    def test_step(self, batch_inputs_dict: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
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

    def predict_step(self, batch_inputs_dict: Dict[str, Any], batch_idx: int) -> Any:
        """Prediction step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Predictions.
        """
        return self(batch_inputs_dict)

    def configure_optimizers(self) -> Union[Optimizer, Dict[str, Any]]:
        """Configure optimizers and schedulers."""
        if self.optimizer_partial is None:
            raise ValueError("Optimizer must be provided.")

        optimizer = self.optimizer_partial(params=self.parameters())

        if self.scheduler_partial is None:
            return optimizer

        scheduler = self.scheduler_partial(optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
