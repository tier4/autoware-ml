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

"""Base datamodule abstractions for Autoware-ML.

This module defines shared configuration containers and abstract datamodule
interfaces used by training, evaluation, and deployment entrypoints.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from autoware_ml.datamodule.collation import CollationStrategy
from autoware_ml.datamodule.pipeline_context import PipelineContext
from autoware_ml.transforms.base import TransformsCompose

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Store configuration values for one dataloader.

    Attributes:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes used by the dataloader.
        pin_memory: Whether to pin host memory before device transfer.
        persistent_workers: Whether worker processes stay alive across epochs.
        shuffle: Whether the dataloader shuffles samples.
        drop_last: Whether to drop the final incomplete batch.
    """

    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = False
    drop_last: bool = False

    def to_dataloader_kwargs(self) -> dict[str, Any]:
        """Convert to keyword arguments accepted by ``DataLoader``.

        Returns:
            Dictionary of DataLoader constructor keyword arguments.
        """
        return {
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
            "drop_last": self.drop_last,
        }


class Dataset(TorchDataset, ABC):
    """Define the base dataset interface for Autoware-ML.

    Subclasses implement :meth:`get_data_info` and may rely on the shared
    transform handling implemented by this class.
    """

    def __init__(self, dataset_transforms: TransformsCompose | None = None, **kwargs: Any):
        """Initialize the dataset base class.

        Args:
            dataset_transforms: Optional transform pipeline applied per sample.
            **kwargs: Additional arguments forwarded to ``TorchDataset``.
        """
        super().__init__(**kwargs)
        self.dataset_transforms = dataset_transforms

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and transform one dataset sample.

        Args:
            index: Sample index.

        Returns:
            Transformed dataset sample.
        """
        input_dict = self.get_data_info(index)
        context = PipelineContext(dataset=self, index=index)
        return self.apply_transforms(input_dict, self.dataset_transforms, context)

    @abstractmethod
    def get_data_info(self, index: int) -> dict[str, Any]:
        """Return raw metadata for a given dataset index.

        Args:
            index: Index of the sample.

        Returns:
            Metadata dictionary consumed by the transform pipeline.
        """
        raise NotImplementedError("Dataset must implement get_data_info")

    def apply_transforms(
        self,
        input_dict: dict[str, Any],
        dataset_transforms: TransformsCompose | None,
        context: PipelineContext,
    ) -> dict[str, Any]:
        """Apply a specific transform pipeline to a metadata sample.

        Args:
            input_dict: Metadata sample dictionary.
            dataset_transforms: Transform pipeline applied to the sample.
            context: Pipeline context associated with the sample.

        Returns:
            Transformed dataset sample.
        """
        if dataset_transforms is None:
            return input_dict
        return dataset_transforms(input_dict, context=context)


class DataModule(L.LightningDataModule, ABC):
    """Define the shared Lightning DataModule behavior for Autoware-ML.

    Subclasses create split-specific datasets while this base class provides
    common setup logic, dataloader construction, and batch collation.

    Collation contract:
        ``collation_map`` is an explicit whitelist of keys that may appear in
        the collated batch. Every key listed must be present in every sample.
        Missing keys are skipped with a warning. Each key declares exactly one strategy.

        Strategies:
            * ``"concat"``: concatenate per-sample tensors along dim 0. The
              first ``"concat"`` key in the map is the primary point cloud
              space. Its per-sample lengths produce the cumulative ``offset``
              tensor that is added to the batch.
            * ``"stack"``: stack fixed-shape per-sample tensors along a new
              dim 0. Shape mismatch raises ``ValueError``.
            * ``"index_concat"``: concatenate integer index tensors along dim
              0 and shift each sample's values by the exclusive cumulative
              offset of the primary point cloud space, making indices globally
              valid across the batch. Requires at least one ``"concat"`` key.
            * ``"list"``: keep per-sample values as a Python list. Numpy
              arrays are converted to tensors element-wise.

    Failure modes:
        Undeclared keys are dropped, declared-but-missing keys are skipped
        with a warning, and a wrong strategy for a key's content fails loudly.
    """

    def __init__(
        self,
        collation_map: Mapping[str, CollationStrategy] | None = None,
        train_transforms: TransformsCompose | None = None,
        val_transforms: TransformsCompose | None = None,
        test_transforms: TransformsCompose | None = None,
        predict_transforms: TransformsCompose | None = None,
        train_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        val_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        test_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        predict_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
    ):
        """Initialize DataModule.

        Args:
            collation_map: Per-key collation strategy applied across all
                splits. Only keys listed here reach the batch. All other
                keys are dropped. See the class docstring for the
                per-strategy contract.
            train_transforms: Transform pipeline applied to training samples.
            val_transforms: Transform pipeline applied to validation samples.
            test_transforms: Transform pipeline applied to test samples.
            predict_transforms: Transform pipeline applied to predict samples.
            train_dataloader_cfg: Configuration for the training dataloader.
            val_dataloader_cfg: Configuration for the validation dataloader.
            test_dataloader_cfg: Configuration for the test dataloader.
            predict_dataloader_cfg: Configuration for the predict dataloader.
        """
        super().__init__()

        self.collation_map: dict[str, CollationStrategy] = dict(collation_map or {})
        # TransformsCompose for each dataset split
        self.train_transforms: TransformsCompose = train_transforms
        self.val_transforms: TransformsCompose = val_transforms
        self.test_transforms: TransformsCompose = test_transforms
        self.predict_transforms: TransformsCompose = predict_transforms
        # Configuration for each dataset split
        self.train_dataloader_cfg = self._coerce_dataloader_cfg(train_dataloader_cfg)
        self.val_dataloader_cfg = self._coerce_dataloader_cfg(val_dataloader_cfg)
        self.test_dataloader_cfg = self._coerce_dataloader_cfg(test_dataloader_cfg)
        self.predict_dataloader_cfg = self._coerce_dataloader_cfg(predict_dataloader_cfg)

        # Dataset splits (to be created in setup)
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None
        self.predict_dataset: Dataset | None = None

    @staticmethod
    def _coerce_dataloader_cfg(
        cfg: DataLoaderConfig | Mapping[str, Any] | None,
    ) -> DataLoaderConfig:
        """Normalize dataloader config values to ``DataLoaderConfig``.

        Hydra composition can pass split dataloader settings as a plain
        ``dict`` or ``DictConfig``. Normalize those mapping inputs at the
        datamodule boundary so downstream code can rely on the dataclass API.

        Args:
            cfg: Optional dataloader config object or mapping.

        Returns:
            Normalized ``DataLoaderConfig`` instance.

        Raises:
            TypeError: If the provided value cannot be converted.
        """
        if cfg is None:
            return DataLoaderConfig()
        if isinstance(cfg, DataLoaderConfig):
            return cfg
        if isinstance(cfg, Mapping):
            return DataLoaderConfig(**dict(cfg))
        raise TypeError(
            "Expected dataloader config to be a DataLoaderConfig, mapping, or None, "
            f"got {type(cfg)!r}."
        )

    @abstractmethod
    def _create_dataset(self, split: str, transforms: TransformsCompose | None = None) -> Dataset:
        """Create dataset for a specific split.

        Subclasses must implement this method to create dataset instances
        for different splits (train, val, test, predict).

        Args:
            split: Dataset split name ("train", "val", "test", "predict").

        Returns:
            Dataset instance for the split.
        """
        raise NotImplementedError("Dataset must implement _create_dataset")

    def setup(self, stage: str | None = None) -> None:
        """Setup datasets for each stage.

        This unified implementation handles all stages and eliminates
        code duplication. It maps stages to dataset splits
        and creates datasets using _create_dataset().

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict') or
                ``None`` to prepare all splits.
        """
        # Define stage to splits mapping
        stage_splits = {
            "fit": ["train", "val"],
            "validate": ["val"],
            "test": ["test"],
            "predict": ["predict"],
        }

        # Get splits for this stage
        splits = ["train", "val", "test", "predict"] if stage is None else stage_splits[stage]

        # Create datasets for required splits
        for split in splits:
            if getattr(self, f"{split}_dataset") is not None:
                continue
            transforms = getattr(self, f"{split}_transforms")
            dataset = self._create_dataset(split, transforms)
            setattr(self, f"{split}_dataset", dataset)

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create a dataloader for the given split.

        Args:
            split: Dataset split name (``train``, ``val``, ``test``, ``predict``).

        Returns:
            Configured DataLoader for the split.
        """
        dataset = getattr(self, f"{split}_dataset")
        cfg: DataLoaderConfig = getattr(self, f"{split}_dataloader_cfg")
        return DataLoader(dataset=dataset, collate_fn=self.collate_fn, **cfg.to_dataloader_kwargs())

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return self._create_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return self._create_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return self._create_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader."""
        return self._create_dataloader("predict")

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        """Convert a per-sample value to a tensor when possible.

        Numpy arrays and Python numeric scalars are converted to tensors so
        downstream strategies can treat all numeric inputs uniformly.
        Existing tensors pass through unchanged. Any other value passes
        through unchanged, which keeps ``"list"`` strategy compatible with
        arbitrary Python objects.

        Args:
            value: A per-sample value drawn from the batch.

        Returns:
            A ``torch.Tensor`` when ``value`` is a numpy array or a Python
            numeric scalar, the original tensor when already a ``Tensor``,
            otherwise the original value.
        """
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            array = value if value.flags.c_contiguous else np.ascontiguousarray(value)
            return torch.from_numpy(array)
        # `bool` is a subclass of `int` and is handled by the same branch.
        if isinstance(value, (int, float)):
            return torch.tensor(value)
        return value

    @staticmethod
    def _apply_stack(key: str, values: list[torch.Tensor]) -> torch.Tensor:
        """Stack fixed-shape per-sample tensors along a new batch dim.

        Args:
            key: Batch key name, used in the error message for diagnostics.
            values: Per-sample tensors. Every tensor must share the same
                shape as ``values[0]``.

        Returns:
            A tensor of shape ``(batch_size, *values[0].shape)``.

        Raises:
            ValueError: When any sample has a shape different from the first.
        """
        expected_shape = values[0].shape
        for sample_idx, value in enumerate(values):
            if value.shape != expected_shape:
                raise ValueError(
                    f"Key '{key}' configured as 'stack' but sample {sample_idx} has shape "
                    f"{list(value.shape)}, expected {list(expected_shape)}."
                )
        return torch.stack(values, dim=0)

    @staticmethod
    def _apply_concat(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate variable-length per-sample tensors along dim 0.

        Args:
            values: Per-sample tensors with matching trailing shape.

        Returns:
            A tuple of:
                * concatenated tensor along dim 0,
                * per-sample lengths as an ``int64`` tensor of shape
                  ``(batch_size,)``.
        """
        lengths = torch.tensor([v.shape[0] for v in values], dtype=torch.long)
        return torch.cat(values, dim=0), lengths

    @staticmethod
    def _apply_index_concat(
        values: list[torch.Tensor], exclusive_offset: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate index tensors and shift each sample by the primary offset.

        Each sample's index values reference positions in the primary point
        cloud space. After concatenation, per-sample chunks are shifted by
        the corresponding exclusive cumulative count of the primary space so
        the resulting indices remain valid across the whole batch.

        Args:
            values: Per-sample integer index tensors.
            exclusive_offset: Exclusive cumulative offset of the primary
                point cloud space. Shape ``(batch_size,)``.

        Returns:
            A 1D integer tensor of globally valid indices.
        """
        lengths = torch.tensor([v.shape[0] for v in values], dtype=torch.long)
        shift = torch.repeat_interleave(exclusive_offset.to(values[0].dtype), lengths)
        return torch.cat(values, dim=0) + shift

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch according to ``self.collation_map``.

        The function dispatches each declared key to its strategy handler,
        derives a cumulative ``offset`` tensor from the first ``"concat"``
        key, and applies the offset shift to any ``"index_concat"`` keys.
        See the class docstring for the full
        per-strategy contract.

        Args:
            batch: Non-empty list of per-sample dictionaries.

        Returns:
            Dictionary of collated values keyed by the names declared in
            ``self.collation_map``. When any ``"concat"`` key is present, an
            additional ``"offset"`` key is added with the inclusive
            cumulative count of the primary point cloud space.

        Raises:
            ValueError: Empty batch, a ``"stack"`` key with mismatched
                shapes, ``"index_concat"`` declared without any ``"concat"``
                key, or an unknown strategy value.
        """
        if not batch:
            raise ValueError("Batch is empty.")

        result: dict[str, Any] = {}
        primary_lengths: torch.Tensor | None = None
        deferred_index_concat: list[tuple[str, list[torch.Tensor]]] = []

        for key, strategy in self.collation_map.items():
            missing = [i for i, sample in enumerate(batch) if key not in sample]
            if missing:
                logger.warning(
                    "Key '%s' declared in collation_map but missing from samples %s. "
                    "Skipping this key during collation. If this comes from deployment/predict, "
                    "it is expected for training-only annotation keys.",
                    key,
                    missing,
                )
                continue

            values = [self._coerce_value(sample[key]) for sample in batch]

            match strategy:
                case CollationStrategy.STACK:
                    result[key] = self._apply_stack(key, values)
                case CollationStrategy.CONCAT:
                    tensor, lengths = self._apply_concat(values)
                    result[key] = tensor
                    if primary_lengths is None:
                        primary_lengths = lengths
                        result["offset"] = torch.cumsum(lengths, dim=0)
                case CollationStrategy.INDEX_CONCAT:
                    deferred_index_concat.append((key, values))
                case CollationStrategy.LIST:
                    result[key] = list(values)
                case _:
                    raise ValueError(f"Unknown CollationStrategy {strategy!r} for key '{key}'.")

        if deferred_index_concat:
            if primary_lengths is None:
                raise ValueError(
                    "'index_concat' requires at least one 'concat' key to define "
                    "the primary point cloud offset. Got: "
                    f"{[k for k, _ in deferred_index_concat]}."
                )
            exclusive_offset = torch.cat(
                [torch.zeros(1, dtype=torch.long), torch.cumsum(primary_lengths, dim=0)[:-1]]
            )
            for key, values in deferred_index_concat:
                result[key] = self._apply_index_concat(values, exclusive_offset)

        return result
