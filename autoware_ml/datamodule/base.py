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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from autoware_ml.datamodule.pipeline_context import PipelineContext
from autoware_ml.preprocessing.base import DataPreprocessing
from autoware_ml.transforms.base import TransformsCompose


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
    common setup logic, dataloader construction, and batch collation helpers.
    """

    def __init__(
        self,
        stack_keys: Sequence[str] | None = None,
        train_transforms: TransformsCompose | None = None,
        val_transforms: TransformsCompose | None = None,
        test_transforms: TransformsCompose | None = None,
        predict_transforms: TransformsCompose | None = None,
        data_preprocessing: DataPreprocessing | None = None,
        train_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        val_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        test_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
        predict_dataloader_cfg: DataLoaderConfig | Mapping[str, Any] | None = None,
    ):
        """Initialize DataModule.

        Args:
            stack_keys: List of keys to stack from the input dictionary.
            train_transforms: TransformsCompose for training dataset.
            val_transforms: TransformsCompose for validation dataset.
            test_transforms: TransformsCompose for test dataset.
            predict_transforms: TransformsCompose for predict dataset.
            data_preprocessing: Data preprocessing module.
            train_dataloader_cfg: Configuration for training data loader.
            val_dataloader_cfg: Configuration for validation data loader.
            test_dataloader_cfg: Configuration for test data loader.
            predict_dataloader_cfg: Configuration for predict data loader.
        """
        super().__init__()

        self.stack_keys: list[str] = list(stack_keys or [])
        # TransformsCompose for each dataset split
        self.train_transforms: TransformsCompose = train_transforms
        self.val_transforms: TransformsCompose = val_transforms
        self.test_transforms: TransformsCompose = test_transforms
        self.predict_transforms: TransformsCompose = predict_transforms
        self.data_preprocessing: DataPreprocessing = data_preprocessing
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
        splits = (
            ["train", "val", "test", "predict"] if stage is None else stage_splits.get(stage, [])
        )

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

    def collate_fn(self, batch_inputs_dicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
        """Collates batch elements into a dictionary of Tensors or Lists.

        1. Converts NumPy arrays, lists, tuples and scalars to PyTorch Tensors.
        2. Collates lists of dictionaries to dictionaries of lists.
        3. Stacks selected lists of tensors to a single tensor.

        Args:
            batch_inputs_dicts: List of dictionaries representing the batch inputs.

        Returns:
            Dictionary mapping keys to Tensors or lists of data.
        """
        if not batch_inputs_dicts:
            raise ValueError("Batch inputs dictionary is empty.")

        all_keys: set[str] = set()
        for input_dict in batch_inputs_dicts:
            all_keys.update(input_dict.keys())

        batch_inputs_dict: dict[str, list[Any]] = {key: [] for key in all_keys}

        for input_dict in batch_inputs_dicts:
            for key in all_keys:
                if key not in input_dict:
                    raise ValueError(f"Key '{key}' not found in input_dict.")

                item = input_dict[key]

                # Convert NumPy arrays to Tensors, enforcing memory contiguity
                if isinstance(item, np.ndarray):
                    if not item.flags.c_contiguous:
                        item = np.ascontiguousarray(item)
                    item = torch.from_numpy(item)

                # Convert scalars to Tensors
                elif isinstance(item, (float, int)):
                    item = torch.tensor(item)

                # Convert lists of numbers to Tensors
                elif isinstance(item, (list, tuple)):
                    if item and isinstance(item[0], (int, float)):
                        item = torch.as_tensor(item)

                batch_inputs_dict[key].append(item)

        for key in self.stack_keys:
            if key not in batch_inputs_dict:
                raise KeyError(f"Stack key '{key}' not found in batch_inputs_dict.")
            batch_inputs_dict[key] = torch.stack(batch_inputs_dict[key], dim=0)

        return batch_inputs_dict

    def on_after_batch_transfer(
        self, batch_inputs_dict: dict[str, Any], dataloader_idx: int
    ) -> dict[str, Any]:
        """Apply optional preprocessing after dataloader device transfer.

        Args:
            batch_inputs_dict: Batch dictionary returned by the dataloader.
            dataloader_idx: Index of the dataloader that produced the batch.

        Returns:
            Batch dictionary after optional preprocessing.
        """
        if self.data_preprocessing is not None:
            batch_inputs_dict = self.data_preprocessing(batch_inputs_dict)
        return batch_inputs_dict
