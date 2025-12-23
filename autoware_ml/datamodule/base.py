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

"""Base DataModule class for Autoware-ML framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from autoware_ml.preprocessing.base import DataPreprocessing
from autoware_ml.transforms import TransformsCompose


@dataclass
class DataLoaderConfig:
    """Configuration options for a single dataloader."""

    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = False
    drop_last: bool = False


class Dataset(TorchDataset, ABC):
    """Base dataset class for Autoware-ML framework."""

    def __init__(self, dataset_transforms: Optional[TransformsCompose] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.dataset_transforms = dataset_transforms

    def __getitem__(self, index: int) -> Dict[str, Any]:
        input_dict = self._get_input_dict(index)
        input_dict = self._transform(input_dict)
        return input_dict

    @abstractmethod
    def _get_input_dict(self, index: int) -> Dict[str, Any]:
        """Get batch inputs for a given index.

        Args:
            index: Index of the sample.

        Returns:
            Input dictionary.
        """
        raise NotImplementedError("Dataset must implement _get_input_dict")

    def _transform(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.dataset_transforms is None:
            return input_dict
        return self.dataset_transforms(input_dict)


class DataModule(L.LightningDataModule, ABC):
    """Top-level DataModule for data loading."""

    def __init__(
        self,
        stack_keys: Optional[List[str]] = None,
        train_transforms: Optional[TransformsCompose] = None,
        val_transforms: Optional[TransformsCompose] = None,
        test_transforms: Optional[TransformsCompose] = None,
        predict_transforms: Optional[TransformsCompose] = None,
        data_preprocessing: Optional[DataPreprocessing] = None,
        train_dataloader_cfg: Optional[DataLoaderConfig] = DataLoaderConfig(),
        val_dataloader_cfg: Optional[DataLoaderConfig] = DataLoaderConfig(),
        test_dataloader_cfg: Optional[DataLoaderConfig] = DataLoaderConfig(),
        predict_dataloader_cfg: Optional[DataLoaderConfig] = DataLoaderConfig(),
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

        self.stack_keys: Optional[List[str]] = stack_keys or []
        # TransformsCompose for each dataset split
        self.train_transforms: TransformsCompose = train_transforms
        self.val_transforms: TransformsCompose = val_transforms
        self.test_transforms: TransformsCompose = test_transforms
        self.predict_transforms: TransformsCompose = predict_transforms
        self.data_preprocessing: DataPreprocessing = data_preprocessing
        # Configuration for each dataset split
        self.train_dataloader_cfg: DataLoaderConfig = train_dataloader_cfg
        self.val_dataloader_cfg: DataLoaderConfig = val_dataloader_cfg
        self.test_dataloader_cfg: DataLoaderConfig = test_dataloader_cfg
        self.predict_dataloader_cfg: DataLoaderConfig = predict_dataloader_cfg

        # Dataset splits (to be created in setup)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.predict_dataset: Optional[Dataset] = None

    @abstractmethod
    def _create_dataset(
        self, split: str, transforms: Optional[TransformsCompose] = None
    ) -> Dataset:
        """Create dataset for a specific split.

        Subclasses must implement this method to create dataset instances
        for different splits (train, val, test, predict).

        Args:
            split: Dataset split name ("train", "val", "test", "predict").

        Returns:
            Dataset instance for the split.
        """
        raise NotImplementedError("Dataset must implement _create_dataset")

    def setup(self, stage: str) -> None:
        """Setup datasets for each stage.

        This unified implementation handles all stages and eliminates
        code duplication. It maps stages to dataset splits
        and creates datasets using _create_dataset().

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        # Define stage to splits mapping
        stage_splits = {
            "fit": ["train", "val"],
            "validate": ["val"],
            "test": ["test"],
            "predict": ["predict"],
        }

        # Get splits for this stage
        splits = stage_splits.get(stage, [])

        # Create datasets for required splits
        for split in splits:
            if getattr(self, f"{split}_dataset") is not None:
                continue
            transforms = getattr(self, f"{split}_transforms")
            dataset = self._create_dataset(split, transforms)
            setattr(self, f"{split}_dataset", dataset)

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader.

        Returns:
            Training DataLoader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_dataloader_cfg.batch_size,
            shuffle=self.train_dataloader_cfg.shuffle,
            num_workers=self.train_dataloader_cfg.num_workers,
            pin_memory=self.train_dataloader_cfg.pin_memory,
            persistent_workers=self.train_dataloader_cfg.persistent_workers
            and self.train_dataloader_cfg.num_workers > 0,
            drop_last=self.train_dataloader_cfg.drop_last,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.

        Returns:
            Validation DataLoader.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_dataloader_cfg.batch_size,
            shuffle=self.val_dataloader_cfg.shuffle,
            num_workers=self.val_dataloader_cfg.num_workers,
            pin_memory=self.val_dataloader_cfg.pin_memory,
            persistent_workers=self.val_dataloader_cfg.persistent_workers
            and self.val_dataloader_cfg.num_workers > 0,
            drop_last=self.val_dataloader_cfg.drop_last,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader.

        Returns:
            Test DataLoader.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_dataloader_cfg.batch_size,
            shuffle=self.test_dataloader_cfg.shuffle,
            num_workers=self.test_dataloader_cfg.num_workers,
            pin_memory=self.test_dataloader_cfg.pin_memory,
            persistent_workers=self.test_dataloader_cfg.persistent_workers
            and self.test_dataloader_cfg.num_workers > 0,
            drop_last=self.test_dataloader_cfg.drop_last,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction dataloader.

        Returns:
            Prediction DataLoader.
        """
        return DataLoader(
            dataset=self.predict_dataset,
            batch_size=self.predict_dataloader_cfg.batch_size,
            shuffle=self.predict_dataloader_cfg.shuffle,
            num_workers=self.predict_dataloader_cfg.num_workers,
            pin_memory=self.predict_dataloader_cfg.pin_memory,
            persistent_workers=self.predict_dataloader_cfg.persistent_workers
            and self.predict_dataloader_cfg.num_workers > 0,
            drop_last=self.predict_dataloader_cfg.drop_last,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch_inputs_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collates batch elements into a dictionary of Tensors or Lists.

        1. Converts NumPy arrays, lists, tuples and scalars to PyTorch Tensors.
        2. Collates lists of dictionaries to dictionaries of lists.
        3. Stacks selected lists of tensors to a single tensor.

        Args:
            batch_inputs_dicts: List of dictionaries representing the batch inputs.

        Returns:
            Dictionary mapping keys to Tensors or lists of data.
        """
        assert len(batch_inputs_dicts) > 0, "Batch inputs dictionary is empty."

        all_keys: Set[str] = set()
        for input_dict in batch_inputs_dicts:
            all_keys.update(input_dict.keys())

        batch_inputs_dict: Dict[str, List[Any]] = {key: [] for key in all_keys}

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
            assert key in batch_inputs_dict, f"Key '{key}' not found in batch_inputs_dict."
            assert len(batch_inputs_dict[key]) > 0, f"List for key '{key}' is empty."
            batch_inputs_dict[key] = torch.stack(batch_inputs_dict[key], dim=0)

        return batch_inputs_dict

    def on_after_batch_transfer(
        self, batch_inputs_dict: Dict[str, Any], dataloader_idx: int
    ) -> Dict[str, Any]:
        if self.data_preprocessing is not None:
            batch_inputs_dict = self.data_preprocessing(batch_inputs_dict)
        return batch_inputs_dict
