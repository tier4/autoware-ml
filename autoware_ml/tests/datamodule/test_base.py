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

"""Unit tests for shared datamodule base classes."""

from __future__ import annotations

from typing import Any

from autoware_ml.datamodule.base import DataLoaderConfig, DataModule, Dataset


class _DummyDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def get_data_info(self, index: int) -> dict[str, Any]:
        return {"index": index}


class _DummyDataModule(DataModule):
    def _create_dataset(self, split: str, transforms=None) -> Dataset:
        return _DummyDataset(dataset_transforms=transforms)


class TestDataModuleBase:
    def test_default_dataloader_configs_are_not_shared(self) -> None:
        first = _DummyDataModule()
        second = _DummyDataModule()

        assert first.train_dataloader_cfg is not second.train_dataloader_cfg
        assert first.val_dataloader_cfg is not second.val_dataloader_cfg
        assert first.test_dataloader_cfg is not second.test_dataloader_cfg
        assert first.predict_dataloader_cfg is not second.predict_dataloader_cfg

    def test_setup_none_initializes_all_splits(self) -> None:
        datamodule = _DummyDataModule()

        datamodule.setup(stage=None)

        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None
        assert datamodule.test_dataset is not None
        assert datamodule.predict_dataset is not None

    def test_explicit_dataloader_config_is_preserved(self) -> None:
        dataloader_cfg = DataLoaderConfig(batch_size=4, num_workers=2)
        datamodule = _DummyDataModule(train_dataloader_cfg=dataloader_cfg)

        assert datamodule.train_dataloader_cfg is dataloader_cfg

    def test_mapping_dataloader_config_is_coerced(self) -> None:
        datamodule = _DummyDataModule(train_dataloader_cfg={"batch_size": 4, "num_workers": 2})

        assert isinstance(datamodule.train_dataloader_cfg, DataLoaderConfig)
        assert datamodule.train_dataloader_cfg.batch_size == 4
        assert datamodule.train_dataloader_cfg.num_workers == 2
