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

"""Regression tests for epoch-end validation with variable-length samplers."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from autoware_ml.utils.lightning_loops import install_epoch_end_validation


class _RangeDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([float(index)])


class _ShrinkingSampler(Sampler[int]):
    """Serves 4 samples on epoch 0 and 3 on every later epoch.

    Mimics ``GroupStreamingSampler``: the per-epoch length depends on the epoch
    while Lightning captures the dataloader length (and thus ``val_check_batch``)
    only once at setup.
    """

    def __init__(self) -> None:
        self.epoch = 0

    def _length(self) -> int:
        return 4 if self.epoch == 0 else 3

    def __iter__(self):
        return iter(range(self._length()))

    def __len__(self) -> int:
        return self._length()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class _TinyModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.val_epochs_run: list[int] = []

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.layer(batch).mean()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.layer(batch).mean()

    def on_validation_epoch_end(self) -> None:
        self.val_epochs_run.append(self.current_epoch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _fit(tmp_path: Path, *, epoch_end_validation: bool) -> list[int]:
    model = _TinyModule()
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
    )
    if epoch_end_validation:
        install_epoch_end_validation(trainer)
    train_loader = DataLoader(_RangeDataset(), batch_size=1, sampler=_ShrinkingSampler())
    val_loader = DataLoader(_RangeDataset(), batch_size=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model.val_epochs_run


def test_stock_loop_skips_validation_on_shorter_epochs(tmp_path: Path) -> None:
    # Documents the upstream defect the override exists for: epochs 1 and 2 run
    # 3 batches while val_check_batch stays 4, so their validation never fires.
    # If this starts failing with [0, 1, 2], Lightning fixed it upstream and
    # EpochEndValidationLoop can be retired.
    assert _fit(tmp_path, epoch_end_validation=False) == [0]


def test_epoch_end_validation_loop_validates_every_epoch(tmp_path: Path) -> None:
    assert _fit(tmp_path, epoch_end_validation=True) == [0, 1, 2]
