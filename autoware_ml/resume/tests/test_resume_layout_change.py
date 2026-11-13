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

"""End to end tests of a resume with another batch size: epoch end and learning rate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autoware_ml.resume.data_loader import (
    ResumableDataLoader,
    ResumableDistributedSampler,
    ResumableTrainLoaderMixin,
)
from autoware_ml.resume.scheduler_horizon import SchedulerHorizon

_NUM_SAMPLES = 40


class _IndexDataset(Dataset):
    """Dataset whose samples are their own indices."""

    def __len__(self) -> int:
        return _NUM_SAMPLES

    def __getitem__(self, index: int) -> int:
        return index


class _Preempted(RuntimeError):
    """Stands in for the kill a requeue delivers without any grace period."""


class _Module(L.LightningModule):
    """Tiny model with an epoch-wise or a step-wise scheduler that records its training."""

    def __init__(self, scheduler: str) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.scheduler = scheduler
        self.batch_indices: list[tuple[int, int]] = []
        self.validated_epochs: list[int] = []

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        self.batch_indices.append((self.current_epoch, batch_idx))
        return self.layer(batch.float().unsqueeze(1)).mean()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.layer(batch.float().unsqueeze(1))

    def on_validation_epoch_end(self) -> None:
        self.validated_epochs.append(self.current_epoch)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        if self.scheduler == "epoch":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.1, total_steps=int(self.trainer.estimated_stepping_batches)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class _DataModule(ResumableTrainLoaderMixin, L.LightningDataModule):
    """Index datamodule carrying its loader position through the framework's mixin."""

    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self) -> ResumableDataLoader:
        dataset = _IndexDataset()
        sampler = ResumableDistributedSampler(dataset, shuffle=True, seed=3)
        loader = ResumableDataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=torch.tensor,
            allow_layout_change=True,
        )
        return self._attach_train_loader(loader)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(_IndexDataset(), batch_size=10, collate_fn=torch.tensor)


class _SaveAndDie(L.Callback):
    """Save a checkpoint after a given batch of a given epoch, then die like a preemption."""

    def __init__(self, path: Path, epoch: int, batch_idx: int) -> None:
        self.path = path
        self.epoch = epoch
        self.batch_idx = batch_idx

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (trainer.current_epoch, batch_idx) == (self.epoch, self.batch_idx):
            trainer.save_checkpoint(self.path)
            raise _Preempted(f"killed after batch {batch_idx}")


def _fit(
    module: _Module,
    batch_size: int,
    max_epochs: int,
    callbacks: list[L.Callback],
    ckpt_path: Path | None = None,
    accumulate_grad_batches: int = 1,
) -> L.Trainer:
    """Fit on CPU with the given callbacks, validating after every epoch."""
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=_DataModule(batch_size), ckpt_path=ckpt_path)
    return trainer


def _learning_rate(trainer: L.Trainer) -> float:
    """Learning rate of the first param group at the end of the run."""
    return float(trainer.optimizers[0].param_groups[0]["lr"])


class TestResumeWithAnotherBatchSize(unittest.TestCase):
    """A run killed mid-epoch and resumed with another batch size."""

    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.checkpoint = Path(temp_dir.name) / "killed.ckpt"

    def _kill(self, scheduler: str, batch_size: int, max_epochs: int, epoch: int, batch: int):
        """Train until the given batch, save and die."""
        with self.assertRaises(_Preempted):
            _fit(
                _Module(scheduler),
                batch_size,
                max_epochs,
                callbacks=[_SaveAndDie(self.checkpoint, epoch, batch)],
            )

    def test_resumed_epoch_ends_with_validation_and_the_epoch_scheduler_step(self) -> None:
        """
        Input: batch size 4 (10 batches per epoch), killed after batch 6 of epoch 1, resumed
        with batch size 8 (5 batches per epoch).
        Expected: the loader continues at batch 3 of 5 and Lightning counts from there, so
        epochs 1 and 2 both end with validation and a scheduler step, and the learning rate
        equals the one of an uninterrupted run.
        """
        reference = _fit(_Module("epoch"), batch_size=4, max_epochs=3, callbacks=[])
        self._kill("epoch", batch_size=4, max_epochs=3, epoch=1, batch=5)

        resumed = _Module("epoch")
        trainer = _fit(resumed, 8, max_epochs=3, callbacks=[], ckpt_path=self.checkpoint)

        self.assertEqual(
            resumed.batch_indices,
            [(1, 3), (1, 4)] + [(2, index) for index in range(5)],
        )
        self.assertEqual(resumed.validated_epochs, [1, 2])
        self.assertEqual(trainer.lr_scheduler_configs[0].scheduler.last_epoch, 3)
        self.assertAlmostEqual(_learning_rate(trainer), _learning_rate(reference))

    def test_one_cycle_refit_to_a_larger_batch_ends_on_the_last_step(self) -> None:
        """
        Input: OneCycleLR over 2 epochs of batch size 4 (20 steps), killed after step 5,
        resumed with batch size 8 (10 steps for the run).
        Expected: the scheduler is refit to 10 steps at step 2 (8 steps remain: 3 of epoch 0
        and 5 of epoch 1) and ends exactly on its last step with the final learning rate.
        """
        reference = _fit(_Module("step"), batch_size=8, max_epochs=2, callbacks=[])
        self._kill("step", batch_size=4, max_epochs=2, epoch=0, batch=4)

        trainer = _fit(
            _Module("step"),
            8,
            max_epochs=2,
            callbacks=[SchedulerHorizon(allow_layout_change=True)],
            ckpt_path=self.checkpoint,
        )

        scheduler = trainer.lr_scheduler_configs[0].scheduler
        self.assertEqual(scheduler.total_steps, 10)
        self.assertEqual(scheduler.last_epoch, 10)
        self.assertAlmostEqual(_learning_rate(trainer), _learning_rate(reference))

    def test_one_cycle_refit_to_a_smaller_batch_does_not_step_past_its_end(self) -> None:
        """
        Input: OneCycleLR over 2 epochs of batch size 4 with accumulation 2 (10 steps),
        killed after batch 6 of epoch 0, resumed with batch size 2 and accumulation 1 (40
        steps for the run).
        Expected: the restored 10-step schedule would raise past its end; the refit one ends
        on step 40 with the final learning rate.
        """
        reference = _fit(_Module("step"), batch_size=2, max_epochs=2, callbacks=[])
        with self.assertRaises(_Preempted):
            _fit(
                _Module("step"),
                4,
                max_epochs=2,
                callbacks=[_SaveAndDie(self.checkpoint, 0, 5)],
                accumulate_grad_batches=2,
            )

        trainer = _fit(
            _Module("step"),
            2,
            max_epochs=2,
            callbacks=[SchedulerHorizon(allow_layout_change=True)],
            ckpt_path=self.checkpoint,
        )

        scheduler = trainer.lr_scheduler_configs[0].scheduler
        self.assertEqual(scheduler.total_steps, 40)
        self.assertEqual(scheduler.last_epoch, 40)
        self.assertAlmostEqual(_learning_rate(trainer), _learning_rate(reference))

    def test_same_layout_keeps_the_restored_scheduler(self) -> None:
        """
        Input: OneCycleLR killed after step 5 and resumed with the same batch size.
        Expected: the callback leaves the restored state alone; the run ends on step 20.
        """
        self._kill("step", batch_size=4, max_epochs=2, epoch=0, batch=4)
        horizon = SchedulerHorizon(allow_layout_change=False)
        trainer = _fit(
            _Module("step"), 4, max_epochs=2, callbacks=[horizon], ckpt_path=self.checkpoint
        )
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        self.assertEqual((scheduler.total_steps, scheduler.last_epoch), (20, 20))

    def test_refit_is_refused_by_default(self) -> None:
        """
        Input: OneCycleLR killed after step 5 and resumed with batch size 8 by a callback
        that does not allow a layout change.
        Expected: the resume fails before training, naming the option.
        """
        self._kill("step", batch_size=4, max_epochs=2, epoch=0, batch=4)
        with self.assertRaisesRegex(RuntimeError, "allow_resume_layout_change"):
            _fit(
                _Module("step"),
                8,
                max_epochs=2,
                callbacks=[SchedulerHorizon()],
                ckpt_path=self.checkpoint,
            )


if __name__ == "__main__":
    unittest.main()
