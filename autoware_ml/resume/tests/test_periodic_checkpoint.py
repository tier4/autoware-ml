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

"""End to end tests of the preemption-safe checkpointing and resume."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autoware_ml.resume.atomic_checkpoint_io import AtomicCheckpointIO
from autoware_ml.resume.checkpoint_locator import CheckpointLocator
from autoware_ml.resume.data_loader import (
    ResumableDataLoader,
    ResumableDistributedSampler,
    ResumableTrainLoaderMixin,
)
from autoware_ml.resume.periodic_checkpoint import PeriodicCheckpoint


class _IndexDataset(Dataset):
    """Dataset whose samples are their own indices."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


class _Preempted(RuntimeError):
    """Stands in for the kill a requeue delivers without any grace period."""


class _RecordingModule(L.LightningModule):
    """Tiny model that records which samples every training step saw."""

    def __init__(self, die_at_batch: int | None = None) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)
        self.seen: list[list[int]] = []
        self.die_at_batch = die_at_batch

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if self.die_at_batch is not None and batch_idx == self.die_at_batch:
            raise _Preempted(f"killed before batch {batch_idx}")
        self.seen.append(batch.tolist())
        return self.layer(batch.float().unsqueeze(1)).mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # The framework's models read this to size their schedulers, which makes the
        # trainer build the dataloaders before the loop state is restored. The resume
        # must survive that.
        if self.trainer.estimated_stepping_batches <= 0:
            raise AssertionError("The trainer estimated no optimizer steps.")
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _ScheduledModule(_RecordingModule):
    """Doubles the learning rate every epoch, records the rate of each epoch and validates."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_lrs: dict[int, float] = {}
        self.validated_epochs: list[int] = []

    def on_train_epoch_start(self) -> None:
        self.epoch_lrs[self.current_epoch] = self.trainer.optimizers[0].param_groups[0]["lr"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.log("val/metric", self.layer(batch.float().unsqueeze(1)).mean())

    def on_validation_epoch_end(self) -> None:
        self.validated_epochs.append(self.current_epoch)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 2.0**epoch)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class _DataModule(ResumableTrainLoaderMixin, L.LightningDataModule):
    """Index datamodule carrying its loader position through the framework's mixin."""

    def __init__(self, size: int = 40, batch_size: int = 4) -> None:
        super().__init__()
        self.size = size
        self.batch_size = batch_size

    def train_dataloader(self) -> ResumableDataLoader:
        dataset = _IndexDataset(self.size)
        sampler = ResumableDistributedSampler(dataset, shuffle=True, seed=11)
        loader = ResumableDataLoader(
            dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=torch.tensor
        )
        return self._attach_train_loader(loader)


class _ValidatedDataModule(_DataModule):
    """Index datamodule with a small validation loader."""

    def val_dataloader(self) -> DataLoader:
        return DataLoader(_IndexDataset(4), batch_size=2, collate_fn=torch.tensor)


class _StopAfter(L.Callback):
    """Create the stop file after a given number of batches, like the launcher does."""

    def __init__(self, stop_file: Path, after_batches: int) -> None:
        self.stop_file = stop_file
        self.after_batches = after_batches

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx + 1 == self.after_batches:
            self.stop_file.write_text("stop\n")


def _trainer(root: Path, callback: PeriodicCheckpoint, **kwargs: Any) -> L.Trainer:
    """CPU trainer with only the periodic callback and the atomic writer."""
    return L.Trainer(
        default_root_dir=str(root),
        accelerator="cpu",
        devices=1,
        callbacks=[callback],
        plugins=[AtomicCheckpointIO()],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        **kwargs,
    )


def _batches_done(checkpoint_path: Path) -> int:
    """Batches of the current epoch that had been trained when the checkpoint was written."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    progress = payload["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"]
    return int(progress["processed"])


def _periodic_files(directory: Path) -> list[Path]:
    """Periodic checkpoint files of a directory, oldest step first."""
    return sorted(directory.glob("periodic-*.ckpt"))


class TestPeriodicCheckpoint(unittest.TestCase):
    """Unit and end to end tests for PeriodicCheckpoint."""

    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)

    def test_saves_rotates_and_links_last(self) -> None:
        """Test the save period, the rotation and the last.ckpt link of one epoch."""
        ckpt_dir = self.root / "ckpt"
        callback = PeriodicCheckpoint(dirpath=str(ckpt_dir), saves_per_epoch=4, keep_last=2)
        _trainer(self.root, callback, max_epochs=1).fit(
            _RecordingModule(), datamodule=_DataModule()
        )

        # 10 optimizer steps per epoch and four saves per epoch: every 3 steps plus the epoch end.
        self.assertEqual(callback.every_n_steps, 3)
        files = _periodic_files(ckpt_dir)
        self.assertEqual(
            [path.name for path in files],
            ["periodic-epoch000-step00000009.ckpt", "periodic-epoch000-step00000010.ckpt"],
        )
        self.assertEqual(list(ckpt_dir.glob("*.tmp")), [])
        last = ckpt_dir / "last.ckpt"
        self.assertTrue(last.is_symlink())
        self.assertEqual(last.resolve(), files[-1].resolve())
        self.assertTrue(all(CheckpointLocator.is_loadable(path) for path in files))
        self.assertEqual(_batches_done(files[-1]), 10)
        self.assertEqual(_batches_done(files[0]), 9)

    def test_resume_from_last_continues_the_epoch_with_the_remaining_batches(self) -> None:
        """Test that a killed run resumes mid-epoch with exactly the unseen batches."""
        # Reference: the full first epoch, uninterrupted.
        reference = _RecordingModule()
        _trainer(
            self.root / "ref",
            PeriodicCheckpoint(dirpath=str(self.root / "ref" / "ckpt")),
            max_epochs=1,
        ).fit(reference, datamodule=_DataModule())
        full_epoch = reference.seen
        self.assertEqual(len(full_epoch), 10)

        # Interrupted run: killed while training batch 8 (index 7). No hook gets to save,
        # so the newest state is the periodic file written after optimizer step 6.
        ckpt_dir = self.root / "run" / "ckpt"
        interrupted = _RecordingModule(die_at_batch=7)
        with self.assertRaises(_Preempted):
            _trainer(
                self.root / "run",
                PeriodicCheckpoint(dirpath=str(ckpt_dir), saves_per_epoch=4, keep_last=2),
                max_epochs=1,
            ).fit(interrupted, datamodule=_DataModule())
        self.assertEqual(interrupted.seen, full_epoch[:7])
        latest = CheckpointLocator(ckpt_dir).find_latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.name, "periodic-epoch000-step00000006.ckpt")
        self.assertEqual(_batches_done(latest), 6)

        # Relaunch: a fresh process resumes from last.ckpt and finishes the epoch.
        resumed = _RecordingModule()
        trainer = _trainer(
            self.root / "run",
            PeriodicCheckpoint(dirpath=str(ckpt_dir), saves_per_epoch=4, keep_last=2),
            max_epochs=1,
        )
        trainer.fit(resumed, datamodule=_DataModule(), ckpt_path=str(latest))
        self.assertEqual(resumed.seen, full_epoch[6:])
        self.assertEqual(trainer.global_step, 10)
        self.assertEqual(trainer.current_epoch, 1)
        # The file the run resumed from was rotated out once newer saves existed.
        names = [path.name for path in _periodic_files(ckpt_dir)]
        self.assertEqual(
            names,
            ["periodic-epoch000-step00000009.ckpt", "periodic-epoch000-step00000010.ckpt"],
        )
        self.assertEqual((ckpt_dir / "last.ckpt").resolve().name, names[-1])

    def test_stop_file_saves_a_checkpoint_and_ends_the_run(self) -> None:
        """Test that the stop file makes the run save at the next batch and stop."""
        stop_file = self.root / "job.stop"
        ckpt_dir = self.root / "ckpt"
        callback = PeriodicCheckpoint(
            dirpath=str(ckpt_dir), saves_per_epoch=1, keep_last=2, stop_file=str(stop_file)
        )
        trainer = _trainer(self.root, callback, max_epochs=3)
        trainer.callbacks.append(_StopAfter(stop_file, after_batches=6))
        trainer.fit(_RecordingModule(), datamodule=_DataModule())

        # The file appears after batch 6 (this test callback runs after the checkpoint one),
        # so the run stops at the end of batch 7 with a checkpoint at that step.
        self.assertEqual(trainer.global_step, 7)
        files = _periodic_files(ckpt_dir)
        self.assertEqual([path.name for path in files], ["periodic-epoch000-step00000007.ckpt"])
        self.assertEqual((ckpt_dir / "last.ckpt").resolve(), files[0].resolve())
        self.assertTrue(callback.stop_requested)
        self.assertEqual(_batches_done(files[0]), 7)

    def test_no_batch_end_save_on_the_last_batch_of_an_epoch(self) -> None:
        """Test that the end-of-epoch save replaces a periodic save on the epoch's last batch."""
        # 12 steps an epoch saved every 3: the fourth periodic save would fall on the last
        # batch, before the epoch's validation and scheduler step, and the end-of-epoch save
        # that follows them would be skipped as a duplicate of that step.
        ckpt_dir = self.root / "ckpt"
        callback = PeriodicCheckpoint(dirpath=str(ckpt_dir), saves_per_epoch=4, keep_last=5)
        _trainer(self.root, callback, max_epochs=1, max_steps=12).fit(
            _ScheduledModule(), datamodule=_ValidatedDataModule(size=48, batch_size=4)
        )
        self.assertEqual(callback.every_n_steps, 3)
        saved = torch.load(ckpt_dir / "last.ckpt", map_location="cpu", weights_only=False)
        self.assertEqual(saved["global_step"], 12)
        self.assertEqual(saved["lr_schedulers"][0]["last_epoch"], 1)

        module = _ScheduledModule()
        _trainer(self.root, PeriodicCheckpoint(dirpath=str(ckpt_dir)), max_epochs=2).fit(
            module,
            datamodule=_ValidatedDataModule(size=48, batch_size=4),
            ckpt_path=str(ckpt_dir / "last.ckpt"),
        )
        self.assertEqual(module.epoch_lrs, {1: 0.02})
        self.assertEqual(module.validated_epochs, [1])

    def test_resume_steps_an_epoch_scheduler_the_checkpoint_left_behind(self) -> None:
        """Test that a resumed run trains each epoch at that epoch's scheduled rate."""
        # A checkpoint of epoch 0's end whose epoch scheduler was never stepped, as a save on
        # an epoch's last batch writes: epoch 1 must still train at its own rate.
        ckpt_dir = self.root / "ckpt"
        _trainer(self.root, PeriodicCheckpoint(dirpath=str(ckpt_dir)), max_epochs=1).fit(
            _ScheduledModule(), datamodule=_DataModule()
        )
        checkpoint = torch.load(
            ckpt_dir / "periodic-epoch000-step00000010.ckpt", map_location="cpu", weights_only=False
        )
        (scheduler_state,) = checkpoint["lr_schedulers"]
        self.assertEqual(scheduler_state["last_epoch"], 1)
        scheduler_state["last_epoch"] = 0
        scheduler_state["_last_lr"] = [0.01]
        for group in checkpoint["optimizer_states"][0]["param_groups"]:
            group["lr"] = 0.01
        lagging = self.root / "lagging.ckpt"
        torch.save(checkpoint, lagging)

        module = _ScheduledModule()
        _trainer(self.root, PeriodicCheckpoint(dirpath=str(ckpt_dir)), max_epochs=3).fit(
            module, datamodule=_DataModule(), ckpt_path=str(lagging)
        )
        self.assertEqual(module.epoch_lrs, {1: 0.02, 2: 0.04})

    def test_stop_file_from_the_environment(self) -> None:
        """Test that the stop file defaults to the environment variable."""
        stop_file = str(self.root / "env.stop")
        with mock.patch.dict(os.environ, {PeriodicCheckpoint.STOP_FILE_ENV: stop_file}):
            self.assertEqual(PeriodicCheckpoint(dirpath=str(self.root)).stop_file, stop_file)
        with mock.patch.dict(os.environ):
            os.environ.pop(PeriodicCheckpoint.STOP_FILE_ENV, None)
            self.assertIsNone(PeriodicCheckpoint(dirpath=str(self.root)).stop_file)

    def test_rejects_bad_counts(self) -> None:
        """Test that non-positive counts are rejected by name."""
        for name in ("saves_per_epoch", "keep_last"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                PeriodicCheckpoint(**{name: 0})

    def test_batch_end_before_train_start_raises(self) -> None:
        """Test that a missing save period is reported instead of asserted."""
        callback = PeriodicCheckpoint(dirpath=str(self.root))
        trainer = mock.Mock(global_step=1)
        with mock.patch.object(callback, "_stop_requested", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "save period"):
                callback.on_train_batch_end(trainer, mock.Mock(), None, None, 0)

    def test_state_dict_round_trip(self) -> None:
        """Test that the period is not saved and older checkpoints with it still load."""
        callback = PeriodicCheckpoint(dirpath=str(self.root))
        callback.saved_paths = ["a.ckpt", "b.ckpt"]
        callback.every_n_steps = 3

        state = callback.state_dict()
        self.assertEqual(state, {"saved_paths": ["a.ckpt", "b.ckpt"], "last_saved_step": -1})

        restored = PeriodicCheckpoint(dirpath=str(self.root))
        restored.load_state_dict(
            {"saved_paths": ["a.ckpt", ""], "every_n_steps": 5, "last_saved_step": 12}
        )
        self.assertEqual(restored.saved_paths, ["a.ckpt"])
        self.assertIsNone(restored.every_n_steps)
        self.assertEqual(restored.state_dict()["last_saved_step"], 12)

    def test_link_last_replaces_the_link_in_place(self) -> None:
        """Test that relinking last.ckpt leaves a relative link and no temporary file."""
        callback = PeriodicCheckpoint(dirpath=str(self.root))
        for name in ("first.ckpt", "second.ckpt"):
            (self.root / name).write_bytes(name.encode())
            callback._link_last(str(self.root / name))

        last = self.root / "last.ckpt"
        self.assertTrue(last.is_symlink())
        self.assertEqual(os.readlink(last), "second.ckpt")
        self.assertEqual(list(self.root.glob("*.tmp")), [])
