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

from pathlib import Path
from typing import Any

import lightning as L
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from autoware_ml.callbacks.periodic_checkpoint import PeriodicCheckpoint
from autoware_ml.datamodule.resumable import ResumableDataLoader, ResumableDistributedSampler
from autoware_ml.plugins.atomic_checkpoint_io import AtomicCheckpointIO
from autoware_ml.utils.checkpoints import (
    checkpoint_is_loadable,
    find_latest_checkpoint,
)


class _IndexDataset(Dataset):
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
        assert self.trainer.estimated_stepping_batches > 0
        return torch.optim.SGD(self.parameters(), lr=0.01)


class _DataModule(L.LightningDataModule):
    """Mirror of the framework datamodule's resumable loader handling."""

    def __init__(self, size: int, batch_size: int) -> None:
        super().__init__()
        self.size = size
        self.batch_size = batch_size
        self._loader: ResumableDataLoader | None = None
        self._loader_state: dict[str, int] | None = None

    def train_dataloader(self) -> ResumableDataLoader:
        dataset = _IndexDataset(self.size)
        sampler = ResumableDistributedSampler(dataset, shuffle=True, seed=11)
        loader = ResumableDataLoader(
            dataset, sampler=sampler, batch_size=self.batch_size, collate_fn=torch.tensor
        )
        if self._loader_state is not None:
            loader.load_state_dict(self._loader_state)
        self._loader = loader
        return loader

    def state_dict(self) -> dict[str, Any]:
        return {"train_dataloader": self._loader.state_dict()} if self._loader else {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._loader_state = state_dict.get("train_dataloader")


def _trainer(root: Path, callback: PeriodicCheckpoint, **kwargs: Any) -> L.Trainer:
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
    return sorted(directory.glob("periodic-*.ckpt"))


def test_periodic_checkpoint_saves_rotates_and_links_last(tmp_path: Path) -> None:
    callback = PeriodicCheckpoint(dirpath=str(tmp_path / "ckpt"), saves_per_epoch=4, keep_last=2)
    trainer = _trainer(tmp_path, callback, max_epochs=1)
    trainer.fit(_RecordingModule(), datamodule=_DataModule(size=40, batch_size=4))

    # 10 optimizer steps per epoch and four saves per epoch: every 3 steps plus the epoch end.
    assert callback.every_n_steps == 3
    files = _periodic_files(tmp_path / "ckpt")
    assert [path.name for path in files] == [
        "periodic-epoch000-step00000009.ckpt",
        "periodic-epoch000-step00000010.ckpt",
    ]
    assert not list((tmp_path / "ckpt").glob("*.tmp"))
    last = tmp_path / "ckpt" / "last.ckpt"
    assert last.is_symlink()
    assert last.resolve() == files[-1].resolve()
    assert all(checkpoint_is_loadable(path) for path in files)
    assert _batches_done(files[-1]) == 10
    assert _batches_done(files[0]) == 9


def test_resume_from_last_continues_the_epoch_with_the_remaining_batches(tmp_path: Path) -> None:
    # Reference: the full first epoch, uninterrupted.
    reference = _RecordingModule()
    _trainer(
        tmp_path / "ref", PeriodicCheckpoint(dirpath=str(tmp_path / "ref" / "ckpt")), max_epochs=1
    ).fit(reference, datamodule=_DataModule(size=40, batch_size=4))
    full_epoch = reference.seen
    assert len(full_epoch) == 10

    # Interrupted run: killed while training batch 8 (index 7). No hook gets to save,
    # so the newest state is the periodic file written after optimizer step 6.
    ckpt_dir = tmp_path / "run" / "ckpt"
    interrupted = _RecordingModule(die_at_batch=7)
    with pytest.raises(_Preempted):
        _trainer(
            tmp_path / "run",
            PeriodicCheckpoint(dirpath=str(ckpt_dir), saves_per_epoch=4, keep_last=2),
            max_epochs=1,
        ).fit(interrupted, datamodule=_DataModule(size=40, batch_size=4))
    assert interrupted.seen == full_epoch[:7]
    latest = find_latest_checkpoint(ckpt_dir)
    assert latest is not None and latest.name == "periodic-epoch000-step00000006.ckpt"
    assert _batches_done(latest) == 6

    # Relaunch: a fresh process resumes from last.ckpt and finishes the epoch.
    resumed = _RecordingModule()
    trainer = _trainer(
        tmp_path / "run",
        PeriodicCheckpoint(dirpath=str(ckpt_dir), saves_per_epoch=4, keep_last=2),
        max_epochs=1,
    )
    trainer.fit(resumed, datamodule=_DataModule(size=40, batch_size=4), ckpt_path=str(latest))
    assert resumed.seen == full_epoch[6:]
    assert trainer.global_step == 10
    assert trainer.current_epoch == 1
    # The file the run resumed from was rotated out once newer saves existed.
    names = [path.name for path in _periodic_files(ckpt_dir)]
    assert names == ["periodic-epoch000-step00000009.ckpt", "periodic-epoch000-step00000010.ckpt"]
    assert (ckpt_dir / "last.ckpt").resolve().name == names[-1]


def test_find_latest_checkpoint_skips_truncated_files(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path / "missing") is None
    good = tmp_path / "good.ckpt"
    torch.save({"state_dict": {}, "optimizer_states": [], "loops": {}}, good)
    (tmp_path / "last.ckpt").write_bytes(b"truncated by a kill")
    newer_but_broken = tmp_path / "newer.ckpt"
    torch.save({"state_dict": {}}, newer_but_broken)
    assert find_latest_checkpoint(tmp_path) == good.resolve()


def test_atomic_checkpoint_io_leaves_no_temporary_file(tmp_path: Path) -> None:
    io = AtomicCheckpointIO()
    target = tmp_path / "state.ckpt"
    io.save_checkpoint({"value": 1}, target)
    assert target.exists()
    assert not (tmp_path / "state.ckpt.tmp").exists()
    assert torch.load(target, weights_only=False)["value"] == 1
    io.remove_checkpoint(target)
    assert not target.exists()


def test_periodic_checkpoint_rejects_bad_counts() -> None:
    with pytest.raises(ValueError, match="saves_per_epoch"):
        PeriodicCheckpoint(saves_per_epoch=0)
    with pytest.raises(ValueError, match="keep_last"):
        PeriodicCheckpoint(keep_last=0)


class _StopAfter(L.Callback):
    """Create the stop file after a given number of batches, like the launcher does."""

    def __init__(self, stop_file: Path, after_batches: int) -> None:
        self.stop_file = stop_file
        self.after_batches = after_batches

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if batch_idx + 1 == self.after_batches:
            self.stop_file.write_text("stop\n")


def test_stop_file_saves_a_checkpoint_and_ends_the_run(tmp_path: Path) -> None:
    stop_file = tmp_path / "job.stop"
    callback = PeriodicCheckpoint(
        dirpath=str(tmp_path / "ckpt"), saves_per_epoch=1, keep_last=2, stop_file=str(stop_file)
    )
    trainer = _trainer(tmp_path, callback, max_epochs=3)
    trainer.callbacks.append(_StopAfter(stop_file, after_batches=6))
    trainer.fit(_RecordingModule(), datamodule=_DataModule(size=40, batch_size=4))

    # The file appears after batch 6 (this test callback runs after the checkpoint one), so
    # the run stops at the end of batch 7 with a checkpoint at that step.
    assert trainer.global_step == 7
    files = _periodic_files(tmp_path / "ckpt")
    assert [path.name for path in files] == ["periodic-epoch000-step00000007.ckpt"]
    assert (tmp_path / "ckpt" / "last.ckpt").resolve() == files[0].resolve()
    assert callback.stop_requested
    assert _batches_done(files[0]) == 7


def test_stop_file_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOWARE_ML_STOP_FILE", str(tmp_path / "env.stop"))
    callback = PeriodicCheckpoint(dirpath=str(tmp_path / "ckpt"))
    assert callback.stop_file == str(tmp_path / "env.stop")
    monkeypatch.delenv("AUTOWARE_ML_STOP_FILE")
    assert PeriodicCheckpoint(dirpath=str(tmp_path / "ckpt")).stop_file is None
