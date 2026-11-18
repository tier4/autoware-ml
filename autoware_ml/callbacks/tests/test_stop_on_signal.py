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

"""Tests for the stop-on-signal callback: a signalled run saves a checkpoint and stops, and
resuming it trains exactly the batches of an uninterrupted run."""

import os
from pathlib import Path
import signal

import lightning as L
import pytest
import torch
from torch.utils.data import DistributedSampler

from autoware_ml.callbacks.stop_on_signal import StopOnSignal, TrainingStopRequested
from autoware_ml.datamodule.resumable_data_loader import ResumableDataLoader

_DATASET = list(range(24))


class _RecordingModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)
        self.seen: list[tuple[int, list[int]]] = []

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        self.seen.append((self.current_epoch, batch.tolist()))
        return self.layer(batch.float().unsqueeze(-1)).sum()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-3)


class _SignalAt(L.Callback):
    def __init__(self, epoch: int, batch_idx: int) -> None:
        self.epoch, self.batch_idx = epoch, batch_idx

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (trainer.current_epoch, batch_idx) == (self.epoch, self.batch_idx):
            os.kill(os.getpid(), signal.SIGUSR1)


def _fit(
    module: _RecordingModule,
    root_dir: Path,
    callbacks: list[L.Callback],
    ckpt_path: Path | None = None,
) -> None:
    sampler = DistributedSampler(_DATASET, num_replicas=1, rank=0, shuffle=True, seed=0)
    loader = ResumableDataLoader(_DATASET, batch_size=4, sampler=sampler)
    trainer = L.Trainer(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=root_dir,
        callbacks=callbacks,
    )
    trainer.fit(module, train_dataloaders=loader, ckpt_path=ckpt_path)


def test_resumed_run_trains_the_batches_of_an_uninterrupted_run(tmp_path: Path) -> None:
    reference, interrupted, resumed = _RecordingModule(), _RecordingModule(), _RecordingModule()
    _fit(reference, tmp_path / "reference", [StopOnSignal(check_every_n_steps=1)])
    with pytest.raises(TrainingStopRequested) as stop:
        _fit(
            interrupted,
            tmp_path / "run",
            [_SignalAt(epoch=1, batch_idx=2), StopOnSignal(check_every_n_steps=1)],
        )
    checkpoint_path = tmp_path / "run" / "checkpoints" / "stopped.ckpt"
    _fit(resumed, tmp_path / "run", [StopOnSignal(check_every_n_steps=1)], checkpoint_path)

    assert stop.value.checkpoint_path == checkpoint_path
    assert signal.getsignal(signal.SIGUSR1) == signal.SIG_DFL
    assert len(reference.seen) == 3 * 6
    assert interrupted.seen == reference.seen[: 6 + 3]
    assert resumed.seen == reference.seen[6 + 3 :]


@pytest.mark.parametrize("signal_name", ["SIGINT", "SIGTERM"])
def test_rejects_the_signals_lightning_handles(signal_name: str) -> None:
    with pytest.raises(ValueError, match=signal_name):
        StopOnSignal(signal_name=signal_name)
