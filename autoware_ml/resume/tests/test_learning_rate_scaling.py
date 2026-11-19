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

"""End to end tests of the learning-rate scaling of a resume with another global batch."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from torch import nn
from torch.utils.data import Dataset

from autoware_ml.resume.data_loader import (
    ResumableDataLoader,
    ResumableDistributedSampler,
    ResumableTrainLoaderMixin,
)
from autoware_ml.resume.learning_rate_scaling import (
    BatchSizeLearningRateScaling,
    LearningRateScalingRule,
)

_LEARNING_RATE = 0.1


class _IndexDataset(Dataset):
    """Dataset whose samples are their own indices."""

    def __len__(self) -> int:
        return 32

    def __getitem__(self, index: int) -> int:
        return index


class _Preempted(RuntimeError):
    """Stands in for the kill a requeue delivers without any grace period."""


class _RecordingSGD(torch.optim.SGD):
    """SGD that records the learning rate every step trains with."""

    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_learning_rates: list[float] = []

    def step(self, closure=None):
        self.step_learning_rates.append(float(self.param_groups[0]["lr"]))
        return super().step(closure)


class _Module(L.LightningModule):
    """Tiny model with a constant learning rate."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.layer(batch.float().unsqueeze(1)).mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return _RecordingSGD(self.parameters(), lr=_LEARNING_RATE)


class _DataModule(ResumableTrainLoaderMixin, L.LightningDataModule):
    """Index datamodule carrying its loader position through the framework's mixin."""

    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self) -> ResumableDataLoader:
        dataset = _IndexDataset()
        sampler = ResumableDistributedSampler(dataset, shuffle=True, seed=5)
        loader = ResumableDataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.batch_size,
            collate_fn=torch.tensor,
            allow_layout_change=True,
        )
        return self._attach_train_loader(loader)


class _SaveAndDie(L.Callback):
    """Save a checkpoint after a given batch of the first epoch, then die."""

    def __init__(self, path: Path, batch_idx: int) -> None:
        self.path = path
        self.batch_idx = batch_idx

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if (trainer.current_epoch, batch_idx) == (0, self.batch_idx):
            trainer.save_checkpoint(self.path)
            raise _Preempted(f"killed after batch {batch_idx}")


class TestBatchSizeLearningRateScaling(unittest.TestCase):
    """A run resumed with another global batch trains with a scaled learning rate."""

    @classmethod
    def setUpClass(cls) -> None:
        # A plain SGD makes torch wrap SGD.step with the step hooks as well, so the recording
        # subclass, which calls it, runs the hooks nested, as any optimizer subclass may.
        torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=_LEARNING_RATE)

    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.directory = Path(temp_dir.name)

    def _fit(
        self,
        batch_size: int,
        rule: str,
        resume_from: Path | None = None,
        die_after: tuple[Path, int] | None = None,
        accumulate_grad_batches: int = 1,
        allow_layout_change: bool = True,
    ) -> tuple[L.Trainer, _Module]:
        """Fit two epochs with the scaling callback; kill the run when asked."""
        module = _Module()
        callbacks: list[L.Callback] = [
            BatchSizeLearningRateScaling(rule, allow_layout_change=allow_layout_change)
        ]
        if die_after is not None:
            callbacks.append(_SaveAndDie(*die_after))
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=2,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        try:
            trainer.fit(module, datamodule=_DataModule(batch_size), ckpt_path=resume_from)
        except _Preempted:
            pass
        return trainer, module

    @staticmethod
    def _step_rates(trainer: L.Trainer) -> list[float]:
        return trainer.optimizers[0].step_learning_rates

    def test_first_launch_keeps_the_learning_rate(self) -> None:
        """
        Input: a fresh run with batch size 4.
        Expected: every step trains with the configured learning rate.
        """
        trainer, _ = self._fit(4, "sqrt")
        self.assertEqual(set(self._step_rates(trainer)), {_LEARNING_RATE})

    def test_resume_with_a_larger_batch_scales_every_step(self) -> None:
        """
        Input: launched with batch size 4, resumed with batch size 8 under each rule.
        Expected: the resumed steps train with 0.1 times sqrt(2), 2 or 1, while the param
        group and the next checkpoint hold the tuned 0.1.
        """
        expected = {"sqrt": math.sqrt(2.0), "linear": 2.0, "none": 1.0}
        for rule, factor in expected.items():
            with self.subTest(rule=rule):
                checkpoint = self.directory / f"{rule}.ckpt"
                self._fit(4, rule, die_after=(checkpoint, 2))
                trainer, _ = self._fit(8, rule, resume_from=checkpoint)

                rates = self._step_rates(trainer)
                self.assertEqual(len(rates), 4 + 4 - 1)
                for rate in rates:
                    self.assertAlmostEqual(rate, _LEARNING_RATE * factor)
                self.assertEqual(trainer.optimizers[0].param_groups[0]["lr"], _LEARNING_RATE)

    def test_accumulation_keeping_the_global_batch_keeps_the_learning_rate(self) -> None:
        """
        Input: launched with batch size 8, resumed with batch size 4 and accumulation 2.
        Expected: the global batch is unchanged, so the learning rate is too.
        """
        checkpoint = self.directory / "killed.ckpt"
        self._fit(8, "linear", die_after=(checkpoint, 1))
        trainer, _ = self._fit(4, "linear", resume_from=checkpoint, accumulate_grad_batches=2)
        self.assertEqual(set(self._step_rates(trainer)), {_LEARNING_RATE})

    def test_relaunches_do_not_compound(self) -> None:
        """
        Input: launched with batch size 4, resumed with batch size 8, killed and resumed
        again with batch size 8.
        Expected: the second relaunch still scales against the first launch's batch size
        (factor 2), not against the previous relaunch (factor 1).
        """
        first = self.directory / "first.ckpt"
        second = self.directory / "second.ckpt"
        self._fit(4, "linear", die_after=(first, 2))
        self._fit(8, "linear", resume_from=first, die_after=(second, 2))

        state = torch.load(second, map_location="cpu", weights_only=False)
        self.assertEqual(state["optimizer_states"][0]["param_groups"][0]["lr"], _LEARNING_RATE)

        trainer, _ = self._fit(8, "linear", resume_from=second)
        self.assertEqual(set(self._step_rates(trainer)), {_LEARNING_RATE * 2.0})

    def test_another_global_batch_is_refused_by_default(self) -> None:
        """
        Input: launched with batch size 8, resumed with batch size 4 and accumulation 1 by a
        callback that does not allow a layout change.
        Expected: the resume fails before training, naming the option.
        """
        checkpoint = self.directory / "killed.ckpt"
        self._fit(8, "sqrt", die_after=(checkpoint, 1))
        with self.assertRaisesRegex(RuntimeError, "allow_resume_layout_change"):
            self._fit(4, "sqrt", resume_from=checkpoint, allow_layout_change=False)

    def test_rule_factors(self) -> None:
        """Test the factor of every rule for a halved global batch."""
        self.assertAlmostEqual(LearningRateScalingRule.SQRT.factor(0.5), math.sqrt(0.5))
        self.assertEqual(LearningRateScalingRule.LINEAR.factor(0.5), 0.5)
        self.assertEqual(LearningRateScalingRule.NONE.factor(0.5), 1.0)
        with self.assertRaises(ValueError):
            LearningRateScalingRule("cubic")


if __name__ == "__main__":
    unittest.main()
