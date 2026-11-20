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

"""Unit tests for the trainer assembly of the train stage: ``fit_model`` and ``optimized_score``."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from jaxtyping import Float32
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from lightning.pytorch.demos.boring_classes import BoringDataModule, BoringModel
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import torch

from autoware_ml.scripts.cli.train import fit_model, optimized_score
from autoware_ml.utils.mlflow_helpers import RUN_METADATA_FILENAME, MlflowRunContext

CONFIG_NAME = "detection3d/tiny/tiny_b1_1e_corpus"


class _LoggingModel(BoringModel):
    """A Lightning demo model that logs ``train/loss`` and ``val/loss`` like ``ModuleBaseModel``."""

    def training_step(
        self, batch: Float32[torch.Tensor, "batch_size 32"], batch_idx: int
    ) -> dict[str, Float32[torch.Tensor, ""]]:
        output = super().training_step(batch, batch_idx)
        self.log("train/loss", output["loss"])
        return output

    def validation_step(
        self, batch: Float32[torch.Tensor, "batch_size 32"], batch_idx: int
    ) -> dict[str, Float32[torch.Tensor, ""]]:
        output = super().validation_step(batch, batch_idx)
        self.log("val/loss", output["x"])
        return output


class _EpochRecorder(L.Callback):
    """Records the epochs the trainer starts, to see where a fit began."""

    def __init__(self) -> None:
        self.started_epochs: list[int] = []

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.started_epochs.append(trainer.current_epoch)


def _cfg(run_dir: Path, configured_checkpoint_dir: Path, max_epochs: int) -> DictConfig:
    """A train config with a CPU trainer, a checkpoint callback and a learning rate monitor."""
    return OmegaConf.create(
        {
            "experiment_run_dir": str(run_dir),
            "trainer": {
                "_target_": "lightning.Trainer",
                "accelerator": "cpu",
                "devices": 1,
                "max_epochs": max_epochs,
                "limit_train_batches": 2,
                "limit_val_batches": 1,
                "num_sanity_val_steps": 0,
                "log_every_n_steps": 1,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
            "callbacks": {
                "checkpoint": {
                    "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "dirpath": str(configured_checkpoint_dir),
                    "save_last": True,
                },
                "lr_monitor": {"_target_": "lightning.pytorch.callbacks.LearningRateMonitor"},
            },
        }
    )


class TestFitModel(unittest.TestCase):
    """Unit tests for ``fit_model``, the trainer assembly every training stage shares."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_dir = self.root / "run"
        self.configured_checkpoint_dir = self.root / "configured_checkpoints"
        self.checkpoint_dir = self.root / "checkpoints"
        torch.manual_seed(0)

    def test_without_a_run_context_the_trainer_has_no_logger(self) -> None:
        """Test that a fit without a tracker runs without a logger and the callbacks that need one.

        Input: the train config, no run context, ``checkpoint_dir`` overriding the configured one.
        Expected: no logger, no ``LearningRateMonitor``, the trainer rooted at
            ``experiment_run_dir``, and ``ModelCheckpoint`` writing ``last.ckpt`` to
            ``checkpoint_dir`` instead of its configured directory.
        Check: the trainer's logger, callbacks and root directory, and the checkpoint files.
        """
        trainer = fit_model(
            _cfg(self.run_dir, self.configured_checkpoint_dir, max_epochs=1),
            _LoggingModel(),
            BoringDataModule(),
            run_context=None,
            config_name=CONFIG_NAME,
            checkpoint_dir=self.checkpoint_dir,
        )

        self.assertIsNone(trainer.logger)
        self.assertFalse(any(isinstance(cb, LearningRateMonitor) for cb in trainer.callbacks))
        self.assertEqual(Path(trainer.default_root_dir), self.run_dir)
        (checkpoint,) = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
        self.assertEqual(Path(checkpoint.dirpath), self.checkpoint_dir)
        self.assertTrue((self.checkpoint_dir / "last.ckpt").is_file())
        self.assertFalse(self.configured_checkpoint_dir.exists())
        self.assertTrue(trainer.state.finished)

    def test_extra_callbacks_follow_the_configured_ones(self) -> None:
        """Test that the stage's extra callbacks are attached after the configured ones and run.

        Input: a configured ``Timer`` and one extra callback that records the epochs the trainer
            starts, a one-epoch fit.
        Expected: the extra callback comes after the ``Timer`` and saw epoch 0.
        Check: the order of the two callbacks in ``trainer.callbacks`` and the recorded epochs.
        """
        cfg = _cfg(self.run_dir, self.configured_checkpoint_dir, max_epochs=1)
        # Not ModelCheckpoint: Lightning moves every ModelCheckpoint to the end of the list.
        cfg.callbacks.timer = {"_target_": "lightning.pytorch.callbacks.Timer"}
        recorder = _EpochRecorder()
        trainer = fit_model(
            cfg,
            _LoggingModel(),
            BoringDataModule(),
            run_context=None,
            config_name=CONFIG_NAME,
            extra_callbacks=[recorder],
        )

        callbacks = list(trainer.callbacks)
        (timer,) = [cb for cb in callbacks if isinstance(cb, Timer)]
        self.assertGreater(callbacks.index(recorder), callbacks.index(timer))
        self.assertEqual(recorder.started_epochs, [0])

    def test_a_resumed_fit_starts_at_the_epoch_after_the_checkpoint(self) -> None:
        """Test that ``resume_checkpoint_path`` reaches ``Trainer.fit``.

        Input: a one-epoch fit that writes ``last.ckpt``, then a two-epoch fit resumed from it.
        Expected: the resumed fit starts at epoch 1 and ends after epoch 1.
        Check: the epochs the resumed fit started and its ``current_epoch``.
        """
        fit_model(
            _cfg(self.run_dir, self.configured_checkpoint_dir, max_epochs=1),
            _LoggingModel(),
            BoringDataModule(),
            run_context=None,
            config_name=CONFIG_NAME,
            checkpoint_dir=self.checkpoint_dir,
        )
        recorder = _EpochRecorder()
        trainer = fit_model(
            _cfg(self.run_dir, self.configured_checkpoint_dir, max_epochs=2),
            _LoggingModel(),
            BoringDataModule(),
            run_context=None,
            config_name=CONFIG_NAME,
            checkpoint_dir=self.checkpoint_dir,
            extra_callbacks=[recorder],
            resume_checkpoint_path=str(self.checkpoint_dir / "last.ckpt"),
        )

        self.assertEqual(recorder.started_epochs, [1])
        self.assertEqual(trainer.current_epoch, 2)

    def test_with_a_run_context_the_trainer_logs_to_the_run_under_the_stage(self) -> None:
        """Test that a fit with a tracker logs to the run and records the stage it fits for.

        Input: an MLflow run in a local SQLite store, its run context, ``stage="quantize"``.
        Expected: an ``MLFlowLogger`` bound to the run, a kept ``LearningRateMonitor``, the
            trainer rooted at the run's artifact directory, ``quantize`` in the run metadata,
            and ``train/loss`` logged to the run.
        Check: the trainer's logger, callbacks and root directory, the metadata file, and
            the run's metrics in the store.
        """
        work_dir = self.root / "hydra_run"
        work_dir.mkdir()
        hydra_cfg = OmegaConf.structured(HydraConf)
        hydra_cfg.runtime.output_dir = str(work_dir)
        hydra_cfg.job.config_name = f"tasks/{CONFIG_NAME}"
        HydraConfig.instance().set_config(OmegaConf.create({"hydra": hydra_cfg}))
        self.addCleanup(setattr, HydraConfig.instance(), "cfg", None)

        tracking_uri = f"sqlite:///{self.root / 'mlflow.db'}"
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_id = client.create_experiment("fit_model")
        run_id = client.create_run(experiment_id, run_name="quantize_tiny").info.run_id
        artifact_dir = self.root / "artifacts"
        artifact_dir.mkdir()
        run_context = MlflowRunContext(
            tracking_uri=tracking_uri,
            experiment_id=experiment_id,
            experiment_name="fit_model",
            run_id=run_id,
            run_name="quantize_tiny",
            tags={"stage": "quantize"},
            artifact_uri=artifact_dir.as_uri(),
            artifact_dir=artifact_dir,
            hydra_dir=work_dir,
        )
        cfg = _cfg(self.run_dir, self.configured_checkpoint_dir, max_epochs=1)
        cfg.logger = {
            "_target_": "lightning.pytorch.loggers.MLFlowLogger",
            "tracking_uri": tracking_uri,
        }

        trainer = fit_model(
            cfg,
            _LoggingModel(),
            BoringDataModule(),
            run_context=run_context,
            config_name=CONFIG_NAME,
            stage="quantize",
        )

        self.assertIsInstance(trainer.logger, MLFlowLogger)
        self.assertEqual(trainer.logger.run_id, run_id)
        self.assertTrue(any(isinstance(cb, LearningRateMonitor) for cb in trainer.callbacks))
        self.assertEqual(Path(trainer.default_root_dir), artifact_dir)
        metadata = json.loads((artifact_dir / RUN_METADATA_FILENAME).read_text())
        self.assertEqual(metadata["stage"], "quantize")
        self.assertIn("train/loss", client.get_run(run_id).data.metrics)


class TestOptimizedScore(unittest.TestCase):
    """Unit tests for ``optimized_score``, the value a hyperparameter sweep minimizes."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        torch.manual_seed(0)
        self.cfg = _cfg(root / "run", root / "checkpoints", max_epochs=1)
        self.trainer = fit_model(
            self.cfg, _LoggingModel(), BoringDataModule(), run_context=None, config_name=CONFIG_NAME
        )

    def test_reads_val_loss_unless_another_metric_is_configured(self) -> None:
        """Test that the score is ``val/loss`` by default and the configured metric otherwise.

        Input: a fitted trainer that logged ``train/loss`` and ``val/loss``; the config without
            ``optimized_metric``, then with ``optimized_metric: train/loss``.
        Expected: the logged value of ``val/loss``, then of ``train/loss``, as floats.
        Check: both returned values against ``trainer.callback_metrics``.
        """
        metrics = self.trainer.callback_metrics
        self.assertEqual(optimized_score(self.trainer, self.cfg), float(metrics["val/loss"]))
        self.cfg.optimized_metric = "train/loss"
        self.assertEqual(optimized_score(self.trainer, self.cfg), float(metrics["train/loss"]))

    def test_a_metric_that_was_not_logged_is_named_with_the_logged_ones(self) -> None:
        """Test that a configured metric the run never logged is refused, naming what was logged.

        Input: the fitted trainer and ``optimized_metric: val/mAP``.
        Expected: ValueError naming ``val/mAP`` and the logged ``['train/loss', 'val/loss']``.
        Check: the error message.
        """
        self.cfg.optimized_metric = "val/mAP"
        with self.assertRaisesRegex(
            ValueError, r"'val/mAP' was not logged.*\['train/loss', 'val/loss'\]"
        ):
            optimized_score(self.trainer, self.cfg)


if __name__ == "__main__":
    unittest.main()
