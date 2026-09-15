"""Tests for checkpoint-resume support: rolling last checkpoint, config-authoritative
callback state, and append-only MLflow param logging."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from autoware_ml.callbacks.early_stopping import EarlyStopping
from autoware_ml.utils.runtime import instantiate_callbacks, log_hyperparameters


def _early_stopping_state(patience: int, wait_count: int) -> dict:
    return {
        "wait_count": wait_count,
        "stopped_epoch": 0,
        "best_score": torch.tensor(1.0),
        "patience": patience,
    }


class TestConfigAuthoritativeEarlyStopping:
    def test_configured_patience_wins_over_checkpoint(self, caplog) -> None:
        callback = EarlyStopping(monitor="val/loss", patience=40, mode="min")

        with caplog.at_level("WARNING"):
            callback.load_state_dict(_early_stopping_state(patience=15, wait_count=7))

        assert callback.patience == 40
        assert "patience" in caplog.text

    def test_runtime_state_is_restored(self) -> None:
        callback = EarlyStopping(monitor="val/loss", patience=40, mode="min")

        callback.load_state_dict(_early_stopping_state(patience=15, wait_count=7))

        assert callback.wait_count == 7
        assert float(callback.best_score) == 1.0

    def test_identical_configuration_restores_silently(self, caplog) -> None:
        callback = EarlyStopping(monitor="val/loss", patience=15, mode="min")

        with caplog.at_level("WARNING"):
            callback.load_state_dict(_early_stopping_state(patience=15, wait_count=3))

        assert callback.patience == 15
        assert caplog.text == ""


class TestResumedParamLogging:
    def _logger_with_params(self, params: dict[str, str]) -> MagicMock:
        trainer_logger = MagicMock(spec=MLFlowLogger)
        trainer_logger.run_id = "run-1"
        trainer_logger.experiment.get_run.return_value.data.params = params
        return trainer_logger

    def test_fresh_run_logs_all_params(self) -> None:
        trainer_logger = self._logger_with_params({})
        cfg = OmegaConf.create({"trainer": {"max_epochs": 50}})

        log_hyperparameters(cfg, trainer_logger)

        trainer_logger.log_hyperparams.assert_called_once_with({"trainer": {"max_epochs": 50}})

    def test_resumed_run_logs_only_new_keys_and_tags_drift(self, caplog) -> None:
        trainer_logger = self._logger_with_params(
            {"callbacks/early_stopping/patience": "15", "trainer/max_epochs": "50"}
        )
        cfg = OmegaConf.create(
            {
                "callbacks": {"early_stopping": {"patience": 40}},
                "trainer": {"max_epochs": 50},
                "seed": 42,
            }
        )

        with caplog.at_level("WARNING"):
            log_hyperparameters(cfg, trainer_logger)

        trainer_logger.log_hyperparams.assert_called_once_with({"seed": 42})
        assert "callbacks/early_stopping/patience: 15 -> 40" in caplog.text
        trainer_logger.experiment.set_tag.assert_called_once_with(
            "run-1", "param_drift", "callbacks/early_stopping/patience: 15 -> 40"
        )

    def test_resumed_run_without_changes_logs_nothing(self) -> None:
        trainer_logger = self._logger_with_params({"trainer/max_epochs": "50"})
        cfg = OmegaConf.create({"trainer": {"max_epochs": 50}})

        log_hyperparameters(cfg, trainer_logger)

        trainer_logger.log_hyperparams.assert_called_once_with({})
        trainer_logger.experiment.set_tag.assert_not_called()

    def test_non_zero_rank_skips_param_reconciliation(self, monkeypatch) -> None:
        # Off rank zero the MLflow experiment property returns a dummy whose
        # get_run() yields None; touching it would crash every DDP launch.
        monkeypatch.setattr(rank_zero_only, "rank", 1)
        trainer_logger = self._logger_with_params({"trainer/max_epochs": "50"})
        cfg = OmegaConf.create({"trainer": {"max_epochs": 50}})

        log_hyperparameters(cfg, trainer_logger)

        trainer_logger.experiment.get_run.assert_not_called()
        trainer_logger.log_hyperparams.assert_called_once_with({"trainer": {"max_epochs": 50}})


class TestDefaultCheckpointCallbacks:
    def test_rolling_last_checkpoint_saves_unconditionally(self, tmp_path) -> None:
        cfg = OmegaConf.create(
            {
                "callbacks": {
                    "model_checkpoint": {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "monitor": "val/loss",
                        "dirpath": str(tmp_path),
                        "filename": "best",
                        "save_top_k": 1,
                        "mode": "min",
                    },
                    "model_checkpoint_last": {
                        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
                        "dirpath": str(tmp_path),
                        "filename": "last",
                        "save_top_k": 1,
                        "enable_version_counter": False,
                    },
                }
            }
        )

        callbacks = instantiate_callbacks(cfg, checkpoint_dir=tmp_path / "checkpoints")

        assert len(callbacks) == 2
        best_callback, last_callback = callbacks
        assert best_callback.monitor == "val/loss"
        assert last_callback.monitor is None
        assert all(cb.dirpath == str(tmp_path / "checkpoints") for cb in callbacks)
