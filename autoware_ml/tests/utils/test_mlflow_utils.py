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

"""Tests for shared MLflow helpers."""

from pathlib import Path

from omegaconf import OmegaConf

from autoware_ml.utils.mlflow import (
    build_run_tags,
    configure_logger,
    generate_run_name,
    load_run_metadata,
    should_enable_logger,
    write_run_metadata,
)

SAMPLE_CONFIG_NAME = "calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2"
SAMPLE_EXPERIMENT_NAME = (
    "calibration_status_calibration_status_classifier_resnet18_t4dataset_j6gen2"
)


class TestRunNaming:
    """Tests for MLflow run naming."""

    def test_generate_run_name(self, tmp_path: Path) -> None:
        work_dir = (
            tmp_path
            / "calibration_status"
            / "calibration_status_classifier"
            / "resnet18_t4dataset_j6gen2"
            / "2026-03-17"
            / "09-00-00"
        )
        work_dir.mkdir(parents=True)

        run_name = generate_run_name(
            SAMPLE_CONFIG_NAME,
            work_dir,
            "train",
        )

        assert run_name == "train:resnet18_t4dataset_j6gen2:2026-03-17/09-00-00"


class TestRunMetadata:
    """Tests for persisted run metadata."""

    def test_write_and_load_run_metadata(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "mlruns" / "task" / "model" / "config" / "2026-03-17" / "09-00-00"
        checkpoints_dir = work_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "best.ckpt"
        checkpoint_path.write_text("", encoding="utf-8")

        metadata = {"run_id": "abc123", "experiment_name": "task_model_config"}
        write_run_metadata(work_dir, metadata)

        assert load_run_metadata(checkpoint_path) == metadata


class TestRunTags:
    """Tests for run tag generation."""

    def test_build_run_tags(self, tmp_path: Path) -> None:
        work_dir = (
            tmp_path
            / "mlruns"
            / "calibration_status"
            / "calibration_status_classifier"
            / "resnet18_t4dataset_j6gen2"
            / "2026-03-17"
            / "09-00-00"
        )
        work_dir.mkdir(parents=True)

        tags = build_run_tags(
            SAMPLE_CONFIG_NAME,
            work_dir,
            "deploy",
            extra_tags={"checkpoint_path": "/tmp/model.ckpt"},
        )

        assert tags["config_name"] == SAMPLE_CONFIG_NAME
        assert tags["task"] == "calibration_status"
        assert tags["model"] == "calibration_status_classifier"
        assert tags["config_variant"] == "resnet18_t4dataset_j6gen2"
        assert tags["stage"] == "deploy"
        assert tags["run_dir"] == str(work_dir)
        assert tags["checkpoint_path"] == "/tmp/model.ckpt"
        assert "hostname" in tags
        assert "git_sha" in tags

    def test_configure_logger_merges_custom_tags(self) -> None:
        logger_cfg = OmegaConf.create(
            {"tracking_uri": "sqlite:///mlruns/mlflow.db", "tags": {"owner": "alice"}}
        )

        configure_logger(
            logger_cfg,
            experiment_name=SAMPLE_EXPERIMENT_NAME,
            run_name="train:resnet18_t4dataset_j6gen2:2026-03-17/09-00-00",
            tags={"stage": "train"},
        )

        assert logger_cfg.experiment_name == SAMPLE_EXPERIMENT_NAME
        assert logger_cfg.run_name == "train:resnet18_t4dataset_j6gen2:2026-03-17/09-00-00"
        assert logger_cfg.tags == {"stage": "train", "owner": "alice"}

    def test_should_enable_logger_requires_logger_and_non_fast_dev_run(self) -> None:
        cfg = OmegaConf.create(
            {
                "logger": {"tracking_uri": "sqlite:///mlruns/mlflow.db"},
                "trainer": {"fast_dev_run": False},
            }
        )
        assert should_enable_logger(cfg) is True

        cfg.trainer.fast_dev_run = True
        assert should_enable_logger(cfg) is False

        cfg = OmegaConf.create({"logger": None, "trainer": {"fast_dev_run": False}})
        assert should_enable_logger(cfg) is False
