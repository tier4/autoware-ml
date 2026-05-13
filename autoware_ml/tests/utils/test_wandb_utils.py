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

"""Tests for W&B tracking helpers."""

from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from autoware_ml.utils.wandb_helpers import (
    build_dataset_metadata,
    build_run_metadata,
    build_wandb_tags,
    configure_logger,
    prepare_run_context,
    write_run_config_artifacts,
    write_run_metadata,
)

SAMPLE_CONFIG_NAME = "calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2"


class TestWandbRunContext:
    """Tests for local W&B run context preparation."""

    def test_prepare_run_context_uses_stable_local_layout(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        logger_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.loggers.WandbLogger",
                "project": "autoware-ml-phase2",
                "entity": "tier4",
                "save_dir": "wandb_runs",
                "id": "abc123",
            }
        )

        context = prepare_run_context(
            logger_cfg,
            SAMPLE_CONFIG_NAME,
            hydra_dir=None,
            stage="train",
            started_at=datetime(2026, 5, 6, 10, 0, 0),
        )

        expected_root = tmp_path / "wandb_runs" / SAMPLE_CONFIG_NAME / "abc123"
        assert context.run_id == "abc123"
        assert context.run_name == "train:resnet18_t4dataset_j6gen2:2026-05-06/10-00-00"
        assert context.hydra_dir == expected_root / "hydra"
        assert context.artifact_dir == expected_root / "artifacts"
        assert context.checkpoints_dir == expected_root / "artifacts" / "checkpoints"
        assert context.tags["tracking_backend"] == "wandb"

    def test_configure_logger_sets_wandb_init_fields(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        logger_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.loggers.WandbLogger",
                "project": "autoware-ml-phase2",
                "entity": None,
                "save_dir": "wandb_runs",
                "tags": ["manual"],
            }
        )
        context = prepare_run_context(
            logger_cfg,
            SAMPLE_CONFIG_NAME,
            hydra_dir=tmp_path / "hydra",
            stage="test",
            started_at=datetime(2026, 5, 6, 10, 0, 0),
        )

        configure_logger(logger_cfg, context)

        assert logger_cfg.project == "autoware-ml-phase2"
        assert logger_cfg.name == context.run_name
        assert logger_cfg.version == context.run_id
        assert logger_cfg.id == context.run_id
        assert logger_cfg.group == SAMPLE_CONFIG_NAME
        assert logger_cfg.job_type == "test"
        assert "manual" in logger_cfg.tags
        assert "tracking_backend:wandb" in logger_cfg.tags
        assert all(len(tag) <= 64 for tag in logger_cfg.tags)

    def test_build_wandb_tags_excludes_long_metadata_values(self) -> None:
        tags = build_wandb_tags(
            {
                "tracking_backend": "wandb",
                "stage": "train",
                "artifact_dir": "/tmp/" + ("very-long-path/" * 10),
                "hydra_dir": "/tmp/" + ("very-long-path/" * 10),
            }
        )

        assert "tracking_backend:wandb" in tags
        assert "stage:train" in tags
        assert all("artifact_dir" not in tag for tag in tags)
        assert all("hydra_dir" not in tag for tag in tags)
        assert all(len(tag) <= 64 for tag in tags)


class TestWandbArtifacts:
    """Tests for W&B metadata artifacts."""

    def test_write_run_config_artifacts_includes_dataset_metadata(self, tmp_path: Path) -> None:
        cfg = OmegaConf.create(
            {
                "datamodule": {
                    "_target_": "autoware_ml.datamodule.coco.COCODetectionDataModule",
                    "data_root": "/data",
                    "train_ann_file": "train.json",
                    "val_ann_file": "val.json",
                    "train_img_root": "images/train",
                    "val_img_root": "images/val",
                    "max_train_samples": 512,
                    "ignored": "nope",
                }
            }
        )

        metadata = build_dataset_metadata(cfg)

        assert metadata == {
            "datamodule_target": "autoware_ml.datamodule.coco.COCODetectionDataModule",
            "data_root": "/data",
            "train_ann_file": "train.json",
            "val_ann_file": "val.json",
            "train_img_root": "images/train",
            "val_img_root": "images/val",
            "max_train_samples": 512,
        }

    def test_write_run_metadata_records_tracking_backend(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        logger_cfg = OmegaConf.create(
            {
                "_target_": "lightning.pytorch.loggers.WandbLogger",
                "project": "autoware-ml-phase2",
                "save_dir": "wandb_runs",
                "id": "abc123",
            }
        )
        context = prepare_run_context(
            logger_cfg,
            SAMPLE_CONFIG_NAME,
            hydra_dir=tmp_path / "hydra",
            stage="train",
            started_at=datetime(2026, 5, 6, 10, 0, 0),
        )

        metadata_path = write_run_metadata(
            context.artifact_dir,
            build_run_metadata(context, SAMPLE_CONFIG_NAME, context.hydra_dir, "train"),
        )

        assert metadata_path.read_text(encoding="utf-8").count('"tracking_backend": "wandb"') == 1
