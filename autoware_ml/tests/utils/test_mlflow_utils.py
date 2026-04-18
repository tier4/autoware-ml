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

from datetime import datetime
from pathlib import Path

from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

from autoware_ml.utils.mlflow_helpers import (
    build_run_tags,
    configure_logger,
    generate_artifact_location,
    generate_hydra_run_dir,
    generate_run_name,
    load_run_metadata,
    load_run_context,
    prepare_run_context,
    resolve_tracking_uri,
    should_enable_logger,
    write_run_metadata,
)

SAMPLE_CONFIG_NAME = "calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2"
SAMPLE_EXPERIMENT_NAME = (
    "calibration_status_calibration_status_classifier_resnet18_t4dataset_j6gen2"
)


class TestRunNaming:
    """Tests for MLflow run naming."""

    def test_generate_run_name(self) -> None:
        run_name = generate_run_name(
            SAMPLE_CONFIG_NAME,
            "train",
            datetime(2026, 3, 17, 9, 0, 0),
        )

        assert run_name == "train:resnet18_t4dataset_j6gen2:2026-03-17/09-00-00"


class TestRunMetadata:
    """Tests for persisted run metadata."""

    def test_write_and_load_run_metadata(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "mlruns" / "task" / "model" / "config" / "run_id" / "artifacts"
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
        assert tags["hydra_dir"] == str(work_dir)
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
        assert logger_cfg.tracking_uri.startswith("sqlite:///")

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


class TestArtifactLayout:
    """Tests for semantic MLflow artifact layout helpers."""

    def test_resolve_tracking_uri_expands_relative_sqlite_path_from_cwd(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        work_dir = tmp_path / "workspace"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)
        resolved = resolve_tracking_uri("sqlite:///mlruns/mlflow.db")
        assert resolved == f"sqlite:///{(work_dir / 'mlruns' / 'mlflow.db').resolve()}"

    def test_generate_artifact_location_uses_semantic_config_path(self, tmp_path: Path) -> None:
        tracking_uri = f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
        artifact_location = generate_artifact_location(SAMPLE_CONFIG_NAME, tracking_uri)
        assert artifact_location == ((tmp_path / "mlruns" / SAMPLE_CONFIG_NAME).resolve().as_uri())

    def test_generate_hydra_run_dir_uses_run_id_when_available(self, tmp_path: Path) -> None:
        tracking_uri = f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
        hydra_dir = generate_hydra_run_dir(
            SAMPLE_CONFIG_NAME,
            tracking_uri=tracking_uri,
            run_id="abc123",
        )
        assert hydra_dir == tmp_path / "mlruns" / SAMPLE_CONFIG_NAME / "abc123" / "hydra"

    def test_prepare_run_context_creates_run_under_semantic_artifact_root(
        self, tmp_path: Path
    ) -> None:
        tracking_uri = f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
        run_context = prepare_run_context(
            tracking_uri,
            SAMPLE_CONFIG_NAME,
            hydra_dir=None,
            stage="train",
            started_at=datetime(2026, 3, 17, 9, 0, 0),
        )

        expected_root = tmp_path / "mlruns" / SAMPLE_CONFIG_NAME
        assert run_context.artifact_dir == expected_root / run_context.run_id / "artifacts"
        assert run_context.hydra_dir == expected_root / run_context.run_id / "hydra"
        assert run_context.tags["artifact_dir"] == str(run_context.artifact_dir)
        assert run_context.tags["hydra_dir"] == str(run_context.hydra_dir)

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(SAMPLE_EXPERIMENT_NAME)
        assert experiment is not None
        assert experiment.artifact_location == expected_root.resolve().as_uri()
        loaded_context = load_run_context(tracking_uri, run_context.run_id)
        assert loaded_context.hydra_dir == run_context.hydra_dir

    def test_prepare_run_context_rejects_legacy_artifact_root(self, tmp_path: Path) -> None:
        tracking_uri = f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}"
        client = MlflowClient(tracking_uri=tracking_uri)
        legacy_root = (tmp_path / "mlruns" / "1").resolve()
        experiment_id = client.create_experiment(
            SAMPLE_EXPERIMENT_NAME,
            artifact_location=legacy_root.as_uri(),
        )
        legacy_run = client.create_run(experiment_id=experiment_id, run_name="legacy")
        legacy_artifact_dir = legacy_root / legacy_run.info.run_id / "artifacts"
        legacy_artifact_dir.mkdir(parents=True)
        (legacy_artifact_dir / "note.txt").write_text("legacy", encoding="utf-8")

        try:
            prepare_run_context(
                tracking_uri,
                SAMPLE_CONFIG_NAME,
                hydra_dir=None,
                stage="train",
                started_at=datetime(2026, 3, 17, 9, 0, 0),
            )
        except RuntimeError as exc:
            assert SAMPLE_EXPERIMENT_NAME in str(exc)
            assert str(legacy_root) in str(exc)
        else:
            raise AssertionError("Expected RuntimeError for legacy experiment artifact root.")
