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

"""Tests for MLflow tracking-store helpers."""

from pathlib import Path

from mlflow.tracking import MlflowClient

from autoware_ml.utils.mlflow import MLFLOW_PARENT_RUN_ID
from autoware_ml.utils.mlflow_store import (
    export_experiment_from_db,
    export_experiment_to_store,
    normalize_experiment_name,
    prepare_export_output_dir,
)

SAMPLE_CONFIG_NAME = "calibration_status/calibration_status_classifier/resnet18_t4dataset_j6gen2"
SAMPLE_EXPERIMENT_NAME = (
    "calibration_status_calibration_status_classifier_resnet18_t4dataset_j6gen2"
)


class TestNormalizeExperimentName:
    """Tests for experiment name normalization."""

    def test_uses_experiment_name_directly(self) -> None:
        assert normalize_experiment_name(SAMPLE_EXPERIMENT_NAME, None) == SAMPLE_EXPERIMENT_NAME

    def test_derives_experiment_name_from_config_name(self) -> None:
        assert normalize_experiment_name(None, SAMPLE_CONFIG_NAME) == SAMPLE_EXPERIMENT_NAME

    def test_strips_tasks_prefix_from_config_name(self) -> None:
        assert (
            normalize_experiment_name(None, f"tasks/{SAMPLE_CONFIG_NAME}") == SAMPLE_EXPERIMENT_NAME
        )


class TestExportExperiment:
    """Tests for MLflow experiment export."""

    def test_export_copies_runs_metrics_and_metadata(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source"
        source_tracking_uri = f"sqlite:///{source_dir / 'mlflow.db'}"
        source_client = MlflowClient(tracking_uri=source_tracking_uri)
        experiment_id = source_client.create_experiment(
            SAMPLE_EXPERIMENT_NAME,
            artifact_location=(source_dir / "artifacts").as_uri(),
        )

        run = source_client.create_run(
            experiment_id=experiment_id,
            tags={"stage": "train"},
            run_name="baseline",
        )
        run_id = run.info.run_id
        source_client.log_param(run_id, "batch_size", "2")
        source_client.log_metric(run_id, "mAP", 0.42, step=0, timestamp=1000)
        source_client.log_metric(run_id, "mAP", 0.51, step=1, timestamp=2000)
        source_client.set_terminated(run_id, status="FINISHED", end_time=3000)

        export_dir = tmp_path / "exported"
        exported_db_path = export_experiment_to_store(
            source_tracking_uri=source_tracking_uri,
            experiment_name=SAMPLE_EXPERIMENT_NAME,
            export_dir=export_dir,
        )

        assert exported_db_path == export_dir / "mlflow.db"
        target_client = MlflowClient(tracking_uri=f"sqlite:///{exported_db_path}")
        target_experiment = target_client.get_experiment_by_name(SAMPLE_EXPERIMENT_NAME)
        assert target_experiment is not None

        exported_runs = target_client.search_runs([target_experiment.experiment_id])
        assert len(exported_runs) == 1
        exported_run = exported_runs[0]
        assert exported_run.data.params["batch_size"] == "2"
        assert exported_run.data.metrics["mAP"] == 0.51

        metric_history = target_client.get_metric_history(exported_run.info.run_id, "mAP")
        assert [metric.value for metric in metric_history] == [0.42, 0.51]
        assert exported_run.data.tags["stage"] == "train"

    def test_export_preserves_nested_run_relationships(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "source_nested"
        source_tracking_uri = f"sqlite:///{source_dir / 'mlflow.db'}"
        source_client = MlflowClient(tracking_uri=source_tracking_uri)
        experiment_id = source_client.create_experiment(
            SAMPLE_EXPERIMENT_NAME,
            artifact_location=(source_dir / "artifacts").as_uri(),
        )

        parent_run = source_client.create_run(experiment_id=experiment_id, run_name="train")
        child_run = source_client.create_run(
            experiment_id=experiment_id,
            run_name="test",
            tags={MLFLOW_PARENT_RUN_ID: parent_run.info.run_id},
        )
        source_client.set_terminated(parent_run.info.run_id, status="FINISHED", end_time=1000)
        source_client.set_terminated(child_run.info.run_id, status="FINISHED", end_time=2000)

        exported_db_path = export_experiment_to_store(
            source_tracking_uri=source_tracking_uri,
            experiment_name=SAMPLE_EXPERIMENT_NAME,
            export_dir=tmp_path / "exported_nested",
        )

        target_client = MlflowClient(tracking_uri=f"sqlite:///{exported_db_path}")
        target_experiment = target_client.get_experiment_by_name(SAMPLE_EXPERIMENT_NAME)
        assert target_experiment is not None

        exported_runs = target_client.search_runs([target_experiment.experiment_id])
        assert len(exported_runs) == 2
        runs_by_name = {run.data.tags["mlflow.runName"]: run for run in exported_runs}
        assert (
            runs_by_name["test"].data.tags[MLFLOW_PARENT_RUN_ID]
            == runs_by_name["train"].info.run_id
        )

    def test_export_raises_when_target_db_exists_without_override(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "existing_export"
        output_dir.mkdir()
        target_db_path = output_dir / "mlflow.db"
        target_db_path.write_text("occupied", encoding="utf-8")

        try:
            prepare_export_output_dir(output_dir, override=False)
        except FileExistsError as exc:
            assert str(target_db_path) in str(exc)
            assert "--override" in str(exc)
        else:
            raise AssertionError("Expected FileExistsError when export DB already exists.")

    def test_export_override_cleans_previous_outputs(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "override_export"
        output_dir.mkdir()
        (output_dir / "mlflow.db").write_text("old-db", encoding="utf-8")
        (output_dir / "mlflow.db-shm").write_text("old-shm", encoding="utf-8")

        target_db_path = prepare_export_output_dir(output_dir, override=True)

        assert target_db_path == output_dir / "mlflow.db"
        assert not (output_dir / "mlflow.db-shm").exists()

    def test_export_from_db_raises_when_source_db_is_missing(self, tmp_path: Path) -> None:
        missing_db_path = tmp_path / "missing.db"

        try:
            export_experiment_from_db(
                db_path=str(missing_db_path),
                experiment_name=SAMPLE_EXPERIMENT_NAME,
            )
        except FileNotFoundError as exc:
            assert str(missing_db_path) in str(exc)
        else:
            raise AssertionError("Expected FileNotFoundError when source MLflow DB is missing.")
