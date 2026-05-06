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

import json
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from autoware_ml.utils.mlflow import (
    _build_folder_manifest,
    _diff_folder_manifests,
    build_dataset_metadata,
    build_dataset_tags,
    build_coco_dataset_records,
    build_run_tags,
    build_stage_metadata,
    build_training_metadata,
    configure_logger,
    generate_run_name,
    log_coco_dataset_inputs,
    load_run_metadata,
    reopen_run,
    resolve_lineage_context,
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

    def test_build_stage_metadata_includes_dataset_and_lineage(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "mlruns" / "task" / "model" / "config" / "2026-03-17" / "09-00-00"
        metadata = build_stage_metadata(
            run_id="run-1",
            experiment_id="exp-1",
            experiment_name="task_model_config",
            config_name=SAMPLE_CONFIG_NAME,
            work_dir=work_dir,
            stage="test",
            source_run_id="parent-1",
            checkpoint_path=tmp_path / "last.ckpt",
            dataset_metadata={"train_ann_file": "train.json"},
        )

        assert metadata["run_id"] == "run-1"
        assert metadata["experiment_id"] == "exp-1"
        assert metadata["stage"] == "test"
        assert metadata["source_run_id"] == "parent-1"
        assert metadata["checkpoint_path"].endswith("last.ckpt")
        assert metadata["dataset"] == {"train_ann_file": "train.json"}

    def test_build_training_metadata_includes_optional_context(self, tmp_path: Path) -> None:
        class DummyLogger:
            run_id = "run-1"
            experiment_id = "exp-1"

        metadata = build_training_metadata(
            DummyLogger(),
            "task_model_config",
            SAMPLE_CONFIG_NAME,
            tmp_path,
            source_run_id="parent-1",
            checkpoint_path=tmp_path / "last.ckpt",
            dataset_metadata={"val_ann_file": "val.json"},
        )

        assert metadata["run_id"] == "run-1"
        assert metadata["stage"] == "train"
        assert metadata["source_run_id"] == "parent-1"
        assert metadata["checkpoint_path"].endswith("last.ckpt")
        assert metadata["dataset"] == {"val_ann_file": "val.json"}

    def test_resolve_lineage_context_falls_back_to_mlflow_run_dir_lookup(
        self, tmp_path: Path
    ) -> None:
        work_dir = tmp_path / "mlruns" / "task" / "model" / "config" / "2026-03-17" / "09-00-00"
        checkpoints_dir = work_dir / "checkpoints"
        hydra_dir = work_dir / ".hydra"
        checkpoints_dir.mkdir(parents=True)
        hydra_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "last.ckpt"
        checkpoint_path.write_text("", encoding="utf-8")

        class DummyExperiment:
            experiment_id = "exp-1"
            name = SAMPLE_EXPERIMENT_NAME

        class DummyRunInfo:
            run_id = "run-123"

        class DummyRunData:
            tags = {"run_dir": str(work_dir.resolve())}

        class DummyRun:
            info = DummyRunInfo()
            data = DummyRunData()

        class DummyClient:
            def __init__(self, tracking_uri: str):
                self.tracking_uri = tracking_uri

            def get_experiment_by_name(self, name: str):
                if name in {SAMPLE_EXPERIMENT_NAME, "task_model_config"}:
                    return DummyExperiment()
                return None

            def search_experiments(self):
                return [DummyExperiment()]

            def search_runs(self, experiment_ids, max_results: int, order_by):
                assert experiment_ids == ["exp-1"]
                assert max_results == 5000
                assert order_by == ["attributes.start_time DESC"]
                return [DummyRun()]

        with patch("autoware_ml.utils.mlflow.MlflowClient", DummyClient):
            experiment_name, run_id = resolve_lineage_context(
                SAMPLE_CONFIG_NAME,
                checkpoint_path,
                tracking_uri="sqlite:///mlruns/mlflow.db",
            )

        assert experiment_name == SAMPLE_EXPERIMENT_NAME
        assert run_id == "run-123"

    def test_resolve_lineage_context_recovers_original_experiment_when_config_name_differs(
        self, tmp_path: Path
    ) -> None:
        work_dir = tmp_path / "mlruns" / "task" / "model" / "config" / "2026-03-17" / "09-00-00"
        checkpoints_dir = work_dir / "checkpoints"
        hydra_dir = work_dir / ".hydra"
        checkpoints_dir.mkdir(parents=True)
        hydra_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "last.ckpt"
        checkpoint_path.write_text("", encoding="utf-8")

        class WrongExperiment:
            experiment_id = "exp-wrong"
            name = "config"

        class CorrectExperiment:
            experiment_id = "exp-right"
            name = SAMPLE_EXPERIMENT_NAME

        class DummyRunInfo:
            run_id = "run-456"

        class DummyRunData:
            tags = {"run_dir": str(work_dir.resolve())}

        class DummyRun:
            info = DummyRunInfo()
            data = DummyRunData()

        class DummyClient:
            def __init__(self, tracking_uri: str):
                self.tracking_uri = tracking_uri

            def get_experiment_by_name(self, name: str):
                if name == "config":
                    return WrongExperiment()
                if name == "task_model_config":
                    return CorrectExperiment()
                return None

            def search_experiments(self):
                return [WrongExperiment(), CorrectExperiment()]

            def search_runs(self, experiment_ids, max_results: int, order_by):
                assert max_results == 5000
                assert order_by == ["attributes.start_time DESC"]
                if experiment_ids == ["exp-right"]:
                    return [DummyRun()]
                return []

        with patch("autoware_ml.utils.mlflow.MlflowClient", DummyClient):
            experiment_name, run_id = resolve_lineage_context(
                "config",
                checkpoint_path,
                tracking_uri="sqlite:///mlruns/mlflow.db",
            )

        assert experiment_name == SAMPLE_EXPERIMENT_NAME
        assert run_id == "run-456"

    def test_resolve_lineage_context_can_match_by_original_run_name_when_run_dir_moved(
        self, tmp_path: Path
    ) -> None:
        work_dir = (
            tmp_path
            / "mlruns"
            / "detection2d"
            / "rtdetrv4"
            / "hgnetv2_s_mapillary_vistas_coco_transfer"
            / "2026-04-17"
            / "07-48-28"
        )
        checkpoints_dir = work_dir / "checkpoints"
        hydra_dir = work_dir / ".hydra"
        checkpoints_dir.mkdir(parents=True)
        hydra_dir.mkdir(parents=True)
        checkpoint_path = checkpoints_dir / "last.ckpt"
        checkpoint_path.write_text("", encoding="utf-8")

        expected_experiment = "detection2d_rtdetrv4_hgnetv2_s_mapillary_vistas_coco_transfer"
        expected_run_name = "train:hgnetv2_s_mapillary_vistas_coco_transfer:2026-04-17/07-48-28"

        class DummyExperiment:
            experiment_id = "exp-1"
            name = expected_experiment

        class DummyRunInfo:
            run_id = "run-789"

        class DummyRunData:
            tags = {
                "run_dir": "/workspace/mlruns/detection2d/rtdetrv4/hgnetv2_s_mapillary_vistas_coco_transfer/2026-04-18/06-11-50",
                "mlflow.runName": expected_run_name,
            }

        class DummyRun:
            info = DummyRunInfo()
            data = DummyRunData()

        class DummyClient:
            def __init__(self, tracking_uri: str):
                self.tracking_uri = tracking_uri

            def get_experiment_by_name(self, name: str):
                if name == expected_experiment:
                    return DummyExperiment()
                return None

            def search_experiments(self):
                return [DummyExperiment()]

            def search_runs(self, experiment_ids, max_results: int, order_by):
                assert experiment_ids == ["exp-1"]
                assert max_results == 5000
                assert order_by == ["attributes.start_time DESC"]
                return [DummyRun()]

        with patch("autoware_ml.utils.mlflow.MlflowClient", DummyClient):
            experiment_name, run_id = resolve_lineage_context(
                "config",
                checkpoint_path,
                tracking_uri="sqlite:///mlruns/mlflow.db",
            )

        assert experiment_name == expected_experiment
        assert run_id == "run-789"


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

    def test_build_dataset_metadata_and_tags(self) -> None:
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
                    "unused_field": "ignore-me",
                }
            }
        )

        metadata = build_dataset_metadata(cfg)
        tags = build_dataset_tags(cfg)

        assert metadata == {
            "datamodule_target": "autoware_ml.datamodule.coco.COCODetectionDataModule",
            "data_root": "/data",
            "train_ann_file": "train.json",
            "val_ann_file": "val.json",
            "train_img_root": "images/train",
            "val_img_root": "images/val",
            "max_train_samples": 512,
        }
        assert tags["dataset.datamodule_target"] == (
            "autoware_ml.datamodule.coco.COCODetectionDataModule"
        )
        assert tags["dataset.train_ann_file"] == "train.json"
        assert tags["dataset.max_train_samples"] == "512"

    def test_build_coco_dataset_records_uses_annotation_content_digest(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        annotations_dir = data_root / "annotations"
        annotations_dir.mkdir(parents=True)
        train_ann = annotations_dir / "train.json"
        train_ann.write_text(
            """
            {
              "images": [{"id": 1, "file_name": "a.jpg"}],
              "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0}],
              "categories": [{"id": 1, "name": "car"}]
            }
            """,
            encoding="utf-8",
        )
        cfg = OmegaConf.create(
            {
                "datamodule": {
                    "_target_": "autoware_ml.datamodule.coco.COCODetectionDataModule",
                    "data_root": str(data_root),
                    "train_ann_file": "annotations/train.json",
                    "train_img_root": "images/train",
                }
            }
        )

        records = build_coco_dataset_records(cfg, {"train": "training"})

        assert len(records) == 1
        record = records[0]
        assert record["manifest"]["stats"] == {
            "images": 1,
            "annotations": 1,
            "categories": 1,
            "category_names": ["car"],
        }
        assert record["manifest"]["annotation_path"] == str(train_ann.resolve())
        assert record["dataset_input"].dataset.name == "train:training"
        tag_map = {tag.key: tag.value for tag in record["dataset_input"].tags}
        assert tag_map["mlflow.data.context"] == "training"
        assert tag_map["split"] == "train"
        assert tag_map["annotation_path"] == str(train_ann.resolve())

    def test_log_coco_dataset_inputs_logs_inputs_and_artifacts(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        annotations_dir = data_root / "annotations"
        annotations_dir.mkdir(parents=True)
        (data_root / "images" / "train").mkdir(parents=True)
        (data_root / "images" / "val").mkdir(parents=True)
        (data_root / "images" / "train" / "a.jpg").write_bytes(b"train-a")
        (data_root / "images" / "val" / "b.jpg").write_bytes(b"val-b")
        train_ann = annotations_dir / "train.json"
        val_ann = annotations_dir / "val.json"
        train_ann.write_text('{"images": [], "annotations": [], "categories": []}', encoding="utf-8")
        val_ann.write_text('{"images": [], "annotations": [], "categories": []}', encoding="utf-8")
        cfg = OmegaConf.create(
            {
                "datamodule": {
                    "_target_": "autoware_ml.datamodule.coco.COCODetectionDataModule",
                    "data_root": str(data_root),
                    "train_ann_file": "annotations/train.json",
                    "val_ann_file": "annotations/val.json",
                    "train_img_root": "images/train",
                    "val_img_root": "images/val",
                }
            }
        )

        class DummyClient:
            def __init__(self) -> None:
                self.logged_inputs = None
                self.artifacts: list[tuple[str, str, str | None]] = []

            def log_inputs(self, run_id: str, datasets=None, models=None) -> None:
                self.logged_inputs = (run_id, datasets, models)

            def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
                self.artifacts.append((run_id, local_path, artifact_path))

        client = DummyClient()

        records = log_coco_dataset_inputs(client, "run-1", cfg, "train", tmp_path / "run")

        assert len(records) == 2
        assert client.logged_inputs is not None
        run_id, datasets, models = client.logged_inputs
        assert run_id == "run-1"
        assert len(datasets) == 2
        assert models is None
        artifact_paths = {(Path(local_path).name, artifact_path) for _, local_path, artifact_path in client.artifacts}
        assert ("manifest.json", "datasets/train") in artifact_paths
        assert ("manifest.json", "datasets/val") in artifact_paths
        assert ("image_manifest.json", "datasets/train") in artifact_paths
        assert ("image_manifest.json", "datasets/val") in artifact_paths
        assert ("train.json", "datasets/train") in artifact_paths
        assert ("val.json", "datasets/val") in artifact_paths

    def test_log_coco_dataset_inputs_compares_against_source_run(self, tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        annotations_dir = data_root / "annotations"
        image_root = data_root / "images" / "val"
        annotations_dir.mkdir(parents=True)
        image_root.mkdir(parents=True)
        (image_root / "keep.jpg").write_bytes(b"current")
        val_ann = annotations_dir / "val.json"
        val_ann.write_text('{"images": [], "annotations": [], "categories": []}', encoding="utf-8")
        cfg = OmegaConf.create(
            {
                "datamodule": {
                    "_target_": "autoware_ml.datamodule.coco.COCODetectionDataModule",
                    "data_root": str(data_root),
                    "test_ann_file": "annotations/val.json",
                    "test_img_root": "images/val",
                }
            }
        )

        source_manifest = {
            "annotation_path": str(val_ann.resolve()),
            "annotation_sha256": "same-annotation",
            "image_root": str(image_root.resolve()),
        }
        source_image_manifest = {
            "root": str(image_root.resolve()),
            "digest": "source123",
            "file_count": 1,
            "total_bytes": 4,
            "files": [{"path": "keep.jpg", "size": 4, "sha256": "old"}],
        }

        class DummyClient:
            def __init__(self) -> None:
                self.logged_inputs = None
                self.artifacts: list[tuple[str, str, str | None]] = []
                self.tags: list[tuple[str, str, str]] = []

            def log_inputs(self, run_id: str, datasets=None, models=None) -> None:
                self.logged_inputs = (run_id, datasets, models)

            def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
                self.artifacts.append((run_id, local_path, artifact_path))

            def set_tag(self, run_id: str, key: str, value: str) -> None:
                self.tags.append((run_id, key, value))

            def list_artifacts(self, run_id: str, path=None):
                del run_id
                if path == "datasets":
                    return [type("Artifact", (), {"is_dir": True, "path": "datasets/val"})()]
                return []

            def download_artifacts(self, run_id: str, path: str, dst_path: str | None = None) -> str:
                del run_id, dst_path
                if path == "datasets/val/manifest.json":
                    payload = source_manifest
                elif path == "datasets/val/image_manifest.json":
                    payload = source_image_manifest
                else:
                    raise FileNotFoundError(path)
                out = tmp_path / "downloads" / Path(path).name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload), encoding="utf-8")
                return str(out)

        client = DummyClient()
        records = log_coco_dataset_inputs(
            client,
            "run-1",
            cfg,
            "test",
            tmp_path / "run",
            compare_to_run_id="source-run",
        )

        assert len(records) == 1
        tag_map = {(key, value) for _, key, value in client.tags}
        assert ("dataset.test.comparison_status", "mismatch") in tag_map
        assert ("dataset.test.image_manifest_match", "false") in tag_map
        assert ("dataset.test.changed_files", "1") in tag_map
        verification_path = tmp_path / "run" / "datasets" / "test" / "verification.json"
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        assert verification["status"] == "mismatch"
        assert verification["image_manifest_diff"]["changed_count"] == 1
        assert verification["image_manifest_diff"]["added_count"] == 0
        assert verification["image_manifest_diff"]["removed_count"] == 0

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


class TestRunResume:
    """Tests for same-run resume helpers."""

    def test_reopen_run_marks_run_running_and_updates_tags(self) -> None:
        class DummyClient:
            def __init__(self, tracking_uri: str):
                self.tracking_uri = tracking_uri
                self.calls: list[tuple[str, str, str | None]] = []

            def set_terminated(self, run_id: str, status: str | None = None, end_time=None) -> None:
                del end_time
                self.calls.append(("set_terminated", run_id, status))

            def set_tag(self, run_id: str, key: str, value: str) -> None:
                self.calls.append((f"set_tag:{key}", run_id, value))

        dummy_client = DummyClient("sqlite:///mlruns/mlflow.db")
        with patch("autoware_ml.utils.mlflow.MlflowClient", return_value=dummy_client):
            reopen_run(
                "sqlite:///mlruns/mlflow.db",
                "run-1",
                tags={"stage": "train", "resume_mode": "same_run"},
            )

        assert dummy_client.calls[0] == ("set_terminated", "run-1", "RUNNING")
        assert ("set_tag:stage", "run-1", "train") in dummy_client.calls
        assert ("set_tag:resume_mode", "run-1", "same_run") in dummy_client.calls


class TestDatasetManifestHelpers:
    """Tests for deep dataset manifest hashing and diffing."""

    def test_build_folder_manifest_is_deterministic(self, tmp_path: Path) -> None:
        root = tmp_path / "images"
        (root / "b").mkdir(parents=True)
        (root / "a").mkdir(parents=True)
        (root / "a" / "1.txt").write_text("one", encoding="utf-8")
        (root / "b" / "2.txt").write_text("two", encoding="utf-8")

        manifest_a = _build_folder_manifest(root)
        manifest_b = _build_folder_manifest(root)

        assert manifest_a["digest"] == manifest_b["digest"]
        assert manifest_a["file_count"] == 2
        assert [entry["path"] for entry in manifest_a["files"]] == ["a/1.txt", "b/2.txt"]

    def test_diff_folder_manifests_reports_added_removed_and_changed(self) -> None:
        source = {
            "digest": "source",
            "file_count": 2,
            "files": [
                {"path": "keep.txt", "size": 3, "sha256": "aaa"},
                {"path": "remove.txt", "size": 4, "sha256": "bbb"},
            ],
        }
        current = {
            "digest": "current",
            "file_count": 2,
            "files": [
                {"path": "keep.txt", "size": 5, "sha256": "ccc"},
                {"path": "add.txt", "size": 2, "sha256": "ddd"},
            ],
        }

        diff = _diff_folder_manifests(source, current)

        assert diff["status"] == "mismatch"
        assert diff["added_files"] == ["add.txt"]
        assert diff["removed_files"] == ["remove.txt"]
        assert diff["changed_files"] == ["keep.txt"]
