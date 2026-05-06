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

"""Shared MLflow helpers for Autoware-ML scripts.

This module centralizes MLflow run management, artifact logging, and metadata
handling shared by training, testing, deployment, and UI tooling.
"""

import hashlib
import json
import logging
import socket
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.meta_dataset import MetaDataset
from mlflow.entities import DatasetInput, InputTag
from mlflow.entities import Param
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf, open_dict

RUN_METADATA_FILENAME = "run_metadata.json"
MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
MLFLOW_RUN_NAME = "mlflow.runName"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_METADATA_KEYS = (
    "_target_",
    "data_root",
    "dataset_root",
    "train_ann_file",
    "val_ann_file",
    "test_ann_file",
    "predict_ann_file",
    "train_img_root",
    "val_img_root",
    "test_img_root",
    "predict_img_root",
    "train_sample_ids",
    "val_sample_ids",
    "test_sample_ids",
    "predict_sample_ids",
    "max_train_samples",
    "max_val_samples",
    "max_test_samples",
    "max_predict_samples",
)
MLFLOW_DATASET_CONTEXT = "mlflow.data.context"
COCO_DATASET_TARGET = "autoware_ml.datamodule.coco.COCODetectionDataModule"
logger = logging.getLogger(__name__)


class LocalFileDatasetSource(DatasetSource):
    """Metadata-only MLflow dataset source for local files.

    This is used to log COCO annotation files as MLflow dataset inputs without
    loading the full dataset into an in-memory table.
    """

    def __init__(self, uri: str, metadata: dict[str, Any] | None = None):
        self._uri = uri
        self._metadata = metadata or {}

    @staticmethod
    def _get_source_type() -> str:
        return "local"

    def load(self) -> str:
        return self._uri

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        return isinstance(raw_source, str | Path)

    @classmethod
    def _resolve(cls, raw_source: Any) -> "LocalFileDatasetSource":
        return cls(str(Path(raw_source).resolve()))

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self._uri, "metadata": self._metadata}

    @classmethod
    def from_dict(cls, source_dict: dict[Any, Any]) -> "LocalFileDatasetSource":
        return cls(
            uri=str(source_dict["uri"]),
            metadata=dict(source_dict.get("metadata", {})),
        )


def get_user_config_name() -> str:
    """Return the user-facing config name without the ``tasks/`` prefix.

    Returns:
        Config name shown to users in run directories and MLflow tags.
    """
    return HydraConfig.get().job.config_name.removeprefix("tasks/")


def generate_experiment_name(config_name: str) -> str:
    """Generate an MLflow experiment name from the config path.

    Args:
        config_name: User-facing config name.

    Returns:
        MLflow experiment name derived from the config path.
    """
    return config_name.replace("/", "_")


def generate_run_name(config_name: str, work_dir: Path, stage: str) -> str:
    """Generate a readable MLflow run name.

    Args:
        config_name: User-facing config name.
        work_dir: Hydra output directory for the run.
        stage: Pipeline stage such as ``train``, ``test``, or ``deploy``.

    Returns:
        Human-readable run name containing the stage, config leaf, and date.
    """
    config_leaf = config_name.split("/")[-1]
    date_part = work_dir.parent.name
    time_part = work_dir.name
    return f"{stage}:{config_leaf}:{date_part}/{time_part}"


def build_run_tags(
    config_name: str,
    work_dir: Path,
    stage: str,
    extra_tags: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build the default MLflow tag set.

    Args:
        config_name: User-facing config name.
        work_dir: Hydra output directory for the run.
        stage: Pipeline stage such as ``train``, ``test``, or ``deploy``.
        extra_tags: Additional user-provided tags merged into the defaults.

    Returns:
        Tag dictionary attached to the MLflow run.
    """
    parts = config_name.split("/")
    tags = {
        "config_name": config_name,
        "task": parts[0] if len(parts) > 0 else "",
        "model": parts[1] if len(parts) > 1 else "",
        "config_variant": parts[-1] if parts else "",
        "stage": stage,
        "run_dir": str(work_dir),
        "hostname": socket.gethostname(),
        "git_sha": get_git_sha(),
    }
    if extra_tags:
        tags.update({key: str(value) for key, value in extra_tags.items()})
    return tags


def get_git_sha() -> str:
    """Return the current Git revision.

    Returns:
        Short Git SHA for the current repository state, or ``unknown`` when it
        cannot be resolved.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "unknown"


def configure_logger(
    logger_cfg: DictConfig,
    experiment_name: str,
    run_name: str,
    tags: dict[str, str],
    run_id: str | None = None,
) -> None:
    """Populate a Lightning ``MLFlowLogger`` configuration node.

    Args:
        logger_cfg: Mutable logger configuration node.
        experiment_name: Target MLflow experiment name.
        run_name: Human-readable run name.
        tags: Default tag dictionary for the run.
        run_id: Existing MLflow run ID when resuming or nesting runs.
    """
    existing_tags = dict(logger_cfg.get("tags", {}))
    merged_tags = dict(tags)
    merged_tags.update({key: str(value) for key, value in existing_tags.items()})
    with open_dict(logger_cfg):
        logger_cfg.experiment_name = experiment_name
        logger_cfg.run_name = run_name
        logger_cfg.tags = merged_tags
        if run_id is not None:
            logger_cfg.run_id = run_id


def should_enable_logger(cfg: DictConfig) -> bool:
    """Return whether MLflow logging should be enabled for the current run.

    Args:
        cfg: Fully composed Hydra configuration.

    Returns:
        ``True`` when a logger configuration is present and the run is not a
        ``fast_dev_run`` smoke test.
    """
    return cfg.get("logger") is not None and not bool(cfg.trainer.get("fast_dev_run", False))


def build_dataset_metadata(cfg: DictConfig) -> dict[str, Any]:
    """Extract dataset identity metadata from the configured datamodule.

    Args:
        cfg: Fully composed Hydra configuration.

    Returns:
        Dataset-related metadata derived from the datamodule configuration.
    """
    datamodule_cfg = cfg.get("datamodule")
    if datamodule_cfg is None:
        return {}

    resolved_cfg = OmegaConf.to_container(datamodule_cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        return {}

    metadata = {
        ("datamodule_target" if key == "_target_" else key): value
        for key, value in resolved_cfg.items()
        if key in DATASET_METADATA_KEYS and value is not None
    }
    return metadata


def build_dataset_tags(cfg: DictConfig) -> dict[str, str]:
    """Build MLflow tags for dataset identity and split configuration.

    Args:
        cfg: Fully composed Hydra configuration.

    Returns:
        Dataset-related MLflow tags as strings.
    """
    return {f"dataset.{key}": str(value) for key, value in build_dataset_metadata(cfg).items()}


def _sha256_hexdigest(value: str | bytes) -> str:
    """Return a SHA-256 hex digest for the provided value."""
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return a SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_data_path(data_root: str | None, value: str | None) -> Path | None:
    """Resolve a possibly-relative dataset path against the data root."""
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute() and data_root is not None:
        path = Path(data_root) / path
    return path.resolve()


def _build_sample_selector_info(
    sample_selector: Any,
    data_root: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return manifest info and digest material for split sample selectors."""
    if sample_selector is None:
        return None, None
    if isinstance(sample_selector, str):
        selector_path = _resolve_data_path(data_root, sample_selector)
        if selector_path is None:
            return None, None
        return (
            {
                "path": str(selector_path),
                "sha256": _sha256_file(selector_path),
            },
            _sha256_hexdigest(
                json.dumps(
                    {
                        "path": str(selector_path),
                        "sha256": _sha256_file(selector_path),
                    },
                    sort_keys=True,
                )
            ),
        )
    return (
        {"values": sample_selector},
        _sha256_hexdigest(json.dumps(sample_selector, sort_keys=True)),
    )


def _read_coco_annotation_stats(annotation_path: Path) -> dict[str, Any]:
    """Read lightweight COCO stats for display and reproducibility metadata."""
    with annotation_path.open(encoding="utf-8") as handle:
        annotation_data = json.load(handle)
    categories = annotation_data.get("categories", [])
    return {
        "images": len(annotation_data.get("images", [])),
        "annotations": len(annotation_data.get("annotations", [])),
        "categories": len(categories),
        "category_names": [category.get("name", "") for category in categories],
    }


def _build_folder_manifest(root: Path) -> dict[str, Any]:
    """Build a deterministic per-file manifest and digest for a folder tree."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {root}")

    files: list[dict[str, Any]] = []
    total_bytes = 0
    digest = hashlib.sha256()
    for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative_path = file_path.relative_to(root).as_posix()
        file_size = file_path.stat().st_size
        file_sha256 = _sha256_file(file_path)
        total_bytes += file_size
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("utf-8"))
        digest.update(b"\0")
        files.append(
            {
                "path": relative_path,
                "size": file_size,
                "sha256": file_sha256,
            }
        )

    return {
        "root": str(root),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "digest": digest.hexdigest()[:8],
        "files": files,
    }


def _diff_folder_manifests(
    source_manifest: dict[str, Any],
    current_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return a structured diff between two folder manifests."""
    source_files = {entry["path"]: entry for entry in source_manifest.get("files", [])}
    current_files = {entry["path"]: entry for entry in current_manifest.get("files", [])}

    added = sorted(path for path in current_files if path not in source_files)
    removed = sorted(path for path in source_files if path not in current_files)
    changed = sorted(
        path
        for path in source_files.keys() & current_files.keys()
        if source_files[path]["sha256"] != current_files[path]["sha256"]
        or source_files[path]["size"] != current_files[path]["size"]
    )

    status = "match" if not added and not removed and not changed else "mismatch"
    return {
        "status": status,
        "source_digest": source_manifest.get("digest"),
        "current_digest": current_manifest.get("digest"),
        "source_file_count": source_manifest.get("file_count"),
        "current_file_count": current_manifest.get("file_count"),
        "added_files": added,
        "removed_files": removed,
        "changed_files": changed,
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
    }


def _load_manifest_artifact(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
) -> dict[str, Any] | None:
    """Download and parse a JSON artifact manifest from a prior run."""
    try:
        downloaded_path = Path(client.download_artifacts(run_id, artifact_path))
    except Exception:
        return None
    return json.loads(downloaded_path.read_text(encoding="utf-8"))


def _find_matching_source_dataset_manifest(
    client: MlflowClient,
    source_run_id: str,
    current_manifest: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Find the source-run dataset manifest that matches the current split inputs."""
    for artifact in client.list_artifacts(source_run_id, "datasets"):
        if not artifact.is_dir:
            continue
        source_manifest = _load_manifest_artifact(client, source_run_id, f"{artifact.path}/manifest.json")
        if source_manifest is None:
            continue
        if source_manifest.get("annotation_path") == current_manifest.get("annotation_path"):
            return artifact.path, source_manifest
        if source_manifest.get("image_root") == current_manifest.get("image_root"):
            return artifact.path, source_manifest
    return None


def build_coco_dataset_records(
    cfg: DictConfig,
    split_contexts: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build MLflow dataset input records for COCO train/val/test splits.

    The returned records are metadata-only MLflow datasets with explicit
    digests derived from COCO annotation content and split selectors.
    """
    dataset_metadata = build_dataset_metadata(cfg)
    if dataset_metadata.get("datamodule_target") != COCO_DATASET_TARGET:
        return []

    data_root = dataset_metadata.get("data_root") or dataset_metadata.get("dataset_root")
    records: list[dict[str, Any]] = []
    for split, context in split_contexts.items():
        ann_path = _resolve_data_path(data_root, dataset_metadata.get(f"{split}_ann_file"))
        if ann_path is None or not ann_path.is_file():
            continue

        image_root = _resolve_data_path(data_root, dataset_metadata.get(f"{split}_img_root"))
        sample_selector_info, sample_selector_digest = _build_sample_selector_info(
            dataset_metadata.get(f"{split}_sample_ids"),
            data_root,
        )
        annotation_sha256 = _sha256_file(ann_path)
        stats = _read_coco_annotation_stats(ann_path)
        image_manifest = None
        if image_root is not None and image_root.is_dir():
            image_manifest = _build_folder_manifest(image_root)
        dataset_digest = _sha256_hexdigest(
            json.dumps(
                {
                    "annotation_sha256": annotation_sha256,
                    "image_root": str(image_root) if image_root is not None else None,
                    "image_root_digest": image_manifest["digest"] if image_manifest else None,
                    "sample_selector_digest": sample_selector_digest,
                    "max_samples": dataset_metadata.get(f"max_{split}_samples"),
                },
                sort_keys=True,
            )
        )[:8]
        dataset_name = f"{ann_path.stem}:{context}"
        source = LocalFileDatasetSource(
            uri=str(ann_path),
            metadata={
                "image_root": str(image_root) if image_root is not None else None,
                "sample_selector": sample_selector_info,
                "max_samples": dataset_metadata.get(f"max_{split}_samples"),
            },
        )
        dataset = MetaDataset(source=source, name=dataset_name, digest=dataset_digest)
        tags = [
            InputTag(key=MLFLOW_DATASET_CONTEXT, value=context),
            InputTag(key="split", value=split),
            InputTag(key="annotation_path", value=str(ann_path)),
            InputTag(key="annotation_sha256", value=annotation_sha256),
            InputTag(key="images", value=str(stats["images"])),
            InputTag(key="annotations", value=str(stats["annotations"])),
            InputTag(key="categories", value=str(stats["categories"])),
        ]
        if image_root is not None:
            tags.append(InputTag(key="image_root", value=str(image_root)))
        if dataset_metadata.get(f"max_{split}_samples") is not None:
            tags.append(
                InputTag(
                    key="max_samples",
                    value=str(dataset_metadata[f"max_{split}_samples"]),
                )
            )

        if image_manifest is not None:
            tags.extend(
                [
                    InputTag(key="image_root_digest", value=image_manifest["digest"]),
                    InputTag(key="image_root_file_count", value=str(image_manifest["file_count"])),
                ]
            )

        records.append(
            {
                "split": split,
                "annotation_path": ann_path,
                "manifest": {
                    "context": context,
                    "name": dataset_name,
                    "digest": dataset_digest,
                    "annotation_path": str(ann_path),
                    "annotation_sha256": annotation_sha256,
                    "image_root": str(image_root) if image_root is not None else None,
                    "image_root_digest": image_manifest["digest"] if image_manifest else None,
                    "image_root_file_count": image_manifest["file_count"] if image_manifest else None,
                    "image_root_total_bytes": image_manifest["total_bytes"] if image_manifest else None,
                    "sample_selector": sample_selector_info,
                    "max_samples": dataset_metadata.get(f"max_{split}_samples"),
                    "stats": stats,
                },
                "image_manifest": image_manifest,
                "dataset_input": DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags),
            }
        )

    return records


def log_coco_dataset_inputs(
    client: MlflowClient,
    run_id: str,
    cfg: DictConfig,
    stage: str,
    work_dir: Path,
    compare_to_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Log COCO dataset inputs plus annotation snapshots for one pipeline stage."""
    if stage == "train":
        split_contexts = {"train": "training", "val": "validation"}
    elif stage == "test":
        split_contexts = {"test": "testing"}
    else:
        return []

    records = build_coco_dataset_records(cfg, split_contexts)
    if not records:
        return []

    client.log_inputs(
        run_id=run_id,
        datasets=[record["dataset_input"] for record in records],
    )

    for record in records:
        split_dir = work_dir / "datasets" / record["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = split_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(record["manifest"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        log_path_as_artifact(client, run_id, manifest_path, f"datasets/{record['split']}")
        if record["image_manifest"] is not None:
            image_manifest_path = split_dir / "image_manifest.json"
            image_manifest_path.write_text(
                json.dumps(record["image_manifest"], indent=2, sort_keys=True),
                encoding="utf-8",
            )
            log_path_as_artifact(
                client,
                run_id,
                image_manifest_path,
                f"datasets/{record['split']}",
            )
        log_path_as_artifact(
            client,
            run_id,
            record["annotation_path"],
            f"datasets/{record['split']}",
        )
        if compare_to_run_id is not None:
            source_match = _find_matching_source_dataset_manifest(
                client,
                compare_to_run_id,
                record["manifest"],
            )
            comparison_status = "unavailable"
            verification_report: dict[str, Any] = {"status": comparison_status}
            if source_match is not None:
                source_artifact_dir, source_manifest = source_match
                verification_report = {
                    "source_run_id": compare_to_run_id,
                    "source_artifact_dir": source_artifact_dir,
                    "annotation_match": source_manifest.get("annotation_sha256")
                    == record["manifest"].get("annotation_sha256"),
                    "annotation_source_sha256": source_manifest.get("annotation_sha256"),
                    "annotation_current_sha256": record["manifest"].get("annotation_sha256"),
                }
                source_image_manifest = _load_manifest_artifact(
                    client,
                    compare_to_run_id,
                    f"{source_artifact_dir}/image_manifest.json",
                )
                if source_image_manifest is None or record["image_manifest"] is None:
                    verification_report["image_manifest_status"] = "unavailable"
                    comparison_status = (
                        "match" if verification_report["annotation_match"] else "mismatch"
                    )
                else:
                    image_diff = _diff_folder_manifests(source_image_manifest, record["image_manifest"])
                    verification_report["image_manifest_status"] = image_diff["status"]
                    verification_report["image_manifest_diff"] = image_diff
                    comparison_status = (
                        "match"
                        if verification_report["annotation_match"] and image_diff["status"] == "match"
                        else "mismatch"
                    )
                verification_report["status"] = comparison_status

            verification_path = split_dir / "verification.json"
            verification_path.write_text(
                json.dumps(verification_report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            log_path_as_artifact(client, run_id, verification_path, f"datasets/{record['split']}")
            client.set_tag(run_id, f"dataset.{record['split']}.comparison_status", comparison_status)
            if source_match is not None:
                client.set_tag(
                    run_id,
                    f"dataset.{record['split']}.annotation_match",
                    str(verification_report["annotation_match"]).lower(),
                )
                image_manifest_diff = verification_report.get("image_manifest_diff")
                if image_manifest_diff is not None:
                    client.set_tag(
                        run_id,
                        f"dataset.{record['split']}.image_manifest_match",
                        str(image_manifest_diff["status"] == "match").lower(),
                    )
                    client.set_tag(
                        run_id,
                        f"dataset.{record['split']}.added_files",
                        str(image_manifest_diff["added_count"]),
                    )
                    client.set_tag(
                        run_id,
                        f"dataset.{record['split']}.removed_files",
                        str(image_manifest_diff["removed_count"]),
                    )
                    client.set_tag(
                        run_id,
                        f"dataset.{record['split']}.changed_files",
                        str(image_manifest_diff["changed_count"]),
                    )
            if comparison_status == "mismatch":
                logger.warning(
                    "Dataset diff detected for split '%s' relative to source run %s.",
                    record["split"],
                    compare_to_run_id,
                )

    return records


def _flatten_params(prefix: str, value: Any) -> dict[str, str]:
    """Flatten nested config values into MLflow parameter strings.

    Args:
        prefix: Current flattened key prefix.
        value: Nested value to flatten.

    Returns:
        Flat mapping from dotted keys to string values.
    """
    if isinstance(value, dict):
        result: dict[str, str] = {}
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_params(nested_prefix, nested_value))
        return result
    if isinstance(value, list):
        return {prefix: json.dumps(value)}
    return {prefix: str(value)}


def log_config_params(client: MlflowClient, run_id: str, cfg: dict[str, Any]) -> None:
    """Log a resolved configuration dictionary as MLflow parameters.

    Args:
        client: MLflow tracking client.
        run_id: Target MLflow run ID.
        cfg: Resolved configuration dictionary.
    """
    flattened = _flatten_params("", cfg)
    items = list(flattened.items())
    for start in range(0, len(items), 100):
        batch = [Param(key=key, value=value[:6000]) for key, value in items[start : start + 100]]
        client.log_batch(run_id, params=batch)


def write_run_metadata(work_dir: Path, metadata: dict[str, Any]) -> Path:
    """Persist run metadata next to Hydra outputs.

    Args:
        work_dir: Hydra output directory for the current run.
        metadata: Metadata payload to persist.

    Returns:
        Path to the written metadata file.
    """
    metadata_path = work_dir / RUN_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def load_run_metadata(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load run metadata associated with a checkpoint path.

    Args:
        checkpoint_path: Checkpoint path whose parent directories should be
            searched for run metadata.

    Returns:
        Parsed run metadata when present, otherwise ``None``.
    """
    for parent in checkpoint_path.parents:
        metadata_path = parent / RUN_METADATA_FILENAME
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
    return None


def _infer_work_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    """Infer the Hydra run directory that owns a checkpoint."""
    for parent in checkpoint_path.parents:
        if (parent / ".hydra").is_dir():
            return parent
    return checkpoint_path.parent


def _infer_config_name_from_work_dir(work_dir: Path) -> str | None:
    """Infer the user-facing config name from a Hydra run directory under mlruns."""
    parts = work_dir.parts
    if "mlruns" not in parts:
        return None
    mlruns_index = parts.index("mlruns")
    config_parts = parts[mlruns_index + 1 : -2]
    if not config_parts:
        return None
    return "/".join(config_parts)


def _find_run_by_run_dir(
    tracking_uri: str,
    experiment_name: str,
    checkpoint_path: Path,
) -> tuple[str | None, str | None]:
    """Find a run and experiment by matching the checkpoint's owning run directory tag."""
    client = MlflowClient(tracking_uri=tracking_uri)

    work_dir = _infer_work_dir_from_checkpoint(checkpoint_path)
    candidate_run_dirs = [str(work_dir), str(work_dir.resolve())]
    inferred_config_name = _infer_config_name_from_work_dir(work_dir)
    inferred_experiment_name = (
        generate_experiment_name(inferred_config_name) if inferred_config_name is not None else None
    )
    inferred_run_name = (
        generate_run_name(inferred_config_name, work_dir, "train")
        if inferred_config_name is not None
        else None
    )
    experiments = []
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        experiments.append(experiment)
    if inferred_experiment_name is not None and inferred_experiment_name != experiment_name:
        inferred_experiment = client.get_experiment_by_name(inferred_experiment_name)
        if inferred_experiment is not None:
            experiments.append(inferred_experiment)
    experiments.extend(
        candidate
        for candidate in client.search_experiments()
        if candidate.name not in {experiment_name, inferred_experiment_name}
    )

    expected_run_dirs = list(dict.fromkeys(candidate_run_dirs))
    for candidate_experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[candidate_experiment.experiment_id],
            max_results=5000,
            order_by=["attributes.start_time DESC"],
        )
        for run in runs:
            run_dir = run.data.tags.get("run_dir")
            run_name = run.data.tags.get(MLFLOW_RUN_NAME)
            if run_dir in expected_run_dirs or (
                inferred_run_name is not None and run_name == inferred_run_name
            ):
                return run.info.run_id, candidate_experiment.name
    return None, None


def resolve_lineage_context(
    config_name: str,
    checkpoint_path: Path | None,
    tracking_uri: str | None = None,
) -> tuple[str, str | None]:
    """Resolve experiment lineage for follow-up stages.

    Args:
        config_name: User-facing config name for the current command.
        checkpoint_path: Checkpoint used by the current stage, if any.
        tracking_uri: Optional MLflow tracking URI used to recover lineage when
            local run metadata is missing.

    Returns:
        Tuple containing the experiment name and optional parent run ID.
    """
    if checkpoint_path is None:
        return generate_experiment_name(config_name), None

    experiment_name = generate_experiment_name(config_name)
    metadata = load_run_metadata(checkpoint_path)
    if metadata is None:
        if tracking_uri is None:
            return experiment_name, None
        parent_run_id, recovered_experiment_name = _find_run_by_run_dir(
            tracking_uri,
            experiment_name,
            checkpoint_path,
        )
        return recovered_experiment_name or experiment_name, parent_run_id
    return metadata["experiment_name"], metadata.get("run_id")


def create_child_run(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    tags: dict[str, str],
    parent_run_id: str | None,
) -> str:
    """Create a child MLflow run and return its ID.

    Args:
        tracking_uri: MLflow tracking backend URI.
        experiment_name: Target experiment name.
        run_name: Human-readable run name.
        tags: Tag dictionary attached to the run.
        parent_run_id: Optional parent run ID for nested lineage.

    Returns:
        Newly created MLflow run ID.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    run_tags = dict(tags)
    if parent_run_id is not None:
        run_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
    run_tags[MLFLOW_RUN_NAME] = run_name
    run = client.create_run(experiment_id=experiment_id, tags=run_tags, run_name=run_name)
    return run.info.run_id


def reopen_run(
    tracking_uri: str,
    run_id: str,
    tags: Mapping[str, Any] | None = None,
) -> None:
    """Mark an existing run as running again and optionally update tags.

    Args:
        tracking_uri: MLflow tracking backend URI.
        run_id: Existing MLflow run ID to reopen.
        tags: Optional tags to upsert on the reopened run.
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_terminated(run_id, status="RUNNING")
    if tags:
        for key, value in tags.items():
            client.set_tag(run_id, key, str(value))


def log_path_as_artifact(
    client: MlflowClient, run_id: str, path: Path, artifact_path: str | None = None
) -> None:
    """Log a file or directory tree as MLflow artifacts.

    Args:
        client: MLflow tracking client.
        run_id: Target MLflow run ID.
        path: File or directory to log as artifacts.
        artifact_path: Optional MLflow artifact subdirectory.
    """
    if not path.exists():
        return
    if path.is_file():
        client.log_artifact(run_id, str(path), artifact_path=artifact_path)
        return

    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file():
            continue
        relative_parent = file_path.relative_to(path).parent.as_posix()
        target_path = artifact_path
        if relative_parent != ".":
            target_path = f"{artifact_path}/{relative_parent}" if artifact_path else relative_parent
        client.log_artifact(run_id, str(file_path), artifact_path=target_path)


def build_stage_metadata(
    *,
    run_id: str | None,
    experiment_id: str | None,
    experiment_name: str,
    config_name: str,
    work_dir: Path,
    stage: str,
    source_run_id: str | None = None,
    checkpoint_path: Path | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build persisted metadata for one pipeline stage.

    Args:
        run_id: MLflow run ID for the current stage.
        experiment_id: MLflow experiment ID for the current stage.
        experiment_name: MLflow experiment name for the current stage.
        config_name: User-facing config name.
        work_dir: Hydra output directory for the run.
        stage: Pipeline stage such as ``train``, ``test``, or ``deploy``.
        source_run_id: Optional parent/source MLflow run ID.
        checkpoint_path: Optional checkpoint path used by the stage.
        dataset_metadata: Optional dataset identity metadata.

    Returns:
        Metadata dictionary persisted alongside run outputs.
    """
    metadata = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "config_name": config_name,
        "work_dir": str(work_dir),
        "stage": stage,
    }
    if source_run_id is not None:
        metadata["source_run_id"] = source_run_id
    if checkpoint_path is not None:
        metadata["checkpoint_path"] = str(checkpoint_path)
    if dataset_metadata:
        metadata["dataset"] = dataset_metadata
    return metadata


def build_training_metadata(
    logger: MLFlowLogger,
    experiment_name: str,
    config_name: str,
    work_dir: Path,
    *,
    source_run_id: str | None = None,
    checkpoint_path: Path | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build persisted metadata for the current training run.

    Args:
        logger: Configured MLflow logger for the training run.
        experiment_name: MLflow experiment name for the run.
        config_name: User-facing config name for the run.
        work_dir: Hydra output directory for the run.
        source_run_id: Optional parent/source MLflow run ID.
        checkpoint_path: Optional checkpoint used to resume training.
        dataset_metadata: Optional dataset identity metadata.

    Returns:
        Metadata dictionary persisted alongside run outputs.
    """
    return build_stage_metadata(
        run_id=logger.run_id,
        experiment_id=logger.experiment_id,
        experiment_name=experiment_name,
        config_name=config_name,
        work_dir=work_dir,
        stage="train",
        source_run_id=source_run_id,
        checkpoint_path=checkpoint_path,
        dataset_metadata=dataset_metadata,
    )
