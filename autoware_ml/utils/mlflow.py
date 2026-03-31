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

import json
import socket
import subprocess
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.entities import Param
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, open_dict

RUN_METADATA_FILENAME = "run_metadata.json"
MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
MLFLOW_RUN_NAME = "mlflow.runName"
REPO_ROOT = Path(__file__).resolve().parents[2]


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


def resolve_lineage_context(
    config_name: str,
    checkpoint_path: Path | None,
) -> tuple[str, str | None]:
    """Resolve experiment lineage for follow-up stages.

    Args:
        config_name: User-facing config name for the current command.
        checkpoint_path: Checkpoint used by the current stage, if any.

    Returns:
        Tuple containing the experiment name and optional parent run ID.
    """
    if checkpoint_path is None:
        return generate_experiment_name(config_name), None

    metadata = load_run_metadata(checkpoint_path)
    if metadata is None:
        return generate_experiment_name(config_name), None
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


def build_training_metadata(
    logger: MLFlowLogger,
    experiment_name: str,
    config_name: str,
    work_dir: Path,
) -> dict[str, Any]:
    """Build persisted metadata for the current training run.

    Args:
        logger: Configured MLflow logger for the training run.
        experiment_name: MLflow experiment name for the run.
        config_name: User-facing config name for the run.
        work_dir: Hydra output directory for the run.

    Returns:
        Metadata dictionary persisted alongside run outputs.
    """
    return {
        "run_id": logger.run_id,
        "experiment_id": logger.experiment_id,
        "experiment_name": experiment_name,
        "config_name": config_name,
        "work_dir": str(work_dir),
    }
