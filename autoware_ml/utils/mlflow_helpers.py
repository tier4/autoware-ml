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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from hydra.core.hydra_config import HydraConfig
from mlflow.entities import Param
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf, open_dict

RUN_METADATA_FILENAME = "run_metadata.json"
MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"
MLFLOW_RUN_NAME = "mlflow.runName"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ARTIFACTS_DIRNAME = "config"
AUTOWARE_ML_RUN_ID_ENV = "AUTOWARE_ML_RUN_ID"
AUTOWARE_ML_HYDRA_RUN_DIR_ENV = "AUTOWARE_ML_HYDRA_RUN_DIR"


@dataclass(frozen=True)
class MlflowRunContext:
    """Resolved MLflow runtime context for one stage run."""

    tracking_uri: str
    experiment_id: str
    experiment_name: str
    run_id: str
    run_name: str
    tags: dict[str, str]
    artifact_uri: str
    artifact_dir: Path
    hydra_dir: Path

    @property
    def checkpoints_dir(self) -> Path:
        """Directory used for model checkpoints."""
        return self.artifact_dir / "checkpoints"

    @property
    def exports_dir(self) -> Path:
        """Directory used for deployment exports."""
        return self.artifact_dir / "exports"

    @property
    def config_dir(self) -> Path:
        """Directory used for persisted run configs."""
        return self.artifact_dir / CONFIG_ARTIFACTS_DIRNAME


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


def generate_run_name(
    config_name: str,
    stage: str,
    started_at: datetime | None = None,
) -> str:
    """Generate a readable MLflow run name.

    Args:
        config_name: User-facing config name.
        stage: Pipeline stage such as ``train``, ``test``, or ``deploy``.
        started_at: Timestamp used to derive a readable run label.

    Returns:
        Human-readable run name containing the stage, config leaf, and date.
    """
    config_leaf = config_name.split("/")[-1]
    timestamp = started_at or datetime.now().astimezone()
    date_part = timestamp.strftime("%Y-%m-%d")
    time_part = timestamp.strftime("%H-%M-%S")
    return f"{stage}:{config_leaf}:{date_part}/{time_part}"


def build_run_tags(
    config_name: str,
    hydra_dir: Path,
    stage: str,
    artifact_dir: Path | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build the default MLflow tag set.

    Args:
        config_name: User-facing config name.
        hydra_dir: Hydra output directory for the run.
        stage: Pipeline stage such as ``train``, ``test``, or ``deploy``.
        artifact_dir: Optional MLflow artifact directory for the run.
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
        "hydra_dir": str(hydra_dir),
        "hostname": socket.gethostname(),
        "git_sha": get_git_sha(),
    }
    if artifact_dir is not None:
        tags["artifact_dir"] = str(artifact_dir)
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
        logger_cfg.tracking_uri = resolve_tracking_uri(logger_cfg.tracking_uri)
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
    """Persist run metadata into the chosen run-owned directory.

    Args:
        work_dir: Run-owned directory receiving the metadata file.
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


def resolve_tracking_uri(tracking_uri: str) -> str:
    """Resolve relative SQLite tracking URIs against the current working directory."""
    sqlite_prefix = "sqlite:///"
    if not tracking_uri.startswith(sqlite_prefix):
        return tracking_uri

    raw_path = tracking_uri.removeprefix(sqlite_prefix)
    db_path = Path(raw_path)
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    return f"{sqlite_prefix}{db_path.resolve()}"


def resolve_tracking_store_dir(tracking_uri: str) -> Path:
    """Return the local filesystem directory backing the MLflow store."""
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri)
    sqlite_prefix = "sqlite:///"
    if not resolved_tracking_uri.startswith(sqlite_prefix):
        raise ValueError(
            "Autoware-ML requires a local SQLite MLflow tracking store for direct artifact writes."
        )
    db_path = Path(resolved_tracking_uri.removeprefix(sqlite_prefix))
    return db_path.parent.resolve()


def generate_artifact_location(config_name: str, tracking_uri: str) -> str:
    """Return the semantic experiment artifact root URI for one config."""
    artifact_root = resolve_tracking_store_dir(tracking_uri) / config_name
    return artifact_root.resolve().as_uri()


def generate_hydra_run_dir(
    config_name: str,
    tracking_uri: str | None = None,
    run_id: str | None = None,
    started_at: datetime | None = None,
) -> Path:
    """Return the Hydra output directory for one command invocation."""
    if tracking_uri is not None:
        root_dir = resolve_tracking_store_dir(tracking_uri)
    else:
        root_dir = REPO_ROOT / "mlruns"
    config_root = (root_dir / config_name).resolve()
    if run_id is not None:
        return config_root / run_id / "hydra"

    timestamp = started_at or datetime.now().astimezone()
    return config_root / "_hydra" / timestamp.strftime("%Y-%m-%d") / timestamp.strftime("%H-%M-%S")


def artifact_uri_to_path(artifact_uri: str) -> Path:
    """Convert a local MLflow artifact URI into a filesystem path."""
    parsed = urlparse(artifact_uri)
    if parsed.scheme in {"", "file"}:
        raw_path = unquote(parsed.path) if parsed.scheme == "file" else artifact_uri
        return Path(raw_path).resolve()
    raise ValueError(
        f"Unsupported MLflow artifact URI '{artifact_uri}'. "
        "Autoware-ML requires local filesystem-backed MLflow artifacts."
    )


def normalize_artifact_location(artifact_location: str) -> Path:
    """Normalize an MLflow artifact location into a local filesystem path."""
    try:
        return artifact_uri_to_path(artifact_location)
    except ValueError:
        return Path(artifact_location).resolve()


def ensure_experiment(
    client: MlflowClient,
    experiment_name: str,
    artifact_location: str,
) -> str:
    """Resolve an experiment and require the semantic artifact root."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return client.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )

    current_location = normalize_artifact_location(experiment.artifact_location)
    target_location = normalize_artifact_location(artifact_location)
    if current_location != target_location:
        raise RuntimeError(
            f"Experiment '{experiment_name}' already exists with artifact root "
            f"'{current_location}', expected '{target_location}'. "
            "Remove or reset the local MLflow store before using the new artifact layout."
        )
    return experiment.experiment_id


def prepare_run_context(
    tracking_uri: str,
    config_name: str,
    hydra_dir: Path | None,
    stage: str,
    parent_run_id: str | None = None,
    experiment_name: str | None = None,
    extra_tags: dict[str, Any] | None = None,
    started_at: datetime | None = None,
) -> MlflowRunContext:
    """Create an MLflow run and resolve its local artifact directory."""
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri)
    resolved_experiment_name = experiment_name or generate_experiment_name(config_name)
    artifact_location = generate_artifact_location(config_name, resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    experiment_id = ensure_experiment(
        client,
        resolved_experiment_name,
        artifact_location,
    )

    run_name = generate_run_name(config_name, stage, started_at)
    provisional_hydra_dir = hydra_dir or generate_hydra_run_dir(
        config_name,
        resolved_tracking_uri,
        started_at=started_at,
    )
    run_tags = build_run_tags(config_name, provisional_hydra_dir, stage, extra_tags=extra_tags)
    if parent_run_id is not None:
        run_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
    run_tags[MLFLOW_RUN_NAME] = run_name

    run = client.create_run(
        experiment_id=experiment_id,
        tags=run_tags,
        run_name=run_name,
    )
    artifact_uri = client.get_run(run.info.run_id).info.artifact_uri
    artifact_dir = artifact_uri_to_path(artifact_uri)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_hydra_dir = hydra_dir or generate_hydra_run_dir(
        config_name,
        resolved_tracking_uri,
        run_id=run.info.run_id,
    )
    resolved_tags = build_run_tags(
        config_name,
        resolved_hydra_dir,
        stage,
        artifact_dir=artifact_dir,
        extra_tags=extra_tags,
    )
    if parent_run_id is not None:
        resolved_tags[MLFLOW_PARENT_RUN_ID] = parent_run_id
    resolved_tags[MLFLOW_RUN_NAME] = run_name
    for key, value in resolved_tags.items():
        client.set_tag(run.info.run_id, key, value)

    context = MlflowRunContext(
        tracking_uri=resolved_tracking_uri,
        experiment_id=experiment_id,
        experiment_name=resolved_experiment_name,
        run_id=run.info.run_id,
        run_name=run_name,
        tags=resolved_tags,
        artifact_uri=artifact_uri,
        artifact_dir=artifact_dir,
        hydra_dir=resolved_hydra_dir,
    )
    return context


def load_run_context(tracking_uri: str, run_id: str) -> MlflowRunContext:
    """Load an existing MLflow run context."""
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    run = client.get_run(run_id)
    experiment = client.get_experiment(run.info.experiment_id)
    artifact_dir = artifact_uri_to_path(run.info.artifact_uri)
    hydra_dir_tag = run.data.tags.get("hydra_dir")
    if hydra_dir_tag is None:
        raise RuntimeError(f"MLflow run '{run_id}' is missing the required 'hydra_dir' tag.")
    return MlflowRunContext(
        tracking_uri=resolved_tracking_uri,
        experiment_id=run.info.experiment_id,
        experiment_name=experiment.name,
        run_id=run.info.run_id,
        run_name=run.data.tags.get(
            MLFLOW_RUN_NAME, getattr(run.info, "run_name", None) or run.info.run_id
        ),
        tags=dict(run.data.tags),
        artifact_uri=run.info.artifact_uri,
        artifact_dir=artifact_dir,
        hydra_dir=Path(hydra_dir_tag),
    )


def build_run_metadata(
    run_context: MlflowRunContext,
    config_name: str,
    hydra_dir: Path,
    stage: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build persisted metadata for any MLflow-backed stage run."""
    metadata = {
        "run_id": run_context.run_id,
        "experiment_id": run_context.experiment_id,
        "experiment_name": run_context.experiment_name,
        "config_name": config_name,
        "hydra_dir": str(hydra_dir),
        "artifact_dir": str(run_context.artifact_dir),
        "stage": stage,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def write_run_config_artifacts(cfg: DictConfig, artifact_dir: Path) -> None:
    """Persist the resolved Hydra config and CLI overrides into MLflow artifacts."""
    config_dir = artifact_dir / CONFIG_ARTIFACTS_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )
    overrides = HydraConfig.get().overrides.task
    overrides_text = "\n".join(overrides)
    if overrides_text:
        overrides_text += "\n"
    (config_dir / "overrides.txt").write_text(overrides_text, encoding="utf-8")
