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
import logging
import re
import socket
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence
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
MLFLOW_PARAM_KEY_REPLACEMENTS = (
    ("&", "and"),
    ("(", ""),
    (")", ""),
)
MLFLOW_PARAM_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.\- /:]+$")


_logger = logging.getLogger(__name__)


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


def generate_experiment_name(experiment_group_name: str, experiment_name: str) -> str:
    """Generate the MLflow experiment name of a config.

    The experiment is ``<experiment_group_name>/<experiment_name>``, the two bookkeeping keys
    every task config defines. It is the single key of a config's MLflow presence: the
    experiment, its artifact root under the tracking store and the Hydra run directories all
    derive from it, so two configs that share both keys share one experiment, and the CLI, the
    entrypoints and deploy all land in the same place.

    Args:
        experiment_group_name: ``experiment_group_name`` of the config, such as
            ``centerpoint/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base``.
        experiment_name: ``experiment_name`` of the config, the leaf naming the variant.

    Returns:
        The ``<experiment_group_name>/<experiment_name>`` experiment name.

    Raises:
        ValueError: If either part is empty.
    """
    group = experiment_group_name.strip("/")
    name = experiment_name.strip("/")
    if not group or not name:
        raise ValueError(
            "Both experiment_group_name and experiment_name must be non-empty, got "
            f"{experiment_group_name!r} and {experiment_name!r}."
        )
    return f"{group}/{name}"


def resolve_experiment_name(cfg: DictConfig) -> str:
    """Resolve the MLflow experiment name from a composed config.

    Args:
        cfg: Composed task config carrying ``experiment_group_name`` and ``experiment_name``.

    Returns:
        The experiment name built by :func:`generate_experiment_name`.

    Raises:
        ValueError: If the config does not define both keys, which every task config must.
    """
    missing = [
        key
        for key in ("experiment_group_name", "experiment_name")
        if key not in cfg or OmegaConf.is_missing(cfg, key)
    ]
    if missing:
        raise ValueError(
            f"The config does not define {missing}, so its MLflow experiment cannot be named. "
            "Every task config must set experiment_group_name and experiment_name."
        )
    return generate_experiment_name(str(cfg.experiment_group_name), str(cfg.experiment_name))


def generate_run_name(stage: str, experiment_name: str, experiment_uid: str) -> str:
    """Generate the MLflow run name of one stage run.

    Args:
        stage: Pipeline stage such as ``train``, ``test``, or ``deploy``.
        experiment_name: Experiment name from :func:`generate_experiment_name`.
        experiment_uid: ``experiment_uid`` of the config, a timestamp by default.

    Returns:
        ``<stage>_<experiment_name leaf>_<experiment_uid>``. The run lives inside its
        experiment, so only the leaf of the experiment name is repeated.
    """
    experiment_leaf = experiment_name.rsplit("/", 1)[-1]
    return f"{stage}_{experiment_leaf}_{experiment_uid}"


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
    _logger.warning(
        "Could not resolve git SHA: %s. Provenance will be recorded as 'unknown'.",
        result.stderr.strip(),
    )
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
        ``True`` when a logger configuration is present.
    """
    return cfg.get("logger") is not None


def sanitize_mlflow_param_key(key: str) -> str:
    """Return a key accepted by MLflow parameter logging.

    Args:
        key: Original config key.

    Returns:
        Sanitized MLflow parameter key.
    """
    sanitized_key = key
    for source, target in MLFLOW_PARAM_KEY_REPLACEMENTS:
        sanitized_key = sanitized_key.replace(source, target)
    if MLFLOW_PARAM_KEY_PATTERN.fullmatch(sanitized_key) is None:
        raise ValueError(
            f"Invalid MLflow parameter key after sanitization: {sanitized_key} (original: {key})"
        )
    return sanitized_key


def sanitize_mlflow_param_keys(value: Any) -> Any:
    """Sanitize mapping keys before sending parameters to MLflow.

    Args:
        value: Resolved configuration value.

    Returns:
        Configuration value with MLflow-compatible mapping keys.
    """
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            sanitized_key = sanitize_mlflow_param_key(str(key))
            if sanitized_key in sanitized:
                raise ValueError(
                    f"MLflow parameter key collision after sanitization: {sanitized_key}"
                )
            sanitized[sanitized_key] = sanitize_mlflow_param_keys(nested_value)
        return sanitized
    if isinstance(value, list):
        return [sanitize_mlflow_param_keys(item) for item in value]
    return value


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
    flattened = _flatten_params("", sanitize_mlflow_param_keys(cfg))
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


def build_source_checkpoint_metadata(
    checkpoint_paths: Sequence[Path],
) -> list[dict[str, str | None]]:
    """Return lineage metadata for all source checkpoint paths."""
    source_checkpoints = []
    for checkpoint_path in checkpoint_paths:
        metadata = load_run_metadata(checkpoint_path)
        source_checkpoints.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "run_id": metadata.get("run_id") if metadata is not None else None,
                "experiment_name": metadata.get("experiment_name")
                if metadata is not None
                else None,
                "config_name": metadata.get("config_name") if metadata is not None else None,
                "stage": metadata.get("stage") if metadata is not None else None,
            }
        )
    return source_checkpoints


def resolve_deploy_lineage(
    checkpoint_paths: Sequence[Path],
) -> tuple[str | None, list[dict[str, str | None]]]:
    """Resolve the source checkpoint lineage of a deploy run.

    The deploy run belongs to the experiment of its own config, like every other stage. This
    only decides which source run it nests under.

    Args:
        checkpoint_paths: Checkpoints the deploy exports from.

    Returns:
        The parent run id, which is the source run when exactly one checkpoint carries run
        metadata and ``None`` otherwise, and the lineage metadata of every checkpoint.
    """
    source_checkpoints = build_source_checkpoint_metadata(checkpoint_paths)
    source_run_ids = [
        source["run_id"] for source in source_checkpoints if source["run_id"] is not None
    ]
    parent_run_id = source_run_ids[0] if len(source_run_ids) == 1 else None
    return parent_run_id, source_checkpoints


def resolve_parent_run_id(checkpoint_path: Path | None) -> str | None:
    """Resolve the run a follow-up stage nests under.

    Args:
        checkpoint_path: Checkpoint used by the current stage, if any.

    Returns:
        The run id recorded next to the checkpoint, or ``None`` without a checkpoint or
        without run metadata.
    """
    if checkpoint_path is None:
        return None
    metadata = load_run_metadata(checkpoint_path)
    if metadata is None:
        return None
    return metadata.get("run_id")


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


def generate_artifact_location(experiment_name: str, tracking_uri: str) -> str:
    """Return the artifact root URI of an experiment.

    Args:
        experiment_name: Experiment name from :func:`generate_experiment_name`.
        tracking_uri: SQLite tracking URI whose directory holds the artifacts.

    Returns:
        ``<store>/<experiment_name>`` as a file URI. The root is keyed on the experiment name,
        the same key as the experiment itself, so an experiment never meets a second root.
    """
    artifact_root = resolve_tracking_store_dir(tracking_uri) / experiment_name
    return artifact_root.resolve().as_uri()


def generate_hydra_run_dir(
    experiment_name: str,
    tracking_uri: str | None = None,
    run_id: str | None = None,
    started_at: datetime | None = None,
) -> Path:
    """Return the Hydra output directory for one command invocation.

    Args:
        experiment_name: Experiment name from :func:`generate_experiment_name`.
        tracking_uri: SQLite tracking URI whose directory holds the artifacts, or ``None`` to
            use ``mlruns/`` under the repository.
        run_id: MLflow run owning the directory, which then lives in the run's artifacts.
        started_at: Timestamp of a run without MLflow, which gets a dated scratch directory.

    Returns:
        ``<store>/<experiment_name>/<run_id>/artifacts/hydra`` with a run id, otherwise
        ``<store>/<experiment_name>/artifacts/_hydra/<date>/<time>``.
    """
    if tracking_uri is not None:
        root_dir = resolve_tracking_store_dir(tracking_uri)
    else:
        root_dir = REPO_ROOT / "mlruns"
    experiment_root = (root_dir / experiment_name).resolve()
    if run_id is not None:
        return experiment_root / run_id / "artifacts" / "hydra"

    timestamp = started_at or datetime.now().astimezone()
    return (
        experiment_root
        / "artifacts"
        / "_hydra"
        / timestamp.strftime("%Y-%m-%d")
        / timestamp.strftime("%H-%M-%S")
    )


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
    return artifact_uri_to_path(artifact_location)


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
    experiment_name: str,
    run_name: str,
    parent_run_id: str | None = None,
    extra_tags: dict[str, Any] | None = None,
    started_at: datetime | None = None,
) -> MlflowRunContext:
    """Create an MLflow run and resolve its local artifact directory.

    Args:
        tracking_uri: MLflow tracking URI.
        config_name: User-facing config name, recorded in the run tags.
        hydra_dir: Hydra output directory of the run, or ``None`` to place it in the run's
            artifacts.
        stage: Pipeline stage such as ``train``, ``test``, or ``deploy``.
        experiment_name: Experiment name from :func:`generate_experiment_name`. It keys the
            experiment, its artifact root and the Hydra directory alike.
        run_name: Run name from :func:`generate_run_name`.
        parent_run_id: Run this one nests under, if any.
        extra_tags: Additional tags merged into the defaults.
        started_at: Timestamp of the dated scratch directory used before the run exists.

    Returns:
        The context of the created run.
    """
    resolved_tracking_uri = resolve_tracking_uri(tracking_uri)
    resolved_experiment_name = experiment_name
    artifact_location = generate_artifact_location(experiment_name, resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    experiment_id = ensure_experiment(
        client,
        resolved_experiment_name,
        artifact_location,
    )

    provisional_hydra_dir = hydra_dir or generate_hydra_run_dir(
        experiment_name,
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
        experiment_name,
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


def mark_run_running(tracking_uri: str, run_id: str) -> None:
    """Set a reused run's status back to RUNNING while it is resumed.

    Args:
        tracking_uri: MLflow tracking URI.
        run_id: Run whose status is reset.
    """
    client = MlflowClient(tracking_uri=resolve_tracking_uri(tracking_uri))
    client.update_run(run_id, status="RUNNING")


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
