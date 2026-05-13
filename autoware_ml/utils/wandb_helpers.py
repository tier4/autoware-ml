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

"""Shared Weights & Biases helpers for Autoware-ML scripts."""

from __future__ import annotations

import json
import socket
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

RUN_METADATA_FILENAME = "run_metadata.json"
CONFIG_ARTIFACTS_DIRNAME = "config"
DATASET_ARTIFACTS_DIRNAME = "datasets"
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
WANDB_TAG_KEYS = (
    "tracking_backend",
    "stage",
    "task",
    "model",
    "config_variant",
)
WANDB_MAX_TAG_LENGTH = 64


@dataclass(frozen=True)
class WandbRunContext:
    """Resolved W&B runtime context for one stage run."""

    project: str
    entity: str | None
    run_id: str
    run_name: str
    group: str
    job_type: str
    tags: dict[str, str]
    save_dir: Path
    artifact_dir: Path
    hydra_dir: Path

    @property
    def experiment_name(self) -> str:
        """Project name used as the experiment name equivalent."""
        return self.project

    @property
    def experiment_id(self) -> str:
        """Project name used as the experiment ID equivalent."""
        return self.project

    @property
    def artifact_uri(self) -> str:
        """Local staging artifact URI."""
        return self.artifact_dir.as_uri()

    @property
    def tracking_uri(self) -> str:
        """Readable W&B tracking URI."""
        owner = f"{self.entity}/" if self.entity else ""
        return f"wandb://{owner}{self.project}"

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

    @property
    def datasets_dir(self) -> Path:
        """Directory used for persisted dataset metadata."""
        return self.artifact_dir / DATASET_ARTIFACTS_DIRNAME


def get_user_config_name() -> str:
    """Return the user-facing config name without the ``tasks/`` prefix."""
    return HydraConfig.get().job.config_name.removeprefix("tasks/")


def generate_run_name(
    config_name: str,
    stage: str,
    started_at: datetime | None = None,
) -> str:
    """Generate a readable W&B run name."""
    config_leaf = config_name.split("/")[-1]
    timestamp = started_at or datetime.now().astimezone()
    return f"{stage}:{config_leaf}:{timestamp.strftime('%Y-%m-%d')}/{timestamp.strftime('%H-%M-%S')}"


def get_git_sha() -> str:
    """Return the current Git revision."""
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


def build_run_tags(
    config_name: str,
    hydra_dir: Path,
    stage: str,
    artifact_dir: Path | None = None,
    extra_tags: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build the default W&B tag metadata set."""
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
        "tracking_backend": "wandb",
    }
    if artifact_dir is not None:
        tags["artifact_dir"] = str(artifact_dir)
    if extra_tags:
        tags.update({key: str(value) for key, value in extra_tags.items()})
    return tags


def resolve_save_dir(save_dir: str | None) -> Path:
    """Resolve W&B local save directory."""
    path = Path(save_dir or "wandb_runs")
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def generate_hydra_run_dir(
    config_name: str,
    save_dir: str | None = None,
    run_id: str | None = None,
    started_at: datetime | None = None,
) -> Path:
    """Return the Hydra output directory for one W&B-backed command."""
    root = resolve_save_dir(save_dir) / config_name
    if run_id is not None:
        return root / run_id / "hydra"
    timestamp = started_at or datetime.now().astimezone()
    return root / "_hydra" / timestamp.strftime("%Y-%m-%d") / timestamp.strftime("%H-%M-%S")


def generate_artifact_dir(config_name: str, save_dir: str | None, run_id: str) -> Path:
    """Return the local artifact staging directory for a W&B run."""
    return resolve_save_dir(save_dir) / config_name / run_id / "artifacts"


def prepare_run_context(
    logger_cfg: DictConfig,
    config_name: str,
    hydra_dir: Path | None,
    stage: str,
    parent_run_id: str | None = None,
    extra_tags: dict[str, Any] | None = None,
    started_at: datetime | None = None,
) -> WandbRunContext:
    """Create a local W&B run context without contacting the W&B service."""
    run_id = str(logger_cfg.get("id") or logger_cfg.get("version") or uuid.uuid4().hex[:8])
    save_dir = resolve_save_dir(logger_cfg.get("save_dir", None))
    resolved_hydra_dir = hydra_dir or generate_hydra_run_dir(
        config_name,
        str(save_dir),
        run_id=run_id,
        started_at=started_at,
    )
    artifact_dir = generate_artifact_dir(config_name, str(save_dir), run_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    project = str(logger_cfg.get("project", "autoware-ml"))
    entity_value = logger_cfg.get("entity", None)
    entity = None if entity_value in {None, "null", ""} else str(entity_value)
    run_name = generate_run_name(config_name, stage, started_at)
    tags = build_run_tags(
        config_name,
        resolved_hydra_dir,
        stage,
        artifact_dir=artifact_dir,
        extra_tags={
            **({"source_run_id": parent_run_id} if parent_run_id else {}),
            **(extra_tags or {}),
        },
    )
    return WandbRunContext(
        project=project,
        entity=entity,
        run_id=run_id,
        run_name=run_name,
        group=config_name,
        job_type=stage,
        tags=tags,
        save_dir=save_dir,
        artifact_dir=artifact_dir,
        hydra_dir=resolved_hydra_dir,
    )


def load_run_context(logger_cfg: DictConfig, run_id: str) -> WandbRunContext:
    """Load a pre-created W&B context from config and environment metadata."""
    hydra_dir = Path(logger_cfg.get("hydra_dir"))
    config_name = str(logger_cfg.get("group", "autoware-ml"))
    artifact_dir = generate_artifact_dir(config_name, logger_cfg.get("save_dir", None), run_id)
    project = str(logger_cfg.get("project", "autoware-ml"))
    entity_value = logger_cfg.get("entity", None)
    entity = None if entity_value in {None, "null", ""} else str(entity_value)
    job_type = str(logger_cfg.get("job_type", "train"))
    return WandbRunContext(
        project=project,
        entity=entity,
        run_id=run_id,
        run_name=str(logger_cfg.get("name", run_id)),
        group=config_name,
        job_type=job_type,
        tags=build_run_tags(config_name, hydra_dir, job_type, artifact_dir=artifact_dir),
        save_dir=resolve_save_dir(logger_cfg.get("save_dir", None)),
        artifact_dir=artifact_dir,
        hydra_dir=hydra_dir,
    )


def configure_logger(logger_cfg: DictConfig, run_context: WandbRunContext) -> None:
    """Populate a Lightning ``WandbLogger`` configuration node."""
    existing_tags = list(logger_cfg.get("tags", []))
    metadata_tags = build_wandb_tags(run_context.tags)
    with open_dict(logger_cfg):
        logger_cfg.project = run_context.project
        logger_cfg.entity = run_context.entity
        logger_cfg.name = run_context.run_name
        logger_cfg.version = run_context.run_id
        logger_cfg.save_dir = str(run_context.save_dir)
        logger_cfg.group = run_context.group
        logger_cfg.job_type = run_context.job_type
        logger_cfg.tags = sorted(set(existing_tags + metadata_tags))
        logger_cfg.id = run_context.run_id


def build_wandb_tags(tags: dict[str, str]) -> list[str]:
    """Convert run metadata into short W&B UI tags."""
    wandb_tags = []
    for key in WANDB_TAG_KEYS:
        value = tags.get(key)
        if value:
            tag = f"{key}:{value}"
            if len(tag) <= WANDB_MAX_TAG_LENGTH:
                wandb_tags.append(tag)
    return sorted(set(wandb_tags))


def write_run_metadata(work_dir: Path, metadata: dict[str, Any]) -> Path:
    """Persist run metadata into the chosen run-owned directory."""
    metadata_path = work_dir / RUN_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def load_run_metadata(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load run metadata associated with a checkpoint path."""
    for parent in checkpoint_path.parents:
        metadata_path = parent / RUN_METADATA_FILENAME
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding="utf-8"))
    return None


def resolve_lineage_context(config_name: str, checkpoint_path: Path | None) -> tuple[str, str | None]:
    """Resolve W&B project and parent run for follow-up stages."""
    if checkpoint_path is None:
        return "autoware-ml", None
    metadata = load_run_metadata(checkpoint_path)
    if metadata is None or metadata.get("tracking_backend") != "wandb":
        return "autoware-ml", None
    return str(metadata.get("project", "autoware-ml")), metadata.get("run_id")


def build_run_metadata(
    run_context: WandbRunContext,
    config_name: str,
    hydra_dir: Path,
    stage: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build persisted metadata for any W&B-backed stage run."""
    metadata = {
        "tracking_backend": "wandb",
        "run_id": run_context.run_id,
        "project": run_context.project,
        "entity": run_context.entity,
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
    """Persist the resolved Hydra config and CLI overrides into local W&B artifacts."""
    config_dir = artifact_dir / CONFIG_ARTIFACTS_DIRNAME
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )
    overrides_text = "\n".join(HydraConfig.get().overrides.task)
    if overrides_text:
        overrides_text += "\n"
    (config_dir / "overrides.txt").write_text(overrides_text, encoding="utf-8")

    dataset_metadata = build_dataset_metadata(cfg)
    if dataset_metadata:
        datasets_dir = artifact_dir / DATASET_ARTIFACTS_DIRNAME
        datasets_dir.mkdir(parents=True, exist_ok=True)
        (datasets_dir / "metadata.json").write_text(
            json.dumps(dataset_metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def build_dataset_metadata(cfg: DictConfig) -> dict[str, Any]:
    """Extract lightweight dataset identity metadata from the datamodule config."""
    datamodule_cfg = cfg.get("datamodule")
    if datamodule_cfg is None:
        return {}
    resolved_cfg = OmegaConf.to_container(datamodule_cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        return {}
    return {
        ("datamodule_target" if key == "_target_" else key): value
        for key, value in resolved_cfg.items()
        if key in DATASET_METADATA_KEYS and value is not None
    }


def _log_path_artifact(
    run: Any,
    path: Path,
    name: str,
    artifact_type: str,
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a file or directory as a W&B artifact using a live W&B run."""
    if not path.exists():
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata)
    if path.is_dir():
        artifact.add_dir(str(path))
    else:
        artifact.add_file(str(path))
    run.log_artifact(artifact, aliases=aliases)


def log_stage_artifacts(run: Any, run_context: WandbRunContext, stage: str) -> None:
    """Log common stage artifacts to W&B."""
    artifact_base = f"autoware-ml-{stage}-{run_context.run_id}"
    _log_path_artifact(
        run,
        run_context.config_dir,
        f"{artifact_base}-config",
        "run_config",
        aliases=["latest"],
        metadata=run_context.tags,
    )
    _log_path_artifact(
        run,
        run_context.artifact_dir / RUN_METADATA_FILENAME,
        f"{artifact_base}-metadata",
        "run_metadata",
        aliases=["latest"],
        metadata=run_context.tags,
    )
    _log_path_artifact(
        run,
        run_context.datasets_dir,
        f"dataset-{run_context.group.replace('/', '-')}",
        "dataset",
        aliases=[stage, "latest"],
        metadata=run_context.tags,
    )
    if stage == "train":
        _log_path_artifact(
            run,
            run_context.checkpoints_dir,
            f"model-{run_context.run_id}",
            "model",
            aliases=["latest", "best"],
            metadata=run_context.tags,
        )
    if stage == "deploy":
        _log_path_artifact(
            run,
            run_context.exports_dir,
            f"deploy-{run_context.run_id}",
            "model",
            aliases=["latest", "candidate"],
            metadata=run_context.tags,
        )


def init_run(run_context: WandbRunContext, cfg: DictConfig) -> Any:
    """Initialize a W&B run for non-Lightning stages such as deploy."""
    import wandb

    return wandb.init(
        project=run_context.project,
        entity=run_context.entity,
        name=run_context.run_name,
        id=run_context.run_id,
        group=run_context.group,
        job_type=run_context.job_type,
        tags=build_wandb_tags(run_context.tags),
        dir=str(run_context.save_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
        resume="allow",
    )
