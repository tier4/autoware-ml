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

"""Deployment entrypoint for Autoware-ML models.

This script wires Hydra configuration, Lightning runtime setup, MLflow
integration, and model deployment execution.
"""

import logging
from types import MappingProxyType

import hydra
from hydra.core.hydra_config import HydraConfig
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch

from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.mlflow_builder import build_mlflow_run_context
from autoware_ml.builders.model_builder import (
    build_model,
    build_data_preprocessor,
    build_weight_checkpoint_paths,
)
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.deployment.deployment_export import DeploymentExport
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.utils.mlflow_helpers import (
    MlflowRunContext,
    build_run_metadata,
    log_config_params,
    resolve_deploy_lineage,
    write_run_config_artifacts,
    write_run_metadata,
)
from autoware_ml.utils.onnx_meta import release_to_model_version
from autoware_ml.utils.deploy import (
    validate_cuda_available,
)
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    restrict_console_logging_to_rank_zero,
    get_config_path,
    log_configuration,
    set_seed,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()
CONFIG_NAME_PREFIX = "tasks/"


def deploy(
    deploy_cfg: DictConfig,
    output_dir: Path,
    datamodule: DataModule,
    model: ModuleBaseModel,
    config_name: str,
    release: str | None,
    tracker: str | None = None,
    run_id: str | None = None,
) -> None:
    """Deploy the trained model for inference.

    Args:
        deploy_cfg: Hydra configuration for deployment.
        output_dir: Directory to save the exported models.
        datamodule: Data module providing the example batch used for tracing.
        model: The trained model to be deployed.
        config_name: User-facing config name stamped into the ONNX metadata.
        release: ``vMAJOR.MINOR.PATCH`` stamped into the ONNX metadata, or ``None``
            for an unversioned export.
        tracker: Active experiment tracker name recorded in the stamp, if any.
        run_id: Deploy run id recorded in the stamp, if any.
    """
    deployment_exporter = DeploymentExport(
        deploy_cfg=deploy_cfg,
        output_dir=output_dir,
        datamodule=datamodule,
        model=model,
        config_name=config_name,
        release=release,
        tracker=tracker,
        run_id=run_id,
    )
    export_outputs = deployment_exporter.export()
    logger.info("Deployment completed successfully.")
    for path in export_outputs.onnx_exported_paths:
        if path.exists():
            logger.info("ONNX module: %s", path)
    for path in export_outputs.tensorrt_exported_paths:
        if path.exists():
            logger.info("TensorRT engine: %s", path)


def record_deploy_run(
    cfg: DictConfig,
    run_context: MlflowRunContext,
    config_name: str,
    parent_run_id: str | None,
    checkpoint_path: Path,
    source_checkpoints: list[dict[str, str | None]],
) -> None:
    """Persist what a deploy run was made from into its MLflow artifacts.

    Writes the resolved config and the CLI overrides under ``config/``, the run metadata with
    the source checkpoints next to them, and logs the config as run parameters. Train and test
    get the same records from the trainer logger builder; deploy has no trainer, so it records
    them here.

    Args:
        cfg: Hydra configuration.
        run_context: The deploy run.
        config_name: User-facing config name.
        parent_run_id: Source training run the deploy nests under, if any.
        checkpoint_path: Primary checkpoint the export reads.
        source_checkpoints: Lineage metadata of every source checkpoint.
    """
    write_run_config_artifacts(cfg, run_context.artifact_dir)
    write_run_metadata(
        run_context.artifact_dir,
        build_run_metadata(
            run_context,
            config_name,
            run_context.hydra_dir,
            "deploy",
            extra_metadata={
                "source_run_id": parent_run_id,
                "checkpoint_path": str(checkpoint_path),
                "source_checkpoints": source_checkpoints,
            },
        ),
    )
    log_config_params(
        MlflowClient(tracking_uri=run_context.tracking_uri),
        run_context.run_id,
        OmegaConf.to_container(cfg, resolve=True),
    )


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint of the deployment.

    Args:
        cfg: Hydra configuration
    """
    run_deploy(cfg)


def run_deploy(cfg: DictConfig) -> None:
    """Export the configured model under its MLflow deploy run.

    The MLflow run, when logging is enabled, is marked FAILED if anything after its creation
    raises and FINISHED otherwise, so a crashed deploy never stays RUNNING.

    Args:
        cfg: Hydra configuration
    """
    # First, so that every rank after rank zero stays silent from the config dump onwards.
    restrict_console_logging_to_rank_zero()
    if "deploy" not in cfg:
        raise ValueError("Config must define a 'deploy' section.")

    release = cfg.get("release", None)
    release_to_model_version(release)  # a malformed release must fail before exporting
    if release is None:
        logger.warning(
            "Deploying without --release — artifacts will be stamped 'unversioned' "
            "(model_version 0). If this model is intended for production, re-run "
            "deploy with an explicit --release vMAJOR.MINOR.PATCH."
        )

    log_configuration(cfg)
    config_name = HydraConfig.get().job.config_name
    if config_name is None:
        raise ValueError("Hydra config name is not available.")
    config_name = config_name.removeprefix(CONFIG_NAME_PREFIX)

    # Configure weights and checkpoint paths
    weights_path, checkpoint_path = build_weight_checkpoint_paths(cfg)

    # Nest the deploy run under the source training run, when there is exactly one
    parent_run_id, source_checkpoints = resolve_deploy_lineage(weights_path)
    source_run_ids = [
        source["run_id"] for source in source_checkpoints if source["run_id"] is not None
    ]
    logger_enabled = cfg.get("logger") is not None
    run_context = build_mlflow_run_context(
        cfg,
        stage="deploy",
        config_name=config_name,
        logger_enabled=logger_enabled,
        parent_run_id=parent_run_id,
        extra_tags=MappingProxyType(
            {
                "checkpoint_path": str(checkpoint_path),
                "source_run_id": parent_run_id or "",
                "source_checkpoint_count": str(len(source_checkpoints)),
                "source_run_ids": ",".join(source_run_ids),
            }
        ),
    )

    if run_context is None:
        _export(cfg, config_name, weights_path, release, run_context=None)
        return

    try:
        record_deploy_run(
            cfg,
            run_context,
            config_name,
            parent_run_id=parent_run_id,
            checkpoint_path=checkpoint_path,
            source_checkpoints=source_checkpoints,
        )
        _export(cfg, config_name, weights_path, release, run_context=run_context)
    except Exception:
        MlflowClient(tracking_uri=run_context.tracking_uri).set_terminated(
            run_context.run_id, status=RunStatus.to_string(RunStatus.FAILED)
        )
        raise
    MlflowClient(tracking_uri=run_context.tracking_uri).set_terminated(
        run_context.run_id, status=RunStatus.to_string(RunStatus.FINISHED)
    )


def _export(
    cfg: DictConfig,
    config_name: str,
    weights_path: list[Path],
    release: str | None,
    run_context: MlflowRunContext | None,
) -> None:
    """Build the data module and the model, then export under the run's exports directory.

    Args:
        cfg: Hydra configuration.
        config_name: User-facing config name.
        weights_path: Checkpoints the model is built from.
        release: ``vMAJOR.MINOR.PATCH`` of the export, or ``None``.
        run_context: The deploy run, whose ``exports`` directory receives the outputs. Without
            one the outputs land in ``experiment_run_dir`` of the config.
    """
    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)

    device = torch.device("cuda")
    logger.info("Using device: %s", device)
    logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    if run_context is not None:
        output_dir = run_context.exports_dir
    elif cfg.get("experiment_run_dir") is not None:
        output_dir = Path(cfg.experiment_run_dir)
    else:
        raise ValueError(
            "Output directory must be specified in the configuration or obtained from MLflow run context."
        )
    logger.info("Output directory for deployment: %s", output_dir)

    # Build datamodule
    datamodule = build_datamodule(cfg)

    # Build model
    data_preprocessor = build_data_preprocessor(cfg)
    model = build_model(
        cfg,
        data_preprocessor=data_preprocessor,
        weights_path=weights_path,
        resume_checkpoint_path=None,
        device=device,
        set_eval=True,
        enforce_full_coverage=True,
    )

    deploy(
        deploy_cfg=cfg.deploy,
        output_dir=output_dir,
        datamodule=datamodule,
        model=model,
        config_name=config_name,
        release=release,
        tracker="mlflow" if run_context is not None else None,
        run_id=run_context.run_id if run_context is not None else None,
    )


if __name__ == "__main__":
    main()
