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
from autoware_ml.builders.logger_builder import build_trainer_logger
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.deployment.deployment_export import DeploymentExport
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.utils.mlflow_helpers import resolve_deploy_lineage, log_config_params
from autoware_ml.utils.onnx_meta import release_to_model_version
from autoware_ml.utils.deploy import (
    validate_cuda_available,
)
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    restrict_console_logging_to_rank_zero,
    get_config_path,
    log_configuration,
    log_hyperparameters,
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


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    """Main training function.

    Args:
        cfg: Hydra configuration
    """
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

    # Configure experiment name, parent_run_id and source checkpoints for MLflow
    experiment_name, parent_run_id, source_checkpoints = resolve_deploy_lineage(
        config_name,
        weights_path,
    )
    source_run_ids = [
        source["run_id"] for source in source_checkpoints if source["run_id"] is not None
    ]
    logger_enabled = cfg.get("logger") is not None
    run_context = build_mlflow_run_context(
        cfg,
        stage="deploy",
        experiment_name=experiment_name,
        config_name=config_name,
        experiment_uid=cfg.experiment_uid,
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

    if run_context is not None:
        mlflow_client = MlflowClient(tracking_uri=run_context.tracking_uri)
    else:
        mlflow_client = None

    validate_cuda_available()
    configure_torch_runtime()
    restrict_console_logging_to_rank_zero()
    set_seed(cfg)

    device = torch.device("cuda")
    logger.info("Using device: %s", device)
    logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    output_dir = cfg.get("experiment_run_dir", None)
    if run_context is not None:
        output_dir = str(run_context.exports_dir)

    if output_dir is None:
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

    if mlflow_client is not None and run_context is not None:
        log_config_params(
            mlflow_client,
            run_context.run_id,
            OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        # Build trainer logger
        trainer_logger = build_trainer_logger(
            cfg,
            ml_flow_run_context=run_context,
            stage="deploy",
            config_name=config_name,
            logger_enabled=logger_enabled,
            extra_metadata=MappingProxyType(
                {
                    "source_run_id": parent_run_id,
                    "checkpoint_path": str(checkpoint_path),
                    "source_checkpoints": source_checkpoints,
                }
            ),
        )
        log_hyperparameters(cfg, trainer_logger)

    deploy(
        deploy_cfg=cfg.deploy,
        output_dir=Path(output_dir),
        datamodule=datamodule,
        model=model,
        config_name=config_name,
        release=release,
        tracker="mlflow" if run_context is not None else None,
        run_id=run_context.run_id if run_context is not None else None,
    )

    if mlflow_client is not None and run_context is not None:
        mlflow_client.set_terminated(
            run_context.run_id,
            status=RunStatus.to_string(RunStatus.FINISHED),
        )
