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

"""Deployment entrypoint for exporting models to ONNX and TensorRT."""

import logging
import os
from pathlib import Path

import hydra
import lightning as L
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import torch

from autoware_ml.utils.checkpoints import load_model_from_checkpoint
from autoware_ml.utils.deploy import (
    build_tensorrt_engine,
    export_to_onnx,
    modify_onnx_graph,
    resolve_export_spec,
    resolve_output_paths,
    should_export_stage,
    should_modify_graph,
    supports_export_stage,
    validate_cuda_available,
)
from autoware_ml.utils.mlflow_helpers import (
    AUTOWARE_ML_RUN_ID_ENV,
    build_run_metadata,
    get_user_config_name,
    load_run_context,
    log_config_params,
    prepare_run_context,
    resolve_lineage_context,
    should_enable_logger,
    write_run_config_artifacts,
    write_run_metadata,
)
from autoware_ml.utils.runtime import (
    configure_torch_runtime,
    get_config_path,
    log_configuration,
    resolve_work_dir,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = get_config_path()


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Export a configured model checkpoint for deployment."""
    if "checkpoint" not in cfg:
        raise ValueError(
            "Checkpoint must be specified (e.g., +checkpoint=path/to/checkpoint.ckpt)."
        )
    if "deploy" not in cfg:
        raise ValueError("Config must define a 'deploy' section.")

    log_configuration(cfg)
    work_dir = resolve_work_dir()
    config_name = get_user_config_name()
    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger_enabled = should_enable_logger(cfg)
    mlflow_client: MlflowClient | None = None
    deploy_run_id: str | None = None
    experiment_name: str | None = None
    parent_run_id: str | None = None

    if logger_enabled:
        experiment_name, parent_run_id = resolve_lineage_context(config_name, checkpoint_path)
        pre_created_run_id = os.environ.get(AUTOWARE_ML_RUN_ID_ENV)
        if pre_created_run_id is not None:
            run_context = load_run_context(cfg.logger.tracking_uri, pre_created_run_id)
            if work_dir != run_context.hydra_dir:
                raise RuntimeError(
                    f"Hydra work directory '{work_dir}' does not match the pre-created MLflow "
                    f"run directory '{run_context.hydra_dir}'."
                )
        else:
            run_context = prepare_run_context(
                cfg.logger.tracking_uri,
                config_name,
                hydra_dir=work_dir,
                stage="deploy",
                parent_run_id=parent_run_id,
                experiment_name=experiment_name,
                extra_tags={
                    "checkpoint_path": str(checkpoint_path),
                    "source_run_id": parent_run_id or "",
                },
            )
        mlflow_client = MlflowClient(tracking_uri=run_context.tracking_uri)
        deploy_run_id = run_context.run_id
        experiment_name = run_context.experiment_name
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
                },
            ),
        )
    else:
        run_context = None

    validate_cuda_available()
    configure_torch_runtime()

    device = torch.device("cuda")
    logger.info("Using device: %s", device)
    logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    configured_output_dir = cfg.get("output_dir", None)
    if run_context is not None and configured_output_dir is None:
        configured_output_dir = str(run_context.exports_dir)
    output_dir, onnx_path, engine_path = resolve_output_paths(
        checkpoint_path,
        cfg.get("output_name", None),
        configured_output_dir,
    )
    if run_context is not None and not output_dir.resolve().is_relative_to(
        run_context.artifact_dir
    ):
        raise ValueError(
            "When MLflow logging is enabled, deployment outputs must stay inside the MLflow "
            f"artifact directory '{run_context.artifact_dir}'. "
            "Leave output_dir unset to use the default exports directory."
        )
    logger.info("Output directory: %s", output_dir)
    logger.info("ONNX output: %s", onnx_path)
    logger.info("TensorRT engine output: %s", engine_path)

    deploy_cfg = cfg.deploy
    try:
        if mlflow_client is not None and deploy_run_id is not None:
            log_config_params(
                mlflow_client,
                deploy_run_id,
                OmegaConf.to_container(cfg, resolve=True),
            )

        logger.info("Instantiating datamodule...")
        datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

        logger.info("Instantiating model...")
        model: L.LightningModule = hydra.utils.instantiate(cfg.model)

        logger.info("Loading checkpoint: %s", checkpoint_path)
        load_model_from_checkpoint(
            model,
            checkpoint_path,
            map_location=device,
            device=device,
            set_eval=True,
        )

        logger.info("Preparing export inputs...")
        export_spec = resolve_export_spec(datamodule, model, device)
        onnx_exported = False
        tensorrt_exported = False

        if should_export_stage(deploy_cfg.onnx):
            if not supports_export_stage(export_spec, "onnx"):
                logger.warning(
                    "Skipping ONNX export: model does not support the ONNX deploy stage."
                )
            else:
                export_to_onnx(
                    export_spec.module,
                    export_spec.args,
                    deploy_cfg,
                    export_spec.input_param_names,
                    export_spec.output_names,
                    onnx_path,
                )
                onnx_exported = True

                modify_graph_cfg = deploy_cfg.onnx.get("modify_graph", None)
                if should_modify_graph(modify_graph_cfg):
                    onnx_path = modify_onnx_graph(onnx_path, modify_graph_cfg)

        if should_export_stage(deploy_cfg.tensorrt):
            if not supports_export_stage(export_spec, "tensorrt"):
                logger.warning(
                    "Skipping TensorRT export: model export path does not support TensorRT."
                )
            else:
                if not onnx_path.exists():
                    raise FileNotFoundError(
                        f"ONNX file not found: {onnx_path}. TensorRT export requires a valid ONNX model."
                    )
                build_tensorrt_engine(onnx_path, deploy_cfg, engine_path)
                tensorrt_exported = True

        if mlflow_client is not None and deploy_run_id is not None:
            mlflow_client.set_terminated(
                deploy_run_id,
                status=RunStatus.to_string(RunStatus.FINISHED),
            )

        logger.info("Deployment completed successfully.")
        if onnx_exported and onnx_path.exists():
            logger.info("ONNX model: %s", onnx_path)
        if tensorrt_exported and engine_path.exists():
            logger.info("TensorRT engine: %s", engine_path)
    except Exception:
        if mlflow_client is not None and deploy_run_id is not None:
            mlflow_client.set_terminated(
                deploy_run_id,
                status=RunStatus.to_string(RunStatus.FAILED),
            )
        raise


if __name__ == "__main__":
    main()
