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
    validate_cuda_available,
)
from autoware_ml.utils.mlflow import (
    build_run_tags,
    generate_run_name,
    get_user_config_name,
    log_config_params,
    log_path_as_artifact,
    resolve_lineage_context,
    should_enable_logger,
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
    experiment_id: str | None = None
    experiment_name: str | None = None
    parent_run_id: str | None = None

    if logger_enabled:
        experiment_name, parent_run_id = resolve_lineage_context(config_name, checkpoint_path)
        run_name = generate_run_name(config_name, work_dir, "deploy")
        run_tags = build_run_tags(
            config_name,
            work_dir,
            "deploy",
            extra_tags={
                "checkpoint_path": str(checkpoint_path),
                "source_run_id": parent_run_id or "",
            },
        )
        mlflow_client = MlflowClient(tracking_uri=cfg.logger.tracking_uri)
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        experiment_id = (
            mlflow_client.create_experiment(experiment_name)
            if experiment is None
            else experiment.experiment_id
        )
        if parent_run_id is not None:
            run_tags["mlflow.parentRunId"] = parent_run_id
        run_tags["mlflow.runName"] = run_name
        deploy_run = mlflow_client.create_run(
            experiment_id=experiment_id,
            tags=run_tags,
            run_name=run_name,
        )
        deploy_run_id = deploy_run.info.run_id

    validate_cuda_available()
    configure_torch_runtime()

    device = torch.device("cuda")
    logger.info("Using device: %s", device)
    logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    output_dir, onnx_path, engine_path = resolve_output_paths(
        checkpoint_path,
        cfg.get("output_name", None),
        cfg.get("output_dir", None),
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

        if should_export_stage(deploy_cfg.onnx):
            export_to_onnx(
                export_spec.module,
                export_spec.args,
                deploy_cfg,
                export_spec.input_param_names,
                export_spec.output_names,
                onnx_path,
            )

            modify_graph_cfg = deploy_cfg.onnx.get("modify_graph", None)
            if should_modify_graph(modify_graph_cfg):
                onnx_path = modify_onnx_graph(onnx_path, modify_graph_cfg)

        if should_export_stage(deploy_cfg.tensorrt):
            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX file not found: {onnx_path}. TensorRT export requires a valid ONNX model."
                )
            build_tensorrt_engine(onnx_path, deploy_cfg, engine_path)

        metadata_path = write_run_metadata(
            work_dir,
            {
                "run_id": deploy_run_id,
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "config_name": config_name,
                "work_dir": str(work_dir),
                "stage": "deploy",
                "source_run_id": parent_run_id,
                "checkpoint_path": str(checkpoint_path),
            },
        )

        if mlflow_client is not None and deploy_run_id is not None:
            log_path_as_artifact(mlflow_client, deploy_run_id, work_dir / ".hydra", "hydra")
            log_path_as_artifact(mlflow_client, deploy_run_id, metadata_path, "metadata")
            log_path_as_artifact(mlflow_client, deploy_run_id, onnx_path, "exports")
            log_path_as_artifact(mlflow_client, deploy_run_id, engine_path, "exports")
            mlflow_client.set_terminated(
                deploy_run_id,
                status=RunStatus.to_string(RunStatus.FINISHED),
            )

        logger.info("Deployment completed successfully.")
        if onnx_path.exists():
            logger.info("ONNX model: %s", onnx_path)
        if engine_path.exists():
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
