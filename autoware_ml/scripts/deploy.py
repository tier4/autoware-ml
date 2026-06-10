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
from autoware_ml.utils.checkpoints import apply_matching_weights
from autoware_ml.utils.deploy import (
    build_tensorrt_engine,
    export_to_onnx,
    merge_module_onnx_cfg,
    modify_onnx_graph,
    resolve_export_specs,
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
    resolve_deploy_lineage,
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
    weights_arg = cfg.get("weights", None)
    if weights_arg is None:
        raise ValueError("--weights <path> (repeatable) must be specified.")
    if "deploy" not in cfg:
        raise ValueError("Config must define a 'deploy' section.")

    log_configuration(cfg)
    work_dir = resolve_work_dir()
    config_name = get_user_config_name()

    weight_paths = (
        [Path(weights_arg)]
        if isinstance(weights_arg, str)
        else [Path(path) for path in weights_arg]
    )
    checkpoint_path = weight_paths[-1]
    for path in weight_paths:
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")

    logger_enabled = should_enable_logger(cfg)
    mlflow_client: MlflowClient | None = None
    deploy_run_id: str | None = None
    experiment_name: str | None = None
    parent_run_id: str | None = None
    source_checkpoints: list[dict[str, str | None]] = []

    if logger_enabled:
        experiment_name, parent_run_id, source_checkpoints = resolve_deploy_lineage(
            config_name,
            weight_paths,
        )
        source_run_ids = [
            source["run_id"] for source in source_checkpoints if source["run_id"] is not None
        ]
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
                    "source_checkpoint_count": str(len(source_checkpoints)),
                    "source_run_ids": ",".join(source_run_ids),
                },
            )
        mlflow_client = MlflowClient(tracking_uri=run_context.tracking_uri)
        deploy_run_id = run_context.run_id
        experiment_name = run_context.experiment_name
    else:
        run_context = None

    try:
        if run_context is not None:
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

        validate_cuda_available()
        configure_torch_runtime()

        device = torch.device("cuda")
        logger.info("Using device: %s", device)
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

        configured_output_dir = cfg.get("output_dir", None)
        if run_context is not None and configured_output_dir is None:
            configured_output_dir = str(run_context.exports_dir)
        output_dir, _, _ = resolve_output_paths(
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

        deploy_cfg = cfg.deploy
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
        model.set_data_preprocessing(hydra.utils.instantiate(cfg.data_preprocessing))

        logger.info(
            "Loading matching weights from %d checkpoint(s): %s", len(weight_paths), weight_paths
        )
        apply_matching_weights(
            model,
            weight_paths,
            map_location=device,
            device=device,
            set_eval=True,
            enforce_full_coverage=True,
            logger=logger,
        )

        logger.info("Preparing export inputs...")
        export_specs = resolve_export_specs(datamodule, model, device)
        onnx_exported_paths: list[Path] = []
        tensorrt_exported_paths: list[Path] = []

        for module_name, export_spec in export_specs.items():
            module_onnx_cfg = merge_module_onnx_cfg(deploy_cfg.onnx, module_name)
            module_onnx_path = output_dir / f"{module_name}.onnx"
            module_engine_path = output_dir / f"{module_name}.engine"

            if should_export_stage(deploy_cfg.onnx):
                if not supports_export_stage(export_spec, "onnx"):
                    raise RuntimeError(
                        f"Module '{module_name}' does not support ONNX export but "
                        "deploy.onnx.enabled=true. Disable the stage or use a supported model."
                    )
                else:
                    export_to_onnx(
                        export_spec.module,
                        export_spec.args,
                        module_onnx_cfg,
                        export_spec.input_param_names,
                        export_spec.output_names,
                        export_spec.dynamic_axes,
                        module_onnx_path,
                    )
                    onnx_exported_paths.append(module_onnx_path)

                    modify_graph_cfg = module_onnx_cfg.get("modify_graph", None)
                    if should_modify_graph(modify_graph_cfg):
                        module_onnx_path = modify_onnx_graph(module_onnx_path, modify_graph_cfg)

            if should_export_stage(deploy_cfg.tensorrt):
                if not supports_export_stage(export_spec, "tensorrt"):
                    raise RuntimeError(
                        f"Module '{module_name}' does not support TensorRT export but "
                        "deploy.tensorrt.enabled=true. Disable the stage or use a supported model."
                    )
                else:
                    if not module_onnx_path.exists():
                        raise FileNotFoundError(
                            f"ONNX file not found: {module_onnx_path}. "
                            "TensorRT export requires a valid ONNX model."
                        )
                    build_tensorrt_engine(module_onnx_path, deploy_cfg, module_engine_path)
                    tensorrt_exported_paths.append(module_engine_path)

    except Exception:
        if mlflow_client is not None and deploy_run_id is not None:
            mlflow_client.set_terminated(
                deploy_run_id,
                status=RunStatus.to_string(RunStatus.FAILED),
            )
        raise

    if mlflow_client is not None and deploy_run_id is not None:
        mlflow_client.set_terminated(
            deploy_run_id,
            status=RunStatus.to_string(RunStatus.FINISHED),
        )

    logger.info("Deployment completed successfully.")
    for path in onnx_exported_paths:
        if path.exists():
            logger.info("ONNX module: %s", path)
    for path in tensorrt_exported_paths:
        if path.exists():
            logger.info("TensorRT engine: %s", path)


if __name__ == "__main__":
    main()
