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

import logging
from pathlib import Path
import os
from types import MappingProxyType
from typing import Any

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from autoware_ml.utils.mlflow_helpers import (
    AUTOWARE_ML_RUN_ID_ENV,
    load_run_context,
    prepare_run_context,
    MlflowRunContext,
    generate_experiment_name,
)

logger = logging.getLogger(__name__)


def build_mlflow_run_context(
    cfg: DictConfig,
    stage: str,
    experiment_name: str,
    experiment_uid: str,
    config_name: str,
    logger_enabled: bool,
    parent_run_id: str | None = None,
    extra_tags: MappingProxyType[str, Any] | None = None,
) -> MlflowRunContext:
    """
    Build an MLFlowRunContext from the Hydra configuration.

    Args:
        cfg: Hydra configuration
        stage: The stage of the run, e.g., "train" or "test".
        experiment_name: The name of the experiment.
        experiment_uid: A unique identifier for the experiment.
        config_name: The name of the user configuration (yaml file).
        logger_enabled: Whether the logger is enabled.

    Returns:
        An MLFlowRunContext object containing the run ID, experiment name, and user config name.
    """
    logger.info("Building MLflow run context...")
    work_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"Hydra work directory: {work_dir}")
    experiment_name = generate_experiment_name(experiment_name)
    if logger_enabled:
        pre_created_run_id = os.environ.get(AUTOWARE_ML_RUN_ID_ENV)
        if pre_created_run_id is not None:
            run_context = load_run_context(cfg.logger.tracking_uri, pre_created_run_id)
            if work_dir != run_context.hydra_dir:
                raise RuntimeError(
                    f"Hydra work directory '{work_dir}' does not match the pre-created MLflow "
                    f"run directory '{run_context.hydra_dir}'."
                )
        else:
            run_name = f"{stage}_{experiment_name}_{experiment_uid}"
            run_context = prepare_run_context(
                cfg.logger.tracking_uri,
                config_name=config_name,
                hydra_dir=work_dir,
                stage=stage,
                experiment_name=experiment_name,
                run_name=run_name,
                parent_run_id=parent_run_id,
                extra_tags=extra_tags,
            )
    else:
        run_context = None

    logger.info(f"MLflow run context built successfully with run context: {run_context}.")
    return run_context
