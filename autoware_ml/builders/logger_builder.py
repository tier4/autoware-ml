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
from typing import Any
from types import MappingProxyType

from hydra.utils import instantiate
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from autoware_ml.utils.mlflow_helpers import (
    MlflowRunContext,
    write_run_config_artifacts,
    write_run_metadata,
    configure_logger,
    build_run_metadata,
)

logger = logging.getLogger(__name__)


def build_trainer_logger(
    cfg: DictConfig,
    ml_flow_run_context: MlflowRunContext,
    stage: str,
    config_name: str,
    logger_enabled: bool,
    extra_metadata: MappingProxyType[str, Any] | None = None,
) -> Logger | None:
    """
    Build an MLFlowRunContext from the Hydra configuration.

    Args:
        cfg: Hydra configuration.
        ml_flow_run_context: An existing MLflowRunContext object.
        stage: The stage of the run, e.g., "train" or "test".
        config_name: The name of the user configuration (yaml file).
        logger_enabled: Whether the logger is enabled.

    Returns:
        An MLFlowRunContext object containing the run ID, experiment name, and user config name.
    """
    logger.info("Building trainer logger...")
    trainer_logger = None
    if not logger_enabled:
        logger.info("Logger is not enabled in the configuration.")
        return trainer_logger

    write_run_config_artifacts(cfg, ml_flow_run_context.artifact_dir)
    write_run_metadata(
        ml_flow_run_context.artifact_dir,
        build_run_metadata(
            ml_flow_run_context,
            config_name,
            ml_flow_run_context.hydra_dir,
            stage,
            extra_metadata=extra_metadata,
        ),
    )
    # Update cfg.logger with the run_id from ml_flow_run_context
    configure_logger(
        cfg.logger,
        ml_flow_run_context.experiment_name,
        ml_flow_run_context.run_name,
        ml_flow_run_context.tags,
        run_id=ml_flow_run_context.run_id,
    )
    trainer_logger = instantiate(cfg.logger)
    logger.info(f"Trainer logger built: {trainer_logger}")
    return trainer_logger
