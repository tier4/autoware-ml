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

"""Runtime helpers for Hydra-backed CLI commands.

This module is intentionally separate from ``autoware_ml.cli.cli`` so shell
completion can import the Typer app without pulling Hydra and MLflow into the
startup path.
"""

import __main__
import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from pathlib import Path

from hydra import compose, initialize_config_dir, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from autoware_ml.utils.cli.helpers import adjust_argv, resolve_config_reference, run_lazy_script
from autoware_ml.utils.mlflow_helpers import (
    AUTOWARE_ML_HYDRA_RUN_DIR_ENV,
    AUTOWARE_ML_RUN_ID_ENV,
    generate_experiment_name,
    generate_hydra_run_dir,
    prepare_run_context,
    resolve_lineage_context,
    should_enable_logger,
)

HYDRA_CONFIG_NAME_OPTION = "--config-name"
HYDRA_CONFIG_PATH_OPTION = "--config-path"
HYDRA_SEARCHPATH_PREFIX = "hydra.searchpath="
TASK_CONFIG_PREFIX = "tasks"


def resolve_module_spec(module_name: str) -> ModuleSpec:
    """Resolve a module spec for a runtime entrypoint."""
    module_spec = find_spec(module_name)
    if module_spec is None:
        raise RuntimeError(f"Could not resolve Hydra entrypoint module '{module_name}'.")
    return module_spec


def resolve_hydra_argv(
    config_value: str,
    config_prefix: str,
    extra_args: Sequence[str] = (),
    hydra_overrides: Sequence[str] = (),
) -> list[str]:
    """Rewrite CLI arguments into the Hydra invocation expected by scripts."""
    resolved_config_path, resolved_config_name, extra_config_overrides = resolve_config_reference(
        config_value, config_prefix
    )
    adjusted_args = adjust_argv(extra_args)
    hydra_argv = [HYDRA_CONFIG_NAME_OPTION, resolved_config_name]

    if resolved_config_path is not None:
        hydra_argv.extend([HYDRA_CONFIG_PATH_OPTION, resolved_config_path])

    hydra_argv.extend(adjusted_args)
    if not any(arg.startswith(HYDRA_SEARCHPATH_PREFIX) for arg in adjusted_args):
        hydra_argv.extend(extra_config_overrides)

    hydra_argv.extend(hydra_overrides)
    return hydra_argv


def resolve_hydra_entrypoint_argv(
    entrypoint_module: str,
    config_value: str,
    config_prefix: str,
    extra_args: Sequence[str] = (),
    hydra_overrides: Sequence[str] = (),
) -> list[str]:
    """Build ``sys.argv`` for a Hydra-backed runtime entrypoint."""
    return [
        entrypoint_module,
        *resolve_hydra_argv(
            config_value,
            config_prefix,
            extra_args=extra_args,
            hydra_overrides=hydra_overrides,
        ),
    ]


@contextmanager
def temporary_main_module(module_spec: ModuleSpec):
    """Temporarily expose a runtime module through ``__main__.__spec__``."""
    previous_spec = getattr(__main__, "__spec__", None)
    previous_package = getattr(__main__, "__package__", None)
    try:
        __main__.__spec__ = module_spec
        __main__.__package__ = module_spec.parent or None
        yield
    finally:
        __main__.__spec__ = previous_spec
        __main__.__package__ = previous_package


@contextmanager
def temporary_environment(updates: dict[str, str | None]):
    """Temporarily apply environment variables for one command invocation."""
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def prepare_runtime_environment(
    config_value: str,
    config_prefix: str,
    stage: str,
    extra_args: Sequence[str] = (),
    hydra_overrides: Sequence[str] = (),
    checkpoint: str | None = None,
) -> dict[str, str | None]:
    """Prepare environment variables used by Hydra-backed runtime commands."""
    adjusted_args = adjust_argv(extra_args)
    resolved_config_path, resolved_config_name, extra_config_overrides = resolve_config_reference(
        config_value, config_prefix
    )
    config_name = resolved_config_name.removeprefix(f"{config_prefix}/")
    compose_overrides = list(adjusted_args)
    if not any(arg.startswith(HYDRA_SEARCHPATH_PREFIX) for arg in compose_overrides):
        compose_overrides.extend(extra_config_overrides)
    compose_overrides.extend(hydra_overrides)

    started_at = datetime.now().astimezone()
    GlobalHydra.instance().clear()
    if resolved_config_path is None:
        with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
            cfg = compose(config_name=resolved_config_name, overrides=compose_overrides)
    else:
        with initialize_config_dir(version_base=None, config_dir=resolved_config_path):
            cfg = compose(config_name=resolved_config_name, overrides=compose_overrides)

    if should_enable_logger(cfg):
        checkpoint_path = Path(checkpoint) if checkpoint is not None else None
        experiment_name = generate_experiment_name(config_name)
        parent_run_id = None
        extra_tags = None
        if checkpoint_path is not None:
            experiment_name, parent_run_id = resolve_lineage_context(config_name, checkpoint_path)
            extra_tags = {
                "checkpoint_path": str(checkpoint_path),
                "source_run_id": parent_run_id or "",
            }
        run_context = prepare_run_context(
            cfg.logger.tracking_uri,
            config_name,
            hydra_dir=None,
            stage=stage,
            parent_run_id=parent_run_id,
            experiment_name=experiment_name,
            extra_tags=extra_tags,
            started_at=started_at,
        )
        return {
            AUTOWARE_ML_RUN_ID_ENV: run_context.run_id,
            AUTOWARE_ML_HYDRA_RUN_DIR_ENV: str(run_context.hydra_dir),
        }

    return {
        AUTOWARE_ML_RUN_ID_ENV: None,
        AUTOWARE_ML_HYDRA_RUN_DIR_ENV: str(
            generate_hydra_run_dir(config_name, started_at=started_at)
        ),
    }


def run_hydra_entrypoint(
    entrypoint_module: str,
    config_name: str,
    stage: str | None,
    extra_args: Sequence[str] = (),
    hydra_overrides: Sequence[str] = (),
    checkpoint: str | None = None,
    config_prefix: str = TASK_CONFIG_PREFIX,
) -> None:
    """Execute one Hydra-backed runtime entrypoint through the CLI wrapper."""
    env_updates: dict[str, str | None] = {}
    if stage is not None:
        env_updates = prepare_runtime_environment(
            config_name,
            config_prefix,
            stage,
            extra_args=extra_args,
            hydra_overrides=hydra_overrides,
            checkpoint=checkpoint,
        )

    sys.argv = resolve_hydra_entrypoint_argv(
        entrypoint_module,
        config_name,
        config_prefix,
        extra_args=extra_args,
        hydra_overrides=hydra_overrides,
    )

    with temporary_main_module(resolve_module_spec(entrypoint_module)), temporary_environment(
        env_updates
    ):
        run_lazy_script(entrypoint_module, "main")
