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

"""Unit tests for the trainer logger builder: the run artifacts it persists, the logger node it
populates and the ``MLFlowLogger`` it instantiates on an existing run (no run is started)."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
import pytest

from autoware_ml.builders.logger_builder import build_trainer_logger
from autoware_ml.builders.tests.conftest import HydraRuntime
from autoware_ml.utils.mlflow_helpers import MlflowRunContext

CONFIG_NAME = "detection3d/tiny/tiny_b1_1e_corpus"
EXPERIMENT_NAME = "centerpoint_t4_v2_gen1_base"
RUN_NAME = "train_centerpoint_t4_v2_gen1_base_2026.10.15.12.00.00"
RUN_ID = "0123456789abcdef0123456789abcdef"


def _run_context(tmp_path: Path, hydra_runtime: HydraRuntime) -> MlflowRunContext:
    """A resolved run context over directories under ``tmp_path``, without an MLflow run."""
    artifact_dir = tmp_path / "mlruns" / CONFIG_NAME / RUN_ID / "artifacts"
    artifact_dir.mkdir(parents=True)
    return MlflowRunContext(
        tracking_uri=f"sqlite:///{tmp_path / 'mlruns' / 'mlflow.db'}",
        experiment_id="7",
        experiment_name=EXPERIMENT_NAME,
        run_id=RUN_ID,
        run_name=RUN_NAME,
        tags={"stage": "train", "config_name": CONFIG_NAME, "owner": "context"},
        artifact_uri=artifact_dir.as_uri(),
        artifact_dir=artifact_dir,
        hydra_dir=hydra_runtime.work_dir,
    )


def _cfg(tracking_uri: str) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {"_target_": "torch.nn.Identity"},
            "trainer": {"max_epochs": 1},
            "logger": {
                "_target_": "lightning.pytorch.loggers.MLFlowLogger",
                "tracking_uri": tracking_uri,
                "tags": {"owner": "alice"},
            },
        }
    )


def test_disabled_logger_returns_none_and_writes_nothing(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: ``logger_enabled=False``.
    Expected: no logger and no file under the artifact directory.
    Check: the return value, the untouched directory and the untouched logger node.
    """
    context = _run_context(tmp_path, hydra_runtime)
    cfg = _cfg(context.tracking_uri)

    trainer_logger = build_trainer_logger(
        cfg,
        ml_flow_run_context=context,
        stage="train",
        config_name=CONFIG_NAME,
        logger_enabled=False,
    )

    assert trainer_logger is None
    assert list(context.artifact_dir.iterdir()) == []
    assert "run_id" not in cfg.logger


def test_enabled_logger_is_an_mlflow_logger_bound_to_the_run(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: ``logger_enabled=True`` on the resolved run context.
    Expected: a ``MLFlowLogger`` attached to the existing run, experiment and run name, with the
        context tags merged under the tags of the logger node.
    Check: the logger type and its constructor state. The run itself is never started.
    """
    context = _run_context(tmp_path, hydra_runtime)
    cfg = _cfg(context.tracking_uri)

    trainer_logger = build_trainer_logger(
        cfg,
        ml_flow_run_context=context,
        stage="train",
        config_name=CONFIG_NAME,
        logger_enabled=True,
    )

    assert isinstance(trainer_logger, MLFlowLogger)
    assert trainer_logger._run_id == RUN_ID
    assert trainer_logger._experiment_name == EXPERIMENT_NAME
    assert trainer_logger._run_name == RUN_NAME
    assert trainer_logger._tracking_uri == context.tracking_uri
    assert dict(trainer_logger.tags) == {
        "stage": "train",
        "config_name": CONFIG_NAME,
        "owner": "alice",
    }
    assert not trainer_logger._initialized


def test_logger_node_is_populated_for_the_trainer(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: the logger node of the config before the build.
    Expected: the builder writes the run id, experiment name, run name and merged tags into the
        node so a re-instantiation reaches the same run.
    Check: the mutated config node.
    """
    context = _run_context(tmp_path, hydra_runtime)
    cfg = _cfg(context.tracking_uri)

    build_trainer_logger(
        cfg,
        ml_flow_run_context=context,
        stage="train",
        config_name=CONFIG_NAME,
        logger_enabled=True,
    )

    assert cfg.logger.run_id == RUN_ID
    assert cfg.logger.experiment_name == EXPERIMENT_NAME
    assert cfg.logger.run_name == RUN_NAME
    assert cfg.logger.tracking_uri == context.tracking_uri
    assert cfg.logger.tags.owner == "alice"
    assert cfg.logger.tags.stage == "train"


def test_run_config_and_metadata_are_persisted_under_the_artifact_dir(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: an enabled logger with the extra metadata the test entrypoint passes.
    Expected: ``config/resolved.yaml`` holds the resolved config, ``config/overrides.txt`` the
        Hydra task overrides one per line, and ``run_metadata.json`` the run identity, the
        stage and the extra metadata.
    Check: the three files.
    """
    context = _run_context(tmp_path, hydra_runtime)
    cfg = _cfg(context.tracking_uri)

    build_trainer_logger(
        cfg,
        ml_flow_run_context=context,
        stage="test",
        config_name=CONFIG_NAME,
        logger_enabled=True,
        extra_metadata=MappingProxyType(
            {"source_run_id": "parent-run", "checkpoint_path": "/ckpt/best.ckpt"}
        ),
    )

    config_dir = context.config_dir
    resolved = OmegaConf.load(config_dir / "resolved.yaml")
    assert resolved.model == cfg.model
    assert resolved.trainer.max_epochs == 1
    assert resolved.logger.tracking_uri == context.tracking_uri
    overrides = (config_dir / "overrides.txt").read_text(encoding="utf-8")
    assert overrides == "\n".join(hydra_runtime.overrides) + "\n"
    metadata = json.loads((context.artifact_dir / "run_metadata.json").read_text("utf-8"))
    assert metadata == {
        "run_id": RUN_ID,
        "experiment_id": "7",
        "experiment_name": EXPERIMENT_NAME,
        "config_name": CONFIG_NAME,
        "hydra_dir": str(hydra_runtime.work_dir),
        "artifact_dir": str(context.artifact_dir),
        "stage": "test",
        "source_run_id": "parent-run",
        "checkpoint_path": "/ckpt/best.ckpt",
    }


def test_enabled_logger_requires_the_logger_node(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: ``logger_enabled=True`` with a config that has no ``logger`` node.
    Expected: the builder fails on the missing node after persisting the run config.
    Check: an AttributeError naming the key is raised.
    """
    context = _run_context(tmp_path, hydra_runtime)
    cfg = _cfg(context.tracking_uri)
    del cfg["logger"]

    with pytest.raises(AttributeError, match="logger"):
        build_trainer_logger(
            cfg,
            ml_flow_run_context=context,
            stage="train",
            config_name=CONFIG_NAME,
            logger_enabled=True,
        )
