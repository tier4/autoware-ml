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

"""Unit tests for the MLflow run-context builder against a SQLite store under ``tmp_path``:
experiment and run naming, the artifact layout, the pre-created run handoff and the disabled
logger."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import pytest

from autoware_ml.builders.mlflow_builder import build_mlflow_run_context
from autoware_ml.builders.tests.conftest import HydraRuntime
from autoware_ml.utils.mlflow_helpers import (
    AUTOWARE_ML_RUN_ID_ENV,
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_RUN_NAME,
    MlflowRunContext,
    generate_run_name,
    prepare_run_context,
    resolve_experiment_name,
)

CONFIG_NAME = "detection3d/tiny/tiny_b1_1e_corpus"
EXPERIMENT_GROUP_NAME = "centerpoint/t4_v2"
EXPERIMENT_NAME = "base"
EXPERIMENT_UID = "2026.10.15.12.00.00"
FULL_EXPERIMENT_NAME = f"{EXPERIMENT_GROUP_NAME}/{EXPERIMENT_NAME}"


def _tracking_uri(tmp_path: Path) -> str:
    """A SQLite tracking URI under ``tmp_path``. The store directory holds the artifacts."""
    store_dir = tmp_path / "mlruns"
    store_dir.mkdir(exist_ok=True)
    return f"sqlite:///{store_dir / 'mlflow.db'}"


def _run_cfg(tmp_path: Path) -> DictConfig:
    """The experiment bookkeeping and logger nodes of ``default_task.yaml``."""
    return OmegaConf.create(
        {
            "experiment_group_name": EXPERIMENT_GROUP_NAME,
            "experiment_name": EXPERIMENT_NAME,
            "experiment_uid": EXPERIMENT_UID,
            "logger": {
                "_target_": "lightning.pytorch.loggers.MLFlowLogger",
                "tracking_uri": _tracking_uri(tmp_path),
            },
        }
    )


def _build(cfg: DictConfig, stage: str, logger_enabled: bool = True) -> MlflowRunContext | None:
    return build_mlflow_run_context(
        cfg,
        stage=stage,
        config_name=CONFIG_NAME,
        logger_enabled=logger_enabled,
    )


def test_experiment_and_run_names_come_from_the_config_keys(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: a config with ``experiment_group_name``, ``experiment_name`` and ``experiment_uid``
        and a train run.
    Expected: the experiment is ``<experiment_group_name>/<experiment_name>`` and the run is
        ``<stage>_<experiment_name>_<uid>``, both registered in the store.
    Check: the context fields and the run the client reads back.
    """
    cfg = _run_cfg(tmp_path)

    context = _build(cfg, stage="train")

    assert context is not None
    assert context.experiment_name == FULL_EXPERIMENT_NAME
    assert context.run_name == f"train_base_{EXPERIMENT_UID}"
    client = MlflowClient(tracking_uri=context.tracking_uri)
    run = client.get_run(context.run_id)
    assert run.info.run_name == context.run_name
    assert run.data.tags[MLFLOW_RUN_NAME] == context.run_name
    assert client.get_experiment(run.info.experiment_id).name == context.experiment_name


def test_builder_and_cli_runs_land_in_the_same_experiment(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: a run prepared the way the CLI prepares it, through the shared naming functions,
        and a run built by the entrypoint builder from the same config.
    Expected: both runs sit in one experiment under one artifact root.
    Check: the experiment ids and the artifact roots of the two runs.
    """
    cfg = _run_cfg(tmp_path)
    experiment_name = resolve_experiment_name(cfg)
    cli_run = prepare_run_context(
        cfg.logger.tracking_uri,
        config_name=CONFIG_NAME,
        hydra_dir=None,
        stage="train",
        experiment_name=experiment_name,
        run_name=generate_run_name("train", experiment_name, cfg.experiment_uid),
    )

    context = _build(cfg, stage="deploy")

    assert context is not None
    assert context.experiment_id == cli_run.experiment_id
    assert context.artifact_dir.parent.parent == cli_run.artifact_dir.parent.parent
    assert context.run_name == f"deploy_base_{EXPERIMENT_UID}"


def test_builder_requires_the_experiment_keys(tmp_path: Path, hydra_runtime: HydraRuntime) -> None:
    """
    Input: a config without ``experiment_group_name``.
    Expected: the experiment cannot be named, so the builder fails before touching the store.
    Check: a ValueError naming the key is raised and no store is created.
    """
    cfg = _run_cfg(tmp_path)
    del cfg["experiment_group_name"]

    with pytest.raises(ValueError, match="experiment_group_name"):
        _build(cfg, stage="train")
    assert not (tmp_path / "mlruns" / "mlflow.db").exists()


def test_run_context_binds_the_hydra_work_dir_and_the_artifact_layout(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: a test run with the seeded Hydra runtime.
    Expected: the hydra directory is the Hydra output directory, the artifact directory lives
        under ``<store>/<experiment_group_name>/<experiment_name>/<run_id>/artifacts`` and the
        tags record the stage, the config name and both directories.
    Check: the context paths, their existence and the tags.
    """
    cfg = _run_cfg(tmp_path)

    context = _build(cfg, stage="test")

    assert context is not None
    assert context.hydra_dir == hydra_runtime.work_dir
    assert context.artifact_dir == (
        tmp_path / "mlruns" / FULL_EXPERIMENT_NAME / context.run_id / "artifacts"
    )
    assert context.artifact_dir.is_dir()
    assert context.checkpoints_dir == context.artifact_dir / "checkpoints"
    assert context.exports_dir == context.artifact_dir / "exports"
    assert context.tags["stage"] == "test"
    assert context.tags["config_name"] == CONFIG_NAME
    assert context.tags["hydra_dir"] == str(hydra_runtime.work_dir)
    assert context.tags["artifact_dir"] == str(context.artifact_dir)


def test_parent_run_and_extra_tags_are_recorded(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: a deploy run nested under a parent run id with the extra tags the deploy entrypoint
        passes.
    Expected: the run carries the parent tag and the extra tags as strings.
    Check: the tags of the run read back from the store.
    """
    cfg = _run_cfg(tmp_path)
    parent = _build(cfg, stage="train")
    assert parent is not None

    context = build_mlflow_run_context(
        cfg,
        stage="deploy",
        config_name=CONFIG_NAME,
        logger_enabled=True,
        parent_run_id=parent.run_id,
        extra_tags=MappingProxyType({"checkpoint_path": "/ckpt/best.ckpt", "count": 2}),
    )

    assert context is not None
    run = MlflowClient(tracking_uri=context.tracking_uri).get_run(context.run_id)
    assert run.data.tags[MLFLOW_PARENT_RUN_ID] == parent.run_id
    assert run.data.tags["checkpoint_path"] == "/ckpt/best.ckpt"
    assert run.data.tags["count"] == "2"
    assert context.run_name == f"deploy_base_{EXPERIMENT_UID}"


def test_disabled_logger_returns_no_context_and_touches_no_store(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: ``logger_enabled=False``.
    Expected: no run context and no tracking store is created.
    Check: the return value and the absence of the SQLite file.
    """
    cfg = _run_cfg(tmp_path)

    context = _build(cfg, stage="train", logger_enabled=False)

    assert context is None
    assert not (tmp_path / "mlruns" / "mlflow.db").exists()


def test_pre_created_run_is_loaded_instead_of_creating_a_new_one(
    tmp_path: Path, hydra_runtime: HydraRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Input: ``AUTOWARE_ML_RUN_ID`` naming a run prepared on the seeded Hydra directory, as the
        CLI does when it resumes a run.
    Expected: that run is loaded and no second run is created.
    Check: the run id, the run name and the run count of the experiment.
    """
    cfg = _run_cfg(tmp_path)
    prepared = prepare_run_context(
        cfg.logger.tracking_uri,
        config_name=CONFIG_NAME,
        hydra_dir=hydra_runtime.work_dir,
        stage="train",
        experiment_name=FULL_EXPERIMENT_NAME,
        run_name="train_base_resumed",
    )
    monkeypatch.setenv(AUTOWARE_ML_RUN_ID_ENV, prepared.run_id)

    context = _build(cfg, stage="train")

    assert context is not None
    assert context.run_id == prepared.run_id
    assert context.run_name == "train_base_resumed"
    assert context.hydra_dir == hydra_runtime.work_dir
    client = MlflowClient(tracking_uri=context.tracking_uri)
    assert len(client.search_runs([context.experiment_id])) == 1


def test_pre_created_run_on_another_hydra_dir_is_rejected(
    tmp_path: Path, hydra_runtime: HydraRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Input: ``AUTOWARE_ML_RUN_ID`` naming a run prepared on a different Hydra directory.
    Expected: the builder refuses to write the run from a directory it was not prepared for.
    Check: a RuntimeError mentioning the mismatch is raised.
    """
    cfg = _run_cfg(tmp_path)
    other_dir = tmp_path / "other_hydra_run"
    other_dir.mkdir()
    prepared = prepare_run_context(
        cfg.logger.tracking_uri,
        config_name=CONFIG_NAME,
        hydra_dir=other_dir,
        stage="train",
        experiment_name=FULL_EXPERIMENT_NAME,
        run_name="train_base_other",
    )
    monkeypatch.setenv(AUTOWARE_ML_RUN_ID_ENV, prepared.run_id)

    with pytest.raises(RuntimeError, match="does not match"):
        _build(cfg, stage="train")


def test_enabled_logger_requires_the_logger_node(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: ``logger_enabled=True`` with a config that has no ``logger`` node.
    Expected: the builder fails on the missing tracking URI.
    Check: an AttributeError naming the key is raised.
    """
    cfg = _run_cfg(tmp_path)
    del cfg["logger"]

    with pytest.raises(AttributeError, match="logger"):
        _build(cfg, stage="train")


def test_builder_requires_the_hydra_runtime(tmp_path: Path) -> None:
    """
    Input: no seeded Hydra runtime.
    Expected: the work directory cannot be resolved, so the builder fails before any store
        access.
    Check: a ValueError is raised and no store is created.
    """
    cfg = _run_cfg(tmp_path)

    with pytest.raises(ValueError):
        _build(cfg, stage="train")
    assert not (tmp_path / "mlruns" / "mlflow.db").exists()
