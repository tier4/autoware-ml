"""Tests for the deploy script: source lineage, the run records and the run status."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
import pytest

from autoware_ml.builders.tests.conftest import HydraRuntime
from autoware_ml.scripts import deploy as deploy_script
from autoware_ml.utils.mlflow_helpers import (
    CONFIG_ARTIFACTS_DIRNAME,
    RUN_METADATA_FILENAME,
    generate_run_name,
    prepare_run_context,
    resolve_deploy_lineage,
    resolve_experiment_name,
)

CONFIG_NAME = "detection3d/tiny/tiny_b1_1e_corpus"


def _write_checkpoint_metadata(
    tmp_path: Path,
    name: str,
    *,
    run_id: str,
    experiment_name: str,
    config_name: str,
) -> Path:
    run_dir = tmp_path / name
    checkpoint_path = run_dir / "artifacts" / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "experiment_name": experiment_name,
                "config_name": config_name,
                "stage": "train",
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path


def test_resolve_deploy_lineage_has_no_parent_for_a_multi_checkpoint_export(
    tmp_path: Path,
) -> None:
    detection3d_checkpoint = _write_checkpoint_metadata(
        tmp_path,
        "detection",
        run_id="det-run",
        experiment_name="ptv3/t4_v2/j6gen2/voxel012",
        config_name="detection3d/ptv3/voxel012",
    )
    segmentation3d_checkpoint = _write_checkpoint_metadata(
        tmp_path,
        "segmentation",
        run_id="seg-run",
        experiment_name="ptv3_seg3d/t4_v2/j6gen2/voxel012",
        config_name="segmentation3d/ptv3/voxel012",
    )

    parent_run_id, source_checkpoints = resolve_deploy_lineage(
        [detection3d_checkpoint, segmentation3d_checkpoint]
    )

    assert parent_run_id is None
    assert [source["run_id"] for source in source_checkpoints] == ["det-run", "seg-run"]
    assert [source["checkpoint_path"] for source in source_checkpoints] == [
        str(detection3d_checkpoint),
        str(segmentation3d_checkpoint),
    ]


def test_resolve_deploy_lineage_keeps_single_checkpoint_parent(tmp_path: Path) -> None:
    config_name = "detection3d/ptv3/voxel012"
    checkpoint = _write_checkpoint_metadata(
        tmp_path,
        "detection",
        run_id="det-run",
        experiment_name="ptv3/t4_v2/j6gen2/voxel012",
        config_name=config_name,
    )

    parent_run_id, source_checkpoints = resolve_deploy_lineage([checkpoint])

    assert parent_run_id == "det-run"
    assert source_checkpoints[0]["experiment_name"] == "ptv3/t4_v2/j6gen2/voxel012"


def _deploy_cfg(tmp_path: Path, logger_enabled: bool = True) -> DictConfig:
    """The nodes of a task config that ``run_deploy`` reads before building anything."""
    store_dir = tmp_path / "mlruns"
    store_dir.mkdir(exist_ok=True)
    return OmegaConf.create(
        {
            "experiment_group_name": "centerpoint/t4_v2",
            "experiment_name": "gen1_base",
            "experiment_uid": "2026.10.15.12.00.00",
            "experiment_run_dir": str(tmp_path / "work" / "gen1_base"),
            "seed": 0,
            "deploy": {"onnx": {"modules": {}}},
            "logger": (
                {
                    "_target_": "lightning.pytorch.loggers.MLFlowLogger",
                    "tracking_uri": f"sqlite:///{store_dir / 'mlflow.db'}",
                }
                if logger_enabled
                else None
            ),
        }
    )


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "source" / "artifacts" / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text("", encoding="utf-8")
    return checkpoint_path


def _run_status(cfg: DictConfig, run_id: str) -> str:
    return MlflowClient(tracking_uri=cfg.logger.tracking_uri).get_run(run_id).info.status


def _only_run_id(cfg: DictConfig) -> str:
    client = MlflowClient(tracking_uri=cfg.logger.tracking_uri)
    experiment = client.get_experiment_by_name(resolve_experiment_name(cfg))
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    return runs[0].info.run_id


def test_record_deploy_run_writes_config_overrides_metadata_and_params(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: a deploy run prepared in a SQLite store under the seeded Hydra runtime, with one
        source checkpoint.
    Expected: the run's artifacts hold ``config/resolved.yaml``, ``config/overrides.txt`` with
        the Hydra task overrides and ``run_metadata.json`` with the deploy stage and the source
        checkpoints, and the config is logged as run parameters.
    Check: the files, their content and the run parameters read back.
    """
    cfg = _deploy_cfg(tmp_path)
    experiment_name = resolve_experiment_name(cfg)
    run_context = prepare_run_context(
        cfg.logger.tracking_uri,
        config_name=CONFIG_NAME,
        hydra_dir=hydra_runtime.work_dir,
        stage="deploy",
        experiment_name=experiment_name,
        run_name=generate_run_name("deploy", experiment_name, cfg.experiment_uid),
    )
    checkpoint_path = _checkpoint(tmp_path)
    source_checkpoints = [{"checkpoint_path": str(checkpoint_path), "run_id": "train-run"}]

    deploy_script.record_deploy_run(
        cfg,
        run_context,
        CONFIG_NAME,
        parent_run_id="train-run",
        checkpoint_path=checkpoint_path,
        source_checkpoints=source_checkpoints,
    )

    config_dir = run_context.artifact_dir / CONFIG_ARTIFACTS_DIRNAME
    assert OmegaConf.load(config_dir / "resolved.yaml").experiment_name == "gen1_base"
    assert (config_dir / "overrides.txt").read_text(encoding="utf-8").splitlines() == list(
        hydra_runtime.overrides
    )
    metadata = json.loads(
        (run_context.artifact_dir / RUN_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    assert metadata["stage"] == "deploy"
    assert metadata["run_id"] == run_context.run_id
    assert metadata["config_name"] == CONFIG_NAME
    assert metadata["source_run_id"] == "train-run"
    assert metadata["checkpoint_path"] == str(checkpoint_path)
    assert metadata["source_checkpoints"] == source_checkpoints
    run = MlflowClient(tracking_uri=cfg.logger.tracking_uri).get_run(run_context.run_id)
    assert run.data.params["experiment_name"] == "gen1_base"


def test_run_deploy_refuses_configured_tensor_names_before_the_run_exists(
    tmp_path: Path,
) -> None:
    """
    Input: MLflow enabled and a deploy config whose module entry names the outputs.
    Expected: the deploy fails naming the entry before the checkpoints are resolved, the
        MLflow run is created or the model is built.
    Check: the raised error, that nothing after the config checks ran, and the absence of the
        SQLite store.
    """
    cfg = _deploy_cfg(tmp_path)
    cfg.deploy.onnx.modules.head = {"output_names": ["feat"]}

    with (
        patch.object(deploy_script, "build_weight_checkpoint_paths") as checkpoint_paths_mock,
        patch.object(deploy_script, "_export") as export_mock,
        pytest.raises(ValueError, match="deploy.onnx.modules.head.output_names"),
    ):
        deploy_script.run_deploy(cfg)

    checkpoint_paths_mock.assert_not_called()
    export_mock.assert_not_called()
    assert not (tmp_path / "mlruns" / "mlflow.db").exists()


def test_run_deploy_marks_the_run_failed_when_the_export_raises(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: MLflow enabled and an export that raises after the deploy run was created.
    Expected: the records are already written, the error propagates and the run is FAILED
        rather than left RUNNING.
    Check: the raised error, the run status and the metadata file.
    """
    cfg = _deploy_cfg(tmp_path)
    checkpoint_path = _checkpoint(tmp_path)

    with (
        patch.object(
            deploy_script,
            "build_weight_checkpoint_paths",
            return_value=([checkpoint_path], checkpoint_path),
        ),
        patch.object(deploy_script, "_export", side_effect=RuntimeError("onnx export broke")),
        pytest.raises(RuntimeError, match="onnx export broke"),
    ):
        deploy_script.run_deploy(cfg)

    run_id = _only_run_id(cfg)
    assert _run_status(cfg, run_id) == RunStatus.to_string(RunStatus.FAILED)
    artifact_dir = Path(
        MlflowClient(tracking_uri=cfg.logger.tracking_uri).get_run(run_id).data.tags["artifact_dir"]
    )
    assert (artifact_dir / RUN_METADATA_FILENAME).is_file()


def test_run_deploy_marks_the_run_finished_and_exports_into_the_run(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: MLflow enabled and an export that succeeds.
    Expected: the export receives the deploy run, whose ``exports`` directory lies in the run's
        artifacts, and the run ends FINISHED.
    Check: the export call and the run status.
    """
    cfg = _deploy_cfg(tmp_path)
    checkpoint_path = _checkpoint(tmp_path)

    with (
        patch.object(
            deploy_script,
            "build_weight_checkpoint_paths",
            return_value=([checkpoint_path], checkpoint_path),
        ),
        patch.object(deploy_script, "_export") as export_mock,
    ):
        deploy_script.run_deploy(cfg)

    run_context = export_mock.call_args.kwargs["run_context"]
    assert run_context is not None
    assert run_context.exports_dir == run_context.artifact_dir / "exports"
    assert _run_status(cfg, run_context.run_id) == RunStatus.to_string(RunStatus.FINISHED)


def test_run_deploy_without_logger_exports_into_the_experiment_run_dir(
    tmp_path: Path, hydra_runtime: HydraRuntime
) -> None:
    """
    Input: MLflow disabled, with the heavy build and export steps stubbed.
    Expected: no store is touched and the export writes to the config's ``experiment_run_dir``
        without a tracker or run id.
    Check: the ``deploy`` call and the absence of the SQLite file.
    """
    cfg = _deploy_cfg(tmp_path, logger_enabled=False)
    checkpoint_path = _checkpoint(tmp_path)

    with (
        patch.object(
            deploy_script,
            "build_weight_checkpoint_paths",
            return_value=([checkpoint_path], checkpoint_path),
        ),
        patch.object(deploy_script, "validate_cuda_available"),
        patch.object(deploy_script.torch.cuda, "get_device_name", return_value="stub"),
        patch.object(deploy_script, "build_datamodule"),
        patch.object(deploy_script, "build_data_preprocessor"),
        patch.object(deploy_script, "build_model"),
        patch.object(deploy_script, "deploy") as deploy_mock,
    ):
        deploy_script.run_deploy(cfg)

    kwargs = deploy_mock.call_args.kwargs
    assert kwargs["output_dir"] == Path(cfg.experiment_run_dir)
    assert kwargs["tracker"] is None
    assert kwargs["run_id"] is None
    assert not (tmp_path / "mlruns" / "mlflow.db").exists()


def test_module_entrypoint_runs_hydra_main() -> None:
    """
    Input: ``python -m autoware_ml.scripts.deploy --help``.
    Expected: the module guard calls the Hydra entrypoint, which prints its help instead of
        doing nothing.
    Check: the exit code and the Hydra help banner.
    """
    result = subprocess.run(
        [sys.executable, "-m", "autoware_ml.scripts.deploy", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Powered by Hydra" in result.stdout
