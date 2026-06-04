"""Tests for deploy script helpers."""

from __future__ import annotations

import json
from pathlib import Path

from autoware_ml.utils.mlflow_helpers import generate_experiment_name, resolve_deploy_lineage


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


def test_resolve_deploy_lineage_uses_final_config_experiment_for_multi_checkpoint_export(
    tmp_path: Path,
) -> None:
    config_name = "segmentation3d_detection3d/ptv3_segdet_transhead/voxel012"
    detection_checkpoint = _write_checkpoint_metadata(
        tmp_path,
        "detection",
        run_id="det-run",
        experiment_name="detection3d_ptv3_transhead_voxel012",
        config_name="detection3d/ptv3/transhead_voxel012",
    )
    segmentation_checkpoint = _write_checkpoint_metadata(
        tmp_path,
        "segmentation",
        run_id="seg-run",
        experiment_name="segmentation3d_ptv3_voxel012",
        config_name="segmentation3d/ptv3/voxel012",
    )

    experiment_name, parent_run_id, source_checkpoints = resolve_deploy_lineage(
        config_name,
        [detection_checkpoint, segmentation_checkpoint],
    )

    assert experiment_name == generate_experiment_name(config_name)
    assert parent_run_id is None
    assert [source["run_id"] for source in source_checkpoints] == ["det-run", "seg-run"]
    assert [source["checkpoint_path"] for source in source_checkpoints] == [
        str(detection_checkpoint),
        str(segmentation_checkpoint),
    ]


def test_resolve_deploy_lineage_keeps_single_checkpoint_parent(tmp_path: Path) -> None:
    config_name = "detection3d/ptv3/transhead_voxel012"
    checkpoint = _write_checkpoint_metadata(
        tmp_path,
        "detection",
        run_id="det-run",
        experiment_name="detection3d_ptv3_transhead_voxel012",
        config_name=config_name,
    )

    experiment_name, parent_run_id, source_checkpoints = resolve_deploy_lineage(
        config_name,
        [checkpoint],
    )

    assert experiment_name == generate_experiment_name(config_name)
    assert parent_run_id == "det-run"
    assert source_checkpoints[0]["experiment_name"] == "detection3d_ptv3_transhead_voxel012"
