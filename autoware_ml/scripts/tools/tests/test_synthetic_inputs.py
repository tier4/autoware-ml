"""Tests for the voxel layout of the synthetic inputs."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    build_serialized_pooling_metadata,
)
from autoware_ml.scripts.tools.synthetic_inputs import (
    REAL_NEIGHBORS,
    mean_neighbors,
    voxel_layout,
    write_export_inputs,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.structures import serialize_point_cloud_batch


def test_voxel_layout_pools_to_the_requested_counts_and_neighbors() -> None:
    counts = [8000, 4000, 1600, 640, 256]
    strides = [2, 2, 2, 2]
    grid_shape = torch.tensor([512, 512, 64])
    generator = torch.Generator().manual_seed(0)
    grid = voxel_layout(counts, strides, grid_shape, REAL_NEIGHBORS, generator)

    point, _ = serialize_point_cloud_batch(
        {
            "coord": grid.float(),
            "feat": grid.float(),
            "grid_coord": grid,
            "offset": torch.tensor([len(grid)]),
        },
        PTv3BaseModel.EXPORT_ORDER,
        torch.tensor(int(grid_shape.max()).bit_length()),
    )
    metadata = build_serialized_pooling_metadata(
        point["grid_coord"],
        point["serialized_code"],
        point["serialized_order"],
        strides,
        [512] * len(counts),
    )
    stages = [grid, *(meta.grid_coord for meta in metadata)]
    assert [len(stage) for stage in stages] == counts
    assert [mean_neighbors(stage) for stage in stages] == pytest.approx(REAL_NEIGHBORS, abs=0.1)


@pytest.mark.parametrize(("counts", "size"), [([100, 10], 64), ([40, 10], 4)])
def test_voxel_layout_rejects_counts_that_cannot_pool(counts: list[int], size: int) -> None:
    grid_shape = torch.tensor([size] * 3)
    with pytest.raises(ValueError, match="cannot pool"):
        voxel_layout(counts, [2], grid_shape, REAL_NEIGHBORS[:2], torch.Generator())


def test_voxel_layout_fills_a_grid_whose_size_is_not_a_multiple_of_the_stride() -> None:
    grid_shape = torch.tensor([8, 8, 5])
    grid = voxel_layout([320, 48], [2], grid_shape, REAL_NEIGHBORS[:2], torch.Generator())

    assert len(torch.unique(grid, dim=0)) == 320
    assert grid.max(dim=0).values.tolist() == [7, 7, 4]


def test_write_export_inputs_writes_raw_tensors_and_trtexec_shapes(tmp_path: Path) -> None:
    feat = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    index = torch.tensor([1, 2])
    specs = {
        "encoder": ExportSpec(
            module=torch.nn.Identity(),
            args=(feat, index),
            input_names=["feat", "index"],
            output_names=["out"],
        )
    }

    write_export_inputs(specs, tmp_path)

    assert (tmp_path / "encoder" / "shapes.txt").read_text() == "feat:2x3,index:2\n"
    assert (
        np.fromfile(tmp_path / "encoder" / "feat.bin", dtype=np.float32).tolist()
        == feat.flatten().tolist()
    )
    assert np.fromfile(tmp_path / "encoder" / "index.bin", dtype=np.int64).tolist() == [1, 2]


def test_module_entrypoint_runs_hydra_main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "autoware_ml.scripts.tools.synthetic_inputs", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Powered by Hydra" in result.stdout
