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

"""Synthetic-inputs entrypoint for Autoware-ML PTv3 models.

This script writes the export inputs of every module at set voxel counts per encoder stage, for
latency benchmarks with trtexec. The voxels are as clustered as in the densest real frames
(``REAL_NEIGHBORS``), since the sparse convolutions cost more the more neighbors a voxel has.
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Mapping, Sequence

import hydra
from omegaconf import DictConfig
import torch

from autoware_ml.builders.model_builder import build_data_preprocessor, build_model
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.models.segmentation3d.encoders.voxel import (
    TIME_LAG_COLUMN,
    MeanVoxelFeatureEncoder,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    prepare_ptv3_export_inputs,
)
from autoware_ml.utils.deploy import ExportSpec, validate_cuda_available
from autoware_ml.utils.runtime import configure_torch_runtime, get_config_path, set_seed

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()

# Occupied cells among the 26 around a voxel, per encoder stage: the p99 over 2184 real frames
# (110 scenes of db_j6gen2_v1 to v14) voxelized at 0.12 m and pooled by 2, as the voxel012 configs.
# TODO measure the voxel006 configs; they use these targets too.
REAL_NEIGHBORS = (8.24, 9.15, 9.44, 9.93, 10.60)
_AROUND = torch.tensor([d for d in itertools.product((-1, 0, 1), repeat=3) if any(d)])


def mean_neighbors(coords: torch.Tensor) -> float:
    """Return how many of the 26 cells around a voxel are occupied, on average."""
    span = int(coords.max()) + 3
    weights = torch.tensor([span * span, span, 1], device=coords.device)
    keys = ((coords.long() + 1) @ weights).sort().values
    around = (coords.long()[:, None] + 1 + _AROUND.to(coords.device)) @ weights
    found = keys[torch.searchsorted(keys, around).clamp(max=len(keys) - 1)] == around
    return float(found.sum()) / len(coords)


def _keep(
    cells: torch.Tensor, parents: torch.Tensor | None, count: int, score: torch.Tensor
) -> torch.Tensor:
    """Return the ``count`` best-scoring cells, the best child of every parent among them."""
    if parents is None:
        return cells[score.argsort()[:count]]
    score = score.reshape(len(parents), -1)
    keep = torch.zeros_like(score, dtype=torch.bool)
    keep[torch.arange(len(parents)), score.argmin(dim=1)] = True
    rest = torch.where(keep, math.inf, score).reshape(-1)
    keep.view(-1)[rest.argsort()[: count - len(parents)]] = True
    return cells[keep.view(-1)]


def voxel_layout(
    counts: Sequence[int],
    strides: Sequence[int],
    grid_shape: torch.Tensor,
    neighbors: Sequence[float],
    generator: torch.Generator,
) -> torch.Tensor:
    """Pick voxels whose pooling stages hold exactly ``counts`` voxels with ``neighbors`` each.

    Stage by stage, coarsest first, keeps the cells nearest the center of the grid floor, ranked
    with noise whose weight is bisected until the stage's mean neighbors reach the target. Every
    stage's grid is the finest one divided by its scale, rounded up as the model pools it.

    Args:
        counts: Voxels per stage, finest first.
        strides: Pooling stride of each stage.
        grid_shape: Finest grid size per axis.
        neighbors: Target mean of occupied cells around a voxel per stage, finest first.
        generator: Source of the ranking noise.

    Returns:
        ``[counts[0], 3]`` grid coordinates.

    Raises:
        ValueError: If a stage cannot pool to the next.
    """
    if not len(counts) == len(strides) + 1 == len(neighbors):
        raise ValueError(
            f"{len(counts)} voxel counts and {len(neighbors)} neighbor targets for "
            f"{len(strides)} pooling stages"
        )
    scale = math.prod(strides)
    cells = torch.cartesian_prod(*(torch.arange(int(n)) for n in (grid_shape + scale - 1) // scale))
    if counts[-1] > len(cells):
        raise ValueError(f"{counts[-1]} voxels cannot pool to a grid of {len(cells)} cells")
    parents = None
    for stage in reversed(range(len(counts))):
        if parents is not None:
            stride = strides[stage]
            scale //= stride
            offsets = torch.tensor(list(itertools.product(range(stride), repeat=3)))
            cells = (parents[:, None] * stride + offsets[None]).reshape(-1, 3)
        inside = (cells < (grid_shape + scale - 1) // scale).all(dim=1)
        if parents is not None and not len(parents) <= counts[stage] <= int(inside.sum()):
            raise ValueError(
                f"{counts[stage]} voxels cannot pool to {len(parents)} with stride {strides[stage]}"
            )
        center = torch.tensor([*(grid_shape[:2] / 2), 0.0]) / scale
        distance = (cells + 0.5 - center).norm(dim=1)
        noise = torch.rand(len(cells), generator=generator) * distance.max()
        distance[~inside] = math.inf
        low, high = 0.0, 64.0
        for _ in range(16):
            mix = (low + high) / 2
            layout = _keep(cells, parents, counts[stage], distance + mix * noise)
            low, high = (mix, high) if mean_neighbors(layout) > neighbors[stage] else (low, mix)
        parents = _keep(cells, parents, counts[stage], distance + low * noise)
    return parents


def write_export_inputs(specs: Mapping[str, ExportSpec], out: Path) -> None:
    """Write every input of every spec as a raw ``.bin`` under ``out/<module>``.

    ``shapes.txt`` next to them lists ``name:AxB`` per input, the ``--shapes`` argument of
    trtexec.
    """
    for module, spec in specs.items():
        module_dir = out / module
        module_dir.mkdir(parents=True, exist_ok=True)
        for name, tensor in zip(spec.input_names, spec.args):
            tensor.detach().cpu().contiguous().numpy().tofile(module_dir / f"{name}.bin")
        shapes = ",".join(
            f"{name}:{'x'.join(map(str, tensor.shape))}"
            for name, tensor in zip(spec.input_names, spec.args)
        )
        (module_dir / "shapes.txt").write_text(shapes + "\n")
        logger.info("%s: %d inputs in %s", module, len(spec.input_names), module_dir)


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Write the export inputs at ``cfg.voxels`` voxels per stage to ``cfg.out``.

    ``cfg.neighbors`` overrides the density targets ``REAL_NEIGHBORS``.

    Args:
        cfg: Hydra configuration.
    """
    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)
    device = torch.device("cuda")
    model = build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        device=device,
        weights_path=None,
        resume_checkpoint_path=None,
        set_eval=True,
    )
    if not isinstance(model, PTv3BaseModel):
        raise NotImplementedError(
            f"synthetic_inputs supports only PTv3 models, got {type(model).__name__}."
        )

    voxels = list(cfg.voxels)
    neighbors = list(cfg.get("neighbors", REAL_NEIGHBORS))
    grid_shape, _ = model.compute_export_geometry(torch.device("cpu"))
    generator = torch.Generator().manual_seed(cfg.seed)
    coords = voxel_layout(voxels, model.encoder.stride, grid_shape, neighbors, generator)
    point_cloud_range = torch.tensor(model.point_cloud_range, dtype=torch.float32)
    xyz = point_cloud_range[:3] + (coords + 0.5) * torch.tensor(model.grid_size)
    intensity = torch.rand(len(coords), 1, generator=generator)
    if isinstance(model.voxel_encoder, MeanVoxelFeatureEncoder):
        points, time_lag_column = torch.cat([xyz, intensity], dim=1), -1
    else:
        time_lag = torch.zeros_like(intensity)
        points, time_lag_column = torch.cat([xyz, intensity, time_lag], dim=1), TIME_LAG_COLUMN
    batch = ModelGTBatch(
        point_cloud_gt_batch=PointCloudGTBatch(
            points=points,
            batch_indices=torch.zeros(len(coords), dtype=torch.int32),
            batch_size=1,
            timestamp_difference_dim=time_lag_column,
        ),
        detection3d_gt_batch=None,
        segmentation3d_gt_batch=None,
        image_gt_batch=None,
        padding_mask=torch.zeros(1, dtype=torch.bool),
    )
    batch_inputs = model.on_after_batch_transfer(batch.to_device(device), dataloader_idx=0)
    dropped = int(batch_inputs.voxels_data.num_dropped_voxels)
    if dropped:
        raise RuntimeError(f"The voxelizer dropped {dropped} voxels; raise its 'max_voxels'.")

    inputs = prepare_ptv3_export_inputs(model, batch_inputs)
    stages = [inputs.grid_coord.cpu(), *(meta.grid_coord.cpu() for meta in inputs.pooling_metadata)]
    counts = [len(stage) for stage in stages]
    if counts != voxels:
        raise RuntimeError(f"The stages hold {counts} voxels, not {voxels}.")
    for k, (stage, real) in enumerate(zip(stages, neighbors)):
        logger.info(
            "stage %d: %d voxels, %.2f neighbors (real p99 %.2f)",
            k,
            len(stage),
            mean_neighbors(stage),
            real,
        )
    write_export_inputs(
        model.build_export_specs(batch_inputs), Path(hydra.utils.to_absolute_path(cfg.out))
    )


if __name__ == "__main__":
    main()
