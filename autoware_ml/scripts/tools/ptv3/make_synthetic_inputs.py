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

This script writes the export inputs of every module at set voxel counts per encoder stage.
The voxels are as clustered as ``target_neighbors`` asks per stage.
"""

from __future__ import annotations

import logging
from pathlib import Path

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
from autoware_ml.utils.deploy import validate_cuda_available
from autoware_ml.utils.runtime import configure_torch_runtime, get_config_path, set_seed
from autoware_ml.utils.synthetic_inputs import (
    VoxelLayoutSpec,
    mean_neighbors,
    voxel_layout,
    write_export_inputs,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Write the export inputs at ``cfg.voxel_counts`` voxels per stage to ``cfg.out``.

    ``cfg.target_neighbors`` sets the mean occupied neighbors per stage, finest first.

    Args:
        cfg: Hydra configuration.

    Raises:
        NotImplementedError: If the configured model is not a PTv3 model.
        ValueError: If the model inputs carry no voxels, or a stage of ``cfg.voxel_counts``
            cannot pool to the next.
        RuntimeError: If the stages do not hold ``cfg.voxel_counts`` voxels.
    """
    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)
    model = build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        device=torch.device("cuda"),
        weights_path=None,
        resume_checkpoint_path=None,
        set_eval=True,
    )
    if not isinstance(model, PTv3BaseModel):
        raise NotImplementedError(
            f"{Path(__file__).stem} supports only PTv3 models, got {type(model).__name__}."
        )

    grid_shape, _ = model.compute_export_geometry(torch.device("cpu"))
    point_cloud_range = torch.tensor(model.point_cloud_range, dtype=torch.float32)
    spec = VoxelLayoutSpec(
        voxel_counts=list(cfg.voxel_counts),
        pooling_strides=model.encoder.stride,
        grid_shape=grid_shape,
        target_neighbors=list(cfg.target_neighbors),
        center=-point_cloud_range[:3] / model.grid_size,
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    coords = voxel_layout(spec, generator)
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
    batch_inputs = model.on_after_batch_transfer(batch.to_device(model.device), dataloader_idx=0)

    inputs = prepare_ptv3_export_inputs(model, batch_inputs)
    stages = [inputs.grid_coord, *(meta.grid_coord for meta in inputs.pooling_metadata)]
    counts = [len(stage) for stage in stages]
    if counts != list(spec.voxel_counts):
        raise RuntimeError(f"The stages hold {counts} voxels, not {spec.voxel_counts}.")
    for index, (stage, target) in enumerate(zip(stages, spec.target_neighbors)):
        logger.info(
            "stage %d: %d voxels, %.2f neighbors (target %.2f)",
            index,
            len(stage),
            mean_neighbors(stage),
            target,
        )
    write_export_inputs(
        model.build_export_specs(batch_inputs), Path(hydra.utils.to_absolute_path(cfg.out))
    )


if __name__ == "__main__":
    main()
