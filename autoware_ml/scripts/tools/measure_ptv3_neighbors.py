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

"""Neighbor-targets entrypoint for the synthetic PTv3 inputs.

This script measures the ``neighbors`` targets of ``make_ptv3_synthetic_inputs`` on the train
split of a config: the p99 over its frames of every stage's mean occupied neighbors, logged as
the ``+neighbors=[...]`` override to pass on.
"""

from __future__ import annotations

from collections.abc import Callable
import functools
import logging
from typing import Sequence

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.datamodule.base_dataset import ConcatDataset
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.models.segmentation3d.ptv3_base import (
    PTv3BaseModel,
    prepare_ptv3_export_inputs,
)
from autoware_ml.types.dataset import SplitType
from autoware_ml.utils.deploy import validate_cuda_available
from autoware_ml.utils.runtime import configure_torch_runtime, get_config_path, set_seed
from autoware_ml.utils.synthetic_inputs import (
    build_ptv3_model,
    check_voxels,
    list_stage_grids,
    measure_stage_neighbors,
)

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


def measure_frame_neighbors(model: PTv3BaseModel, batch: ModelGTBatch, frame: int) -> list[float]:
    """Return the mean neighbors of every stage of one frame, voxelized and pooled as exported."""
    batch_inputs = model.on_after_batch_transfer(batch.to_device(model.device), dataloader_idx=0)
    check_voxels(batch_inputs.voxels_data, f"train frame {frame}")
    inputs = prepare_ptv3_export_inputs(model, batch_inputs)
    return measure_stage_neighbors(list_stage_grids(inputs.grid_coord, inputs.pooling_metadata))


def measure_neighbors(
    datamodule: DataModule,
    neighbors_of: Callable[[ModelGTBatch, int], Sequence[float]],
    num_workers: int,
) -> list[float]:
    """Return the p99 over the train split of every stage's mean neighbors, finest first.

    Every source of the split is read once, whatever its training repeat, and ``neighbors_of``
    measures one frame, given with its index.
    """
    datamodule.prepare_data()
    datamodule.setup_splits((SplitType.TRAIN,))
    split = datamodule.datasets[SplitType.TRAIN]
    dataset = ConcatDataset(split.datasets, [1] * len(split.datasets))
    loader = DataLoader(
        dataset, batch_size=1, num_workers=num_workers, collate_fn=dataset.collate_fn
    )
    with torch.no_grad():
        neighbors = [list(neighbors_of(batch, frame)) for frame, batch in enumerate(loader)]
    return torch.tensor(neighbors, dtype=torch.float32).quantile(0.99, dim=0).tolist()


@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig) -> None:
    """Log the neighbor targets of the config, the p99 over its train split per stage.

    The frames go through the test transforms, without the training augmentations, as the
    deployed model sees them.

    Args:
        cfg: Hydra configuration.
    """
    validate_cuda_available()
    configure_torch_runtime()
    set_seed(cfg)
    model = build_ptv3_model(cfg)
    cfg.datasets.train.transforms = cfg.datasets.test.transforms
    targets = measure_neighbors(
        build_datamodule(cfg), functools.partial(measure_frame_neighbors, model), cfg.num_workers
    )
    logger.info(
        "p99 over the train split: +neighbors=[%s]", ",".join(f"{value:.2f}" for value in targets)
    )


if __name__ == "__main__":
    main()
