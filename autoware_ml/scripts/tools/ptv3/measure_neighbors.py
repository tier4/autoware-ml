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

This script measures the ``target_neighbors`` of ``make_synthetic_inputs`` on the train
split of a config: the p99 over its frames of every stage's mean occupied neighbors, logged as
the ``+target_neighbors=[...]`` override to pass on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
import logging

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from autoware_ml.builders.database_builder import build_datamodule
from autoware_ml.builders.model_builder import build_data_preprocessor, build_model
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
from autoware_ml.utils.synthetic_inputs import mean_neighbors

logger = logging.getLogger(__name__)
_CONFIG_PATH = get_config_path()


def measure_frame_neighbors(model: PTv3BaseModel, batch: ModelGTBatch) -> list[float]:
    """Return the mean neighbors of every stage of one frame, voxelized and pooled as exported."""
    batch_inputs = model.on_after_batch_transfer(batch.to_device(model.device), dataloader_idx=0)
    inputs = prepare_ptv3_export_inputs(model, batch_inputs)
    stages = [inputs.grid_coord, *(meta.grid_coord for meta in inputs.pooling_metadata)]
    return [mean_neighbors(stage) for stage in stages]


def measure_neighbors(
    datamodule: DataModule,
    neighbors_of: Callable[[ModelGTBatch], Sequence[float]],
    num_workers: int,
) -> list[float]:
    """Return the p99 over the train split of every stage's mean neighbors, finest first.

    Every source of the split is read once, whatever its training repeat, and ``neighbors_of``
    measures one frame.
    """
    datamodule.prepare_data()
    datamodule.setup_splits((SplitType.TRAIN,))
    split = datamodule.datasets[SplitType.TRAIN]
    dataset = ConcatDataset(split.datasets, [1] * len(split.datasets))
    loader = DataLoader(
        dataset, batch_size=1, num_workers=num_workers, collate_fn=dataset.collate_fn
    )
    with torch.no_grad():
        neighbors = [list(neighbors_of(batch)) for batch in loader]
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
            f"Only PTv3 models have synthetic inputs, got {type(model).__name__}."
        )
    cfg.datasets.train.transforms = cfg.datasets.test.transforms
    targets = measure_neighbors(
        build_datamodule(cfg), functools.partial(measure_frame_neighbors, model), cfg.num_workers
    )
    logger.info(
        "p99 over the train split: +target_neighbors=[%s]",
        ",".join(f"{value:.2f}" for value in targets),
    )


if __name__ == "__main__":
    main()
