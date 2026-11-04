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

"""Every runnable experiment config composes and builds its model and pipelines."""

from __future__ import annotations

import inspect
from pathlib import Path

import hydra
import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from autoware_ml.utils.config.tests.config_roots import compose_task, task_config_names

# Building blocks and bases are not runnable on their own, only the leaves are.
_BLOCKS = (
    "base",
    "large",
    "bev_neck",
    "encoder",
    "seg3d_head",
    "voxel_encoder",
    "default_task",
)
EXPERIMENT_CONFIGS = [
    name for name in task_config_names() if not name.rsplit("/", 1)[-1].startswith(_BLOCKS)
]


def write_placeholder_artifacts(cfg: DictConfig) -> None:
    """Write the files a run prepares beforehand: the anchors of a Sparse4D head."""
    anchor_file = OmegaConf.select(cfg, "model.bbox_head.anchor_file")
    if anchor_file is not None:
        Path(anchor_file).parent.mkdir(parents=True, exist_ok=True)
        np.save(anchor_file, np.zeros((cfg.model.bbox_head.num_anchor, 10), dtype=np.float32))


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS)
def test_every_experiment_instantiates_its_model_and_pipelines(
    config_name: str, tmp_path: Path
) -> None:
    cfg = compose_task(config_name, overrides=[f"cache_root_path={tmp_path}"])
    write_placeholder_artifacts(cfg)

    data_preprocessor = hydra.utils.instantiate(cfg.data_preprocessor)
    hydra.utils.instantiate(cfg.datasets.train.transforms)
    hydra.utils.instantiate(cfg.datasets.test.transforms)
    hydra.utils.instantiate(cfg.model, data_preprocessor=data_preprocessor)


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS)
def test_every_warmup_decay_schedule_spans_the_training_epochs(config_name: str) -> None:
    # The scheduler is a partial, so a `???` epoch count only surfaces when training starts, and
    # a schedule shorter than the run keeps evaluating the cosine past its end and climbs back
    # towards the peak learning rate. Every leaf therefore sets both counts to add up to
    # trainer.max_epochs. A top-level `scheduler` key is read by nothing and is a misplaced one.
    cfg = compose_task(config_name)
    scheduler = cfg.model.get("scheduler")
    if scheduler is None or "warmup_epochs" not in scheduler or "decay_epochs" not in scheduler:
        return

    assert not OmegaConf.is_missing(scheduler, "warmup_epochs"), "warmup_epochs is `???`"
    assert not OmegaConf.is_missing(scheduler, "decay_epochs"), "decay_epochs is `???`"
    assert scheduler.warmup_epochs + scheduler.decay_epochs == cfg.trainer.max_epochs
    assert cfg.get("scheduler") is None, "a top-level `scheduler` key is read by nothing"


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS)
def test_every_database_config_names_only_arguments_of_its_database(config_name: str) -> None:
    # The database is built from the records on disk, so the model and pipeline test above does
    # not build it. A key its class does not take only fails when a run reads the records.
    database = compose_task(config_name).database
    parameters = inspect.signature(hydra.utils.get_class(database._target_)).parameters

    assert set(database) - {"_target_", "_convert_"} <= set(parameters)
