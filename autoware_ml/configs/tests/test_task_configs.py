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

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from autoware_ml.configs.resolvers import register_config_resolvers

_CONFIGS = Path(__file__).resolve().parents[1]
# Building blocks and bases are not runnable on their own, only the leaves are.
_BLOCKS = (
    "base",
    "tiny",
    "bev_neck",
    "encoder_",
    "seg3d_head_",
    "voxel_encoder",
    "default_task",
)
EXPERIMENT_CONFIGS = sorted(
    path.relative_to(_CONFIGS).with_suffix("").as_posix()
    for path in (_CONFIGS / "tasks").rglob("*.yaml")
    if not path.stem.startswith(_BLOCKS)
)


def compose_experiment(config_name: str):
    """Compose one experiment config with the run context the training entrypoint provides."""
    register_config_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return cfg


@pytest.mark.parametrize("config_name", EXPERIMENT_CONFIGS)
def test_every_experiment_instantiates_its_model_and_pipelines(config_name: str) -> None:
    cfg = compose_experiment(config_name)

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
    cfg = compose_experiment(config_name)
    scheduler = cfg.model.get("scheduler")
    if scheduler is None or "warmup_epochs" not in scheduler or "decay_epochs" not in scheduler:
        return

    assert not OmegaConf.is_missing(scheduler, "warmup_epochs"), "warmup_epochs is `???`"
    assert not OmegaConf.is_missing(scheduler, "decay_epochs"), "decay_epochs is `???`"
    assert scheduler.warmup_epochs + scheduler.decay_epochs == cfg.trainer.max_epochs
    assert cfg.get("scheduler") is None, "a top-level `scheduler` key is read by nothing"
