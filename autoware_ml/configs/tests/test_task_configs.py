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

"""Every runnable task config composes and builds its model and pipelines."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from autoware_ml.configs.resolvers import register_config_resolvers

_CONFIGS = Path(__file__).resolve().parents[1]
TASK_CONFIGS = sorted(
    path.relative_to(_CONFIGS).with_suffix("").as_posix()
    for path in (_CONFIGS / "tasks").rglob("*.yaml")
    if path.name.endswith("_nuscenes.yaml") or "t4dataset" in path.name
)


def compose_task(config_name: str):
    """Compose one task config with the run context the training entrypoint provides."""
    register_config_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return cfg


@pytest.mark.parametrize("config_name", TASK_CONFIGS)
def test_every_task_instantiates_its_model_and_pipeline(config_name: str) -> None:
    cfg = compose_task(config_name)

    hydra.utils.instantiate(cfg.data_preprocessing)
    hydra.utils.instantiate(cfg.datamodule.train_dataset.transforms)
    hydra.utils.instantiate(cfg.model)
