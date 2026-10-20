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

"""Shared fixtures of the builder tests.

The MLflow and logger builders read the Hydra runtime (``HydraConfig.get()``) for the run
output directory and the task overrides, so the fixture seeds a structured ``HydraConf`` the
way ``hydra.main`` would before the entrypoint runs, and clears it afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import pytest

HYDRA_CONFIG_NAME = "tasks/detection3d/tiny/tiny_b1_1e_corpus"
HYDRA_TASK_OVERRIDES = ("+weights=[/tmp/best.ckpt]", "trainer.max_epochs=1")


@dataclass(frozen=True)
class HydraRuntime:
    """What the seeded Hydra runtime exposes to the builders.

    Attributes:
        work_dir: ``HydraConfig.get().runtime.output_dir`` of the run.
        config_name: ``HydraConfig.get().job.config_name`` of the run.
        overrides: ``HydraConfig.get().overrides.task`` of the run.
    """

    work_dir: Path
    config_name: str
    overrides: tuple[str, ...]


@pytest.fixture
def hydra_runtime(tmp_path: Path) -> Iterator[HydraRuntime]:
    """Seed the global Hydra runtime with a run directory under ``tmp_path``."""
    work_dir = tmp_path / "hydra_run"
    work_dir.mkdir()
    hydra_cfg = OmegaConf.structured(HydraConf)
    hydra_cfg.runtime.output_dir = str(work_dir)
    hydra_cfg.overrides.task = list(HYDRA_TASK_OVERRIDES)
    hydra_cfg.job.config_name = HYDRA_CONFIG_NAME
    HydraConfig.instance().set_config(OmegaConf.create({"hydra": hydra_cfg}))
    yield HydraRuntime(
        work_dir=work_dir, config_name=HYDRA_CONFIG_NAME, overrides=HYDRA_TASK_OVERRIDES
    )
    HydraConfig.instance().cfg = None
