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

"""Unit tests for the datamodule configs a model config composes."""

from __future__ import annotations

import inspect
import unittest

import hydra
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.datamodule.streaming_data_module import StreamingDataModule

# An experiment composing a database datamodule, which does not stream.
EXPERIMENT = "tasks/detection3d/centerpoint/t4dataset/voxel024_second_secfpn_b16_30e_t4dataset_120m"


def compose_streaming_experiment(experiment: str) -> DictConfig:
    """
    Compose an experiment with the streaming config after it, as a temporal model's config does.

    The two are composed by the defaults list of a primary config held in the config store.

    Args:
      experiment: Experiment config to compose, relative to the bundled configs.

    Returns:
      DictConfig: The composed experiment.
    """
    register_config_resolvers()
    ConfigStore.instance().store(
        name="streaming_experiment",
        node={
            "defaults": [
                f"/{experiment}",
                {"/datamodule/streaming@datamodule": "default_streaming_datamodule"},
                "_self_",
            ]
        },
        package="_global_",
    )
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        cfg = compose(config_name="streaming_experiment", return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return cfg


class TestStreamingDataModuleConfig(unittest.TestCase):
    """The streaming config layered on the database datamodule of an experiment."""

    def test_turns_the_database_datamodule_into_a_streaming_one(self) -> None:
        """
        Input: an experiment whose database datamodule is followed by the streaming config.
        Expected: the streaming config swaps the class for StreamingDataModule and passes it
        the experiment seed, keeping the sources and dataloaders of the database datamodule,
        so the composed node holds exactly the arguments StreamingDataModule takes.
        Check: the class the node instantiates, its seed and its keys against the parameters
        of the constructor.
        """
        cfg = compose_streaming_experiment(EXPERIMENT)

        datamodule = cfg.datamodule
        arguments = set(datamodule) - {"_target_", "_convert_"}
        parameters = set(inspect.signature(StreamingDataModule.__init__).parameters) - {"self"}
        self.assertIs(hydra.utils.get_class(datamodule._target_), StreamingDataModule)
        self.assertEqual(datamodule.seed, cfg.seed)
        self.assertEqual(arguments, parameters)


if __name__ == "__main__":
    unittest.main()
