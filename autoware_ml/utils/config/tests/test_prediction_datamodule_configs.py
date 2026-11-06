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

"""Unit tests for the Argoverse 2 Motion Forecasting database and prediction datamodule configs."""

from __future__ import annotations

import inspect
from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from autoware_ml.databases.argoverse2.av2_motion_database import AV2MotionDatabase
from autoware_ml.databases.argoverse2.av2_motion_scenarios import AV2MotionScenarios
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.datamodule.prediction.dataset import PredictionDataset
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.prediction.loading import LoadAgentTracks, LoadVectorMap
from autoware_ml.types.motion import PolylineType
from autoware_ml.utils.config.resolvers import register_config_resolvers
from autoware_ml.utils.config.settings import CONFIGS_ROOT, Settings
from autoware_ml.utils.config.tests.config_roots import SEARCHPATH_OVERRIDE

# The classes a prediction experiment would declare over the coarse taxonomy.
CLASS_NAMES = ["vehicle", "pedestrian", "cyclist"]


def compose_prediction_experiment(settings: Settings, root_path: Path) -> DictConfig:
    """
    Compose the Argoverse 2 database with the prediction datamodule, as a prediction experiment
    does: the paths, the database, the datamodule and the taxonomy the model declares its
    classes under.

    The config is only composed, not instantiated: instantiating the database or the datamodule
    lists the scenario directories of the dataset.

    Args:
      settings: Settings the path resolvers read, with the data and cache roots set.
      root_path: Directory holding the split directories of the dataset, the database has no
        default for it.

    Returns:
      DictConfig: The composed experiment.
    """
    register_config_resolvers(settings)
    ConfigStore.instance().store(
        name="prediction_experiment",
        node={
            "defaults": [
                "/defaults/modules/paths",
                {"/database@database": "argoverse2/argoverse2_motion_forecasting_base"},
                {"/datamodule@datamodule": "argoverse2/default_prediction_datamodule"},
                {"/database/argoverse2/taxonomy/prediction@taxonomy.prediction": "default_coarse"},
                "_self_",
            ],
            "batch_size": 2,
            "num_workers": 0,
            "taxonomy": {"prediction": {"class_names": CLASS_NAMES}},
        },
        package="_global_",
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_ROOT)):
        cfg = compose(
            config_name="prediction_experiment",
            overrides=[SEARCHPATH_OVERRIDE, f"database.root_path={root_path}"],
            return_hydra_config=True,
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    # The settings come from .env, not the process environment, so the roots are passed in.
    settings = Settings(data_path=tmp_path / "data", cache_path=tmp_path / "cache")
    return compose_prediction_experiment(settings, tmp_path / "data" / "motion_forecasting")


def test_datamodule_config_names_the_arguments_of_the_datamodule(cfg: DictConfig) -> None:
    datamodule = cfg.datamodule
    arguments = set(datamodule) - {"_target_", "_convert_"}
    parameters = set(inspect.signature(DataModule.__init__).parameters) - {"self"}

    assert hydra.utils.get_class(datamodule._target_) is DataModule
    assert arguments == parameters
    assert set(datamodule.sources) == {"main"}


def test_database_config_names_only_arguments_of_the_database(cfg: DictConfig) -> None:
    database = cfg.database
    parameters = set(inspect.signature(AV2MotionDatabase.__init__).parameters) - {"self"}

    assert hydra.utils.get_class(database._target_) is AV2MotionDatabase
    assert set(database) - {"_target_", "_convert_"} <= parameters
    assert database.root_path.endswith("motion_forecasting")
    assert database.cache_path.endswith("prediction/")


def test_scenarios_config_names_the_three_splits_with_a_unit_stride(cfg: DictConfig) -> None:
    scenarios = cfg.database.scenarios.db_av2_motion_forecasting
    parameters = set(inspect.signature(AV2MotionScenarios).parameters)

    assert hydra.utils.get_class(scenarios._target_) is AV2MotionScenarios
    assert set(scenarios) - {"_target_", "_convert_"} <= parameters
    assert scenarios.scenario_root_path == cfg.database.root_path
    assert [params.dataset_name for params in scenarios.dataset_params] == ["train", "val", "test"]
    assert all(params.sample_steps == 1 for params in scenarios.dataset_params)


@pytest.mark.parametrize("split", ["train", "test"])
def test_dataset_configs_name_only_arguments_of_the_dataset(cfg: DictConfig, split: str) -> None:
    dataset = cfg.datasets[split]
    parameters = set(inspect.signature(PredictionDataset.__init__).parameters) - {"self"}

    assert hydra.utils.get_class(dataset._target_) is PredictionDataset
    assert set(dataset) - {"_target_", "_convert_"} <= parameters
    assert dataset.database_root_path == cfg.database.root_path


@pytest.mark.parametrize("split", ["train", "test"])
def test_dataset_pipelines_load_the_tracks_then_the_map(cfg: DictConfig, split: str) -> None:
    # The transforms do not touch the data when built, so they are instantiated: the tracks
    # loader resolves the agent types with the prediction taxonomy of the experiment.
    transforms = hydra.utils.instantiate(cfg.datasets[split].transforms)

    assert isinstance(transforms, TransformsCompose)
    load_tracks, load_map = transforms.pipeline
    assert isinstance(load_tracks, LoadAgentTracks)
    assert isinstance(load_map, LoadVectorMap)
    assert isinstance(load_tracks.taxonomy, LabelTaxonomy)
    assert load_tracks.taxonomy.class_names == tuple(CLASS_NAMES)
    assert load_tracks.taxonomy.resolve_index("bus") == 0
    assert load_tracks.taxonomy.resolve_index("motorcyclist") == 2
    assert load_tracks.taxonomy.resolve_index("riderless_bicycle") == -1
    assert load_map.num_points_per_polyline == 20
    assert load_map.polyline_types == tuple(PolylineType)


def test_taxonomy_config_coarsens_every_fine_name_of_the_vocabulary(cfg: DictConfig) -> None:
    taxonomy = hydra.utils.instantiate(cfg.taxonomy.prediction)

    assert taxonomy.vocabulary == hydra.utils.instantiate(cfg.database.vocabulary)
    assert set(taxonomy.class_mapping) == set(taxonomy.vocabulary.fine_names)
    assert taxonomy.ignore_index == -1
