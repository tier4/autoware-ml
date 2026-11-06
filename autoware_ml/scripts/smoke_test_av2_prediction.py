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

"""Smoke test of the Argoverse 2 Motion Forecasting data layer on the real dataset.

The script composes the bundled database and prediction datamodule configs over a subset of
the scenarios, builds the record table, serves one training batch and one test batch and checks
their shapes. It takes the directory holding the ``train``, ``val`` and ``test`` split
directories of the dataset, laid out as ``<split>/<scenario_id>/``::

    python -m autoware_ml.scripts.smoke_test_av2_prediction --root-path /path/to/motion_forecasting

Every ``--sample-steps``-th scenario of each split is kept, so the default stride of 1000 covers
about 250 of the 250k scenarios. ``--sample-steps 1`` builds the full record table.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.utils.config.resolvers import register_config_resolvers
from autoware_ml.utils.config.settings import CONFIGS_ROOT, SETTINGS_FILE, Settings

logger = logging.getLogger(__name__)

# The classes the smoke run declares over the coarse taxonomy, as a prediction model would.
CLASS_NAMES = ("vehicle", "pedestrian", "cyclist")
# Steps of the window, the train / val scenarios ship the observed and the future steps, the
# test scenarios the observed ones only.
TRAIN_STEPS = 110
TEST_STEPS = 50
CURRENT_STEP_INDEX = 49


def parse_args() -> argparse.Namespace:
    """
    Parse the command line.

    Returns:
      argparse.Namespace: The arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--root-path",
        type=Path,
        required=True,
        help="Directory holding the train, val and test split directories of the dataset.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Root of the record tables, AUTOWARE_ML_CACHE_PATH of .env or .cache of the "
        "repository when not given.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=1000,
        help="Stride over the scenarios of every split, 1 keeps every scenario.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size of the loaders.")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Worker processes of the dataloaders."
    )
    parser.add_argument(
        "--database-num-workers",
        type=int,
        default=8,
        help="Worker processes building the record table.",
    )
    return parser.parse_args()


def resolve_settings(cache_path: Path | None) -> Settings:
    """
    Settings of the run: the .env of the checkout, with the given cache root on top.

    Args:
      cache_path: Cache root, None reads it from .env.

    Returns:
      Settings: The settings.
    """
    settings = Settings.read(SETTINGS_FILE) if SETTINGS_FILE.is_file() else Settings()
    return Settings(
        data_path=settings.data_path,
        cache_path=cache_path.resolve() if cache_path is not None else settings.cache_path,
        configs_path=settings.configs_path,
        scenarios_path=settings.scenarios_path,
        working_dir=settings.working_dir,
    )


def compose_config(settings: Settings, args: argparse.Namespace) -> DictConfig:
    """
    Compose the database and the prediction datamodule over a subset of the scenarios.

    Args:
      settings: Settings the path resolvers read.
      args: Command line arguments.

    Returns:
      DictConfig: The composed config.
    """
    register_config_resolvers(settings)
    ConfigStore.instance().store(
        name="av2_prediction_smoke_test",
        node={
            "defaults": [
                "/defaults/modules/paths",
                {"/database@database": "argoverse2/argoverse2_motion_forecasting_base"},
                {"/datamodule@datamodule": "argoverse2/default_prediction_datamodule"},
                {"/database/argoverse2/taxonomy/prediction@taxonomy.prediction": "default_coarse"},
                "_self_",
            ],
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "taxonomy": {"prediction": {"class_names": list(CLASS_NAMES)}},
        },
        package="_global_",
    )
    scenarios = "database.scenarios.db_av2_motion_forecasting.dataset_params"
    overrides = [
        f"database.root_path={args.root_path.resolve()}",
        f"database.num_workers={args.database_num_workers}",
        *(f"{scenarios}.{index}.sample_steps={args.sample_steps}" for index in range(3)),
    ]
    if settings.configs_path is not None:
        overrides.append(f"hydra.searchpath=[file://{settings.configs_path}]")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_ROOT)):
        return compose(config_name="av2_prediction_smoke_test", overrides=overrides)


def check_batch(batch: ModelGTBatch, num_steps: int, batch_size: int) -> None:
    """
    Check the shapes of a collated prediction batch.

    Args:
      batch: The batch.
      num_steps: Steps of the window every scenario of the split ships.
      batch_size: Scenarios in the batch.
    """
    gt = batch.prediction_gt_batch
    if gt is None:
        raise AssertionError("The batch carries no prediction ground truth.")
    num_agents = gt.agent_positions.shape[1]
    num_polylines = gt.map_points.shape[1]
    num_points = gt.map_points.shape[2]
    assert gt.agent_positions.shape == (batch_size, num_agents, num_steps, 3), (
        gt.agent_positions.shape
    )
    assert gt.agent_headings.shape == (batch_size, num_agents, num_steps), gt.agent_headings.shape
    assert gt.agent_velocities.shape == (batch_size, num_agents, num_steps, 2)
    assert gt.agent_valid.shape == (batch_size, num_agents, num_steps)
    assert (
        gt.agent_types.shape
        == gt.agent_roles.shape
        == gt.agent_mask.shape
        == (batch_size, num_agents)
    )
    assert gt.map_points.shape == (batch_size, num_polylines, num_points, 3), gt.map_points.shape
    assert gt.map_valid.shape == (batch_size, num_polylines, num_points)
    assert gt.map_polyline_types.shape == gt.polyline_mask.shape == (batch_size, num_polylines)
    assert gt.current_step_index == CURRENT_STEP_INDEX, gt.current_step_index
    assert gt.agent_mask.any(dim=1).all(), "a scenario without agents"
    assert gt.polyline_mask.any(dim=1).all(), "a scenario without map polylines"
    # The focal track is row 0 and is valid at the current step
    assert gt.agent_valid[:, 0, CURRENT_STEP_INDEX].all(), "focal track invalid at the current step"
    logger.info(
        f"Batch of {batch_size} scenarios {list(batch.frame_meta_batch.scene_tokens)}: "
        f"agents {tuple(gt.agent_positions.shape)}, polylines {tuple(gt.map_points.shape)}, "
        f"agents per scenario {gt.agent_mask.sum(dim=1).tolist()}, "
        f"polylines per scenario {gt.polyline_mask.sum(dim=1).tolist()}, "
        f"step {gt.step_seconds} s, current step {gt.current_step_index}"
    )


def main() -> None:
    """Build the record table of the subset and serve one batch of the train and test splits."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    settings = resolve_settings(args.cache_path)
    cfg = compose_config(settings, args)
    logger.info(
        f"Database root {cfg.database.root_path}, cache {cfg.database.cache_path}, "
        f"scenario stride {args.sample_steps}"
    )

    start = time.perf_counter()
    datamodule: DataModule = hydra.utils.instantiate(cfg.datamodule)
    logger.info(f"Instantiated the datamodule in {time.perf_counter() - start:.1f} s")

    start = time.perf_counter()
    datamodule.prepare_data()
    database = next(iter(datamodule.databases()))
    logger.info(
        f"Record table {database.cache_file_path} ready in {time.perf_counter() - start:.1f} s"
    )

    datamodule.setup("fit")
    start = time.perf_counter()
    train_batch = next(iter(datamodule.train_dataloader()))
    logger.info(f"Served the first training batch in {time.perf_counter() - start:.2f} s")
    check_batch(train_batch, TRAIN_STEPS, args.batch_size)
    val_batch = next(iter(datamodule.val_dataloader()))
    check_batch(val_batch, TRAIN_STEPS, args.batch_size)

    datamodule.setup("test")
    test_batch = next(iter(datamodule.test_dataloader()))
    check_batch(test_batch, TEST_STEPS, args.batch_size)
    logger.info("Smoke test passed.")


if __name__ == "__main__":
    main()
