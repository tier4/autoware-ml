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

"""Tests for the prediction dataset, from a synthetic Argoverse 2 corpus to a collated batch."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("av2")

from autoware_ml.databases.argoverse2.av2_motion_database import AV2MotionDatabase  # noqa: E402
from autoware_ml.databases.argoverse2.av2_motion_scenarios import AV2MotionScenarios  # noqa: E402
from autoware_ml.databases.argoverse2.tests.av2_fixtures import (  # noqa: E402
    FOCAL_TRACK_ID,
    av2_prediction_taxonomy,
    av2_vocabulary,
    write_scenario,
)
from autoware_ml.databases.scenarios import DatasetParams  # noqa: E402
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema  # noqa: E402
from autoware_ml.datamodule.data_module import DataLoaderConfig, DataModule  # noqa: E402
from autoware_ml.datamodule.prediction.dataset import PredictionDataset  # noqa: E402
from autoware_ml.datamodule.sources import DatasetSource  # noqa: E402
from autoware_ml.datamodule.splitters.dataset_record_splitter import ScenarioSplitter  # noqa: E402
from autoware_ml.transforms.base import TransformsCompose  # noqa: E402
from autoware_ml.transforms.prediction.loading import LoadAgentTracks, LoadVectorMap  # noqa: E402
from autoware_ml.types.dataset import DatasetFormat  # noqa: E402

TRAIN_IDS = ("a0", "a1")
VAL_IDS = ("b0",)
TEST_IDS = ("c0", "c1", "c2")


@pytest.fixture
def database(tmp_path: Path) -> AV2MotionDatabase:
    """A database over a synthetic corpus with its record table written."""
    root = tmp_path / "av2"
    for scenario_id in TRAIN_IDS:
        write_scenario(root / "train", scenario_id)
    for scenario_id in VAL_IDS:
        write_scenario(root / "val", scenario_id)
    for scenario_id in TEST_IDS:
        write_scenario(root / "test", scenario_id, num_steps=50)
    database = AV2MotionDatabase(
        version="AV2MotionForecasting-test-v1.0.0",
        root_path=str(root),
        scenarios={
            "av2": AV2MotionScenarios(
                scenario_root_path=root,
                dataset_params=[
                    DatasetParams(dataset_name=split, sample_steps=1)
                    for split in ("train", "val", "test")
                ],
            )
        },
        cache_path=str(tmp_path / "cache"),
        cache_file_prefix_name="database",
        num_workers=1,
        vocabulary=av2_vocabulary(),
    )
    database.process_scenario_records()
    return database


def _dataset(database: AV2MotionDatabase, with_map: bool = True) -> PredictionDataset:
    pipeline = [LoadAgentTracks(av2_prediction_taxonomy())]
    if with_map:
        pipeline.append(LoadVectorMap(num_points_per_polyline=5))
    return PredictionDataset(
        database_root_path=str(database.root_path), transforms=TransformsCompose(pipeline)
    )


def test_a_declared_dataset_serves_nothing_until_bound(database: AV2MotionDatabase) -> None:
    dataset = _dataset(database)

    assert not dataset.is_bound
    assert not dataset.det3d_supervised
    assert not dataset.seg3d_supervised
    with pytest.raises(ValueError, match="not bound"):
        len(dataset)


def test_bound_dataset_serves_one_sample_per_scenario(database: AV2MotionDatabase) -> None:
    records = database.load_polars_scenario_dataframe()

    dataset = _dataset(database).bind(records)

    assert dataset.is_bound
    assert len(dataset) == len(TRAIN_IDS) + len(VAL_IDS) + len(TEST_IDS)
    assert dataset.samples_per_record == 1
    assert dataset.scene_index_groups() == [[index] for index in range(len(dataset))]
    assert list(dataset.dataset_records_dataframe.columns) == [
        DatasetTableSchema.SCENARIO_ID.name,
        DatasetTableSchema.TIMESTAMP_SECONDS.name,
        DatasetTableSchema.PREDICTION.name,
    ]


def test_data_sample_names_the_files_under_the_dataset_root(database: AV2MotionDatabase) -> None:
    dataset = _dataset(database).bind(database.load_polars_scenario_dataframe())

    sample = dataset.get_data_sample(0)

    source = sample.prediction_source_sample
    assert source is not None
    assert source.tracks_path == str(database.root_path / "train/a0/scenario_a0.parquet")
    assert source.map_path == str(database.root_path / "train/a0/log_map_archive_a0.json")
    assert source.dataset_format is DatasetFormat.AV2
    assert source.scenario_key is None
    assert source.focal_track_id == FOCAL_TRACK_ID
    assert source.first_step_index == 0
    assert (source.current_step_index, source.num_steps, source.num_tracks) == (49, 110, 3)
    assert source.step_seconds == pytest.approx(0.1)
    assert sample.frame_meta.scene_token == "a0"
    assert sample.frame_meta.timestamp == pytest.approx(1.0 + 4.9)
    assert sample.frame_meta.ego2global.diag().tolist() == [1.0, 1.0, 1.0, 1.0]
    assert sample.frame_meta.lanelet_map_path is None
    assert sample.prediction_gt_sample is None
    for field in (
        "lidar_point_cloud_samples",
        "image_samples",
        "point_cloud_data",
        "camera_image_data",
        "detection3d_gt_bboxes_3d",
        "segmentation3d_gt_sample",
    ):
        assert getattr(sample, field) is None


def test_getitem_runs_the_loading_pipeline_and_the_batch_collates(
    database: AV2MotionDatabase,
) -> None:
    dataset = _dataset(database).bind(database.load_polars_scenario_dataframe())

    first, second = dataset[0], dataset[1]

    assert first.prediction_gt_sample.agents.positions.shape == (3, 110, 3)
    assert first.prediction_gt_sample.map_polylines.points.shape == (6, 5, 3)
    assert first.io_processing_time > 0.0
    assert not first.is_padding

    batch = dataset.collate_fn([first, second])

    assert batch.point_cloud_gt_batch is None
    assert batch.detection3d_gt_batch is None
    assert batch.segmentation3d_gt_batch is None
    assert batch.image_gt_batch is None
    assert batch.frame_meta_batch.scene_tokens == ["a0", "a1"]
    assert batch.batch_size == 2
    prediction = batch.prediction_gt_batch
    assert prediction.agent_positions.shape == (2, 3, 110, 3)
    assert prediction.agent_mask.all()
    assert prediction.map_points.shape == (2, 6, 5, 3)
    assert prediction.current_step_index == 49


def test_a_pipeline_without_the_map_loader_serves_tracks_only(
    database: AV2MotionDatabase,
) -> None:
    dataset = _dataset(database, with_map=False).bind(database.load_polars_scenario_dataframe())

    batch = dataset.collate_fn([dataset[0]])

    assert batch.prediction_gt_batch.map_points is None
    assert batch.prediction_gt_batch.agent_positions.shape == (1, 3, 110, 3)


def test_records_without_a_scenario_are_rejected(database: AV2MotionDatabase) -> None:
    records = database.load_polars_scenario_dataframe()
    perception = records.with_columns(
        pl.lit(None)
        .cast(DatasetTableSchema.PREDICTION.dtype)
        .alias(DatasetTableSchema.PREDICTION.name)
    )

    with pytest.raises(ValueError, match="carry no prediction scenario"):
        _dataset(database).bind(perception)
    with pytest.raises(ValueError, match="carry no prediction scenario"):
        PredictionDataset(
            database_root_path=str(database.root_path),
            transforms=None,
            dataset_records_dataframe=pl.concat([records, perception]),
        )


def test_datamodule_serves_the_splits_end_to_end(database: AV2MotionDatabase) -> None:
    loader = DataLoaderConfig(batch_size=2, num_workers=0)
    datamodule = DataModule(
        splitter=ScenarioSplitter(),
        sources={
            "av2": DatasetSource(
                database=database,
                train_dataset=_dataset(database),
                validation_dataset=_dataset(database),
                test_dataset=_dataset(database),
            )
        },
        train_dataloader=loader,
        validation_dataloader=loader,
        test_dataloader=loader,
        predict_dataloader=None,
        train_frame_sampling=None,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")
    train_batch = next(iter(datamodule.train_dataloader()))
    assert train_batch.batch_size == 2
    assert train_batch.prediction_gt_batch.agent_positions.shape == (2, 3, 110, 3)
    assert train_batch.prediction_gt_batch.map_points.shape == (2, 6, 5, 3)
    assert sorted(train_batch.frame_meta_batch.scene_tokens) == list(TRAIN_IDS)
    assert not train_batch.padding_mask.any()
    val_batch = next(iter(datamodule.val_dataloader()))
    assert val_batch.frame_meta_batch.scene_tokens == list(VAL_IDS)

    datamodule.setup("test")
    test_batches = list(datamodule.test_dataloader())
    assert [batch.batch_size for batch in test_batches] == [2, 1]
    assert test_batches[0].prediction_gt_batch.agent_positions.shape == (2, 3, 50, 3)
    assert test_batches[0].prediction_gt_batch.current_step_index == 49
