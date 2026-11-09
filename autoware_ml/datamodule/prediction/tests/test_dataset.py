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

import importlib.util
import tempfile
import unittest
from pathlib import Path

import polars as pl

if importlib.util.find_spec("av2") is None:
    raise unittest.SkipTest("the av2 devkit is not installed")

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


def _write_database(tmp_path: Path) -> AV2MotionDatabase:
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
    """An unbound dataset over the database root, loading the tracks and, by default, the map."""
    pipeline = [LoadAgentTracks(av2_prediction_taxonomy())]
    if with_map:
        pipeline.append(LoadVectorMap(num_points_per_polyline=5))
    return PredictionDataset(
        database_root_path=str(database.root_path), transforms=TransformsCompose(pipeline)
    )


class PredictionDatasetTestCase(unittest.TestCase):
    """Base case that writes the synthetic corpus and its record table per test."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.database = _write_database(Path(self._tmp_dir.name))


class TestPredictionDataset(PredictionDatasetTestCase):
    """The dataset serves one sample per prediction record of the bound table."""

    def test_a_declared_dataset_serves_nothing_until_bound(self) -> None:
        """
        Input: a dataset declared over the database root without a record table.
        Expected: the dataset is unbound, supervises neither perception task, and has no
        length yet.
        Check: is_bound, the two supervision flags and ValueError from len.
        """
        dataset = _dataset(self.database)

        self.assertFalse(dataset.is_bound)
        self.assertFalse(dataset.det3d_supervised)
        self.assertFalse(dataset.seg3d_supervised)
        with self.assertRaisesRegex(ValueError, "not bound"):
            len(dataset)

    def test_bound_dataset_serves_one_sample_per_scenario(self) -> None:
        """
        Input: the dataset bound to the record table of the six scenarios.
        Expected: one sample per record, each scenario its own scene group, and a table
        narrowed to the scenario id, the timestamp and the prediction struct.
        Check: is_bound, the length, samples_per_record, the scene index groups and the
        column names.
        """
        records = self.database.load_polars_scenario_dataframe()

        dataset = _dataset(self.database).bind(records)

        self.assertTrue(dataset.is_bound)
        self.assertEqual(len(dataset), len(TRAIN_IDS) + len(VAL_IDS) + len(TEST_IDS))
        self.assertEqual(dataset.samples_per_record, 1)
        self.assertEqual(dataset.scene_index_groups(), [[index] for index in range(len(dataset))])
        self.assertEqual(
            list(dataset.dataset_records_dataframe.columns),
            [
                DatasetTableSchema.SCENARIO_ID.name,
                DatasetTableSchema.TIMESTAMP_SECONDS.name,
                DatasetTableSchema.PREDICTION.name,
            ],
        )

    def test_data_sample_names_the_files_under_the_dataset_root(self) -> None:
        """
        Input: the first data sample of the bound dataset, the train scenario "a0".
        Expected: the source sample resolves the relative paths of the record under the
        database root and copies the window, the frame meta names the scenario with an
        identity ego pose and no lanelet map, and every other sample field is None before the
        pipeline runs.
        Check: every source field, the frame meta fields, the absent GT sample and the None
        perception fields.
        """
        dataset = _dataset(self.database).bind(self.database.load_polars_scenario_dataframe())

        sample = dataset.get_data_sample(0)

        source = sample.prediction_source_sample
        self.assertIsNotNone(source)
        self.assertEqual(
            source.tracks_path, str(self.database.root_path / "train/a0/scenario_a0.parquet")
        )
        self.assertEqual(
            source.map_path, str(self.database.root_path / "train/a0/log_map_archive_a0.json")
        )
        self.assertIs(source.dataset_format, DatasetFormat.AV2)
        self.assertIsNone(source.scenario_key)
        self.assertEqual(source.focal_track_id, FOCAL_TRACK_ID)
        self.assertEqual(source.first_step_index, 0)
        self.assertEqual(
            (source.current_step_index, source.num_steps, source.num_tracks), (49, 110, 3)
        )
        self.assertAlmostEqual(source.step_seconds, 0.1)
        self.assertEqual(sample.frame_meta.scene_token, "a0")
        self.assertAlmostEqual(sample.frame_meta.timestamp, 1.0 + 4.9)
        self.assertEqual(sample.frame_meta.ego2global.diag().tolist(), [1.0, 1.0, 1.0, 1.0])
        self.assertIsNone(sample.frame_meta.lanelet_map_path)
        self.assertIsNone(sample.prediction_gt_sample)
        for field in (
            "lidar_point_cloud_samples",
            "image_samples",
            "point_cloud_data",
            "camera_image_data",
            "detection3d_gt_bboxes_3d",
            "segmentation3d_gt_sample",
        ):
            self.assertIsNone(getattr(sample, field), field)

    def test_getitem_runs_the_loading_pipeline_and_the_batch_collates(self) -> None:
        """
        Input: the first two samples of the bound dataset, read through getitem and collated.
        Expected: getitem runs the tracks and map loaders and times the IO, and the collated
        batch carries the prediction GT batch only, with the two train scenarios in order.
        Check: the GT sample shapes and timing, the None perception batches, the scene
        tokens, the batch size and the prediction batch shapes, mask and current step.
        """
        dataset = _dataset(self.database).bind(self.database.load_polars_scenario_dataframe())

        first, second = dataset[0], dataset[1]

        self.assertEqual(first.prediction_gt_sample.agents.positions.shape, (3, 110, 3))
        self.assertEqual(first.prediction_gt_sample.map_polylines.points.shape, (6, 5, 3))
        self.assertGreater(first.io_processing_time, 0.0)
        self.assertFalse(first.is_padding)

        batch = dataset.collate_fn([first, second])

        self.assertIsNone(batch.point_cloud_gt_batch)
        self.assertIsNone(batch.detection3d_gt_batch)
        self.assertIsNone(batch.segmentation3d_gt_batch)
        self.assertIsNone(batch.image_gt_batch)
        self.assertEqual(batch.frame_meta_batch.scene_tokens, ["a0", "a1"])
        self.assertEqual(batch.batch_size, 2)
        prediction = batch.prediction_gt_batch
        self.assertEqual(prediction.agent_positions.shape, (2, 3, 110, 3))
        self.assertTrue(prediction.agent_mask.all().item())
        self.assertEqual(prediction.map_points.shape, (2, 6, 5, 3))
        self.assertEqual(prediction.current_step_index, 49)

    def test_a_pipeline_without_the_map_loader_serves_tracks_only(self) -> None:
        """
        Input: a dataset whose pipeline loads the tracks only, collated from one sample.
        Expected: the prediction batch has no map and the agent tensors of one sample.
        Check: map_points and the agent positions shape.
        """
        dataset = _dataset(self.database, with_map=False).bind(
            self.database.load_polars_scenario_dataframe()
        )

        batch = dataset.collate_fn([dataset[0]])

        self.assertIsNone(batch.prediction_gt_batch.map_points)
        self.assertEqual(batch.prediction_gt_batch.agent_positions.shape, (1, 3, 110, 3))

    def test_records_without_a_scenario_are_rejected(self) -> None:
        """
        Input: the record table with its prediction column nulled, bound on its own and
        concatenated with the real records.
        Expected: a prediction dataset serves prediction records only, so a table with a
        record without a scenario is refused, both through bind and the constructor.
        Check: ValueError saying the records carry no prediction scenario.
        """
        records = self.database.load_polars_scenario_dataframe()
        perception = records.with_columns(
            pl.lit(None)
            .cast(DatasetTableSchema.PREDICTION.dtype)
            .alias(DatasetTableSchema.PREDICTION.name)
        )

        with self.assertRaisesRegex(ValueError, "carry no prediction scenario"):
            _dataset(self.database).bind(perception)
        with self.assertRaisesRegex(ValueError, "carry no prediction scenario"):
            PredictionDataset(
                database_root_path=str(self.database.root_path),
                transforms=None,
                dataset_records_dataframe=pl.concat([records, perception]),
            )


class TestPredictionDataModule(PredictionDatasetTestCase):
    """The data module splits the database by scenario and serves every split."""

    def test_datamodule_serves_the_splits_end_to_end(self) -> None:
        """
        Input: a data module over the database with the scenario splitter and loaders of
        batch size 2 without workers.
        Expected: the fit stage serves the two train scenarios in one batch and the val
        scenario alone, and the test stage serves the three 50 step scenarios as batches of
        2 and 1 with the current step at 49.
        Check: the batch sizes, the prediction batch shapes, the scene tokens, the padding
        mask and the current step of the test batch.
        """
        loader = DataLoaderConfig(batch_size=2, num_workers=0)
        datamodule = DataModule(
            splitter=ScenarioSplitter(),
            sources={
                "av2": DatasetSource(
                    database=self.database,
                    train_dataset=_dataset(self.database),
                    validation_dataset=_dataset(self.database),
                    test_dataset=_dataset(self.database),
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
        self.assertEqual(train_batch.batch_size, 2)
        self.assertEqual(train_batch.prediction_gt_batch.agent_positions.shape, (2, 3, 110, 3))
        self.assertEqual(train_batch.prediction_gt_batch.map_points.shape, (2, 6, 5, 3))
        self.assertEqual(sorted(train_batch.frame_meta_batch.scene_tokens), list(TRAIN_IDS))
        self.assertFalse(train_batch.padding_mask.any().item())
        val_batch = next(iter(datamodule.val_dataloader()))
        self.assertEqual(val_batch.frame_meta_batch.scene_tokens, list(VAL_IDS))

        datamodule.setup("test")
        test_batches = list(datamodule.test_dataloader())
        self.assertEqual([batch.batch_size for batch in test_batches], [2, 1])
        self.assertEqual(test_batches[0].prediction_gt_batch.agent_positions.shape, (2, 3, 50, 3))
        self.assertEqual(test_batches[0].prediction_gt_batch.current_step_index, 49)
