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

"""Unit tests for serving one split from several dataset sources."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
import tempfile
import unittest

import polars as pl

from autoware_ml.databases.base_database import BaseDatabase
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.taxonomy import DatabaseTaxonomy, DetectionTaxonomy
from autoware_ml.datamodule.data_module import DataLoaderConfig
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.datamodule.samplers import (
    DistributedWeightedRandomSampler,
    FrameSamplingConfig,
)
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.common.detection3d import Detection3DTask
from autoware_ml.datamodule.t4dataset.segmentation3d import T4Segmentation3DTask
from autoware_ml.datamodule.tests.corpus import (
    SEGMENTATION_TAXONOMY,
    VOCABULARY,
    build_transforms,
    write_corpus,
)
from autoware_ml.types.dataset import SplitType


FRAME_SAMPLING = FrameSamplingConfig(
    repeat_sampling_factor=0.3,
    object_bev_range=[-100.0, -100.0, 100.0, 100.0],
    pedestrian_class_name="pedestrian",
    low_pedestrian_height_threshold=1.5,
    low_pedestrian_bev_range=[-50.0, -50.0, 50.0, 50.0],
    class_names=["car", "pedestrian"],
    ignore_index=-1,
    filter_attributes=[],
    seed=0,
)


def build_taxonomy(class_name: str = "car") -> DatabaseTaxonomy:
    """
    Build the taxonomy of the synthetic corpora.

    Args:
      class_name: Detection class the car labels train as, changed to get a different
        taxonomy.

    Returns:
      DatabaseTaxonomy: A fresh taxonomy object on every call.
    """
    detection = DetectionTaxonomy(
        VOCABULARY,
        (class_name,),
        {"car": class_name},
        -1,
        {"vehicle": (class_name,)},
        eval_range={class_name: 121.0},
        collision_kinds={class_name: "wheeled"},
        living_speeds={},
    )
    return DatabaseTaxonomy(detection3d=detection, segmentation3d=SEGMENTATION_TAXONOMY)


class FakeDatabase(BaseDatabase):
    """A database serving a record table written to a temporary directory."""

    def __init__(
        self, root: Path, records: pl.DataFrame, version: str, taxonomy: DatabaseTaxonomy
    ) -> None:
        """
        Build a database over an already written corpus.

        Args:
          root: Root directory the record paths resolve against.
          records: Record table of the corpus.
          version: Version naming the database.
          taxonomy: Taxonomy the datamodule compares across sources.
        """
        self._root_path = root
        self._records = records
        self._version = version
        self._taxonomy = taxonomy
        self._lidar_intensity_scale = 255.0
        self.processed = 0

    def __str__(self) -> str:
        """String representation of the database."""
        return f"FakeDatabase({self._version})"

    def __hash__(self) -> int:
        """Hash the database by its version."""
        return hash(self._version)

    def __eq__(self, other: object) -> bool:
        """Compare two databases by their version."""
        return isinstance(other, FakeDatabase) and self._version == other._version

    @property
    def root_path(self) -> Path:
        """Root directory of the corpus."""
        return self._root_path

    @property
    def version(self) -> str:
        """Version of the database."""
        return self._version

    @property
    def taxonomy(self) -> DatabaseTaxonomy:
        """Taxonomy the labels of the database are built with."""
        return self._taxonomy

    @property
    def database_hash(self) -> str:
        """Hash identifying the record table of the database."""
        return self._version

    @property
    def scenarios(self) -> Mapping[str, object]:
        """Scenario groups of the database."""
        return {}

    def load_polars_scenario_dataframe(self) -> pl.DataFrame:
        """Return the record table of the corpus."""
        return self._records

    def process_scenario_records(self) -> None:
        """Count the calls, the datamodule prepares every database once."""
        self.processed += 1


class SingleSplitSplitter:
    """A splitter serving every record of a database to every split."""

    def __str__(self) -> str:
        """String representation of the splitter."""
        return "SingleSplitSplitter"

    def split_by_polars_dataframe(
        self, dataset_records_dataframe: pl.DataFrame, scenarios: Mapping[str, object]
    ) -> Mapping[SplitType, pl.DataFrame]:
        """Return the same records for every split."""
        return {split: dataset_records_dataframe for split in SplitType}


def build_dataset(database: BaseDatabase, det3d: bool = True, seg3d: bool = True) -> T4Dataset:
    """
    Return the dataset of a split as a config declares it, reading a database, bound to no
    records yet.

    Args:
      database: Database the dataset reads, as a config names it through the root path.
      det3d: Whether the boxes of the corpus supervise the run.
      seg3d: Whether the semantic masks of the corpus supervise the run.

    Returns:
      T4Dataset: The declared dataset.
    """
    return T4Dataset(
        database_root_path=str(database.root_path),
        max_num_3d_gt_bboxes=8,
        transforms=build_transforms(),
        dataset_tasks={
            "detection3d": Detection3DTask,
            "segmentation3d": partial(T4Segmentation3DTask, taxonomy=SEGMENTATION_TAXONOMY),
        },
        lidar_intensity_scale=database.lidar_intensity_scale,
        det3d_supervised=det3d,
        seg3d_supervised=seg3d,
    )


def build_source(
    database: BaseDatabase,
    det3d: bool = True,
    seg3d: bool = True,
    repeat: int = 1,
    train: bool = True,
    validation: bool = True,
) -> DatasetSource:
    """
    Declare a source over a database with a fresh dataset for every split it serves.

    Args:
      database: Database of the source, read by every dataset of the source.
      det3d: Whether the boxes of its datasets supervise the run.
      seg3d: Whether the semantic masks of its datasets supervise the run.
      repeat: Share of the training epoch.
      train: Whether the source serves the training split.
      validation: Whether the source serves the validation split.

    Returns:
      DatasetSource: The declared source.
    """
    return DatasetSource(
        database=database,
        train_dataset=build_dataset(database, det3d, seg3d) if train else None,
        validation_dataset=build_dataset(database, det3d, seg3d) if validation else None,
        repeat=repeat,
    )


class MultiSourceTestCase(unittest.TestCase):
    """Shared fixtures writing two corpora and declaring them as sources."""

    def setUp(self) -> None:
        """Write two synthetic corpora, one standing in for a seg3d GT one."""
        self._directory = tempfile.TemporaryDirectory()
        root = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)

        self.main_root = root / "main"
        self.seg3d_gt_root = root / "seg3d_gt"
        self.main_records = write_corpus(self.main_root, num_records=3)
        self.seg3d_gt_records = write_corpus(self.seg3d_gt_root, num_records=1)
        self.main_database = FakeDatabase(
            self.main_root, self.main_records, "main", build_taxonomy()
        )
        self.seg3d_gt_database = FakeDatabase(
            self.seg3d_gt_root, self.seg3d_gt_records, "seg3d_gt", build_taxonomy()
        )

    def build_data_module(
        self,
        sources: Mapping[str, DatasetSource],
        train_dataloader: DataLoaderConfig | None = None,
        train_frame_sampling: FrameSamplingConfig | None = None,
    ) -> DataModule:
        """Build a datamodule over the given sources with a splitter serving every record."""
        return DataModule(
            splitter=SingleSplitSplitter(),
            sources=sources,
            train_dataloader=train_dataloader,
            validation_dataloader=None,
            test_dataloader=None,
            predict_dataloader=None,
            train_frame_sampling=train_frame_sampling,
        )


class TestDatasetSource(unittest.TestCase):
    """The declaration of one dataset source."""

    def setUp(self) -> None:
        """Build a database the sources can name."""
        self.database = FakeDatabase(Path("/"), pl.DataFrame(), "v", build_taxonomy())

    def test_rejects_a_repeat_below_one(self) -> None:
        """
        Input: a source asking to appear zero times per epoch.
        Expected: a source that contributes nothing is a mistake, not a way to disable it.
        Check: a ValueError naming the repeat is raised.
        """
        with self.assertRaisesRegex(ValueError, "repeat must be at least 1"):
            DatasetSource(
                database=self.database, train_dataset=build_dataset(self.database), repeat=0
            )

    def test_rejects_a_source_without_a_dataset(self) -> None:
        """
        Input: a source declaring no dataset for any split.
        Expected: a source serves the splits it declares a dataset for, so one without any
        would serve nothing.
        Check: a ValueError naming the splits is raised.
        """
        with self.assertRaisesRegex(ValueError, "at least one split"):
            DatasetSource(database=self.database)

    def test_rejects_a_repeat_without_a_training_dataset(self) -> None:
        """
        Input: a repeated source that stays out of the training split.
        Expected: repeats only balance the training epoch, a repeated frame would count twice
        in the metrics, so the repeat would have no effect.
        Check: a ValueError naming the training split is raised.
        """
        with self.assertRaisesRegex(ValueError, "training"):
            DatasetSource(
                database=self.database, test_dataset=build_dataset(self.database), repeat=2
            )

    def test_rejects_a_dataset_reading_another_database(self) -> None:
        """
        Input: a source whose dataset config names the root path of another database.
        Expected: a dataset reads the database of its source, a dataset node shared across
        sources would silently read the wrong corpus, so the mismatch is rejected at
        declaration.
        Check: a ValueError naming both root paths is raised.
        """
        other = FakeDatabase(Path("/elsewhere"), pl.DataFrame(), "other", build_taxonomy())

        with self.assertRaisesRegex(ValueError, "reads /elsewhere"):
            DatasetSource(database=self.database, train_dataset=build_dataset(other))

    def test_maps_the_declared_datasets_to_their_splits(self) -> None:
        """
        Input: a source declaring a training and a test dataset.
        Expected: the source serves exactly those two splits.
        Check: the dataset of each split is the declared one, the others are None.
        """
        train_dataset = build_dataset(self.database)
        test_dataset = build_dataset(self.database)

        source = DatasetSource(
            database=self.database, train_dataset=train_dataset, test_dataset=test_dataset
        )

        self.assertEqual(set(source.datasets), {SplitType.TRAIN, SplitType.TEST})
        self.assertIs(source.dataset(SplitType.TRAIN), train_dataset)
        self.assertIs(source.dataset(SplitType.TEST), test_dataset)
        self.assertIsNone(source.dataset(SplitType.VAL))
        self.assertIsNone(source.dataset(SplitType.PREDICT))


class TestDataModuleSources(MultiSourceTestCase):
    """Assembling the splits from several corpora."""

    def test_keeps_the_declared_sources(self) -> None:
        """
        Input: a source instantiated by the config.
        Expected: the datamodule keeps the sources as declared, by name.
        Check: the kept source is the declared one with its database, datasets and repeat.
        """
        source = build_source(self.main_database, seg3d=False, repeat=3)

        data_module = self.build_data_module({"main": source})

        self.assertEqual(data_module.sources, {"main": source})
        self.assertIs(source.database, self.main_database)
        assert source.train_dataset is not None
        self.assertTrue(source.train_dataset.det3d_supervised)
        self.assertFalse(source.train_dataset.seg3d_supervised)
        self.assertEqual(source.repeat, 3)

    def test_rejects_a_datamodule_without_sources(self) -> None:
        """
        Input: no source at all.
        Expected: a datamodule must read at least one source.
        Check: a ValueError is raised at construction.
        """
        with self.assertRaisesRegex(ValueError, "at least one dataset source"):
            self.build_data_module({})

    def test_a_split_is_served_by_the_sources_declaring_its_dataset(self) -> None:
        """
        Input: a source serving training and validation beside one serving training only.
        Expected: a split reads the sources declaring a dataset for it, in declaration order,
        and a split no source declares is empty.
        Check: the names of the sources of every split.
        """
        data_module = self.build_data_module(
            {
                "main": build_source(self.main_database),
                "seg3d_gt": build_source(self.seg3d_gt_database, validation=False),
            }
        )

        self.assertEqual(list(data_module.split_sources(SplitType.TRAIN)), ["main", "seg3d_gt"])
        self.assertEqual(list(data_module.split_sources(SplitType.VAL)), ["main"])
        self.assertEqual(dict(data_module.split_sources(SplitType.TEST)), {})

    def test_a_single_source_serves_its_own_records(self) -> None:
        """
        Input: one source over a three frame corpus.
        Expected: the ordinary single corpus run is one source, so the split holds exactly
        the records of that corpus.
        Check: read the length of the training split.
        """
        data_module = self.build_data_module({"main": build_source(self.main_database)})

        data_module.setup("fit")

        self.assertEqual(len(data_module.datasets[SplitType.TRAIN]), 3)

    def test_sources_are_concatenated_in_declaration_order(self) -> None:
        """
        Input: a three frame corpus followed by a one frame corpus.
        Expected: the split serves every record of every source in declaration order, so
        mixing corpora needs no change to the datasets themselves.
        Check: the split holds the records of both corpora, the main one first.
        """
        data_module = self.build_data_module(
            {
                "main": build_source(self.main_database),
                "seg3d_gt": build_source(self.seg3d_gt_database),
            }
        )

        data_module.setup("fit")

        split = data_module.datasets[SplitType.TRAIN]
        self.assertEqual(len(split), 4)
        self.assertEqual(
            [dataset.database_root_path for dataset in split.datasets],
            [self.main_root, self.seg3d_gt_root],
        )

    def test_a_repeated_source_takes_a_larger_share_of_the_epoch(self) -> None:
        """
        Input: a one frame corpus repeated six times beside a three frame one.
        Expected: repetition is how a small corpus keeps a share of the epoch next to a large
        one, so its frames appear that many times.
        Check: the split holds three plus six samples.
        """
        data_module = self.build_data_module(
            {
                "main": build_source(self.main_database),
                "seg3d_gt": build_source(self.seg3d_gt_database, repeat=6),
            }
        )

        data_module.setup("fit")

        self.assertEqual(len(data_module.datasets[SplitType.TRAIN]), 9)

    def test_a_repeat_only_balances_the_training_split(self) -> None:
        """
        Input: a repeated source that also serves the validation split.
        Expected: a repeated frame would count twice in the metrics, so the validation split
        reads every source once.
        Check: the validation split holds three plus one samples.
        """
        data_module = self.build_data_module(
            {
                "main": build_source(self.main_database),
                "seg3d_gt": build_source(self.seg3d_gt_database, repeat=6),
            }
        )

        data_module.setup("fit")

        self.assertEqual(len(data_module.datasets[SplitType.VAL]), 4)

    def test_every_database_is_prepared_once(self) -> None:
        """
        Input: one corpus declared by two sources with different supervision.
        Expected: the record table of a database is generated once however many splits and
        sources read it.
        Check: the database counted one preparation call.
        """
        data_module = self.build_data_module(
            {
                "main": build_source(self.main_database),
                "main_boxless": build_source(self.main_database, det3d=False),
            }
        )

        data_module.prepare_data()

        self.assertEqual(self.main_database.processed, 1)

    def test_accepts_sources_whose_taxonomies_are_equal(self) -> None:
        """
        Input: two corpora whose taxonomies are separate but equal objects.
        Expected: taxonomies are compared by their definition, not by identity.
        Check: the datamodule builds and keeps both sources.
        """
        self.assertIsNot(self.main_database.taxonomy, self.seg3d_gt_database.taxonomy)

        data_module = self.build_data_module(
            {
                "main": build_source(self.main_database),
                "seg3d_gt": build_source(self.seg3d_gt_database),
            }
        )

        self.assertEqual(len(data_module.sources), 2)

    def test_rejects_sources_whose_taxonomies_disagree(self) -> None:
        """
        Input: two corpora built with different taxonomies.
        Expected: the class index of a label is what the model learns, so corpora that name
        their classes differently cannot share a run.
        Check: a ValueError is raised at construction.
        """
        other = FakeDatabase(
            self.seg3d_gt_root, self.seg3d_gt_records, "other", build_taxonomy("vehicle")
        )

        with self.assertRaisesRegex(ValueError, "same taxonomy"):
            self.build_data_module(
                {"main": build_source(self.main_database), "other": build_source(other)}
            )

    def test_a_split_without_sources_has_no_dataloader(self) -> None:
        """
        Input: a datamodule whose sources declare no test dataset.
        Expected: an undeclared split is absent rather than silently empty.
        Check: asking for the test dataloader raises.
        """
        data_module = self.build_data_module({"main": build_source(self.main_database)})
        data_module.setup(None)

        with self.assertRaisesRegex(ValueError, "has no dataset"):
            data_module.test_dataloader()


class TestDatasetBinding(MultiSourceTestCase):
    """Binding the dataset of a source to its records."""

    def test_a_declared_dataset_has_no_samples(self) -> None:
        """
        Input: a dataset as a config declares it, without records.
        Expected: it serves nothing until the datamodule binds it to records.
        Check: asking for its length raises a ValueError naming the binding.
        """
        dataset = build_dataset(self.main_database)

        self.assertFalse(dataset.is_bound)
        with self.assertRaisesRegex(ValueError, "not bound"):
            len(dataset)

    def test_bind_returns_a_copy_serving_the_records(self) -> None:
        """
        Input: a declared dataset whose masks are unused and the records of its database.
        Expected: the copy serves the records with the root path, the intensity scale and the
        supervision the dataset was declared with, while the declared dataset stays unbound.
        Check: compare the copy and the declared dataset.
        """
        dataset = build_dataset(self.main_database, seg3d=False)

        bound = dataset.bind(self.main_records)

        self.assertIsNot(bound, dataset)
        self.assertEqual(bound.database_root_path, self.main_database.root_path)
        self.assertEqual(bound.lidar_intensity_scale, self.main_database.lidar_intensity_scale)
        self.assertTrue(bound.det3d_supervised)
        self.assertFalse(bound.seg3d_supervised)
        self.assertEqual(len(bound), 3)
        self.assertFalse(dataset.is_bound)

    def test_every_source_binds_its_dataset_to_its_own_records(self) -> None:
        """
        Input: two sources over different corpora, each with a dataset reading its database.
        Expected: the bound dataset of every source reads the root path of that database and
        serves its records, and the declared datasets stay unbound.
        Check: the bound datasets of the split and the declared ones.
        """
        main = build_source(self.main_database)
        seg3d_gt = build_source(self.seg3d_gt_database)
        data_module = self.build_data_module({"main": main, "seg3d_gt": seg3d_gt})

        data_module.setup("fit")

        bound_main, bound_seg3d_gt = data_module.datasets[SplitType.TRAIN].datasets
        self.assertEqual(bound_main.database_root_path, self.main_database.root_path)
        self.assertEqual(bound_seg3d_gt.database_root_path, self.seg3d_gt_database.root_path)
        assert main.train_dataset is not None
        assert seg3d_gt.train_dataset is not None
        self.assertFalse(main.train_dataset.is_bound)
        self.assertFalse(seg3d_gt.train_dataset.is_bound)


class TestSupervisionToggles(MultiSourceTestCase):
    """Sources whose annotations are not used."""

    def build_sample(self, det3d: bool = True, seg3d: bool = True):
        """Build the first training sample of a source with the given supervision."""
        data_module = self.build_data_module(
            {"main": build_source(self.main_database, det3d=det3d, seg3d=seg3d)}
        )
        data_module.setup("fit")
        return data_module.datasets[SplitType.TRAIN][0]

    def test_a_supervised_source_keeps_its_annotations(self) -> None:
        """
        Input: a source supervising both tasks.
        Expected: the sample carries the boxes and the labels of the corpus.
        Check: the box set is not empty and some point carries a named class.
        """
        sample = self.build_sample()

        assert sample.detection3d_gt_bboxes_3d is not None
        assert sample.segmentation3d_gt_sample is not None
        self.assertGreater(len(sample.detection3d_gt_bboxes_3d), 0)
        self.assertTrue(bool((sample.segmentation3d_gt_sample.gt_semantic_mask >= 0).any()))

    def test_an_unsupervised_detection_source_carries_an_empty_box_set(self) -> None:
        """
        Input: a source whose boxes do not supervise the run.
        Expected: the field stays so the sample still collates with the supervised corpora,
        but it holds nothing to learn from.
        Check: the box set is empty and the labels are untouched.
        """
        sample = self.build_sample(det3d=False)

        assert sample.detection3d_gt_bboxes_3d is not None
        assert sample.segmentation3d_gt_sample is not None
        self.assertEqual(len(sample.detection3d_gt_bboxes_3d), 0)
        self.assertTrue(bool((sample.segmentation3d_gt_sample.gt_semantic_mask >= 0).any()))

    def test_an_unsupervised_segmentation_source_ignores_every_point(self) -> None:
        """
        Input: a source whose masks do not supervise the run, as a pseudo labelled corpus
        scored on its boxes alone.
        Expected: every point takes the ignore index, so the labels reach neither the loss nor
        the metrics while the sample still collates with the corpora that do carry them.
        Check: every label equals the ignore index and the boxes are untouched.
        """
        sample = self.build_sample(seg3d=False)

        assert sample.detection3d_gt_bboxes_3d is not None
        assert sample.segmentation3d_gt_sample is not None
        labels = sample.segmentation3d_gt_sample.gt_semantic_mask
        self.assertTrue(bool((labels == sample.segmentation3d_gt_sample.ignore_index).all()))
        self.assertGreater(len(sample.detection3d_gt_bboxes_3d), 0)

    def test_an_unsupervised_segmentation_source_needs_no_mask_on_disk(self) -> None:
        """
        Input: a corpus whose records name no semantic mask, declared without segmentation
        supervision, as a detection only pseudo corpus is.
        Expected: the labels of such a corpus are written rather than read, so the frames
        serve without a mask file and every point still takes the ignore index.
        Check: the sample carries one ignored label per point.
        """
        records = self.main_records.with_columns(
            pl.col(DatasetTableSchema.LIDAR_FRAMES.name).list.eval(
                pl.element().struct.with_fields(
                    lidar_pointcloud_semantic_mask_path=pl.lit(None, dtype=pl.String)
                )
            )
        )
        database = FakeDatabase(self.main_root, records, "maskless", build_taxonomy())
        data_module = self.build_data_module({"main": build_source(database, seg3d=False)})
        data_module.setup("fit")

        sample = data_module.datasets[SplitType.TRAIN][0]

        assert sample.segmentation3d_gt_sample is not None
        assert sample.point_cloud_data is not None
        labels = sample.segmentation3d_gt_sample.gt_semantic_mask
        self.assertEqual(labels.shape[0], len(sample.point_cloud_data))
        self.assertTrue(bool((labels == sample.segmentation3d_gt_sample.ignore_index).all()))

    def test_mixed_supervision_still_collates(self) -> None:
        """
        Input: one supervised sample and one whose masks are ignored, as a mixed supervision
        split serves them in one batch.
        Expected: both samples still collate, which is why disabled tasks keep their fields.
        Check: the collated batch carries both the boxes and the labels.
        """
        data_module = self.build_data_module(
            {
                "main": build_source(self.main_database, seg3d=False),
                "seg3d_gt": build_source(self.seg3d_gt_database),
            }
        )
        data_module.setup("fit")
        split = data_module.datasets[SplitType.TRAIN]

        batch = split.collate_fn([split[0], split[len(split) - 1]])

        assert batch.point_cloud_gt_batch is not None
        assert batch.segmentation3d_gt_batch is not None
        assert batch.detection3d_gt_batch is not None
        self.assertEqual(int(batch.point_cloud_gt_batch.batch_size), 2)


class TestFramesWithoutMasks(unittest.TestCase):
    """A supervised corpus must carry a semantic mask on every frame it serves."""

    def test_a_frame_without_a_mask_is_rejected(self) -> None:
        """
        Input: a segmentation supervised corpus whose second record names no mask.
        Expected: a supervised frame without labels is a broken corpus, not an empty mask.
        Check: the first frame serves its labels and the second one raises a ValueError.
        """
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        records = write_corpus(root, num_records=2, unmasked_records=frozenset({1}))
        database = FakeDatabase(root, records, "unmasked", build_taxonomy())
        dataset = build_dataset(database).bind(records)

        self.assertIsNotNone(dataset[0].segmentation3d_gt_sample)
        with self.assertRaisesRegex(ValueError, "carries no semantic mask"):
            dataset[1]


class TestFrameSampling(MultiSourceTestCase):
    """The training split drawn by repeat factor sampling."""

    def test_the_training_loader_draws_through_the_weighted_sampler(self) -> None:
        """
        Input: a training split with frame sampling settings.
        Expected: the training loader draws its order from the weighted sampler.
        Check: the sampler is the weighted one and draws one epoch of the split.
        """
        data_module = self.build_data_module(
            {"main": build_source(self.main_database)},
            train_dataloader=DataLoaderConfig(batch_size=1),
            train_frame_sampling=FRAME_SAMPLING,
        )
        data_module.setup("fit")

        loader = data_module.train_dataloader()

        self.assertIsInstance(loader.sampler, DistributedWeightedRandomSampler)
        self.assertEqual(len(list(loader.sampler)), len(self.main_records))

    def test_rejects_shuffling_next_to_the_weighted_sampler(self) -> None:
        """
        Input: a training loader asking to shuffle while frame sampling is set.
        Expected: the weighted sampler draws the order, so a shuffle would be ignored.
        Check: building the training loader raises a ValueError naming the shuffle.
        """
        data_module = self.build_data_module(
            {"main": build_source(self.main_database)},
            train_dataloader=DataLoaderConfig(batch_size=1, shuffle=True),
            train_frame_sampling=FRAME_SAMPLING,
        )
        data_module.setup("fit")

        with self.assertRaisesRegex(ValueError, "shuffle"):
            data_module.train_dataloader()


if __name__ == "__main__":
    unittest.main()
