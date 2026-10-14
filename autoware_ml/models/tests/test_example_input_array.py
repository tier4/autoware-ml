"""Unit tests for the example input the base model exposes to the Lightning model summary."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import lightning as L
from lightning.pytorch.utilities.model_summary import summarize
import polars as pl
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch, ModelGTSample
from autoware_ml.dataclasses.geometry.point_clouds import PointCloudGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.datamodule.base_dataset import BaseDataset, ConcatDataset
from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.datamodule.sources import DatasetSource
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.types.dataset import SplitType
from autoware_ml.types.geometry import PointFeatureName


class _Dataset(BaseDataset):
    """Minimal new-style dataset whose samples hold a two-point cloud filled with the index."""

    def __init__(self, num_samples: int = 3) -> None:
        super().__init__(
            database_root_path="/",
            max_num_3d_gt_bboxes=0,
            dataset_records_dataframe=pl.DataFrame({"index": list(range(num_samples))}),
            transforms=None,
        )
        self.requested: list[int] = []

    def get_data_sample(self, index: int) -> ModelGTSample:
        self.requested.append(index)
        points = LiDARPoints(
            torch.full((2, 4), float(index)),
            [
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ],
            timestamp=0.0,
        )
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=points,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )


def _datamodule(train_dataset: BaseDataset | None) -> DataModule:
    """Build a DataModule serving ``train_dataset`` as its set-up training split.

    The source declares a test dataset so it is valid without a training one, and the
    training split is bound directly instead of through ``setup``, which needs a database.
    """
    database = SimpleNamespace(root_path="/", taxonomy="taxonomy", version="v1", database_hash="h")
    datamodule = DataModule(
        splitter=SimpleNamespace(),  # type: ignore[arg-type]
        sources={
            "main": DatasetSource(
                database=database,  # type: ignore[arg-type]
                train_dataset=train_dataset,
                test_dataset=_Dataset(),
            )
        },
        train_dataloader=None,
        validation_dataloader=None,
        test_dataloader=None,
        predict_dataloader=None,
        train_frame_sampling=None,
    )
    if train_dataset is not None:
        datamodule.datasets[SplitType.TRAIN] = ConcatDataset(datasets=[train_dataset], repeats=[1])
    return datamodule


class _Model(ModuleBaseModel):
    """Tiny model whose forward runs one matmul on the collated points."""

    def __init__(self, datamodule: L.LightningDataModule | None) -> None:
        super().__init__(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            log_dict_configs=LogDictConfigs(prog_bar=False),
        )
        self.linear = torch.nn.Linear(4, 8, bias=False)
        # Only the datamodule of the trainer is read by the hooks under test.
        self._trainer = SimpleNamespace(datamodule=datamodule)  # type: ignore[assignment]

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        point_cloud_gt_batch = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        assert isinstance(point_cloud_gt_batch, PointCloudGTBatch)
        self.linear(point_cloud_gt_batch.points)
        return ModelOutputs(detection3d_head_outputs=None)


class TestBuildExampleInputBatch(unittest.TestCase):
    """The method collating the first training sample."""

    def test_collates_only_the_first_sample(self) -> None:
        dataset = _Dataset()

        batch = _Model(_datamodule(dataset)).build_example_input_batch()

        self.assertIsInstance(batch, ModelGTBatch)
        assert batch is not None and batch.point_cloud_gt_batch is not None
        self.assertEqual(dataset.requested, [0])
        self.assertEqual(tuple(batch.point_cloud_gt_batch.points.shape), (2, 4))
        self.assertEqual(batch.point_cloud_gt_batch.batch_indices.tolist(), [0, 0])

    def test_none_without_a_usable_training_dataset(self) -> None:
        self.assertIsNone(_Model(None).build_example_input_batch())
        self.assertIsNone(_Model(L.LightningDataModule()).build_example_input_batch())
        self.assertIsNone(_Model(_datamodule(None)).build_example_input_batch())
        self.assertIsNone(_Model(_datamodule(_Dataset(0))).build_example_input_batch())


class TestModuleBaseModelExampleInput(unittest.TestCase):
    """The setup hook and the Lightning summary reading the example input."""

    def test_setup_fit_exposes_the_first_training_sample(self) -> None:
        model = _Model(_datamodule(_Dataset()))

        model.setup("fit")

        self.assertIsInstance(model.example_input_array, ModelGTBatch)

    def test_setup_other_stages_leave_the_example_input_alone(self) -> None:
        for stage in ("validate", "test", "predict"):
            model = _Model(_datamodule(_Dataset()))
            model.setup(stage)
            self.assertIsNone(model.example_input_array, stage)

    def test_setup_keeps_a_preset_example_input(self) -> None:
        model = _Model(_datamodule(_Dataset()))
        preset = _Model(_datamodule(_Dataset(1))).build_example_input_batch()
        model.example_input_array = preset

        model.setup("fit")

        self.assertIs(model.example_input_array, preset)

    def test_setup_without_a_training_sample_warns_and_skips(self) -> None:
        model = _Model(L.LightningDataModule())

        with self.assertLogs("autoware_ml.models.module_base_model", level="WARNING") as logs:
            model.setup("fit")

        self.assertIsNone(model.example_input_array)
        self.assertIn("FLOPs", logs.output[0])

    def test_model_summary_reports_flops_from_the_example_input(self) -> None:
        model = _Model(_datamodule(_Dataset()))
        model.setup("fit")
        model._trainer = None

        summary = summarize(model, max_depth=1)

        # One matmul of the (2, 4) points with the (4, 8) weight: 2 * 2 * 4 * 8 FLOPs.
        self.assertEqual(summary.total_flops, 2 * 2 * 4 * 8)

    def test_model_summary_without_example_input_reports_zero_flops(self) -> None:
        model = _Model(None)
        model._trainer = None

        self.assertEqual(summarize(model, max_depth=1).total_flops, 0)


if __name__ == "__main__":
    unittest.main()
