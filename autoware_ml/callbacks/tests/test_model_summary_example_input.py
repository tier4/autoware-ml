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

"""Unit tests for the ModelSummaryExampleInput callback: the collated first training sample,
the setup hook, the misconfiguration errors, and the FLOPs the Lightning summary reports."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import lightning as L
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_summary import summarize
import polars as pl
import torch

from autoware_ml.callbacks.model_summary_example_input import ModelSummaryExampleInput
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
    """Minimal dataset whose samples hold a two-point cloud filled with the index."""

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


def _trainer(datamodule: L.LightningDataModule | None) -> SimpleNamespace:
    """Build a trainer stand-in carrying only the datamodule the callback reads."""
    return SimpleNamespace(datamodule=datamodule)


class _Model(ModuleBaseModel):
    """Tiny model whose forward runs one matmul on the collated points."""

    def __init__(self) -> None:
        super().__init__(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            log_dict_configs=LogDictConfigs(prog_bar=False),
        )
        self.linear = torch.nn.Linear(4, 8, bias=False)

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        point_cloud_gt_batch = multi_task_batch_inputs.multi_task_gt_batch.point_cloud_gt_batch
        assert isinstance(point_cloud_gt_batch, PointCloudGTBatch)
        self.linear(point_cloud_gt_batch.points)
        return ModelOutputs(detection3d_head_outputs=None)


class TestBuildExampleInputBatch(unittest.TestCase):
    """The method collating the first training sample."""

    def test_collates_only_the_first_sample(self) -> None:
        """
        Input: a trainer whose datamodule serves a three-sample training split.
        Expected: a single-sample batch holding sample 0 and no other sample loaded.
        Check: the requested indices and the collated points and batch indices.
        """
        dataset = _Dataset()

        batch = ModelSummaryExampleInput.build_example_input_batch(_trainer(_datamodule(dataset)))

        self.assertIsInstance(batch, ModelGTBatch)
        assert batch.point_cloud_gt_batch is not None
        self.assertEqual(dataset.requested, [0])
        self.assertEqual(tuple(batch.point_cloud_gt_batch.points.shape), (2, 4))
        self.assertEqual(batch.point_cloud_gt_batch.batch_indices.tolist(), [0, 0])

    def test_rejects_a_trainer_without_a_datamodule(self) -> None:
        """
        Input: a trainer with no datamodule.
        Expected: a misconfiguration error naming the missing DataModule.
        Check: the exception type and message.
        """
        with self.assertRaisesRegex(MisconfigurationException, "needs a DataModule"):
            ModelSummaryExampleInput.build_example_input_batch(_trainer(None))

    def test_rejects_a_foreign_datamodule(self) -> None:
        """
        Input: a trainer with a plain Lightning datamodule.
        Expected: a misconfiguration error naming the missing DataModule.
        Check: the exception type and message.
        """
        with self.assertRaisesRegex(MisconfigurationException, "needs a DataModule"):
            ModelSummaryExampleInput.build_example_input_batch(_trainer(L.LightningDataModule()))

    def test_rejects_a_datamodule_without_a_training_split(self) -> None:
        """
        Input: a DataModule whose sources declare no training dataset.
        Expected: a misconfiguration error naming the missing training split.
        Check: the exception type and message.
        """
        with self.assertRaisesRegex(MisconfigurationException, "serves none"):
            ModelSummaryExampleInput.build_example_input_batch(_trainer(_datamodule(None)))

    def test_rejects_an_empty_training_split(self) -> None:
        """
        Input: a DataModule whose training split has zero samples.
        Expected: a misconfiguration error naming the empty training split.
        Check: the exception type and message.
        """
        with self.assertRaisesRegex(MisconfigurationException, "is empty"):
            ModelSummaryExampleInput.build_example_input_batch(_trainer(_datamodule(_Dataset(0))))


class TestSetup(unittest.TestCase):
    """The setup hook setting the example input on the module."""

    def setUp(self) -> None:
        self.callback = ModelSummaryExampleInput()
        self.model = _Model()
        self.trainer = _trainer(_datamodule(_Dataset()))

    def test_fit_exposes_the_first_training_sample(self) -> None:
        """
        Input: the fit stage with a training split available.
        Expected: the module carries the collated first sample as its example input.
        Check: the type of the example input.
        """
        self.callback.setup(self.trainer, self.model, "fit")  # type: ignore[arg-type]

        self.assertIsInstance(self.model.example_input_array, ModelGTBatch)

    def test_other_stages_leave_the_example_input_alone(self) -> None:
        """
        Input: the validate, test and predict stages.
        Expected: the module keeps no example input.
        Check: the example input stays None for every stage.
        """
        for stage in ("validate", "test", "predict"):
            self.callback.setup(self.trainer, self.model, stage)  # type: ignore[arg-type]
            self.assertIsNone(self.model.example_input_array, stage)

    def test_keeps_a_preset_example_input(self) -> None:
        """
        Input: a module that already carries an example input.
        Expected: the preset input is kept and the training split is not read.
        Check: identity of the example input and the dataset's requested indices.
        """
        dataset = _Dataset()
        self.trainer = _trainer(_datamodule(dataset))
        preset = ModelSummaryExampleInput.build_example_input_batch(
            _trainer(_datamodule(_Dataset(1)))
        )
        self.model.example_input_array = preset

        self.callback.setup(self.trainer, self.model, "fit")  # type: ignore[arg-type]

        self.assertIs(self.model.example_input_array, preset)
        self.assertEqual(dataset.requested, [])

    def test_fit_without_a_training_sample_raises(self) -> None:
        """
        Input: the fit stage with a plain Lightning datamodule on the trainer.
        Expected: setup fails instead of skipping the example input.
        Check: the exception type and that no example input is set.
        """
        with self.assertRaises(MisconfigurationException):
            self.callback.setup(_trainer(L.LightningDataModule()), self.model, "fit")  # type: ignore[arg-type]

        self.assertIsNone(self.model.example_input_array)


class TestModelSummary(unittest.TestCase):
    """The Lightning summary reading the example input the callback sets."""

    def test_reports_flops_from_the_example_input(self) -> None:
        """
        Input: a module set up by the callback for fit.
        Expected: the summary counts the forward pass of the single-sample batch.
        Check: one matmul of the (2, 4) points with the (4, 8) weight, 2 * 2 * 4 * 8 FLOPs.
        """
        model = _Model()
        ModelSummaryExampleInput().setup(_trainer(_datamodule(_Dataset())), model, "fit")  # type: ignore[arg-type]

        summary = summarize(model, max_depth=1)

        self.assertEqual(summary.total_flops, 128)

    def test_without_example_input_reports_zero_flops(self) -> None:
        """
        Input: a module without an example input.
        Expected: the summary reports no FLOPs.
        Check: the total FLOPs.
        """
        self.assertEqual(summarize(_Model(), max_depth=1).total_flops, 0)


if __name__ == "__main__":
    unittest.main()
