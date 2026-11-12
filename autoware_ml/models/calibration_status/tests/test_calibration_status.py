"""Tests of the calibration status classifier's export specification."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.models.calibration_status.calibration_status import (
    CalibrationStatusClassifier,
    _CalibrationStatusExportModule,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.tests.batch_inputs_fixtures import build_batch_inputs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

BATCH_SIZE = 2
NUM_CAMERAS = 3
NUM_IMAGE_CHANNELS = 3
NUM_DEPTH_CHANNELS = 2
HEIGHT = 4
WIDTH = 6
NUM_CLASSES = 2


class _Head(nn.Module):
    """A classifier over the pooled features with the ``predict`` the model calls."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(NUM_IMAGE_CHANNELS + NUM_DEPTH_CHANNELS, NUM_CLASSES)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.classifier(feats.mean(dim=(2, 3)))

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.softmax(dim=1)


def _image_batch() -> ImageGTBatch:
    return ImageGTBatch(
        images=torch.rand(BATCH_SIZE, NUM_CAMERAS, NUM_IMAGE_CHANNELS, HEIGHT, WIDTH),
        depth_maps=torch.rand(BATCH_SIZE, NUM_CAMERAS, NUM_DEPTH_CHANNELS, HEIGHT, WIDTH),
        camera_intrinsics=torch.eye(3).expand(BATCH_SIZE, NUM_CAMERAS, 3, 3),
        image_augmentation_matrices=torch.eye(4).expand(BATCH_SIZE, NUM_CAMERAS, 4, 4),
        lidar2images=torch.eye(4).expand(BATCH_SIZE, NUM_CAMERAS, 4, 4),
        lidar2cams=torch.eye(4).expand(BATCH_SIZE, NUM_CAMERAS, 4, 4),
        calibration_statuses=torch.zeros(BATCH_SIZE, NUM_CAMERAS, dtype=torch.long),
    )


class TestCalibrationStatusExportSpec(unittest.TestCase):
    def test_export_spec_declares_the_node_names_and_the_image_axes(self) -> None:
        """
        Input: a batch of two samples with three cameras, images and depth maps.
        Expected: the spec names the tensors the node binds, ``input`` and ``output``, declares
            the image height and width dynamic, and traces one fused image per camera.
        Check: the export spec of ``build_export_specs``.
        """
        model = CalibrationStatusClassifier(
            backbone=nn.Identity(),
            neck=nn.Identity(),
            head=_Head(),
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        )

        spec = model.build_export_specs(build_batch_inputs(images=_image_batch()))["end_to_end"]

        self.assertIsInstance(spec.module, _CalibrationStatusExportModule)
        self.assertEqual(spec.input_names, ["input"])
        self.assertEqual(spec.output_names, ["output"])
        self.assertEqual(spec.dynamic_axes, {"input": {2: "height", 3: "width"}})
        (fused_images,) = spec.args
        self.assertEqual(
            tuple(fused_images.shape),
            (BATCH_SIZE * NUM_CAMERAS, NUM_IMAGE_CHANNELS + NUM_DEPTH_CHANNELS, HEIGHT, WIDTH),
        )
        self.assertEqual(
            tuple(spec.module(fused_images).shape), (BATCH_SIZE * NUM_CAMERAS, NUM_CLASSES)
        )
