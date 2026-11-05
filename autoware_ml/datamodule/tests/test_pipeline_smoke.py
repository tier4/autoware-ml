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

"""End to end smoke tests of the data pipeline, from the record table to a training step.

The tests build a small synthetic T4 corpus on disk, read it through the real dataset, the
real transform pipeline and the real runtime preprocessing, and run one optimizer step of
the PTv3 models on the result.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.datamodule.tests.corpus import (
    POINT_CLOUD_RANGE,
    SCENE,
    build_dataset,
    write_corpus,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_bev_neck,
    build_preprocessor,
    build_ptv3_encoder,
    build_seg3d_head,
    build_seg3d_model,
    build_trans_model,
    build_transfusion_head,
)
from autoware_ml.models.multi.ptv3_det3d_seg3d import PTv3Det3DSeg3DModel
from autoware_ml.models.segmentation3d.encoders.voxel import PastFutureVoxelFeatureEncoder
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


class TestPipelineSmoke(unittest.TestCase):
    """The dataset, the transforms and the preprocessing carry a batch into the models."""

    def setUp(self) -> None:
        """Write a two frame corpus and collate a batch from it."""
        torch.manual_seed(0)
        np.random.seed(0)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.records = write_corpus(self.root, num_records=2)
        self.dataset = build_dataset(self.root, self.records)
        self.batch = self.dataset.collate_fn([self.dataset[0], self.dataset[1]])

    def test_dataset_serves_a_collated_batch_of_every_task(self) -> None:
        """
        Input: a batch collated from the two frames of the synthetic corpus.
        Expected: the dataset serves the points, the boxes, the semantic masks and the frame
        metadata of both frames in one batch, and the frame metadata carry the scene and the
        sample time of every record, which place its frame in the stream of its scene.
        Check: every task batch is present and the batch size is two; read the scene tokens and
        the sample times.
        """
        assert isinstance(self.batch, ModelGTBatch)
        assert self.batch.point_cloud_gt_batch is not None
        assert self.batch.detection3d_gt_batch is not None
        assert self.batch.segmentation3d_gt_batch is not None
        assert self.batch.frame_meta_batch is not None
        self.assertEqual(self.batch.batch_size, 2)
        self.assertEqual(list(self.batch.frame_meta_batch.scene_tokens), [SCENE, SCENE])
        # The sample time of every record places its frame in the stream of its scene
        self.assertEqual(self.batch.frame_meta_batch.timestamps.tolist(), [0.0, 1.0])

    def test_every_point_keeps_its_semantic_label_through_the_pipeline(self) -> None:
        point_batch = self.batch.point_cloud_gt_batch
        segment_batch = self.batch.segmentation3d_gt_batch
        assert point_batch is not None and segment_batch is not None

        self.assertEqual(point_batch.points.shape[0], segment_batch.gt_semantic_masks.shape[0])
        self.assertEqual(point_batch.batch_indices.tolist(), segment_batch.batch_indices.tolist())

    def test_the_sweeps_carry_the_time_lag_and_the_ignore_label(self) -> None:
        point_batch = self.batch.point_cloud_gt_batch
        segment_batch = self.batch.segmentation3d_gt_batch
        assert point_batch is not None and segment_batch is not None

        time_lag = point_batch.points[:, point_batch.timestamp_difference_dim]
        self.assertEqual(point_batch.timestamp_difference_dim, 4)
        # A past sweep was captured before the sample and a future one after it, so the two
        # sides arrive with opposite signs and the current frame keeps the exact zero
        self.assertTrue(bool((time_lag > 0).any()))
        self.assertTrue(bool((time_lag < 0).any()))
        self.assertTrue(bool((time_lag == 0).any()))
        # The current frame is labelled and every sweep return takes the ignore index
        self.assertTrue(bool((segment_batch.gt_semantic_masks[time_lag != 0] == -1).all()))
        self.assertTrue(bool((segment_batch.gt_semantic_masks[time_lag == 0] >= 0).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "the PTv3 stem runs on CUDA only")
    def test_ptv3_segmentation3d_runs_a_training_step(self) -> None:
        device = torch.device("cuda")
        model = build_seg3d_model(POINT_CLOUD_RANGE, PastFutureVoxelFeatureEncoder).to(device)
        model.log_dict = lambda *args, **kwargs: None
        batch_inputs = DataPreprocessor(
            preprocessor_modules=[build_preprocessor(POINT_CLOUD_RANGE)]
        )(self.batch.to_device(device), is_training=True)

        loss = model.training_step(batch_inputs, batch_idx=0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "the PTv3 stem runs on CUDA only")
    def test_ptv3_detection3d_runs_a_training_step(self) -> None:
        device = torch.device("cuda")
        model = build_trans_model(
            point_cloud_range=POINT_CLOUD_RANGE, voxel_encoder_type=PastFutureVoxelFeatureEncoder
        ).to(device)
        model.log_dict = lambda *args, **kwargs: None
        batch_inputs = DataPreprocessor(
            preprocessor_modules=[build_preprocessor(POINT_CLOUD_RANGE)]
        )(self.batch.to_device(device), is_training=True)

        loss = model.training_step(batch_inputs, batch_idx=0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(), "the PTv3 stem runs on CUDA only")
    def test_ptv3_det3d_seg3d_runs_a_training_step(self) -> None:
        device = torch.device("cuda")
        model = PTv3Det3DSeg3DModel(
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            encoder=build_ptv3_encoder(PastFutureVoxelFeatureEncoder.out_channels),
            voxel_encoder=PastFutureVoxelFeatureEncoder(),
            seg3d_head=build_seg3d_head(),
            bev_neck=build_bev_neck(),
            bbox_head=build_transfusion_head(POINT_CLOUD_RANGE),
            export_output_names=["pred_labels", "pred_probs"],
            optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
            grid_size=1.0,
            point_cloud_range=list(POINT_CLOUD_RANGE),
        ).to(device)
        model.log_dict = lambda *args, **kwargs: None
        batch_inputs = DataPreprocessor(
            preprocessor_modules=[build_preprocessor(POINT_CLOUD_RANGE)]
        )(self.batch.to_device(device), is_training=True)

        loss = model.training_step(batch_inputs, batch_idx=0)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(), "the PTv3 stem runs on CUDA only")
    def test_evaluation_step_builds_the_metric_frames(self) -> None:
        model = build_seg3d_model(POINT_CLOUD_RANGE, PastFutureVoxelFeatureEncoder).to(
            torch.device("cuda")
        )
        device = next(model.parameters()).device
        model.log_dict = lambda *args, **kwargs: None
        batch_inputs = DataPreprocessor(
            preprocessor_modules=[build_preprocessor(POINT_CLOUD_RANGE)]
        )(self.batch.to_device(device), is_training=False)

        with torch.no_grad():
            outputs = model(batch_inputs)
            eval_output = model.build_eval_output(batch_inputs, outputs)

        frames = eval_output["seg3d_frames"]
        self.assertEqual(len(frames), 2)
        for frame in frames:
            self.assertEqual(frame["coord"].shape[0], frame["pred"].shape[0])
            self.assertEqual(frame["coord"].shape[0], frame["target"].shape[0])
            self.assertEqual(frame["scene_token"], SCENE)
