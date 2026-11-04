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

"""Tests for the losses and the eval output of the SparseDrive model with its motion head."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch, FrameMetaSample
from autoware_ml.dataclasses.batch.motion import AgentTracksBatch, AgentTracksSample, MotionGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids
from autoware_ml.metrics.eval_output import Detection3DEvalOutput
from autoware_ml.metrics.tests.eval_output_fixtures import detection_frame, prediction
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import POINT_CLOUD_RANGE
from autoware_ml.models.sparsedrive.tests.sparse4d_fixtures import gt_batch, random_stage
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.sparsedrive.tests.motion_fixtures import build_motion_head
from autoware_ml.models.sparsedrive.tests.test_motion_export import build_model
from autoware_ml.models.sparsedrive.model import SparseDriveModel
from autoware_ml.models.sparsedrive.motion_head import motion_eval_frames
from autoware_ml.models.sparsedrive.tests.test_model import frame_inputs
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.types.motion import AgentRole

requires_cuda_spconv = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)


def _tracks(positions: torch.Tensor, track_ids: list[str]) -> AgentTracksSample:
    """Agent tracks of five steps whose every step is known, the current step first."""
    count = len(track_ids)
    return AgentTracksSample(
        positions=positions,
        headings=torch.zeros(count, 5),
        velocities=torch.zeros(count, 5, 2),
        valid=torch.ones(count, 5, dtype=torch.bool),
        agent_types=torch.zeros(count, dtype=torch.int64),
        agent_roles=torch.full((count,), AgentRole.SCORED, dtype=torch.int64),
        track_ids=track_ids,
        current_step_index=0,
        step_seconds=0.5,
        ignore_index=-1,
    )


def _frame_with_tracks(device: torch.device, timestamp: float) -> ModelBatchInputs:
    """One frame of the test scene with its boxes and the tracks of their agents."""
    batch = frame_inputs(device, timestamp)
    detections = batch.multi_task_gt_batch.detection3d_gt_batch
    batch_size, num_boxes = detections.gt_bboxes_3d.shape[:2]
    # Every box is an agent, its track id hashes to the instance id of the box
    track_ids = [str(slot) for slot in range(num_boxes)]
    instance_ids = torch.from_numpy(hash_instance_ids(track_ids)).to(device)
    agents = AgentTracksBatch.collate(
        [_tracks(torch.zeros(num_boxes, 5, 3), track_ids)] * batch_size
    ).to_device(device)
    return batch.replace(
        multi_task_gt_batch=batch.multi_task_gt_batch._replace(
            detection3d_gt_batch=detections._replace(
                gt_bboxes_instance_ids=instance_ids.expand(batch_size, -1).clone()
            ),
            motion_gt_batch=MotionGTBatch(agents=agents),
        )
    )


def _two_frame_step(
    model: SparseDriveModel,
) -> tuple[ModelBatchInputs, ModelOutputs, Detection3DEvalOutput]:
    """Two frames, two cars in the first and none in the second, with the head outputs on them.

    Returns:
        The batch, the outputs of the detection and the motion head, and the detection frames
        with the point counts and the metadata of their boxes.
    """
    cars = torch.tensor(
        [
            [4.0, 4.0, 0.0, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0],
            [-3.0, 2.0, 0.0, 4.5, 1.9, 1.7, 0.0, 0.0, 0.0],
        ]
    )
    track_ids = ["first", "second"]
    instance_ids = torch.from_numpy(hash_instance_ids(track_ids))
    labels = [torch.tensor([0, 1]), torch.zeros(0, dtype=torch.int64)]
    gt = gt_batch([cars, torch.zeros(0, 9)], labels, [instance_ids, instance_ids[:0]])
    positions = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    agents = AgentTracksBatch.collate(
        [_tracks(positions, track_ids), _tracks(torch.zeros(0, 5, 3), [])]
    )
    frame_meta = FrameMetaBatch.collate_gt_samples(
        [
            FrameMetaSample(torch.eye(4, dtype=torch.float64), f"db/scene/{lane}", 100.0, True)
            for lane in range(2)
        ],
        [None, None],
    )
    batch = ModelBatchInputs(
        multi_task_gt_batch=ModelGTBatch(
            point_cloud_gt_batch=None,
            detection3d_gt_batch=gt,
            segmentation3d_gt_batch=None,
            image_gt_batch=None,
            padding_mask=torch.zeros(2, dtype=torch.bool),
            frame_meta_batch=frame_meta,
            motion_gt_batch=MotionGTBatch(agents=agents),
        ),
        voxels_data=None,
        image_data=None,
        range_view_data=None,
    )
    stages = [
        random_stage(grid.grid_dims, 16, batch_size=2) for grid in model.bbox_head.level_grids
    ]
    det3d_outputs = model.bbox_head(
        stages,
        timestamps_seconds=frame_meta.timestamps,
        lidar2global=frame_meta.ego2globals,
        scene_tokens=frame_meta.scene_tokens,
        gt_detections=gt,
    )
    motion_outputs = model.motion_head(
        det3d_outputs.instance_feature,
        det3d_outputs.prediction[:, -1],
        det3d_outputs.classification[:, -1],
        model.bbox_head.anchor_encoder,
        model.bbox_head.instance_bank.lane_continues,
        None,
        frame_meta.ego2globals,
    )
    outputs = ModelOutputs(
        detection3d_head_outputs=Detection3DHeadOutputs(sparse4d_head_outputs=det3d_outputs),
        motion_head_outputs=motion_outputs,
    )
    empty = prediction(torch.zeros(0, 9), torch.zeros(0), torch.zeros(0, dtype=torch.int64))
    detection = Detection3DEvalOutput(
        frames=[
            detection_frame(empty, boxes, frame_labels).model_copy(
                update={
                    "gt_num_points": torch.full((len(boxes),), 5),
                    "frame_meta": frame_meta.sample(lane),
                }
            )
            for lane, (boxes, frame_labels) in enumerate(
                zip([cars, torch.zeros(0, 9)], labels, strict=True)
            )
        ]
    )
    return batch, outputs, detection


class TestSparseDriveMotion(unittest.TestCase):
    """The losses and the eval output of the SparseDrive model with its motion head."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_the_model_refuses_a_motion_head_of_other_classes(self) -> None:
        """
        Input: a motion head with anchors of three classes over a detection head of two.
        Expected: ValueError naming both class counts.
        """
        model = build_model(self.tmp_path)

        with self.assertRaisesRegex(
            ValueError, "anchors for 3 classes, the detection head predicts 2"
        ):
            SparseDriveModel(
                motion_head=build_motion_head(3),
                encoder=model.encoder,
                voxel_encoder=model.voxel_encoder,
                seg3d_head=model.seg3d_head,
                det_fusion=model.det_fusion,
                bbox_head=model.bbox_head,
                log_dict_configs=LogDictConfigs(prog_bar=False),
                data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
                grid_size=1.0,
                point_cloud_range=POINT_CLOUD_RANGE,
            )

    def test_the_motion_frames_add_the_query_center_and_pair_the_box_futures(self) -> None:
        """
        Input: two frames, two cars with known futures in the first and no box in the second.
        Expected: the trajectories of the most confident queries move from their box center to
          the frame, the futures follow the box slots, the frame without boxes is empty and
          a batch without detection frames is refused.
        """
        model = build_model(self.tmp_path).eval()
        with torch.no_grad():
            batch, outputs, detection = _two_frame_step(model)

            frames = motion_eval_frames(
                model.motion_head, model.bbox_head, batch, outputs, detection, 3
            )

        det3d_outputs = outputs.detection3d().sparse4d_head()
        scores = model.bbox_head.query_scores(det3d_outputs).max(dim=-1).values
        top = scores[0].topk(3).indices
        anchor = det3d_outputs.prediction[0, -1, top]
        expected = anchor[:, None, None, :2] + outputs.motion().agent_trajectories[0, top]
        torch.testing.assert_close(frames[0].pred_trajectories, expected)
        futures = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)[:, 1:, :2]
        torch.testing.assert_close(frames[0].gt_futures, futures)
        self.assertEqual(
            (frames[1].predictions.bboxes_3d.shape[0], frames[1].gt_boxes.shape[0]), (0, 0)
        )
        with self.assertRaisesRegex(ValueError, "boxes of the detection frames"):
            motion_eval_frames(model.motion_head, model.bbox_head, batch, outputs, None, 3)

    @requires_cuda_spconv
    def test_the_motion_model_trains_on_a_stream_and_reaches_every_parameter(self) -> None:
        """
        Input: two frames of one scene with the tracks of their boxes.
        Expected: a finite loss with the motion terms whose gradient reaches every parameter at
          every frame.
        """
        device = torch.device("cuda")
        model = build_model(self.tmp_path).to(device)

        for frame_index in range(2):
            model.zero_grad()
            batch = _frame_with_tracks(device, 100.0 + 0.1 * frame_index)
            metrics = model.compute_metrics(batch, model(batch))
            metrics["loss"].backward()

            self.assertTrue(torch.isfinite(metrics["loss"]))
            self.assertIn("weighted_motion_loss", metrics)
            missing = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            self.assertEqual(missing, [])
