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

"""End to end smoke test of StreamPETR, from the record table to streamed training steps."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import hydra
from jaxtyping import Float, Float64
import numpy as np
from omegaconf import DictConfig
import polars as pl
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.common.detection3d import Detection3DTask
from autoware_ml.datamodule.samplers import GroupStreamingSampler
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.tests.corpus import SCENE, image_frame, write_corpus
from autoware_ml.models.common.necks.fpn import FinestLevelRefinedFPN
from autoware_ml.models.detection2d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.streampetr import StreamPETRDetection3DModel
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.task_modules.streaming import StreamContinuity
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.types.geometry import Box2DFieldIndex, Box3DFieldIndex
from autoware_ml.utils.config.tests.config_roots import compose_task

# The T4Dataset leaf of the test config root, the shape of the T4 stages.
T4_STAGE = "tasks/detection3d/streampetr/t4dataset/vov99_flrfpn_480x640_b16_35e_t4dataset_51m"
POINT_CLOUD_RANGE = [-10.0, -10.0, -5.0, 10.0, 10.0, 5.0]
# [height, width] of the network input the T4 pipeline is shrunk to, and of the stored images.
IMAGE_SIZE = [32, 48]
STORED_IMAGE_SIZE = (48, 64)
# Every camera has its own viewing direction (yaw about the lidar z axis), focal length and
# gray level, so a shuffle that separated the pixels of a camera from its calibration shows.
CAMERAS = {
    "CAM_FRONT": (0.0, 40.0, 64),
    "CAM_BACK": (np.pi, 30.0, 192),
}
# The two scenarios are stored in two scene directories, which give the scene of every frame.
SCENES = {"b": SCENE, "a": SCENE.replace("scene_0", "scene_1")}


def lidar_to_camera(yaw: float) -> Float64[np.ndarray, "4 4"]:
    """Return the pose of a camera at the lidar origin looking along ``yaw``.

    Args:
        yaw: Viewing direction of the camera about the lidar z axis, in radians.

    Returns:
        The lidar-to-camera transform.
    """
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = [
        [np.sin(yaw), -np.cos(yaw), 0.0],  # x right
        [0.0, 0.0, -1.0],  # y down
        [np.cos(yaw), np.sin(yaw), 0.0],  # z forward
    ]
    return lidar2cam


def write_cameras(root: Path, records: pl.DataFrame) -> pl.DataFrame:
    """Write one flat gray image per camera per record and attach the image frames.

    Args:
        root: Database root the relative paths resolve against.
        records: Record table of the corpus.

    Returns:
        The record table with the image frames of every camera.
    """
    timestamps = records.get_column(DatasetTableSchema.TIMESTAMP_SECONDS.name).to_list()
    image_frames = []
    for record_index, timestamp in enumerate(timestamps):
        frames = []
        for camera, (yaw, focal, gray) in CAMERAS.items():
            relative = f"{SCENE}/data/{camera}/{record_index}.jpg"
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / relative), np.full((*STORED_IMAGE_SIZE, 3), gray, np.uint8))
            intrinsics = np.array(
                [
                    [focal, 0.0, STORED_IMAGE_SIZE[1] / 2.0],
                    [0.0, focal, STORED_IMAGE_SIZE[0] / 2.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            frames.append(
                image_frame(
                    record_index,
                    camera,
                    relative,
                    timestamp,
                    STORED_IMAGE_SIZE,
                    intrinsics,
                    lidar_to_camera(yaw),
                )
            )
        image_frames.append([frames])
    return records.with_columns(
        pl.Series(
            DatasetTableSchema.IMAGE_FRAMES.name,
            image_frames,
            dtype=DatasetTableSchema.IMAGE_FRAMES.dtype,
        )
    )


def place_boxes_ahead(records: pl.DataFrame, distance: float) -> pl.DataFrame:
    """Move the box of every record ahead, in view of the front camera only.

    Args:
        records: Record table of the corpus.
        distance: Distance of the boxes along the lidar x axis, in meters.

    Returns:
        The record table with the moved boxes.
    """
    boxes = records.get_column(DatasetTableSchema.BOXES_3D.name).to_list()
    for record_boxes in boxes:
        for box in record_boxes:
            box["box3d_params"][Box3DFieldIndex.X] = distance
    return records.with_columns(
        pl.Series(
            DatasetTableSchema.BOXES_3D.name,
            boxes,
            dtype=DatasetTableSchema.BOXES_3D.dtype,
        )
    )


def move_to_scene(
    root: Path, records: pl.DataFrame, rows: Sequence[int], scene: str
) -> pl.DataFrame:
    """Store the lidar frames of the given records under another scene directory.

    Args:
        root: Database root the relative paths resolve against.
        records: Record table of the corpus.
        rows: Rows of the records to move.
        scene: Scene directory the records move to.

    Returns:
        The record table with the moved lidar frames.
    """
    lidar_frames = records.get_column(DatasetTableSchema.LIDAR_FRAMES.name).to_list()
    for row in rows:
        for frame in lidar_frames[row]:
            for key in (
                "lidar_pointcloud_relative_path",
                "lidar_pointcloud_semantic_mask_relative_path",
            ):
                if frame[key] is None:
                    continue
                moved = frame[key].replace(SCENE, scene, 1)
                (root / moved).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(root / frame[key], root / moved)
                frame[key] = moved
    return records.with_columns(
        pl.Series(
            DatasetTableSchema.LIDAR_FRAMES.name,
            lidar_frames,
            dtype=DatasetTableSchema.LIDAR_FRAMES.dtype,
        )
    )


class _TinyBackbone(torch.nn.Module):
    """Two strided convolutions standing in for the two VoVNet stages the neck reads."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = torch.nn.Conv2d(3, 16, kernel_size=8, stride=8)
        self.stage2 = torch.nn.Conv2d(16, 32, kernel_size=2, stride=2)

    def forward(
        self, x: Float[torch.Tensor, "num_images 3 height width"]
    ) -> tuple[
        Float[torch.Tensor, "num_images 16 stride8_height stride8_width"],
        Float[torch.Tensor, "num_images 32 stride16_height stride16_width"],
    ]:
        """Return the stride-8 and stride-16 features of the images.

        Args:
            x: The normalized camera images.

        Returns:
            The features at stride 8 and at stride 16.
        """
        c3 = self.stage1(x)
        return c3, self.stage2(c3)


class _RecordedStreamContinuity(StreamContinuity):
    """The stream continuity of the head, keeping the scene, time and continuity of every call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[list[str], list[float], list[float]]] = []

    def __call__(
        self, scene_tokens: Sequence[str], timestamp: Float[torch.Tensor, " batch_size"]
    ) -> Float[torch.Tensor, " batch_size"]:
        """Return the continuity of every lane and keep it with the scene and time of the lane.

        Args:
            scene_tokens: Scene of the frame of every lane.
            timestamp: Sample time of the frame of every lane in seconds.

        Returns:
            ``prev_exists`` of every lane, as StreamContinuity computes it.
        """
        prev_exists = super().__call__(scene_tokens, timestamp)
        self.calls.append((list(scene_tokens), timestamp.tolist(), prev_exists.tolist()))
        return prev_exists


def build_model(cfg: DictConfig) -> StreamPETRDetection3DModel:
    """Build a small StreamPETR on the stride-8 tokens of the FinestLevelRefinedFPN neck.

    Args:
        cfg: Composed config of the T4 stage, whose data preprocessor, classes and partial
            annotations the model takes.

    Returns:
        The model with the auxiliary 2D head.
    """
    class_names = list(cfg.taxonomy.detection3d.class_names)
    return StreamPETRDetection3DModel(
        data_preprocessor=hydra.utils.instantiate(cfg.data_preprocessor),
        img_backbone=_TinyBackbone(),
        img_neck=FinestLevelRefinedFPN(in_channels=[16, 32], out_channels=32),
        bbox_head=StreamPETRHead(
            class_names=class_names,
            in_channels=32,
            hidden_dim=32,
            num_queries=8,
            num_decoder_layers=2,
            num_heads=4,
            feedforward_channels=64,
            memory_len=8,
            topk_proposals=4,
            num_propagated=4,
            depth_num=8,
            position_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
            scalar=2,
            noise_scale=0.5,
            split=0.5,
            bbox_coder=NMSFreeBBoxCoder3D(pc_range=POINT_CLOUD_RANGE, max_num=8),
            assigner=HungarianAssigner3D(
                cls_cost=ClassificationCost(weight=2.0),
                reg_cost=BBox3DL1Cost(weight=0.25),
                iou_cost=IoU3DCost(weight=0.0),
                point_cloud_range=POINT_CLOUD_RANGE,
            ),
            point_cloud_range=POINT_CLOUD_RANGE,
            code_weights=[2.0, 2.0] + [1.0] * 8,
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        auxiliary_2d_head=FocalHead2D(
            class_names=class_names, in_channels=32, embed_dims=32, stride=8
        ),
        partial_annotations=hydra.utils.instantiate(cfg.model.partial_annotations),
    )


class TestStreamPETRPipelineSmoke(unittest.TestCase):
    """The ported data path carries two streamed scenes through StreamPETR training steps."""

    def setUp(self) -> None:
        """Write two scenes of two frames, stored out of scene and time order."""
        torch.manual_seed(0)
        np.random.seed(0)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        records = write_corpus(root, num_records=4).with_columns(
            pl.Series(DatasetTableSchema.SCENARIO_ID.name, ["b", "a", "b", "a"]),
            # Scenario a follows scenario b in time, so only the scene tells the streams apart.
            pl.Series(DatasetTableSchema.TIMESTAMP_SECONDS.name, [20.5, 21.0, 20.0, 21.5]),
            # Scenario a was annotated without the partially annotated classes.
            pl.Series(
                DatasetTableSchema.TRAFFIC_CONE_BARRIER_BBOX_STATUS.name,
                [True, False, True, False],
            ),
        )
        records = write_cameras(root, place_boxes_ahead(records, distance=6.0))
        records = move_to_scene(root, records, rows=[1, 3], scene=SCENES["a"])
        # The training pipeline and the preprocessing of the T4 stage, shrunk to small images.
        self.cfg = compose_task(T4_STAGE)
        taxonomy = hydra.utils.instantiate(self.cfg.taxonomy.detection3d)
        self.cfg.image_size = IMAGE_SIZE
        self.dataset = T4Dataset(
            lidar_intensity_scale=255.0,
            database_root_path=str(root),
            max_num_3d_gt_bboxes=8,
            max_num_2d_gt_bboxes=8,
            dataset_records_dataframe=records,
            transforms=hydra.utils.instantiate(self.cfg.datasets.train.transforms),
            dataset_tasks={"detection3d": partial(Detection3DTask, taxonomy=taxonomy)},
            det3d_supervised=True,
            seg3d_supervised=True,
            camera_names=list(CAMERAS),
        )
        self.sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False, seed=0)

    def assert_cameras_keep_their_data(self, sample: ModelGTSample) -> None:
        """Check that every camera slot holds the pixels, focal length and 2D boxes of its camera.

        Args:
            sample: A sample of the training pipeline.
        """
        image_data = sample.camera_image_data
        boxes_2d = sample.detection2d_gt_sample.gt_bboxes_2d
        for slot, camera in enumerate(image_data.camera_names):
            _, focal, gray = CAMERAS[camera]
            # The median ignores the border a resize below the covering scale pads with zeros.
            self.assertAlmostEqual(
                float(image_data.images[slot].median()), gray / 255.0, delta=0.01
            )
            self.assertEqual(float(image_data.camera_intrinsics[slot][0, 0]), focal)
            rows = boxes_2d[slot]
            self.assertEqual(len(rows) > 0, camera == "CAM_FRONT")
            self.assertTrue(bool((rows[:, Box2DFieldIndex.X_MIN] >= 0.0).all()))
            self.assertTrue(bool((rows[:, Box2DFieldIndex.Y_MIN] >= 0.0).all()))
            self.assertTrue(bool((rows[:, Box2DFieldIndex.X_MAX] <= IMAGE_SIZE[1]).all()))
            self.assertTrue(bool((rows[:, Box2DFieldIndex.Y_MAX] <= IMAGE_SIZE[0]).all()))

    def assert_normalized(
        self,
        images: Float[torch.Tensor, "batch_size num_cams 3 height width"],
        loaded: Float[torch.Tensor, "batch_size num_cams 3 height width"],
    ) -> None:
        """Check that the preprocessed images are the loaded ones normalized.

        Args:
            images: The images the preprocessor hands to the model, zero where a grid cell
                hides them.
            loaded: The unit-range images the pipeline loaded.
        """
        normalize = self.cfg.data_preprocessor.preprocessor_modules[0]
        mean = torch.tensor(list(normalize.mean)).view(1, 1, 3, 1, 1)
        std = torch.tensor(list(normalize.std)).view(1, 1, 3, 1, 1)
        visible = images != 0.0
        self.assertTrue(bool(visible.any()))
        torch.testing.assert_close(images[visible], ((loaded - mean) / std)[visible])

    def test_a_lane_streams_every_scene_in_time_order(self) -> None:
        """Test that one lane serves the scenes one after the other, each in time order.

        Input: the four records of two scenes, stored out of scene and time order, served
        through one lane without shuffling.
        Expected: the lane serves one scene after the other, in the order the scenes first
        appear in the records, and every scene in the order of its sample times.
        Check: the indices the sampler serves.
        """
        self.assertEqual(list(self.sampler), [2, 0, 1, 3])

    def test_the_camera_shuffle_keeps_every_camera_with_its_data(self) -> None:
        """Test that the shuffled cameras keep their pixels, calibration and 2D boxes.

        Input: the four frames served through one lane, each loaded and augmented by the
        training pipeline of the T4 stage, its camera shuffle and 2D projection included.
        Expected: the shuffle reorders the cameras of a frame at least once, and every camera
        keeps its pixels, its calibration and its 2D boxes; the box ahead projects onto the
        front camera only, inside the image.
        Check: the gray level, the focal length and the 2D boxes of every camera slot, and
        the camera order of every frame.
        """
        orders = []
        for index in self.sampler:
            sample = self.dataset[index]
            self.assert_cameras_keep_their_data(sample)
            orders.append(list(sample.camera_image_data.camera_names))

        self.assertTrue(any(order != list(CAMERAS) for order in orders))

    def test_streamed_steps_train_both_heads_on_normalized_images(self) -> None:
        """Test that every streamed step trains both heads on the normalized images of its frame.

        Input: the four frames served through one lane, collated and preprocessed by the data
        preprocessor of the T4 stage, and one training step per frame of a small StreamPETR
        with the auxiliary 2D head and the partial annotations of the stage.
        Expected: the images reach the model normalized, the augmentation is folded into the
        frame pose, every step trains both heads on the box of its frame, and the weights of
        the 2D head get gradients.
        Check: the normalized pixels, the frame pose, the finite total loss and the non-zero
        3D and 2D regression and 2D classification losses of every step, and the gradients of
        the 2D head.
        """
        model = build_model(self.cfg).train()
        for index in self.sampler:
            batch = self.dataset.collate_fn([self.dataset[index]])
            # The preprocessor runs where Lightning hands the batch to the model.
            batch_inputs = model.on_after_batch_transfer(batch, dataloader_idx=0)
            self.assert_normalized(batch_inputs.image_data.images, batch.image_gt_batch.images)
            frame_meta = batch_inputs.multi_task_gt_batch.frame_meta_batch
            self.assertIsNotNone(frame_meta)
            self.assertFalse(torch.equal(frame_meta.ego2globals[0], torch.eye(4)))

            losses = model.compute_metrics(batch_inputs, model(batch_inputs))
            losses["loss"].backward()
            self.assertTrue(torch.isfinite(losses["loss"]))
            for key in ("loss_bbox", "loss_bbox2d", "loss_cls2d"):
                self.assertGreater(float(losses[key]), 0.0, key)

        gradients = [parameter.grad for parameter in model.auxiliary_2d_head.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_every_frame_brings_its_scene_and_annotation_status_to_the_model(self) -> None:
        """Test that the memory follows the scenes and the annotation status reaches the batch.

        Input: the four frames of two scenes served through one lane into a small StreamPETR
        in training mode, the second scene later in time.
        Expected: the memory continues within a scene and restarts at the next one, which
        only its scene tells apart from the first; the annotation status of every record
        reaches the batch.
        Check: the scene, the sample time and the continuity of every frame, and the
        annotation statuses of the batches.
        """
        model = build_model(self.cfg).train()
        continuity = _RecordedStreamContinuity()
        model.bbox_head.stream_continuity = continuity
        statuses = []
        for index in self.sampler:
            batch = self.dataset.collate_fn([self.dataset[index]])
            batch_inputs = model.on_after_batch_transfer(batch, dataloader_idx=0)
            detection3d = batch_inputs.multi_task_gt_batch.detection3d_gt_batch
            self.assertIsNotNone(detection3d)
            statuses.append(detection3d.gt_traffic_cone_barrier_bbox_status.tolist())
            with torch.no_grad():
                model(batch_inputs)

        self.assertEqual(
            continuity.calls,
            [
                ([SCENES["b"]], [20.0], [0.0]),
                ([SCENES["b"]], [20.5], [1.0]),
                ([SCENES["a"]], [21.0], [0.0]),
                ([SCENES["a"]], [21.5], [1.0]),
            ],
        )
        self.assertEqual(statuses, [[True], [True], [False], [False]])

    def test_evaluation_decodes_the_streamed_frames(self) -> None:
        """Test that evaluation decodes every streamed frame without the auxiliary 2D head.

        Input: the four frames served through one lane into the small StreamPETR in evaluation
        mode.
        Expected: evaluation runs without the auxiliary 2D head and decodes the detections of
        every frame.
        Check: no auxiliary outputs, and one prediction per frame.
        """
        model = build_model(self.cfg).eval()
        for index in self.sampler:
            batch = self.dataset.collate_fn([self.dataset[index]])
            with torch.no_grad():
                batch_inputs = model.on_after_batch_transfer(batch, dataloader_idx=0)
                outputs = model(batch_inputs)
                predictions = model.decode_outputs(batch_inputs, outputs)
            self.assertIsNone(outputs.detection3d().streampetr_head().auxiliary_2d)
            self.assertEqual(len(predictions.detection3d_predictions), 1)
