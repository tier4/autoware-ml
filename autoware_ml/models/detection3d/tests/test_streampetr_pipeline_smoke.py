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

from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import hydra
import numpy as np
import polars as pl
import torch

from autoware_ml.configs.tests.test_task_configs import compose_experiment
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.common.detection3d import Detection3DTask
from autoware_ml.datamodule.samplers import GroupStreamingSampler
from autoware_ml.datamodule.t4dataset.dataset import T4Dataset
from autoware_ml.datamodule.tests.corpus import SCENE, write_corpus
from autoware_ml.models.common.necks.cp_fpn import CPFPN
from autoware_ml.models.detection3d.heads.focal2d import FocalHead2D
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.streampetr import StreamPETRDetection3DModel
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.types.geometry import Box2DFieldIndex, Box3DFieldIndex

T4_STAGE = (
    "tasks/detection3d/streampetr/t4dataset/vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base"
)
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


def lidar_to_camera(yaw: float) -> np.ndarray:
    """Pose of a camera at the lidar origin looking along ``yaw``, as lidar-to-camera."""
    lidar2cam = np.eye(4)
    lidar2cam[:3, :3] = [
        [np.sin(yaw), -np.cos(yaw), 0.0],  # x right
        [0.0, 0.0, -1.0],  # y down
        [np.cos(yaw), np.sin(yaw), 0.0],  # z forward
    ]
    return lidar2cam


def write_cameras(root: Path, records: pl.DataFrame) -> pl.DataFrame:
    """Write one flat gray image per camera per record and attach the image frames."""
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
            camera_matrix = np.eye(4)
            camera_matrix[:3, :3] = intrinsics
            lidar2cam = lidar_to_camera(yaw)
            frames.append(
                {
                    "image_frame_id": f"image-{record_index}-{camera}",
                    "image_keyframe": True,
                    "image_sensor_id": camera,
                    "image_sensor_channel_name": camera,
                    "image_timestamp_seconds": timestamp,
                    "image_relative_path": relative,
                    "image_height": STORED_IMAGE_SIZE[0],
                    "image_width": STORED_IMAGE_SIZE[1],
                    "cam2img": intrinsics.tolist(),
                    "image_distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "image_distortion_model": "plumb_bob",
                    "image_sensor_to_ego_pose_matrix": np.eye(4).tolist(),
                    "image_frame_ego_pose_to_global_matrix": np.eye(4).tolist(),
                    "lidar2cam": lidar2cam.tolist(),
                    "lidar2img": (camera_matrix @ lidar2cam).tolist(),
                }
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
    """Move the box of every record ``distance`` meters ahead, in view of the front camera only."""
    boxes = records.get_column(DatasetTableSchema.BOXES_3D.name).to_list()
    for record_boxes in boxes:
        for box in record_boxes:
            box["box3d_params"][Box3DFieldIndex.X] = distance
    return records.with_columns(
        pl.Series(DatasetTableSchema.BOXES_3D.name, boxes, dtype=DatasetTableSchema.BOXES_3D.dtype)
    )


def move_to_scene(root: Path, records: pl.DataFrame, rows: list[int], scene: str) -> pl.DataFrame:
    """Store the lidar frames of the given records under another scene directory."""
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
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = torch.nn.Conv2d(3, 16, kernel_size=8, stride=8)
        self.stage2 = torch.nn.Conv2d(16, 32, kernel_size=2, stride=2)

    def forward(self, x):
        c3 = self.stage1(x)
        return c3, self.stage2(c3)


def build_model(cfg) -> StreamPETRDetection3DModel:
    """A small StreamPETR with the CPFPN neck and the auxiliary 2D head, on stride-8 tokens.

    The data preprocessor, the classes and the partial annotations are those of the T4 stage.
    """
    class_names = list(cfg.database.taxonomy.detection3d.class_names)
    return StreamPETRDetection3DModel(
        data_preprocessor=hydra.utils.instantiate(cfg.data_preprocessor),
        img_backbone=_TinyBackbone(),
        img_neck=CPFPN(in_channels=[16, 32], out_channels=32),
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
        img_roi_head=FocalHead2D(class_names=class_names, in_channels=32, embed_dims=32, stride=8),
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
        self.cfg = compose_experiment(T4_STAGE)
        self.cfg.image_size = IMAGE_SIZE
        self.dataset = T4Dataset(
            lidar_intensity_scale=255.0,
            database_root_path=str(root),
            max_num_3d_gt_bboxes=8,
            max_num_2d_gt_bboxes=8,
            dataset_records_dataframe=records,
            transforms=hydra.utils.instantiate(self.cfg.datasets.train.transforms),
            dataset_tasks={"detection3d": Detection3DTask},
            det3d_supervised=True,
            seg3d_supervised=True,
            camera_names=list(CAMERAS),
        )

    def assert_cameras_keep_their_data(self, sample) -> None:
        """Every camera slot holds the pixels, the focal length and the 2D boxes of its camera."""
        image_data = sample.camera_image_data
        boxes_2d = sample.detection2d_gt_sample.gt_bboxes_2d
        for slot, camera in enumerate(image_data.camera_names):
            _, focal, gray = CAMERAS[camera]
            # The median ignores the border a resize below the covering scale pads with zeros.
            self.assertAlmostEqual(
                float(image_data.images[slot].median()), gray / 255.0, delta=0.01
            )
            self.assertEqual(float(image_data.camera_intrinsics[slot][0, 0]), focal)
            # The box ahead is in view of the front camera only, inside its image.
            rows = boxes_2d[slot]
            self.assertEqual(len(rows) > 0, camera == "CAM_FRONT")
            self.assertTrue(bool((rows[:, Box2DFieldIndex.X_MIN] >= 0.0).all()))
            self.assertTrue(bool((rows[:, Box2DFieldIndex.Y_MIN] >= 0.0).all()))
            self.assertTrue(bool((rows[:, Box2DFieldIndex.X_MAX] <= IMAGE_SIZE[1]).all()))
            self.assertTrue(bool((rows[:, Box2DFieldIndex.Y_MAX] <= IMAGE_SIZE[0]).all()))

    def assert_normalized(self, images: torch.Tensor, loaded: torch.Tensor) -> None:
        """The preprocessed images are the loaded ones normalized, where no grid cell hides them."""
        mean = torch.tensor(list(self.cfg.img_norm_cfg.mean)).view(1, 1, 3, 1, 1)
        std = torch.tensor(list(self.cfg.img_norm_cfg.std)).view(1, 1, 3, 1, 1)
        visible = images != 0.0
        self.assertTrue(bool(visible.any()))
        torch.testing.assert_close(images[visible], ((loaded - mean) / std)[visible])

    def test_a_lane_streams_every_scene_in_time_order(self) -> None:
        """
        Input: the four records of two scenes, stored out of scene and time order, served
        through one lane without shuffling.
        Expected: the lane serves one scene after the other, in the order the scenes first
        appear in the records, and every scene in the order of its sample times.
        Check: the indices the sampler serves.
        """
        sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False, seed=0)

        self.assertEqual(list(sampler), [2, 0, 1, 3])

    def test_streamed_frames_train_the_detector_and_carry_its_memory(self) -> None:
        """
        Input: the four frames served through one lane, each loaded, augmented and collated by
        the training pipeline of the T4 stage (its camera shuffle and 2D projection included)
        and preprocessed by its data preprocessor, and one training step per frame of a small
        StreamPETR with the auxiliary 2D head and the partial annotations of the stage.
        Expected: every camera keeps its pixels, its calibration and its 2D boxes through the
        shuffle, which reorders the cameras of a frame at least once; the box ahead projects onto
        the front camera only, inside the image; the images reach the model normalized; every
        step trains both heads on the box of its frame, with the augmentation folded into the
        frame pose; the memory continues within a scenario and restarts at the next one, which
        follows in time, so only the scene tells them apart; the annotation status of every
        record reaches the batch; the weights of the 2D head get gradients.
        Check: the camera data of every slot, the reordering, the normalized pixels, the folded
        pose, the finite loss and the non-zero 3D and 2D regression and 2D classification losses
        of every step, the stream continuity recorded for every frame, the annotation statuses
        and the gradients of the 2D head.
        """
        model = build_model(self.cfg).train()
        logs = []
        model.log_dict = lambda logged, **kwargs: logs.append(
            {key: float(value) for key, value in logged.items()}
        )
        sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False, seed=0)
        continuity = []
        statuses = []
        reordered = False
        stream_continuity = model.bbox_head.stream_continuity

        def record_continuity(scene_tokens, timestamp):
            prev_exists = stream_continuity(scene_tokens, timestamp)
            continuity.append((list(scene_tokens), timestamp.tolist(), prev_exists.tolist()))
            return prev_exists

        model.bbox_head.stream_continuity = record_continuity

        for index in sampler:
            sample = self.dataset[index]
            self.assert_cameras_keep_their_data(sample)
            reordered |= list(sample.camera_image_data.camera_names) != list(CAMERAS)
            batch = self.dataset.collate_fn([sample])
            # The preprocessor runs where Lightning hands the batch to the model.
            batch_inputs = model.on_after_batch_transfer(batch, dataloader_idx=0)
            self.assert_normalized(batch_inputs.image_data.images, batch.image_gt_batch.images)
            gt_batch = batch_inputs.multi_task_gt_batch
            frame_meta = gt_batch.frame_meta_batch
            assert frame_meta is not None
            # The augmentation of the frame is folded into its pose for the memory warp.
            self.assertFalse(torch.equal(frame_meta.ego2globals[0], torch.eye(4)))
            assert gt_batch.detection3d_gt_batch is not None
            statuses.append(gt_batch.detection3d_gt_batch.gt_traffic_cone_barrier_bbox_status)

            loss = model.training_step(batch_inputs, batch_idx=0)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            # Both heads train on the box of the frame.
            self.assertGreater(logs[-1]["train/loss_bbox"], 0.0)
            self.assertGreater(logs[-1]["train/loss_bbox2d"], 0.0)
            self.assertGreater(logs[-1]["train/loss_cls2d"], 0.0)

        self.assertTrue(reordered)
        self.assertEqual(
            continuity,
            [
                ([SCENES["b"]], [20.0], [0.0]),
                ([SCENES["b"]], [20.5], [1.0]),
                ([SCENES["a"]], [21.0], [0.0]),
                ([SCENES["a"]], [21.5], [1.0]),
            ],
        )
        self.assertEqual(
            [status.tolist() for status in statuses], [[True], [True], [False], [False]]
        )
        gradients = [parameter.grad for parameter in model.img_roi_head.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_evaluation_decodes_the_streamed_frames(self) -> None:
        """
        Input: the four frames served through one lane into the small StreamPETR in evaluation
        mode.
        Expected: evaluation runs without the auxiliary 2D head and decodes the detections of
        every frame.
        Check: no auxiliary outputs, and one prediction per frame.
        """
        model = build_model(self.cfg).eval()
        sampler = GroupStreamingSampler(self.dataset, batch_size=1, shuffle=False, seed=0)
        for index in sampler:
            batch = self.dataset.collate_fn([self.dataset[index]])
            with torch.no_grad():
                batch_inputs = model.on_after_batch_transfer(batch, dataloader_idx=0)
                outputs = model(batch_inputs)
                predictions = model.decode_outputs(batch_inputs, outputs)
            self.assertIsNone(outputs.detection3d().streampetr_head().auxiliary_2d)
            self.assertEqual(len(predictions.detection3d_predictions), 1)
