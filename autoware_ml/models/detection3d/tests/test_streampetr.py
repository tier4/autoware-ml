"""Unit tests for the native StreamPETR detector."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import onnxruntime
import pytest
import torch
import torch.nn as nn

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.metrics.base import EvalStage, MetricSuite
from autoware_ml.models.common.necks.lss_fpn import GeneralizedLSSFPN
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.streampetr import StreamPETRDetection3DModel, _ground_truth
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.streaming import StreamContinuity
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_detection3d_gt_batch,
)
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import ExportSpec, export_to_onnx

NUM_CAMS = 6
IMAGE_HW = (96, 160)


class _TinyMultiScaleBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = torch.nn.Conv2d(3, 128, kernel_size=8, stride=8)
        self.stage2 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.stage3 = torch.nn.Conv2d(256, 512, kernel_size=2, stride=2)

    def forward(self, x):
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return c3, c4, c5


class _ResetCountingSuite(MetricSuite[None]):
    """A stateless suite that only counts how often it is reset."""

    prefix = "fake"

    def __init__(self) -> None:
        super().__init__(components=[])
        self.resets = 0

    def runs_at(self, stage: EvalStage) -> bool:
        del stage
        return True

    def reset(self) -> None:
        super().reset()
        self.resets += 1

    def update(self, eval_out: dict) -> None:
        del eval_out

    def state_for(self, metric_range, metric_filter) -> None:
        del metric_range, metric_filter


def _build_model(metrics: Sequence[MetricSuite] | None = None) -> StreamPETRDetection3DModel:
    return StreamPETRDetection3DModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        img_backbone=_TinyMultiScaleBackbone(),
        img_neck=GeneralizedLSSFPN(in_channels=[128, 256, 512], out_channels=128),
        bbox_head=_build_head(),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        metrics=metrics,
    )


def _build_head(memory_len: int = 32, topk_proposals: int = 8, num_propagated: int = 8):
    return StreamPETRHead(
        class_names=["car", "truck", "bus"],
        in_channels=128,
        hidden_dim=128,
        num_queries=32,
        num_decoder_layers=3,
        num_heads=4,
        feedforward_channels=256,
        memory_len=memory_len,
        topk_proposals=topk_proposals,
        num_propagated=num_propagated,
        with_dn=True,
        with_ego_pos=True,
        depth_num=8,
        LID=True,
        position_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
        scalar=2,
        noise_scale=0.5,
        dn_weight=1.0,
        split=0.5,
        bbox_coder=NMSFreeBBoxCoder3D(
            pc_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
            post_center_range=[-12.0, -12.0, -6.0, 12.0, 12.0, 6.0],
            score_threshold=0.01,
            max_num=16,
        ),
        assigner=HungarianAssigner3D(
            cls_cost=ClassificationCost(weight=2.0),
            reg_cost=BBox3DL1Cost(
                weight=0.25, code_weights=(2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
            ),
            iou_cost=IoU3DCost(weight=0.0),
            point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
        ),
        point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
        code_weights=[2.0, 2.0] + [1.0] * 8,
    )


def _build_images(
    img: torch.Tensor,
    camera_intrinsics: torch.Tensor | None = None,
    pixel_affines: torch.Tensor | None = None,
    lidar2cams: torch.Tensor | None = None,
) -> ImageGTBatch:
    """Image batch for the given ``(B, N, 3, H, W)`` images, identity calibration by default.

    ``pixel_affines`` ``(B, N, 3, 3)`` are the image-space augmentations applied to the raw
    pixels; ``lidar2images`` is composed from the augmented intrinsics as the loaders do.
    """
    batch_size, num_cams = img.shape[:2]
    identity_4x4 = torch.eye(4).expand(batch_size, num_cams, 4, 4).clone()
    if camera_intrinsics is None:
        camera_intrinsics = torch.eye(3).expand(batch_size, num_cams, 3, 3).clone()
    if pixel_affines is None:
        pixel_affines = torch.eye(3).expand(batch_size, num_cams, 3, 3).clone()
    if lidar2cams is None:
        lidar2cams = identity_4x4.clone()
    augmentation_matrices = identity_4x4.clone()
    augmentation_matrices[..., :2, :2] = pixel_affines[..., :2, :2]
    augmentation_matrices[..., :2, 3] = pixel_affines[..., :2, 2]
    augmented_intrinsics = pixel_affines @ camera_intrinsics
    homogeneous_intrinsics = identity_4x4.clone()
    homogeneous_intrinsics[..., :3, :3] = augmented_intrinsics
    return ImageGTBatch(
        images=img,
        depth_maps=None,
        camera_intrinsics=camera_intrinsics,
        image_augmentation_matrices=augmentation_matrices,
        lidar2images=homogeneous_intrinsics @ lidar2cams,
        lidar2cams=lidar2cams,
        calibration_statuses=None,
    )


def _resize_crop_flip_affines(batch_size: int) -> torch.Tensor:
    """Per-camera pixel affines of a resize, a bottom crop and a flip of every other camera."""
    affines = torch.eye(3).repeat(batch_size, NUM_CAMS, 1, 1)
    for index in range(NUM_CAMS):
        scale = 0.5 + 0.01 * index
        affines[:, index, 0, 0] = scale
        affines[:, index, 1, 1] = scale
        affines[:, index, 1, 2] = -20.0 - index
        if index % 2 == 1:
            flip = torch.tensor([[-1.0, 0.0, IMAGE_HW[1] - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            affines[:, index] = flip @ affines[:, index]
    return affines


def _build_frame_meta(
    timestamps: Sequence[float],
    scene_tokens: Sequence[str] | None = None,
    ego2globals: torch.Tensor | None = None,
) -> FrameMetaBatch:
    batch_size = len(timestamps)
    return FrameMetaBatch(
        ego2globals=torch.eye(4).repeat(batch_size, 1, 1) if ego2globals is None else ego2globals,
        scene_tokens=list(scene_tokens or [f"db/scene_{index}/0" for index in range(batch_size)]),
        timestamps_seconds=torch.tensor(timestamps, dtype=torch.float64),
        lanelet_map_paths=[None] * batch_size,
    )


def _build_batch_inputs(
    img: torch.Tensor,
    timestamps: Sequence[float] | None = None,
    scene_tokens: Sequence[str] | None = None,
    gt_boxes: Sequence[torch.Tensor] | None = None,
    gt_labels: Sequence[torch.Tensor] | None = None,
    ego2globals: torch.Tensor | None = None,
) -> ModelBatchInputs:
    """Typed model inputs of one frame per lane; ``gt_boxes`` are 10-wide collated boxes."""
    batch_size = img.shape[0]
    detection = None
    if gt_boxes is not None and gt_labels is not None:
        detection = build_detection3d_gt_batch(gt_boxes, gt_labels)
    return build_batch_inputs(
        detection=detection,
        images=_build_images(img),
        frame_meta=_build_frame_meta(
            [0.0] * batch_size if timestamps is None else timestamps, scene_tokens, ego2globals
        ),
    )


def _collated_box(*values: float) -> torch.Tensor:
    """One collated box ``[x, y, z, l, w, h, yaw, vx, vy, vz]``."""
    return torch.tensor([values], dtype=torch.float32)


def test_streampetr_forward_returns_per_layer_decoder_outputs() -> None:
    """
    Input: two samples of six random camera images, each the first frame of its own scene.
    Expected: the head returns the class logits and the boxes of every decoder layer for the 32
    learnable queries and the 8 propagated from the temporal memory, ``(3, 2, 40, 3)`` and
    ``(3, 2, 40, 10)`` with the three-layer test head.
    Check: compare the shapes of the two head outputs.
    """
    model = _build_model()
    outputs = model(_build_batch_inputs(torch.randn(2, NUM_CAMS, 3, *IMAGE_HW)))

    head_outputs = outputs.detection3d().streampetr_head()
    # 3 decoder layers x (32 learnable queries + 8 propagated from the temporal memory).
    assert head_outputs.all_cls_scores.shape == (3, 2, 40, 3)
    assert head_outputs.all_bbox_preds.shape == (3, 2, 40, 10)


def test_forward_reads_the_stream_metadata_of_the_batch() -> None:
    """
    Input: two frames of one lane in the same scene, 0.1 s apart at an epoch time of 1.7e9 s,
    the second one 65 km from the map origin.
    Expected: the head remembers the scene and the float64 time of the last frame, and leads the
    map-frame bank with an entry carrying the second frame's pose.
    Check: the remembered scene tokens, the dtype of the remembered times and the first bank
    pose.
    """
    model = _build_model().eval()
    head = model.bbox_head
    ego2global = torch.eye(4)
    ego2global[:3, 3] = torch.tensor([65460.2, 675.7, 715.4])
    img = torch.randn(1, NUM_CAMS, 3, *IMAGE_HW)

    with torch.no_grad():
        model(_build_batch_inputs(img, timestamps=[1.7e9], scene_tokens=["db/scene_a/0"]))
        model(
            _build_batch_inputs(
                img,
                timestamps=[1.7e9 + 0.1],
                scene_tokens=["db/scene_a/0"],
                ego2globals=ego2global.unsqueeze(0),
            ),
        )

    assert head.stream_continuity.scene_tokens == ["db/scene_a/0"]
    assert head.stream_continuity.timestamps.dtype == torch.float64
    # The bank is stored in the map frame, so the stored pose is the frame's pose.
    torch.testing.assert_close(head.memory_egopose[0, 0], ego2global)


def test_checkpoint_keys_keep_the_image_encoder_prefixes() -> None:
    """
    Input: the state dict of the test model.
    Expected: every weight outside the head lives under ``image_feature_extractor.img_backbone``
    or ``image_feature_extractor.img_neck``, the prefixes by which trained checkpoints and the
    AWML converter name the image weights.
    Check: the two-level prefixes of the keys outside ``bbox_head`` are exactly those two.
    """
    keys = _build_model().state_dict()

    encoder_prefixes = {
        ".".join(key.split(".")[:2]) for key in keys if not key.startswith("bbox_head.")
    }

    assert encoder_prefixes == {
        "image_feature_extractor.img_backbone",
        "image_feature_extractor.img_neck",
    }


def test_forward_rejects_a_batch_without_images_or_frame_meta() -> None:
    """
    Input: a batch that carries the images but no frame metadata.
    Expected: without the scene and the time of every frame the memory cannot tell a continuing
    stream from a new one, so the forward refuses the batch instead of guessing.
    Check: a ValueError naming the images and the frame metadata is raised.
    """
    model = _build_model()
    with pytest.raises(ValueError, match="images and the frame metadata"):
        model(build_batch_inputs(images=_build_images(torch.randn(1, NUM_CAMS, 3, *IMAGE_HW))))


def test_ground_truth_drops_the_vertical_velocity() -> None:
    """
    Input: one collated ground truth box ``[x, y, z, l, w, h, yaw, vx, vy, vz]`` with label 1.
    Expected: the head regresses ``[x, y, z, l, w, h, yaw, vx, vy]``, so the conversion drops
    the vertical velocity, keeps the other fields in order and hands the labels over as int64.
    Check: compare the converted box and read the dtype of the labels.
    """
    gt_detections = build_detection3d_gt_batch(
        [_collated_box(1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.3, 0.4, 0.9)], [torch.tensor([1])]
    )

    gt_boxes, gt_labels = _ground_truth(gt_detections, torch.zeros(1, dtype=torch.bool))

    # The head regresses [x, y, z, l, w, h, yaw, vx, vy]; the vertical velocity is dropped.
    torch.testing.assert_close(
        gt_boxes[0], torch.tensor([[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.3, 0.4]])
    )
    assert gt_labels[0].dtype == torch.int64


def test_training_step_runs_with_denoising_queries_and_predicts() -> None:
    """
    Input: a training batch of two samples of which only the first carries a box.
    Expected: the training step seeds the denoising queries with that box and returns a finite
    loss that backpropagates, and the decode returns one typed prediction per sample, the
    contract the metric suite consumes.
    Check: the loss is finite, the outputs carry denoising queries, and the decode returns two
    ``Detection3DSamplePredictions``.
    """
    model = _build_model().train()
    model.log_dict = lambda *args, **kwargs: None
    batch_inputs = _build_batch_inputs(
        torch.randn(2, NUM_CAMS, 3, *IMAGE_HW),
        gt_boxes=[
            _collated_box(0.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.0, 0.0, 0.0),
            torch.zeros(0, 10),
        ],
        gt_labels=[torch.tensor([1]), torch.zeros(0, dtype=torch.int64)],
    )

    loss = model.training_step(batch_inputs, batch_idx=0)
    loss.backward()
    outputs = model(batch_inputs)
    predictions = model.decode_outputs(batch_inputs, outputs)

    assert torch.isfinite(loss)
    assert outputs.detection3d().streampetr_head().denoising is not None
    # Predictions follow the typed detection contract consumed by the metric suite.
    assert len(predictions.detection3d_predictions) == 2
    assert all(
        isinstance(prediction, Detection3DSamplePredictions)
        for prediction in predictions.detection3d_predictions
    )


def test_build_eval_output_pairs_predictions_with_ground_truth() -> None:
    """
    Input: one eval frame with one ground truth box.
    Expected: the eval output pairs the predictions of the frame with its ground truth boxes, in
    the 10-wide collated layout, and with its scene token.
    Check: count the predictions and read the shape of the ground truth boxes and the scene
    tokens.
    """
    model = _build_model().eval()
    batch_inputs = _build_batch_inputs(
        torch.randn(1, NUM_CAMS, 3, *IMAGE_HW),
        gt_boxes=[_collated_box(0.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.0, 0.0, 0.0)],
        gt_labels=[torch.tensor([1])],
    )

    with torch.no_grad():
        eval_out = model.build_eval_output(batch_inputs, model(batch_inputs))

    assert len(eval_out["predictions"]) == 1
    assert eval_out["gt_boxes"][0].shape == (1, 10)
    assert eval_out["scene_token"] == ["db/scene_0/0"]


def test_stream_continuity_follows_the_scene_and_the_time_of_every_lane() -> None:
    """
    Input: three lanes without a previous frame; then lane 0 later in its scene, lane 1 in
    another scene and lane 2 back at an earlier time; then a batch of one lane.
    Expected: only a lane that stays in its scene and moves forward in time continues; a batch
    of another size cannot be matched lane by lane and starts anew.
    Check: the ``prev_exists`` flags of the three calls.
    """
    head = _build_model().bbox_head

    # No previous frame: every lane starts a stream.
    first = head.stream_continuity(["a", "b", "c"], torch.tensor([0.0, 0.0, 5.0]))
    # Lane 0 continues, lane 1 moved to another scene, lane 2 restarted its scene.
    second = head.stream_continuity(["a", "d", "c"], torch.tensor([0.5, 0.5, 1.0]))
    # A batch of another size cannot be matched lane by lane.
    third = head.stream_continuity(["a"], torch.tensor([1.0]))

    assert first.tolist() == [0.0, 0.0, 0.0]
    assert second.tolist() == [1.0, 0.0, 0.0]
    assert third.tolist() == [0.0]


def test_streampetr_memory_resets_when_the_lane_changes_scene() -> None:
    """
    Input: two frames of one lane, the first in scene ``a`` and the second in scene ``b``.
    Expected: the scene change drops the first frame's bank; only the second frame's top-k
    proposals remain, at the front.
    Check: the slots behind the first top-k are zero, the first top-k are not, and the bank is
    finite.
    """
    model = _build_model().eval()
    head = model.bbox_head
    topk = head.topk_proposals

    with torch.no_grad():
        model(
            _build_batch_inputs(
                torch.randn(1, NUM_CAMS, 3, *IMAGE_HW), timestamps=[0.0], scene_tokens=["a"]
            ),
        )
        assert (head.memory_embedding[:, :topk] != 0).any()
        model(
            _build_batch_inputs(
                torch.randn(1, NUM_CAMS, 3, *IMAGE_HW), timestamps=[1.0], scene_tokens=["b"]
            ),
        )

    # The first frame's proposals were pushed to slots [topk, 2 * topk) and must be gone.
    assert (head.memory_embedding[:, topk:] == 0).all()
    assert (head.memory_embedding[:, :topk] != 0).any()
    assert torch.isfinite(head.memory_embedding).all()


def _far_pose(x: float, y: float, yaw: float) -> torch.Tensor:
    """A kilometer-scale, rotated lidar-to-map pose ``(1, 4, 4)``."""
    pose = torch.eye(4)
    pose[:2, :2] = torch.tensor([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    pose[:3, 3] = torch.tensor([x, y, 1.8])
    return pose.unsqueeze(0)


def _exported_head_outputs(
    model: StreamPETRDetection3DModel,
    batch_inputs: ModelBatchInputs,
    pose: torch.Tensor,
    pre_memory: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    """Run the three deployed graphs of ``model`` on one frame with the given bank.

    Returns the outputs of the memory graph by name, and the export specs they came from.
    """
    head = model.bbox_head
    img = batch_inputs.image_data.images
    img_features = model.image_feature_extractor(img)
    pos_embed, cone = head.position_embedding(
        img_features,
        batch_inputs.image_data.augmented_camera_intrinsics(),
        *IMAGE_HW,
        img2lidar=torch.inverse(batch_inputs.image_data.lidar2images),
    )
    specs = model.build_export_specs(batch_inputs)
    assert list(specs) == ["extract_img_feat", "position_embedding", "pts_head_memory"]
    # Inverted in float64 like forward does; a float32 inverse at kilometer scale would move the
    # bank by 1e-4 m and fail the exact comparison.
    outputs = specs["pts_head_memory"].module(
        img_features, pos_embed, cone, pose, torch.linalg.inv(pose.double()).float(), *pre_memory
    )
    return dict(zip(specs["pts_head_memory"].output_names, outputs)), specs


def test_exported_head_matches_eval_forward_on_a_propagated_frame() -> None:
    """
    Input: a warm-up frame and a second frame of the same scene 0.5 s later, kilometers from the
    map origin with a yaw; the warm-up bank feeds both the export wrapper and the head.
    Expected: the graph emits exactly what ``predict`` evaluates, through the seven outputs the
    runtime binds and with the last decoder layer only.
    Check: the output names and shapes, the last-layer predictions, the first ``memory_len``
    entries of every ``post_memory_*`` tensor, and that the decoded centers are raw centers.
    """
    model = _build_model().eval()
    head = model.bbox_head
    img = torch.randn(1, NUM_CAMS, 3, *IMAGE_HW)
    pose = _far_pose(3000.0, -1200.0, 0.3)
    # The second frame sits at t = 0, where the head's absolute-time bookkeeping and the
    # frame-relative ages the runtime feeds the graph coincide.
    warm_up = _build_batch_inputs(
        img, timestamps=[-0.5], scene_tokens=["a"], ego2globals=_far_pose(2998.0, -1201.0, 0.25)
    )
    second = _build_batch_inputs(img, timestamps=[0.0], scene_tokens=["a"], ego2globals=pose)

    with torch.no_grad():
        model(warm_up)
        pre_memory = (
            head.memory_embedding.clone(),
            head.memory_reference_point.clone(),
            head.memory_timestamp.clone().float(),
            head.memory_egopose.clone(),
            head.memory_velo.clone(),
        )

        reference = model(second).detection3d().streampetr_head()
        reference_memory = (
            head.memory_embedding,
            head.memory_reference_point,
            head.memory_timestamp.float(),
            head.memory_egopose,
            head.memory_velo,
        )

        exported, specs = _exported_head_outputs(model, second, pose, pre_memory)

    # The outputs the runtime binds; nothing else leaves the graphs.
    assert specs["position_embedding"].args[1].shape == (1, NUM_CAMS, 4, 4)
    assert specs["pts_head_memory"].output_names == [
        "all_cls_scores",
        "all_bbox_preds",
        "post_memory_embedding",
        "post_memory_reference_point",
        "post_memory_timestamp",
        "post_memory_egopose",
        "post_memory_velo",
    ]
    # Only the last decoder layer leaves the graph, the one predict() decodes: the runtime
    # decodes every row of the two outputs, so nothing else may be there.
    num_queries = reference.all_bbox_preds.shape[2]
    assert num_queries == head.num_queries + head.num_propagated
    assert exported["all_cls_scores"].shape == (1, head.num_classes, num_queries)
    assert exported["all_bbox_preds"].shape == (1, 10, num_queries)
    assert exported["post_memory_embedding"].shape == (
        1,
        head.memory_len + head.topk_proposals,
        head.hidden_dim,
    )
    torch.testing.assert_close(exported["all_bbox_preds"][0].T, reference.all_bbox_preds[-1, 0])
    torch.testing.assert_close(exported["all_cls_scores"][0].T, reference.all_cls_scores[-1, 0])
    # The graph emits the concatenated bank; the head keeps the first memory_len entries.
    for name, expected in zip(
        ("embedding", "reference_point", "timestamp", "egopose", "velo"), reference_memory
    ):
        torch.testing.assert_close(exported[f"post_memory_{name}"][:, : head.memory_len], expected)
    # And the offline decode must consume the graph's centers unchanged.
    decoded = head.predict(reference)[0].bboxes_3d
    raw_centers = reference.all_bbox_preds[-1, 0, :, :3]
    assert all(any(torch.allclose(center, raw) for raw in raw_centers) for center in decoded[:, :3])


def test_exported_head_matches_eval_forward_on_the_first_frame_of_a_stream() -> None:
    """
    Input: the first frame of a stream, kilometers from the map origin, with the all-zero bank
    the runtime feeds first.
    Expected: the graph, which has no ``prev_exists`` input, recognizes the zero bank and seeds
    it like the eval head; otherwise its entries would land at the map origin.
    Check: the last-layer boxes and the first ``memory_len`` post-memory reference points.
    """
    model = _build_model().eval()
    head = model.bbox_head
    img = torch.randn(1, NUM_CAMS, 3, *IMAGE_HW)
    pose = _far_pose(3000.0, -1200.0, 0.3)
    first = _build_batch_inputs(img, timestamps=[0.0], scene_tokens=["a"], ego2globals=pose)
    zero_bank = (
        torch.zeros(1, head.memory_len, head.hidden_dim),
        torch.zeros(1, head.memory_len, 3),
        torch.zeros(1, head.memory_len, 1),
        torch.zeros(1, head.memory_len, 4, 4),
        torch.zeros(1, head.memory_len, 2),
    )

    with torch.no_grad():
        reference = model(first).detection3d().streampetr_head()
        reference_points = head.memory_reference_point
        exported, _ = _exported_head_outputs(model, first, pose, zero_bank)

    torch.testing.assert_close(exported["all_bbox_preds"][0].T, reference.all_bbox_preds[-1, 0])
    torch.testing.assert_close(
        exported["post_memory_reference_point"][:, : head.memory_len], reference_points
    )


def test_export_wrappers_are_built_in_eval_mode() -> None:
    """
    Input: the export specs of an eval-mode model.
    Expected: both wrappers are built in eval mode, since the exporter restores a wrapper's mode
    onto the shared head after tracing and a training wrapper would leave the decoder dropout on.
    Check: neither wrapper nor its head is training, and both wrappers share one head.
    """
    model = _build_model().eval()
    batch_inputs = _build_batch_inputs(torch.randn(1, NUM_CAMS, 3, *IMAGE_HW))

    with torch.no_grad():
        specs = model.build_export_specs(batch_inputs)

    for name in ("position_embedding", "pts_head_memory"):
        assert not specs[name].module.training
        assert not specs[name].module.head.training
    assert specs["position_embedding"].module.head is specs["pts_head_memory"].module.head


def _onnx_sessions(
    model: StreamPETRDetection3DModel, batch_inputs: ModelBatchInputs, directory: Path
) -> tuple[dict[str, ExportSpec], dict[str, onnxruntime.InferenceSession]]:
    """Export the three graphs with the settings of the deploy config and open them."""
    specs = model.build_export_specs(batch_inputs)
    onnx_cfg = OmegaConf.create({"opset_version": 17, "dynamo": False, "do_constant_folding": True})
    sessions = {}
    for name, spec in specs.items():
        path = directory / f"{name}.onnx"
        export_to_onnx(
            model=spec.module,
            input_sample=spec.args,
            onnx_cfg=onnx_cfg,
            input_param_names=spec.input_param_names,
            output_names_override=spec.output_names,
            dynamic_axes_override=spec.dynamic_axes,
            output_path=path,
        )
        sessions[name] = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return specs, sessions


@pytest.mark.parametrize("feeding", ["map_frame", "ego_relative"])
def test_onnx_graphs_match_the_eval_forward_over_a_stream(tmp_path: Path, feeding: str) -> None:
    """
    Input: three frames of one scene kilometers from the map origin, turning, seen by six
    differently calibrated cameras; the exported graphs run in ONNX Runtime the way the node
    runs them: a zero bank first, then the previous post bank with its ages dated. The poses
    bound are the map-frame ones, or the identity and the motion since the previous frame.
    Expected: the ONNX graphs predict what the eval forward predicts on every frame, either way.
    Check: the last-layer logits and boxes, and the post-bank reference points and poses in the
    vehicle frame.
    """
    torch.manual_seed(0)
    model = _build_model().eval()
    head = model.bbox_head
    # Spread the logits around zero: the 1 % prior of an untrained head puts every score in the
    # sigmoid tail, where ONNX Runtime rounds them into ties that would decide the top-k order.
    with torch.no_grad():
        nn.init.normal_(head.cls_branches[0][-1].weight, std=0.15)
        nn.init.zeros_(head.cls_branches[0][-1].bias)
    intrinsics = torch.eye(3).repeat(1, NUM_CAMS, 1, 1)
    intrinsics[0, :, 0, 0] = 100.0 + 40.0 * torch.arange(NUM_CAMS)
    intrinsics[0, :, 1, 1] = 1.05 * intrinsics[0, :, 0, 0]
    intrinsics[0, :, 0, 2] = IMAGE_HW[1] / 2
    intrinsics[0, :, 1, 2] = IMAGE_HW[0] / 2
    lidar2cams = torch.eye(4).repeat(1, NUM_CAMS, 1, 1)
    # Optical axes along the lidar x axis turned by each camera's heading.
    lidar_to_optical = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    for index in range(NUM_CAMS):
        heading = _far_pose(0.0, 0.0, 2 * math.pi * index / NUM_CAMS)[0, :3, :3]
        rotation = lidar_to_optical @ heading.T
        lidar2cams[0, index, :3, :3] = rotation
        lidar2cams[0, index, :3, 3] = -rotation @ torch.tensor([0.3 * index, 0.1, 1.6])
    images = [torch.randn(1, NUM_CAMS, 3, *IMAGE_HW) for _ in range(3)]
    poses = [
        _far_pose(3000.0 + 1.2 * index, -1200.0 + 0.4 * index, 0.3 + 0.02 * index)
        for index in range(3)
    ]

    def frame_inputs(index: int) -> ModelBatchInputs:
        return build_batch_inputs(
            images=_build_images(
                images[index], camera_intrinsics=intrinsics, lidar2cams=lidar2cams
            ),
            frame_meta=_build_frame_meta([0.5 * index], ["a"], poses[index]),
        )

    with torch.no_grad():
        specs, sessions = _onnx_sessions(model, frame_inputs(0), tmp_path)
    calibration = specs["position_embedding"]
    pos_embed, cone = sessions["position_embedding"].run(
        None,
        {name: arg.numpy() for name, arg in zip(calibration.input_param_names, calibration.args)},
    )
    head_spec = specs["pts_head_memory"]
    # The bank the runtime clears at start-up, and the date of every entry it keeps between frames.
    bank = {
        name: arg.numpy() for name, arg in zip(head_spec.input_param_names[5:], head_spec.args[5:])
    }
    dates = np.zeros_like(bank["pre_memory_timestamp"])

    def in_vehicle_frame(points: torch.Tensor, to_vehicle: torch.Tensor) -> torch.Tensor:
        return points.double() @ to_vehicle[:3, :3].T + to_vehicle[:3, 3]

    for index, pose in enumerate(poses):
        frame_time = np.float32(0.5 * index)
        if feeding == "map_frame":
            ego_pose = pose.double()
            ego_pose_inv = torch.linalg.inv(ego_pose)
        else:
            # The bank stays in the previous vehicle frame; the motion is computed in float64.
            ego_pose = torch.eye(4, dtype=torch.float64).unsqueeze(0)
            ego_pose_inv = torch.linalg.inv(pose.double()) @ poses[max(index - 1, 0)].double()
        with torch.no_grad():
            expected = model(frame_inputs(index)).detection3d().streampetr_head()
        img_feats = sessions["extract_img_feat"].run(None, {"img": images[index].numpy()})[0]
        feeds = {
            **bank,
            "x": img_feats,
            "pos_embed": pos_embed,
            "cone": cone,
            "data_ego_pose": ego_pose.float().numpy(),
            "data_ego_pose_inv": ego_pose_inv.float().numpy(),
            "pre_memory_timestamp": dates + frame_time,
        }
        outputs = dict(zip(head_spec.output_names, sessions["pts_head_memory"].run(None, feeds)))

        # The float32 map-frame bank has a 0.24 mm ulp at 3 km and ONNX Runtime sums in another
        # order, so the boxes are compared to 1 mm.
        torch.testing.assert_close(
            torch.from_numpy(outputs["all_cls_scores"][0].T),
            expected.all_cls_scores[-1, 0],
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            torch.from_numpy(outputs["all_bbox_preds"][0].T),
            expected.all_bbox_preds[-1, 0],
            atol=1e-3,
            rtol=1e-4,
        )
        # The banks are compared in the vehicle frame, where the eval head's float32 map-frame bank
        # rounds to a fraction of a millimeter at 3 km.
        graph_to_vehicle = torch.linalg.inv(ego_pose)[0]
        head_to_vehicle = torch.linalg.inv(pose.double())[0]
        torch.testing.assert_close(
            in_vehicle_frame(
                torch.from_numpy(outputs["post_memory_reference_point"][0, : head.memory_len]),
                graph_to_vehicle,
            ),
            in_vehicle_frame(head.memory_reference_point[0], head_to_vehicle),
            atol=2e-3,
            rtol=0.0,
        )
        torch.testing.assert_close(
            graph_to_vehicle
            @ torch.from_numpy(outputs["post_memory_egopose"][0, : head.memory_len]).double(),
            head_to_vehicle @ head.memory_egopose[0].double(),
            atol=2e-3,
            rtol=0.0,
        )
        bank = {
            f"pre_memory_{name}": outputs[f"post_memory_{name}"][:, : head.memory_len]
            for name in ("embedding", "reference_point", "egopose", "velo")
        }
        dates = outputs["post_memory_timestamp"][:, : head.memory_len] - frame_time


def test_pseudo_reference_points_cover_the_whole_range_on_every_axis() -> None:
    """
    Input: the test head with 8 propagated queries, and a bare head with the 256 of the
    production recipe.
    Expected: the frozen pseudo reference points are picked evenly from the 7 x 7 x 7 grid, so
    they reach both ends of every axis (the reference's first 256 cells stop at x = 0.79).
    Check: the shapes, the frozen flag, and per-axis minima below 0.1 and maxima above 0.9.
    """
    points = _build_model().bbox_head.pseudo_reference_points
    assert points.shape == (8, 3)
    assert not points.requires_grad
    # 256 of a 7 x 7 x 7 grid, the production setting: the reference's first 256 cells
    # never reach x > 0.79, the even pick does on every axis.
    production = StreamPETRHead.__new__(StreamPETRHead)
    nn.Module.__init__(production)
    production.num_propagated = 256
    production._init_pseudo_reference_points()
    points = production.pseudo_reference_points
    assert points.shape == (256, 3)
    assert (points.amin(dim=0) < 0.1).all() and (points.amax(dim=0) > 0.9).all()


def test_rejects_a_bank_too_small_for_the_propagated_entries() -> None:
    """
    Input: a bank of 4 entries for 8 propagated ones, and a top-k of 4 for 8 propagated ones.
    Expected: the bank has to hold the propagated entries, and the top-k has to cover them
    because they are the strongest proposals of the previous frame, so both heads are rejected
    at construction.
    Check: a ValueError naming ``memory_len`` and one naming ``topk_proposals`` are raised.
    """
    with pytest.raises(ValueError, match="memory_len"):
        _build_head(memory_len=4, topk_proposals=8, num_propagated=8)
    with pytest.raises(ValueError, match="topk_proposals"):
        _build_head(memory_len=32, topk_proposals=4, num_propagated=8)


def test_rejects_a_top_k_larger_than_the_queries_of_a_frame() -> None:
    """
    Input: a top-k of 41 for a frame of 32 + 8 queries, and the largest accepted top-k, 40.
    Expected: 41 is rejected at construction, since the runtime reads
    ``memory_len + topk_proposals`` post-bank entries without checking; 40 fills the bank.
    Check: a ValueError naming the 40 queries, and ``memory_len + 40`` entries in every tensor
    ``push_memory`` returns.
    """
    with pytest.raises(ValueError, match="exceeds the 40 queries"):
        _build_head(memory_len=64, topk_proposals=41, num_propagated=8)
    head = _build_head(memory_len=64, topk_proposals=40, num_propagated=8)
    num_layers, num_slots = head.num_decoder_layers, head.num_queries + head.num_propagated
    pose = torch.eye(4).unsqueeze(0)
    head.pre_update_memory(torch.zeros(1, dtype=torch.float64), torch.zeros(1), pose)

    bank = head.push_memory(
        pose,
        torch.randn(num_layers, 1, num_slots, head.num_classes),
        torch.randn(num_layers, 1, num_slots, 10),
        torch.randn(num_layers, 1, num_slots, head.hidden_dim),
    )

    assert all(tensor.shape[1] == head.memory_len + 40 for tensor in bank)


def test_every_token_carries_the_focal_length_of_its_own_camera() -> None:
    """
    Input: six cameras with focal lengths of 1000 to 6000 pixels and 3 x 5 feature maps.
    Expected: the tokens are camera-major, so the cone of every token carries the focal lengths
    of its own camera, in thousands of pixels; the reference tiles them camera-minor, which
    gives most tokens the focal length of another camera.
    Check: compare both focal channels of the cone with the focal lengths repeated per token.
    """
    head = _build_model().bbox_head
    intrinsics = torch.eye(3).repeat(1, NUM_CAMS, 1, 1)
    focal_lengths = 1000.0 * torch.arange(1, NUM_CAMS + 1, dtype=torch.float32)
    intrinsics[0, :, 0, 0] = intrinsics[0, :, 1, 1] = focal_lengths
    feature_hw = (3, 5)
    img_features = torch.zeros(1, NUM_CAMS, head.hidden_dim, *feature_hw)
    img2lidar = torch.eye(4).repeat(1, NUM_CAMS, 1, 1)

    _, cone = head.position_embedding(img_features, intrinsics, *IMAGE_HW, img2lidar=img2lidar)

    tokens_per_camera = feature_hw[0] * feature_hw[1]
    expected = (focal_lengths / 1e3).repeat_interleave(tokens_per_camera)
    torch.testing.assert_close(cone[0, :, 0], expected)
    torch.testing.assert_close(cone[0, :, 1], expected)


def test_position_embedding_and_export_see_the_intrinsics_of_the_augmented_images() -> None:
    """
    Input: two samples of six cameras, resized, bottom-cropped and every other one flipped, with
    random lidar-to-camera translations.
    Expected: the head and the exported graph see the intrinsics of the augmented images, as the
    runtime binds them, not the raw ones the batch carries.
    Check: the augmented intrinsics reproduce ``lidar2images`` and differ from the raw ones, the
    forward passes them to ``position_embedding``, and the export binds the first sample's,
    padded to 4 x 4.
    """
    model = _build_model().eval()
    head = model.bbox_head
    batch_size = 2
    raw_intrinsics = torch.eye(3).repeat(batch_size, NUM_CAMS, 1, 1)
    raw_intrinsics[..., 0, 0] = 1000.0
    raw_intrinsics[..., 1, 1] = 1100.0
    raw_intrinsics[..., 0, 2] = IMAGE_HW[1]
    raw_intrinsics[..., 1, 2] = IMAGE_HW[0]
    lidar2cams = torch.eye(4).repeat(batch_size, NUM_CAMS, 1, 1)
    lidar2cams[..., :3, 3] = torch.randn(batch_size, NUM_CAMS, 3)
    image_data = _build_images(
        torch.randn(batch_size, NUM_CAMS, 3, *IMAGE_HW),
        camera_intrinsics=raw_intrinsics,
        pixel_affines=_resize_crop_flip_affines(batch_size),
        lidar2cams=lidar2cams,
    )
    batch_inputs = build_batch_inputs(images=image_data, frame_meta=_build_frame_meta([0.0, 0.0]))
    expected = image_data.augmented_camera_intrinsics()
    # The augmented intrinsics are what lidar2images carries on top of the extrinsics.
    torch.testing.assert_close(
        (image_data.lidar2images @ torch.inverse(image_data.lidar2cams))[..., :3, :3],
        expected,
        atol=1e-3,
        rtol=1e-5,
    )
    assert not torch.allclose(expected, raw_intrinsics)

    seen: dict[str, torch.Tensor] = {}
    original = head.position_embedding

    def spy(img_features, camera_intrinsics, *args, **kwargs):
        seen["intrinsics"] = camera_intrinsics
        return original(img_features, camera_intrinsics, *args, **kwargs)

    head.position_embedding = spy
    with torch.no_grad():
        model(batch_inputs)
    head.position_embedding = original
    torch.testing.assert_close(seen["intrinsics"], expected)

    with torch.no_grad():
        specs = model.build_export_specs(batch_inputs)
    # The export binds the first sample's augmented intrinsics, padded to 4x4.
    torch.testing.assert_close(specs["position_embedding"].args[1][..., :3, :3], expected[:1])


def test_phase_hooks_start_new_streams() -> None:
    """
    Input: an eval model that ran one frame, then the start of a validation epoch.
    Expected: the validation scenes start new streams, so the hook clears the bank and the
    previous frame of every lane.
    Check: the bank and the remembered scene tokens are None.
    """
    model = _build_model().eval()
    with torch.no_grad():
        model(_build_batch_inputs(torch.randn(1, NUM_CAMS, 3, *IMAGE_HW)))
    assert model.bbox_head.memory_embedding is not None

    model.on_validation_epoch_start()

    assert model.bbox_head.memory_embedding is None
    assert model.bbox_head.stream_continuity.scene_tokens is None


def test_phase_hooks_reset_the_metrics_before_starting_new_streams() -> None:
    """
    Input: a model whose metric suite counts its resets, at the start of a validation and of a
    test epoch.
    Expected: the stream hooks keep the metric reset of the eval mixin, so the monitored mAP
    does not accumulate across validation epochs.
    Check: each suite was reset once more, and the streams were reset as well.
    """
    model = _build_model(metrics=[_ResetCountingSuite()])
    val_suite = model._stage_metrics(EvalStage.VAL)[0]
    test_suite = model._stage_metrics(EvalStage.TEST)[0]
    resets_after_construction = (val_suite.resets, test_suite.resets)

    model.on_validation_epoch_start()
    model.on_test_epoch_start()

    assert val_suite.resets == resets_after_construction[0] + 1
    assert test_suite.resets == resets_after_construction[1] + 1
    assert model.bbox_head.stream_continuity.scene_tokens is None


@pytest.mark.parametrize(
    "hook",
    [
        "on_train_epoch_start",
        "on_validation_epoch_start",
        "on_test_epoch_start",
        "on_predict_epoch_start",
    ],
)
def test_every_phase_hook_runs_the_base_hook(hook: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Input: each of the four epoch-start hooks, with the hook of the base model replaced by a
    recorder.
    Expected: the stream reset does not hide the base hook: every hook calls it once and starts
    new streams.
    Check: the recorder saw exactly that hook, and the remembered scene tokens are None.
    """
    calls: list[str] = []
    monkeypatch.setattr(ModuleBaseModel, hook, lambda self: calls.append(hook))
    model = _build_model()

    getattr(model, hook)()

    assert calls == [hook]
    assert model.bbox_head.stream_continuity.scene_tokens is None


def test_stream_continuity_breaks_over_a_long_gap() -> None:
    """
    Input: two lanes whose next frames come 0.5 s and 1.5 s after their previous ones, with a
    longest stream gap of 1 s.
    Expected: the lane whose gap exceeds the limit starts a new stream, and the other one
    continues.
    Check: compare the ``prev_exists`` flags of the second call.
    """
    continuity = StreamContinuity(max_stream_gap=1.0)
    continuity(["a", "b"], torch.tensor([0.0, 0.0]))

    assert continuity(["a", "b"], torch.tensor([0.5, 1.5])).tolist() == [1.0, 0.0]


def test_loss_keys_and_graph_are_uniform_without_ground_truth() -> None:
    """
    Input: a training batch of one sample without any box.
    Expected: every rank emits the same loss keys and reaches the same parameters whatever its
    batch holds, so the denoising losses are present and every regression branch gets a gradient
    even without ground truth.
    Check: the two denoising loss keys exist, and after backward every parameter of the
    regression branches has a gradient.
    """
    model = _build_model().train()
    batch_inputs = _build_batch_inputs(
        torch.randn(1, NUM_CAMS, 3, *IMAGE_HW),
        gt_boxes=[torch.zeros(0, 10)],
        gt_labels=[torch.zeros(0, dtype=torch.int64)],
    )

    losses = model.compute_metrics(batch_inputs, model(batch_inputs))

    # Every rank must emit the same loss keys and reach the same parameters,
    # regardless of ground-truth presence in its batch.
    assert "dn_loss_cls" in losses
    assert "dn_loss_bbox" in losses
    losses["loss"].backward()
    assert all(
        parameter.grad is not None for parameter in model.bbox_head.reg_branches.parameters()
    )


def test_memory_alignment_preserves_meter_motion_under_bf16_autocast() -> None:
    """
    Input: a bank entry stored at an ego origin 65 km from the map origin, aligned under bf16
    autocast to the next frame, 2.5 m further along x.
    Expected: the alignment runs in float32 regardless of autocast, so the 2.5 m of motion
    survives; bf16 would round the map-frame point to 256 m steps.
    Check: the bank stays float32, and the entry lands at (-2.5, 0, 0) within 5 cm.
    """
    head = _build_model().bbox_head
    first_pose = torch.eye(4)
    first_pose[:3, 3] = torch.tensor([65460.2, 675.7, 715.4])
    second_pose = first_pose.clone()
    second_pose[0, 3] += 2.5

    head.reset_memory()
    zero_time = torch.zeros(1, dtype=torch.float64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        head.pre_update_memory(zero_time, torch.zeros(1), torch.inverse(first_pose).unsqueeze(0))
        # Store one bank entry at the first ego origin, in the global frame.
        head.memory_reference_point[:, 0] = first_pose[:3, 3]
        head.pre_update_memory(zero_time, torch.ones(1), torch.inverse(second_pose).unsqueeze(0))

    assert head.memory_reference_point.dtype == torch.float32
    assert head.memory_egopose.dtype == torch.float32
    local_point = head.memory_reference_point[0, 0]
    assert torch.allclose(local_point, torch.tensor([-2.5, 0.0, 0.0]), atol=0.05)


@pytest.fixture
def cli_matmul_precision() -> Iterator[None]:
    """Run a test under the float32 matmul precision the CLI runtime sets, then restore it."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 matmuls run on CUDA only")
@pytest.mark.usefixtures("cli_matmul_precision")
def test_memory_alignment_keeps_millimeters_under_the_cli_matmul_precision() -> None:
    """
    Input: on CUDA, under the ``"medium"`` matmul precision the CLI sets, two consecutive
    rotated poses 65 km from the map origin, every query of the first frame predicting the same
    center.
    Expected: the bank geometry does not depend on the matmul precision; a TF32 pose product
    would be off by meters here.
    Check: the pushed poses and reference points are within 1 cm of the float64 relative pose
    and center.
    """
    device = torch.device("cuda")
    head = _build_head().to(device).eval()
    first_pose = _far_pose(65460.2, 675.7, 0.4).to(device)
    second_pose = _far_pose(65461.4, 676.5, 0.41).to(device)
    num_layers, num_slots = head.num_decoder_layers, head.num_queries + head.num_propagated
    # Every query predicts the same center, so every entry pushed to the bank carries it.
    local_center = torch.tensor([12.3, -4.5, 0.7], device=device)
    boxes = torch.zeros(num_layers, 1, num_slots, 10, device=device)
    boxes[..., :3] = local_center
    zero_time = torch.zeros(1, dtype=torch.float64, device=device)

    head.reset_memory()
    head.pre_update_memory(
        zero_time, torch.zeros(1, device=device), torch.linalg.inv(first_pose.double()).float()
    )
    head.post_update_memory(
        zero_time,
        first_pose,
        torch.randn(num_layers, 1, num_slots, head.num_classes, device=device),
        boxes,
        torch.randn(num_layers, 1, num_slots, head.hidden_dim, device=device),
    )
    head.pre_update_memory(
        zero_time, torch.ones(1, device=device), torch.linalg.inv(second_pose.double()).float()
    )

    relative = (torch.linalg.inv(second_pose.double()) @ first_pose.double())[0]
    pushed_poses = head.memory_egopose[0, : head.topk_proposals].double()
    assert (pushed_poses[:, :3, 3] - relative[:3, 3]).norm(dim=-1).max() < 0.01
    # The reference points the bank stores are map-frame too and move by elementwise products
    # as well, so they hold the same precision.
    pushed_centers = head.memory_reference_point[0, : head.topk_proposals].double()
    expected_center = relative[:3, :3] @ local_center.double() + relative[:3, 3]
    assert (pushed_centers - expected_center).norm(dim=-1).max() < 0.01


def test_forward_sanitizes_nan_poisoned_memory() -> None:
    """
    Input: the second frame of a stream whose bank holds four NaN embeddings.
    Expected: the forward replaces non-finite post-normed queries like the reference does, so a
    poisoned bank cannot turn the predictions into NaN.
    Check: every class score and box of the outputs is finite.
    """
    model = _build_model().eval()
    img = torch.randn(1, NUM_CAMS, 3, *IMAGE_HW)

    model(_build_batch_inputs(img, timestamps=[0.0], scene_tokens=["a"]))
    head = model.bbox_head
    head.memory_embedding[:, :4] = float("nan")
    outputs = model(_build_batch_inputs(img, timestamps=[0.5], scene_tokens=["a"]))

    head_outputs = outputs.detection3d().streampetr_head()
    assert torch.isfinite(head_outputs.all_cls_scores).all()
    assert torch.isfinite(head_outputs.all_bbox_preds).all()


def _with_padding(batch_inputs: ModelBatchInputs, padding: Sequence[bool]) -> ModelBatchInputs:
    """Flag the given samples as padding slots, as a streaming batch collates them."""
    gt_batch = batch_inputs.multi_task_gt_batch._replace(padding_mask=torch.tensor(padding))
    return batch_inputs.replace(multi_task_gt_batch=gt_batch)


def _independent_samples(model: StreamPETRDetection3DModel) -> StreamPETRDetection3DModel:
    """Keep the training mode but switch off what couples or randomizes the samples.

    BatchNorm in training mode mixes the samples through its batch statistics (StreamPETR's
    VoVNet runs with ``norm_eval`` and its neck has no normalization, but the test neck has
    BatchNorm), and dropout draws its masks over the whole batch.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
        elif isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
    return model


def _two_samples_with_boxes() -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """Images of two samples and one box of class 1 for each."""
    images = torch.randn(2, NUM_CAMS, 3, *IMAGE_HW)
    boxes = [
        _collated_box(1.0, 0.5, 0.5, 4.0, 2.0, 1.5, 0.2, 0.0, 0.0, 0.0),
        _collated_box(-2.0, 3.0, 0.4, 3.5, 1.8, 1.4, -0.4, 0.5, 0.0, 0.0),
    ]
    return images, boxes, [torch.tensor([1]), torch.tensor([1])]


def test_a_padding_sample_takes_no_part_in_the_loss() -> None:
    """
    Input: two real training samples with one box each, once alone and once followed by a
    padding sample that carries a box too, every image requiring a gradient.
    Expected: the padding sample seeds no denoising query, matches no query and gets no
    classification weight, so the loss is that of the real samples and none of it reaches the
    padding sample's images.
    Check: the loss terms of the two batches, the denoising queries, and the image gradients
    (zero only for the padding sample).
    """
    torch.manual_seed(0)
    model = _independent_samples(_build_model().train())
    images, boxes, labels = _two_samples_with_boxes()
    padded_images = torch.cat([images, images[:1]]).requires_grad_()
    real_batch = _build_batch_inputs(images, gt_boxes=boxes, gt_labels=labels)
    padded_batch = _with_padding(
        _build_batch_inputs(
            padded_images, gt_boxes=[*boxes, boxes[0]], gt_labels=[*labels, labels[0]]
        ),
        [False, False, True],
    )

    torch.manual_seed(1)
    real_outputs = model(real_batch)
    real_losses = model.compute_metrics(real_batch, real_outputs)
    model.bbox_head.reset_memory()
    torch.manual_seed(1)
    padded_outputs = model(padded_batch)
    padded_losses = model.compute_metrics(padded_batch, padded_outputs)
    padded_losses["loss"].backward()

    assert padded_losses.keys() == real_losses.keys()
    for key, value in real_losses.items():
        torch.testing.assert_close(padded_losses[key], value, msg=key)
    denoising = padded_outputs.detection3d().streampetr_head().denoising
    assert denoising is not None
    assert 2 not in denoising.batch_indices.tolist()
    assert denoising.batch_indices.numel() == (
        real_outputs.detection3d().streampetr_head().denoising.batch_indices.numel()
    )
    assert padded_images.grad is not None
    assert torch.count_nonzero(padded_images.grad[2]) == 0
    assert torch.count_nonzero(padded_images.grad[0]) > 0
    assert torch.count_nonzero(padded_images.grad[1]) > 0


def test_a_batch_of_padding_alone_gives_a_zero_loss_over_the_whole_graph() -> None:
    """
    Input: a training batch whose two samples are both padding slots carrying boxes.
    Expected: the batch supervises nothing, but its loss is a zero that keeps the whole graph,
    so backward reaches every branch and the ranks stay in step.
    Check: the loss is zero and requires a gradient, no denoising query is seeded, the loss keys
    are those of a batch without boxes, and every branch parameter gets a gradient.
    """
    model = _build_model().train()
    images, boxes, labels = _two_samples_with_boxes()
    batch_inputs = _with_padding(
        _build_batch_inputs(images, gt_boxes=boxes, gt_labels=labels), [True, True]
    )

    outputs = model(batch_inputs)
    losses = model.compute_metrics(batch_inputs, outputs)
    losses["loss"].backward()

    assert outputs.detection3d().streampetr_head().denoising is None
    assert float(losses["loss"].detach()) == 0.0
    assert losses["loss"].requires_grad
    assert "dn_loss_cls" in losses and "dn_loss_bbox" in losses
    for branches in (model.bbox_head.cls_branches, model.bbox_head.reg_branches):
        assert all(parameter.grad is not None for parameter in branches.parameters())


def test_a_padded_eval_batch_keeps_one_entry_per_sample() -> None:
    """
    Input: an eval batch of three frames with one box each, the middle one a padding slot.
    Expected: the predictions and the eval output keep one entry per sample in batch order, and
    the eval output flags the padding slot for the metrics.
    Check: the number of predictions and per-frame entries, the padding flags and the scene
    tokens.
    """
    model = _build_model().eval()
    images = torch.randn(3, NUM_CAMS, 3, *IMAGE_HW)
    box = _collated_box(0.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.0, 0.0, 0.0)
    batch_inputs = _with_padding(
        _build_batch_inputs(
            images,
            scene_tokens=["db/a/0", "db/b/0", "db/c/0"],
            gt_boxes=[box, box, box],
            gt_labels=[torch.tensor([1])] * 3,
        ),
        [False, True, False],
    )

    with torch.no_grad():
        outputs = model(batch_inputs)
        eval_out = model.build_eval_output(batch_inputs, outputs)
        predictions = model.decode_outputs(batch_inputs, outputs)

    assert len(predictions.detection3d_predictions) == 3
    assert len(eval_out["predictions"]) == len(eval_out["gt_boxes"]) == 3
    assert eval_out["padding_mask"] == [False, True, False]
    assert eval_out["scene_token"] == ["db/a/0", "db/b/0", "db/c/0"]


def test_a_padding_slot_starts_a_new_stream_and_leaves_the_next_epoch_alone() -> None:
    """
    Input: a lane that ends on a frame of scene ``a`` and is padded with a slot repeating that
    frame, followed by the first frame of the next epoch.
    Expected: the slot starts a new stream, and the next epoch's first frame does not see it.
    Check: the continuity flag of the repeated frame, and the outputs of the slot and of the
    next epoch's first frame against a model that saw neither.
    """
    continuity = StreamContinuity()
    last_time = torch.tensor([4.0], dtype=torch.float64)
    continuity(["a"], last_time)
    assert continuity(["a"], last_time).tolist() == [0.0]

    model = _build_model().eval()
    fresh = deepcopy(model)
    earlier, last, next_first = (torch.randn(1, NUM_CAMS, 3, *IMAGE_HW) for _ in range(3))
    with torch.no_grad():
        model(_build_batch_inputs(earlier, timestamps=[3.5], scene_tokens=["a"]))
        model(_build_batch_inputs(last, timestamps=[4.0], scene_tokens=["a"]))
        slot = model(
            _with_padding(_build_batch_inputs(last, timestamps=[4.0], scene_tokens=["a"]), [True]),
        )
        model.on_train_epoch_start()
        after_slot = model(_build_batch_inputs(next_first, timestamps=[0.0], scene_tokens=["b"]))

        first_frame = fresh(_build_batch_inputs(last, timestamps=[4.0], scene_tokens=["a"]))
        fresh.on_train_epoch_start()
        alone = fresh(_build_batch_inputs(next_first, timestamps=[0.0], scene_tokens=["b"]))

    for observed, expected in ((slot, first_frame), (after_slot, alone)):
        observed_head = observed.detection3d().streampetr_head()
        expected_head = expected.detection3d().streampetr_head()
        torch.testing.assert_close(observed_head.all_cls_scores, expected_head.all_cls_scores)
        torch.testing.assert_close(observed_head.all_bbox_preds, expected_head.all_bbox_preds)
