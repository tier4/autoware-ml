"""Unit tests for the native StreamPETR detector."""

from __future__ import annotations

from collections.abc import Sequence
import math

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
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.tests.batch_inputs_fixtures import (
    build_batch_inputs,
    build_detection3d_gt_batch,
)
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor

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


def _build_images(img: torch.Tensor) -> ImageGTBatch:
    """Image batch with identity calibration for the given ``(B, N, 3, H, W)`` images."""
    batch_size, num_cams = img.shape[:2]
    identity_4x4 = torch.eye(4).expand(batch_size, num_cams, 4, 4).clone()
    return ImageGTBatch(
        images=img,
        depth_maps=None,
        camera_intrinsics=torch.eye(3).expand(batch_size, num_cams, 3, 3).clone(),
        image_augmentation_matrices=identity_4x4.clone(),
        lidar2images=identity_4x4.clone(),
        lidar2cams=identity_4x4.clone(),
        calibration_statuses=None,
    )


def _build_frame_meta(
    timestamps: Sequence[float],
    scene_tokens: Sequence[str] | None = None,
    ego2globals: torch.Tensor | None = None,
) -> FrameMetaBatch:
    batch_size = len(timestamps)
    return FrameMetaBatch(
        ego2globals=torch.eye(4).repeat(batch_size, 1, 1) if ego2globals is None else ego2globals,
        scene_tokens=list(scene_tokens or [f"db/scene_{index}/0" for index in range(batch_size)]),
        lanelet_map_paths=[None] * batch_size,
        timestamps=torch.tensor(timestamps, dtype=torch.float64),
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


def _run(model: StreamPETRDetection3DModel, batch_inputs: ModelBatchInputs):
    return model(batch_inputs)


def test_streampetr_forward_returns_per_layer_decoder_outputs() -> None:
    model = _build_model()
    outputs = _run(model, _build_batch_inputs(torch.randn(2, NUM_CAMS, 3, *IMAGE_HW)))

    head_outputs = outputs.detection3d().streampetr_head()
    # 3 decoder layers x (32 learnable queries + 8 propagated from the temporal memory).
    assert head_outputs.all_cls_scores.shape == (3, 2, 40, 3)
    assert head_outputs.all_bbox_preds.shape == (3, 2, 40, 10)


def test_forward_reads_the_calibration_and_the_stream_metadata_of_the_batch() -> None:
    """The lane memory follows the scene tokens and the poses the batch carries."""
    model = _build_model().eval()
    head = model.bbox_head
    ego2global = torch.eye(4)
    ego2global[:3, 3] = torch.tensor([65460.2, 675.7, 715.4])
    img = torch.randn(1, NUM_CAMS, 3, *IMAGE_HW)

    with torch.no_grad():
        _run(model, _build_batch_inputs(img, timestamps=[1.7e9], scene_tokens=["db/scene_a/0"]))
        _run(
            model,
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


def test_forward_rejects_a_batch_without_images_or_frame_meta() -> None:
    model = _build_model()
    with pytest.raises(ValueError, match="images and the frame metadata"):
        model(build_batch_inputs(images=_build_images(torch.randn(1, NUM_CAMS, 3, *IMAGE_HW))))


def test_ground_truth_drops_the_vertical_velocity() -> None:
    gt_detections = build_detection3d_gt_batch(
        [_collated_box(1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.3, 0.4, 0.9)], [torch.tensor([1])]
    )

    gt_boxes, gt_labels = _ground_truth(gt_detections)

    # The head regresses [x, y, z, l, w, h, yaw, vx, vy]; the vertical velocity is dropped.
    torch.testing.assert_close(
        gt_boxes[0], torch.tensor([[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.3, 0.4]])
    )
    assert gt_labels[0].dtype == torch.int64


def test_training_step_runs_with_denoising_queries_and_predicts() -> None:
    """One training step on a batch where only one sample carries boxes."""
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
    outputs = _run(model, batch_inputs)
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
    model = _build_model().eval()
    batch_inputs = _build_batch_inputs(
        torch.randn(1, NUM_CAMS, 3, *IMAGE_HW),
        gt_boxes=[_collated_box(0.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.0, 0.0, 0.0)],
        gt_labels=[torch.tensor([1])],
    )

    with torch.no_grad():
        eval_out = model.build_eval_output(batch_inputs, _run(model, batch_inputs))

    assert len(eval_out["predictions"]) == 1
    assert eval_out["gt_boxes"][0].shape == (1, 10)
    assert eval_out["scene_token"] == ["db/scene_0/0"]


def test_stream_continuity_follows_the_scene_and_the_time_of_every_lane() -> None:
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
    """A new scene drops the carried bank; only the frame's top-k survives."""
    model = _build_model().eval()
    head = model.bbox_head
    topk = head.topk_proposals

    with torch.no_grad():
        _run(
            model,
            _build_batch_inputs(
                torch.randn(1, NUM_CAMS, 3, *IMAGE_HW), timestamps=[0.0], scene_tokens=["a"]
            ),
        )
        assert (head.memory_embedding[:, :topk] != 0).any()
        _run(
            model,
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
    """Run the three deployed graphs of ``model`` on one frame with the given bank."""
    head = model.bbox_head
    img = batch_inputs.image_data.images
    img_features = model.image_feature_extractor(img)
    pos_embed, cone = head.position_embedding(
        img_features,
        batch_inputs.image_data.camera_intrinsics,
        *IMAGE_HW,
        img2lidar=torch.inverse(batch_inputs.image_data.lidar2images),
    )
    specs = model.build_export_specs(batch_inputs)
    assert list(specs) == ["extract_img_feat", "position_embedding", "pts_head_memory"]
    # The runtime binds the inverse pose; computed in float64 like forward does, a
    # float32 LU at kilometer scale would move the bank by 1e-4 m and fail the bit-exact
    # comparison for a reason that has nothing to do with the graphs.
    outputs = specs["pts_head_memory"].module(
        img_features, pos_embed, cone, pose, torch.linalg.inv(pose.double()).float(), *pre_memory
    )
    return dict(zip(specs["pts_head_memory"].output_names, outputs))


def test_exported_head_matches_eval_forward_on_a_propagated_frame() -> None:
    """The deployed graphs must emit exactly what ``predict`` evaluates offline.

    Runs one warm-up frame through the training-path forward so the memory
    bank is populated, then feeds the resulting bank to the export wrapper as
    its ``pre_memory_*`` inputs and to the head as a stream-continuous second
    frame. Every prediction and every ``post_memory_*`` tensor has to agree
    element-wise; any re-centering, dtype island or ordering difference between
    the two code paths shows up here instead of on the vehicle. The frames sit
    kilometers from the map origin with a yaw, so a swapped ``data_ego_pose``
    and ``data_ego_pose_inv`` would show up too.
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
        _run(model, warm_up)
        pre_memory = (
            head.memory_embedding.clone(),
            head.memory_reference_point.clone(),
            head.memory_timestamp.clone().float(),
            head.memory_egopose.clone(),
            head.memory_velo.clone(),
        )

        reference = _run(model, second).detection3d().streampetr_head()
        reference_memory = (
            head.memory_embedding,
            head.memory_reference_point,
            head.memory_timestamp.float(),
            head.memory_egopose,
            head.memory_velo,
        )

        exported = _exported_head_outputs(model, second, pose, pre_memory)
        specs = model.build_export_specs(second)

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
    num_layers, _, num_queries, _ = reference.all_bbox_preds.shape
    assert exported["all_cls_scores"].shape == (1, head.num_classes, num_layers * num_queries)
    assert exported["post_memory_embedding"].shape == (
        1,
        head.memory_len + head.topk_proposals,
        head.hidden_dim,
    )
    exported_boxes = exported["all_bbox_preds"].transpose(2, 1).view(num_layers, 1, num_queries, 10)
    exported_scores = (
        exported["all_cls_scores"].transpose(2, 1).view(num_layers, 1, num_queries, -1)
    )
    torch.testing.assert_close(exported_boxes, reference.all_bbox_preds)
    torch.testing.assert_close(exported_scores, reference.all_cls_scores)
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
    """A wiped (all-zero) bank, as the runtime feeds at start-up, is the head's reset state.

    The eval head seeds the first ``num_propagated`` entries of a fresh bank with the
    pseudo reference points in the lidar frame; the graph has no ``prev_exists`` input and
    must recognize the zero bank itself, otherwise its entries land at the map origin.
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
        reference = _run(model, first).detection3d().streampetr_head()
        reference_points = head.memory_reference_point
        exported = _exported_head_outputs(model, first, pose, zero_bank)

    num_layers, _, num_queries, _ = reference.all_bbox_preds.shape
    exported_boxes = exported["all_bbox_preds"].transpose(2, 1).view(num_layers, 1, num_queries, 10)
    torch.testing.assert_close(exported_boxes, reference.all_bbox_preds)
    torch.testing.assert_close(
        exported["post_memory_reference_point"][:, : head.memory_len], reference_points
    )


def test_pseudo_reference_points_cover_the_whole_range_on_every_axis() -> None:
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
    with pytest.raises(ValueError, match="memory_len"):
        _build_head(memory_len=4, topk_proposals=8, num_propagated=8)
    with pytest.raises(ValueError, match="topk_proposals"):
        _build_head(memory_len=32, topk_proposals=4, num_propagated=8)


def test_every_token_carries_the_focal_length_of_its_own_camera() -> None:
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


def test_phase_hooks_start_new_streams() -> None:
    model = _build_model().eval()
    with torch.no_grad():
        _run(model, _build_batch_inputs(torch.randn(1, NUM_CAMS, 3, *IMAGE_HW)))
    assert model.bbox_head.memory_embedding is not None

    model.on_validation_epoch_start()

    assert model.bbox_head.memory_embedding is None
    assert model.bbox_head.stream_continuity.scene_tokens is None


def test_phase_hooks_reset_the_metrics_before_starting_new_streams() -> None:
    """Regression test: the stream hooks used to hide the metric reset of the eval mixin.

    Without the reset, the detection metric accumulated across validation epochs and the
    monitored mAP, hence the checkpoint selection, drifted with the history of the run.
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


def test_stream_continuity_breaks_over_a_long_gap() -> None:
    continuity = StreamContinuity(max_time_gap=1.0)
    continuity(["a", "b"], torch.tensor([0.0, 0.0]))

    assert continuity(["a", "b"], torch.tensor([0.5, 1.5])).tolist() == [1.0, 0.0]


def test_loss_keys_and_graph_are_uniform_without_ground_truth() -> None:
    model = _build_model().train()
    batch_inputs = _build_batch_inputs(
        torch.randn(1, NUM_CAMS, 3, *IMAGE_HW),
        gt_boxes=[torch.zeros(0, 10)],
        gt_labels=[torch.zeros(0, dtype=torch.int64)],
    )

    losses = model.compute_metrics(batch_inputs, _run(model, batch_inputs))

    # Every rank must emit the same loss keys and reach the same parameters,
    # regardless of ground-truth presence in its batch.
    assert "dn_loss_cls" in losses
    assert "dn_loss_bbox" in losses
    losses["loss"].backward()
    assert all(
        parameter.grad is not None for parameter in model.bbox_head.reg_branches.parameters()
    )


def test_memory_alignment_preserves_meter_motion_under_bf16_autocast() -> None:
    """Kilometer-scale global poses must not lose meter-level ego motion.

    Regression test: the memory alignment matrix products previously ran under
    autocast, quantizing global-frame reference points (~6.5e4 m) to 256 m
    bf16 steps and destroying the propagated queries.
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


def test_forward_sanitizes_nan_poisoned_memory() -> None:
    model = _build_model().eval()
    img = torch.randn(1, NUM_CAMS, 3, *IMAGE_HW)

    _run(model, _build_batch_inputs(img, timestamps=[0.0], scene_tokens=["a"]))
    head = model.bbox_head
    head.memory_embedding[:, :4] = float("nan")
    outputs = _run(model, _build_batch_inputs(img, timestamps=[0.5], scene_tokens=["a"]))

    head_outputs = outputs.detection3d().streampetr_head()
    assert torch.isfinite(head_outputs.all_cls_scores).all()
    assert torch.isfinite(head_outputs.all_bbox_preds).all()
