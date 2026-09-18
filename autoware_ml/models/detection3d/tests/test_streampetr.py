"""Unit tests for the native StreamPETR detector."""

from __future__ import annotations

import torch

from autoware_ml.models.common.necks.lss_fpn import GeneralizedLSSFPN
from autoware_ml.models.detection3d.heads.streampetr import StreamPETRHead
from autoware_ml.models.detection3d.streampetr import StreamPETRDetectionModel
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import NMSFreeBBoxCoder3D
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBox3DL1Cost,
    ClassificationCost,
    IoU3DCost,
)


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


def _build_model() -> StreamPETRDetectionModel:
    return StreamPETRDetectionModel(
        img_backbone=_TinyMultiScaleBackbone(),
        img_neck=GeneralizedLSSFPN(in_channels=[128, 256, 512], out_channels=128),
        bbox_head=StreamPETRHead(
            num_classes=3,
            in_channels=128,
            hidden_dim=128,
            num_queries=32,
            num_decoder_layers=3,
            num_heads=4,
            feedforward_channels=256,
            memory_len=32,
            topk_proposals=8,
            num_propagated=8,
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
            ),
            point_cloud_range=[-10.0, -10.0, -5.0, 10.0, 10.0, 5.0],
            code_weights=[2.0, 2.0] + [1.0] * 8,
        ),
    )


def _build_geometry(batch_size: int, num_cams: int) -> tuple[torch.Tensor, torch.Tensor]:
    intrinsics = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
    lidar2cam = torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
    return intrinsics, lidar2cam


def _stream_state(
    batch_size: int, prev_exists: float = 0.0, timestamp: float = 0.0
) -> dict[str, torch.Tensor]:
    identity = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    return {
        "timestamp": torch.full((batch_size,), timestamp, dtype=torch.float64),
        "prev_exists": torch.full((batch_size,), prev_exists),
        "ego_pose": identity,
        "ego_pose_inv": identity.clone(),
    }


def test_streampetr_forward_returns_per_layer_decoder_outputs() -> None:
    model = _build_model()
    img = torch.randn(2, 6, 3, 96, 160)
    intrinsics, lidar2cam = _build_geometry(batch_size=2, num_cams=6)
    outputs = model(img=img, camera_intrinsics=intrinsics, lidar2cam=lidar2cam, **_stream_state(2))

    # 3 decoder layers x (32 learnable queries + 8 propagated from the temporal memory).
    assert outputs["all_cls_scores"].shape == (3, 2, 40, 3)
    assert outputs["all_bbox_preds"].shape == (3, 2, 40, 10)


def test_streampetr_loss_and_predict_run_with_denoising_queries() -> None:
    model = _build_model()
    img = torch.randn(1, 6, 3, 96, 160)
    intrinsics, lidar2cam = _build_geometry(batch_size=1, num_cams=6)
    gt_boxes = [torch.tensor([[0.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.2, 0.0, 0.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]
    outputs = model(
        img=img,
        camera_intrinsics=intrinsics,
        lidar2cam=lidar2cam,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        **_stream_state(1),
    )

    metrics = model.compute_metrics({"gt_boxes": gt_boxes, "gt_labels": gt_labels}, outputs)
    predictions = model.bbox_head.predict(outputs)

    assert "loss" in metrics
    assert outputs["dn_mask_dict"] is not None
    assert isinstance(predictions, list)
    # Predictions must follow the shared detection contract consumed by the
    # metric suite (bboxes_3d / scores_3d / labels_3d).
    for prediction in predictions:
        assert set(prediction) >= {"bboxes_3d", "scores_3d", "labels_3d"}


def test_streampetr_memory_resets_without_stream_continuity() -> None:
    """``prev_exists == 0`` drops the carried bank; only the frame's top-k survives."""
    model = _build_model().eval()
    head = model.bbox_head
    intrinsics, lidar2cam = _build_geometry(batch_size=1, num_cams=6)
    topk = head.topk_proposals

    with torch.no_grad():
        model(
            img=torch.randn(1, 6, 3, 96, 160),
            camera_intrinsics=intrinsics,
            lidar2cam=lidar2cam,
            **_stream_state(1),
        )
        assert (head.memory_embedding[:, :topk] != 0).any()
        model(
            img=torch.randn(1, 6, 3, 96, 160),
            camera_intrinsics=intrinsics,
            lidar2cam=lidar2cam,
            **_stream_state(1, timestamp=1.0),
        )

    # The first frame's proposals were pushed to slots [topk, 2 * topk) and must be gone.
    assert (head.memory_embedding[:, topk:] == 0).all()
    assert (head.memory_embedding[:, :topk] != 0).any()
    assert torch.isfinite(head.memory_embedding).all()


def test_streampetr_builds_three_module_runtime_export() -> None:
    model = _build_model().eval()
    intrinsics, lidar2cam = _build_geometry(batch_size=1, num_cams=6)
    batch = {
        "img": [torch.randn(6, 3, 96, 160)],
        "camera_intrinsics": [intrinsics[0]],
        "lidar2cam": [lidar2cam[0]],
    }

    specs = model.build_export_specs(batch)

    assert list(specs) == ["extract_img_feat", "position_embedding", "pts_head_memory"]
    head = model.bbox_head
    outputs = specs["pts_head_memory"].module(*specs["pts_head_memory"].args)
    named = dict(zip(specs["pts_head_memory"].output_names, outputs))
    num_layers = head.num_decoder_layers
    num_queries = head.num_queries + head.num_propagated
    assert named["all_cls_scores"].shape == (1, head.num_classes, num_layers * num_queries)
    assert named["all_bbox_preds"].shape == (1, 10, num_layers * num_queries)
    assert named["post_memory_embedding"].shape == (
        1,
        head.memory_len + head.topk_proposals,
        head.hidden_dim,
    )
    # The runtime binds exactly these; nothing else leaves the graph.
    assert specs["pts_head_memory"].output_names == [
        "all_cls_scores",
        "all_bbox_preds",
        "post_memory_embedding",
        "post_memory_reference_point",
        "post_memory_timestamp",
        "post_memory_egopose",
        "post_memory_velo",
    ]


def test_exported_head_matches_eval_forward_on_a_propagated_frame() -> None:
    """The deployed graph must emit exactly what ``predict`` evaluates offline.

    Runs one warm-up frame through the training-path forward so the memory
    bank is populated, then feeds the resulting bank to the export wrapper as
    its ``pre_memory_*`` inputs and to the head as a stream-continuous second
    frame. Every prediction and every ``post_memory_*`` tensor has to agree
    element-wise; any re-centering, dtype island or ordering difference between
    the two code paths shows up here instead of on the vehicle.
    """
    model = _build_model().eval()
    head = model.bbox_head
    img = torch.randn(1, 6, 3, 96, 160)
    intrinsics, lidar2cam = _build_geometry(batch_size=1, num_cams=6)

    with torch.no_grad():
        model(img=img, camera_intrinsics=intrinsics, lidar2cam=lidar2cam, **_stream_state(1))
        pre_memory = (
            head.memory_embedding.clone(),
            head.memory_reference_point.clone(),
            head.memory_timestamp.clone().float(),
            head.memory_egopose.clone(),
            head.memory_velo.clone(),
        )
        img_features = model.image_feature_extractor(img)
        pos_embed, cone = head.position_embedding(
            img_features, intrinsics, lidar2cam, image_height=96, image_width=160
        )
        identity = torch.eye(4).unsqueeze(0)

        reference = model(
            img=img,
            camera_intrinsics=intrinsics,
            lidar2cam=lidar2cam,
            **_stream_state(1, prev_exists=1.0),
        )
        reference_memory = (
            head.memory_embedding,
            head.memory_reference_point,
            head.memory_timestamp.float(),
            head.memory_egopose,
            head.memory_velo,
        )

        specs = model.build_export_specs(
            {"img": [img[0]], "camera_intrinsics": [intrinsics[0]], "lidar2cam": [lidar2cam[0]]}
        )
        exported = dict(
            zip(
                specs["pts_head_memory"].output_names,
                specs["pts_head_memory"].module(
                    img_features, pos_embed, cone, identity, identity, *pre_memory
                ),
            )
        )

    num_layers, _, num_queries, _ = reference["all_bbox_preds"].shape
    exported_boxes = exported["all_bbox_preds"].transpose(2, 1).view(num_layers, 1, num_queries, 10)
    exported_scores = (
        exported["all_cls_scores"].transpose(2, 1).view(num_layers, 1, num_queries, -1)
    )
    torch.testing.assert_close(exported_boxes, reference["all_bbox_preds"])
    torch.testing.assert_close(exported_scores, reference["all_cls_scores"])
    # The graph emits the concatenated bank; the head keeps the first memory_len entries.
    for name, expected in zip(
        ("embedding", "reference_point", "timestamp", "egopose", "velo"), reference_memory
    ):
        torch.testing.assert_close(exported[f"post_memory_{name}"][:, : head.memory_len], expected)
    # And the offline decode must consume the graph's centers unchanged.
    decoded = head.predict(reference)[0]["bboxes_3d"]
    raw_centers = reference["all_bbox_preds"][-1, 0, :, :3]
    assert all(any(torch.allclose(center, raw) for raw in raw_centers) for center in decoded[:, :3])


def test_loss_keys_and_graph_are_uniform_without_ground_truth() -> None:
    model = _build_model().train()
    img = torch.randn(1, 6, 3, 96, 160)
    intrinsics, lidar2cam = _build_geometry(batch_size=1, num_cams=6)
    empty_boxes = [torch.zeros(0, 9)]
    empty_labels = [torch.zeros(0, dtype=torch.long)]

    outputs = model(
        img=img,
        camera_intrinsics=intrinsics,
        lidar2cam=lidar2cam,
        gt_boxes=empty_boxes,
        gt_labels=empty_labels,
        **_stream_state(1),
    )
    losses = model.bbox_head.loss(outputs, empty_boxes, empty_labels)

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
    first_state = head._build_stream_state(
        device=torch.device("cpu"),
        timestamp=torch.zeros(1),
        prev_exists=torch.zeros(1),
        ego_pose=first_pose.unsqueeze(0),
        ego_pose_inv=torch.inverse(first_pose).unsqueeze(0),
    )
    second_state = head._build_stream_state(
        device=torch.device("cpu"),
        timestamp=torch.zeros(1),
        prev_exists=torch.ones(1),
        ego_pose=second_pose.unsqueeze(0),
        ego_pose_inv=torch.inverse(second_pose).unsqueeze(0),
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        head.pre_update_memory(first_state)
        # Store one bank entry at the first ego origin, in the global frame.
        head.memory_reference_point[:, 0] = first_pose[:3, 3]
        head.pre_update_memory(second_state)

    assert head.memory_reference_point.dtype == torch.float32
    assert head.memory_egopose.dtype == torch.float32
    local_point = head.memory_reference_point[0, 0]
    assert torch.allclose(local_point, torch.tensor([-2.5, 0.0, 0.0]), atol=0.05)


def test_forward_sanitizes_nan_poisoned_memory_and_keeps_float64_timestamps() -> None:
    model = _build_model().eval()
    img = torch.randn(1, 6, 3, 96, 160)
    intrinsics, lidar2cam = _build_geometry(batch_size=1, num_cams=6)

    model(img=img, camera_intrinsics=intrinsics, lidar2cam=lidar2cam, **_stream_state(1))
    head = model.bbox_head
    assert head.memory_timestamp.dtype == torch.float64

    head.memory_embedding[:, :4] = float("nan")
    outputs = model(
        img=img,
        camera_intrinsics=intrinsics,
        lidar2cam=lidar2cam,
        **_stream_state(1, prev_exists=1.0),
    )

    assert torch.isfinite(outputs["all_cls_scores"]).all()
    assert torch.isfinite(outputs["all_bbox_preds"]).all()
