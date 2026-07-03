"""Export-contract tests for combined PTv3 segmentation+detection export."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autoware_ml.models.detection3d.heads.transfusion import TransFusionHead
from autoware_ml.models.detection3d.ptv3 import PTv3BEVProjection
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.multi.ptv3_segdet import (
    PTv3SegDetModel as PTv3SegmentationDetectionExportModel,
)
from autoware_ml.models.segmentation3d.ptv3 import PTv3SegmentationModel
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.tests.models.ptv3_detection_fixtures import (
    build_bev_encoder,
    build_inputs,
    build_ptv3_backbone,
    build_trans_model,
    move_batch_to_device,
)
from autoware_ml.utils.checkpoints import apply_matching_weights

EXPECTED_PTV3_INPUT_NAMES = [
    "grid_coord",
    "feat",
    "serialized_code",
    "serialized_pooling_0_indices",
    "serialized_pooling_0_indptr",
    "serialized_pooling_0_cluster",
    "serialized_pooling_0_head_indices",
    "serialized_pooling_0_grid_coord",
    "serialized_pooling_0_serialized_order",
    "serialized_pooling_0_serialized_inverse",
]


class _DummyBBoxHead:
    def predict(self, det_outputs: object) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "bboxes_3d": torch.zeros((0, 9), dtype=torch.float32),
                "scores_3d": torch.zeros((0,), dtype=torch.float32),
                "labels_3d": torch.zeros((0,), dtype=torch.long),
            }
        ]


def _save_segmentation_checkpoint(tmp_path: Path) -> Path:
    segmentation_model = PTv3SegmentationModel(
        backbone=build_ptv3_backbone(),
        num_classes=3,
        backbone_out_channels=8,
        ignore_index=-1,
        optimizer=lambda params: torch.optim.AdamW(params, lr=1e-3),
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    )
    checkpoint_path = tmp_path / "ptv3_segmentation.ckpt"
    torch.save({"state_dict": segmentation_model.state_dict()}, checkpoint_path)
    return checkpoint_path


def test_ptv3_segdet_eval_output_scatters_segmentation_to_original_points() -> None:
    model = SimpleNamespace(bbox_head=_DummyBBoxHead())
    outputs = {
        "det_outputs": object(),
        "seg_logits": torch.tensor(
            [
                [3.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=torch.float32,
        ),
    }
    batch = {
        "gt_boxes": [torch.zeros((0, 9), dtype=torch.float32)],
        "gt_labels": [torch.zeros((0,), dtype=torch.long)],
        "inverse": torch.tensor([0, 1, 1, 0], dtype=torch.long),
        "origin_segment": torch.tensor([0, 1, 1, 0], dtype=torch.long),
        "origin_coord": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 3.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    }

    eval_out = PTv3SegmentationDetectionExportModel.build_eval_output(model, batch, outputs)

    assert eval_out["seg_pred_labels"].tolist() == [0, 1, 1, 0]
    assert torch.equal(eval_out["seg_target_labels"], batch["origin_segment"])
    assert torch.equal(eval_out["seg_coord"], batch["origin_coord"])


def _save_detection_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    *,
    drift_backbone: bool = False,
) -> Path:
    state_dict = model.state_dict()
    if drift_backbone:
        state_dict = dict(state_dict)
        key = next(
            key
            for key, value in state_dict.items()
            if key.startswith("backbone.") and torch.is_tensor(value) and value.is_floating_point()
        )
        state_dict[key] = state_dict[key].clone()
        state_dict[key].flatten()[0].add_(1.0)
    torch.save(
        {
            "state_dict": state_dict,
            "autoware_ml_checkpoint_recipe": {
                "type": "ptv3_detection",
                "freeze_backbone": True,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_transhead_segdet_export_uses_named_joint_outputs(tmp_path: Path) -> None:
    segmentation_checkpoint = _save_segmentation_checkpoint(tmp_path)
    detection_model = build_trans_model(freeze_backbone=True)
    apply_matching_weights(detection_model, (segmentation_checkpoint,))
    detection_checkpoint = tmp_path / "ptv3_trans_detection.ckpt"
    _save_detection_checkpoint(detection_checkpoint, detection_model)

    model = PTv3SegmentationDetectionExportModel(
        backbone=build_ptv3_backbone(),
        seg3d_head=torch.nn.Linear(8, 3),
        bev_projector=PTv3BEVProjection(
            in_channels=8, out_channels=16, output_shape=[8, 8], bev_stride=1
        ),
        bev_encoder=build_bev_encoder(),
        bbox_head=TransFusionHead(
            num_proposals=8,
            auxiliary=False,
            in_channels=64,
            hidden_channel=32,
            num_classes=2,
            num_decoder_layers=1,
            num_heads=4,
            feedforward_channels=64,
            common_heads={
                "center": (2, 2),
                "height": (1, 2),
                "dim": (3, 2),
                "rot": (2, 2),
                "vel": (2, 2),
            },
            bbox_coder=TransFusionBBoxCoder(
                pc_range=[0.0, 0.0],
                out_size_factor=1,
                voxel_size=[1.0, 1.0],
                post_center_range=[-1.0, -1.0, -5.0, 10.0, 10.0, 5.0],
                code_size=10,
            ),
            assigner=HungarianAssigner3D(
                cls_cost=ClassificationCost(weight=0.15),
                reg_cost=BBoxBEVL1Cost(weight=0.25),
                iou_cost=IoU3DCost(weight=0.25),
            ),
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[1.0, 1.0, 4.0],
            out_size_factor=1,
            code_weights=[1.0] * 8 + [0.2, 0.2],
            min_radius=1,
            gaussian_overlap=0.1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
        ),
        segmentation_num_classes=3,
        export_output_names=[
            "pred_labels",
            "pred_probs",
            "dense_heatmap",
            "query_heatmap_score",
            "query_labels",
            "heatmap",
            "center",
            "height",
            "dim",
            "rot",
            "vel",
        ],
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    ).cuda()
    apply_matching_weights(
        model,
        (detection_checkpoint, segmentation_checkpoint),
        map_location=torch.device("cuda"),
        device=torch.device("cuda"),
        set_eval=True,
    )

    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))
    spec = model.build_export_spec(batch)
    outputs = spec.module(*spec.args)

    assert spec.input_param_names == EXPECTED_PTV3_INPUT_NAMES
    assert spec.dynamic_axes is not None
    assert spec.dynamic_axes["pred_probs"] == {0: "num_voxels"}
    assert spec.dynamic_axes["serialized_pooling_0_serialized_inverse"] == {
        1: "serialized_pooling_0_out_voxels"
    }
    assert spec.output_names == [
        "pred_labels",
        "pred_probs",
        "dense_heatmap",
        "query_heatmap_score",
        "query_labels",
        "heatmap",
        "center",
        "height",
        "dim",
        "rot",
        "vel",
    ]
    assert len(outputs) == 11
    assert outputs[0].dtype == torch.long
    assert outputs[1].shape[1] == 3
    assert outputs[2].shape[:2] == (1, 2)
    assert outputs[4].dtype == torch.long
