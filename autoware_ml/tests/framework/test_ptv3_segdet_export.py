"""Export-contract tests for combined PTv3 segmentation+detection export."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autoware_ml.models.multi.ptv3_segdet import PTv3SegDetModel
from autoware_ml.models.segmentation3d.ptv3_base import seg_head_export_input_names
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.tests.models.ptv3_detection_fixtures import (
    build_bev_neck,
    build_inputs,
    build_ptv3_encoder,
    build_seg_head,
    build_seg_model,
    build_trans_model,
    build_transfusion_head,
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

JOINT_EXPORT_OUTPUT_NAMES = [
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
    checkpoint_path = tmp_path / "ptv3_segmentation.ckpt"
    torch.save({"state_dict": build_seg_model().state_dict()}, checkpoint_path)
    return checkpoint_path


def test_seg_head_export_input_names_follow_dec_depths_rule() -> None:
    """The split seg-head contract: base set for depths-0, per-block-stage metadata otherwise."""
    base_names = [
        "point_feat_0",
        "point_feat_1",
        "point_feat_2",
        "point_feat_3",
        "point_feat_4",
        "pooling_cluster_0",
        "pooling_cluster_1",
        "pooling_cluster_2",
        "pooling_cluster_3",
    ]
    assert seg_head_export_input_names(5, [0, 0, 0, 0]) == base_names
    assert seg_head_export_input_names(5, [0, 0, 1, 1]) == base_names + [
        "serialized_pooling_1_serialized_order",
        "serialized_pooling_1_serialized_inverse",
        "serialized_pooling_1_grid_coord",
        "serialized_pooling_2_serialized_order",
        "serialized_pooling_2_serialized_inverse",
        "serialized_pooling_2_grid_coord",
    ]
    assert seg_head_export_input_names(2, [1]) == [
        "point_feat_0",
        "point_feat_1",
        "pooling_cluster_0",
        "serialized_code",
        "grid_coord",
    ]
    with pytest.raises(ValueError, match="entries"):
        seg_head_export_input_names(5, [0, 0, 0])


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_seg_split_export_supports_decoder_blocks() -> None:
    """A blocks decoder (dec_depths (1,) at stage 0) exports and runs as a split seg head."""
    model = build_seg_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))

    specs = model.build_export_specs(batch)

    # The encoder-only encoder graph never consumes pooling clusters; the
    # tracer would prune them, so the declared interface must not list them.
    assert not any("_cluster" in name for name in specs["encoder"].input_param_names)
    with torch.no_grad():
        encoder_outputs = specs["encoder"].module(*specs["encoder"].args)
    assert len(encoder_outputs) == 2

    spec = specs["seg3d_head"]
    assert spec.input_param_names == [
        "point_feat_0",
        "point_feat_1",
        "pooling_cluster_0",
        "serialized_code",
        "grid_coord",
    ]
    assert spec.dynamic_axes["serialized_code"] == {1: "num_voxels"}
    assert spec.dynamic_axes["grid_coord"] == {0: "num_voxels"}

    with torch.no_grad():
        pred_labels, pred_probs = spec.module(*spec.args)

    num_voxels = batch["coord"].shape[0]
    assert pred_labels.shape == (num_voxels,)
    assert pred_probs.shape == (num_voxels, 3)
    assert pred_labels.dtype == torch.long


def test_ptv3_segdet_eval_output_scatters_segmentation_to_original_points() -> None:
    model = SimpleNamespace(
        bbox_head=_DummyBBoxHead(),
        _detection_frame_mask=PTv3SegDetModel._detection_frame_mask,
        _mask_detection_outputs=PTv3SegDetModel._mask_detection_outputs,
        _mask_list=PTv3SegDetModel._mask_list,
    )
    outputs = {
        "det_outputs": {"heatmap": torch.zeros((1, 2, 8), dtype=torch.float32)},
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

    eval_out = PTv3SegDetModel.build_eval_output(model, batch, outputs)

    assert eval_out["seg_pred_labels"].tolist() == [0, 1, 1, 0]
    assert torch.equal(eval_out["seg_target_labels"], batch["origin_segment"])
    assert torch.equal(eval_out["seg_coord"], batch["origin_coord"])
    assert len(eval_out["predictions"]) == 1
    assert len(eval_out["gt_boxes"]) == 1


def _save_detection_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    *,
    drift_encoder: bool = False,
) -> Path:
    state_dict = model.state_dict()
    if drift_encoder:
        state_dict = dict(state_dict)
        key = next(
            key
            for key, value in state_dict.items()
            if key.startswith("encoder.") and torch.is_tensor(value) and value.is_floating_point()
        )
        state_dict[key] = state_dict[key].clone()
        state_dict[key].flatten()[0].add_(1.0)
    torch.save(
        {
            "state_dict": state_dict,
            "autoware_ml_checkpoint_recipe": {
                "type": "ptv3_detection",
                "freeze_encoder": True,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _build_segdet_model() -> PTv3SegDetModel:
    return PTv3SegDetModel(
        encoder=build_ptv3_encoder(),
        seg3d_head=build_seg_head(),
        bev_neck=build_bev_neck(),
        bbox_head=build_transfusion_head(),
        export_output_names=JOINT_EXPORT_OUTPUT_NAMES,
        grid_size=1.0,
        point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
    )


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_transhead_segdet_export_uses_named_joint_outputs(tmp_path: Path) -> None:
    segmentation_checkpoint = _save_segmentation_checkpoint(tmp_path)
    detection_model = build_trans_model(freeze_encoder=True)
    apply_matching_weights(detection_model, (segmentation_checkpoint,))
    detection_checkpoint = tmp_path / "ptv3_trans_detection.ckpt"
    _save_detection_checkpoint(detection_checkpoint, detection_model)

    model = _build_segdet_model().cuda()
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
    assert spec.output_names == JOINT_EXPORT_OUTPUT_NAMES
    assert len(outputs) == 11
    assert outputs[0].dtype == torch.long
    assert outputs[1].shape[1] == 3
    assert outputs[2].shape[:2] == (1, 2)
    assert outputs[4].dtype == torch.long


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)
def test_ptv3_segdet_detection_outputs_invariant_to_seg_head() -> None:
    """The core stage-4 guarantee: with a fixed encoder, changing the seg head
    must not change detection outputs (the det branch taps the encoder only)."""
    torch.manual_seed(0)
    model = _build_segdet_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))

    with torch.no_grad():
        reference = model(**batch)
        for parameter in model.seg3d_head.parameters():
            parameter.add_(1.0)
        perturbed = model(**batch)

    for name, value in reference["det_outputs"].items():
        assert torch.equal(value, perturbed["det_outputs"][name]), name
    assert not torch.equal(reference["seg_logits"], perturbed["seg_logits"])


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution tests require CUDA spconv",
)
def test_ptv3_segdet_seg_logits_invariant_to_det_branch_pass() -> None:
    """The BEV neck must read the encoder chain non-destructively: seg logits
    are identical whether or not the detection branch ran first."""
    torch.manual_seed(0)
    model = _build_segdet_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))

    with torch.no_grad():
        joint_logits = model(**batch)["seg_logits"]
        point = model.encoder(
            {
                "coord": batch["coord"],
                "feat": batch["feat"],
                "grid_coord": batch["grid_coord"],
                "offset": batch["offset"],
            }
        )
        seg_only_logits = model.seg3d_head(point)

    torch.testing.assert_close(joint_logits, seg_only_logits)
