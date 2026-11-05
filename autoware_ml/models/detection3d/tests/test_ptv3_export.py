"""Export-contract tests for PTv3 detection models."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_inputs,
    build_trans_model,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.onnx_modifiers import TransHeadTensorRTModifier

EXPECTED_PTV3_INPUT_NAMES = [
    "voxels",
    "num_points_per_voxel",
    "grid_coord",
    "patch_order",
    "serialized_inverse",
    "serialized_pooling_0_indices",
    "serialized_pooling_0_indptr",
    "serialized_pooling_0_cluster",
    "serialized_pooling_0_head_indices",
    "serialized_pooling_0_grid_coord",
    "serialized_pooling_0_patch_order",
    "serialized_pooling_0_serialized_inverse",
]


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_transhead_build_export_spec_uses_named_detection3d_outputs() -> None:
    device = torch.device("cuda")
    model = build_trans_model().to(device)
    batch = build_inputs(device=device)

    spec = model.build_export_spec(batch)
    outputs = spec.module(*spec.args)

    assert spec.input_names == EXPECTED_PTV3_INPUT_NAMES
    assert spec.dynamic_axes is not None
    assert spec.dynamic_axes["serialized_pooling_0_serialized_inverse"] == {
        1: "serialized_pooling_0_out_voxels"
    }
    assert spec.output_names == [
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
    assert len(outputs) == 9
    assert outputs[0].shape[:2] == (1, 2)
    assert outputs[2].dtype == torch.long


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_split_export_declares_the_transhead_tensorrt_patches_on_the_head() -> None:
    """
    Input: the split export of a TransHead PTv3 detector.
    Expected: the detection head spec declares the TransHead TensorRT patches with the head's
        own proposal count; the encoder spec declares none.
    Check: ``onnx_modifiers`` of both specs.
    """
    device = torch.device("cuda")
    model = build_trans_model().to(device)

    specs = model.build_export_specs(build_inputs(device=device))

    assert specs["ptv3_det3d_head"].onnx_modifiers == (
        TransHeadTensorRTModifier(k=model.bbox_head.num_proposals),
    )
    assert specs["ptv3_encoder"].onnx_modifiers == ()
