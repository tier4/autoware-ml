"""Export-contract tests for PTv3 detection models."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.tests.models.ptv3_detection_fixtures import (
    build_center_model,
    build_inputs,
    build_trans_model,
    move_batch_to_device,
)

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


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_centerhead_build_export_spec_uses_ptv3_inputs() -> None:
    device = torch.device("cuda")
    model = build_center_model().to(device)
    batch = move_batch_to_device(build_inputs(), device)

    spec = model.build_export_spec(batch)
    outputs = spec.module(*spec.args)

    assert spec.input_param_names == EXPECTED_PTV3_INPUT_NAMES
    assert spec.dynamic_axes is not None
    assert spec.dynamic_axes["serialized_pooling_0_indices"] == {
        0: "serialized_pooling_0_in_voxels"
    }
    assert spec.output_names == ["heatmap", "reg", "height", "dim", "rot", "vel"]
    assert len(outputs) == 6
    assert outputs[0].shape[:2] == (1, 2)


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="PTv3 sparse-convolution export tests require CUDA spconv",
)
def test_ptv3_transhead_build_export_spec_uses_named_detection_outputs() -> None:
    device = torch.device("cuda")
    model = build_trans_model().to(device)
    batch = move_batch_to_device(build_inputs(), device)

    spec = model.build_export_spec(batch)
    outputs = spec.module(*spec.args)

    assert spec.input_param_names == EXPECTED_PTV3_INPUT_NAMES
    assert spec.dynamic_axes is not None
    assert spec.dynamic_axes["serialized_pooling_0_serialized_order"] == {
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
