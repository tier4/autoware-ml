"""Export-contract tests for PTv3 detection models."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.detection3d.ptv3 import (
    det_head_export_dynamic_axes,
    det_head_export_input_names,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.tests.models.ptv3_detection_fixtures import (
    build_inputs,
    build_trans_model,
    move_batch_to_device,
)

EXPECTED_PTV3_INPUT_NAMES = [
    "level_0_grid_coord",
    "feat",
    "level_0_serialized_order",
    "level_0_serialized_inverse",
    "serialized_pooling_0_indices",
    "serialized_pooling_0_indptr",
    "serialized_pooling_0_cluster",
    "serialized_pooling_0_head_indices",
    "level_1_grid_coord",
    "level_1_serialized_order",
    "level_1_serialized_inverse",
]


def test_det_head_export_names_address_levels_not_pooling_stages() -> None:
    """The det head taps the two coarsest levels; every input names the level it belongs to.

    The skip level's coordinates used to be ``point_grid_coord_<l>`` here while the encoder and
    the seg head called the same tensor ``serialized_pooling_<l-1>_grid_coord``.
    """
    names = det_head_export_input_names(stage_count=5)

    assert names == [
        "point_feat_3",
        "point_feat_4",
        "pooling_cluster_3",
        "level_3_grid_coord",
    ]

    axes = det_head_export_dynamic_axes(stage_count=5)

    # pooling_cluster_3 is sized by stage 3's input, which is level 3 - the same symbol the skip
    # level's features and coordinates use, not a separate "in_voxels" name.
    assert axes["point_feat_3"] == {0: "level_3_voxels"}
    assert axes["pooling_cluster_3"] == {0: "level_3_voxels"}
    assert axes["level_3_grid_coord"] == {0: "level_3_voxels"}
    assert axes["point_feat_4"] == {0: "level_4_voxels"}


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
    assert spec.dynamic_axes["level_1_serialized_order"] == {1: "level_1_voxels"}
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
