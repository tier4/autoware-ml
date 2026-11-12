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

"""Export-contract tests for PTv3 detection models."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_inputs,
    build_trans_model,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

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
