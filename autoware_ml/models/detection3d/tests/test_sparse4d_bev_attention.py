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

"""Tests for the CMT positional embedding and the BEV cross attention of the Sparse4D head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.models.detection3d.heads.sparse4d import (
    BEVCrossAttention,
    CMTPositionEmbedding,
    Sparse4DHead,
    sincos_position_embedding,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import POINT_CLOUD_RANGE
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import BEVLevel

EMBED_DIMS = 8
BEV_CHANNELS = 4


def _embedding() -> CMTPositionEmbedding:
    return CMTPositionEmbedding(EMBED_DIMS, POINT_CLOUD_RANGE)


def _block(embedding: CMTPositionEmbedding) -> BEVCrossAttention:
    return BEVCrossAttention(
        embed_dims=EMBED_DIMS,
        bev_channels=BEV_CHANNELS,
        num_heads=2,
        position_embedding=embedding,
    )


def _level(cells: int, cell_size: float, seed: int) -> BEVLevel:
    generator = torch.Generator().manual_seed(seed)
    feat = torch.randn((1, BEV_CHANNELS, cells, cells), generator=generator).requires_grad_(True)
    return BEVLevel(feat, (POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1]), cell_size)


def _attention_head(tmp_path: Path) -> Sparse4DHead:
    return build_sparse4d_head(
        make_anchor_file(tmp_path),
        lidar_sampling="bev_attention",
        bev_cell_size=1.0,
        bev_channels=BEV_CHANNELS,
    )


def test_sincos_embedding_alternates_the_sine_and_the_cosine_with_y_first() -> None:
    at_origin = sincos_position_embedding(torch.zeros(2), 4)
    quarter_x = sincos_position_embedding(torch.tensor([0.25, 0.0]), 2)

    assert at_origin.shape == (8,)
    assert torch.equal(at_origin[0::2], torch.zeros(4))
    assert torch.equal(at_origin[1::2], torch.ones(4))
    assert torch.allclose(quarter_x, torch.tensor([0.0, 1.0, 1.0, 0.0]), atol=1e-6)


def test_the_embedding_needs_even_dims() -> None:
    with pytest.raises(ValueError, match="even"):
        CMTPositionEmbedding(7, POINT_CLOUD_RANGE)


def test_a_query_and_the_cell_under_it_share_one_embedding() -> None:
    embedding = _embedding()
    level = BEVLevel(torch.zeros(1, BEV_CHANNELS, 4, 4), (0.0, 0.0), 2.0)

    cells = embedding.cells(level)
    query = embedding(torch.tensor([[3.0, 5.0]]))

    assert cells.shape == (16, EMBED_DIMS)
    # The query sits in the cell of column 1 and row 2, the map flattens row major
    assert torch.allclose(query[0], cells[2 * 4 + 1])
    assert not torch.allclose(query[0], cells[0])


def test_the_block_attends_to_every_cell_of_every_level() -> None:
    embedding = _embedding()
    block = _block(embedding)
    levels = [_level(4, 2.0, 0), _level(2, 4.0, 1)]
    instance_feature = torch.randn(1, 3, EMBED_DIMS)
    anchor = torch.zeros(1, 3, 10)
    anchor[..., :2] = torch.tensor([[1.0, 1.0], [7.0, 7.0], [4.0, 4.0]])

    output = block(instance_feature, anchor, torch.zeros_like(instance_feature), [], levels)
    output.sum().backward()

    assert output.shape == (1, 3, EMBED_DIMS)
    assert all(level.feat.grad is not None and level.feat.grad.abs().sum() > 0 for level in levels)
    assert all(parameter.grad is not None for parameter in embedding.parameters())


def test_the_block_needs_bev_maps_and_reads_no_context() -> None:
    block = _block(_embedding())
    instance_feature = torch.zeros(1, 2, EMBED_DIMS)
    anchor = torch.zeros(1, 2, 10)

    with pytest.raises(ValueError, match="BEV maps"):
        block(instance_feature, anchor, instance_feature, [], [])
    with pytest.raises(ValueError, match="frame context"):
        block(instance_feature, anchor, instance_feature, [], [_level(2, 4.0, 0)], context=1)


def test_an_attention_head_needs_one_bev_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_sparse4d_head(
            make_anchor_file(tmp_path), lidar_sampling="bev_attention", bev_channels=BEV_CHANNELS
        )


def test_an_attention_head_trains_every_parameter_and_samples_no_voxels(tmp_path: Path) -> None:
    head = _attention_head(tmp_path)
    head.cls_threshold_to_reg = 1.0
    bev_map = torch.randn(1, BEV_CHANNELS, 8, 8)
    gt_boxes = [torch.tensor([[4.0, 4.0, 0.0, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]

    outputs = head(bev_map)
    losses = head.loss(outputs, gt_boxes, [torch.tensor([1])], [torch.tensor([0])], [0])
    losses["loss"].backward()

    missing = [
        name
        for name, parameter in head.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    assert head.levels(bev_map) == []
    assert all(isinstance(block, BEVCrossAttention) for block in head.deformable)
    shared = head.deformable[0].position_embedding
    assert all(block.position_embedding is shared for block in head.deformable)


def test_an_attention_head_exports_the_training_forward(tmp_path: Path) -> None:
    head = _attention_head(tmp_path).eval()
    export_head = head.prepare_for_export()
    bev_map = torch.randn(1, BEV_CHANNELS, 8, 8)

    with torch.no_grad():
        train_outputs = head(bev_map)
        export_outputs = export_head(bev_map)

    classification, prediction = export_outputs.export_tensors(["classification", "prediction"])
    assert torch.allclose(classification, train_outputs.classification[:, -1], atol=1e-5)
    assert torch.allclose(prediction, train_outputs.prediction[:, -1], atol=1e-5)
