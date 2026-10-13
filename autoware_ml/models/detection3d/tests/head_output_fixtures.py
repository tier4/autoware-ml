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

"""Builders of the typed detection head outputs the head tests feed in."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    TransFusionHeadOutputs,
    TransFusionSeparateHeadOutputs,
)


def build_transfusion_outputs(tensors: Mapping[str, torch.Tensor]) -> TransFusionHeadOutputs:
    """Build TransFusion head outputs from named tensors.

    The per proposal branches are required. The dense heatmap, the query scores and the query
    labels default to neutral values sized from the proposal heatmap, and the velocity branch
    defaults to absent.

    Args:
        tensors: Head output tensors keyed by their export names.

    Returns:
        The typed head outputs.
    """
    heatmap = tensors["heatmap"]
    batch_size, num_classes, num_proposals = heatmap.shape
    dense_heatmap = (
        tensors["dense_heatmap"]
        if "dense_heatmap" in tensors
        else heatmap.new_zeros((batch_size, num_classes, 4, 4))
    )
    query_heatmap_score = (
        tensors["query_heatmap_score"]
        if "query_heatmap_score" in tensors
        else heatmap.new_ones((batch_size, num_classes, num_proposals))
    )
    query_labels = (
        tensors["query_labels"]
        if "query_labels" in tensors
        else torch.zeros((batch_size, num_proposals), dtype=torch.long)
    )
    return TransFusionHeadOutputs(
        dense_heatmap=dense_heatmap,
        query_heatmap_score=query_heatmap_score,
        query_labels=query_labels,
        separate_head_outputs=TransFusionSeparateHeadOutputs(
            heatmap=heatmap,
            center=tensors["center"],
            height=tensors["height"],
            dim=tensors["dim"],
            rot=tensors["rot"],
            vel=tensors["vel"] if "vel" in tensors else None,
        ),
    )
