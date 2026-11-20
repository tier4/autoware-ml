# Copyright (c) Horizon Robotics. All rights reserved.
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

"""Sparse4D v3 query detection head (arXiv 2311.11722) sampling a sparse voxel stage."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Float32, Int64, Integer
from scipy.optimize import linear_sum_assignment

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DDenoisingOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.anchored import DataAnchored, anchor_tensor
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.models.detection3d.ptv3 import PTv3StageBEVPyramid, SparseStage
from autoware_ml.models.detection3d.task_modules.bbox_coders import (
    denormalize_boxes3d,
    normalize_boxes3d,
)
from autoware_ml.models.detection3d.task_modules.match_costs import ClassificationCost
from autoware_ml.models.detection3d.task_modules.streaming import (
    StreamContinuity,
    transform_reference_points,
)
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import (
    BEVLevel,
    bev_deformable_aggregation,
)
from autoware_ml.ops.deformable_aggregation.sparse_deformable_aggregation import (
    EMPTY_ROW,
    SparseLevel,
    sparse_deformable_aggregation,
)
from autoware_ml.types.geometry import ANCHOR_DIM, AnchorFieldIndex
from autoware_ml.utils.distributed import reduce_mean_count
from autoware_ml.utils.point_cloud.batching import offset_to_batch

_X, _Y, _Z, _LN_L, _LN_W, _LN_H, _SIN_YAW, _COS_YAW, _VX, _VY = (
    int(field) for field in AnchorFieldIndex
)
# Columns of the quality estimate
_CNS, _YNS = range(2)


class LidarSampling(StrEnum):
    """Lidar features the decoder blocks sample at the keypoints.

    Attributes:
        SPARSE: The sparse voxels of the fused stages.
        SPARSE_BEV: The sparse voxels and dense BEV maps of the stages.
        BEV: The dense BEV maps alone.
    """

    SPARSE = "sparse"
    SPARSE_BEV = "sparse_bev"
    BEV = "bev"


class TemporalStateInputs(NamedTuple):
    """Instance bank state of the last frame at the temporal export graph boundary.

    A reset feeds zeros and ``temp_valid = 0``.

    Attributes:
        temp_instance_feature: Cached instance features.
        temp_anchor: Cached anchors in the previous frame.
        temp_confidence: Cached instance confidences.
        temp_valid: Continuity gate of every lane in ``{0, 1}``.
        time_interval: Frame gap of every lane in seconds.
        prev2current: Transform from the previous into the current frame of every lane.
    """

    temp_instance_feature: Float[torch.Tensor, "batch_size num_temp_instances embed_dims"]
    temp_anchor: Float[torch.Tensor, "batch_size num_temp_instances anchor_dim"]
    temp_confidence: Float[torch.Tensor, "batch_size num_temp_instances"]
    temp_valid: Float[torch.Tensor, " batch_size"]
    time_interval: Float[torch.Tensor, " batch_size"]
    prev2current: Float[torch.Tensor, "batch_size 4 4"]


class Sparse4DFrameOutputs(NamedTuple):
    """Last layer outputs of one frame of the temporal export graph and the next bank state.

    The field names are the graph output names.

    Attributes:
        classification: Class logits of every query.
        prediction: Anchor state of every query.
        quality: Centerness and yawness logits of every query.
        instance_feature: Instance feature of every query.
        temp_instance_feature_out: Instance features of the state slots of the next frame.
        temp_anchor_out: Anchors of the state slots.
        temp_confidence_out: Confidences of the state slots.
        temp_query_index_out: Query every state slot was selected from, the runtime carries its
            track ids into the slots.
    """

    classification: Float[torch.Tensor, "batch_size num_queries num_classes"]
    prediction: Float[torch.Tensor, "batch_size num_queries anchor_dim"]
    quality: Float[torch.Tensor, "batch_size num_queries 2"]
    instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"]
    temp_instance_feature_out: Float[torch.Tensor, "batch_size num_temp_instances embed_dims"]
    temp_anchor_out: Float[torch.Tensor, "batch_size num_temp_instances anchor_dim"]
    temp_confidence_out: Float[torch.Tensor, "batch_size num_temp_instances"]
    temp_query_index_out: Int64[torch.Tensor, "batch_size num_temp_instances"]


# Instance bank state at the temporal export graph boundary
TEMPORAL_STATE_INPUT_NAMES = TemporalStateInputs._fields
TEMPORAL_STATE_OUTPUT_NAMES = (
    "temp_instance_feature_out",
    "temp_anchor_out",
    "temp_confidence_out",
)
# Query every state slot was selected from, the runtime carries its track ids into the slots
TEMPORAL_QUERY_INDEX_OUTPUT_NAME = "temp_query_index_out"


def _det3x3(matrix: Float[torch.Tensor, "*batch 3 3"]) -> Float[torch.Tensor, "*batch"]:
    """Batched 3x3 determinant from elementary ops, which ONNX and TensorRT support."""
    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    g, h, i = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


@dataclass(frozen=True)
class EncodedGroundTruth:
    """Ground truth of one sample in anchor space.

    Attributes:
        anchor: Anchor state targets, unknown velocities set to zero.
        labels: Class indices.
        channel_valid: Whether every anchor state channel of every box is known.
    """

    anchor: Float[torch.Tensor, "num_gt_boxes anchor_dim"]
    labels: Int64[torch.Tensor, " num_gt_boxes"]
    channel_valid: Bool[torch.Tensor, "num_gt_boxes anchor_dim"]


def encode_ground_truth(
    boxes: Float[torch.Tensor, "num_gt_boxes box_dim"],
    labels: Integer[torch.Tensor, " num_gt_boxes"],
    device: torch.device,
    dtype: torch.dtype,
) -> EncodedGroundTruth:
    """Encode the ground truth of one sample into anchor space.

    A missing velocity masks its channels out of the regression.

    Args:
        boxes: Ground truth boxes ``(M, >=9)`` in the repository layout.
        labels: Ground truth class indices ``(M,)``.
        device: Device of the encoded tensors.
        dtype: Floating point type of the encoded tensors.

    Returns:
        The encoded ground truth.

    Raises:
        ValueError: If a box has an unknown center, size or yaw.
    """
    boxes = boxes.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(boxes[:, :7]).all()):
        raise ValueError("Every ground truth box needs a finite center, size and yaw.")
    channel_valid = torch.ones((boxes.shape[0], ANCHOR_DIM), dtype=torch.bool, device=device)
    channel_valid[:, _VX : _VY + 1] = torch.isfinite(boxes[:, 7:9])
    anchor = torch.nan_to_num(normalize_boxes3d(boxes), nan=0.0)
    return EncodedGroundTruth(
        anchor=anchor,
        labels=labels.to(device=device, dtype=torch.long),
        channel_valid=channel_valid,
    )


def _rotate(
    vectors: Float[torch.Tensor, "batch_size num_vectors 2"],
    linear: Float[torch.Tensor, "batch_size 2 2"],
) -> Float[torch.Tensor, "batch_size num_vectors 2"]:
    """``vectors @ linear.T`` per sample, exact under TF32 as products and a sum."""
    return (vectors.unsqueeze(-2) * linear[:, None]).sum(-1)


def linear_relu_ln(
    embed_dims: int, in_loops: int, out_loops: int, input_dims: int
) -> list[nn.Module]:
    """Stack ``Linear + ReLU`` blocks terminated by LayerNorm per outer loop."""
    layers: list[nn.Module] = []
    current_dims = input_dims
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(current_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            current_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class BankState(NamedTuple):
    """Instances of one frame and the cached instances warped into it.

    Attributes:
        instance_feature: Fresh instance features.
        anchor: Fresh anchors.
        time_interval: Frame gap of every lane in seconds, the default gap for a lane that starts
            a stream.
        temp_feature: Cached instance features, zero for a lane that starts a stream.
        temp_anchor: Cached anchors warped into the frame, zero for such a lane.
        temp_valid: Continuity gate of every lane in ``{0, 1}``.
    """

    instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"]
    time_interval: Float[torch.Tensor, " batch_size"]
    temp_feature: Float[torch.Tensor, "batch_size num_temp_instances embed_dims"]
    temp_anchor: Float[torch.Tensor, "batch_size num_temp_instances anchor_dim"]
    temp_valid: Float[torch.Tensor, " batch_size"]


class GatedTemporalState(NamedTuple):
    """Cached state warped into the current frame, zero for the lanes that start a stream.

    Attributes:
        feature: Cached instance features.
        anchor: Cached anchors in the current frame.
        confidence: Cached instance confidences.
    """

    feature: Float[torch.Tensor, "batch_size num_temp_instances embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_temp_instances anchor_dim"]
    confidence: Float[torch.Tensor, "batch_size num_temp_instances"]


class InstanceQueries(NamedTuple):
    """Instance features and anchors of a set of queries.

    Attributes:
        instance_feature: Instance feature of every query.
        anchor: Anchor of every query.
    """

    instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"]


class TemporalSelection(NamedTuple):
    """Most confident instances of a frame, the state of the next frame.

    Attributes:
        feature: Instance features of the state slots.
        anchor: Anchors of the state slots.
        confidence: Decayed confidences of the state slots.
        query_index: Query every state slot was selected from.
    """

    feature: Float[torch.Tensor, "batch_size num_temp_instances embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_temp_instances anchor_dim"]
    confidence: Float[torch.Tensor, "batch_size num_temp_instances"]
    query_index: Int64[torch.Tensor, "batch_size num_temp_instances"]


class InstanceBank(DataAnchored):
    """Hold the anchors and propagate confident instances across frames.

    The bank needs scene continuous batch slots, a ``StreamingDataModule`` composed with
    ``/datamodule/streaming@datamodule: default_streaming_datamodule``.
    """

    anchor_names = ("anchor",)

    def __init__(
        self,
        *,
        num_anchor: int,
        embed_dims: int,
        anchor_recipe: AnchorRecipe,
        num_temp_instances: int,
        default_time_interval: float = 0.1,
        min_time_interval: float = 0.05,
        max_time_interval: float = 2.0,
        confidence_decay: float = 0.36,
    ) -> None:
        """Initialize the instance bank.

        Args:
            num_anchor: Total number of instances per frame.
            embed_dims: Instance feature dimension.
            anchor_recipe: Recipe of the initial anchors ``(num_anchor, 10)``.
            num_temp_instances: Number of instances propagated across frames.
            default_time_interval: Time interval in seconds used without
                temporal history.
            min_time_interval: Smallest frame gap in seconds that keeps the
                cached state of a slot. A shorter gap is not a frame interval,
                the velocity refinement divides by it.
            max_time_interval: Largest frame gap in seconds that keeps the
                cached state of a slot.
            confidence_decay: Decay of the cached instance confidence per
                second, applied as ``confidence_decay ** dt``, so the memory
                horizon does not depend on the frame rate.

        Raises:
            ValueError: If the temporal instances do not fit the anchors or the time gate is
                empty.
        """
        super().__init__(anchor_recipe)
        if not 0 < num_temp_instances < num_anchor:
            raise ValueError(
                f"num_temp_instances must be in (0, num_anchor), got {num_temp_instances} "
                f"with num_anchor={num_anchor}."
            )
        if not 0 < min_time_interval < max_time_interval:
            raise ValueError(
                "The time gate needs 0 < min_time_interval < max_time_interval, got "
                f"{min_time_interval} and {max_time_interval}."
            )
        self.num_anchor = num_anchor
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.min_time_interval = min_time_interval
        self.max_time_interval = max_time_interval
        self.confidence_decay = confidence_decay
        self.stream_continuity = StreamContinuity(
            max_stream_gap=max_time_interval, min_stream_gap=min_time_interval
        )
        # Filled from the anchor recipe before training, see DataAnchored
        self.anchor = nn.Parameter(torch.zeros(num_anchor, ANCHOR_DIM))
        self.instance_feature = nn.Parameter(
            torch.zeros(num_anchor, embed_dims), requires_grad=False
        )
        self.reset()

    def load_anchors(self, arrays: Mapping[str, Float32[np.ndarray, "..."]]) -> None:
        """Copy the k-means anchors of the recipe into the bank.

        Args:
            arrays: Anchor arrays of the recipe, ``anchors`` ``(num_anchor, 10)``.
        """
        self.anchor.copy_(anchor_tensor(arrays, "anchors", (self.num_anchor, ANCHOR_DIM)))

    def reset(self) -> None:
        """Drop the cached temporal state of every slot."""
        self.stream_continuity.reset()
        self._cached_feature: (
            Float[torch.Tensor, "batch_size num_temp_instances embed_dims"] | None
        ) = None
        self._cached_anchor: (
            Float[torch.Tensor, "batch_size num_temp_instances anchor_dim"] | None
        ) = None
        self._cached_confidence: Float[torch.Tensor, "batch_size num_temp_instances"] | None = None
        self._cached_pose: Float[torch.Tensor, "batch_size 4 4"] | None = None
        self._slot_valid: Bool[torch.Tensor, " batch_size"] | None = None
        self._temp_time_interval: Float[torch.Tensor, " batch_size"] | None = None
        self._temp_prev_confidence: Float[torch.Tensor, "batch_size num_temp_instances"] | None = (
            None
        )
        self.cached_query_index: Int64[torch.Tensor, "batch_size num_temp_instances"] | None = None
        self.carried_query_index: Int64[torch.Tensor, "batch_size num_temp_instances"] | None = None

    @property
    def lane_continues(self) -> Bool[torch.Tensor, " batch_size"]:
        """Whether every batch lane continues the stream of its previous frame.

        Raises:
            RuntimeError: If the bank served no frame yet.
        """
        if self._slot_valid is None:
            raise RuntimeError("The lane continuity is known only after get() of the bank.")
        return self._slot_valid

    def fresh_instances(self, batch_size: int) -> InstanceQueries:
        """Instance features and anchors every frame starts from.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            The instance features and the anchors, broadcast over the batch.
        """
        return InstanceQueries(
            instance_feature=self.instance_feature[None].expand(batch_size, -1, -1),
            anchor=self.anchor[None].expand(batch_size, -1, -1),
        )

    @staticmethod
    def project_anchor(
        anchor: Float[torch.Tensor, "batch_size num_instances anchor_dim"],
        time_interval: Float[torch.Tensor, " batch_size"],
        transform: Float[torch.Tensor, "batch_size 4 4"],
    ) -> Float[torch.Tensor, "batch_size num_instances anchor_dim"]:
        """Warp cached anchors into the current frame.

        Args:
            anchor: Cached anchors ``(B, K, 10)``.
            time_interval: Frame gaps ``(B,)`` in seconds.
            transform: Frame to frame transforms ``(B, 4, 4)``.

        Returns:
            Warped anchors ``(B, K, 10)``.
        """
        linear = transform[:, :3, :3]

        velocity = anchor[..., _VX : _VY + 1]
        advanced_xy = anchor[..., _X : _Y + 1] + velocity * time_interval[:, None, None]
        center = torch.cat([advanced_xy, anchor[..., _Z : _Z + 1]], dim=-1)
        # Sums of elementwise products, a float32 matmul follows the global TF32 precision
        center = transform_reference_points(center, transform)

        heading = torch.stack([anchor[..., _COS_YAW], anchor[..., _SIN_YAW]], dim=-1)
        heading = _rotate(heading, linear[:, :2, :2])
        heading = F.normalize(heading, dim=-1, eps=1e-6)

        velocity = _rotate(velocity, linear[:, :2, :2])

        scale = _det3x3(linear).abs().clamp(min=1e-6) ** (1.0 / 3.0)
        sizes = anchor[..., _LN_L : _LN_H + 1] + scale.log()[:, None, None]

        return torch.cat([center, sizes, heading[..., 1:2], heading[..., 0:1], velocity], dim=-1)

    @classmethod
    def gate_temporal_state(
        cls,
        feature: Float[torch.Tensor, "batch_size num_temp_instances embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_temp_instances anchor_dim"],
        confidence: Float[torch.Tensor, "batch_size num_temp_instances"],
        valid: Float[torch.Tensor, " batch_size"],
        time_interval: Float[torch.Tensor, " batch_size"],
        transform: Float[torch.Tensor, "batch_size 4 4"],
    ) -> GatedTemporalState:
        """Warp the cached state into the current frame and zero the slots of restarted lanes.

        Args:
            feature: Cached instance features ``(B, K, C)``.
            anchor: Cached anchors ``(B, K, D)`` in the previous frame.
            confidence: Cached instance confidences ``(B, K)``.
            valid: Continuity gates of the lanes ``(B,)`` in ``{0, 1}``.
            time_interval: Frame gaps ``(B,)`` in seconds.
            transform: Transforms from the previous into the current frame ``(B, 4, 4)``.

        Returns:
            The gated features, the warped anchors and the confidences.
        """
        # Selected, not multiplied, so a restart also drops a non-finite cache
        keep = valid > 0.5
        warped_anchor = cls.project_anchor(anchor, time_interval, transform)
        return GatedTemporalState(
            feature=feature.masked_fill(~keep[:, None, None], 0.0),
            anchor=warped_anchor.masked_fill(~keep[:, None, None], 0.0),
            confidence=confidence.masked_fill(~keep[:, None], 0.0),
        )

    def get(
        self,
        batch_size: int,
        timestamps_seconds: Float[torch.Tensor, " batch_size"],
        lidar2global: Float[torch.Tensor, "batch_size 4 4"],
        scene_tokens: Sequence[str],
    ) -> BankState:
        """Return the fresh instances and the warped temporal instances.

        Args:
            batch_size: Number of samples in the batch.
            timestamps_seconds: Frame timestamps ``(B,)`` in seconds.
            lidar2global: Poses ``(B, 4, 4)`` with the augmentations folded in.
            scene_tokens: Scene identifiers of the frames.

        Returns:
            The instances of the frame, the cached ones zeroed for the slots without valid
            history.
        """
        fresh = self.fresh_instances(batch_size)
        device = self.anchor.device
        timestamp = timestamps_seconds.to(device=device, dtype=torch.float64)
        lidar2global = lidar2global.to(device=device, dtype=torch.float64)
        valid = self.stream_continuity(scene_tokens, timestamp).bool()

        if self._cached_feature is None or self._cached_feature.shape[0] != batch_size:
            self._cached_feature = self.anchor.new_zeros(
                (batch_size, self.num_temp_instances, self.embed_dims)
            )
            self._cached_anchor = self.anchor.new_zeros(
                (batch_size, self.num_temp_instances, ANCHOR_DIM)
            )
            self._cached_confidence = self.anchor.new_zeros((batch_size, self.num_temp_instances))
            self._cached_pose = lidar2global.clone()
            # The slots of the previous batch size index no query of this one
            self.cached_query_index = None

        time_interval = torch.where(
            valid,
            self.stream_continuity.gap,
            torch.full_like(timestamp, self.default_time_interval),
        ).to(dtype=torch.float32)

        transform = torch.linalg.solve(lidar2global, self._cached_pose).to(dtype=torch.float32)
        gate = valid.to(dtype=torch.float32)
        gated = self.gate_temporal_state(
            self._cached_feature,
            self._cached_anchor,
            self._cached_confidence,
            gate,
            time_interval,
            transform,
        )

        self._slot_valid = valid
        self._temp_prev_confidence = gated.confidence
        self._temp_time_interval = time_interval
        self._cached_anchor = gated.anchor
        self._cached_feature = gated.feature
        # Query of the previous frame every slot of this frame carries, kept until the next get
        self.carried_query_index = self.cached_query_index
        return BankState(
            instance_feature=fresh.instance_feature,
            anchor=fresh.anchor,
            time_interval=time_interval,
            temp_feature=gated.feature,
            temp_anchor=gated.anchor,
            temp_valid=gate,
        )

    @staticmethod
    def merge_temporal(
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        cls_logits: Float[torch.Tensor, "batch_size num_queries num_classes"],
        temp_feature: Float[torch.Tensor, "batch_size num_temp_instances embed_dims"],
        temp_anchor: Float[torch.Tensor, "batch_size num_temp_instances anchor_dim"],
        gate: Float[torch.Tensor, " batch_size"],
        num_temp_instances: int,
    ) -> InstanceQueries:
        """Merge the propagated instances into the working set after block 0.

        The temporal block replaces the least confident fresh instances at the leading positions.

        Args:
            instance_feature: Fresh instance features ``(B, N, C)``.
            anchor: Anchors refined by block 0 ``(B, N, 10)``.
            cls_logits: Class logits of block 0 ``(B, N, num_classes)``.
            temp_feature: Propagated instance features ``(B, K, C)``.
            temp_anchor: Propagated (warped) anchors ``(B, K, 10)``.
            gate: Continuity gate of every lane ``(B,)`` in ``{0, 1}``.
            num_temp_instances: Size ``K`` of the temporal block.

        Returns:
            The merged instance features and anchors, the temporal block leading.
        """
        num_fresh = instance_feature.shape[1] - num_temp_instances
        confidence = cls_logits.max(dim=-1).values
        order = confidence.argsort(dim=1, descending=True)
        top_index = order[:, :num_fresh]
        remainder_index = order[:, num_fresh:]

        def gather(
            source: Float[torch.Tensor, "batch_size num_queries channels"],
            index: Int64[torch.Tensor, "batch_size num_selected"],
        ) -> Float[torch.Tensor, "batch_size num_selected channels"]:
            return source.gather(1, index[..., None].expand(-1, -1, source.shape[-1]))

        keep = gate[:, None, None] > 0.5
        temporal_feature = torch.where(
            keep, temp_feature, gather(instance_feature, remainder_index)
        )
        temporal_anchor = torch.where(keep, temp_anchor, gather(anchor, remainder_index))

        return InstanceQueries(
            instance_feature=torch.cat(
                [temporal_feature, gather(instance_feature, top_index)], dim=1
            ),
            anchor=torch.cat([temporal_anchor, gather(anchor, top_index)], dim=1),
        )

    @staticmethod
    def select_temporal_state(
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        cls_logits: Float[torch.Tensor, "batch_size num_queries num_classes"],
        prev_confidence: Float[torch.Tensor, "batch_size num_temp_instances"],
        confidence_decay: float,
        time_interval: Float[torch.Tensor, " batch_size"],
        num_temp_instances: int,
    ) -> TemporalSelection:
        """Select the most confident instances as the state of the next frame.

        The previous confidence decays as ``confidence_decay ** dt``.

        Args:
            instance_feature: Instance features of the last layer ``(B, N, C)``.
            anchor: Anchors refined by the last layer ``(B, N, 10)``.
            cls_logits: Class logits of the last layer ``(B, N, num_classes)``.
            prev_confidence: Previous confidence of the temporal block
                ``(B, K)``, zeros for reset slots.
            confidence_decay: Decay per second.
            time_interval: Frame gaps ``(B,)`` in seconds.
            num_temp_instances: Size ``K`` of the temporal state.

        Returns:
            The state of the next frame and the query every state slot was selected from.
        """
        confidence = cls_logits.max(dim=-1).values.sigmoid()
        decay = torch.pow(
            confidence.new_tensor(confidence_decay), time_interval.to(confidence.dtype)
        )
        decayed = torch.cat(
            [
                prev_confidence * decay[:, None],
                torch.zeros_like(confidence[:, num_temp_instances:]),
            ],
            dim=1,
        )
        confidence = torch.maximum(confidence, decayed)

        top_confidence, top_index = confidence.topk(num_temp_instances, dim=1)
        index = top_index[..., None]
        # The anchor width is a constant of the exported graph
        return TemporalSelection(
            feature=instance_feature.gather(1, index.expand(-1, -1, instance_feature.shape[-1])),
            anchor=anchor.gather(1, index.expand(-1, -1, ANCHOR_DIM)),
            confidence=top_confidence,
            query_index=top_index,
        )

    def cache(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        cls_logits: Float[torch.Tensor, "batch_size num_queries num_classes"],
        lidar2global: Float[torch.Tensor, "batch_size 4 4"],
    ) -> None:
        """Write the most confident instances back as the state of the next frame.

        Args:
            instance_feature: Instance features of the last layer ``(B, N, C)``.
            anchor: Anchors refined by the last layer ``(B, N, 10)``.
            cls_logits: Class logits of the last layer ``(B, N, num_classes)``.
            lidar2global: Poses ``(B, 4, 4)`` with the augmentations folded in.
        """
        selection = self.select_temporal_state(
            instance_feature,
            anchor,
            cls_logits,
            self._temp_prev_confidence,
            self.confidence_decay,
            self._temp_time_interval,
            self.num_temp_instances,
        )
        self._cached_feature = selection.feature.detach()
        self._cached_anchor = selection.anchor.detach()
        self._cached_confidence = selection.confidence.detach()
        self.cached_query_index = selection.query_index
        self._cached_pose = lidar2global.to(device=self.anchor.device, dtype=torch.float64).clone()


class InstanceTracker:
    """Identity state of the temporal slots of every batch lane.

    Per lane it keeps the ground truth instances observed in the current stream, the instance
    every cached slot follows and the track id every cached slot carries. The state rolls with
    the slots the instance bank caches and resets with the lanes the bank resets.
    """

    def __init__(
        self, num_temp_instances: int, min_observed_points: int, track_birth_score: float
    ) -> None:
        """Initialize the tracker.

        Args:
            num_temp_instances: Number of slots the instance bank propagates.
            min_observed_points: Points an instance needs in one frame of the stream to
                become a target for the rest of the stream.
            track_birth_score: Score from which a query without a track starts a new one.

        Raises:
            ValueError: If the point threshold is negative.
        """
        if min_observed_points < 0:
            raise ValueError(
                f"min_observed_points must be non-negative, got {min_observed_points}."
            )
        self.num_temp_instances = num_temp_instances
        self.min_observed_points = min_observed_points
        self.track_birth_score = track_birth_score
        self.reset()

    def reset(self) -> None:
        """Forget every stream."""
        self._observed: list[set[int]] = []
        self._slot_owner: Int64[torch.Tensor, "batch_size num_temp_instances"] | None = None
        self._slot_track_id: Int64[torch.Tensor, "batch_size num_temp_instances"] | None = None
        self._query_owner: Int64[torch.Tensor, "batch_size num_queries"] | None = None
        self._query_track_id: Int64[torch.Tensor, "batch_size num_queries"] | None = None
        self._stream_id: Int64[torch.Tensor, " batch_size"] | None = None
        self._frame_observed: list[Bool[torch.Tensor, " _num_gt_boxes"]] | None = None
        self._next_track_id = 0
        self._next_stream_id = 0

    def _new_stream_ids(self, count: int, device: torch.device) -> Int64[torch.Tensor, " count"]:
        """Stream ids unique over the ranks, the rank fills the high bits."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        ids = torch.arange(self._next_stream_id, self._next_stream_id + count, device=device)
        self._next_stream_id += count
        return ids + (rank << 40)

    def begin_frame(
        self,
        lane_continues: Bool[torch.Tensor, " batch_size"],
        query_index: Int64[torch.Tensor, "batch_size num_temp_instances"] | None,
        num_query: int,
    ) -> None:
        """Carry the state into the cached slots and reset the lanes that start a stream.

        Every query starts without an instance except the leading temporal slots, which keep the
        instance of their slot until an assignment of their lane replaces it.

        Args:
            lane_continues: Whether every lane continues its stream ``(B,)``.
            query_index: Query index of every slot the bank cached at the previous frame
                ``(B, K)``, None at the first frame.
            num_query: Number of queries of the frame.
        """
        batch_size = lane_continues.shape[0]
        device = lane_continues.device
        if self._stream_id is None or self._stream_id.shape[0] != batch_size:
            self._observed = [set() for _ in range(batch_size)]
            self._slot_owner = torch.full(
                (batch_size, self.num_temp_instances), -1, dtype=torch.long, device=device
            )
            self._slot_track_id = self._slot_owner.clone()
            self._stream_id = self._new_stream_ids(batch_size, device)
        else:
            self._slot_owner = self._query_owner.gather(1, query_index)
            self._slot_track_id = self._query_track_id.gather(1, query_index)
        restart = ~lane_continues
        self._slot_owner[restart] = -1
        self._slot_track_id[restart] = -1
        for lane in restart.nonzero().flatten().tolist():
            self._observed[lane] = set()
        # Out of place, a stream id made under trainer.validate or test is an inference tensor
        self._stream_id = self._stream_id.masked_scatter(
            restart, self._new_stream_ids(int(restart.sum()), device)
        )
        # The temporal slots lead the queries, see InstanceBank.merge_temporal
        self._query_owner = torch.full((batch_size, num_query), -1, dtype=torch.long, device=device)
        self._query_owner[:, : self.num_temp_instances] = self._slot_owner
        self._frame_observed = None

    def observe(
        self,
        instance_ids: Sequence[Int64[torch.Tensor, " _num_gt_boxes"]],
        num_points: Sequence[Integer[torch.Tensor, " _num_gt_boxes"]],
    ) -> list[Bool[torch.Tensor, " _num_gt_boxes"]]:
        """Add the instances seen in this frame and tell which boxes are targets.

        Args:
            instance_ids: Instance id of every ground truth box of every lane.
            num_points: Point count of every ground truth box of every lane.

        Returns:
            Per lane mask of the boxes whose instance was observed in the stream so far.
        """
        masks = []
        for lane, (ids, points) in enumerate(zip(instance_ids, num_points, strict=True)):
            seen = self._observed[lane]
            seen.update(ids[points >= self.min_observed_points].tolist())
            masks.append(
                torch.tensor(
                    [value in seen for value in ids.tolist()], dtype=torch.bool, device=ids.device
                )
            )
        self._frame_observed = masks
        return masks

    def observed_masks(self) -> list[Bool[torch.Tensor, " _num_gt_boxes"]]:
        """Per lane mask of the boxes of this frame whose instance was observed so far.

        Raises:
            ValueError: If forward() got no detection ground truth.
        """
        if self._frame_observed is None:
            raise ValueError(
                "The Sparse4D head needs forward() to receive the detection ground truth with the "
                "instance ids and the point counts of the boxes."
            )
        return self._frame_observed

    def slot_owners(
        self, lanes: Sequence[int]
    ) -> Int64[torch.Tensor, "num_lanes num_temp_instances"]:
        """Instance every slot of the given lanes follows ``(len(lanes), K)``, -1 when free."""
        return self._slot_owner[list(lanes)]

    def set_query_owners(
        self,
        lanes: Sequence[int],
        owners: Int64[torch.Tensor, "num_lanes num_queries"],
        scores: Float[torch.Tensor, "num_lanes num_queries"],
    ) -> None:
        """Record the instance every confident query of the given lanes was assigned to.

        Only a query scoring at least ``track_birth_score`` takes or keeps its instance, so the
        unconfident matches of early training never lock an instance to a slot. The other lanes
        keep the instances their slots carried in.

        Args:
            lanes: Lanes the owners belong to.
            owners: Assigned instance id of every query ``(len(lanes), N)``, -1 for background.
            scores: Detection score of every query ``(len(lanes), N)``.
        """
        self._query_owner[list(lanes)] = torch.where(
            scores >= self.track_birth_score, owners, torch.full_like(owners, -1)
        )

    def assign_track_ids(
        self, scores: Float[torch.Tensor, "batch_size num_queries"]
    ) -> Int64[torch.Tensor, "batch_size num_queries"]:
        """Give every query the track of its slot or a new track when it is confident enough.

        The temporal slots lead the queries, see :meth:`InstanceBank.merge_temporal`.

        Args:
            scores: Detection score of every query ``(B, N)``.

        Returns:
            Track id of every query ``(B, N)``, -1 for queries without a track.
        """
        track_ids = torch.full(scores.shape, -1, dtype=torch.long, device=scores.device)
        track_ids[:, : self.num_temp_instances] = self._slot_track_id
        births = (scores >= self.track_birth_score) & (track_ids < 0)
        num_births = int(births.sum())
        track_ids[births] = torch.arange(
            self._next_track_id, self._next_track_id + num_births, device=scores.device
        )
        self._next_track_id += num_births
        self._query_track_id = track_ids
        return track_ids

    @property
    def stream_ids(self) -> Int64[torch.Tensor, " batch_size"]:
        """Stream id of every lane ``(B,)``, new whenever a lane starts a stream.

        Raises:
            ValueError: If no frame began since the last reset.
        """
        if self._stream_id is None:
            raise ValueError("The stream ids are known only after the first frame of the lanes.")
        return self._stream_id


class SparseBox3DEncoder(nn.Module):
    """Embed anchor states into positional features."""

    def __init__(self, embed_dims: int) -> None:
        """Initialize the anchor encoder.

        Args:
            embed_dims: Output embedding dimension.
        """
        super().__init__()
        self.pos_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 3))
        self.size_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 3))
        self.yaw_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 2))
        self.vel_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 2))
        self.output_fc = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, embed_dims))

    def forward(
        self, anchor: Float[torch.Tensor, "*batch anchor_dim"]
    ) -> Float[torch.Tensor, "*batch embed_dims"]:
        """Encode anchors ``(..., 10)`` into embeddings ``(..., C)``."""
        pos_feat = self.pos_fc(anchor[..., _X : _Z + 1])
        size_feat = self.size_fc(anchor[..., _LN_L : _LN_H + 1])
        yaw_feat = self.yaw_fc(anchor[..., _SIN_YAW : _COS_YAW + 1])
        vel_feat = self.vel_fc(anchor[..., _VX : _VY + 1])
        return self.output_fc(pos_feat + size_feat + yaw_feat + vel_feat)


class SparseBox3DKeyPointsGenerator(nn.Module):
    """Generate 3D keypoints from anchors in the shared metric frame."""

    def __init__(
        self,
        *,
        embed_dims: int,
        fix_scale: Sequence[Sequence[float]],
        num_learnable_pts: int,
    ) -> None:
        """Initialize the keypoint generator.

        Args:
            embed_dims: Instance feature dimension driving the learnable offsets.
            fix_scale: Fixed keypoint offsets as fractions of the box size.
            num_learnable_pts: Number of keypoints the instance feature places.

        Raises:
            ValueError: If the fixed offsets are not 3D points.
        """
        super().__init__()
        fix_scale_tensor = torch.tensor(fix_scale, dtype=torch.float32)
        if fix_scale_tensor.ndim != 2 or fix_scale_tensor.shape[1] != 3:
            raise ValueError(f"fix_scale must be (P, 3), got {tuple(fix_scale_tensor.shape)}.")
        self.fix_scale: Float[torch.Tensor, "num_fixed_points 3"]
        self.register_buffer("fix_scale", fix_scale_tensor, persistent=False)
        self.num_learnable_pts = num_learnable_pts
        self.num_pts: int = fix_scale_tensor.shape[0] + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(embed_dims, num_learnable_pts * 3)

    def forward(
        self,
        anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
    ) -> Float[torch.Tensor, "batch_size num_queries num_points 3"]:
        """Generate keypoints ``(B, N, P, 3)`` from anchors ``(B, N, 10)``."""
        batch_size, num_anchor = anchor.shape[:2]
        size = anchor[..., None, _LN_L : _LN_H + 1].exp()
        key_points = self.fix_scale * size
        if self.num_learnable_pts > 0:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(batch_size, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)

        sin_yaw = anchor[..., _SIN_YAW]
        cos_yaw = anchor[..., _COS_YAW]
        rotation = anchor.new_zeros((batch_size, num_anchor, 3, 3))
        rotation[..., 0, 0] = cos_yaw
        rotation[..., 0, 1] = -sin_yaw
        rotation[..., 1, 0] = sin_yaw
        rotation[..., 1, 1] = cos_yaw
        rotation[..., 2, 2] = 1.0
        key_points = torch.matmul(rotation[:, :, None], key_points[..., None]).squeeze(-1)
        return key_points + anchor[..., None, _X : _Z + 1]


# Below this occupied weight the aggregated features fade to zero instead of amplifying a sliver
MIN_OCCUPANCY = 0.05


class SparseLevelGrid(nn.Module):
    """Dense voxel grid of one sparse encoder stage, it builds the lookup of the stage voxels."""

    def __init__(
        self, point_cloud_range: Sequence[float], grid_size: float, feature_stride: int
    ) -> None:
        """Initialize the stage grid.

        Args:
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            feature_stride: Total pooling stride of the stage.

        Raises:
            ValueError: If the range or the stride is invalid.
        """
        super().__init__()
        if feature_stride < 1:
            raise ValueError(f"feature_stride must be positive, got {feature_stride}.")
        pc_range = [float(value) for value in point_cloud_range]
        if len(pc_range) != 6:
            raise ValueError(f"point_cloud_range must have 6 values, got {pc_range}.")
        base_cells = [round((pc_range[axis + 3] - pc_range[axis]) / grid_size) for axis in range(3)]
        self.feature_stride = feature_stride
        size_x, size_y, size_z = (math.ceil(cells / feature_stride) for cells in base_cells)
        self.grid_dims: tuple[int, int, int] = (size_x, size_y, size_z)
        self.num_cells: int = size_x * size_y * size_z
        self.voxel_size: float = grid_size * feature_stride
        self.grid_origin: tuple[float, float, float] = (pc_range[0], pc_range[1], pc_range[2])

    def forward(
        self,
        feat: Float[torch.Tensor, "num_voxels channels"],
        grid_coord: Integer[torch.Tensor, "num_voxels 3"],
        offset: Integer[torch.Tensor, " batch_size"],
    ) -> SparseLevel:
        """Look up the stage voxels in the dense grid of every sample.

        Args:
            feat: Stage features ``(V, C)``.
            grid_coord: Stage voxel coordinates ``(V, 3)`` as ``(x, y, z)``.
            offset: Cumulative voxel offsets of the samples ``(B,)``.

        Returns:
            The stage with its voxel lookup.

        Raises:
            ValueError: If a voxel lies outside the grid, outside an export.
        """
        if not torch.onnx.is_in_onnx_export():
            outside = (grid_coord < 0) | (grid_coord >= grid_coord.new_tensor(self.grid_dims))
            num_outside = int(outside.any(dim=1).sum())
            if num_outside > 0:
                raise ValueError(
                    f"{num_outside} stage voxels lie outside the grid {self.grid_dims}, the head "
                    "range or grid size differs from the voxelizer."
                )
        size_x, size_y, size_z = self.grid_dims
        batch_index = offset_to_batch(offset, grid_coord)
        flat_cells = (
            (batch_index * size_z + grid_coord[:, 2].long()) * size_y + grid_coord[:, 1].long()
        ) * size_x + grid_coord[:, 0].long()
        # The rows fit int32, which halves the dense grid against int64
        index_grid = torch.full(
            (offset.shape[0] * self.num_cells,),
            EMPTY_ROW,
            dtype=torch.int32,
            device=feat.device,
        )
        index_grid[flat_cells] = torch.arange(
            1, feat.shape[0] + 1, dtype=torch.int32, device=feat.device
        )
        padded = torch.cat([feat.new_zeros((1, feat.shape[1])), feat])
        return SparseLevel(padded, index_grid, self.grid_dims, self.grid_origin, self.voxel_size)


class SparseDeformableAggregation(nn.Module):
    """Sample sparse voxel features at anchor keypoints with learned weights.

    Keypoints interpolate trilinearly over every sampled stage, the weights normalize over the
    stages and the keypoints of every feature group. Keypoints in empty space read zeros, so the
    weighted sum is divided by the weight that landed on occupied voxels. A BEV branch samples
    dense BEV maps at the same keypoints and adds to the voxel features, or replaces them.
    """

    def __init__(
        self,
        *,
        embed_dims: int,
        num_groups: int,
        num_levels: int,
        kps_generator: SparseBox3DKeyPointsGenerator,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        bev_aggregation: BEVDeformableAggregation | None = None,
        sample_sparse: bool = True,
    ) -> None:
        """Initialize the sparse deformable aggregation branch.

        Args:
            embed_dims: Instance and voxel feature dimension.
            num_groups: Number of feature groups sharing one weight.
            num_levels: Number of sampled encoder stages.
            kps_generator: Shared keypoint generator.
            proj_drop: Dropout on the output projection.
            attn_drop: Probability of dropping whole keypoints in training.
            bev_aggregation: BEV map sampling at the same keypoints, None without BEV maps.
            sample_sparse: Whether the block samples the sparse voxels.

        Raises:
            ValueError: If the features do not split into the groups or the block samples no
                lidar features.
        """
        super().__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(f"embed_dims {embed_dims} not divisible by num_groups {num_groups}.")
        if not sample_sparse and bev_aggregation is None:
            raise ValueError("The block samples the sparse voxels, BEV maps or both.")
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.kps_generator = kps_generator
        self.num_pts: int = kps_generator.num_pts
        self.attn_drop = attn_drop
        self.sample_sparse = sample_sparse
        # A BEV only block has no voxel weights, every parameter must train under DDP
        self.weights_fc: nn.Linear | None = (
            nn.Linear(embed_dims, num_groups * num_levels * self.num_pts) if sample_sparse else None
        )
        self.output_proj: nn.Linear | None = (
            nn.Linear(embed_dims, embed_dims) if sample_sparse else None
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.bev_aggregation = bev_aggregation
        if self.weights_fc is not None:
            nn.init.zeros_(self.weights_fc.weight)
            nn.init.zeros_(self.weights_fc.bias)

    def forward(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        anchor_embed: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        levels: Sequence[SparseLevel],
        bev_levels: Sequence[BEVLevel] = (),
    ) -> Float[torch.Tensor, "batch_size num_queries embed_dims"]:
        """Aggregate sparse voxel features into the instance features.

        Args:
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchor states ``(B, N, 10)``.
            anchor_embed: Anchor embeddings ``(B, N, C)``.
            levels: Sampled encoder stages with their voxel lookup.
            bev_levels: Dense BEV maps of the BEV branch.

        Returns:
            Updated instance features ``(B, N, C)``.
        """
        key_points = self.kps_generator(anchor, instance_feature)
        query = instance_feature + anchor_embed
        bev = (
            None
            if self.bev_aggregation is None
            else self.bev_aggregation(query, key_points, bev_levels)
        )
        if self.sample_sparse:
            weights = keypoint_dropout(
                _group_softmax(self.weights_fc(query), self.num_levels * self.num_pts),
                self.attn_drop,
                self.training,
            )
            features = sparse_deformable_aggregation(key_points, weights, levels, MIN_OCCUPANCY)
            output = self.output_proj(features) if bev is None else self.output_proj(features) + bev
        else:
            output = bev
        return instance_feature + self.proj_drop(output)


def _group_softmax(
    logits: Float[torch.Tensor, "batch_size num_queries num_weights"], num_samples: int
) -> Float[torch.Tensor, "batch_size num_queries num_samples num_groups"]:
    """Softmax over the samples of every feature group, ``(B, N, S * G)`` to ``(B, N, S, G)``."""
    batch_size, num_anchor = logits.shape[:2]
    return logits.reshape(batch_size, num_anchor, num_samples, -1).softmax(dim=-2)


def keypoint_dropout(
    weights: Float[torch.Tensor, "batch_size num_queries num_samples num_groups"],
    drop: float,
    training: bool,
) -> Float[torch.Tensor, "batch_size num_queries num_samples num_groups"]:
    """Drop whole samples of the aggregation weights in training and rescale the rest."""
    if not training or drop == 0.0:
        return weights
    mask = torch.rand_like(weights[..., :1]) > drop
    return weights * mask / (1.0 - drop)


class BEVDeformableAggregation(nn.Module):
    """Sample dense BEV maps at the keypoints' (x, y) with learned weights per level and point."""

    def __init__(
        self,
        *,
        embed_dims: int,
        bev_channels: int,
        num_groups: int,
        num_levels: int,
        num_points: int,
        attn_drop: float = 0.0,
    ) -> None:
        """Initialize the BEV branch.

        Args:
            embed_dims: Instance feature dimension.
            bev_channels: Feature dimension of the BEV maps.
            num_groups: Number of feature groups sharing one weight.
            num_levels: Number of sampled BEV levels.
            num_points: Keypoints per instance.
            attn_drop: Probability of dropping whole samples in training.

        Raises:
            ValueError: If the BEV channels do not split into the groups.
        """
        super().__init__()
        if bev_channels % num_groups != 0:
            raise ValueError(f"bev_channels {bev_channels} not divisible by {num_groups} groups.")
        self.num_levels = num_levels
        self.num_samples: int = num_levels * num_points
        self.attn_drop = attn_drop
        self.weights_fc = nn.Linear(embed_dims, num_groups * self.num_samples)
        self.output_proj = nn.Linear(bev_channels, embed_dims)
        nn.init.zeros_(self.weights_fc.weight)
        nn.init.zeros_(self.weights_fc.bias)

    def forward(
        self,
        query: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        key_points: Float[torch.Tensor, "batch_size num_queries num_points 3"],
        levels: Sequence[BEVLevel],
    ) -> Float[torch.Tensor, "batch_size num_queries embed_dims"]:
        """Aggregate the BEV samples of the keypoints into instance features.

        Args:
            query: Instance features with the anchor embedding ``(B, N, C)``.
            key_points: Metric keypoints ``(B, N, P, 3)``.
            levels: BEV levels, finest first.

        Returns:
            BEV features of the instances ``(B, N, C)``.

        Raises:
            ValueError: If the number of levels does not match the branch.
        """
        if len(levels) != self.num_levels:
            raise ValueError(f"The BEV branch samples {self.num_levels} levels, got {len(levels)}.")
        weights = keypoint_dropout(
            _group_softmax(self.weights_fc(query), self.num_samples), self.attn_drop, self.training
        )
        return self.output_proj(bev_deformable_aggregation(key_points, weights, levels))


class DecoupledInstanceAttention(nn.Module):
    """Instance attention with decoupled (concatenated) anchor embeddings.

    With key inputs it reads the temporal memory.
    """

    def __init__(self, embed_dims: int, num_heads: int, dropout: float) -> None:
        """Initialize the decoupled attention block.

        Args:
            embed_dims: Instance feature dimension.
            num_heads: Number of attention heads.
            dropout: Attention and residual dropout probability.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dims * 2, num_heads, dropout=dropout, batch_first=True
        )
        self.fc_value = nn.Linear(embed_dims, embed_dims * 2, bias=False)
        self.fc_out = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Factor of the block output, one until an export folds a scale in
        self.output_scale: Float[torch.Tensor, ""]
        self.register_buffer("output_scale", torch.ones(()), persistent=False)

    def fold_output_scale(self, norm: nn.LayerNorm) -> None:
        """Divide the block output by a power of two into the fp16 range, the norm output unchanged.

        Args:
            norm: LayerNorm the block output feeds, its eps follows the scale.
        """
        value_weight = self.attn.in_proj_weight.chunk(3)[2]
        gain = math.prod(
            torch.linalg.matrix_norm(weight.detach().float(), ord=2).item()
            for weight in (
                self.fc_value.weight,
                value_weight,
                self.attn.out_proj.weight,
                self.fc_out.weight,
            )
        )
        # A power of two above the value path gain keeps the fold exact and the output at norm scale
        scale = 2.0 ** math.ceil(math.log2(max(gain, 1.0)))
        with torch.no_grad():
            self.fc_value.weight.div_(scale)
            self.attn.in_proj_bias.chunk(3)[2].div_(scale)
            self.attn.out_proj.bias.div_(scale)
            self.output_scale.fill_(1.0 / scale)
        norm.eps /= scale**2

    def forward(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        anchor_embed: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        key_feature: Float[torch.Tensor, "batch_size num_keys embed_dims"] | None = None,
        key_embed: Float[torch.Tensor, "batch_size num_keys embed_dims"] | None = None,
        attn_mask: Bool[torch.Tensor, "num_queries num_keys"] | None = None,
        key_padding_mask: Bool[torch.Tensor, "batch_size num_keys"] | None = None,
    ) -> Float[torch.Tensor, "batch_size num_queries embed_dims"]:
        """Attend with anchor position aware queries and keys.

        Args:
            instance_feature: Query instance features ``(B, N, C)``.
            anchor_embed: Query anchor embeddings ``(B, N, C)``.
            key_feature: Optional key and value instance features ``(B, K, C)``.
            key_embed: Optional key anchor embeddings ``(B, K, C)``.
            attn_mask: Optional boolean mask ``(N, K)`` where ``True`` blocks
                attention, the denoising isolation mask of training steps.
            key_padding_mask: Optional boolean mask ``(B, K)`` where ``True`` hides a key from
                every query of its row.

        Returns:
            Updated instance features ``(B, N, C)``.
        """
        tokens = torch.cat([instance_feature, anchor_embed], dim=-1)
        if key_feature is None:
            key_tokens = tokens
            value = self.fc_value(instance_feature)
        else:
            key_tokens = torch.cat([key_feature, key_embed], dim=-1)
            value = self.fc_value(key_feature)
        attended = self.attn(
            tokens, key_tokens, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )[0]
        return self.fc_out(tokens * self.output_scale + self.dropout(attended))


class Refinement(NamedTuple):
    """Outputs of one refinement block.

    Attributes:
        anchor: Refined anchors.
        cls_logits: Class logits.
        quality: Centerness and yawness logits.
    """

    anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"]
    cls_logits: Float[torch.Tensor, "batch_size num_queries num_classes"]
    quality: Float[torch.Tensor, "batch_size num_queries 2"]


class SparseBox3DRefinementModule(nn.Module):
    """Refine anchor states, classify instances and estimate the box quality."""

    def __init__(self, *, embed_dims: int, num_classes: int) -> None:
        """Initialize the refinement module.

        Args:
            embed_dims: Instance feature dimension.
            num_classes: Number of detection classes.
        """
        super().__init__()
        self.refine_layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2, embed_dims),
            nn.Linear(embed_dims, ANCHOR_DIM),
        )
        self.cls_layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, embed_dims),
            nn.Linear(embed_dims, num_classes),
        )
        self.quality_layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2, embed_dims),
            nn.Linear(embed_dims, 2),
        )
        cls_bias = -math.log((1.0 - 0.01) / 0.01)
        nn.init.constant_(self.cls_layers[-1].bias, cls_bias)

    def forward(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        anchor_embed: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        time_interval: Float[torch.Tensor, " batch_size"],
    ) -> Refinement:
        """Refine anchors, classify instances and estimate their quality.

        Args:
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchor states ``(B, N, 10)``.
            anchor_embed: Anchor embeddings ``(B, N, C)``.
            time_interval: Time intervals of the samples ``(B,)`` in seconds.

        Returns:
            The refined anchors, the class logits and the centerness and yawness logits.
        """
        feature = instance_feature + anchor_embed
        delta = self.refine_layers(feature)
        refined = anchor + delta
        velocity = (
            delta[..., _VX : _VY + 1] / time_interval[:, None, None] + anchor[..., _VX : _VY + 1]
        )
        return Refinement(
            anchor=torch.cat([refined[..., :_VX], velocity], dim=-1),
            cls_logits=self.cls_layers(instance_feature),
            quality=self.quality_layers(feature),
        )


@dataclass(frozen=True)
class Sparse4DTargets:
    """Training targets of one decoder layer from Hungarian matching.

    Attributes:
        cls_targets: One-hot class targets.
        box_targets: Anchor state targets.
        box_weights: Regression weights, zero for unmatched queries and unknown channels.
        matched: Whether every query is matched to a box.
        assigned_instance: Instance id of the box every query is matched to, -1 for background.
        num_pos: Number of matched queries.
    """

    cls_targets: Float[torch.Tensor, "batch_size num_queries num_classes"]
    box_targets: Float[torch.Tensor, "batch_size num_queries anchor_dim"]
    box_weights: Float[torch.Tensor, "batch_size num_queries anchor_dim"]
    matched: Bool[torch.Tensor, "batch_size num_queries"]
    assigned_instance: Int64[torch.Tensor, "batch_size num_queries"]
    num_pos: int


@dataclass(frozen=True)
class DenoisingQueries:
    """Denoising queries of one batch and their fixed targets.

    Each group holds noised positives followed by displaced negatives.

    Attributes:
        anchor: Noised anchors of the denoising queries.
        cls_targets: One-hot class targets, zero for the negatives.
        box_targets: Anchor state targets of the positives.
        box_weights: Regression weights, zero for the negatives and the padding.
        cls_weights: Classification weights, zero for the padding.
        attn_mask: Isolation mask of the regular and the denoising queries, True blocks.
    """

    anchor: Float[torch.Tensor, "batch_size num_dn_queries anchor_dim"]
    cls_targets: Float[torch.Tensor, "batch_size num_dn_queries num_classes"]
    box_targets: Float[torch.Tensor, "batch_size num_dn_queries anchor_dim"]
    box_weights: Float[torch.Tensor, "batch_size num_dn_queries anchor_dim"]
    cls_weights: Float[torch.Tensor, "batch_size num_dn_queries"]
    attn_mask: Bool[torch.Tensor, "num_all_queries num_all_queries"]


def _signed_uniform(
    shape: tuple[int, ...], low: float, high: float, device: torch.device
) -> Float[torch.Tensor, "..."]:
    """Sample ``sign * U(low, high)`` with an independent random sign per element."""
    magnitude = torch.empty(shape, device=device).uniform_(low, high)
    sign = torch.randint(0, 2, shape, device=device, dtype=torch.float32) * 2.0 - 1.0
    return magnitude * sign


def _noise_denoising_anchors(
    encoded_gt: Float[torch.Tensor, "num_gt_boxes anchor_dim"],
    num_groups: int,
    *,
    center_low: float,
    center_high: float,
    noise_size: float,
    noise_yaw: float,
    noise_velocity: float,
) -> Float[torch.Tensor, "num_groups num_gt_boxes anchor_dim"]:
    """Corrupt encoded ground truth boxes with noise relative to the box size.

    Args:
        encoded_gt: Clean anchor space boxes ``(M, 10)``.
        num_groups: Number of independent noised copies ``G``.
        center_low: Lower bound of the per axis center offset magnitude.
        center_high: Upper bound of the per axis center offset magnitude.
        noise_size: Half range of the uniform log size jitter.
        noise_yaw: Half range of the uniform heading rotation in radians.
        noise_velocity: Half range of the uniform velocity jitter in m/s.

    Returns:
        Noised anchors ``(G, M, 10)``.
    """
    device = encoded_gt.device
    num_gt = encoded_gt.shape[0]
    noised = encoded_gt[None].repeat(num_groups, 1, 1)

    half_extent = noised[..., _LN_L : _LN_H + 1].exp() / 2.0
    units = _signed_uniform((num_groups, num_gt, 3), center_low, center_high, device)
    offsets = units * half_extent
    sin_yaw = noised[..., _SIN_YAW].clone()
    cos_yaw = noised[..., _COS_YAW].clone()
    noised[..., _X] += offsets[..., 0] * cos_yaw - offsets[..., 1] * sin_yaw
    noised[..., _Y] += offsets[..., 0] * sin_yaw + offsets[..., 1] * cos_yaw
    noised[..., _Z] += offsets[..., 2]

    noised[..., _LN_L : _LN_H + 1] += _signed_uniform(
        (num_groups, num_gt, 3), 0.0, noise_size, device
    )

    theta = _signed_uniform((num_groups, num_gt), 0.0, noise_yaw, device)
    noised[..., _SIN_YAW] = sin_yaw * theta.cos() + cos_yaw * theta.sin()
    noised[..., _COS_YAW] = cos_yaw * theta.cos() - sin_yaw * theta.sin()

    noised[..., _VX : _VY + 1] += _signed_uniform(
        (num_groups, num_gt, 2), 0.0, noise_velocity, device
    )
    return noised


def build_denoising_attention_mask(
    num_query: int, num_groups: int, group_width: int, device: torch.device
) -> Bool[torch.Tensor, "num_all_queries num_all_queries"]:
    """Build the boolean isolation mask of a query set with a denoising tail.

    Regular queries see no group and groups see no other group, ``True`` blocks attention.

    Args:
        num_query: Number of regular queries ``N`` at the leading positions.
        num_groups: Number of denoising groups ``G``.
        group_width: Queries per denoising group.
        device: Device to allocate the mask on.

    Returns:
        Boolean mask ``(N + G * group_width, N + G * group_width)``.
    """
    # Group of every position, -1 for the regular queries
    tail = torch.arange(num_groups * group_width, device=device) // max(group_width, 1)
    group = torch.cat([torch.full((num_query,), -1, device=device), tail])
    return (group[None, :] >= 0) & (group[:, None] != group[None, :])


@torch.no_grad()
def build_denoising_queries(
    gt_boxes: Sequence[Float[torch.Tensor, "_num_gt_boxes box_dim"]],
    gt_labels: Sequence[Integer[torch.Tensor, " _num_gt_boxes"]],
    *,
    num_query: int,
    num_classes: int,
    query_budget: int,
    reg_weights: Float[torch.Tensor, " anchor_dim"],
    noise_center: float,
    noise_center_neg: float,
    noise_size: float,
    noise_yaw: float,
    noise_velocity: float,
) -> DenoisingQueries:
    """Build noised ground truth queries with their targets fixed at construction.

    Groups follow the DINO budget ``max(query_budget // (2 * max_gt), 1)``. A batch without
    boxes gets an empty group, so every training step reports the same denoising terms.

    Args:
        gt_boxes: Ground truth boxes ``(M, >=9)`` of every sample.
        gt_labels: Ground truth labels ``(M,)`` of every sample.
        num_query: Number of regular queries before the denoising tail.
        num_classes: Number of detection classes.
        query_budget: Approximate denoising query count per sample.
        reg_weights: Regression weight of every anchor channel ``(10,)``, it
            also gives the device and the type of every built tensor.
        noise_center: Upper bound of the positive center offset magnitude, as
            a fraction of the box half extent.
        noise_center_neg: Upper bound of the negative center offset magnitude,
            negatives are displaced by ``U(noise_center, noise_center_neg)``.
        noise_size: Half range of the uniform log size jitter.
        noise_yaw: Half range of the uniform heading rotation in radians.
        noise_velocity: Half range of the uniform velocity jitter in m/s.

    Returns:
        The built queries.
    """
    max_gt = max(boxes.shape[0] for boxes in gt_boxes)
    device = reg_weights.device
    group_width = 2 * max_gt
    num_groups = max(query_budget // group_width, 1) if group_width else 1
    total = num_groups * group_width
    batch_size = len(gt_boxes)

    anchor = reg_weights.new_zeros((batch_size, total, ANCHOR_DIM))
    cls_targets = reg_weights.new_zeros((batch_size, total, num_classes))
    box_targets = reg_weights.new_zeros((batch_size, total, ANCHOR_DIM))
    box_weights = reg_weights.new_zeros((batch_size, total, ANCHOR_DIM))
    cls_weights = reg_weights.new_zeros((batch_size, total))

    group_offsets = torch.arange(num_groups, device=device) * group_width
    for sample_index, (boxes, labels) in enumerate(zip(gt_boxes, gt_labels)):
        if boxes.shape[0] == 0:
            continue
        ground_truth = encode_ground_truth(boxes, labels, device, reg_weights.dtype)
        num_gt = ground_truth.anchor.shape[0]

        positive = _noise_denoising_anchors(
            ground_truth.anchor,
            num_groups,
            center_low=0.0,
            center_high=noise_center,
            noise_size=noise_size,
            noise_yaw=noise_yaw,
            noise_velocity=noise_velocity,
        )
        negative = _noise_denoising_anchors(
            ground_truth.anchor,
            num_groups,
            center_low=noise_center,
            center_high=noise_center_neg,
            noise_size=noise_size,
            noise_yaw=noise_yaw,
            noise_velocity=noise_velocity,
        )

        pos_index = (group_offsets[:, None] + torch.arange(num_gt, device=device)[None]).reshape(-1)
        neg_index = pos_index + max_gt
        anchor[sample_index, pos_index] = positive.reshape(-1, ANCHOR_DIM)
        anchor[sample_index, neg_index] = negative.reshape(-1, ANCHOR_DIM)
        cls_targets[sample_index, pos_index, ground_truth.labels.repeat(num_groups)] = 1.0
        box_targets[sample_index, pos_index] = ground_truth.anchor.repeat(num_groups, 1)
        box_weights[sample_index, pos_index] = (
            reg_weights * ground_truth.channel_valid.to(reg_weights.dtype)
        ).repeat(num_groups, 1)
        cls_weights[sample_index, pos_index] = 1.0
        cls_weights[sample_index, neg_index] = 1.0

    attn_mask = build_denoising_attention_mask(num_query, num_groups, group_width, device)
    return DenoisingQueries(
        anchor=anchor,
        cls_targets=cls_targets,
        box_targets=box_targets,
        box_weights=box_weights,
        cls_weights=cls_weights,
        attn_mask=attn_mask,
    )


class _DecoderOutputs(NamedTuple):
    """Per layer outputs of the decoder and its final state.

    Attributes:
        classification: Class logits of every layer over the regular and the denoising queries.
        prediction: Anchor states of every layer over the same queries.
        quality: Quality logits of every layer over the same queries.
        instance_feature: Final instance features of the regular queries.
        anchor: Final anchors of the regular queries.
        cls_logits: Final class logits of the regular queries.
    """

    classification: list[Float[torch.Tensor, "batch_size num_all_queries num_classes"]]
    prediction: list[Float[torch.Tensor, "batch_size num_all_queries anchor_dim"]]
    quality: list[Float[torch.Tensor, "batch_size num_all_queries 2"]]
    instance_feature: Float[torch.Tensor, "batch_size num_queries embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"]
    cls_logits: Float[torch.Tensor, "batch_size num_queries num_classes"]


class QualityLosses(NamedTuple):
    """Quality losses of the regressed queries of one layer.

    Attributes:
        centerness: Centerness loss.
        yawness: Yawness loss.
    """

    centerness: Float[torch.Tensor, ""]
    yawness: Float[torch.Tensor, ""]


class Sparse4DHead(nn.Module):
    """Temporal Sparse4D query head sampling the sparse PTv3 encoder stages.

    Every block attends to the temporal memory and to the other queries, samples the stages at
    keypoints of its anchors, runs an FFN and refines the anchors. Training needs scene continuous
    batch slots, a ``StreamingDataModule`` composed with
    ``/datamodule/streaming@datamodule: default_streaming_datamodule``.
    """

    def __init__(
        self,
        *,
        num_classes: int,
        embed_dims: int,
        anchor_recipe: AnchorRecipe,
        num_anchor: int,
        point_cloud_range: Sequence[float],
        grid_size: float,
        feature_strides: Sequence[int],
        num_temp_instances: int,
        min_observed_points: int,
        track_birth_score: float,
        num_decoder: int = 6,
        num_heads: int = 8,
        num_groups: int = 8,
        feedforward_channels: int = 512,
        dropout: float = 0.1,
        fix_scale: Sequence[Sequence[float]] = (
            (0.0, 0.0, 0.0),
            (0.45, 0.0, 0.0),
            (-0.45, 0.0, 0.0),
            (0.0, 0.45, 0.0),
            (0.0, -0.45, 0.0),
            (0.0, 0.0, 0.45),
            (0.0, 0.0, -0.45),
        ),
        num_learnable_pts: int = 6,
        attn_drop: float = 0.15,
        reg_weights: Sequence[float] = (2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        match_reg_weights: Sequence[float] = (2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
        cls_cost_weight: float = 2.0,
        box_cost_weight: float = 0.25,
        loss_cls_weight: float = 2.0,
        loss_box_weight: float = 0.25,
        loss_cns_weight: float = 1.0,
        loss_yns_weight: float = 1.0,
        cls_threshold_to_reg: float = 0.05,
        num_output: int = 300,
        default_time_interval: float = 0.1,
        min_time_interval: float = 0.05,
        max_time_interval: float = 2.0,
        confidence_decay: float = 0.36,
        dn_query_budget: int = 0,
        dn_noise_center: float = 0.5,
        dn_noise_center_neg: float = 1.0,
        dn_noise_size: float = 0.2,
        dn_noise_yaw: float = 0.39,
        dn_noise_velocity: float = 1.0,
        lidar_sampling: LidarSampling | str = LidarSampling.SPARSE,
        bev: PTv3StageBEVPyramid | None = None,
    ) -> None:
        """Initialize the Sparse4D head.

        Args:
            num_classes: Number of detection classes.
            embed_dims: Instance feature dimension, the feature width of the
                sampled stage.
            anchor_recipe: Recipe of the k-means anchors ``(num_anchor, 10)``.
            num_anchor: Number of instances per frame.
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            feature_strides: Total pooling strides of the sampled stages, from the finest to
                the coarsest.
            num_temp_instances: Instances propagated across frames.
            min_observed_points: Points an instance needs in one frame of a stream to become a
                target for the rest of the stream.
            track_birth_score: Score from which a query without a track starts a new one.
            num_decoder: Number of decoder blocks.
            num_heads: Attention heads of the instance attention.
            num_groups: Feature groups of the deformable aggregation.
            feedforward_channels: Hidden width of the FFN blocks.
            dropout: Dropout probability in the attention and FFN blocks.
            fix_scale: Fixed keypoint offsets as fractions of the box size.
            num_learnable_pts: Keypoints per anchor that the instance feature places.
            attn_drop: Keypoint dropout of the aggregation in training.
            reg_weights: Regression loss weight of every anchor channel.
            match_reg_weights: Matching cost weight of every anchor channel.
            cls_cost_weight: Classification cost weight of the matching.
            box_cost_weight: Regression cost weight of the matching.
            loss_cls_weight: Classification loss weight.
            loss_box_weight: Regression loss weight.
            loss_cns_weight: Centerness loss weight.
            loss_yns_weight: Yawness loss weight.
            cls_threshold_to_reg: Matched queries whose best class score is at
                most this value get no regression and quality loss. They still
                count as positives.
            num_output: Largest number of decoded boxes per sample.
            default_time_interval: Time interval in seconds without history.
            min_time_interval: Smallest frame gap that keeps the state of a slot.
            max_time_interval: Largest frame gap that keeps the state of a slot.
            confidence_decay: Decay of the cached instance confidence per second.
            dn_query_budget: Approximate number of denoising queries appended
                per sample in training steps, ``0`` turns denoising off.
            dn_noise_center: Upper bound of the positive center offset
                magnitude, as a fraction of the box half extent.
            dn_noise_center_neg: Upper bound of the negative center offset,
                negatives are displaced by ``U(dn_noise_center, dn_noise_center_neg)``.
            dn_noise_size: Half range of the uniform log size jitter.
            dn_noise_yaw: Half range of the uniform heading rotation in radians.
            dn_noise_velocity: Half range of the uniform velocity jitter in m/s.
            lidar_sampling: Lidar features the decoder blocks sample, a ``LidarSampling`` or its
                string value.
            bev: Builds the BEV levels of a frame from the stages, exactly for BEV sampling.

        Raises:
            ValueError: If the lidar sampling is unknown, the weights do not cover the anchor
                channels or the denoising or the BEV settings do not go together.
        """
        super().__init__()
        lidar_sampling = LidarSampling(lidar_sampling)
        if (bev is None) != (lidar_sampling == LidarSampling.SPARSE):
            raise ValueError(
                f"lidar_sampling {lidar_sampling} needs BEV levels exactly with BEV maps."
            )
        if len(reg_weights) != ANCHOR_DIM or len(match_reg_weights) != ANCHOR_DIM:
            raise ValueError(
                f"reg_weights and match_reg_weights need {ANCHOR_DIM} entries, got "
                f"{len(reg_weights)} and {len(match_reg_weights)}."
            )
        if dn_query_budget < 0:
            raise ValueError(f"dn_query_budget must be non-negative, got {dn_query_budget}.")
        if dn_query_budget > 0 and not 0.0 < dn_noise_center < dn_noise_center_neg:
            raise ValueError(
                f"Denoising needs 0 < dn_noise_center < dn_noise_center_neg, got "
                f"{dn_noise_center} and {dn_noise_center_neg}."
            )
        if min(dn_noise_size, dn_noise_yaw, dn_noise_velocity) < 0.0:
            raise ValueError("Denoising noise half ranges must be non-negative.")
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_decoder = num_decoder
        self.bev = bev
        self.sample_sparse = lidar_sampling != LidarSampling.BEV
        self.cls_cost = ClassificationCost(weight=cls_cost_weight)
        self.box_cost_weight = box_cost_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_box_weight = loss_box_weight
        self.loss_cns_weight = loss_cns_weight
        self.loss_yns_weight = loss_yns_weight
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.num_output = num_output
        self.dn_query_budget = dn_query_budget
        self.dn_noise_center = dn_noise_center
        self.dn_noise_center_neg = dn_noise_center_neg
        self.dn_noise_size = dn_noise_size
        self.dn_noise_yaw = dn_noise_yaw
        self.dn_noise_velocity = dn_noise_velocity
        self.reg_weights: Float[torch.Tensor, " anchor_dim"]
        self.register_buffer(
            "reg_weights", torch.tensor(list(reg_weights), dtype=torch.float32), persistent=False
        )
        self.match_reg_weights: Float[torch.Tensor, " anchor_dim"]
        self.register_buffer(
            "match_reg_weights",
            torch.tensor(list(match_reg_weights), dtype=torch.float32),
            persistent=False,
        )

        self.instance_bank = InstanceBank(
            num_anchor=num_anchor,
            embed_dims=embed_dims,
            anchor_recipe=anchor_recipe,
            num_temp_instances=num_temp_instances,
            default_time_interval=default_time_interval,
            min_time_interval=min_time_interval,
            max_time_interval=max_time_interval,
            confidence_decay=confidence_decay,
        )
        self.tracker = InstanceTracker(num_temp_instances, min_observed_points, track_birth_score)
        self.anchor_encoder = SparseBox3DEncoder(embed_dims)
        self.gnn = nn.ModuleList(
            [
                DecoupledInstanceAttention(embed_dims, num_heads, dropout)
                for _ in range(num_decoder - 1)
            ]
        )
        self.gnn_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_decoder - 1)])
        self.temp_gnn = nn.ModuleList(
            [
                DecoupledInstanceAttention(embed_dims, num_heads, dropout)
                for _ in range(num_decoder - 1)
            ]
        )
        self.temp_gnn_norm = nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(num_decoder - 1)]
        )
        self.level_grids = nn.ModuleList(
            [SparseLevelGrid(point_cloud_range, grid_size, stride) for stride in feature_strides]
        )
        self.deformable = nn.ModuleList(
            [
                SparseDeformableAggregation(
                    embed_dims=embed_dims,
                    num_groups=num_groups,
                    num_levels=len(feature_strides),
                    kps_generator=SparseBox3DKeyPointsGenerator(
                        embed_dims=embed_dims,
                        fix_scale=fix_scale,
                        num_learnable_pts=num_learnable_pts,
                    ),
                    proj_drop=dropout,
                    attn_drop=attn_drop,
                    bev_aggregation=None
                    if bev is None
                    else BEVDeformableAggregation(
                        embed_dims=embed_dims,
                        bev_channels=bev.channels,
                        num_groups=num_groups,
                        num_levels=bev.num_levels,
                        num_points=len(fix_scale) + num_learnable_pts,
                        attn_drop=attn_drop,
                    ),
                    sample_sparse=self.sample_sparse,
                )
                for _ in range(num_decoder)
            ]
        )
        self.deformable_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_decoder)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dims, feedforward_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(feedforward_channels, embed_dims),
                    nn.Dropout(dropout),
                )
                for _ in range(num_decoder)
            ]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_decoder)])
        self.refine = nn.ModuleList(
            [
                SparseBox3DRefinementModule(embed_dims=embed_dims, num_classes=num_classes)
                for _ in range(num_decoder)
            ]
        )
        self.loss_cls = SigmoidFocalLoss()
        self.loss_yns = GaussianFocalLoss()

    def reset(self) -> None:
        """Forget the cached instances and the tracks of every lane."""
        self.instance_bank.reset()
        self.tracker.reset()

    def levels(self, stages: Sequence[SparseStage]) -> list[SparseLevel]:
        """Look up the voxels of every sampled stage once per frame, none for a BEV only head."""
        if not self.sample_sparse:
            return []
        return [
            grid(stage.feat, stage.grid_coord, stage.offset)
            for grid, stage in zip(self.level_grids, stages, strict=True)
        ]

    def bev_levels(self, stages: Sequence[SparseStage]) -> list[BEVLevel]:
        """Build the BEV maps of the stages once per frame, none for a head without them."""
        return [] if self.bev is None else self.bev(stages)

    def _decode(
        self,
        levels: Sequence[SparseLevel],
        state: BankState,
        denoising: DenoisingQueries | None,
        bev_levels: Sequence[BEVLevel],
    ) -> _DecoderOutputs:
        """Run the decoder blocks over the sampled sparse encoder stages.

        The temporal instances join the working set after block 0. Denoising queries ride at the
        tail and never reach the merge or the final state.

        Args:
            levels: Sampled stages with their voxel lookup.
            state: Fresh instances of the frame and the cached ones warped into it.
            denoising: Optional denoising queries appended at the tail with
                zero features and the isolation mask on the instance attention.
            bev_levels: BEV maps of a head with a BEV branch.

        Returns:
            The per layer outputs over the regular and the denoising queries
            and the final state of the regular queries.
        """
        instance_feature, anchor = state.instance_feature, state.anchor
        num_query = instance_feature.shape[1]
        attn_mask = None
        if denoising is not None:
            dn_feature = instance_feature.new_zeros(
                (instance_feature.shape[0], denoising.anchor.shape[1], instance_feature.shape[-1])
            )
            instance_feature = torch.cat([instance_feature, dn_feature], dim=1)
            anchor = torch.cat([anchor, denoising.anchor], dim=1)
            attn_mask = denoising.attn_mask
        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = self.anchor_encoder(state.temp_anchor)
        classification: list[Float[torch.Tensor, "batch_size num_all_queries num_classes"]] = []
        prediction: list[Float[torch.Tensor, "batch_size num_all_queries anchor_dim"]] = []
        quality: list[Float[torch.Tensor, "batch_size num_all_queries 2"]] = []
        for block_index in range(self.num_decoder):
            if block_index > 0:
                instance_feature = self.temp_gnn[block_index - 1](
                    instance_feature, anchor_embed, state.temp_feature, temp_anchor_embed
                )
                instance_feature = self.temp_gnn_norm[block_index - 1](instance_feature)
                instance_feature = self.gnn[block_index - 1](
                    instance_feature, anchor_embed, attn_mask=attn_mask
                )
                instance_feature = self.gnn_norm[block_index - 1](instance_feature)
            instance_feature = self.deformable[block_index](
                instance_feature, anchor, anchor_embed, levels, bev_levels
            )
            instance_feature = self.deformable_norm[block_index](instance_feature)
            instance_feature = self.ffn_norm[block_index](
                instance_feature + self.ffn[block_index](instance_feature)
            )
            refined = self.refine[block_index](
                instance_feature, anchor, anchor_embed, state.time_interval
            )
            anchor = refined.anchor
            classification.append(refined.cls_logits)
            prediction.append(anchor)
            quality.append(refined.quality)
            if block_index == 0:
                merged = InstanceBank.merge_temporal(
                    instance_feature[:, :num_query],
                    anchor[:, :num_query],
                    refined.cls_logits[:, :num_query],
                    state.temp_feature,
                    state.temp_anchor,
                    state.temp_valid,
                    self.instance_bank.num_temp_instances,
                )
                instance_feature = torch.cat(
                    [merged.instance_feature, instance_feature[:, num_query:]], dim=1
                )
                anchor = torch.cat([merged.anchor, anchor[:, num_query:]], dim=1)
            if block_index != self.num_decoder - 1:
                anchor_embed = self.anchor_encoder(anchor)
                # Read the temporal memory at the refined anchors
                if block_index > 0:
                    temp_anchor_embed = anchor_embed[:, : temp_anchor_embed.shape[1]]
        return _DecoderOutputs(
            classification=classification,
            prediction=prediction,
            quality=quality,
            instance_feature=instance_feature[:, :num_query],
            anchor=anchor[:, :num_query],
            cls_logits=classification[-1][:, :num_query],
        )

    def forward(
        self,
        stages: Sequence[SparseStage],
        timestamps_seconds: Float[torch.Tensor, " batch_size"],
        lidar2global: Float[torch.Tensor, "batch_size 4 4"],
        scene_tokens: Sequence[str],
        gt_detections: Detection3DGTBatch | None = None,
    ) -> Sparse4DHeadOutputs:
        """Run the decoder over the sampled sparse encoder stages and advance the tracks.

        With the ground truth the head records which boxes were observed in the stream so far,
        only those feed the denoising queries and the loss.

        Args:
            stages: Fused stages from the finest to the coarsest.
            timestamps_seconds: Frame timestamps ``(B,)`` in seconds.
            lidar2global: Poses ``(B, 4, 4)`` with the augmentations folded in.
            scene_tokens: Scene identifiers of the frames.
            gt_detections: Detection ground truth of the batch, to train and evaluate.

        Returns:
            Per layer outputs of the regular queries and, in training steps
            with denoising, of the denoising queries.

        Raises:
            ValueError: If a denoising training step misses the ground truth.
        """
        if self.training and self.dn_query_budget > 0 and gt_detections is None:
            raise ValueError("Denoising training steps need the ground truth boxes.")
        batch_size = stages[0].offset.shape[0]
        state = self.instance_bank.get(batch_size, timestamps_seconds, lidar2global, scene_tokens)
        num_query = state.anchor.shape[1]
        self.tracker.begin_frame(
            self.instance_bank.lane_continues, self.instance_bank.carried_query_index, num_query
        )
        denoising = None
        if gt_detections is not None:
            observed = self.tracker.observe(
                gt_detections.valid_bboxes_instance_ids(), gt_detections.valid_bboxes_num_points()
            )
            if self.training and self.dn_query_budget > 0:
                denoising = build_denoising_queries(
                    [
                        boxes[mask]
                        for boxes, mask in zip(
                            gt_detections.valid_bboxes_3d(), observed, strict=True
                        )
                    ],
                    [
                        labels[mask]
                        for labels, mask in zip(
                            gt_detections.valid_labels_3d(), observed, strict=True
                        )
                    ],
                    num_query=num_query,
                    num_classes=self.num_classes,
                    query_budget=self.dn_query_budget,
                    reg_weights=self.reg_weights,
                    noise_center=self.dn_noise_center,
                    noise_center_neg=self.dn_noise_center_neg,
                    noise_size=self.dn_noise_size,
                    noise_yaw=self.dn_noise_yaw,
                    noise_velocity=self.dn_noise_velocity,
                )

        decoded = self._decode(self.levels(stages), state, denoising, self.bev_levels(stages))
        self.instance_bank.cache(
            decoded.instance_feature, decoded.anchor, decoded.cls_logits, lidar2global
        )

        classification = torch.stack(decoded.classification, dim=1)
        prediction = torch.stack(decoded.prediction, dim=1)
        quality = torch.stack(decoded.quality, dim=1)
        scores = self._query_scores(classification[:, -1, :num_query], quality[:, -1, :num_query])
        instance_ids = self.tracker.assign_track_ids(scores.detach().max(dim=-1).values)
        return Sparse4DHeadOutputs(
            classification=classification[:, :, :num_query],
            prediction=prediction[:, :, :num_query],
            quality=quality[:, :, :num_query],
            instance_ids=instance_ids,
            instance_feature=decoded.instance_feature,
            denoising=None
            if denoising is None
            else Sparse4DDenoisingOutputs(
                classification=classification[:, :, num_query:],
                prediction=prediction[:, :, num_query:],
                cls_targets=denoising.cls_targets,
                box_targets=denoising.box_targets,
                box_weights=denoising.box_weights,
                cls_weights=denoising.cls_weights,
            ),
        )

    def forward_temporal_export(
        self, stages: Sequence[SparseStage], state: TemporalStateInputs
    ) -> Sparse4DFrameOutputs:
        """Run the decoder with the bank state at the graph boundary.

        A frame's ``*_out`` state feeds the next one. The query index of every state slot lets
        the runtime carry its track ids into the slots.

        Args:
            stages: Fused stages from the finest to the coarsest.
            state: Bank state of the last frame.

        Returns:
            The last layer outputs and the state of the next frame.
        """
        bank = self.instance_bank
        fresh = bank.fresh_instances(stages[0].offset.shape[0])
        gated = bank.gate_temporal_state(
            state.temp_instance_feature,
            state.temp_anchor,
            state.temp_confidence,
            state.temp_valid,
            state.time_interval,
            state.prev2current,
        )
        decoded = self._decode(
            self.levels(stages),
            BankState(
                instance_feature=fresh.instance_feature,
                anchor=fresh.anchor,
                time_interval=state.time_interval,
                temp_feature=gated.feature,
                temp_anchor=gated.anchor,
                temp_valid=state.temp_valid,
            ),
            None,
            self.bev_levels(stages),
        )
        selection = bank.select_temporal_state(
            decoded.instance_feature,
            decoded.anchor,
            decoded.cls_logits,
            gated.confidence,
            bank.confidence_decay,
            state.time_interval,
            bank.num_temp_instances,
        )
        return Sparse4DFrameOutputs(
            classification=decoded.classification[-1],
            prediction=decoded.prediction[-1],
            quality=decoded.quality[-1],
            instance_feature=decoded.instance_feature,
            temp_instance_feature_out=selection.feature,
            temp_anchor_out=selection.anchor,
            temp_confidence_out=selection.confidence,
            temp_query_index_out=selection.query_index,
        )

    def temporal_state_export_inputs(self, device: torch.device) -> TemporalStateInputs:
        """Return the reset state of one lane, the sample inputs of the temporal export."""
        bank = self.instance_bank
        num_temp = bank.num_temp_instances
        return TemporalStateInputs(
            temp_instance_feature=torch.zeros(1, num_temp, self.embed_dims, device=device),
            temp_anchor=torch.zeros(1, num_temp, ANCHOR_DIM, device=device),
            temp_confidence=torch.zeros(1, num_temp, device=device),
            temp_valid=torch.zeros(1, device=device),
            time_interval=torch.full((1,), bank.default_time_interval, device=device),
            prev2current=torch.eye(4, device=device).unsqueeze(0),
        )

    def prepare_for_export(self) -> Sparse4DHead:
        """Return an export ready copy in the fp16 range for :meth:`forward_temporal_export`."""
        head = deepcopy(self)
        attention = (*zip(head.gnn, head.gnn_norm), *zip(head.temp_gnn, head.temp_gnn_norm))
        for attention_block, norm in attention:
            attention_block.fold_output_scale(norm)
            attention_block.attn = ExportableMultiheadAttention(attention_block.attn)
        return head.eval()

    @torch.no_grad()
    def _assign_targets(
        self,
        cls_logits: Float[torch.Tensor, "batch_size num_queries num_classes"],
        pred_anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        ground_truth: Sequence[EncodedGroundTruth],
        instance_ids: Sequence[Int64[torch.Tensor, " _num_gt_boxes"]],
        slot_owners: Int64[torch.Tensor, "batch_size num_temp_instances"] | None,
    ) -> Sparse4DTargets:
        """Match the predictions of one layer to the ground truth one to one.

        A leading temporal slot that follows an instance keeps that instance as its target, or
        gets background when the instance is gone. The other queries are Hungarian matched to the
        instances no slot follows.

        Args:
            cls_logits: Class logits ``(B, N, num_classes)``.
            pred_anchor: Predicted anchor states ``(B, N, 10)``.
            ground_truth: Encoded ground truth of every sample.
            instance_ids: Instance id of every ground truth box of every sample.
            slot_owners: Instance each leading temporal slot follows ``(B, K)``, -1 for free
                slots, or None to match every query freely.

        Returns:
            Stacked classification and regression targets.

        Raises:
            ValueError: If the matching cost is not finite.
        """
        batch_size, num_anchor = cls_logits.shape[:2]
        device = cls_logits.device
        cls_targets = cls_logits.new_zeros((batch_size, num_anchor, self.num_classes))
        box_targets = pred_anchor.new_zeros((batch_size, num_anchor, ANCHOR_DIM))
        box_weights = pred_anchor.new_zeros((batch_size, num_anchor, ANCHOR_DIM))
        matched = torch.zeros((batch_size, num_anchor), dtype=torch.bool, device=device)
        assigned_instance = torch.full(
            (batch_size, num_anchor), -1, dtype=torch.long, device=device
        )
        num_pos = 0

        for sample_index, (sample_gt, sample_ids) in enumerate(
            zip(ground_truth, instance_ids, strict=True)
        ):
            sample_ids = sample_ids.to(device)
            free_pred = torch.ones(num_anchor, dtype=torch.bool, device=device)
            free_gt = torch.ones(sample_ids.shape[0], dtype=torch.bool, device=device)
            fixed_pred = torch.zeros(0, dtype=torch.long, device=device)
            fixed_gt = torch.zeros(0, dtype=torch.long, device=device)
            if slot_owners is not None:
                owners = slot_owners[sample_index]
                # Instance ids are unique in a frame, so every slot follows at most one box
                fixed_pred, fixed_gt = (owners[:, None] == sample_ids[None, :]).nonzero(
                    as_tuple=True
                )
                free_pred[: owners.shape[0]] = owners < 0
                free_gt[fixed_gt] = False
            if sample_gt.anchor.shape[0] == 0:
                continue
            cls_cost = self.cls_cost(
                cls_logits[sample_index, None].float(),
                sample_gt.labels[None],
                torch.ones_like(sample_gt.labels, dtype=torch.bool)[None],
            )[0]

            box_delta = pred_anchor[sample_index].float()[:, None] - sample_gt.anchor[None]
            box_cost = (
                box_delta.abs() * self.match_reg_weights * sample_gt.channel_valid[None]
            ).sum(dim=-1)

            cost = cls_cost + self.box_cost_weight * box_cost
            if not bool(torch.isfinite(cost).all()):
                raise ValueError("The matching cost is not finite, the predictions diverged.")
            free_pred_indices = free_pred.nonzero().flatten()
            free_gt_indices = free_gt.nonzero().flatten()
            free_cost = cost[free_pred_indices][:, free_gt_indices]
            hungarian_pred, hungarian_gt = linear_sum_assignment(free_cost.cpu().numpy())
            pred_indices = torch.cat(
                [fixed_pred, free_pred_indices[torch.from_numpy(hungarian_pred).to(device)]]
            )
            gt_indices = torch.cat(
                [fixed_gt, free_gt_indices[torch.from_numpy(hungarian_gt).to(device)]]
            )

            cls_targets[sample_index, pred_indices, sample_gt.labels[gt_indices]] = 1.0
            box_targets[sample_index, pred_indices] = sample_gt.anchor[gt_indices]
            box_weights[sample_index, pred_indices] = self.reg_weights * sample_gt.channel_valid[
                gt_indices
            ].to(self.reg_weights.dtype)
            matched[sample_index, pred_indices] = True
            assigned_instance[sample_index, pred_indices] = sample_ids[gt_indices]
            num_pos += int(pred_indices.shape[0])

        return Sparse4DTargets(
            cls_targets=cls_targets,
            box_targets=box_targets,
            box_weights=box_weights,
            matched=matched,
            assigned_instance=assigned_instance,
            num_pos=num_pos,
        )

    def _quality_losses(
        self,
        quality: Float[torch.Tensor, "batch_size num_queries 2"],
        pred_anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
        targets: Sparse4DTargets,
        reg_mask: Bool[torch.Tensor, "batch_size num_queries"],
        avg_factor: float,
    ) -> QualityLosses:
        """Supervise the centerness and the yawness of the regressed queries.

        Args:
            quality: Quality logits ``(B, N, 2)``.
            pred_anchor: Predicted anchor states ``(B, N, 10)``.
            targets: Matching targets of the layer.
            reg_mask: Queries that get regression and quality losses ``(B, N)``.
            avg_factor: Normalization factor of the centerness and the yawness loss.

        Returns:
            The centerness and the yawness loss.
        """
        # Mask by weight, not indexing, so DDP always sees the quality layers
        quality = quality.float()
        weights = reg_mask.to(quality.dtype)
        anchor = pred_anchor.detach().float()
        center_distance = torch.linalg.vector_norm(
            anchor[..., _X : _Z + 1] - targets.box_targets[..., _X : _Z + 1], dim=-1
        )
        loss_cns = (
            F.binary_cross_entropy_with_logits(
                quality[..., _CNS], torch.exp(-center_distance), reduction="none"
            )
            * weights
        ).sum() / avg_factor
        yaw_agrees = (
            F.cosine_similarity(
                anchor[..., _SIN_YAW : _COS_YAW + 1],
                targets.box_targets[..., _SIN_YAW : _COS_YAW + 1],
                dim=-1,
            )
            > 0
        ).to(quality.dtype) * weights
        loss_yns = self.loss_yns(
            quality[..., _YNS], yaw_agrees, weights=weights, avg_factor=avg_factor
        )
        return QualityLosses(centerness=loss_cns, yawness=loss_yns)

    def assign(
        self,
        outputs: Sparse4DHeadOutputs,
        gt_boxes: Sequence[Float[torch.Tensor, "_num_gt_boxes box_dim"]],
        gt_labels: Sequence[Integer[torch.Tensor, " _num_gt_boxes"]],
        gt_instance_ids: Sequence[Int64[torch.Tensor, " _num_gt_boxes"]],
        lanes: Sequence[int],
    ) -> list[Sparse4DTargets]:
        """Match the queries of every layer to the ground truth one to one.

        The head keeps the boxes observed in the stream so far and lets its slots follow their
        instances from the second layer on, where the temporal slots lead the queries. The
        assignment leaves the stream state alone, see :meth:`follow_assignment`.

        Args:
            outputs: Raw head outputs from :meth:`forward` of the given lanes.
            gt_boxes: Ground truth boxes ``(M, >=9)`` of every sample.
            gt_labels: Ground truth labels ``(M,)`` of every sample.
            gt_instance_ids: Instance id ``(M,)`` of every ground truth box.
            lanes: Batch lane of every sample.

        Returns:
            The targets of every decoder layer, the last one also names the instance every
            query was matched to.
        """
        device = outputs.prediction.device
        masks = self.tracker.observed_masks()
        observed = [masks[lane] for lane in lanes]
        gt_boxes = [boxes[mask] for boxes, mask in zip(gt_boxes, observed, strict=True)]
        gt_labels = [labels[mask] for labels, mask in zip(gt_labels, observed, strict=True)]
        gt_instance_ids = [ids[mask] for ids, mask in zip(gt_instance_ids, observed, strict=True)]
        slot_owners = self.tracker.slot_owners(lanes)
        ground_truth = [
            encode_ground_truth(boxes, labels, device, torch.float32)
            for boxes, labels in zip(gt_boxes, gt_labels, strict=True)
        ]
        return [
            self._assign_targets(
                outputs.classification[:, layer_index],
                outputs.prediction[:, layer_index],
                ground_truth,
                gt_instance_ids,
                None if layer_index == 0 else slot_owners,
            )
            for layer_index in range(outputs.classification.shape[1])
        ]

    def follow_assignment(
        self, outputs: Sparse4DHeadOutputs, targets: Sequence[Sparse4DTargets], lanes: Sequence[int]
    ) -> None:
        """Let the temporal slots of the next frame follow the instances the last layer assigned.

        The model calls it once per step after :meth:`assign`, so computing a loss again changes
        no stream state.

        Args:
            outputs: Raw head outputs from :meth:`forward` of the given lanes.
            targets: Targets of every decoder layer from :meth:`assign`.
            lanes: Batch lane of every sample.
        """
        scores = self.query_scores(outputs).detach()
        self.tracker.set_query_owners(
            lanes, targets[-1].assigned_instance, scores.max(dim=-1).values
        )

    def loss_from_targets(
        self,
        outputs: Sparse4DHeadOutputs,
        targets: Sequence[Sparse4DTargets],
        lanes: Sequence[int],
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"] | None = None,
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Compute the matched losses of every layer from its targets and the denoising losses.

        Denoising queries are graded against their source box without matching.

        Args:
            outputs: Raw head outputs from :meth:`forward` of the given lanes.
            targets: Targets of every decoder layer from :meth:`assign`.
            lanes: Batch lane of every sample.
            gt_annotated_class_masks: Per sample and class mask ``(B, num_classes)``, True where
                the class is annotated in the sample. None supervises every class.

        Returns:
            Loss dictionary with the terms of every layer and their sum under ``loss``.
        """
        loss_dict: dict[str, Float[torch.Tensor, ""]] = {}
        num_layers = outputs.classification.shape[1]
        class_weights = self._class_weights(
            len(lanes), gt_annotated_class_masks, outputs.classification.device
        )
        total_pos = 0
        for layer_index, layer_targets in enumerate(targets):
            cls_logits = outputs.classification[:, layer_index]
            pred_anchor = outputs.prediction[:, layer_index]
            total_pos += layer_targets.num_pos
            avg_factor = float(
                reduce_mean_count(cls_logits.new_tensor(float(layer_targets.num_pos)))
            )

            # A matched query is vouched for by its box, the others skip the unannotated classes
            cls_weights = torch.where(layer_targets.matched[..., None], 1.0, class_weights[:, None])
            loss_cls = self.loss_cls(
                cls_logits.flatten(end_dim=1).float(),
                layer_targets.cls_targets.flatten(end_dim=1),
                weights=cls_weights.flatten(end_dim=1),
                avg_factor=avg_factor,
            )
            confident = cls_logits.detach().max(dim=-1).values.sigmoid() > self.cls_threshold_to_reg
            reg_mask = layer_targets.matched & confident
            box_error = (
                pred_anchor.float() - layer_targets.box_targets
            ).abs() * layer_targets.box_weights
            loss_box = (box_error * reg_mask[..., None]).sum() / avg_factor
            quality_losses = self._quality_losses(
                outputs.quality[:, layer_index], pred_anchor, layer_targets, reg_mask, avg_factor
            )

            prefix = "layer_-1" if layer_index == num_layers - 1 else f"layer_{layer_index}"
            loss_dict[f"{prefix}_loss_cls"] = self.loss_cls_weight * loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = self.loss_box_weight * loss_box
            loss_dict[f"{prefix}_loss_cns"] = self.loss_cns_weight * quality_losses.centerness
            loss_dict[f"{prefix}_loss_yns"] = self.loss_yns_weight * quality_losses.yawness

        loss_dict["num_followed"] = outputs.classification.new_tensor(
            float((self.tracker.slot_owners(lanes) >= 0).sum())
        )
        loss_dict["num_pos"] = outputs.classification.new_tensor(total_pos / max(num_layers, 1))
        if outputs.denoising is not None:
            loss_dict.update(self._denoising_losses(outputs.denoising, class_weights))
        loss_dict["loss"] = sum(value for key, value in loss_dict.items() if "loss" in key)
        return loss_dict

    def _class_weights(
        self,
        batch_size: int,
        gt_annotated_class_masks: Bool[torch.Tensor, "batch_size num_classes"] | None,
        device: torch.device,
    ) -> Float[torch.Tensor, "batch_size num_classes"]:
        """Classification weights of every sample and class, zero for the unannotated classes.

        Args:
            batch_size: Number of samples.
            gt_annotated_class_masks: Per sample and class mask ``(B, num_classes)``, True where
                the class is annotated in the sample. None supervises every class.
            device: Device of the weights.

        Returns:
            Weights ``(B, num_classes)``, one for the annotated classes and zero for the others.

        Raises:
            ValueError: If the mask does not cover every sample and class.
        """
        if gt_annotated_class_masks is None:
            return torch.ones((batch_size, self.num_classes), device=device)
        if tuple(gt_annotated_class_masks.shape) != (batch_size, self.num_classes):
            raise ValueError(
                f"gt_annotated_class_masks must have shape ({batch_size}, {self.num_classes}), "
                f"got {tuple(gt_annotated_class_masks.shape)}."
            )
        return gt_annotated_class_masks.to(device=device, dtype=torch.float32)

    def _denoising_losses(
        self,
        denoising: Sparse4DDenoisingOutputs,
        class_weights: Float[torch.Tensor, "batch_size num_classes"],
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Compute the classification and regression losses of the denoising queries.

        Args:
            denoising: Denoising outputs and targets of the batch.
            class_weights: Classification weights of every sample and class ``(B, num_classes)``.

        Returns:
            Denoising loss terms of every layer and the denoising positive count.
        """
        loss_dict: dict[str, Float[torch.Tensor, ""]] = {}
        num_layers = denoising.classification.shape[1]
        positive = denoising.cls_targets.sum(dim=-1) > 0
        dn_avg_factor = float(reduce_mean_count(positive.sum().float()))
        # A positive is vouched for by its source box, a negative skips the unannotated classes
        cls_weights = torch.where(positive[..., None], 1.0, class_weights[:, None])
        cls_weights = (cls_weights * denoising.cls_weights[..., None]).flatten(end_dim=1)
        for layer_index in range(num_layers):
            dn_loss_cls = self.loss_cls(
                denoising.classification[:, layer_index].flatten(end_dim=1).float(),
                denoising.cls_targets.flatten(end_dim=1),
                weights=cls_weights,
                avg_factor=dn_avg_factor,
            )
            dn_box_error = (
                denoising.prediction[:, layer_index].float() - denoising.box_targets
            ).abs() * denoising.box_weights

            prefix = "layer_-1" if layer_index == num_layers - 1 else f"layer_{layer_index}"
            loss_dict[f"{prefix}_loss_dn_cls"] = self.loss_cls_weight * dn_loss_cls
            loss_dict[f"{prefix}_loss_dn_bbox"] = (
                self.loss_box_weight * dn_box_error.sum() / dn_avg_factor
            )
        loss_dict["dn_num_pos"] = denoising.classification.new_tensor(dn_avg_factor)
        return loss_dict

    def query_scores(
        self, outputs: Sparse4DHeadOutputs
    ) -> Float[torch.Tensor, "batch_size num_queries num_classes"]:
        """Detection score of every query and class of the last layer.

        Args:
            outputs: Raw head outputs from :meth:`forward`.

        Returns:
            The class probabilities weighted by the centerness.
        """
        return self._query_scores(outputs.classification[:, -1], outputs.quality[:, -1])

    @staticmethod
    def _query_scores(
        cls_logits: Float[torch.Tensor, "batch_size num_queries num_classes"],
        quality: Float[torch.Tensor, "batch_size num_queries 2"],
    ) -> Float[torch.Tensor, "batch_size num_queries num_classes"]:
        """Class probability times centerness of every query and class."""
        return cls_logits.float().sigmoid() * quality[..., _CNS].float().sigmoid()[..., None]

    def decode_outputs(self, outputs: Detection3DHeadOutputs) -> ModelPredictions:
        """Decode the last layer into metric space boxes, as the deployed runtime does.

        Every query gives one box of its best class, scored by the class probability times the
        centerness. The top ``num_output`` boxes stay, each carrying the track id of its query.
        Nothing is suppressed, the one-to-one matching of training leaves one query per object.

        Args:
            outputs: Raw head outputs from :meth:`forward`.

        Returns:
            One Detection3DSamplePredictions per sample of the batch, wrapped in ModelPredictions.
        """
        outputs = outputs.sparse4d_head()
        # Mixed precision evaluation hands over float16 outputs, the boxes decode in float32
        # The centerness is per query, so the best class is the one with the largest logit
        scores, labels = self.query_scores(outputs).max(dim=-1)
        pred_anchor = outputs.prediction[:, -1].float()

        results = []
        for sample_scores, sample_labels, sample_anchor, sample_ids in zip(
            scores, labels, pred_anchor, outputs.instance_ids, strict=True
        ):
            keep_count = min(self.num_output, sample_scores.numel())
            top_scores, top_indices = sample_scores.topk(keep_count)
            results.append(
                Detection3DSamplePredictions(
                    bboxes_3d=denormalize_boxes3d(sample_anchor[top_indices]),
                    scores_3d=top_scores,
                    labels_3d=sample_labels[top_indices],
                    instance_ids=sample_ids[top_indices],
                )
            )
        return ModelPredictions(detection3d_predictions=results)
