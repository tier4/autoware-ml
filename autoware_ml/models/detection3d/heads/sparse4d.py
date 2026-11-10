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
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DDenoisingOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.dataclasses.models.model_predictions import ModelPredictions
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.models.detection3d.task_modules.nms_groups import (
    NMSGroupConfig,
    filter_nms_groups,
    resolve_nms_groups,
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
from autoware_ml.utils.point_cloud.batching import offset_to_batch

ANCHOR_DIM = 10
_X, _Y, _Z, _LN_L, _LN_W, _LN_H, _SIN_YAW, _COS_YAW, _VX, _VY = range(ANCHOR_DIM)
# Columns of the quality estimate
_CNS, _YNS = range(2)

# Instance bank state at the temporal export graph boundary
TEMPORAL_STATE_INPUT_NAMES = (
    "temp_instance_feature",
    "temp_anchor",
    "temp_confidence",
    "temp_valid",
    "time_interval",
    "prev2current",
)
TEMPORAL_STATE_OUTPUT_NAMES = (
    "temp_instance_feature_out",
    "temp_anchor_out",
    "temp_confidence_out",
)
# Query every state slot was selected from, the runtime carries its track ids into the slots
TEMPORAL_QUERY_INDEX_OUTPUT_NAME = "temp_query_index_out"


def _det3x3(matrix: torch.Tensor) -> torch.Tensor:
    """Batched 3x3 determinant from elementary ops, which ONNX and TensorRT support."""
    a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    d, e, f = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    g, h, i = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def encode_boxes_to_anchor_space(boxes: torch.Tensor) -> torch.Tensor:
    """Encode repository layout boxes into the anchor state layout.

    Args:
        boxes: Boxes ``(N, >=9)`` led by ``[cx, cy, cz, l, w, h, yaw, vx, vy]``.

    Returns:
        Anchor states ``(N, 10)`` as
        ``[cx, cy, cz, ln l, ln w, ln h, sin yaw, cos yaw, vx, vy]``.
    """
    encoded = boxes.new_zeros((boxes.shape[0], ANCHOR_DIM))
    encoded[:, _X : _Z + 1] = boxes[:, 0:3]
    encoded[:, _LN_L : _LN_H + 1] = boxes[:, 3:6].log()
    encoded[:, _SIN_YAW] = torch.sin(boxes[:, 6])
    encoded[:, _COS_YAW] = torch.cos(boxes[:, 6])
    encoded[:, _VX : _VY + 1] = boxes[:, 7:9]
    return encoded


def decode_anchor_space_to_boxes(anchor: torch.Tensor) -> torch.Tensor:
    """Decode anchor states back into repository layout boxes.

    Args:
        anchor: Anchor states ``(..., 10)``.

    Returns:
        Boxes ``(..., 9)`` as ``[cx, cy, cz, l, w, h, yaw, vx, vy]``.
    """
    dims = anchor[..., _LN_L : _LN_H + 1].exp()
    yaw = torch.atan2(anchor[..., _SIN_YAW : _SIN_YAW + 1], anchor[..., _COS_YAW : _COS_YAW + 1])
    return torch.cat(
        [anchor[..., _X : _Z + 1], dims, yaw, anchor[..., _VX : _VY + 1]],
        dim=-1,
    )


@dataclass
class EncodedGroundTruth:
    """Ground truth of one sample in anchor space.

    Attributes:
        anchor: Anchor state targets ``(M, 10)``, unknown velocities set to zero.
        labels: Class indices ``(M,)``.
        channel_valid: Whether every anchor state channel of every box is known ``(M, 10)``.
    """

    anchor: torch.Tensor
    labels: torch.Tensor
    channel_valid: torch.Tensor


def encode_ground_truth(
    boxes: torch.Tensor, labels: torch.Tensor, device: torch.device, dtype: torch.dtype
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
    anchor = torch.nan_to_num(encode_boxes_to_anchor_space(boxes), nan=0.0)
    return EncodedGroundTruth(
        anchor=anchor,
        labels=labels.to(device=device, dtype=torch.long),
        channel_valid=channel_valid,
    )


def _linear_relu_ln(
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


class InstanceBank(nn.Module):
    """Hold the anchors and propagate confident instances across frames.

    Temporal mode needs scene continuous batch slots, see ``DataModule(streaming=True)``.
    """

    def __init__(
        self,
        *,
        num_anchor: int,
        embed_dims: int,
        anchor_file: str,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.1,
        min_time_interval: float = 0.05,
        max_time_interval: float = 2.0,
        confidence_decay: float = 0.36,
        anchor_dim: int = ANCHOR_DIM,
    ) -> None:
        """Initialize the instance bank.

        Args:
            num_anchor: Total number of instances per frame.
            embed_dims: Instance feature dimension.
            anchor_file: ``.npy`` file with the initial anchors ``(num_anchor, anchor_dim)``.
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
                horizon does not depend on the frame rate. Sparse4D decays by
                0.6 per frame at 2 Hz, about 0.36 per second.
            anchor_dim: Width of an anchor, the 3D box layout by default.
        """
        super().__init__()
        if not 0 <= num_temp_instances < num_anchor:
            raise ValueError(
                f"num_temp_instances must be in [0, num_anchor), got {num_temp_instances} "
                f"with num_anchor={num_anchor}."
            )
        if not 0 < min_time_interval < max_time_interval:
            raise ValueError(
                "The time gate needs 0 < min_time_interval < max_time_interval, got "
                f"{min_time_interval} and {max_time_interval}."
            )
        if not Path(anchor_file).is_file():
            raise FileNotFoundError(
                f"Anchor file not found: {anchor_file}. Generate it with "
                "`python -m autoware_ml.scripts.generate_anchors --config-name <task config>`."
            )
        anchor = np.load(anchor_file)
        if anchor.ndim != 2 or anchor.shape[1] != anchor_dim:
            raise ValueError(
                f"Anchor file {anchor_file} must contain an (N, {anchor_dim}) array, "
                f"got {anchor.shape}."
            )
        if anchor.shape[0] != num_anchor:
            raise ValueError(
                f"Anchor file {anchor_file} holds {anchor.shape[0]} anchors, "
                f"config expects num_anchor={num_anchor}."
            )
        self.num_anchor = num_anchor
        self.anchor_dim = anchor_dim
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.min_time_interval = min_time_interval
        self.max_time_interval = max_time_interval
        self.confidence_decay = confidence_decay
        self.anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float32))
        self.instance_feature = nn.Parameter(
            torch.zeros(num_anchor, embed_dims), requires_grad=False
        )
        self.reset()

    def reset(self) -> None:
        """Drop the cached temporal state of every slot."""
        self._cached_feature: torch.Tensor | None = None
        self._cached_anchor: torch.Tensor | None = None
        self._cached_confidence: torch.Tensor | None = None
        self._cached_timestamp: torch.Tensor | None = None
        self._cached_pose: torch.Tensor | None = None
        self._cached_scene_tokens: list[str] | None = None
        self._slot_valid: torch.Tensor | None = None
        self._temp_time_interval: torch.Tensor | None = None
        self._temp_prev_confidence: torch.Tensor | None = None
        self.cached_query_index: torch.Tensor | None = None

    @property
    def lane_continues(self) -> torch.Tensor:
        """Whether every batch lane continues the stream of its previous frame ``(B,)``."""
        if self._slot_valid is None:
            raise RuntimeError("The lane continuity is known only after get() of a temporal bank.")
        return self._slot_valid

    @staticmethod
    def _project_anchor(
        anchor: torch.Tensor, time_interval: torch.Tensor, transform: torch.Tensor
    ) -> torch.Tensor:
        """Warp cached anchors into the current frame.

        Args:
            anchor: Cached anchors ``(B, K, 10)``.
            time_interval: Frame gaps ``(B,)`` in seconds.
            transform: Frame to frame transforms ``(B, 4, 4)``.

        Returns:
            Warped anchors ``(B, K, 10)``.
        """
        linear = transform[:, :3, :3]
        translation = transform[:, :3, 3]

        velocity = anchor[..., _VX : _VY + 1]
        advanced_xy = anchor[..., _X : _Y + 1] + velocity * time_interval[:, None, None]
        center = torch.cat([advanced_xy, anchor[..., _Z : _Z + 1]], dim=-1)
        center = center @ linear.transpose(-1, -2) + translation[:, None, :]

        heading = torch.stack([anchor[..., _COS_YAW], anchor[..., _SIN_YAW]], dim=-1)
        heading = heading @ linear[:, :2, :2].transpose(-1, -2)
        heading = F.normalize(heading, dim=-1, eps=1e-6)

        velocity = velocity @ linear[:, :2, :2].transpose(-1, -2)

        scale = _det3x3(linear).abs().clamp(min=1e-6) ** (1.0 / 3.0)
        sizes = anchor[..., _LN_L : _LN_H + 1] + scale.log()[:, None, None]

        return torch.cat([center, sizes, heading[..., 1:2], heading[..., 0:1], velocity], dim=-1)

    def get(
        self,
        batch_size: int,
        timestamps_seconds: torch.Tensor | None = None,
        lidar2global: torch.Tensor | None = None,
        scene_tokens: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return the fresh instances and the warped temporal instances.

        Args:
            batch_size: Number of samples in the batch.
            timestamps_seconds: Frame timestamps ``(B,)`` in seconds, temporal
                mode only.
            lidar2global: Poses ``(B, 4, 4)`` with the augmentations folded in,
                temporal mode only.
            scene_tokens: Scene identifiers of the frames, temporal mode only.

        Returns:
            Tuple of instance features ``(B, N, C)``, anchors ``(B, N, 10)``,
            time intervals ``(B,)`` and, in temporal mode, the warped cached
            features ``(B, K, C)`` and anchors ``(B, K, 10)``, zeroed for the
            slots without valid history, otherwise ``None`` twice.

        Raises:
            ValueError: If a temporal bank misses the frame metadata.
        """
        instance_feature = self.instance_feature[None].expand(batch_size, -1, -1)
        anchor = self.anchor[None].expand(batch_size, -1, -1)
        if self.num_temp_instances == 0:
            time_interval = self.anchor.new_full((batch_size,), self.default_time_interval)
            return instance_feature, anchor, time_interval, None, None

        if timestamps_seconds is None or lidar2global is None or scene_tokens is None:
            raise ValueError(
                "Temporal instance propagation needs the timestamp, the pose and the scene "
                "token of every frame."
            )
        device = self.anchor.device
        timestamp = timestamps_seconds.to(device=device, dtype=torch.float64)
        lidar2global = lidar2global.to(device=device, dtype=torch.float64)

        if self._cached_feature is None or self._cached_feature.shape[0] != batch_size:
            self._cached_feature = self.anchor.new_zeros(
                (batch_size, self.num_temp_instances, self.embed_dims)
            )
            self._cached_anchor = self.anchor.new_zeros(
                (batch_size, self.num_temp_instances, self.anchor_dim)
            )
            self._cached_confidence = self.anchor.new_zeros((batch_size, self.num_temp_instances))
            self._cached_timestamp = timestamp.clone()
            self._cached_pose = lidar2global.clone()
            self._cached_scene_tokens = [""] * batch_size

        time_gap = timestamp - self._cached_timestamp
        same_scene = torch.tensor(
            [
                bool(previous) and previous == current
                for previous, current in zip(self._cached_scene_tokens, scene_tokens)
            ],
            device=device,
        )
        valid = (
            same_scene & (time_gap >= self.min_time_interval) & (time_gap <= self.max_time_interval)
        )
        time_interval = torch.where(
            valid, time_gap, torch.full_like(time_gap, self.default_time_interval)
        ).to(dtype=torch.float32)

        transform = torch.linalg.solve(lidar2global, self._cached_pose).to(dtype=torch.float32)
        warped_anchor = self._project_anchor(self._cached_anchor, time_interval, transform)

        gate = valid[:, None, None].to(dtype=torch.float32)
        temp_feature = self._cached_feature * gate
        temp_anchor = warped_anchor * gate

        self._slot_valid = valid
        self._temp_prev_confidence = self._cached_confidence * valid[:, None]
        self._temp_time_interval = time_interval
        self._cached_anchor = temp_anchor
        self._cached_feature = temp_feature
        return instance_feature, anchor, time_interval, temp_feature, temp_anchor

    @staticmethod
    def merge_temporal(
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        temp_feature: torch.Tensor,
        temp_anchor: torch.Tensor,
        gate: torch.Tensor,
        num_temp_instances: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge the propagated instances into the working set after block 0.

        The temporal block replaces the least confident fresh instances at the leading positions.

        Args:
            instance_feature: Fresh instance features ``(B, N, C)``.
            anchor: Anchors refined by block 0 ``(B, N, 10)``.
            cls_logits: Class logits of block 0 ``(B, N, num_classes)``.
            temp_feature: Propagated instance features ``(B, K, C)``.
            temp_anchor: Propagated (warped) anchors ``(B, K, 10)``.
            gate: Per slot validity gate ``(B, 1, 1)`` in ``{0, 1}``.
            num_temp_instances: Size ``K`` of the temporal block.

        Returns:
            Tuple of merged instance features and anchors ``(B, N, ...)``.
        """
        num_fresh = instance_feature.shape[1] - num_temp_instances
        confidence = cls_logits.max(dim=-1).values
        order = confidence.argsort(dim=1, descending=True)
        top_index = order[:, :num_fresh]
        remainder_index = order[:, num_fresh:]

        def gather(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return source.gather(1, index[..., None].expand(-1, -1, source.shape[-1]))

        gate = gate.to(dtype=instance_feature.dtype)
        temporal_feature = temp_feature * gate + gather(instance_feature, remainder_index) * (
            1.0 - gate
        )
        temporal_anchor = temp_anchor * gate + gather(anchor, remainder_index) * (1.0 - gate)

        merged_feature = torch.cat([temporal_feature, gather(instance_feature, top_index)], dim=1)
        merged_anchor = torch.cat([temporal_anchor, gather(anchor, top_index)], dim=1)
        return merged_feature, merged_anchor

    @staticmethod
    def select_temporal_state(
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        prev_confidence: torch.Tensor,
        confidence_decay: float,
        time_interval: torch.Tensor,
        num_temp_instances: int,
        anchor_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the most confident instances as the state of the next frame.

        The previous confidence decays as ``confidence_decay ** dt``.

        Args:
            instance_feature: Instance features of the last layer ``(B, N, C)``.
            anchor: Anchors refined by the last layer ``(B, N, D)``.
            cls_logits: Class logits of the last layer ``(B, N, num_classes)``.
            prev_confidence: Previous confidence of the temporal block
                ``(B, K)``, zeros for reset slots.
            confidence_decay: Decay per second.
            time_interval: Frame gaps ``(B,)`` in seconds.
            num_temp_instances: Size ``K`` of the temporal state.
            anchor_dim: Width ``D`` of an anchor, a constant of the exported graph.

        Returns:
            Tuple of state features ``(B, K, C)``, anchors ``(B, K, D)``,
            confidences ``(B, K)`` and the query index of every state slot ``(B, K)``.
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
        state_feature = instance_feature.gather(1, index.expand(-1, -1, instance_feature.shape[-1]))
        state_anchor = anchor.gather(1, index.expand(-1, -1, anchor_dim))
        return state_feature, state_anchor, top_confidence, top_index

    def update(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Merge the cached instances into the working set after block 0."""
        return self.merge_temporal(
            instance_feature,
            anchor,
            cls_logits,
            self._cached_feature,
            self._cached_anchor,
            self._slot_valid[:, None, None],
            self.num_temp_instances,
        )

    def cache(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        cls_logits: torch.Tensor,
        timestamps_seconds: torch.Tensor,
        lidar2global: torch.Tensor,
        scene_tokens: Sequence[str],
    ) -> None:
        """Write the most confident instances back as the state of the next frame.

        Args:
            instance_feature: Instance features of the last layer ``(B, N, C)``.
            anchor: Anchors refined by the last layer ``(B, N, 10)``.
            cls_logits: Class logits of the last layer ``(B, N, num_classes)``.
            timestamps_seconds: Frame timestamps ``(B,)`` in seconds.
            lidar2global: Poses ``(B, 4, 4)`` with the augmentations folded in.
            scene_tokens: Scene identifiers of the frames.
        """
        state_feature, state_anchor, state_confidence, query_index = self.select_temporal_state(
            instance_feature,
            anchor,
            cls_logits,
            self._temp_prev_confidence,
            self.confidence_decay,
            self._temp_time_interval,
            self.num_temp_instances,
            self.anchor_dim,
        )
        self._cached_feature = state_feature.detach()
        self._cached_anchor = state_anchor.detach()
        self._cached_confidence = state_confidence.detach()
        self.cached_query_index = query_index
        self._cached_timestamp = timestamps_seconds.to(
            device=self.anchor.device, dtype=torch.float64
        ).clone()
        self._cached_pose = lidar2global.to(device=self.anchor.device, dtype=torch.float64).clone()
        self._cached_scene_tokens = [str(token) for token in scene_tokens]


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
        self._slot_owner: torch.Tensor | None = None
        self._slot_track_id: torch.Tensor | None = None
        self._query_owner: torch.Tensor | None = None
        self._query_track_id: torch.Tensor | None = None
        self._stream_id: torch.Tensor | None = None
        self.frame_observed: list[torch.Tensor] | None = None
        self._next_track_id = 0
        self._next_stream_id = 0

    def _new_stream_ids(self, count: int, device: torch.device) -> torch.Tensor:
        """Stream ids unique over the ranks, the rank fills the high bits."""
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        ids = torch.arange(self._next_stream_id, self._next_stream_id + count, device=device)
        self._next_stream_id += count
        return ids + (rank << 40)

    def begin_frame(self, lane_continues: torch.Tensor, query_index: torch.Tensor | None) -> None:
        """Carry the state into the cached slots and reset the lanes that start a stream.

        Args:
            lane_continues: Whether every lane continues its stream ``(B,)``.
            query_index: Query index of every slot the bank cached at the previous frame
                ``(B, K)``, None at the first frame.
        """
        batch_size = lane_continues.shape[0]
        device = lane_continues.device
        empty = torch.full(
            (batch_size, self.num_temp_instances), -1, dtype=torch.long, device=device
        )
        if self._stream_id is None or self._stream_id.shape[0] != batch_size:
            self._observed = [set() for _ in range(batch_size)]
            self._slot_owner = empty
            self._slot_track_id = empty.clone()
            self._stream_id = self._new_stream_ids(batch_size, device)
        else:
            self._slot_owner = (
                empty if self._query_owner is None else self._query_owner.gather(1, query_index)
            )
            self._slot_track_id = (
                empty.clone()
                if self._query_track_id is None
                else self._query_track_id.gather(1, query_index)
            )
        restart = ~lane_continues
        self._slot_owner[restart] = -1
        self._slot_track_id[restart] = -1
        for lane in restart.nonzero().flatten().tolist():
            self._observed[lane] = set()
        num_restart = int(restart.sum())
        if num_restart:
            self._stream_id[restart] = self._new_stream_ids(num_restart, device)
        self._query_owner = None
        self._query_track_id = None
        self.frame_observed = None

    def observe(
        self, instance_ids: Sequence[torch.Tensor], num_points: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
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
        self.frame_observed = masks
        return masks

    def slot_owners(self, lanes: Sequence[int]) -> torch.Tensor:
        """Instance every slot of the given lanes follows ``(len(lanes), K)``, -1 when free."""
        return self._slot_owner[list(lanes)]

    def set_query_owners(
        self, lanes: Sequence[int], owners: torch.Tensor, scores: torch.Tensor
    ) -> None:
        """Record the instance every confident query of the given lanes was assigned to.

        Only a query scoring at least ``track_birth_score`` takes or keeps its instance, so the
        unconfident matches of early training never lock an instance to a slot.

        Args:
            lanes: Lanes the owners belong to.
            owners: Assigned instance id of every query ``(len(lanes), N)``, -1 for background.
            scores: Detection score of every query ``(len(lanes), N)``.
        """
        self._query_owner = torch.full(
            (self._slot_owner.shape[0], owners.shape[1]), -1, dtype=torch.long, device=owners.device
        )
        self._query_owner[list(lanes)] = torch.where(
            scores >= self.track_birth_score, owners, torch.full_like(owners, -1)
        )

    def assign_track_ids(self, scores: torch.Tensor) -> torch.Tensor:
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
    def stream_ids(self) -> torch.Tensor:
        """Stream id of every lane ``(B,)``, new whenever a lane starts a stream."""
        return self._stream_id


class SparseBox3DEncoder(nn.Module):
    """Embed anchor states into positional features."""

    def __init__(self, embed_dims: int) -> None:
        """Initialize the anchor encoder.

        Args:
            embed_dims: Output embedding dimension.
        """
        super().__init__()
        self.pos_fc = nn.Sequential(*_linear_relu_ln(embed_dims, 1, 2, 3))
        self.size_fc = nn.Sequential(*_linear_relu_ln(embed_dims, 1, 2, 3))
        self.yaw_fc = nn.Sequential(*_linear_relu_ln(embed_dims, 1, 2, 2))
        self.vel_fc = nn.Sequential(*_linear_relu_ln(embed_dims, 1, 2, 2))
        self.output_fc = nn.Sequential(*_linear_relu_ln(embed_dims, 1, 2, embed_dims))

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        """Encode anchors ``(..., 10)`` into embeddings ``(..., C)``."""
        pos_feat = self.pos_fc(anchor[..., _X : _Z + 1])
        size_feat = self.size_fc(anchor[..., _LN_L : _LN_H + 1])
        yaw_feat = self.yaw_fc(anchor[..., _SIN_YAW : _COS_YAW + 1])
        vel_feat = self.vel_fc(anchor[..., _VX : _VY + 1])
        return self.output_fc(pos_feat + size_feat + yaw_feat + vel_feat)


class KeyPointsGenerator(Protocol):
    """What the deformable aggregation needs from a keypoint generator."""

    num_pts: int

    def __call__(self, anchor: torch.Tensor, instance_feature: torch.Tensor) -> torch.Tensor:
        """Keypoints ``(B, N, num_pts, 3)`` of anchors ``(B, N, D)``."""
        ...


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
        """
        super().__init__()
        fix_scale_tensor = torch.tensor(fix_scale, dtype=torch.float32)
        if fix_scale_tensor.ndim != 2 or fix_scale_tensor.shape[1] != 3:
            raise ValueError(f"fix_scale must be (P, 3), got {tuple(fix_scale_tensor.shape)}.")
        self.register_buffer("fix_scale", fix_scale_tensor, persistent=False)
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = fix_scale_tensor.shape[0] + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(embed_dims, num_learnable_pts * 3)

    def forward(self, anchor: torch.Tensor, instance_feature: torch.Tensor) -> torch.Tensor:
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


# Features, voxel coordinates and batch offsets of one sparse encoder stage
SparseStage = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# The lidar features a head samples: the sparse stages, or the dense BEV map of a BEV only head
LidarFeatures = Sequence[SparseStage] | torch.Tensor
# Below this occupied weight the aggregated features fade to zero instead of amplifying a sliver
MIN_OCCUPANCY = 0.05


class SparseLevelGrid(nn.Module):
    """Dense voxel grid of one sparse encoder stage, it builds the lookup of the stage voxels."""

    def __init__(self, point_cloud_range: Sequence[float], grid_size: float, feature_stride: int):
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
        self.grid_dims = tuple(math.ceil(cells / feature_stride) for cells in base_cells)
        self.num_cells = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
        self.voxel_size = grid_size * feature_stride
        self.grid_origin = (pc_range[0], pc_range[1], pc_range[2])

    def forward(
        self, feat: torch.Tensor, grid_coord: torch.Tensor, offset: torch.Tensor
    ) -> SparseLevel:
        """Look up the stage voxels in the dense grid of every sample.

        Args:
            feat: Stage features ``(V, C)``.
            grid_coord: Stage voxel coordinates ``(V, 3)`` as ``(x, y, z)``.
            offset: Cumulative voxel offsets of the samples ``(B,)``.

        Returns:
            The stage with its voxel lookup.
        """
        size_x, size_y, size_z = self.grid_dims
        batch_index = offset_to_batch(offset, grid_coord)
        flat_cells = (
            (batch_index * size_z + grid_coord[:, 2].long()) * size_y + grid_coord[:, 1].long()
        ) * size_x + grid_coord[:, 0].long()
        index_grid = torch.full(
            (offset.shape[0] * self.num_cells,),
            EMPTY_ROW,
            dtype=torch.long,
            device=feat.device,
        )
        index_grid[flat_cells] = torch.arange(1, feat.shape[0] + 1, device=feat.device)
        padded = torch.cat([feat.new_zeros((1, feat.shape[1])), feat])
        return SparseLevel(padded, index_grid, self.grid_dims, self.grid_origin, self.voxel_size)


class SparseDeformableAggregation(nn.Module):
    """Sample sparse voxel features at anchor keypoints with learned weights.

    Keypoints interpolate trilinearly over every sampled stage, the weights normalize over the
    stages and the keypoints of every feature group. Keypoints in empty space read zeros, so the
    weighted sum is divided by the weight that landed on occupied voxels. A BEV branch samples
    dense BEV maps at the same keypoints and adds to the voxel features, or replaces them. A
    subclass joins the features of its frame context, see :meth:`join_context`.
    """

    def __init__(
        self,
        *,
        embed_dims: int,
        num_groups: int,
        num_levels: int,
        kps_generator: KeyPointsGenerator,
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
            ValueError: If the block samples no lidar features.
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
        self.num_pts = kps_generator.num_pts
        self.attn_drop = attn_drop
        self.sample_sparse = sample_sparse
        # A BEV only block has no voxel weights, every parameter must train under DDP
        self.weights_fc = (
            nn.Linear(embed_dims, num_groups * num_levels * self.num_pts) if sample_sparse else None
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims) if sample_sparse else None
        self.proj_drop = nn.Dropout(proj_drop)
        self.bev_aggregation = bev_aggregation
        if self.weights_fc is not None:
            nn.init.zeros_(self.weights_fc.weight)
            nn.init.zeros_(self.weights_fc.bias)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        levels: Sequence[SparseLevel],
        bev_levels: Sequence[BEVLevel] = (),
        context: Any = None,
    ) -> torch.Tensor:
        """Aggregate sparse voxel features into the instance features.

        Args:
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchor states ``(B, N, 10)``.
            anchor_embed: Anchor embeddings ``(B, N, C)``.
            levels: Sampled encoder stages with their voxel lookup.
            bev_levels: Dense BEV maps of the BEV branch.
            context: Frame context of a subclass block, None for the lidar block.

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
        output = self.join_context(output, query, key_points, context)
        return instance_feature + self.proj_drop(output)

    def join_context(
        self, output: torch.Tensor, query: torch.Tensor, key_points: torch.Tensor, context: Any
    ) -> torch.Tensor:
        """Join the features of the frame context to the lidar features ``(B, N, C)``.

        Args:
            output: Lidar features of every instance ``(B, N, C)``.
            query: Instance features with the anchor embedding ``(B, N, C)``.
            key_points: Metric keypoints ``(B, N, P, 3)``.
            context: Frame context of the block.

        Returns:
            The lidar features, the lidar block reads no context.

        Raises:
            ValueError: If the lidar block gets a context.
        """
        if context is not None:
            raise ValueError("The lidar aggregation block reads no frame context.")
        return output


def _group_softmax(logits: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Softmax over the samples of every feature group, ``(B, N, S * G)`` to ``(B, N, S, G)``."""
    batch_size, num_anchor = logits.shape[:2]
    return logits.reshape(batch_size, num_anchor, num_samples, -1).softmax(dim=-2)


def keypoint_dropout(weights: torch.Tensor, drop: float, training: bool) -> torch.Tensor:
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
        self.num_samples = num_levels * num_points
        self.attn_drop = attn_drop
        self.weights_fc = nn.Linear(embed_dims, num_groups * self.num_samples)
        self.output_proj = nn.Linear(bev_channels, embed_dims)
        nn.init.zeros_(self.weights_fc.weight)
        nn.init.zeros_(self.weights_fc.bias)

    def forward(
        self, query: torch.Tensor, key_points: torch.Tensor, levels: Sequence[BEVLevel]
    ) -> torch.Tensor:
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


def sincos_position_embedding(
    position: torch.Tensor, num_feats: int, temperature: float = 10000.0
) -> torch.Tensor:
    """Sine and cosine embedding of unit range ``(x, y)`` positions, the ``pos2embed`` of CMT.

    Every axis gets ``num_feats`` features over the wavelengths of the transformer embedding,
    the sine and the cosine of every wavelength alternating. The y features come first.

    Args:
        position: Positions ``(..., 2)`` in the unit range.
        num_feats: Features per axis, even.
        temperature: Base of the wavelength progression.

    Returns:
        Embedding ``(..., 2 * num_feats)``.
    """
    scaled = position * (2.0 * math.pi)
    wavelength = torch.arange(num_feats, dtype=scaled.dtype, device=scaled.device)
    wavelength = temperature ** (2.0 * wavelength.div(2, rounding_mode="floor") / num_feats)
    embeddings = []
    for axis in (_Y, _X):
        phase = scaled[..., axis, None] / wavelength
        embeddings.append(
            torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), dim=-1).flatten(-2)
        )
    return torch.cat(embeddings, dim=-1)


class CMTPositionEmbedding(nn.Module):
    """Positional embedding of metric ``(x, y)`` shared by the BEV cells and the queries.

    CMT (arXiv 2301.01283) puts the queries and the BEV map in one positional space: the
    position, normalized to the unit range over the point cloud range, becomes a sine and cosine
    embedding that one MLP maps to the feature dimension. A query reads it at its anchor center,
    a BEV cell at its center, so a query and the cell under it share one embedding.
    """

    def __init__(
        self, embed_dims: int, point_cloud_range: Sequence[float], temperature: float = 10000.0
    ) -> None:
        """Initialize the shared embedding.

        Args:
            embed_dims: Feature dimension of the embedding, even.
            point_cloud_range: Metric range ``(x_min, y_min, z_min, x_max, y_max, z_max)`` that
                normalizes the positions.
            temperature: Base of the wavelength progression of the sine and cosine embedding.

        Raises:
            ValueError: If the feature dimension is odd.
        """
        super().__init__()
        if embed_dims % 2 != 0:
            raise ValueError(f"The sine and cosine embedding needs even dims, got {embed_dims}.")
        self.embed_dims = embed_dims
        self.temperature = temperature
        self.register_buffer(
            "origin",
            torch.tensor([float(point_cloud_range[0]), float(point_cloud_range[1])]),
            persistent=False,
        )
        self.register_buffer(
            "extent",
            torch.tensor(
                [
                    float(point_cloud_range[3] - point_cloud_range[0]),
                    float(point_cloud_range[4] - point_cloud_range[1]),
                ]
            ),
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """Embed metric ``(x, y)`` positions ``(..., 2)`` into ``(..., C)``."""
        unit = (position - self.origin) / self.extent
        return self.mlp(sincos_position_embedding(unit, self.embed_dims, self.temperature))

    def cells(self, level: BEVLevel) -> torch.Tensor:
        """Embed the cell centers of a BEV level ``(H * W, C)``, row major like its flattened map."""
        height, width = level.feat.shape[-2:]
        origin = self.origin.new_tensor(level.grid_origin)
        xs = (torch.arange(width, device=origin.device) + 0.5) * level.cell_size + origin[0]
        ys = (torch.arange(height, device=origin.device) + 0.5) * level.cell_size + origin[1]
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return self(torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))


class BEVCrossAttention(nn.Module):
    """Attend from every query to every cell of the BEV maps, in place of the aggregation.

    The CMT decoder reads the lidar map with plain cross attention: the queries and the cells
    carry the shared :class:`CMTPositionEmbedding`, so the attention logits compare positions
    next to features, and the values are the cell features alone. The block takes the inputs of
    :class:`SparseDeformableAggregation` and reads no keypoints and no sparse voxels. The anchor
    embedding of Sparse4D stays with the instance attention and the refinement.
    """

    def __init__(
        self,
        *,
        embed_dims: int,
        bev_channels: int,
        num_heads: int,
        position_embedding: CMTPositionEmbedding,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        """Initialize the cross attention block.

        Args:
            embed_dims: Instance feature dimension.
            bev_channels: Feature dimension of the BEV maps.
            num_heads: Number of attention heads.
            position_embedding: Embedding shared by the queries, the cells and the other blocks.
            dropout: Dropout on the attended features.
            attn_drop: Dropout on the attention weights in training.

        Raises:
            ValueError: If the feature dimension does not split into the heads.
        """
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims {embed_dims} not divisible by num_heads {num_heads}.")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.attn_drop = attn_drop
        self.position_embedding = position_embedding
        self.input_proj = nn.Linear(bev_channels, embed_dims)
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.out_proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        levels: Sequence[SparseLevel],
        bev_levels: Sequence[BEVLevel] = (),
        context: Any = None,
    ) -> torch.Tensor:
        """Attend to the BEV cells and add the result to the instance features.

        Args:
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchor states ``(B, N, 10)``, the queries take their position from them.
            anchor_embed: Anchor embeddings ``(B, N, C)``, not read.
            levels: Sparse stages, not read.
            bev_levels: BEV maps the queries attend to, every cell of every level is one key.
            context: Frame context of a subclass block, None for the lidar block.

        Returns:
            Updated instance features ``(B, N, C)``.

        Raises:
            ValueError: If the block gets no BEV map or a frame context.
        """
        if context is not None:
            raise ValueError("The BEV attention block reads no frame context.")
        if len(bev_levels) == 0:
            raise ValueError("The BEV attention block needs BEV maps.")
        query = instance_feature + self.position_embedding(anchor[..., _X : _Y + 1])
        value = torch.cat(
            [self.input_proj(level.feat.flatten(2).transpose(1, 2)) for level in bev_levels], dim=1
        )
        key = value + torch.cat([self.position_embedding.cells(level) for level in bev_levels])
        return instance_feature + self.proj_drop(self._attend(query, key, value))

    def _attend(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Multi-head scaled dot product attention over the cells, ``(B, N, C)``."""
        batch_size = query.shape[0]
        head_dims = self.embed_dims // self.num_heads

        def split_heads(tokens: torch.Tensor) -> torch.Tensor:
            return tokens.reshape(batch_size, -1, self.num_heads, head_dims).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            split_heads(self.q_proj(query)),
            split_heads(self.k_proj(key)),
            split_heads(self.v_proj(value)),
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        return self.out_proj(attended.transpose(1, 2).reshape(batch_size, -1, self.embed_dims))


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

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
        key_feature: torch.Tensor | None = None,
        key_embed: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        if attn_mask is None and key_padding_mask is None:
            attended = self.attn(tokens, key_tokens, value)[0]
        else:
            attended = self.attn(
                tokens, key_tokens, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )[0]
        return self.fc_out(tokens + self.dropout(attended))


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
            *_linear_relu_ln(embed_dims, 2, 2, embed_dims),
            nn.Linear(embed_dims, ANCHOR_DIM),
        )
        self.cls_layers = nn.Sequential(
            *_linear_relu_ln(embed_dims, 1, 2, embed_dims),
            nn.Linear(embed_dims, num_classes),
        )
        self.quality_layers = nn.Sequential(
            *_linear_relu_ln(embed_dims, 1, 2, embed_dims),
            nn.Linear(embed_dims, 2),
        )
        cls_bias = -math.log((1.0 - 0.01) / 0.01)
        nn.init.constant_(self.cls_layers[-1].bias, cls_bias)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine anchors, classify instances and estimate their quality.

        Args:
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchor states ``(B, N, 10)``.
            anchor_embed: Anchor embeddings ``(B, N, C)``.
            time_interval: Time intervals of the samples ``(B,)`` in seconds.

        Returns:
            Tuple of refined anchors ``(B, N, 10)``, class logits
            ``(B, N, num_classes)`` and centerness and yawness logits ``(B, N, 2)``.
        """
        feature = instance_feature + anchor_embed
        delta = self.refine_layers(feature)
        refined = anchor + delta
        velocity = (
            delta[..., _VX : _VY + 1] / time_interval[:, None, None] + anchor[..., _VX : _VY + 1]
        )
        refined = torch.cat([refined[..., :_VX], velocity], dim=-1)
        return refined, self.cls_layers(instance_feature), self.quality_layers(feature)


@dataclass
class Sparse4DTargets:
    """Training targets of one decoder layer from Hungarian matching.

    Attributes:
        cls_targets: One-hot class targets ``(B, N, num_classes)``.
        box_targets: Anchor state targets ``(B, N, 10)``.
        box_weights: Regression weights ``(B, N, 10)``, zero for unmatched
            queries and unknown channels.
        matched: Whether every query is matched to a box ``(B, N)``.
        assigned_instance: Instance id of the box every query is matched to ``(B, N)``, -1
            for background.
        num_pos: Number of matched queries.
    """

    cls_targets: torch.Tensor
    box_targets: torch.Tensor
    box_weights: torch.Tensor
    matched: torch.Tensor
    assigned_instance: torch.Tensor
    num_pos: int


@dataclass
class DenoisingQueries:
    """Denoising queries of one batch and their fixed targets.

    Each group holds noised positives followed by displaced negatives.
    """

    anchor: torch.Tensor
    cls_targets: torch.Tensor
    box_targets: torch.Tensor
    box_weights: torch.Tensor
    cls_weights: torch.Tensor
    attn_mask: torch.Tensor


def _signed_uniform(
    shape: tuple[int, ...], low: float, high: float, device: torch.device
) -> torch.Tensor:
    """Sample ``sign * U(low, high)`` with an independent random sign per element."""
    magnitude = torch.empty(shape, device=device).uniform_(low, high)
    sign = torch.randint(0, 2, shape, device=device, dtype=torch.float32) * 2.0 - 1.0
    return magnitude * sign


def _noise_denoising_anchors(
    encoded_gt: torch.Tensor,
    num_groups: int,
    *,
    center_low: float,
    center_high: float,
    noise_size: float,
    noise_yaw: float,
    noise_velocity: float,
) -> torch.Tensor:
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
) -> torch.Tensor:
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
    total = num_query + num_groups * group_width
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    mask[:num_query, num_query:] = True
    for group in range(num_groups):
        start = num_query + group * group_width
        end = start + group_width
        mask[start:end, num_query:start] = True
        mask[start:end, end:] = True
    return mask


@torch.no_grad()
def build_denoising_queries(
    gt_boxes: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],
    *,
    num_query: int,
    num_classes: int,
    query_budget: int,
    reg_weights: torch.Tensor,
    noise_center: float,
    noise_center_neg: float,
    noise_size: float,
    noise_yaw: float,
    noise_velocity: float,
) -> DenoisingQueries | None:
    """Build noised ground truth queries with their targets fixed at construction.

    Groups follow the DINO budget ``max(query_budget // (2 * max_gt), 1)``.

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
        The built queries, or ``None`` when no sample carries a box.
    """
    max_gt = max(boxes.shape[0] for boxes in gt_boxes)
    if max_gt == 0:
        return None
    device = reg_weights.device
    num_groups = max(query_budget // (2 * max_gt), 1)
    group_width = 2 * max_gt
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


@dataclass
class _DecoderOutputs:
    """Per layer outputs of the decoder and its final state.

    Attributes:
        classification: Class logits of every layer ``(B, N + D, num_classes)``.
        prediction: Anchor states of every layer ``(B, N + D, 10)``.
        quality: Quality logits of every layer ``(B, N + D, 2)``.
        instance_feature: Final instance features of the regular queries ``(B, N, C)``.
        anchor: Final anchors of the regular queries ``(B, N, 10)``.
        cls_logits: Final class logits of the regular queries ``(B, N, num_classes)``.
    """

    classification: list[torch.Tensor]
    prediction: list[torch.Tensor]
    quality: list[torch.Tensor]
    instance_feature: torch.Tensor
    anchor: torch.Tensor
    cls_logits: torch.Tensor


class SparseQueryHead(nn.Module):
    """Decoder of instance queries over sparse encoder stages, as in Sparse4D.

    Every block attends to the temporal memory and to the other queries, samples the stages at
    keypoints of its anchors, runs an FFN and refines the anchors. A subclass sets the anchor
    encoding, the keypoints and the refinement of its anchor kind.
    """

    def _build_decoder(
        self,
        *,
        embed_dims: int,
        num_decoder: int,
        num_heads: int,
        num_groups: int,
        feedforward_channels: int,
        dropout: float,
        attn_drop: float,
        temporal: bool,
        point_cloud_range: Sequence[float],
        grid_size: float,
        feature_strides: Sequence[int],
        anchor_encoder: nn.Module,
        kps_generator: Callable[[], KeyPointsGenerator],
        refine: Callable[[], nn.Module],
        bev: nn.Module | None = None,
        bev_cell_size: float | None = None,
        bev_aggregation: Callable[[], BEVDeformableAggregation | None] = lambda: None,
        sample_sparse: bool = True,
        aggregation: Callable[..., SparseDeformableAggregation] = SparseDeformableAggregation,
        bev_attention: Callable[[], BEVCrossAttention] | None = None,
    ) -> None:
        """Create the decoder blocks.

        Args:
            embed_dims: Instance feature dimension.
            num_decoder: Number of decoder blocks.
            num_heads: Attention heads of the instance attention.
            num_groups: Feature groups of the deformable aggregation.
            feedforward_channels: Hidden width of the FFN blocks.
            dropout: Dropout probability in the attention and FFN blocks.
            attn_drop: Keypoint dropout of the aggregation, or the attention dropout of the BEV
                attention, in training.
            temporal: Whether the blocks attend to a temporal memory.
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            feature_strides: Total pooling strides of the sampled stages.
            anchor_encoder: Embedding of the anchors.
            kps_generator: Builds the keypoint generator of one block.
            refine: Builds the refinement of one block.
            bev: Builds the BEV levels of a frame from the stages, None without one.
            bev_cell_size: Metric cell size of the BEV map the head gets as its lidar features,
                which it places on the grid of ``point_cloud_range`` as its one BEV level. None
                without one.
            bev_aggregation: Builds the BEV sampling of one block, None without BEV maps.
            sample_sparse: Whether the blocks sample the sparse voxels.
            aggregation: Builds the deformable aggregation of one block from its keyword
                arguments.
            bev_attention: Builds the BEV cross attention of one block, which then takes the
                place of its deformable aggregation. None for the aggregation.
        """
        self.num_decoder = num_decoder
        self.bev = bev
        self.bev_cell_size = None if bev_cell_size is None else float(bev_cell_size)
        self.bev_grid_origin = (float(point_cloud_range[0]), float(point_cloud_range[1]))
        self.sample_sparse = sample_sparse
        self.anchor_encoder = anchor_encoder
        self.gnn = nn.ModuleList(
            [
                DecoupledInstanceAttention(embed_dims, num_heads, dropout)
                for _ in range(num_decoder - 1)
            ]
        )
        self.gnn_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_decoder - 1)])
        if temporal:
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
        # The lidar block of every decoder, the deformable aggregation or the BEV attention
        self.deformable = nn.ModuleList(
            [
                aggregation(
                    embed_dims=embed_dims,
                    num_groups=num_groups,
                    num_levels=len(feature_strides),
                    kps_generator=kps_generator(),
                    proj_drop=dropout,
                    attn_drop=attn_drop,
                    bev_aggregation=bev_aggregation(),
                    sample_sparse=sample_sparse,
                )
                if bev_attention is None
                else bev_attention()
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
        self.refine = nn.ModuleList([refine() for _ in range(num_decoder)])

    def _refine(
        self,
        block_index: int,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Refine the anchors of one block and classify the queries.

        Args:
            block_index: Index of the decoder block.
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchors ``(B, N, D)``.
            anchor_embed: Anchor embeddings ``(B, N, C)``.
            time_interval: Frame gaps ``(B,)`` in seconds.

        Returns:
            Refined anchors, class logits and quality logits, None for a head without quality.
        """
        raise NotImplementedError

    def _block_context(self, anchor: torch.Tensor, context: Any) -> Any:
        """Frame context the blocks read for the current anchors, the head context as given."""
        return context

    def levels(self, lidar_features: LidarFeatures) -> list[SparseLevel]:
        """Look up the voxels of every sampled stage once per frame, none for a BEV only head.

        Raises:
            ValueError: If a head sampling the sparse voxels gets a BEV map.
        """
        if not self.sample_sparse:
            return []
        if isinstance(lidar_features, torch.Tensor):
            raise ValueError("A head sampling the sparse voxels needs the sparse stages.")
        return [grid(*stage) for grid, stage in zip(self.level_grids, lidar_features, strict=True)]

    def bev_levels(self, lidar_features: LidarFeatures) -> list[BEVLevel]:
        """Build the BEV levels of the frame once, none for a head without them.

        The levels come from the ``bev`` module over the stages, or the lidar features are one
        BEV map placed on the voxel grid with ``bev_cell_size``.

        Raises:
            ValueError: If a head placing a BEV map gets the sparse stages.
        """
        if self.bev is not None:
            return self.bev(lidar_features)
        if self.bev_cell_size is None:
            return []
        if not isinstance(lidar_features, torch.Tensor):
            raise ValueError("A head sampling a BEV map needs the map as its lidar features.")
        return [BEVLevel(lidar_features, self.bev_grid_origin, self.bev_cell_size)]

    @staticmethod
    def _batch_size(lidar_features: LidarFeatures) -> int:
        """Batch size of the lidar features, from the BEV map or the offsets of the first stage."""
        if isinstance(lidar_features, torch.Tensor):
            return lidar_features.shape[0]
        return lidar_features[0][2].shape[0]

    def _decode(
        self,
        levels: Sequence[SparseLevel],
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        time_interval: torch.Tensor,
        temp_feature: torch.Tensor | None,
        temp_anchor: torch.Tensor | None,
        merge: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None,
        denoising: DenoisingQueries | None = None,
        context: Any = None,
        bev_levels: Sequence[BEVLevel] = (),
    ) -> _DecoderOutputs:
        """Run the decoder blocks over the sampled sparse encoder stages.

        Denoising queries ride at the tail and never reach the merge or the final state.

        Args:
            levels: Sampled stages with their voxel lookup.
            instance_feature: Fresh instance features ``(B, N, C)``.
            anchor: Fresh anchors ``(B, N, D)``.
            time_interval: Frame gaps of the slots ``(B,)`` in seconds.
            temp_feature: Gated temporal instance features ``(B, K, C)``, or
                ``None`` in single frame mode.
            temp_anchor: Warped temporal anchors ``(B, K, D)``.
            merge: Temporal merge applied after block 0, or ``None``.
            denoising: Optional denoising queries appended at the tail with
                zero features and the isolation mask on the instance attention.
            context: Frame context of a subclass head, None for the lidar head.
            bev_levels: BEV maps of a head with a BEV branch.

        Returns:
            The per layer outputs over the regular and the denoising queries
            and the final state of the regular queries.
        """
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
        temp_anchor_embed = None if temp_anchor is None else self.anchor_encoder(temp_anchor)
        block_context = self._block_context(anchor, context)
        classification: list[torch.Tensor] = []
        prediction: list[torch.Tensor] = []
        quality: list[torch.Tensor] = []
        for block_index in range(self.num_decoder):
            if block_index > 0:
                if temp_feature is not None:
                    instance_feature = self.temp_gnn[block_index - 1](
                        instance_feature, anchor_embed, temp_feature, temp_anchor_embed
                    )
                    instance_feature = self.temp_gnn_norm[block_index - 1](instance_feature)
                instance_feature = self.gnn[block_index - 1](
                    instance_feature, anchor_embed, attn_mask=attn_mask
                )
                instance_feature = self.gnn_norm[block_index - 1](instance_feature)
            instance_feature = self.deformable[block_index](
                instance_feature, anchor, anchor_embed, levels, bev_levels, block_context
            )
            instance_feature = self.deformable_norm[block_index](instance_feature)
            instance_feature = self.ffn_norm[block_index](
                instance_feature + self.ffn[block_index](instance_feature)
            )
            anchor, cls_logits, quality_logits = self._refine(
                block_index, instance_feature, anchor, anchor_embed, time_interval
            )
            classification.append(cls_logits)
            prediction.append(anchor)
            if quality_logits is not None:
                quality.append(quality_logits)
            if block_index == 0 and merge is not None:
                merged_feature, merged_anchor = merge(
                    instance_feature[:, :num_query],
                    anchor[:, :num_query],
                    cls_logits[:, :num_query],
                )
                instance_feature = torch.cat(
                    [merged_feature, instance_feature[:, num_query:]], dim=1
                )
                anchor = torch.cat([merged_anchor, anchor[:, num_query:]], dim=1)
                # The merge brings in the temporal instances, the context follows the new set
                block_context = self._block_context(anchor, context)
            if block_index != self.num_decoder - 1:
                anchor_embed = self.anchor_encoder(anchor)
                # Read the temporal memory at the refined anchors
                if block_index > 0 and temp_anchor_embed is not None:
                    temp_anchor_embed = anchor_embed[:, : temp_anchor_embed.shape[1]]
        return _DecoderOutputs(
            classification=classification,
            prediction=prediction,
            quality=quality,
            instance_feature=instance_feature[:, :num_query],
            anchor=anchor[:, :num_query],
            cls_logits=cls_logits[:, :num_query],
        )

    def forward_temporal_export(
        self,
        lidar_features: LidarFeatures,
        temp_instance_feature: torch.Tensor,
        temp_anchor: torch.Tensor,
        temp_confidence: torch.Tensor,
        temp_valid: torch.Tensor,
        time_interval: torch.Tensor,
        prev2current: torch.Tensor,
        context: Any = None,
    ) -> dict[str, torch.Tensor]:
        """Run the temporal decoder with the bank state at the graph boundary.

        A frame's ``*_out`` state feeds the next one, a reset feeds zeros and ``temp_valid = 0``.
        The query index of every state slot lets the runtime carry its track ids into the slots.

        Args:
            lidar_features: Features, voxel coordinates and batch offsets of the sampled stages,
                from the finest to the coarsest, or the dense BEV map of a BEV only head.
            temp_instance_feature: Cached instance features ``(B, K, C)``.
            temp_anchor: Cached anchors ``(B, K, D)`` in the previous frame.
            temp_confidence: Cached instance confidences ``(B, K)``.
            temp_valid: Continuity gates of the slots ``(B,)`` in ``{0, 1}``.
            time_interval: Frame gaps ``(B,)`` in seconds.
            prev2current: Transforms from the previous into the current frame ``(B, 4, 4)``.
            context: Frame context of a subclass head, None for the lidar head.

        Returns:
            Last layer ``classification``, ``prediction``, ``instance_feature`` and, for a head
            with quality, ``quality``, the state of the next frame, ``temp_instance_feature_out``, ``temp_anchor_out`` and
            ``temp_confidence_out``, and the query of every state slot, ``temp_query_index_out``.

        Raises:
            ValueError: If the head has no temporal memory.
        """
        bank = self.instance_bank
        if bank.num_temp_instances == 0:
            raise ValueError(
                "forward_temporal_export needs a temporal head (num_temp_instances > 0)."
            )
        batch_size = self._batch_size(lidar_features)
        instance_feature = bank.instance_feature[None].expand(batch_size, -1, -1)
        anchor = bank.anchor[None].expand(batch_size, -1, -1)

        gate = temp_valid[:, None, None]
        warped_anchor = bank._project_anchor(temp_anchor, time_interval, prev2current)
        temp_feature = temp_instance_feature * gate
        temp_anchor = warped_anchor * gate
        prev_confidence = temp_confidence * temp_valid[:, None]

        def merge(
            feature: torch.Tensor, anchor_state: torch.Tensor, cls_logits: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return bank.merge_temporal(
                feature,
                anchor_state,
                cls_logits,
                temp_feature,
                temp_anchor,
                gate,
                bank.num_temp_instances,
            )

        decoded = self._decode(
            self.levels(lidar_features),
            instance_feature,
            anchor,
            time_interval,
            temp_feature,
            temp_anchor,
            merge,
            context=context,
            bev_levels=self.bev_levels(lidar_features),
        )
        state_feature, state_anchor, state_confidence, query_index = bank.select_temporal_state(
            decoded.instance_feature,
            decoded.anchor,
            decoded.cls_logits,
            prev_confidence,
            bank.confidence_decay,
            time_interval,
            bank.num_temp_instances,
            bank.anchor_dim,
        )
        outputs = {
            "classification": decoded.classification[-1],
            "prediction": decoded.prediction[-1],
            "instance_feature": decoded.instance_feature,
            "temp_instance_feature_out": state_feature,
            "temp_anchor_out": state_anchor,
            "temp_confidence_out": state_confidence,
            TEMPORAL_QUERY_INDEX_OUTPUT_NAME: query_index,
        }
        if decoded.quality:
            outputs["quality"] = decoded.quality[-1]
        return outputs

    @property
    def is_temporal(self) -> bool:
        """Whether the head propagates instances across frames."""
        return self.instance_bank.num_temp_instances > 0

    def temporal_state_export_inputs(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Return reset state sample inputs for tracing the temporal export.

        Raises:
            ValueError: If the head has no temporal memory.
        """
        bank = self.instance_bank
        if bank.num_temp_instances == 0:
            raise ValueError(
                "Temporal export inputs need a temporal head (num_temp_instances > 0)."
            )
        num_temp = bank.num_temp_instances
        return (
            torch.zeros(1, num_temp, self.embed_dims, device=device),
            torch.zeros(1, num_temp, bank.anchor_dim, device=device),
            torch.zeros(1, num_temp, device=device),
            torch.zeros(1, device=device),
            torch.full((1,), bank.default_time_interval, device=device),
            torch.eye(4, device=device).unsqueeze(0),
        )

    def prepare_for_export(self) -> "SparseQueryHead":
        """Return an export ready copy that emits the last layer only.

        Temporal heads export through :meth:`forward_temporal_export`.
        """
        head = deepcopy(self)
        head._export_mode = True
        for attention_block in head.gnn:
            attention_block.attn = ExportableMultiheadAttention(attention_block.attn)
        if head.is_temporal:
            for attention_block in head.temp_gnn:
                attention_block.attn = ExportableMultiheadAttention(attention_block.attn)
        return head.eval()


class Sparse4DHead(SparseQueryHead):
    """Sparse4D query head sampling the sparse stages or the BEV map of a lidar branch."""

    def __init__(
        self,
        *,
        num_classes: int,
        embed_dims: int,
        anchor_file: str,
        num_anchor: int,
        point_cloud_range: Sequence[float],
        grid_size: float,
        feature_strides: Sequence[int],
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
        score_threshold: float | None = None,
        class_names: Sequence[str] | None = None,
        nms_group_configs: Sequence[NMSGroupConfig] | None = None,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.1,
        min_time_interval: float = 0.05,
        max_time_interval: float = 2.0,
        confidence_decay: float = 0.36,
        min_observed_points: int | None = None,
        track_birth_score: float | None = None,
        dn_query_budget: int = 0,
        dn_noise_center: float = 0.5,
        dn_noise_center_neg: float = 1.0,
        dn_noise_size: float = 0.2,
        dn_noise_yaw: float = 0.39,
        dn_noise_velocity: float = 1.0,
        lidar_sampling: str = "sparse",
        bev: nn.Module | None = None,
        bev_cell_size: float | None = None,
        bev_channels: int = 0,
        aggregation: Callable[..., SparseDeformableAggregation] = SparseDeformableAggregation,
        yaw_free_classes: Sequence[str] = (),
        yaw_reversible_classes: Sequence[str] = (),
        global_num_pos: bool = False,
    ) -> None:
        """Initialize the Sparse4D head.

        Args:
            num_classes: Number of detection classes.
            embed_dims: Instance feature dimension, the feature width of the
                sampled stage.
            anchor_file: ``.npy`` file with the k-means anchors ``(num_anchor, 10)``.
            num_anchor: Number of instances per frame.
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            feature_strides: Total pooling strides of the sampled stages, from the finest to
                the coarsest.
            num_decoder: Number of decoder blocks.
            num_heads: Attention heads of the instance attention.
            num_groups: Feature groups of the deformable aggregation.
            feedforward_channels: Hidden width of the FFN blocks.
            dropout: Dropout probability in the attention and FFN blocks.
            fix_scale: Fixed keypoint offsets as fractions of the box size.
            num_learnable_pts: Keypoints per anchor that the instance feature places.
            attn_drop: Keypoint dropout of the aggregation, or the attention dropout of the BEV
                attention, in training.
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
            score_threshold: Optional score threshold of the decoding.
            class_names: Class names in label order, needed by ``nms_group_configs``.
            nms_group_configs: Optional circle NMS groups, None runs no NMS. Boxes of classes
                outside every group are not suppressed.
            num_temp_instances: Instances propagated across frames, ``0``
                turns the temporal memory off. Temporal training needs scene
                continuous batch slots (``DataModule(streaming=True)``).
            default_time_interval: Time interval in seconds without history.
            min_time_interval: Smallest frame gap that keeps the state of a slot.
            max_time_interval: Largest frame gap that keeps the state of a slot.
            confidence_decay: Decay of the cached instance confidence per second.
            min_observed_points: Points an instance needs in one frame of a stream to become a
                target for the rest of the stream. Required by a temporal head.
            track_birth_score: Score from which a query without a track starts a new one.
                Required by a temporal head.
            dn_query_budget: Approximate number of denoising queries appended
                per sample in training steps, ``0`` turns denoising off.
            dn_noise_center: Upper bound of the positive center offset
                magnitude, as a fraction of the box half extent.
            dn_noise_center_neg: Upper bound of the negative center offset,
                negatives are displaced by ``U(dn_noise_center, dn_noise_center_neg)``.
            dn_noise_size: Half range of the uniform log size jitter.
            dn_noise_yaw: Half range of the uniform heading rotation in radians.
            dn_noise_velocity: Half range of the uniform velocity jitter in m/s.
            lidar_sampling: ``sparse`` samples the sparse voxels, ``sparse_bev`` adds the BEV
                maps and ``bev`` samples the BEV maps alone. ``bev_attention`` attends from
                every query to every cell of the BEV maps with the shared CMT positional
                embedding of :class:`BEVCrossAttention` in place of the deformable aggregation.
            bev: Builds the BEV levels of a frame from the stages. BEV sampling takes exactly
                one of ``bev`` and ``bev_cell_size``.
            bev_cell_size: Metric cell size of the BEV map the head gets as its lidar features
                in place of the stages, placed on the grid of ``point_cloud_range``.
            bev_channels: Feature dimension of the BEV maps.
            aggregation: Builds the deformable aggregation of one decoder block from its keyword
                arguments.
            yaw_free_classes: Classes without a heading, such as traffic cones, whose yaw
                takes no part in the matching and the regression.
            yaw_reversible_classes: Classes whose heading is defined up to a half turn, such as
                barriers. A query facing away from its target regresses the reversed heading.
            global_num_pos: Normalize the matched losses by the mean positive count over the
                ranks instead of the count of each rank.
        """
        super().__init__()
        if lidar_sampling not in {"sparse", "sparse_bev", "bev", "bev_attention"}:
            raise ValueError(
                "lidar_sampling must be sparse, sparse_bev, bev or bev_attention, got "
                f"{lidar_sampling}."
            )
        num_bev_sources = int(bev is not None) + int(bev_cell_size is not None)
        if num_bev_sources != int(lidar_sampling != "sparse"):
            raise ValueError(
                f"lidar_sampling {lidar_sampling} samples BEV maps with exactly one of bev and "
                "bev_cell_size, and none without."
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
        if (yaw_free_classes or yaw_reversible_classes) and class_names is None:
            raise ValueError(
                "Sparse4D yaw_free_classes and yaw_reversible_classes need class_names."
            )
        unknown = set(yaw_free_classes) | set(yaw_reversible_classes)
        unknown -= set(class_names or ())
        if unknown:
            raise ValueError(f"Unknown classes in the yaw settings: {sorted(unknown)}.")
        self.register_buffer(
            "yaw_free",
            torch.tensor([name in yaw_free_classes for name in class_names or [""] * num_classes]),
            persistent=False,
        )
        self.register_buffer(
            "yaw_reversible",
            torch.tensor(
                [name in yaw_reversible_classes for name in class_names or [""] * num_classes]
            ),
            persistent=False,
        )
        self.global_num_pos = global_num_pos
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.cls_cost_weight = cls_cost_weight
        self.box_cost_weight = box_cost_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_box_weight = loss_box_weight
        self.loss_cns_weight = loss_cns_weight
        self.loss_yns_weight = loss_yns_weight
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.num_output = num_output
        self.score_threshold = score_threshold
        if nms_group_configs is not None and class_names is None:
            raise ValueError("Sparse4D nms_group_configs need class_names.")
        self.nms_groups = (
            None
            if nms_group_configs is None
            else resolve_nms_groups(
                nms_group_configs, class_names, default_radius=0.0, default_max_size=num_output
            )
        )
        self.dn_query_budget = dn_query_budget
        self.dn_noise_center = dn_noise_center
        self.dn_noise_center_neg = dn_noise_center_neg
        self.dn_noise_size = dn_noise_size
        self.dn_noise_yaw = dn_noise_yaw
        self.dn_noise_velocity = dn_noise_velocity
        self.register_buffer(
            "reg_weights", torch.tensor(list(reg_weights), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "match_reg_weights",
            torch.tensor(list(match_reg_weights), dtype=torch.float32),
            persistent=False,
        )

        self.instance_bank = InstanceBank(
            num_anchor=num_anchor,
            embed_dims=embed_dims,
            anchor_file=anchor_file,
            num_temp_instances=num_temp_instances,
            default_time_interval=default_time_interval,
            min_time_interval=min_time_interval,
            max_time_interval=max_time_interval,
            confidence_decay=confidence_decay,
        )
        self.tracker: InstanceTracker | None = None
        if num_temp_instances > 0:
            if min_observed_points is None or track_birth_score is None:
                raise ValueError(
                    "A temporal Sparse4D head tracks its instances, set min_observed_points and "
                    "track_birth_score."
                )
            self.tracker = InstanceTracker(
                num_temp_instances, min_observed_points, track_birth_score
            )
        # CMT shares one positional embedding between the queries, the cells and the blocks
        position_embedding = (
            CMTPositionEmbedding(embed_dims, point_cloud_range)
            if lidar_sampling == "bev_attention"
            else None
        )
        self._build_decoder(
            embed_dims=embed_dims,
            num_decoder=num_decoder,
            num_heads=num_heads,
            num_groups=num_groups,
            feedforward_channels=feedforward_channels,
            dropout=dropout,
            attn_drop=attn_drop,
            temporal=num_temp_instances > 0,
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            feature_strides=feature_strides,
            anchor_encoder=SparseBox3DEncoder(embed_dims),
            kps_generator=lambda: SparseBox3DKeyPointsGenerator(
                embed_dims=embed_dims,
                fix_scale=fix_scale,
                num_learnable_pts=num_learnable_pts,
            ),
            refine=lambda: SparseBox3DRefinementModule(
                embed_dims=embed_dims, num_classes=num_classes
            ),
            bev=bev,
            bev_cell_size=bev_cell_size,
            bev_aggregation=lambda: None
            if lidar_sampling in {"sparse", "bev_attention"}
            else BEVDeformableAggregation(
                embed_dims=embed_dims,
                bev_channels=bev_channels,
                num_groups=num_groups,
                num_levels=1 if bev is None else bev.num_levels,
                num_points=len(fix_scale) + num_learnable_pts,
                attn_drop=attn_drop,
            ),
            sample_sparse=lidar_sampling in {"sparse", "sparse_bev"},
            aggregation=aggregation,
            bev_attention=None
            if lidar_sampling != "bev_attention"
            else lambda: BEVCrossAttention(
                embed_dims=embed_dims,
                bev_channels=bev_channels,
                num_heads=num_heads,
                position_embedding=position_embedding,
                dropout=dropout,
                attn_drop=attn_drop,
            ),
        )
        self.loss_cls = SigmoidFocalLoss()
        self.loss_yns = GaussianFocalLoss()
        self._export_mode = False

    def forward(
        self,
        lidar_features: LidarFeatures,
        timestamps_seconds: torch.Tensor | None = None,
        lidar2global: torch.Tensor | None = None,
        scene_tokens: Sequence[str] | None = None,
        gt_boxes: Sequence[torch.Tensor] | None = None,
        gt_labels: Sequence[torch.Tensor] | None = None,
        gt_instance_ids: Sequence[torch.Tensor] | None = None,
        gt_num_points: Sequence[torch.Tensor] | None = None,
        context: Any = None,
    ) -> Sparse4DHeadOutputs:
        """Run the decoder over the sampled lidar features.

        A temporal head also advances its tracks. With the ground truth it records which boxes
        were observed in the stream so far, only those feed the denoising queries and the loss.

        Args:
            lidar_features: Features, voxel coordinates and batch offsets of the sampled stages,
                from the finest to the coarsest, or the dense BEV map of a BEV only head.
            timestamps_seconds: Frame timestamps ``(B,)`` in seconds, temporal
                mode only.
            lidar2global: Poses ``(B, 4, 4)`` with the augmentations folded in,
                temporal mode only.
            scene_tokens: Scene identifiers of the frames, temporal mode only.
            gt_boxes: Ground truth boxes ``(M, >=9)`` of every sample, read by
                the denoising queries of training steps only.
            gt_labels: Ground truth labels ``(M,)`` of every sample.
            gt_instance_ids: Instance id ``(M,)`` of every ground truth box, temporal mode only.
            gt_num_points: Point count ``(M,)`` of every ground truth box, temporal mode only.
            context: Frame context of a subclass head, None for the lidar head.

        Returns:
            Per layer outputs of the regular queries and, in training steps
            with denoising, of the denoising queries. In export mode only the
            last layer.

        Raises:
            ValueError: If a denoising training step misses the ground truth.
        """
        batch_size = self._batch_size(lidar_features)
        instance_feature, anchor, time_interval, temp_feature, temp_anchor = self.instance_bank.get(
            batch_size, timestamps_seconds, lidar2global, scene_tokens
        )
        num_query = anchor.shape[1]
        if self.tracker is not None:
            self.tracker.begin_frame(
                self.instance_bank.lane_continues, self.instance_bank.cached_query_index
            )
            if gt_instance_ids is not None and gt_num_points is not None:
                observed = self.tracker.observe(gt_instance_ids, gt_num_points)
                gt_boxes = [boxes[mask] for boxes, mask in zip(gt_boxes, observed, strict=True)]
                gt_labels = [labels[mask] for labels, mask in zip(gt_labels, observed, strict=True)]

        denoising = None
        if self.training and self.dn_query_budget > 0:
            if gt_boxes is None or gt_labels is None:
                raise ValueError("Denoising training steps need the ground truth boxes.")
            denoising = build_denoising_queries(
                gt_boxes,
                gt_labels,
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

        merge = self.instance_bank.update if temp_feature is not None else None
        decoded = self._decode(
            self.levels(lidar_features),
            instance_feature,
            anchor,
            time_interval,
            temp_feature,
            temp_anchor,
            merge,
            denoising,
            context,
            self.bev_levels(lidar_features),
        )
        if temp_feature is not None:
            self.instance_bank.cache(
                decoded.instance_feature,
                decoded.anchor,
                decoded.cls_logits,
                timestamps_seconds,
                lidar2global,
                scene_tokens,
            )

        layers = slice(-1, None) if self._export_mode else slice(None)
        classification = torch.stack(decoded.classification[layers], dim=1)
        prediction = torch.stack(decoded.prediction[layers], dim=1)
        quality = torch.stack(decoded.quality[layers], dim=1)
        instance_ids = None
        if self.tracker is not None:
            scores = self._query_scores(
                classification[:, -1, :num_query], quality[:, -1, :num_query]
            )
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

    def _refine(
        self,
        block_index: int,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Refine the boxes of one block, see :meth:`SparseQueryHead._refine`."""
        return self.refine[block_index](instance_feature, anchor, anchor_embed, time_interval)

    @torch.no_grad()
    def _assign_targets(
        self,
        cls_logits: torch.Tensor,
        pred_anchor: torch.Tensor,
        ground_truth: Sequence[EncodedGroundTruth],
        instance_ids: Sequence[torch.Tensor],
        slot_owners: torch.Tensor | None,
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
            channel_valid = sample_gt.channel_valid.clone()
            channel_valid[self.yaw_free[sample_gt.labels], _SIN_YAW : _COS_YAW + 1] = False
            prob = cls_logits[sample_index].float().sigmoid()
            alpha, gamma = 0.25, 2.0
            pos_cost = alpha * ((1.0 - prob) ** gamma) * (-(prob + 1e-8).log())
            neg_cost = (1.0 - alpha) * (prob**gamma) * (-(1.0 - prob + 1e-8).log())
            cls_cost = (pos_cost - neg_cost)[:, sample_gt.labels]

            box_delta = pred_anchor[sample_index].float()[:, None] - sample_gt.anchor[None]
            box_cost = (box_delta.abs() * self.match_reg_weights * channel_valid[None]).sum(dim=-1)

            cost = self.cls_cost_weight * cls_cost + self.box_cost_weight * box_cost
            cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
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
            box_weights[sample_index, pred_indices] = self.reg_weights * channel_valid[
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
        quality: torch.Tensor,
        pred_anchor: torch.Tensor,
        targets: Sparse4DTargets,
        reg_mask: torch.Tensor,
        avg_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Supervise the centerness and the yawness of the regressed queries.

        Args:
            quality: Quality logits ``(B, N, 2)``.
            pred_anchor: Predicted anchor states ``(B, N, 10)``.
            targets: Matching targets of the layer.
            reg_mask: Queries that get regression and quality losses ``(B, N)``.
            avg_factor: Normalization factor of the centerness loss.

        Returns:
            Tuple of the centerness and the yawness loss.
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
        loss_yns = self.loss_yns(quality[..., _YNS], yaw_agrees, weights=weights)
        return loss_cns, loss_yns

    def assign(
        self,
        outputs: Sparse4DHeadOutputs,
        gt_boxes: Sequence[torch.Tensor],
        gt_labels: Sequence[torch.Tensor],
        gt_instance_ids: Sequence[torch.Tensor],
        lanes: Sequence[int],
    ) -> list[Sparse4DTargets]:
        """Match the queries of every layer to the ground truth one to one.

        A temporal head keeps the boxes observed in the stream so far and lets its slots follow
        their instances from the second layer on, where the temporal slots lead the queries. The
        instances the last layer assigns become the owners the slots follow in the next frame.

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
        slot_owners = None
        if self.tracker is not None:
            if self.tracker.frame_observed is None:
                raise ValueError(
                    "A temporal head loss needs forward() to receive the instance ids and the "
                    "point counts of the ground truth."
                )
            observed = [self.tracker.frame_observed[lane] for lane in lanes]
            gt_boxes = [boxes[mask] for boxes, mask in zip(gt_boxes, observed, strict=True)]
            gt_labels = [labels[mask] for labels, mask in zip(gt_labels, observed, strict=True)]
            gt_instance_ids = [
                ids[mask] for ids, mask in zip(gt_instance_ids, observed, strict=True)
            ]
            slot_owners = self.tracker.slot_owners(lanes)
        ground_truth = [
            encode_ground_truth(boxes, labels, device, torch.float32)
            for boxes, labels in zip(gt_boxes, gt_labels, strict=True)
        ]
        targets = [
            self._assign_targets(
                outputs.classification[:, layer_index],
                outputs.prediction[:, layer_index],
                ground_truth,
                gt_instance_ids,
                None if layer_index == 0 else slot_owners,
            )
            for layer_index in range(outputs.classification.shape[1])
        ]
        if self.tracker is not None:
            scores = self._query_scores(
                outputs.classification[:, -1].detach(), outputs.quality[:, -1].detach()
            )
            self.tracker.set_query_owners(
                lanes, targets[-1].assigned_instance, scores.max(dim=-1).values
            )
        return targets

    def loss_from_targets(
        self,
        outputs: Sparse4DHeadOutputs,
        targets: Sequence[Sparse4DTargets],
        lanes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Compute the matched losses of every layer from its targets and the denoising losses.

        Args:
            outputs: Raw head outputs from :meth:`forward` of the given lanes.
            targets: Targets of every decoder layer from :meth:`assign`.
            lanes: Batch lane of every sample.

        Returns:
            Loss dictionary with the terms of every layer and their sum under ``loss``.
        """
        loss_dict: dict[str, torch.Tensor] = {}
        num_layers = outputs.classification.shape[1]
        total_pos = 0
        for layer_index, layer_targets in enumerate(targets):
            cls_logits = outputs.classification[:, layer_index]
            pred_anchor = outputs.prediction[:, layer_index]
            total_pos += layer_targets.num_pos
            avg_factor = self._avg_factor(layer_targets.num_pos, cls_logits.device)
            layer_targets = self._reverse_facing_away(layer_targets, pred_anchor)

            loss_cls = self.loss_cls(
                cls_logits.flatten(end_dim=1).float(),
                layer_targets.cls_targets.flatten(end_dim=1),
                avg_factor=avg_factor,
            )
            confident = cls_logits.detach().max(dim=-1).values.sigmoid() > self.cls_threshold_to_reg
            reg_mask = layer_targets.matched & confident
            box_error = (
                pred_anchor.float() - layer_targets.box_targets
            ).abs() * layer_targets.box_weights
            loss_box = (box_error * reg_mask[..., None]).sum() / avg_factor
            loss_cns, loss_yns = self._quality_losses(
                outputs.quality[:, layer_index], pred_anchor, layer_targets, reg_mask, avg_factor
            )

            prefix = "layer_-1" if layer_index == num_layers - 1 else f"layer_{layer_index}"
            loss_dict[f"{prefix}_loss_cls"] = self.loss_cls_weight * loss_cls
            loss_dict[f"{prefix}_loss_bbox"] = self.loss_box_weight * loss_box
            loss_dict[f"{prefix}_loss_cns"] = self.loss_cns_weight * loss_cns
            loss_dict[f"{prefix}_loss_yns"] = self.loss_yns_weight * loss_yns

        if self.tracker is not None:
            loss_dict["num_followed"] = outputs.classification.new_tensor(
                float((self.tracker.slot_owners(lanes) >= 0).sum())
            )
        loss_dict["num_pos"] = outputs.classification.new_tensor(total_pos / max(num_layers, 1))
        if outputs.denoising is not None:
            loss_dict.update(self._denoising_losses(outputs.denoising))
        loss_dict["loss"] = sum(value for key, value in loss_dict.items() if "loss" in key)
        return loss_dict

    def _avg_factor(self, num_pos: int, device: torch.device) -> float:
        """Normalization of the matched losses, at least one.

        Args:
            num_pos: Matched queries of this rank.
            device: Device of the reduction.

        Returns:
            The positive count of this rank, or the mean over the ranks with global_num_pos.
        """
        if not self.global_num_pos or not torch.distributed.is_initialized():
            return float(max(num_pos, 1))
        count = torch.tensor(float(num_pos), device=device)
        torch.distributed.all_reduce(count)
        return max(float(count) / torch.distributed.get_world_size(), 1.0)

    def _reverse_facing_away(
        self, targets: Sparse4DTargets, pred_anchor: torch.Tensor
    ) -> Sparse4DTargets:
        """Reverse the heading target of the yaw reversible queries facing away from it.

        Args:
            targets: Targets of one decoder layer.
            pred_anchor: Predicted anchors of the layer ``(B, N, 10)``.

        Returns:
            The targets with the reversed headings.
        """
        if not bool(self.yaw_reversible.any()):
            return targets
        reversible = (targets.cls_targets[..., self.yaw_reversible] > 0).any(dim=-1)
        heading = targets.box_targets[..., _SIN_YAW : _COS_YAW + 1]
        facing_away = (pred_anchor.detach().float()[..., _SIN_YAW : _COS_YAW + 1] * heading).sum(
            -1
        ) < 0
        box_targets = targets.box_targets.clone()
        box_targets[..., _SIN_YAW : _COS_YAW + 1] = torch.where(
            (reversible & facing_away)[..., None], -heading, heading
        )
        return replace(targets, box_targets=box_targets)

    def loss(
        self,
        outputs: Sparse4DHeadOutputs,
        gt_boxes: Sequence[torch.Tensor],
        gt_labels: Sequence[torch.Tensor],
        gt_instance_ids: Sequence[torch.Tensor],
        lanes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Compute the matched and the denoising losses of every layer.

        Denoising queries are graded against their source box without matching, see
        :meth:`assign` for the matching of the regular queries.

        Args:
            outputs: Raw head outputs from :meth:`forward` of the given lanes.
            gt_boxes: Ground truth boxes ``(M, >=9)`` of every sample.
            gt_labels: Ground truth labels ``(M,)`` of every sample.
            gt_instance_ids: Instance id ``(M,)`` of every ground truth box.
            lanes: Batch lane of every sample.

        Returns:
            Loss dictionary with the terms of every layer and their sum under ``loss``.
        """
        targets = self.assign(outputs, gt_boxes, gt_labels, gt_instance_ids, lanes)
        return self.loss_from_targets(outputs, targets, lanes)

    def _denoising_losses(self, denoising: Sparse4DDenoisingOutputs) -> dict[str, torch.Tensor]:
        """Compute the classification and regression losses of the denoising queries.

        Args:
            denoising: Denoising outputs and targets of the batch.

        Returns:
            Denoising loss terms of every layer and the denoising positive count.
        """
        loss_dict: dict[str, torch.Tensor] = {}
        num_layers = denoising.classification.shape[1]
        dn_avg_factor = float(max(int((denoising.cls_targets.sum(dim=-1) > 0).sum()), 1))
        cls_weights = denoising.cls_weights.flatten()[:, None].expand(-1, self.num_classes)
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

    def query_scores(self, outputs: Sparse4DHeadOutputs) -> torch.Tensor:
        """Detection score of every query and class of the last layer ``(B, N, num_classes)``.

        Args:
            outputs: Raw head outputs from :meth:`forward`.

        Returns:
            The class probabilities weighted by the centerness.
        """
        return self._query_scores(outputs.classification[:, -1], outputs.quality[:, -1])

    @staticmethod
    def _query_scores(cls_logits: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        """Class probability times centerness of every query and class ``(B, N, C)``."""
        return cls_logits.float().sigmoid() * quality[..., _CNS].float().sigmoid()[..., None]

    def decode_outputs(self, outputs: Detection3DHeadOutputs) -> ModelPredictions:
        """Decode the last layer into metric space boxes.

        The score is class probability times centerness. A query may yield boxes of several
        classes, ranked by score, every box carrying the track id of its query. The NMS groups
        suppress across their classes and rank the confident tracked boxes first, so a new box
        never replaces a track.

        Args:
            outputs: Raw head outputs from :meth:`forward`.

        Returns:
            One Detection3DSamplePredictions per sample of the batch, wrapped in ModelPredictions.
        """
        outputs = outputs.sparse4d_head()
        # Mixed precision evaluation hands over float16 outputs, the boxes decode in float32
        scores = self._query_scores(outputs.classification[:, -1], outputs.quality[:, -1])
        pred_anchor = outputs.prediction[:, -1].float()

        results = []
        for sample_index, (sample_scores, sample_anchor) in enumerate(zip(scores, pred_anchor)):
            keep_count = min(self.num_output, sample_scores.numel())
            top_scores, top_pairs = sample_scores.flatten().topk(keep_count)
            top_labels = top_pairs % self.num_classes
            top_indices = torch.div(top_pairs, self.num_classes, rounding_mode="floor")
            boxes = decode_anchor_space_to_boxes(sample_anchor[top_indices])
            if self.score_threshold is not None:
                keep = top_scores > self.score_threshold
                top_scores = top_scores[keep]
                top_labels = top_labels[keep]
                top_indices = top_indices[keep]
                boxes = boxes[keep]
            track_ids = (
                None
                if outputs.instance_ids is None
                else outputs.instance_ids[sample_index, top_indices]
            )
            if self.nms_groups is not None:
                tracked = (
                    torch.zeros_like(top_scores)
                    if track_ids is None
                    else ((track_ids >= 0) & (top_scores >= self.tracker.track_birth_score)).float()
                )
                ranking = top_scores + tracked
                keep = filter_nms_groups(
                    boxes[None, :, :2],
                    ranking[None],
                    top_labels[None],
                    torch.ones_like(ranking[None], dtype=torch.bool),
                    self.nms_groups,
                    self.num_classes,
                )[0]
                top_scores = top_scores[keep]
                top_labels = top_labels[keep]
                boxes = boxes[keep]
                track_ids = None if track_ids is None else track_ids[keep]
            results.append(
                Detection3DSamplePredictions(
                    bboxes_3d=boxes,
                    scores_3d=top_scores,
                    labels_3d=top_labels,
                    instance_ids=track_ids,
                )
            )
        return ModelPredictions(detection3d_predictions=results)
