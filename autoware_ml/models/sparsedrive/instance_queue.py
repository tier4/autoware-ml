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

"""History of the detection queries, per batch lane."""

from __future__ import annotations

from typing import NamedTuple

import torch
from jaxtyping import Bool, Float, Int64, Shaped

from autoware_ml.models.sparsedrive.sparse4d_head import ANCHOR_DIM, InstanceBank


class QueueState(NamedTuple):
    """History of every query of every lane.

    Attributes:
        feature: Detached features of every entry ``(B, L, Q, C)``, newest entry first.
        anchor: Anchors of every entry in the frame of the latest advance ``(B, L, Q, 10)``.
        query_map: Query of every entry each current query maps to ``(B, L, Q)``, -1 for none.
        count: Entries every lane holds ``(B,)``.
        age: Frames since the newest entry of every lane ``(B,)``.
    """

    feature: Float[torch.Tensor, "batch_size history_length num_queries embed_dims"]
    anchor: Float[torch.Tensor, "batch_size history_length num_queries anchor_dim"]
    query_map: Int64[torch.Tensor, "batch_size history_length num_queries"]
    count: Int64[torch.Tensor, " batch_size"]
    age: Int64[torch.Tensor, " batch_size"]

    @staticmethod
    def empty(
        batch_size: int,
        length: int,
        num_queries: int,
        embed_dims: int,
        stride: int,
        device: torch.device,
    ) -> QueueState:
        """State of lanes without history, due for an entry at the next push.

        Args:
            batch_size: Number of lanes.
            length: Entries per lane.
            num_queries: Detection queries per frame.
            embed_dims: Feature width of the queries.
            stride: Frames between two pushed entries.
            device: Device of the state.

        Returns:
            The empty state.
        """
        return QueueState(
            feature=torch.zeros(batch_size, length, num_queries, embed_dims, device=device),
            anchor=torch.zeros(batch_size, length, num_queries, ANCHOR_DIM, device=device),
            query_map=torch.full(
                (batch_size, length, num_queries), -1, dtype=torch.long, device=device
            ),
            count=torch.zeros(batch_size, dtype=torch.long, device=device),
            age=torch.full((batch_size,), stride, dtype=torch.long, device=device),
        )


def advance_queue(
    state: QueueState,
    lane_continues: Bool[torch.Tensor, " batch_size"],
    carried_index: Int64[torch.Tensor, "batch_size num_temp_instances"],
    prev2current: Float[torch.Tensor, "batch_size 4 4"],
    stride: int,
) -> QueueState:
    """Move the history to the queries and into the frame of the new frame.

    Args:
        state: History after the previous frame.
        lane_continues: Whether every lane continues its stream ``(B,)``.
        carried_index: Query of the previous frame every leading temporal slot of this frame comes
            from ``(B, K)``.
        prev2current: Transform from the previous to the current frame ``(B, 4, 4)``.
        stride: Frames between two pushed entries.

    Returns:
        The history of the new frame, empty for the restarted lanes.
    """
    batch_size, length, total = state.query_map.shape
    previous = torch.full((batch_size, total), -1, dtype=torch.long, device=state.query_map.device)
    previous[:, : carried_index.shape[1]] = carried_index
    gathered = state.query_map.gather(2, previous.clamp(min=0)[:, None].expand(-1, length, -1))
    query_map = torch.where(previous[:, None] >= 0, gathered, torch.full_like(gathered, -1))
    restart = ~lane_continues
    anchor = InstanceBank.project_anchor(
        state.anchor.reshape(batch_size, length * total, ANCHOR_DIM),
        state.anchor.new_zeros(batch_size),
        prev2current.to(state.anchor.dtype),
    ).reshape(state.anchor.shape)
    # No in-place writes, a state built under trainer.validate or test is an inference tensor
    return QueueState(
        feature=state.feature,
        anchor=anchor,
        query_map=torch.where(restart[:, None, None], torch.full_like(query_map, -1), query_map),
        count=torch.where(restart, torch.zeros_like(state.count), state.count),
        age=torch.where(restart, torch.full_like(state.age, stride), state.age + 1),
    )


class QueueHistory(NamedTuple):
    """Entries of every current query, newest first.

    Attributes:
        feature: Features of every entry ``(B, Q, L, C)``.
        anchor: Anchors of every entry ``(B, Q, L, 10)``.
        valid: Whether each entry exists ``(B, Q, L)``.
    """

    feature: Float[torch.Tensor, "batch_size num_queries history_length embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_queries history_length anchor_dim"]
    valid: Bool[torch.Tensor, "batch_size num_queries history_length"]


def queue_history(state: QueueState) -> QueueHistory:
    """Read the entries of every current query, newest first.

    Args:
        state: History of the current frame.

    Returns:
        The entries of every query.
    """
    index = state.query_map.clamp(min=0)
    feature = state.feature.gather(2, index[..., None].expand(-1, -1, -1, state.feature.shape[-1]))
    anchor = state.anchor.gather(2, index[..., None].expand(-1, -1, -1, ANCHOR_DIM))
    entry = torch.arange(state.query_map.shape[1], device=index.device)
    valid = (state.query_map >= 0) & (entry[None, :, None] < state.count[:, None, None])
    return QueueHistory(feature.transpose(1, 2), anchor.transpose(1, 2), valid.transpose(1, 2))


def push_queue(
    state: QueueState,
    feature: Float[torch.Tensor, "batch_size num_queries embed_dims"],
    anchor: Float[torch.Tensor, "batch_size num_queries anchor_dim"],
    stride: int,
) -> QueueState:
    """Store this frame as the newest entry of the lanes whose stride has elapsed.

    Args:
        state: History of the current frame.
        feature: Features of every query of this frame ``(B, Q, C)``.
        anchor: Anchors of every query of this frame ``(B, Q, 10)``.
        stride: Frames between two pushed entries.

    Returns:
        The history with the new entries.
    """
    due = state.age >= stride
    batch_size, length, total = state.query_map.shape

    def rolled(
        previous: Shaped[torch.Tensor, "batch_size history_length ..."],
        newest: Shaped[torch.Tensor, "batch_size ..."],
    ) -> Shaped[torch.Tensor, "batch_size history_length ..."]:
        shifted = torch.cat([newest[:, None], previous[:, :-1]], dim=1)
        return torch.where(due.view(-1, *([1] * (previous.dim() - 1))), shifted, previous)

    identity = torch.arange(total, device=feature.device).expand(batch_size, -1)
    return QueueState(
        feature=rolled(state.feature, feature.detach()),
        anchor=rolled(state.anchor, anchor.detach()),
        query_map=rolled(state.query_map, identity),
        count=torch.where(due & (state.count < length), state.count + 1, state.count),
        age=torch.where(due, torch.zeros_like(state.age), state.age),
    )
