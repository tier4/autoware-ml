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

"""Helpers of the distributed (DDP) training and evaluation runs."""

from __future__ import annotations

from jaxtyping import Float, Num
import torch


def reduce_mean_count(
    value: Num[torch.Tensor, "*counts"], min_total: float = 1.0
) -> Float[torch.Tensor, "*counts"]:
    """Average a count over the DDP ranks, flooring the sum before the division.

    A loss summed over the objects of a rank is divided by this so that its gradient matches
    single-device training. DDP averages the gradients of the ranks in the backward pass,
    which is one more division by the world size: dividing the loss by the count summed over
    the ranks would leave a gradient of ``1 / (N * world_size)`` instead of ``1 / N`` for the
    ``N`` objects of the step, so the summed count is divided by the world size first.

    The floor applies to the sum, as it would on one device. (The mmdetection heads floor the
    mean of ``reduce_mean`` instead, which halves the only positive of a two-rank step.) The
    ``all_reduce`` is a collective, so every rank has to call it the same number of times per
    step.

    A loss divided by this count is ``world_size`` times the rank's share of the per-object loss
    of the step; it is that per-object loss only once averaged over the ranks, as DDP does for
    the gradients and ``sync_dist`` does for the logged values.

    Args:
        value: Count or counts of this rank, reduced element by element.
        min_total: Floor of the count summed over the ranks.

    Returns:
        The floored mean count as float; without a process group, the floored count of this
        rank.
    """
    total = value.float()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        total = total.clone()
        torch.distributed.all_reduce(total)
        return total.clamp_(min=min_total).div_(torch.distributed.get_world_size())
    return total.clamp(min=min_total)
