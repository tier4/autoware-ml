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

"""Activation checkpointing of a block that keeps its batch normalization statistics honest.

A non-reentrant checkpoint drops the activations of a block in the forward pass and runs the
block again in the backward pass to rebuild them. In train mode that second run is a second
batch normalization update: every ``BatchNorm`` of the block moves its running mean and
variance twice per step and counts two batches, which doubles the effective momentum of the
model. The recomputation here runs with the running statistics frozen, so the block
normalizes with the batch statistics exactly as the forward did and the statistics move once.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.checkpoint import checkpoint


@contextmanager
def frozen_batch_norm_statistics(module: nn.Module) -> Iterator[None]:
    """Keep the running statistics of every batch normalization of a module where they are.

    The normalization still uses the batch statistics in train mode and the autograd graph is
    the one of the original forward. Only the running mean, the running variance and the batch
    count stay untouched: the momentum is zero while the context is open, which keeps the
    running statistics, and the batch counts are restored on exit.

    Args:
        module: Module whose batch normalization layers are frozen while the context is open.

    Yields:
        None
    """
    tracking = [
        layer
        for layer in module.modules()
        if isinstance(layer, _BatchNorm) and layer.track_running_stats
    ]
    momenta = [layer.momentum for layer in tracking]
    batch_counts = [
        None if layer.num_batches_tracked is None else layer.num_batches_tracked.clone()
        for layer in tracking
    ]
    for layer in tracking:
        layer.momentum = 0.0
    try:
        yield
    finally:
        for layer, momentum, batch_count in zip(tracking, momenta, batch_counts):
            layer.momentum = momentum
            if batch_count is not None:
                layer.num_batches_tracked.copy_(batch_count)


def checkpoint_block(block: nn.Module, *inputs: Any) -> Any:
    """Run a block under a non-reentrant checkpoint whose recomputation updates no statistics.

    Args:
        block: Module to run.
        *inputs: Inputs of the block.

    Returns:
        The outputs of the block.
    """
    return checkpoint(
        block,
        *inputs,
        use_reentrant=False,
        context_fn=lambda: (nullcontext(), frozen_batch_norm_statistics(block)),
    )


def checkpointing_active(module: nn.Module, activation_checkpointing: bool) -> bool:
    """Whether a forward of the module should checkpoint its blocks.

    A checkpoint only pays off when a backward pass follows, so evaluation, inference and
    export run the blocks directly.

    Args:
        module: Module about to run its forward.
        activation_checkpointing: Whether the module was configured to checkpoint.

    Returns:
        bool: True when the blocks are to be checkpointed.
    """
    return activation_checkpointing and module.training and torch.is_grad_enabled()
