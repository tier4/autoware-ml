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

"""Fold BatchNorm into the preceding convolution or linear layer.

The fold is for paths that must see the weight the deployed graph carries: a weight
quantizer calibrating before the runtime fuses the BatchNorm, or a plugin node that takes
the folded bias. Plain float exports keep their BatchNorm nodes, since the runtimes fuse
them and folding into fp16 weights loses precision. Pairs come only from structure that
proves the dataflow, never from registration order.
"""

import logging
from itertools import pairwise

import torch
from torch import nn

from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE

if IS_SPCONV_AVAILABLE:
    from spconv.pytorch import SparseSequential
    from spconv.pytorch.conv import SparseConvolution

logger = logging.getLogger(__name__)


def fuse_bn_weights(
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    bn_mean: torch.Tensor,
    bn_var: torch.Tensor,
    bn_eps: float,
    bn_weight: torch.Tensor | None,
    bn_bias: torch.Tensor | None,
    is_transposed: bool = False,
) -> tuple[nn.Parameter, nn.Parameter]:
    """Fold a BatchNorm into a convolution's weight and bias.

    With ``s = gamma / sqrt(var + eps)``: ``W' = W * s`` per output channel and
    ``b' = (b - mean) * s + beta``. The output channels are dim 0 of the weight, or dim 1
    with ``is_transposed`` (``ConvTranspose2d`` weights are ``[C_in, C_out, ...]``).
    """
    if conv_bias is None:
        conv_bias = torch.zeros_like(bn_mean)
    if bn_weight is None:
        bn_weight = torch.ones_like(bn_mean)
    if bn_bias is None:
        bn_bias = torch.zeros_like(bn_mean)

    bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)
    scale = bn_weight * bn_var_rsqrt
    if is_transposed:
        shape = [1, -1] + [1] * (conv_weight.ndim - 2)
    else:
        shape = [-1] + [1] * (conv_weight.ndim - 1)

    fused_weight = conv_weight * scale.reshape(shape)
    fused_bias = (conv_bias - bn_mean) * scale + bn_bias
    return nn.Parameter(fused_weight.contiguous()), nn.Parameter(fused_bias.contiguous())


def fuse_conv_bn(conv: nn.Module, bn: nn.Module):
    """Fold ``bn`` into ``conv`` in place; ``bn`` must then be replaced by an identity."""
    assert not conv.training and not bn.training, "Fusion only works in eval mode"
    if bn.running_mean is None or bn.running_var is None:
        raise ValueError(
            f"Cannot fold {type(bn).__name__} without running statistics "
            "(track_running_stats=False): there is nothing to fold into the convolution."
        )

    # The only transposed type in _CONV_TO_BN.
    is_transposed = isinstance(conv, nn.ConvTranspose2d)

    conv.weight, conv.bias = fuse_bn_weights(
        conv.weight,
        conv.bias,
        bn.running_mean,
        bn.running_var,
        bn.eps,
        bn.weight,
        bn.bias,
        is_transposed=is_transposed,
    )


#: Set to ``True`` on a container whose ``forward`` applies its children in registration
#: order, like ``nn.Sequential``, so that adjacent children can be paired.
SEQUENTIAL_MARKER = "applies_children_in_order"

#: Set on a block to name the ``(conv, bn)`` child pairs its ``forward`` connects directly
#: (``bn(conv(x))``, a reshape between them allowed), e.g. ``(("conv", "norm"),)``. A named
#: BatchNorm that is an ``nn.Identity`` is skipped.
FUSION_PAIRS_MARKER = "bn_fusion_pairs"

#: Conv/Linear type -> the BatchNorm type that folds into it.
_CONV_TO_BN: dict[type, type] = {
    nn.Conv1d: nn.BatchNorm1d,
    nn.Conv2d: nn.BatchNorm2d,
    nn.ConvTranspose2d: nn.BatchNorm2d,
    nn.Linear: nn.BatchNorm1d,
}

#: Containers whose ``forward`` applies their children in registration order.
_SEQUENTIAL_CONTAINERS: tuple[type, ...] = (nn.Sequential,)

if IS_SPCONV_AVAILABLE:
    # Sparse convolutions are normalized by BatchNorm1d over the features, and their weight
    # ``[C_out, k1, k2, k3, C_in]`` has the output channels on dim 0, like a dense one.
    _CONV_TO_BN[SparseConvolution] = nn.BatchNorm1d
    _SEQUENTIAL_CONTAINERS = (*_SEQUENTIAL_CONTAINERS, SparseSequential)


def _applies_children_in_order(module: nn.Module) -> bool:
    return isinstance(module, _SEQUENTIAL_CONTAINERS) or bool(
        getattr(module, SEQUENTIAL_MARKER, False)
    )


def _is_fusible_pair(left: nn.Module, right: nn.Module) -> bool:
    """Whether ``left`` is a Conv/Linear whose output channels ``right`` normalizes."""
    for conv_type, bn_type in _CONV_TO_BN.items():
        if isinstance(left, conv_type) and isinstance(right, bn_type):
            out_channels = left.out_features if isinstance(left, nn.Linear) else left.out_channels
            return out_channels == right.num_features
    return False


def find_conv_bn_pairs(model: nn.Module) -> list[tuple[str, str]]:
    """Return the ``(conv, bn)`` module paths the model's structure proves are connected.

    A pair is two adjacent children of an in-order container (``nn.Sequential`` or
    ``applies_children_in_order``) or a pair a block names in ``bn_fusion_pairs``. Adjacent
    children of any other parent are not paired, since its ``forward`` may put an activation
    between them or not connect them at all; they are logged so the block can declare them.
    """
    pairs: list[tuple[str, str]] = []
    undeclared: list[str] = []
    for parent_name, parent in model.named_modules():
        prefix = f"{parent_name}." if parent_name else ""
        declared = getattr(parent, FUSION_PAIRS_MARKER, None)
        if declared is not None:
            for conv_attr, bn_attr in declared:
                conv, bn = parent._modules.get(conv_attr), parent._modules.get(bn_attr)
                if isinstance(bn, nn.Identity):
                    continue  # built without a BatchNorm, or already folded
                if conv is None or bn is None or not _is_fusible_pair(conv, bn):
                    raise ValueError(
                        f"{type(parent).__name__}.bn_fusion_pairs names ({conv_attr!r}, "
                        f"{bn_attr!r}) under {parent_name or '<root>'!r}, which is not a "
                        "Conv/Linear + matching BatchNorm child pair."
                    )
                pairs.append((f"{prefix}{conv_attr}", f"{prefix}{bn_attr}"))
            continue
        children = [(name, child) for name, child in parent._modules.items() if child is not None]
        for (left_name, left), (right_name, right) in pairwise(children):
            if not _is_fusible_pair(left, right):
                continue
            if _applies_children_in_order(parent):
                pairs.append((f"{prefix}{left_name}", f"{prefix}{right_name}"))
            else:
                undeclared.append(f"{prefix}{left_name} + {prefix}{right_name}")
    if undeclared:
        logger.warning(
            "BN fusion skipped %d adjacent Conv/Linear + BatchNorm registration(s) whose "
            "parent neither applies its children in order nor declares bn_fusion_pairs "
            "(registration order does not prove dataflow): %s. If the block computes "
            "bn(conv(x)), declare `bn_fusion_pairs = ((conv_attr, bn_attr),)` on it.",
            len(undeclared),
            ", ".join(undeclared),
        )
    return pairs


def _get_parent_module(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    """Return ``(parent module, attribute name)`` for a dotted module path."""
    parent_name, _, attr_name = name.rpartition(".")
    # get_submodule also resolves numeric child names of containers without __getitem__.
    return (model.get_submodule(parent_name) if parent_name else model), attr_name


def _replace_bn_with_identity(model: nn.Module, bn_name: str):
    parent, attr = _get_parent_module(model, bn_name)
    # add_module also covers numeric names and containers without __setitem__.
    parent.add_module(attr, nn.Identity())


def fuse_model_bn(model: nn.Module) -> list[tuple[str, str]]:
    """Fold every pair :func:`find_conv_bn_pairs` finds, in place, and return the pairs.

    Switches the model to eval mode; each folded BatchNorm becomes an ``nn.Identity``.
    """
    model.eval()

    pairs = find_conv_bn_pairs(model)
    if len(pairs) == 0:
        logger.info("No Conv-BN pairs found to fuse")
        return pairs

    modules_dict = dict(model.named_modules())

    for conv_name, bn_name in pairs:
        conv = modules_dict[conv_name]
        bn = modules_dict[bn_name]
        fuse_conv_bn(conv, bn)
        _replace_bn_with_identity(model, bn_name)

    logger.info("Fused %d Conv-BN pairs", len(pairs))
    return pairs
