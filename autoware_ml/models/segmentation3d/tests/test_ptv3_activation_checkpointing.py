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

"""Activation checkpointing of the PTv3 blocks must not change the training function."""

from __future__ import annotations

import torch

from autoware_ml.models.segmentation3d.encoders.ptv3 import PointTransformerV3Encoder


def _encoder(checkpointing: bool) -> PointTransformerV3Encoder:
    torch.manual_seed(0)
    return PointTransformerV3Encoder(
        in_channels=5,
        order=("z", "hilbert"),
        stride=(2,),
        enc_depths=(2, 2),
        enc_channels=(8, 16),
        enc_num_head=(1, 2),
        enc_patch_size=(8, 8),
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        pre_norm=True,
        shuffle_orders=False,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        # Attention-only blocks: the checkpoint wraps attention and MLP, and the sparse
        # convolutions have no CPU kernels in every environment.
        enc_conv=False,
        activation_checkpointing=checkpointing,
    )


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    grid_coord = torch.unique(torch.randint(0, 32, (400, 3), generator=generator), dim=0)
    count = grid_coord.shape[0]
    feat = torch.randn(count, 5, generator=generator)
    return {
        "feat": feat,
        "coord": grid_coord.float() * 0.1,
        "grid_coord": grid_coord,
        "offset": torch.tensor([count // 2, count]),
        "grid_size": torch.tensor(0.1),
    }


def _run(checkpointing: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
    encoder = _encoder(checkpointing).train()
    inputs = _inputs()
    inputs["feat"].requires_grad_(True)
    point = encoder(inputs)
    # The recomputation in the backward pass rewrites the point's features, so read them first.
    output = point.feat.detach().clone()
    loss = point.feat.float().square().mean()
    loss.backward()
    grads = [p.grad.clone() for p in encoder.parameters() if p.grad is not None]
    return output, grads


def test_checkpointed_blocks_reproduce_outputs_and_gradients() -> None:
    plain_out, plain_grads = _run(False)
    ckpt_out, ckpt_grads = _run(True)
    torch.testing.assert_close(ckpt_out, plain_out)
    assert len(plain_grads) == len(ckpt_grads) > 0
    for plain, ckpt in zip(plain_grads, ckpt_grads):
        torch.testing.assert_close(ckpt, plain)


def test_checkpointing_is_inactive_in_eval_mode() -> None:
    encoder = _encoder(True).eval()
    assert all(
        block.activation_checkpointing
        for block in encoder.modules()
        if hasattr(block, "_attention_and_mlp")
    )
    with torch.no_grad():
        point = encoder(_inputs())
    assert torch.isfinite(point.feat).all()
