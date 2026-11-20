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

"""Unit tests for the hybrid encoder of RT-DETR."""

from __future__ import annotations

from collections import Counter
import math
import unittest

import torch

from autoware_ml.models.common.necks.hybrid_encoder import HybridEncoder, RepVggBlock


def _features(height: int, width: int, channels: int = 8) -> list[torch.Tensor]:
    """Three levels at strides 8, 16 and 32 of an image of the given size."""
    torch.manual_seed(0)
    return [torch.randn(2, channels, height // s, width // s) for s in (8, 16, 32)]


class TestHybridEncoder(unittest.TestCase):
    """The hybrid encoder: AIFI on the coarsest level, then CCFF across the levels."""

    def test_returns_hidden_dim_maps_at_the_input_resolutions(self) -> None:
        """
        Input: three levels of a 64x96 image with different channel counts.
        Expected: three maps of hidden_dim channels at the input resolutions.
        Check: the output shapes and out_channels.
        """
        encoder = HybridEncoder(in_channels=(8, 16, 32), hidden_dim=32).eval()
        feats = [
            torch.randn(2, channels, 64 // s, 96 // s)
            for channels, s in ((8, 8), (16, 16), (32, 32))
        ]

        with torch.no_grad():
            outs = encoder(feats)

        self.assertEqual(encoder.out_channels, [32, 32, 32])
        self.assertEqual(
            [tuple(out.shape) for out in outs], [(2, 32, 8, 12), (2, 32, 4, 6), (2, 32, 2, 3)]
        )

    def test_rejects_a_wrong_number_of_levels(self) -> None:
        """
        Input: two feature maps for an encoder of three levels.
        Expected: refused instead of silently dropping a level.
        Check: the ValueError.
        """
        encoder = HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32)

        with self.assertRaises(ValueError):
            encoder(_features(64, 64)[:2])

    def test_rejects_aifi_levels_out_of_range_or_out_of_order(self) -> None:
        """
        Input: use_encoder_idx past the last of three levels, negative, repeated and
            decreasing.
        Expected: each is refused at construction, instead of an IndexError, a level
            attended twice, or transformers swapped against the checkpoint.
        Check: the ValueError.
        """
        for use_encoder_idx in ((3,), (-1,), (2, 2), (2, 1)):
            with (
                self.subTest(use_encoder_idx=use_encoder_idx),
                self.assertRaises(ValueError),
            ):
                HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32, use_encoder_idx=use_encoder_idx)

    def test_runs_without_aifi(self) -> None:
        """
        Input: an encoder with an empty use_encoder_idx.
        Expected: no transformer is built and the CCFF fusion alone returns every level.
        Check: the transformer count and the output shapes.
        """
        encoder = HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32, use_encoder_idx=()).eval()

        with torch.no_grad():
            outs = encoder(_features(64, 64))

        self.assertEqual(len(encoder.encoder), 0)
        self.assertEqual(
            [tuple(out.shape) for out in outs], [(2, 32, 8, 8), (2, 32, 4, 4), (2, 32, 2, 2)]
        )

    def test_cached_position_embedding_matches_the_generated_one(self) -> None:
        """
        Input: an encoder built for a 64x96 input, whose stride-32 level is 2x3.
        Expected: the cached embedding equals the one built on the fly, and it stays out of
            the state dict, so a checkpoint of a model built for another size still loads.
        Check: exact equality and the state dict keys.
        """
        encoder = HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32, eval_spatial_size=(64, 96))

        generated = encoder.build_2d_sincos_position_embedding(3, 2, 32)

        torch.testing.assert_close(encoder.pos_embed2, generated, rtol=0, atol=0)
        self.assertNotIn("pos_embed2", encoder.state_dict())

    def test_position_embedding_encodes_the_row_then_the_column_of_each_token(self) -> None:
        """
        Input: the embedding of a 2x3 map (2 rows, 3 columns) with 8 channels, 2 frequencies
            per quarter.
        Expected: token k, at row k // 3 and column k % 3 as the map is flattened row-major,
            holds sin and cos of its row, then sin and cos of its column.
        Check: every value against the formula.
        """
        # RT-DETRv2 weights expect this layout. Before lyuwenyu/RT-DETR#692 the grid was
        # built as (width, height) and the column came first; both mistakes cancel on square
        # maps, so only a non-square map like this one tells the layouts apart.
        embedding = HybridEncoder.build_2d_sincos_position_embedding(
            width=3, height=2, embed_dim=8, temperature=10000.0
        )

        omegas = (1.0, 1.0 / 100.0)
        expected = torch.tensor(
            [
                [math.sin(row * w) for w in omegas]
                + [math.cos(row * w) for w in omegas]
                + [math.sin(col * w) for w in omegas]
                + [math.cos(col * w) for w in omegas]
                for row, col in (divmod(token, 3) for token in range(6))
            ]
        )
        self.assertEqual(tuple(embedding.shape), (1, 6, 8))
        torch.testing.assert_close(embedding[0], expected)

    def test_cached_embedding_is_used_in_eval_at_its_size_only(self) -> None:
        """
        Input: an encoder cached for 64x96 whose cached embedding is replaced by a marker,
            next to a plain encoder with the same weights, both run on a 64x96 input.
        Expected: eval reads the cache, so the marker changes the output; training builds
            the embedding on the fly, so the outputs stay equal.
        Check: the outputs of both encoders in eval and in train mode.
        """
        cached = HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32, eval_spatial_size=(64, 96))
        plain = HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32)
        plain.load_state_dict(cached.state_dict())
        cached.pos_embed2.fill_(100.0)
        feats = _features(64, 96)

        with torch.no_grad():
            eval_outs = cached.eval()(feats)
            eval_expected = plain.eval()(feats)
            train_outs = cached.train()(feats)
            train_expected = plain.train()(feats)

        self.assertFalse(torch.allclose(eval_outs[-1], eval_expected[-1]))
        for out, reference in zip(train_outs, train_expected):
            torch.testing.assert_close(out, reference, rtol=0, atol=0)

    def test_eval_at_another_size_than_the_cached_one(self) -> None:
        """
        Input: an encoder cached for 64x64, run in eval on a 96x160 input.
        Expected: the embedding is built for the actual size, and eval gives the same output
            as an encoder without the cache.
        Check: the output shapes and equality with the plain encoder.
        """
        cached = HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32, eval_spatial_size=(64, 64))
        plain = HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32)
        plain.load_state_dict(cached.state_dict())
        feats = _features(96, 160)

        with torch.no_grad():
            outs = cached.eval()(feats)
            expected = plain.eval()(feats)

        self.assertEqual([tuple(out.shape[2:]) for out in outs], [(12, 20), (6, 10), (3, 5)])
        for out, reference in zip(outs, expected):
            torch.testing.assert_close(out, reference, rtol=0, atol=0)

    def test_state_dict_follows_rt_detr_v2_checkpoints(self) -> None:
        """
        Input: the encoders of RT-DETRv2-S and -M, on the R18 and R50 stage channels, with
            the hidden dimension, expansion and feed-forward width of RT-DETRv2.
        Expected: the layout of the encoder weights of RT-DETRv2 checkpoints trained with the
            official code, so they load unchanged: the entry counts under each submodule, and
            the shapes of the input projections, the AIFI layer and the CCFF blocks. The
            cached position embedding is not part of them.
        Check: the state dict entry counts and the shapes of the listed entries.
        """
        counts = {
            "input_proj": 18,
            "encoder": 12,
            "lateral_convs": 12,
            "fpn_blocks": 108,
            "downsample_convs": 12,
            "pan_blocks": 108,
        }
        shared_shapes = {
            "encoder.0.layers.0.self_attn.in_proj_weight": (768, 256),
            "encoder.0.layers.0.linear1.weight": (1024, 256),
            "encoder.0.layers.0.norm2.bias": (256,),
            "lateral_convs.0.conv.weight": (256, 256, 1, 1),
            "fpn_blocks.0.conv1.conv.weight": (128, 512, 1, 1),
            "fpn_blocks.0.bottlenecks.2.conv1.conv.weight": (128, 128, 3, 3),
            "fpn_blocks.0.bottlenecks.2.conv2.conv.weight": (128, 128, 1, 1),
            "fpn_blocks.0.conv3.conv.weight": (256, 128, 1, 1),
            "downsample_convs.1.conv.weight": (256, 256, 3, 3),
            "pan_blocks.1.conv2.norm.running_var": (128,),
        }

        for name, in_channels in (("r18vd", (128, 256, 512)), ("r50vd_m", (512, 1024, 2048))):
            with self.subTest(name=name):
                encoder = HybridEncoder(
                    in_channels=in_channels,
                    hidden_dim=256,
                    expansion=0.5,
                    dim_feedforward=1024,
                    eval_spatial_size=(640, 640),
                )
                shapes = {
                    **shared_shapes,
                    "input_proj.0.conv.weight": (256, in_channels[0], 1, 1),
                    "input_proj.2.conv.weight": (256, in_channels[2], 1, 1),
                }

                state = encoder.state_dict()

                self.assertEqual(dict(Counter(key.split(".")[0] for key in state)), counts)
                self.assertEqual({key: tuple(state[key].shape) for key in shapes}, shapes)

    def test_aifi_runs_on_the_selected_levels_only(self) -> None:
        """
        Input: an encoder with AIFI on levels 1 and 2.
        Expected: one transformer per selected level, each with num_encoder_layers layers.
        Check: the encoder module lists.
        """
        encoder = HybridEncoder(
            in_channels=(8, 8, 8), hidden_dim=32, use_encoder_idx=(1, 2), num_encoder_layers=2
        )

        self.assertEqual(len(encoder.encoder), 2)
        self.assertEqual([len(stack.layers) for stack in encoder.encoder], [2, 2])

    def test_gradients_reach_every_parameter_and_level(self) -> None:
        """
        Input: an encoder in train mode and three levels that require gradients.
        Expected: a loss on the outputs reaches every parameter, the AIFI transformer
            included, and every input level, so no branch of the fusion is cut off.
        Check: every gradient is present and non-zero.
        """
        torch.manual_seed(0)
        encoder = HybridEncoder(in_channels=(8, 16, 32), hidden_dim=32).train()
        feats = [
            torch.randn(2, channels, 64 // s, 64 // s, requires_grad=True)
            for channels, s in ((8, 8), (16, 16), (32, 32))
        ]

        outs = encoder(feats)
        sum((out * torch.randn_like(out)).sum() for out in outs).backward()

        for name, param in encoder.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertGreater(param.grad.abs().sum().item(), 0.0, name)
        for level, feat in enumerate(feats):
            self.assertGreater(feat.grad.abs().sum().item(), 0.0, f"level {level}")

    def test_fusing_every_rep_vgg_block_preserves_the_encoder_output(self) -> None:
        """
        Input: an encoder in eval with random batch norm statistics, whose RepVgg blocks are
            all fused for deployment by walking its modules.
        Expected: the fused encoder gives the output of the unfused one.
        Check: the outputs before and after fusing, and the number of fused blocks.
        """
        torch.manual_seed(0)
        encoder = HybridEncoder(in_channels=(8, 8, 8), hidden_dim=32)
        with torch.no_grad():
            for module in encoder.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.running_mean.uniform_(-1, 1)
                    module.running_var.uniform_(0.5, 2)
                    module.weight.uniform_(0.5, 2)
                    module.bias.uniform_(-1, 1)
        encoder.eval()
        feats = _features(64, 64)

        with torch.no_grad():
            expected = encoder(feats)
            blocks = [m for m in encoder.modules() if isinstance(m, RepVggBlock)]
            for block in blocks:
                block.convert_to_deploy()
            fused = encoder(feats)

        self.assertEqual(len(blocks), 12)
        for out, reference in zip(fused, expected):
            torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-4)


def _random_rep_vgg_block() -> RepVggBlock:
    """A RepVgg block in eval with random batch norm statistics and affine parameters."""
    torch.manual_seed(0)
    block = RepVggBlock(8, 8)
    with torch.no_grad():
        for norm in (block.conv1.norm, block.conv2.norm):
            norm.running_mean.uniform_(-1, 1)
            norm.running_var.uniform_(0.5, 2)
            norm.weight.uniform_(0.5, 2)
            norm.bias.uniform_(-1, 1)
    return block.eval()


class TestRepVggBlock(unittest.TestCase):
    """The RepVgg block of the cross-scale fusion."""

    def test_fusion_preserves_the_output(self) -> None:
        """
        Input: a block with random batch norm statistics, in eval, fused for deployment.
        Expected: the single 3x3 convolution gives the output of the two branches.
        Check: the outputs before and after fusing.
        """
        block = _random_rep_vgg_block()
        x = torch.randn(2, 8, 6, 6)

        with torch.no_grad():
            expected = block(x)
            block.convert_to_deploy()
            fused = block(x)

        torch.testing.assert_close(fused, expected, rtol=1e-4, atol=1e-5)

    def test_fused_state_dict_follows_rt_detr(self) -> None:
        """
        Input: a block fused for deployment.
        Expected: the branches are kept and the fused convolution is named conv, as in
            RT-DETRv2, so converted checkpoints load in both directions.
        Check: the state dict keys.
        """
        block = RepVggBlock(8, 8).eval()

        block.convert_to_deploy()

        keys = set(block.state_dict())
        self.assertIn("conv.weight", keys)
        self.assertIn("conv.bias", keys)
        self.assertIn("conv1.conv.weight", keys)
        self.assertIn("conv2.norm.running_var", keys)

    def test_fusion_keeps_the_dtype_of_the_block(self) -> None:
        """
        Input: a block cast to float64 (and to float16 on a GPU), then fused.
        Expected: the fused convolution takes the dtype of the block, so the next forward
            runs instead of failing on mixed dtypes.
        Check: the fused weight dtype and the output against the unfused block.
        """
        cases = [(torch.float64, "cpu", 1e-7, 1e-10)]
        if torch.cuda.is_available():
            cases.append((torch.float16, "cuda", 1e-2, 1e-2))
        for dtype, device, rtol, atol in cases:
            with self.subTest(dtype=dtype):
                block = _random_rep_vgg_block().to(device=device, dtype=dtype)
                x = torch.randn(2, 8, 6, 6, device=device, dtype=dtype)

                with torch.no_grad():
                    expected = block(x)
                    block.convert_to_deploy()
                    fused = block(x)

                self.assertEqual(block.conv.weight.dtype, dtype)
                torch.testing.assert_close(fused, expected, rtol=rtol, atol=atol)

    def test_converting_again_refolds_the_current_branches(self) -> None:
        """
        Input: a fused block whose branch batch norm is changed, then converted again.
        Expected: the same fused convolution is updated with the changed branches.
        Check: the fused module identity and the output against the unfused branches.
        """
        block = _random_rep_vgg_block()
        block.convert_to_deploy()
        fused = block.conv
        x = torch.randn(2, 8, 6, 6)

        with torch.no_grad():
            block.conv1.norm.running_mean.add_(0.5)
            expected = block.act(block.conv1(x) + block.conv2(x))
            block.convert_to_deploy()
            out = block(x)

        self.assertIs(block.conv, fused)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)
