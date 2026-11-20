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

"""Unit tests for the ResNet-vd backbone of RT-DETR."""

from __future__ import annotations

from collections import Counter
import unittest

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

from autoware_ml.models.common.backbones.presnet import PResNet


class TestPResNet(unittest.TestCase):
    """The ResNet-vd backbone."""

    def test_returns_the_selected_stages_at_strides_8_16_32(self) -> None:
        """
        Input: a 64x64 image through the depth 18 and depth 50 backbones.
        Expected: the default stages (1, 2, 3) at strides 8, 16 and 32, with the channels of
            basic blocks and of bottlenecks.
        Check: out_channels, out_strides and the output shapes.
        """
        for depth, channels in ((18, [128, 256, 512]), (50, [512, 1024, 2048])):
            with self.subTest(depth=depth):
                backbone = PResNet(depth=depth).eval()

                with torch.no_grad():
                    outs = backbone(torch.randn(1, 3, 64, 64))

                self.assertEqual(backbone.out_channels, channels)
                self.assertEqual(backbone.out_strides, [8, 16, 32])
                self.assertEqual(
                    [tuple(out.shape) for out in outs],
                    [
                        (1, channels[0], 8, 8),
                        (1, channels[1], 4, 4),
                        (1, channels[2], 2, 2),
                    ],
                )

    def test_out_indices_select_the_stride_4_stage(self) -> None:
        """
        Input: a depth 18 backbone with out_indices (0, 3).
        Expected: the first stage at stride 4 and the last at stride 32.
        Check: the output shapes.
        """
        backbone = PResNet(depth=18, out_indices=(0, 3)).eval()

        with torch.no_grad():
            outs = backbone(torch.randn(1, 3, 64, 64))

        self.assertEqual([tuple(out.shape) for out in outs], [(1, 64, 16, 16), (1, 512, 2, 2)])

    def test_rejects_unsupported_depth_and_stage_indices(self) -> None:
        """
        Input: depth 26, empty, out-of-range and decreasing out_indices.
        Expected: each is refused at construction.
        Check: the ValueError.
        """
        for kwargs in (
            {"depth": 26},
            {"depth": 18, "out_indices": ()},
            {"depth": 18, "out_indices": (1, 4)},
            {"depth": 18, "out_indices": (2, 1)},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                PResNet(**kwargs)

    def test_state_dict_follows_rt_detr_v2_checkpoints(self) -> None:
        """
        Input: the depth 18 backbone and the depth 50 one with frozen batch norms, as
            RT-DETRv2-S and -M train them.
        Expected: the layout of the backbone weights of RT-DETRv2 checkpoints trained with the
            official code, so they load unchanged: the entry counts under the stem and the
            stages, and the shapes of the stem, the shortcuts and the last blocks. Frozen
            batch norms carry no num_batches_tracked, as in those checkpoints.
        Check: the state dict entry counts and the shapes of the listed entries.
        """
        cases = (
            (
                PResNet(depth=18),
                {"conv1": 18, "res_layers": 120},
                {
                    "conv1.conv1_1.conv.weight": (32, 3, 3, 3),
                    "conv1.conv1_3.conv.weight": (64, 32, 3, 3),
                    "res_layers.0.blocks.0.branch2a.conv.weight": (64, 64, 3, 3),
                    "res_layers.0.blocks.0.short.conv.weight": (64, 64, 1, 1),
                    "res_layers.1.blocks.0.short.conv.conv.weight": (128, 64, 1, 1),
                    "res_layers.3.blocks.1.branch2b.norm.running_var": (512,),
                },
            ),
            (
                PResNet(depth=50, freeze_at=0, freeze_norm=True),
                {"conv1": 15, "res_layers": 260},
                {
                    "conv1.conv1_1.conv.weight": (32, 3, 3, 3),
                    "res_layers.0.blocks.0.branch2a.conv.weight": (64, 64, 1, 1),
                    "res_layers.0.blocks.0.short.conv.weight": (256, 64, 1, 1),
                    "res_layers.1.blocks.0.short.conv.conv.weight": (512, 256, 1, 1),
                    "res_layers.2.blocks.5.branch2c.conv.weight": (1024, 256, 1, 1),
                    "res_layers.3.blocks.2.branch2c.norm.running_var": (2048,),
                },
            ),
        )

        for name, (backbone, counts, shapes) in zip(("r18vd", "r50vd_m"), cases):
            with self.subTest(name=name):
                state = backbone.state_dict()

                self.assertEqual(dict(Counter(key.split(".")[0] for key in state)), counts)
                self.assertEqual({key: tuple(state[key].shape) for key in shapes}, shapes)

    def test_downsampling_shortcut_average_pools_before_the_projection(self) -> None:
        """
        Input: a ramp through the shortcut of the first block of stage 1, with the 1x1
            convolution copying the 64 input channels and the batch norm in its initial eval
            state.
        Expected: each output position is the mean of a 2x2 input window, not one of its four
            positions as with a strided 1x1 convolution.
        Check: the copied channels equal the 2x2 average pool, up to the batch norm epsilon.
        """
        shortcut = PResNet(depth=18).res_layers[1].blocks[0].short.eval()
        with torch.no_grad():
            shortcut.conv.conv.weight.copy_(torch.eye(128, 64)[:, :, None, None])
            x = torch.arange(64 * 8 * 8, dtype=torch.float32).reshape(1, 64, 8, 8)

            out = shortcut(x)

        expected = nn.functional.avg_pool2d(x, 2, 2)[:, :64] / (1 + 1e-5) ** 0.5
        self.assertEqual(tuple(out.shape), (1, 128, 4, 4))
        torch.testing.assert_close(out[:, :64], expected)

    def test_freezing_matches_the_r50_m_settings(self) -> None:
        """
        Input: depth 50 with freeze_at=0 and freeze_norm=True, as RT-DETRv2-M trains it.
        Expected: the stem is frozen, every stage trains, and every batch norm is frozen.
        Check: requires_grad of the parameters and the batch norm types.
        """
        backbone = PResNet(depth=50, freeze_at=0, freeze_norm=True)

        self.assertFalse(any(p.requires_grad for p in backbone.conv1.parameters()))
        self.assertTrue(all(p.requires_grad for p in backbone.res_layers.parameters()))
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in backbone.modules()))
        self.assertTrue(any(isinstance(m, FrozenBatchNorm2d) for m in backbone.modules()))

    def test_freeze_at_n_freezes_the_first_n_stages(self) -> None:
        """
        Input: depth 18 with freeze_at=2.
        Expected: the stem and stages 0 and 1 are frozen, stages 2 and 3 train.
        Check: requires_grad of the parameters per stage.
        """
        backbone = PResNet(depth=18, freeze_at=2)

        trainable = [
            all(p.requires_grad for p in stage.parameters()) for stage in backbone.res_layers
        ]
        self.assertFalse(any(p.requires_grad for p in backbone.conv1.parameters()))
        self.assertEqual(trainable, [False, False, True, True])

    def test_gradients_reach_every_parameter_and_the_image(self) -> None:
        """
        Input: a trainable depth 18 and depth 50 backbone in train mode, and an image that
            requires gradients.
        Expected: a loss on the selected stages reaches every parameter and the image, so no
            block is skipped and the in-place activations keep the graph intact.
        Check: every gradient is present and non-zero.
        """
        for depth in (18, 50):
            with self.subTest(depth=depth):
                torch.manual_seed(0)
                backbone = PResNet(depth=depth, out_indices=(0, 1, 2, 3)).train()
                image = torch.randn(2, 3, 64, 64, requires_grad=True)

                outs = backbone(image)
                sum((out * torch.randn_like(out)).sum() for out in outs).backward()

                for name, param in backbone.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertGreater(param.grad.abs().sum().item(), 0.0, name)
                self.assertGreater(image.grad.abs().sum().item(), 0.0)
