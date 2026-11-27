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

"""Unit tests for the RT-DETRv2 transformer decoder."""

from __future__ import annotations

from collections import Counter
import unittest

from jaxtyping import Float, Int32
import torch
from torch import Tensor

from autoware_ml.models.detection2d.heads.rtdetrv2_decoder import RTDETRv2Decoder

NUM_CLASSES = 4
HIDDEN_DIM = 32
NUM_QUERIES = 12
STRIDES = (8, 16, 32)
# A non-square input: 8 x 12, 4 x 6 and 2 x 3 tokens.
HEIGHT, WIDTH = 64, 96


def _decoder(**kwargs: object) -> RTDETRv2Decoder:
    """Build a small seeded decoder of 3 layers over three 32-channel levels."""
    torch.manual_seed(0)
    config = {
        "num_classes": NUM_CLASSES,
        "hidden_dim": HIDDEN_DIM,
        "num_queries": NUM_QUERIES,
        "feat_channels": (HIDDEN_DIM,) * 3,
        "feat_strides": STRIDES,
        "num_heads": 4,
        "num_layers": 3,
        "dim_feedforward": 64,
        "num_denoising": 12,
        "eval_spatial_size": (HEIGHT, WIDTH),
    }
    config.update(kwargs)
    return RTDETRv2Decoder(**config)


def _feats(
    height: int = HEIGHT, width: int = WIDTH, batch_size: int = 2
) -> list[Float[Tensor, "batch_size channels height width"]]:
    """Random encoder outputs of an input of the given size."""
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randn(batch_size, HIDDEN_DIM, height // s, width // s, generator=generator)
        for s in STRIDES
    ]


def _ground_truth(
    box_counts: tuple[int, ...] = (3, 1),
) -> tuple[
    Int32[Tensor, "batch_size max_num_boxes"],
    Float[Tensor, "batch_size max_num_boxes 4"],
    Int32[Tensor, " batch_size"],
]:
    """Labels and normalized (cx, cy, w, h) boxes of images holding the given box counts.

    Padded as the ground truth is collated: two rows past the largest count, labels -1 and
    zero boxes past the valid count of each image.
    """
    generator = torch.Generator().manual_seed(2)
    width = max(box_counts) + 2
    labels = torch.full((len(box_counts), width), -1, dtype=torch.int32)
    boxes = torch.zeros(len(box_counts), width, 4)
    for image, count in enumerate(box_counts):
        labels[image, :count] = torch.randint(0, NUM_CLASSES, (count,), generator=generator)
        boxes[image, :count] = torch.rand(count, 4, generator=generator) * torch.tensor(
            [0.5, 0.5, 0.2, 0.2]
        ) + torch.tensor([0.25, 0.25, 0.05, 0.05])
    return labels, boxes, torch.tensor(box_counts, dtype=torch.int32)


class TestRTDETRv2Decoder(unittest.TestCase):
    """The query selection, decoder layers and outputs of the RT-DETRv2 decoder."""

    def test_training_returns_every_layer_the_query_selection_and_the_denoising(self) -> None:
        """
        Input: a 3-layer decoder in training, ``num_denoising=12``, two images with 3 and 1 boxes.
        Expected: the logits and boxes of all 3 layers for the 12 matching queries, the query
        selection outputs for the same queries, and 12 // 3 = 4 groups of 6 denoising
        queries; every box is within [0, 1].
        Check: the output shapes, the group count and the box range.
        """
        decoder = _decoder().train()

        out = decoder(_feats(), *_ground_truth())

        self.assertEqual(tuple(out.all_cls_scores.shape), (3, 2, NUM_QUERIES, NUM_CLASSES))
        self.assertEqual(tuple(out.all_bbox_preds.shape), (3, 2, NUM_QUERIES, 4))
        self.assertEqual(tuple(out.encoder_cls_scores.shape), (2, NUM_QUERIES, NUM_CLASSES))
        self.assertEqual(tuple(out.encoder_bbox_preds.shape), (2, NUM_QUERIES, 4))
        self.assertEqual(out.denoising.num_groups, 4)
        self.assertEqual(tuple(out.denoising.cls_scores.shape), (3, 2, 24, NUM_CLASSES))
        self.assertTrue(((out.all_bbox_preds >= 0) & (out.all_bbox_preds <= 1)).all())

    def test_inference_returns_the_eval_idx_layer_of_training(self) -> None:
        """
        Input: a 3-layer decoder with ``eval_idx=1``, run in training with denoising and at
        inference. Its input projection batch norms stay in eval mode in both.
        Expected: inference returns only the second layer, equal to the second layer of
        training. This holds because the denoising queries do not reach the matching
        queries, and the cached anchors equal the generated ones.
        Check: the inference shapes, the absent training outputs, and the predictions
        against the training ones.
        """
        decoder = _decoder(eval_idx=1)
        feats = _feats()
        decoder.train()
        decoder.input_proj.eval()
        with torch.no_grad():
            trained = decoder(feats, *_ground_truth())

            inferred = decoder.eval()(feats)

        self.assertEqual(tuple(inferred.all_cls_scores.shape), (1, 2, NUM_QUERIES, NUM_CLASSES))
        self.assertIsNone(inferred.encoder_cls_scores)
        self.assertIsNone(inferred.denoising)
        torch.testing.assert_close(inferred.all_cls_scores[0], trained.all_cls_scores[1])
        torch.testing.assert_close(inferred.all_bbox_preds[0], trained.all_bbox_preds[1])

    def test_cached_anchors_are_used_at_their_size_only(self) -> None:
        """
        Input: a decoder cached for 64 x 96 whose cached anchors are replaced by a marker,
        next to a decoder without cache, both run at 64 x 96 and at 96 x 64.
        Expected: at 64 x 96 the marker changes the prediction; at 96 x 64 the anchors are
        generated, so both decoders agree.
        Check: the predictions of both decoders at both sizes.
        """
        cached = _decoder().eval()
        plain = _decoder(eval_spatial_size=None).eval()
        plain.load_state_dict(cached.state_dict())
        with torch.no_grad():
            cached.anchors.fill_(0.0)
            for height, width, same in ((HEIGHT, WIDTH, False), (WIDTH, HEIGHT, True)):
                with self.subTest(size=(height, width)):
                    feats = _feats(height, width)

                    a = cached(feats).all_bbox_preds
                    b = plain(feats).all_bbox_preds

                    self.assertEqual(torch.allclose(a, b), same)

    def test_anchors_sit_on_the_token_centers(self) -> None:
        """
        Input: the anchors of a single 2 x 3 level.
        Expected: token k, at row k // 3 and column k % 3, is centered on
        ((column + 0.5) / 3, (row + 0.5) / 2) with size 0.05. None is within 0.01 of the
        border, so all are valid and finite.
        Check: the sigmoid of the anchors and the mask.
        """
        decoder = _decoder(eval_spatial_size=None)

        anchors, valid_mask = decoder._generate_anchors([[2, 3]])

        expected = torch.tensor(
            [[(col + 0.5) / 3, (row + 0.5) / 2, 0.05, 0.05] for row in range(2) for col in range(3)]
        )
        torch.testing.assert_close(torch.sigmoid(anchors[0]), expected)
        self.assertTrue(valid_mask.all())

    def test_anchors_near_the_border_are_masked_out(self) -> None:
        """
        Input: the anchors of a 1 x 60 level, whose first token is centered at x = 0.5 / 60.
        Expected: that token is within 0.01 of the border, so it is masked and its anchor is
        +inf; its neighbor is valid.
        Check: the mask and the anchors of the first two tokens.
        """
        decoder = _decoder(eval_spatial_size=None)

        anchors, valid_mask = decoder._generate_anchors([[1, 60]])

        self.assertEqual(valid_mask[0, :2, 0].tolist(), [False, True])
        self.assertTrue(torch.isinf(anchors[0, 0]).all())
        self.assertTrue(torch.isfinite(anchors[0, 1]).all())

    def test_box_loss_of_a_layer_also_trains_the_previous_box_head(self) -> None:
        """
        Input: a decoder in training without denoising, and a loss on the boxes of its
        second layer only.
        Expected: the loss reaches the box heads of the first and second layers (look
        forward twice) but not of the third.
        Check: the gradients of the three box heads.
        """
        decoder = _decoder(num_denoising=0).train()

        out = decoder(_feats())
        out.all_bbox_preds[1].sum().backward()

        reached = [
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
            for head in decoder.dec_bbox_head
        ]
        self.assertEqual(reached, [True, True, False])

    def test_gradients_reach_every_parameter_even_without_boxes(self) -> None:
        """
        Input: a decoder in training with denoising, once for images with boxes and once for
        a batch without any box.
        Expected: a loss on every output reaches every parameter in both cases, so
        distributed ranks with and without boxes update the same parameters.
        Check: every gradient is present.
        """
        for box_counts in ((3, 1), (0, 0)):
            with self.subTest(box_counts=box_counts):
                decoder = _decoder().train()

                out = decoder(_feats(), *_ground_truth(box_counts))
                tensors = [
                    out.all_cls_scores,
                    out.all_bbox_preds,
                    out.encoder_cls_scores,
                    out.encoder_bbox_preds,
                    out.denoising.cls_scores,
                    out.denoising.bbox_preds,
                ]
                sum((t * torch.randn_like(t)).sum() for t in tensors).backward()

                for name, param in decoder.named_parameters():
                    self.assertIsNotNone(param.grad, name)

    def test_ground_truth_is_required_only_with_denoising(self) -> None:
        """
        Input: a decoder with denoising and one without, both trained without ground truth.
        Expected: the first raises a ValueError; the second runs and returns no denoising
        outputs.
        Check: the ValueError and the absent denoising outputs.
        """
        with self.assertRaises(ValueError):
            _decoder().train()(_feats())

        out = _decoder(num_denoising=0).train()(_feats())

        self.assertIsNone(out.denoising)
        self.assertIsNotNone(out.encoder_cls_scores)

    def test_rejects_mismatched_levels_and_an_eval_idx_beyond_the_layers(self) -> None:
        """
        Input: three channel counts with two strides; and ``eval_idx=3`` and ``eval_idx=-4`` with 3
        layers.
        Expected: each is refused at construction.
        Check: a ValueError is raised for each.
        """
        for kwargs in (
            {"feat_strides": (8, 16)},
            {"eval_idx": 3},
            {"eval_idx": -4},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                _decoder(**kwargs)

    def test_initial_scores_are_1_percent_and_boxes_start_at_their_reference(self) -> None:
        """
        Input: a fresh decoder.
        Expected: every score head starts at a 1% probability and every box head at a zero
        offset, so the initial boxes are the anchors of the selected tokens.
        Check: the score biases and the last box layer of every head.
        """
        decoder = _decoder()

        for head in [decoder.enc_score_head, *decoder.dec_score_head]:
            torch.testing.assert_close(torch.sigmoid(head.bias), torch.full((NUM_CLASSES,), 0.01))
        for head in [decoder.enc_bbox_head, *decoder.dec_bbox_head]:
            self.assertEqual(head.layers[-1].weight.abs().sum().item(), 0.0)
            self.assertEqual(head.layers[-1].bias.abs().sum().item(), 0.0)

    def test_state_dict_follows_rt_detr_v2_checkpoints(self) -> None:
        """
        Input: the decoders of RT-DETRv2-S (3 layers) and -M (6 layers) as trained on T4
        data: 8 classes, 256 channels, 300 queries, 3 levels of 4 points, cached anchors.
        Expected: the same entry count per submodule and the same shapes of sampled entries
        as the decoder in RT-DETRv2 checkpoints of the official code, except the anchors,
        which are generated rather than saved.
        Check: the state dict entry counts and the shapes of the listed entries.
        """
        shapes = {
            "input_proj.2.conv.weight": (256, 256, 1, 1),
            "decoder.layers.0.self_attn.in_proj_weight": (768, 256),
            "decoder.layers.0.cross_attn.sampling_offsets.weight": (192, 256),
            "decoder.layers.0.cross_attn.attention_weights.weight": (96, 256),
            "decoder.layers.0.cross_attn.num_points_scale": (12,),
            "decoder.layers.0.linear1.weight": (1024, 256),
            "denoising_class_embed.weight": (9, 256),
            "query_pos_head.layers.0.weight": (512, 4),
            "enc_bbox_head.layers.2.weight": (4, 256),
            "dec_score_head.0.weight": (8, 256),
        }
        for num_layers in (3, 6):
            with self.subTest(num_layers=num_layers):
                decoder = RTDETRv2Decoder(
                    num_classes=8,
                    feat_channels=(256, 256, 256),
                    num_layers=num_layers,
                    eval_spatial_size=(960, 960),
                )
                state = decoder.state_dict()

                self.assertEqual(
                    dict(Counter(key.split(".")[0] for key in state)),
                    {
                        "input_proj": 18,
                        "decoder": 23 * num_layers,
                        "denoising_class_embed": 1,
                        "query_pos_head": 4,
                        "enc_output": 4,
                        "enc_score_head": 2,
                        "enc_bbox_head": 6,
                        "dec_score_head": 2 * num_layers,
                        "dec_bbox_head": 6 * num_layers,
                    },
                )
                self.assertEqual({key: tuple(state[key].shape) for key in shapes}, shapes)
                self.assertEqual(
                    decoder.anchors.shape, (1, sum((960 // s) ** 2 for s in STRIDES), 4)
                )
