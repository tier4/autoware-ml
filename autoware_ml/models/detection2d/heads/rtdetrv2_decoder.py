# Copyright (c) 2023 lyuwenyu. All Rights Reserved.
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

"""RT-DETRv2 transformer decoder, ported from RT-DETRv2's src/zoo/rtdetr/rtdetrv2_decoder.py.

The decoder scores every encoder token, takes the top ones as the initial queries and their
boxes, and refines the boxes through deformable-attention decoder layers. In training,
contrastive denoising queries run next to them. Only the configuration every released
RT-DETRv2 model uses is kept: class-aware query selection, no learned query content, one
decoder level per encoder output, and grid-sample attention. Module names match the source,
so its checkpoints load unchanged.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import copy
import math

from jaxtyping import Bool, Float, Int64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from autoware_ml.dataclasses.models.detection2d.head_outputs import (
    RTDETRv2DenoisingOutputs,
    RTDETRv2HeadOutputs,
)
from autoware_ml.models.common.layers.ms_deformable_attention import MSDeformableAttention
from autoware_ml.models.detection2d.task_modules.denoising import (
    INVERSE_SIGMOID_EPS,
    DenoisingQueries,
    build_contrastive_denoising_queries,
)
from autoware_ml.models.detection3d.task_modules.streaming import inverse_sigmoid


class MLP(nn.Module):
    """Linear layers with a ReLU between each two."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Input dimension.
            hidden_dim: Output dimension of every layer but the last.
            output_dim: Output dimension.
            num_layers: Number of linear layers.
        """
        super().__init__()
        dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim, *dims], [*dims, output_dim])
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Float[Tensor, "*batch input_dim"]) -> Float[Tensor, "*batch output_dim"]:
        """Apply the layers.

        Args:
            x: Input features.

        Returns:
            Output features.
        """
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


class TransformerDecoderLayer(nn.Module):
    """Post-norm decoder layer: self-attention, deformable cross-attention, ReLU FFN."""

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_levels: int = 4,
        num_points: int | Sequence[int] = 4,
    ) -> None:
        """Initialize the layer.

        Args:
            d_model: Query dimension.
            num_heads: Number of attention heads.
            dim_feedforward: Hidden dimension of the feed-forward network.
            dropout: Dropout probability.
            num_levels: Number of memory levels.
            num_points: Sampling points per head and level, or one count per level.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformableAttention(d_model, num_heads, num_levels, num_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(
        self,
        target: Float[Tensor, "batch_size num_queries d_model"],
        reference_points: Float[Tensor, "batch_size num_queries 1 4"],
        memory: Float[Tensor, "batch_size num_tokens d_model"],
        memory_spatial_shapes: Sequence[Sequence[int]],
        attn_mask: Bool[Tensor, "num_queries num_queries"] | None = None,
        query_pos_embed: Float[Tensor, "batch_size num_queries d_model"] | None = None,
    ) -> Float[Tensor, "batch_size num_queries d_model"]:
        """Update the queries.

        Args:
            target: Query content.
            reference_points: ``(cx, cy, w, h)`` reference box of every query, normalized by
                the image size.
            memory: Encoder tokens of every level, flattened row by row and concatenated
                level by level.
            memory_spatial_shapes: ``(height, width)`` of each memory level.
            attn_mask: Self-attention mask, True where attention is blocked.
            query_pos_embed: Query position embedding, added to the self-attention queries and
                keys and to the cross-attention queries.

        Returns:
            Updated query content.
        """
        q = k = target if query_pos_embed is None else target + query_pos_embed
        attended, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask, need_weights=False)
        target = self.norm1(target + self.dropout1(attended))

        query = target if query_pos_embed is None else target + query_pos_embed
        attended = self.cross_attn(query, reference_points, memory, memory_spatial_shapes)
        target = self.norm2(target + self.dropout2(attended))

        ffn = self.linear2(self.dropout3(self.activation(self.linear1(target))))
        return self.norm3(target + self.dropout4(ffn))


class TransformerDecoder(nn.Module):
    """Decoder layers with iterative box refinement.

    Every layer predicts a box offset relative to the box of the previous layer. Training
    runs every layer and returns all of their predictions for the auxiliary losses.
    Inference stops after ``eval_idx``, so a model trained with N layers can be deployed
    with fewer.
    """

    def __init__(
        self, decoder_layer: TransformerDecoderLayer, num_layers: int, eval_idx: int = -1
    ) -> None:
        """Initialize the decoder.

        Args:
            decoder_layer: Layer to replicate.
            num_layers: Number of layers.
            eval_idx: Index of the layer whose prediction inference returns; negative
                values count from the end.

        Raises:
            ValueError: If ``eval_idx`` selects no layer.
        """
        super().__init__()
        if not -num_layers <= eval_idx < num_layers:
            raise ValueError(
                f"eval_idx must select one of the {num_layers} layers, got {eval_idx}."
            )
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        target: Float[Tensor, "batch_size num_queries d_model"],
        ref_points_unact: Float[Tensor, "batch_size num_queries 4"],
        memory: Float[Tensor, "batch_size num_tokens d_model"],
        memory_spatial_shapes: Sequence[Sequence[int]],
        bbox_head: nn.ModuleList,
        score_head: nn.ModuleList,
        query_pos_head: nn.Module,
        attn_mask: Bool[Tensor, "num_queries num_queries"] | None = None,
    ) -> tuple[
        Float[Tensor, "num_outputs batch_size num_queries 4"],
        Float[Tensor, "num_outputs batch_size num_queries num_classes"],
    ]:
        """Run the layers.

        Args:
            target: Initial query content.
            ref_points_unact: Inverse sigmoid of the initial ``(cx, cy, w, h)`` boxes.
            memory: Encoder tokens of every level, flattened and concatenated.
            memory_spatial_shapes: ``(height, width)`` of each memory level.
            bbox_head: Box refinement head of every layer.
            score_head: Classification head of every layer.
            query_pos_head: Maps a normalized box to a query position embedding.
            attn_mask: Self-attention mask, True where attention is blocked.

        Returns:
            Boxes and logits of every layer in training, of the ``eval_idx`` layer only at
            inference.
        """
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        # The boxes of the previous layer with their gradient; read from the second layer on.
        ref_points = ref_points_detach

        output = target
        for i, layer in enumerate(self.layers):
            query_pos_embed = query_pos_head(ref_points_detach)
            output = layer(
                output,
                ref_points_detach.unsqueeze(2),
                memory,
                memory_spatial_shapes,
                attn_mask,
                query_pos_embed,
            )
            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach, eps=INVERSE_SIGMOID_EPS)
            )

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    # Refine the previous boxes before their detach, so the box loss of a
                    # layer also trains the layer before it (look forward twice).
                    dec_out_bboxes.append(
                        F.sigmoid(
                            bbox_head[i](output)
                            + inverse_sigmoid(ref_points, eps=INVERSE_SIGMOID_EPS)
                        )
                    )
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)


class RTDETRv2Decoder(nn.Module):
    """RT-DETRv2 query selection, transformer decoder and prediction heads."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 300,
        feat_channels: Sequence[int] = (256, 256, 256),
        feat_strides: Sequence[int] = (8, 16, 32),
        num_points: int | Sequence[int] = 4,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        eval_spatial_size: Sequence[int] | None = None,
        eval_idx: int = -1,
        eps: float = 1e-2,
    ) -> None:
        """Initialize the decoder.

        Args:
            num_classes: Number of object classes.
            hidden_dim: Query and memory dimension.
            num_queries: Number of queries selected from the encoder tokens.
            feat_channels: Channel count of each encoder output level.
            feat_strides: Stride of each encoder output level.
            num_points: Deformable-attention sampling points per head and level, or one
                count per level.
            num_heads: Number of attention heads.
            num_layers: Number of decoder layers run in training.
            dim_feedforward: Feed-forward dimension of the decoder layers.
            dropout: Dropout of the decoder layers.
            num_denoising: Target number of denoising queries per image in training; 0
                disables them.
            label_noise_ratio: Twice the probability that the label of a denoising query is
                replaced by a random class.
            box_noise_scale: Box noise of the denoising queries, relative to half the box
                size.
            eval_spatial_size: Input ``(height, width)`` whose anchors are cached for
                inference. Other sizes, and training, generate them on the fly.
            eval_idx: Decoder layer whose prediction inference returns; negative values count
                from the end.
            eps: Anchors closer than this to the image border, in normalized coordinates, are
                masked out.

        Raises:
            ValueError: If ``feat_channels`` and ``feat_strides`` differ in length, or
                ``eval_idx`` selects no layer.
        """
        super().__init__()
        if len(feat_channels) != len(feat_strides):
            raise ValueError(
                "feat_channels and feat_strides must describe the same levels, got "
                f"{list(feat_channels)} and {list(feat_strides)}."
            )

        self.hidden_dim = hidden_dim
        self.feat_strides = [int(s) for s in feat_strides]
        self.num_levels = len(feat_channels)
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers

        self.input_proj = nn.ModuleList(
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(channels, hidden_dim, 1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )
            for channels in feat_channels
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, num_heads, dim_feedforward, dropout, self.num_levels, num_points
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            nn.init.normal_(self.denoising_class_embed.weight[:-1])

        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)
        self.enc_output = nn.Sequential(
            OrderedDict(
                [("proj", nn.Linear(hidden_dim, hidden_dim)), ("norm", nn.LayerNorm(hidden_dim))]
            )
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)
        self.dec_score_head = nn.ModuleList(
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        )
        self.dec_bbox_head = nn.ModuleList(
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        )

        # Non-persistent, unlike the source: the cache follows eval_spatial_size instead of
        # the size a checkpoint was trained at.
        self._anchor_spatial_shapes: list[list[int]] | None = None
        if eval_spatial_size is not None:
            eval_height, eval_width = eval_spatial_size
            self._anchor_spatial_shapes = [
                [int(eval_height / s), int(eval_width / s)] for s in self.feat_strides
            ]
            anchors, valid_mask = self._generate_anchors(self._anchor_spatial_shapes)
            self.register_buffer("anchors", anchors, persistent=False)
            self.register_buffer("valid_mask", valid_mask, persistent=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Start every class at a 1% score and every box at its anchor or reference."""
        bias = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.enc_score_head.bias, bias)
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        for score_head, bbox_head in zip(self.dec_score_head, self.dec_bbox_head):
            nn.init.constant_(score_head.bias, bias)
            nn.init.constant_(bbox_head.layers[-1].weight, 0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0)
        nn.init.xavier_uniform_(self.enc_output[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        nn.init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)

    def _get_encoder_input(
        self, feats: Sequence[Float[Tensor, "batch_size channels height width"]]
    ) -> tuple[Float[Tensor, "batch_size num_tokens hidden_dim"], list[list[int]]]:
        """Project the encoder outputs and flatten them into one token sequence.

        Args:
            feats: Encoder output of every level.

        Returns:
            The tokens of every level, flattened row by row and concatenated level by level,
            and the ``(height, width)`` of each level.
        """
        feat_flatten = []
        spatial_shapes = []
        for proj, feat in zip(self.input_proj, feats):
            feat = proj(feat)
            spatial_shapes.append(list(feat.shape[2:]))
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(
        self,
        spatial_shapes: Sequence[Sequence[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> tuple[Float[Tensor, "1 num_tokens 4"], Bool[Tensor, "1 num_tokens 1"]]:
        """Build one anchor box per memory token.

        Each anchor is centered on its token, with a size of ``grid_size`` of the image on
        the finest level, doubled on each coarser one.

        Args:
            spatial_shapes: ``(height, width)`` of each memory level.
            grid_size: Anchor size on the finest level, relative to the image size.
            dtype: Anchor dtype.
            device: Anchor device.

        Returns:
            The inverse sigmoid of every anchor, +inf for the ones closer than ``eps`` to the
            border, and the mask of the others.
        """
        anchors = []
        for level, (height, width) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height), torch.arange(width), indexing="ij"
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([width, height], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**level)
            anchors.append(torch.concat([grid_xy, wh], dim=-1).reshape(-1, height * width, 4))

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _get_decoder_input(
        self,
        memory: Float[Tensor, "batch_size num_tokens hidden_dim"],
        spatial_shapes: list[list[int]],
        denoising: DenoisingQueries | None = None,
    ) -> tuple[
        Float[Tensor, "batch_size num_decoder_queries hidden_dim"],
        Float[Tensor, "batch_size num_decoder_queries 4"],
        Float[Tensor, "batch_size num_queries num_classes"],
        Float[Tensor, "batch_size num_queries 4"],
    ]:
        """Score every memory token and select the top ``num_queries`` as initial queries.

        Args:
            memory: Encoder tokens of every level, flattened and concatenated.
            spatial_shapes: ``(height, width)`` of each memory level.
            denoising: Denoising queries to put in front of the selected ones.

        Returns:
            The initial query content and the inverse sigmoid of the initial boxes, the
            denoising queries first, both detached from the query selection; and the logits
            and the boxes the query selection gave the selected tokens, for its loss.
        """
        if not self.training and spatial_shapes == self._anchor_spatial_shapes:
            anchors, valid_mask = self.anchors, self.valid_mask
        else:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)

        # A multiply rather than torch.where keeps the ONNX export type-consistent.
        memory = valid_mask.to(memory.dtype) * memory

        output_memory = self.enc_output(memory)
        enc_outputs_logits = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        # Class-aware selection: the tokens with the highest score of any class.
        _, topk_ind = torch.topk(enc_outputs_logits.max(-1).values, self.num_queries, dim=-1)
        topk_ind = topk_ind.unsqueeze(-1)
        enc_topk_bbox_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.repeat(1, 1, enc_outputs_coord_unact.shape[-1])
        )
        enc_topk_logits = enc_outputs_logits.gather(
            dim=1, index=topk_ind.repeat(1, 1, enc_outputs_logits.shape[-1])
        )
        enc_topk_memory = output_memory.gather(
            dim=1, index=topk_ind.repeat(1, 1, output_memory.shape[-1])
        )
        enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)

        content = enc_topk_memory.detach()
        ref_points_unact = enc_topk_bbox_unact.detach()
        if denoising is not None:
            ref_points_unact = torch.concat([denoising.boxes_unact, ref_points_unact], dim=1)
            content = torch.concat([denoising.content, content], dim=1)

        return content, ref_points_unact, enc_topk_logits, enc_topk_bboxes

    def forward(
        self,
        feats: Sequence[Float[Tensor, "batch_size channels height width"]],
        gt_labels: Sequence[Int64[Tensor, " num_boxes"]] | None = None,
        gt_boxes: Sequence[Float[Tensor, "num_boxes 4"]] | None = None,
    ) -> RTDETRv2HeadOutputs:
        """Predict the boxes and class logits of the queries.

        Args:
            feats: Encoder output of every level of ``feat_channels``.
            gt_labels: Class index of every box of every image; required in training with
                denoising.
            gt_boxes: ``(cx, cy, w, h)`` of every box of every image, normalized by the image
                size; required in training with denoising.

        Returns:
            The predictions of every decoder layer, the query selection and the denoising
            queries in training; of the ``eval_idx`` layer only at inference.

        Raises:
            ValueError: If the ground truth is missing in training with denoising.
        """
        memory, spatial_shapes = self._get_encoder_input(feats)

        denoising = None
        if self.training and self.num_denoising > 0:
            if gt_labels is None or gt_boxes is None:
                raise ValueError("Training with denoising needs gt_labels and gt_boxes.")
            denoising = build_contrastive_denoising_queries(
                gt_labels,
                gt_boxes,
                self.num_classes,
                self.num_queries,
                self.denoising_class_embed,
                num_denoising=self.num_denoising,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
            )

        content, ref_points_unact, enc_topk_logits, enc_topk_bboxes = self._get_decoder_input(
            memory, spatial_shapes, denoising
        )
        out_bboxes, out_logits = self.decoder(
            content,
            ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=denoising.attn_mask if denoising is not None else None,
        )

        if not self.training:
            return RTDETRv2HeadOutputs(all_cls_scores=out_logits, all_bbox_preds=out_bboxes)

        denoising_outputs = None
        if denoising is not None:
            split = [denoising.num_queries, self.num_queries]
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, split, dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, split, dim=2)
            denoising_outputs = RTDETRv2DenoisingOutputs(
                cls_scores=dn_out_logits,
                bbox_preds=dn_out_bboxes,
                positive_indices=denoising.positive_indices,
                num_groups=denoising.num_groups,
            )
        return RTDETRv2HeadOutputs(
            all_cls_scores=out_logits,
            all_bbox_preds=out_bboxes,
            encoder_cls_scores=enc_topk_logits,
            encoder_bbox_preds=enc_topk_bboxes,
            denoising=denoising_outputs,
        )
