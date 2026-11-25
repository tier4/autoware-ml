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

"""
Modules to save raw outputs from a detection2d head.
"""

from __future__ import annotations

from jaxtyping import Float, Float32, Int64
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class FocalHead2DOutputs(BaseModel):
    """
    Dataclass to save the dense outputs of an auxiliary 2D head over the image tokens.

    Every camera image of every sample gets one prediction per feature map token, the images
    ordered sample by sample.

    Attributes:
      cls_scores: Class logits of every token.
      bbox_preds: Box ``(cx, cy, w, h)`` of every token, normalized by the image size.
      centers_2d: Projected 3D center of every token, normalized by the image size.
      heatmap: Center heatmap logit of every token, one per class.
      feature_height: Rows of the token grid of every image; the tokens are flattened row by
        row, so the heatmap targets are drawn on this grid.
      feature_width: Columns of the token grid of every image.
      image_height: Height in pixels of the images the predictions are normalized by.
      image_width: Width in pixels of the images the predictions are normalized by.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    cls_scores: Float32[torch.Tensor, "num_images num_tokens num_classes"]
    bbox_preds: Float32[torch.Tensor, "num_images num_tokens 4"]
    centers_2d: Float32[torch.Tensor, "num_images num_tokens 2"]
    heatmap: Float32[torch.Tensor, "num_images num_tokens num_classes"]
    feature_height: int
    feature_width: int
    image_height: int
    image_width: int

    @model_validator(mode="after")
    def _check_images_and_tokens(self) -> FocalHead2DOutputs:
        """Check that every prediction covers the same images and the tokens of the grid."""
        images_and_tokens = self.cls_scores.shape[:2]
        for name, tensor in (
            ("bbox_preds", self.bbox_preds),
            ("centers_2d", self.centers_2d),
            ("heatmap", self.heatmap),
        ):
            if tensor.shape[:2] != images_and_tokens:
                raise ValueError(
                    f"{name} must cover {tuple(images_and_tokens)} images and tokens, got "
                    f"{tuple(tensor.shape)}."
                )
        if self.heatmap.shape[2] != self.cls_scores.shape[2]:
            raise ValueError(
                f"heatmap must have one channel per class of cls_scores, "
                f"{self.cls_scores.shape[2]}, got {self.heatmap.shape[2]}."
            )
        if self.feature_height * self.feature_width != images_and_tokens[1]:
            raise ValueError(
                f"The predictions cover {images_and_tokens[1]} tokens per image but the token "
                f"grid is {self.feature_height} x {self.feature_width}."
            )
        return self


class RTDETRv2DenoisingOutputs(BaseModel):
    """
    Dataclass to save the predictions of the denoising queries of an RT-DETRv2 decoder.

    The queries form ``num_groups`` groups of noised copies of the ground truth boxes, each
    group a positive per box, padded to the largest box count of the batch, then as many
    negatives. The loss matches every positive with the box it was noised from.

    Attributes:
      cls_scores: Class logits of every denoising query, per decoder layer.
      bbox_preds: ``(cx, cy, w, h)`` box of every denoising query, per decoder layer,
        normalized by the image size.
      positive_indices: For every image, the query index of each of its positives, group by
        group and box by box within a group.
      num_groups: Number of groups.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    cls_scores: Float[torch.Tensor, "num_layers batch_size num_denoising num_classes"]
    bbox_preds: Float[torch.Tensor, "num_layers batch_size num_denoising 4"]
    positive_indices: tuple[Int64[torch.Tensor, " num_positives"], ...]
    num_groups: int

    @model_validator(mode="after")
    def _check_queries_and_images(self) -> RTDETRv2DenoisingOutputs:
        """Check that the logits and the boxes cover the same queries and every image."""
        if self.cls_scores.shape[:3] != self.bbox_preds.shape[:3]:
            raise ValueError(
                "cls_scores and bbox_preds must cover the same layers, images and queries, got "
                f"{tuple(self.cls_scores.shape)} and {tuple(self.bbox_preds.shape)}."
            )
        if len(self.positive_indices) != self.cls_scores.shape[1]:
            raise ValueError(
                f"positive_indices must hold one entry per image ({self.cls_scores.shape[1]}), "
                f"got {len(self.positive_indices)}."
            )
        return self


class RTDETRv2HeadOutputs(BaseModel):
    """
    Dataclass to save the outputs of an RT-DETRv2 decoder.

    Attributes:
      all_cls_scores: Class logits of the matching queries: of every decoder layer in
        training, of the ``eval_idx`` layer only at inference.
      all_bbox_preds: ``(cx, cy, w, h)`` box of the matching queries, normalized by the image
        size, for the same layers as ``all_cls_scores``.
      encoder_cls_scores: Class logits the query selection gave the selected encoder tokens,
        in training only.
      encoder_bbox_preds: Box the query selection gave the selected encoder tokens, in
        training only.
      denoising: Predictions of the denoising queries, in training with denoising only.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    all_cls_scores: Float[torch.Tensor, "num_layers batch_size num_queries num_classes"]
    all_bbox_preds: Float[torch.Tensor, "num_layers batch_size num_queries 4"]
    encoder_cls_scores: Float[torch.Tensor, "batch_size num_queries num_classes"] | None = None
    encoder_bbox_preds: Float[torch.Tensor, "batch_size num_queries 4"] | None = None
    denoising: RTDETRv2DenoisingOutputs | None = None

    @model_validator(mode="after")
    def _check_layers_images_and_queries(self) -> RTDETRv2HeadOutputs:
        """Check that every prediction describes the same images, queries and classes."""
        if self.all_cls_scores.shape[:3] != self.all_bbox_preds.shape[:3]:
            raise ValueError(
                "all_cls_scores and all_bbox_preds must cover the same layers, images and "
                f"queries, got {tuple(self.all_cls_scores.shape)} and "
                f"{tuple(self.all_bbox_preds.shape)}."
            )
        num_layers, batch_size, num_queries, num_classes = self.all_cls_scores.shape
        if (self.encoder_cls_scores is None) != (self.encoder_bbox_preds is None):
            raise ValueError("encoder_cls_scores and encoder_bbox_preds must be set together.")
        if self.encoder_cls_scores is not None and (
            self.encoder_cls_scores.shape != (batch_size, num_queries, num_classes)
            or self.encoder_bbox_preds.shape != (batch_size, num_queries, 4)
        ):
            raise ValueError(
                "The encoder predictions must cover the images, queries and classes of "
                f"all_cls_scores {tuple(self.all_cls_scores.shape)}, got "
                f"{tuple(self.encoder_cls_scores.shape)} and "
                f"{tuple(self.encoder_bbox_preds.shape)}."
            )
        if self.denoising is not None:
            shape = self.denoising.cls_scores.shape
            if (shape[0], shape[1], shape[3]) != (num_layers, batch_size, num_classes):
                raise ValueError(
                    "The denoising queries must share the layers, images and classes of the "
                    f"matching queries, got {tuple(shape)} for "
                    f"{tuple(self.all_cls_scores.shape)}."
                )
        return self
