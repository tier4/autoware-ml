"""
Modules to save raw outputs from a detection3d head.
"""

from __future__ import annotations

from typing import Sequence
from types import MappingProxyType

from jaxtyping import Float32, Int64
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class TransFusionSeparateHeadOutputs(BaseModel):
    """
    Dataclass to save the outputs (1D) from a separate head in a Transfusion-based 3D detection model.

    Attributes:
      heatmaps: Heatmap to save probability for each class in a BEV heatmap.
      centers: Center_x and center_y translation from each cell in a BEV heatmap.
      heights: Height value from each cell in a BEV heatmap.
      dims: Dimension values (length, width, height) from each cell in a BEV heatmap.
      rots: Rotation values (sin, cos) from each cell in a BEV heatmap.
      vels: Velocity values (vel_x, vel_y) from each cell in a BEV heatmap.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    heatmaps: Float32[torch.Tensor, "batch_size num_classes num_proposals"]
    centers: Float32[torch.Tensor, "batch_size 2 num_proposals"]
    heights: Float32[torch.Tensor, "batch_size 1 num_proposals"]
    dims: Float32[torch.Tensor, "batch_size 3 num_proposals"]
    rots: Float32[torch.Tensor, "batch_size 2 num_proposals"]
    vels: Float32[torch.Tensor, "batch_size 2 num_proposals"] | None

    @model_validator(mode="after")
    def _check_batch_and_proposals(self) -> TransFusionSeparateHeadOutputs:
        """Check that every branch covers the same batch and the same proposals as heatmaps."""
        batch_size, _, num_proposals = self.heatmaps.shape
        for name, tensor in (
            ("centers", self.centers),
            ("heights", self.heights),
            ("dims", self.dims),
            ("rots", self.rots),
            ("vels", self.vels),
        ):
            if tensor is None:
                continue
            if tensor.shape[0] != batch_size or tensor.shape[2] != num_proposals:
                raise ValueError(
                    f"{name} must have shape (batch_size={batch_size}, channels, "
                    f"num_proposals={num_proposals}), got {tuple(tensor.shape)}."
                )
        return self

    @classmethod
    def from_dict(
        cls, data: MappingProxyType[str, Float32[torch.Tensor, "batch_size channels num_proposals"]]
    ) -> TransFusionSeparateHeadOutputs:
        """
        Create a TransFusionSeparateHeadOutputs instance from a dictionary.

        Args:
            data: A dictionary containing the output tensors.
        """
        vels = None if "vels" not in data else data.get("vels")
        return cls(
            heatmaps=data.get("heatmaps"),
            centers=data.get("centers"),
            heights=data.get("heights"),
            dims=data.get("dims"),
            rots=data.get("rots"),
            vels=vels,
        )

    @property
    def ordered_keys(self) -> Sequence[str]:
        """
        Get the ordered keys of the output tensors.

        Returns:
            A list of ordered keys.
        """
        if self.vels is not None:
            return ["heatmaps", "centers", "heights", "dims", "rots", "vels"]
        return ["heatmaps", "centers", "heights", "dims", "rots"]


class TransFusionHeadOutputs(BaseModel):
    """
    Dataclass to save Transfusion-based outputs from a 3D detection model.

    Attributes:
      dense_heatmaps: Heatmap to save probability for each class in a BEV heatmap.
      query_heatmap_scores: Heatmap scores gathered at each query position.
      query_labels: Class index predicted for each query.
      separate_head_outputs: Per-proposal outputs from the separate head.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    dense_heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"]
    query_heatmap_scores: Float32[torch.Tensor, "batch_size num_classes num_queries"]
    query_labels: Int64[torch.Tensor, "batch_size num_queries"]

    separate_head_outputs: TransFusionSeparateHeadOutputs

    @model_validator(mode="after")
    def _check_batch_classes_and_queries(self) -> TransFusionHeadOutputs:
        """
        Check that the dense, query and per-proposal outputs share a batch and class count.

        ``num_queries`` and ``num_proposals`` are deliberately not compared: with auxiliary
        decoder layers the separate head concatenates the proposals of every layer.
        """
        batch_size, num_classes = self.dense_heatmaps.shape[:2]
        num_queries = self.query_labels.shape[1]
        if self.query_labels.shape[0] != batch_size:
            raise ValueError(
                f"query_labels must have batch_size={batch_size}, got {tuple(self.query_labels.shape)}."
            )
        if self.query_heatmap_scores.shape != (batch_size, num_classes, num_queries):
            raise ValueError(
                f"query_heatmap_scores must have shape ({batch_size}, {num_classes}, "
                f"{num_queries}), got {tuple(self.query_heatmap_scores.shape)}."
            )
        if self.separate_head_outputs.heatmaps.shape[:2] != (batch_size, num_classes):
            raise ValueError(
                f"separate_head_outputs.heatmaps must have batch_size={batch_size} and "
                f"num_classes={num_classes}, got {tuple(self.separate_head_outputs.heatmaps.shape)}."
            )
        return self


class CenterHeadOutputs(BaseModel):
    """
    Dataclass to save CenterHead-based outputs from a 3D detection model.

    Attributes:
      heatmaps: Heatmap to save probability for each class in a BEV heatmap.
      centers: Center_x and center_y translation from each cell in a BEV heatmap.
      heights: Height value from each cell in a BEV heatmap.
      dims: Dimension values (length, width, height) from each cell in a BEV heatmap.
      rots: Rotation values (sin, cos) from each cell in a BEV heatmap.
      vels: Velocity values (vel_x, vel_y) from each cell in a BEV heatmap.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)
    heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"]
    centers: Float32[torch.Tensor, "batch_size 2 height width"]
    heights: Float32[torch.Tensor, "batch_size 1 height width"]
    dims: Float32[torch.Tensor, "batch_size 3 height width"]
    rots: Float32[torch.Tensor, "batch_size 2 height width"]
    vels: Float32[torch.Tensor, "batch_size 2 height width"] | None

    @model_validator(mode="after")
    def _check_batch_and_grid(self) -> CenterHeadOutputs:
        """Check that every branch covers the same batch on the same BEV grid as heatmaps."""
        batch_size, _, height, width = self.heatmaps.shape
        for name, tensor in (
            ("centers", self.centers),
            ("heights", self.heights),
            ("dims", self.dims),
            ("rots", self.rots),
            ("vels", self.vels),
        ):
            if tensor is None:
                continue
            if tensor.shape[0] != batch_size or tensor.shape[2:] != (height, width):
                raise ValueError(
                    f"{name} must have shape (batch_size={batch_size}, channels, height={height}, "
                    f"width={width}), got {tuple(tensor.shape)}."
                )
        return self


class Detection3DHeadOutputs(BaseModel):
    """
    Dataclass to save outputs from 3D detection models.

    Exactly one of the two heads must be set.

    Attributes:
      center_head_outputs: Outputs from a CenterHead-based 3D detection model.
      transfusion_head_outputs: Outputs from a TransFusion-based 3D detection model.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    center_head_outputs: CenterHeadOutputs | None
    transfusion_head_outputs: TransFusionHeadOutputs | None

    @model_validator(mode="after")
    def _check_exactly_one_head(self) -> Detection3DHeadOutputs:
        """Check that exactly one head family produced the outputs."""
        has_center_head = self.center_head_outputs is not None
        has_transfusion_head = self.transfusion_head_outputs is not None
        if has_center_head == has_transfusion_head:
            raise ValueError(
                "Exactly one of center_head_outputs and transfusion_head_outputs must be set, "
                f"got center_head_outputs={has_center_head} and "
                f"transfusion_head_outputs={has_transfusion_head}."
            )
        return self
