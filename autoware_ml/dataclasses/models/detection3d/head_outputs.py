"""
Modules to save raw outputs from a detection3d head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jaxtyping import Bool, Float, Int64
from pydantic import BaseModel, ConfigDict, model_validator

import torch


class TransFusionSeparateHeadOutputs(BaseModel):
    """
    Dataclass to save the outputs (1D) from a separate head in a Transfusion-based 3D detection model.

    Attributes:
      heatmap: Heatmap to save probability for each class in a BEV heatmap.
      center: Center_x and center_y translation from each cell in a BEV heatmap.
      height: Height value from each cell in a BEV heatmap.
      dim: Dimension values (length, width, height) from each cell in a BEV heatmap.
      rot: Rotation values (sin, cos) from each cell in a BEV heatmap.
      vel: Velocity values (vel_x, vel_y) from each cell in a BEV heatmap.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    heatmap: Float[torch.Tensor, "batch_size num_classes num_proposals"]
    center: Float[torch.Tensor, "batch_size 2 num_proposals"]
    height: Float[torch.Tensor, "batch_size 1 num_proposals"]
    dim: Float[torch.Tensor, "batch_size 3 num_proposals"]
    rot: Float[torch.Tensor, "batch_size 2 num_proposals"]
    vel: Float[torch.Tensor, "batch_size 2 num_proposals"] | None

    @model_validator(mode="after")
    def _check_batch_and_proposals(self) -> TransFusionSeparateHeadOutputs:
        """Check that every branch covers the same batch and the same proposals as heatmap."""
        batch_size, _, num_proposals = self.heatmap.shape
        for name, tensor in (
            ("center", self.center),
            ("height", self.height),
            ("dim", self.dim),
            ("rot", self.rot),
            ("vel", self.vel),
        ):
            if tensor is None:
                continue
            if tensor.shape[0] != batch_size or tensor.shape[2] != num_proposals:
                raise ValueError(
                    f"{name} must have shape (batch_size={batch_size}, channels, "
                    f"num_proposals={num_proposals}), got {tuple(tensor.shape)}."
                )
        return self


class TransFusionHeadOutputs(BaseModel):
    """
    Dataclass to save Transfusion-based outputs from a 3D detection model.

    Attributes:
      dense_heatmap: Heatmap to save probability for each class in a BEV heatmap.
      query_heatmap_score: Heatmap scores gathered at each query position.
      query_labels: Class index predicted for each query.
      separate_head_outputs: Per-proposal outputs from the separate head.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    dense_heatmap: Float[torch.Tensor, "batch_size num_classes height width"]
    query_heatmap_score: Float[torch.Tensor, "batch_size num_classes num_queries"]
    query_labels: Int64[torch.Tensor, "batch_size num_queries"]

    separate_head_outputs: TransFusionSeparateHeadOutputs

    @model_validator(mode="after")
    def _check_batch_classes_and_queries(self) -> TransFusionHeadOutputs:
        """
        Check that the dense, query and per-proposal outputs share a batch and class count.

        ``num_queries`` and ``num_proposals`` are deliberately not compared: with auxiliary
        decoder layers the separate head concatenates the proposals of every layer.
        """
        batch_size, num_classes = self.dense_heatmap.shape[:2]
        num_queries = self.query_labels.shape[1]
        if self.query_labels.shape[0] != batch_size:
            raise ValueError(
                f"query_labels must have batch_size={batch_size}, got {tuple(self.query_labels.shape)}."
            )
        if self.query_heatmap_score.shape != (batch_size, num_classes, num_queries):
            raise ValueError(
                f"query_heatmap_score must have shape ({batch_size}, {num_classes}, "
                f"{num_queries}), got {tuple(self.query_heatmap_score.shape)}."
            )
        if self.separate_head_outputs.heatmap.shape[:2] != (batch_size, num_classes):
            raise ValueError(
                f"separate_head_outputs.heatmap must have batch_size={batch_size} and "
                f"num_classes={num_classes}, got {tuple(self.separate_head_outputs.heatmap.shape)}."
            )
        return self

    def select_samples(self, mask: Bool[torch.Tensor, " batch_size"]) -> TransFusionHeadOutputs:
        """
        Keep the outputs of the samples a mask selects.

        Args:
            mask: True for every sample to keep.

        Returns:
            The outputs of the selected samples.
        """
        separate = self.separate_head_outputs
        return TransFusionHeadOutputs(
            dense_heatmap=self.dense_heatmap[mask],
            query_heatmap_score=self.query_heatmap_score[mask],
            query_labels=self.query_labels[mask],
            separate_head_outputs=TransFusionSeparateHeadOutputs(
                heatmap=separate.heatmap[mask],
                center=separate.center[mask],
                height=separate.height[mask],
                dim=separate.dim[mask],
                rot=separate.rot[mask],
                vel=None if separate.vel is None else separate.vel[mask],
            ),
        )

    def export_tensors(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """
        Pick the output tensors in the order an export names them.

        Args:
            names: Output names in the exported order, taken from both the dense and the
                per-proposal outputs.

        Returns:
            The tensors of the names, in the same order.

        Raises:
            ValueError: If a name is not an output of the head or the head left it unset.
        """
        separate = self.separate_head_outputs
        tensors = {
            "dense_heatmap": self.dense_heatmap,
            "query_heatmap_score": self.query_heatmap_score,
            "query_labels": self.query_labels,
        } | {name: getattr(separate, name) for name in type(separate).model_fields}
        return _pick_export_tensors(tensors, names)


class CenterHeadOutputs(BaseModel):
    """
    Dataclass to save CenterHead-based outputs from a 3D detection model.

    Attributes:
      heatmap: Heatmap to save probability for each class in a BEV heatmap.
      reg: Center_x and center_y offset from each cell in a BEV heatmap.
      height: Height value from each cell in a BEV heatmap.
      dim: Dimension values (length, width, height) from each cell in a BEV heatmap.
      rot: Rotation values (sin, cos) from each cell in a BEV heatmap.
      vel: Velocity values (vel_x, vel_y) from each cell in a BEV heatmap.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)
    heatmap: Float[torch.Tensor, "batch_size num_classes height width"]
    reg: Float[torch.Tensor, "batch_size 2 height width"]
    height: Float[torch.Tensor, "batch_size 1 height width"]
    dim: Float[torch.Tensor, "batch_size 3 height width"]
    rot: Float[torch.Tensor, "batch_size 2 height width"]
    vel: Float[torch.Tensor, "batch_size 2 height width"] | None

    @model_validator(mode="after")
    def _check_batch_and_grid(self) -> CenterHeadOutputs:
        """Check that every branch covers the same batch on the same BEV grid as heatmap."""
        batch_size, _, height, width = self.heatmap.shape
        for name, tensor in (
            ("reg", self.reg),
            ("height", self.height),
            ("dim", self.dim),
            ("rot", self.rot),
            ("vel", self.vel),
        ):
            if tensor is None:
                continue
            if tensor.shape[0] != batch_size or tensor.shape[2:] != (height, width):
                raise ValueError(
                    f"{name} must have shape (batch_size={batch_size}, channels, height={height}, "
                    f"width={width}), got {tuple(tensor.shape)}."
                )
        return self

    def export_tensors(self, names: Sequence[str]) -> tuple[torch.Tensor, ...]:
        """
        Pick the output tensors in the order an export names them.

        Args:
            names: Output names in the exported order.

        Returns:
            The tensors of the names, in the same order.

        Raises:
            ValueError: If a name is not an output of the head or the head left it unset.
        """
        tensors = {name: getattr(self, name) for name in type(self).model_fields}
        return _pick_export_tensors(tensors, names)


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

    def center_head(self) -> CenterHeadOutputs:
        """
        Read the outputs of a CenterHead-based model.

        Raises:
            ValueError: If a TransFusion-based head produced the outputs.
        """
        if self.center_head_outputs is None:
            raise ValueError("The detection outputs come from a TransFusion head.")
        return self.center_head_outputs

    def transfusion_head(self) -> TransFusionHeadOutputs:
        """
        Read the outputs of a TransFusion-based model.

        Raises:
            ValueError: If a CenterHead-based head produced the outputs.
        """
        if self.transfusion_head_outputs is None:
            raise ValueError("The detection outputs come from a CenterHead.")
        return self.transfusion_head_outputs

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


def _pick_export_tensors(
    tensors: Mapping[str, torch.Tensor | None], names: Sequence[str]
) -> tuple[torch.Tensor, ...]:
    """
    Pick the named head outputs in the exported order.

    Args:
        tensors: Every output of a head keyed by its tensor name, unset ones as None.
        names: Output names in the exported order.

    Returns:
        The tensors of the names, in the same order.

    Raises:
        ValueError: If a name is not an output of the head or the head left it unset.
    """
    unknown = [name for name in names if name not in tensors or tensors[name] is None]
    if unknown:
        raise ValueError(
            f"The head has no output named {unknown}, it provides "
            f"{[name for name, tensor in tensors.items() if tensor is not None]}."
        )
    return tuple(tensors[name] for name in names)
