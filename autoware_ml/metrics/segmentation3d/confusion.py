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

"""Confusion-matrix state shared by the segmentation metrics.

A confusion matrix is a bounded sufficient statistic: every segmentation metric
is an exact closed form of its integer counts. ``ConfusionState`` is what the
suite hands to each metric. It exposes the cheap shared primitives (true
positives, predicted/actual marginals, support) as cached properties, so each
metric computes its own quantity (IoU, accuracy, precision/recall/F1) without
re-deriving the marginals.
"""

from __future__ import annotations

from functools import cached_property

import torch
from jaxtyping import Bool, Float, Float64, Int64
from pydantic import BaseModel, ConfigDict, model_validator


def macro(
    values: Float[torch.Tensor, " num_classes"], has_support: Bool[torch.Tensor, " num_classes"]
) -> float:
    """Mean over classes with ground-truth support, or NaN when none have it.

    Args:
        values: Per-class values.
        has_support: Mask of classes present in ground truth.

    Returns:
        The mean over supported classes.
    """
    if not bool(has_support.any()):
        return float("nan")
    return float(values[has_support].mean().item())


def class_name_token(class_index: int, class_names: tuple[str, ...] | None) -> str:
    """Per-class key token, falling back to ``class_{index}``.

    Args:
        class_index: Class index.
        class_names: Class names in index order, or ``None``.

    Returns:
        The class token.
    """
    if class_names is not None and class_index < len(class_names):
        return class_names[class_index]
    return f"class_{class_index}"


class ConfusionState(BaseModel):
    """
    The synced confusion matrix of one bucket, with cached marginals.

    In a grouped suite the matrix is already folded onto the behaviour taxonomy, so
    ``class_names`` are the grouped names and every metric reads it exactly as it reads the
    per-class matrix.

    Attributes:
      confusion: Point counts, the ground truth class along the rows and the predicted class
        along the columns.
      class_names: Class names in index order for the metric keys, or ``None``.
      num_classes: Number of classes the matrix covers.
      num_frames: Frames the suite folded into the filter slice of this bucket, the same for
        every range window.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    confusion: Int64[torch.Tensor, "num_classes num_classes"]
    class_names: tuple[str, ...] | None
    num_classes: int
    num_frames: int

    @model_validator(mode="after")
    def _check_num_classes(self) -> ConfusionState:
        """Check that the matrix and the names cover exactly the declared classes."""
        if self.confusion.shape[0] != self.num_classes:
            raise ValueError(
                f"confusion covers {self.confusion.shape[0]} classes, num_classes is "
                f"{self.num_classes}."
            )
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names names {len(self.class_names)} classes, num_classes is "
                f"{self.num_classes}."
            )
        return self

    @cached_property
    def _double(self) -> Float64[torch.Tensor, "num_classes num_classes"]:
        return self.confusion.double()

    @cached_property
    def true_positive(self) -> Float64[torch.Tensor, " num_classes"]:
        """Return per-class diagonal counts."""
        return self._double.diag()

    @cached_property
    def predicted(self) -> Float64[torch.Tensor, " num_classes"]:
        """Return predicted counts per class."""
        return self._double.sum(dim=0)  # column sums: predicted count per class

    @cached_property
    def actual(self) -> Float64[torch.Tensor, " num_classes"]:
        """Return ground-truth counts per class."""
        return self._double.sum(dim=1)  # row sums: ground-truth count per class

    @cached_property
    def total(self) -> Float64[torch.Tensor, ""]:
        """Return the total number of valid points in the matrix."""
        return self._double.sum()

    @cached_property
    def has_support(self) -> Bool[torch.Tensor, " num_classes"]:
        """Return a mask for classes present in ground truth."""
        return self.actual > 0

    @cached_property
    def frequency(self) -> Float64[torch.Tensor, " num_classes"]:
        """Return per-class ground-truth frequencies."""
        zeros = torch.zeros_like(self.actual)
        return self.actual / self.total if self.total > 0 else zeros
