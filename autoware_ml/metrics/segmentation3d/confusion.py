"""Confusion-matrix state shared by the segmentation metrics.

A confusion matrix is a bounded sufficient statistic: every segmentation metric
is an exact closed form of its integer counts. ``ConfusionState`` is what the
suite hands to each metric. It exposes the cheap shared primitives (true
positives, predicted/actual marginals, support) as cached properties, so each
metric computes its own quantity (IoU, accuracy, precision/recall/F1) without
re-deriving the marginals.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch


def macro(values: torch.Tensor, has_support: torch.Tensor) -> float:
    """Mean over classes with ground-truth support, or NaN when none have it."""
    if not bool(has_support.any()):
        return float("nan")
    return float(values[has_support].mean().item())


def class_name_token(class_index: int, class_names: tuple[str, ...] | None) -> str:
    """Per-class key token, falling back to ``class_{index}``."""
    if class_names is not None and class_index < len(class_names):
        return class_names[class_index]
    return f"class_{class_index}"


@dataclass
class ConfusionState:
    """Synced confusion matrix for one bucket, with cached marginals."""

    confusion: torch.Tensor
    class_names: tuple[str, ...] | None
    num_classes: int

    @cached_property
    def _double(self) -> torch.Tensor:
        return self.confusion.double()

    @cached_property
    def true_positive(self) -> torch.Tensor:
        """Return per-class diagonal counts."""
        return self._double.diag()

    @cached_property
    def predicted(self) -> torch.Tensor:
        """Return predicted counts per class."""
        return self._double.sum(dim=0)  # column sums: predicted count per class

    @cached_property
    def actual(self) -> torch.Tensor:
        """Return ground-truth counts per class."""
        return self._double.sum(dim=1)  # row sums: ground-truth count per class

    @cached_property
    def total(self) -> torch.Tensor:
        """Return the total number of valid points in the matrix."""
        return self._double.sum()

    @cached_property
    def has_support(self) -> torch.Tensor:
        """Return a mask for classes present in ground truth."""
        return self.actual > 0

    @cached_property
    def frequency(self) -> torch.Tensor:
        """Return per-class ground-truth frequencies."""
        zeros = torch.zeros_like(self.actual)
        return self.actual / self.total if self.total > 0 else zeros
