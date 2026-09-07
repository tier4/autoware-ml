"""Shared emission of a confusion matrix as flat metric keys.

Both the detection (matched-pair) and segmentation (point) confusion metrics turn
their accumulated ``(C, C)`` matrix (rows = true class, columns = predicted class)
into ``confusion_<true>__<pred>`` count keys. The ``__`` separator is safe
because no trained class name contains a double underscore. Raw counts are emitted
(not fractions): the report row-normalizes for the heatmap, so both the absolute
count and the per-true-class rate stay recoverable.
"""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.detection3d.naming import metric_token


def confusion_class_token(index: int, class_names: tuple[str, ...] | None) -> str:
    """Sanitized per-class key token, ``class_{index}`` only when no names exist.

    With ``class_names`` configured an out-of-range index is a folding or
    configuration bug and raises instead of silently minting a phantom class key.

    Args:
        index: Class index in the reported space.
        class_names: Class names in index order, or ``None``.

    Returns:
        The sanitized class token.
    """
    if class_names is None:
        return f"class_{index}"
    if not 0 <= index < len(class_names):
        raise ValueError(
            f"class index {index} is outside the configured class set ({len(class_names)} classes)."
        )
    return metric_token(class_names[index])


def confusion_cells(matrix: np.ndarray, class_names: tuple[str, ...] | None) -> dict[str, float]:
    """Flatten a ``(C, C)`` true vs. pred matrix into ``confusion_<true>__<pred>`` counts.

    Args:
        matrix: Confusion counts with true classes as rows.
        class_names: Class names in index order, or ``None``.

    Returns:
        One count per cell, keyed by the true and predicted class tokens.
    """
    num_classes = int(matrix.shape[0])
    cells: dict[str, float] = {}
    for true_index in range(num_classes):
        true_name = confusion_class_token(true_index, class_names)
        for pred_index in range(num_classes):
            pred_name = confusion_class_token(pred_index, class_names)
            cells[f"confusion_{true_name}__{pred_name}"] = float(matrix[true_index, pred_index])
    return cells
