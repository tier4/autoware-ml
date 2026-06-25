"""Metric-key token helpers for detection components.

Keeps key formatting in one place so components stay focused on selecting values.
"""

from __future__ import annotations


def metric_token(value: str) -> str:
    """Lowercase, underscore-separated token safe for a metric key."""
    return value.lower().replace(" ", "_").replace("/", "_")


def label_metric_name(label: int, class_names: tuple[str, ...] | None) -> str:
    """Token for a class label, falling back to ``class_{label}``."""
    if class_names is not None and 0 <= label < len(class_names):
        return metric_token(class_names[label])
    return f"class_{label}"


def threshold_token(threshold: float) -> str:
    """Collision-free token for a distance threshold, e.g. ``0p5m`` for 0.5."""
    token = f"{float(threshold):g}".replace("-", "minus").replace(".", "p")
    return f"{token}m"
