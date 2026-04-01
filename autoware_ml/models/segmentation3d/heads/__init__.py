"""Segmentation3D head exports used by segmentation models.

This package re-exports reusable prediction heads shared across segmentation
architectures.
"""

from autoware_ml.models.segmentation3d.heads.frnet import FRHead, FrustumHead

__all__ = ["FRHead", "FrustumHead"]
