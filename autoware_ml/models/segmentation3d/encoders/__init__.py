"""Segmentation3D encoder exports used by segmentation models.

This package re-exports reusable encoder components shared across point and
range-view segmentation architectures.
"""

from autoware_ml.models.segmentation3d.encoders.frnet import FrustumFeatureEncoder

__all__ = ["FrustumFeatureEncoder"]
