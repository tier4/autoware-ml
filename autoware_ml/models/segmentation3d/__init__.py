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

"""Segmentation3D model namespace for Autoware-ML.

This package re-exports point-cloud and range-view segmentation model
components used by training and deployment entrypoints.
"""

from autoware_ml.models.segmentation3d.backbones import FRNetBackbone, PointTransformerV3Backbone
from autoware_ml.models.segmentation3d.base import BaseSegmentationModel
from autoware_ml.models.segmentation3d.encoders import FrustumFeatureEncoder
from autoware_ml.models.segmentation3d.frnet import FRNet
from autoware_ml.models.segmentation3d.heads import FRHead, FrustumHead
from autoware_ml.models.segmentation3d.ptv3 import PTv3SegmentationModel

__all__ = [
    "BaseSegmentationModel",
    "FRHead",
    "FRNet",
    "FRNetBackbone",
    "FrustumFeatureEncoder",
    "FrustumHead",
    "PTv3SegmentationModel",
    "PointTransformerV3Backbone",
]
