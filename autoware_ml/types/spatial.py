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

"""Enumerations of coordinate systems, BEV directions and rotation axes."""

from enum import Enum, StrEnum, IntEnum


class CoordinateSystem(str, Enum):
    """
    Coordinate system.

    Attributes:
        LIDAR_COMMON: Coordinate system in a multi-LiDARs setup, for example, concatenated pointclouds.
        LIDAR_LOCAL: Coordinate system in a single LiDAR sensor.
        CAMERA_CAMERA: Coordinate system in a multi-Cameras setup.
        CAMERA_LOCAL: Coordinate system in a single Camera sensor.
        EGO: Ego coordinate system, for example, the vehicle's coordinate system.
        GLOBAL: Global coordinate system.
        MAP: Map coordinate system.
    """

    LIDAR_COMMON = "lidar_common"
    LIDAR_LOCAL = "lidar_local"
    CAMERA_CAMERA = "camera_common"
    CAMERA_LOCAL = "camera_local"
    EGO = "ego"
    GLOBAL = "global"
    MAP = "map"


class BEVDirection(StrEnum):
    """
    BEV direction.

    Attributes:
      HORIZONTAL: Horizontal direction.
      VERTICAL: Vertical direction.
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class RotationAxis(IntEnum):
    """
    Rotation axis.

    Attributes:
      X: X axis.
      Y: Y axis.
      Z: Z axis.
    """

    X = 0
    Y = 1
    Z = 2
