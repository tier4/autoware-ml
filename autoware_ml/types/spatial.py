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
