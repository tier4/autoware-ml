from enum import Enum


class CoordinateSystem(str, Enum):
    """
    Coordinate system.

    Attributes:
      LIDAR: Lidar coordinate system.
      EGO: Ego coordinate system.
      GLOBAL: Global coordinate system.
    """

    LIDAR = "lidar"
    EGO = "ego"
    GLOBAL = "global"
    MAP = "map"
