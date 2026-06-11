from enum import Enum


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
