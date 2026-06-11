from enum import Enum


class Modality(str, Enum):
    """
    Modality.

    Attributes:
      LIDAR: Lidar modality.
      CAMERA: Camera modality.
      RADAR: Radar modality.
    """

    LIDAR = "lidar"
    CAMERA = "camera"
    RADAR = "radar"


class LidarChannel(str, Enum):
    """
    Lidar channel in Dataset.

    Attributes:
      LIDAR_TOP: Top lidar channel.
      LIDAR_CONCAT: Concatenated lidar channel.
    """

    LIDAR_TOP = "LIDAR_TOP"
    LIDAR_CONCAT = "LIDAR_CONCAT"
