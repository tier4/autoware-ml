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


class CameraChannel(str, Enum):
    """
    Camera channel in Dataset.

    Attributes:
      CAM_FRONT: Front camera channel.
      CAM_FRONT_RIGHT: Front right camera channel.
      CAM_FRONT_LEFT: Front left camera channel.
      CAM_BACK: Back camera channel.
      CAM_BACK_LEFT: Back left camera channel.
      CAM_BACK_RIGHT: Back right camera channel.
      CAM_FRONT_WIDE: Front wide camera channel.
      CAM_FRONT_LEFT_WIDE: Front left wide camera channel.
      CAM_FRONT_RIGHT_WIDE: Front right wide camera channel.
      CAM_BACK_LEFT_WIDE: Back left wide camera channel.
      CAM_BACK_RIGHT_WIDE: Back right wide camera channel.
    """

    CAM_FRONT = "CAM_FRONT"
    CAM_FRONT_RIGHT = "CAM_FRONT_RIGHT"
    CAM_FRONT_LEFT = "CAM_FRONT_LEFT"
    CAM_BACK = "CAM_BACK"
    CAM_BACK_LEFT = "CAM_BACK_LEFT"
    CAM_BACK_RIGHT = "CAM_BACK_RIGHT"
    # jpntaxigen2 wide camera channels
    CAM_FRONT_WIDE = "CAM_FRONT_WIDE"
    CAM_FRONT_LEFT_WIDE = "CAM_FRONT_LEFT_WIDE"
    CAM_FRONT_RIGHT_WIDE = "CAM_FRONT_RIGHT_WIDE"
    CAM_BACK_LEFT_WIDE = "CAM_BACK_LEFT_WIDE"
    CAM_BACK_RIGHT_WIDE = "CAM_BACK_RIGHT_WIDE"
