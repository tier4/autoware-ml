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


class SplitType(str, Enum):
    """
    Split type.

    Attributes:
      TRAIN: Training split.
      VAL: Validation split.
      TEST: Test split.
      PREDICT: Predict split.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"


class LidarChannel(str, Enum):
    """
    Lidar channel in Dataset.

    Attributes:
      LIDAR_TOP: Top lidar channel.
      LIDAR_CONCAT: Concatenated lidar channel.
    """

    LIDAR_TOP = "LIDAR_TOP"
    LIDAR_CONCAT = "LIDAR_CONCAT"


class Box3DFieldIndex(Enum):
    """
    Box 3D field index.

    Attributes:
      X: X coordinate of the center of the box.
      Y: Y coordinate of the center of the box.
      Z: Z coordinate of the center of the box.
      LENGTH: Length of the box.
      WIDTH: Width of the box.
      HEIGHT: Height of the box.
      YAW: Yaw angle of the box.
      VELOCITY_X: Velocity in the X direction.
      VELOCITY_Y: Velocity in the Y direction.
      VELOCITY_Z: Velocity in the Z direction.
    """

    X = 0
    Y = 1
    Z = 2
    LENGTH = 3
    WIDTH = 4
    HEIGHT = 5
    YAW = 6
    VELOCITY_X = 7
    VELOCITY_Y = 8
    VELOCITY_Z = 9
