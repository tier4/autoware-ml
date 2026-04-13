from enum import Enum


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
