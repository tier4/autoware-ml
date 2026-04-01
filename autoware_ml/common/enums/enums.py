from enum import Enum


class SplitType(str, Enum):
    """
    Split type.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICT = "predict"


class LidarChannel(str, Enum):
    """
    Lidar channel in Dataset.
    """

    LIDAR_TOP = "LIDAR_TOP"
    LIDAR_CONCAT = "LIDAR_CONCAT"
