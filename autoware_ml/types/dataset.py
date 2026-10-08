from enum import Enum, StrEnum, auto


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


class PCDFileFormat(StrEnum):
    """
    File format the point clouds of a dataset are read from.

    Attributes:
      AUTO: The t4pack file when the record has a pack location, else the ``.pcd.bin`` file.
      BIN: The ``.pcd.bin`` file of every frame.
      T4PACK: The ``data/<channel>.pack`` file of the channel; every record needs a pack location.
    """

    AUTO = auto()
    BIN = auto()
    T4PACK = auto()
