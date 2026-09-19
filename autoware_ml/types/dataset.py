from enum import Enum, StrEnum


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


class SweepDirection(StrEnum):
    """
    Side of a sample to collect lidar sweeps from. The lidar frames of a scene form a chain,
    and each direction names the link to follow.

    Attributes:
      PAST: Frames captured before the sample, reached through the prev link.
      FUTURE: Frames captured after the sample, reached through the next link.
    """

    PAST = "past"
    FUTURE = "future"

    @property
    def link(self) -> str:
        """
        Name of the sample data link to follow in this direction.

        Returns:
          str: Link name.
        """

        return "prev" if self is SweepDirection.PAST else "next"
