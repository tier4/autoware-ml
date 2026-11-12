# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Enumerations of dataset splits, sweep directions and point cloud file formats."""

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
