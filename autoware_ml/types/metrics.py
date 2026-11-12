from enum import StrEnum, auto


class MotionModel(StrEnum):
    """
    Reachable-set shape of a collision participant.

    Attributes:
      WHEELED: Agent bound to a road surface, reaching along kinematically feasible arcs.
      OMNIDIRECTIONAL: Agent free to move any direction, reaching an isotropic disc.
      STATIC: Agent that never moves and keeps its own footprint.
    """

    WHEELED = auto()
    OMNIDIRECTIONAL = auto()
    STATIC = auto()


class MatchCost(StrEnum):
    """
    Cost the detection matching pairs a prediction with a ground truth box by.

    Attributes:
      CENTER: BEV center distance in meters, the nuScenes matching.
      CORNER: Mean BEV corner distance in meters, which also charges size and yaw errors.
    """

    CENTER = auto()
    CORNER = auto()
