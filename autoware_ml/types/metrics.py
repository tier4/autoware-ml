from enum import StrEnum, auto


class AgentKind(StrEnum):
    """
    Reachable-set shape of a collision participant.

    Attributes:
      WHEELED: Agent bound to a road surface, reaching along kinematically feasible arcs.
      VRU: Agent free to move any direction, reaching an isotropic disc.
      STATIC: Agent that never moves and keeps its own footprint.
    """

    WHEELED = auto()
    VRU = auto()
    STATIC = auto()
