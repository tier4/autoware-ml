from enum import IntEnum, StrEnum


class Box3DFieldIndex(IntEnum):
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


class Box3DCenterCoordinateType(StrEnum):
    """
    Box 3D center coordinate type.

    Attributes:
      GravityCenter: The center of the box is at the gravity center of the box, where z coordinate
          is at the center of the bbox
    """

    GRAVITY_CENTER = "gravity_center"


class PointFieldIndex(IntEnum):
    """
    Point field index.

    Attributes:
      X: X coordinate of the point.
      Y: Y coordinate of the point.
      Z: Z coordinate of the point.
      INTENSITY: Intensity of the point.
      RING: Ring index of the point.
      TIME: Timestamp of the point.
    """

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4
    TIME = 5


class PointFeatureName(StrEnum):
    """
    Point feature name.

    Attributes:
      X: X coordinate of the point.
      Y: Y coordinate of the point.
      Z: Z coordinate of the point.
      INTENSITY: Intensity of the point.
      RING: Ring index of the point.
      TIME: Timestamp of the point.
    """

    X = "x"
    Y = "y"
    Z = "z"
    INTENSITY = "intensity"
    RING = "ring"
    TIME = "time"
