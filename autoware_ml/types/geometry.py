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


class Box2DFieldIndex(IntEnum):
    """
    Box 2D field index.

    A 2D box is axis-aligned in the pixels of the image it is drawn on: x runs along the image
    width and y down the image height, so the minimum corner is the top-left one.

    Attributes:
      X_MIN: X coordinate of the left edge of the box.
      Y_MIN: Y coordinate of the top edge of the box.
      X_MAX: X coordinate of the right edge of the box.
      Y_MAX: Y coordinate of the bottom edge of the box.
      CENTER_X: X coordinate of the object center, which is not the middle of the box: for a
        box projected from a 3D box, the projection of its gravity center.
      CENTER_Y: Y coordinate of the object center.
    """

    X_MIN = 0
    Y_MIN = 1
    X_MAX = 2
    Y_MAX = 3
    CENTER_X = 4
    CENTER_Y = 5


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
    # Timestamp is stored separately (e.g., BasePoints.timestamp_seconds) and is not a
    # point field index for now.

    Attributes:
      X: X coordinate of the point.
      Y: Y coordinate of the point.
      Z: Z coordinate of the point.
      INTENSITY: Intensity of the point.
      RING: Ring index of the point.
    """

    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4


class PointFeatureName(StrEnum):
    """
    Point feature name.

    Attributes:
      X: X coordinate of the point.
      Y: Y coordinate of the point.
      Z: Z coordinate of the point.
      INTENSITY: Intensity of the point.
      RING: Ring index of the point.
      TIMESTAMP_DIFFERENCE: Timestamp difference with respect to another point data.
    """

    X = "x"
    Y = "y"
    Z = "z"
    INTENSITY = "intensity"
    RING = "ring"
    TIMESTAMP_DIFFERENCE = "timestamp_difference"


class TransformationName(StrEnum):
    """
    Transformation name.

    Attributes:
      ROTATION: Rotation transformation.
      SCALING: Scaling transformation.
      TRANSLATION: Translation transformation.
      HORIZONTAL_FLIP: Horizontal flip transformation.
      VERTICAL_FLIP: Vertical flip transformation.
    """

    ROTATION = "rotation"
    SCALING = "scaling"
    TRANSLATION = "translation"
    HORIZONTAL_FLIP = "horizontal_flip"
    VERTICAL_FLIP = "vertical_flip"


class ImageChannel(StrEnum):
    """
    Supported image channel order.
    """

    RGB = "RGB"
