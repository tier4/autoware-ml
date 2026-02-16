from enum import Enum 

class TaskType(str, Enum):
    """
    Task type.
    """
    CALIBRATION_STATUS = "calibration_status"
    DETECTION3D = "detection3d"
    DETECTION2D = "detection2d"
  
class BoundingBox3D(Enum):
  """
  Index for each 3D bounding box dimension.
  """
  CENTER_X = 0
  CENTER_Y = 1
  CENTER_Z = 2
  LENGTH = 3
  WIDTH = 4
  HEIGHT = 5
  YAW = 6
  VELOCITY_X = 7
  VELOCITY_Y = 8

class BoundingBox2D(Enum):
  """
  Index for each 2D bounding box dimension.
  """
  CENTER_X = 0
  CENTER_Y = 1
  WIDTH = 2
  HEIGHT = 3


class SplitType(str, Enum):
  """
  Split type.
  """
  
  TRAIN = "train"
  VAL = "val"
  TEST = "test"
  PREDICT = "predict"
