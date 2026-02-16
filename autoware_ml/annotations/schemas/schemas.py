from dataclasses import dataclass
from typing import NamedTuple 

import polars as pl 


class AnnotationTableColumn(NamedTuple):
    """
    Annotation table column.
    """
    name: str
    dtype: pl.DataType


@dataclass(frozen=True)
class AnnotationTableSchema:
    """
    Annotation table schema.
    """
    SCENARIO_ID = AnnotationTableColumn("scenario_id", pl.String)
    SAMPLE_ID = AnnotationTableColumn("sample_id", pl.String)
    LOCATION = AnnotationTableColumn("location", pl.String)
    VEHICLE_TYPE = AnnotationTableColumn("vehicle_type", pl.String)
    # List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y
    BBOX_3D = AnnotationTableColumn("bbox_3d", pl.List(pl.Array(pl.Float64, shape=(9))))
    # List of 2D bounding boxes with center_x, center_y, width, height
    BBOX_2D = AnnotationTableColumn("bbox_2d", pl.List(pl.Array(pl.Float64, shape=(4))))

    @classmethod
    def to_polars_schema(cls) -> pl.Schema:
        """
        Convert the annotation table schema to a Polars schema.
        """
        return pl.Schema({v.name: v.dtype for k, v in cls.__dict__.items() if not k.startswith("__") and isinstance(v, AnnotationTableColumn)})
