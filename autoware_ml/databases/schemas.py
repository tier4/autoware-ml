from dataclasses import dataclass
from typing import NamedTuple

import polars as pl
from pydantic import BaseModel, ConfigDict


class DatasetTableColumn(NamedTuple):
    """
    Annotation table column.
    """

    name: str
    dtype: pl.DataType


@dataclass(frozen=True)
class DatasetTableSchema:
    """
    Annotation table schema.
    """

    SCENARIO_ID = DatasetTableColumn("scenario_id", pl.String)
    SAMPLE_ID = DatasetTableColumn("sample_id", pl.String)
    SAMPLE_INDEX = DatasetTableColumn("sample_index", pl.Int32)
    LOCATION = DatasetTableColumn("location", pl.String)
    VEHICLE_TYPE = DatasetTableColumn("vehicle_type", pl.String)
    # List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y
    # BBOX_3D = DatasetTableColumn("bbox_3d", pl.List(pl.Array(pl.Float64, shape=(9))))

    @classmethod
    def to_polars_schema(cls) -> pl.Schema:
        """
        Convert the dataset table schema to a Polars schema.
        """
        return pl.Schema(
            {
                v.name: v.dtype
                for k, v in cls.__dict__.items()
                if not k.startswith("__") and isinstance(v, DatasetTableColumn)
            }
        )


class DatasetRecord(BaseModel):
    """
    Data class to save a record for each column in the annotation table.
    :param scenario_id: Scenario id.
    :param sample_id: Sample id.
    :param location: Location of the vehicle.
    :param vehicle_type: Type of the vehicle.
    :param bbox_3d: List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y.
    :param bbox_2d: List of 2D bounding boxes with center_x, center_y, width, height.
    """

    # Set model config to frozen
    model_config = ConfigDict(frozen=True, strict=True)

    scenario_id: str
    sample_id: str
    sample_index: int
    location: str
    vehicle_type: str
    # TODO (KokSeang): Add more annotation fields here
    # bbox_3d: Sequence[npt.NDArray[np.float64]]
