---
icon: lucide/table
---

# Dataset Schema

The output schema lives in the `autoware_ml/databases/schemas/` package. It is split into a top-level table definition and reusable nested data models that can be shared across dataset families. Every database implementation emits `Sequence[DatasetRecord]` from `process_scenario_records()` and persists rows as a Polars `DataFrame` (typically Parquet) using `DatasetTableSchema`.

## Base building blocks (`base_schemas.py`)

- **`DatasetTableColumn`** - a `NamedTuple` pairing a column name with a Polars data type.
- **`BaseFieldSchema`** - base class for nested struct schemas. Subclasses declare `DatasetTableColumn` attributes and expose `to_polars_field_schema()` to produce `pl.Field` definitions for struct columns.
- **`DataModelInterface`** - abstract interface requiring `to_dictionary()` and `load_from_dictionary()` so every Pydantic data model can round-trip through a Polars `DataFrame`.

## Top-level table (`dataset_schemas.py`)

- **`DatasetTableSchema`** - a frozen dataclass whose class-level attributes are `DatasetTableColumn` entries. Call `DatasetTableSchema.to_polars_schema()` to get a `pl.Schema` for constructing or validating a Polars `DataFrame`.
- **`DatasetRecord`** - a frozen Pydantic model (implementing `DataModelInterface`) representing a single row. One record is emitted per sample/frame by `process_scenario_records()`.

```python
class DatasetTableSchema:
    # Basic metadata
    SCENARIO_ID = DatasetTableColumn("scenario_id", pl.String)
    SAMPLE_ID = DatasetTableColumn("sample_id", pl.String)
    SAMPLE_INDEX = DatasetTableColumn("sample_index", pl.Int32)
    TIMESTAMP_SECONDS = DatasetTableColumn("timestamp_seconds", pl.Float64)
    LOCATION = DatasetTableColumn("location", pl.String)
    VEHICLE_TYPE = DatasetTableColumn("vehicle_type", pl.String)
    SCENARIO_NAME = DatasetTableColumn("scenario_name", pl.String)

    # Nested sensor data columns
    LIDAR_FRAMES = DatasetTableColumn("lidar_frames", pl.List(pl.Struct(...)))
    LIDAR_SOURCES = DatasetTableColumn("lidar_sources", pl.List(pl.Struct(...)))
    # Annotation fields
    CATEGORY_MAPPING = DatasetTableColumn("category_mapping", pl.Struct(...))

    @classmethod
    def to_polars_schema(cls) -> pl.Schema: ...

class DatasetRecord(BaseModel, DataModelInterface):
    scenario_id: str
    sample_id: str
    sample_index: int
    timestamp_seconds: float
    location: str | None
    vehicle_type: str | None
    scenario_name: str
    lidar_frames: Sequence[LidarFrameDataModel]
    lidar_sources: Sequence[LidarSourceDataModel] | None
    category_mapping: CategoryMappingDataModel | None

    def to_dictionary(self) -> Mapping[str, Any]: ...
    @classmethod
    def load_from_dictionary(cls, data_model: Mapping[str, Any]) -> DatasetRecord: ...
```

## Top-level columns

| Column              | Python type                              | Polars type    | Description                                        |
| ------------------- | ---------------------------------------- | -------------- | -------------------------------------------------- |
| `scenario_id`       | `str`                                    | `String`       | Unique identifier of the driving scenario          |
| `sample_id`         | `str`                                    | `String`       | Unique identifier of the individual sample/frame   |
| `sample_index`      | `int`                                    | `Int32`        | Zero-based index of the sample within the scenario |
| `timestamp_seconds` | `float`                                  | `Float64`      | Sample timestamp in seconds                        |
| `location`          | `str \| None`                            | `String`       | Geographic location where the data was captured    |
| `vehicle_type`      | `str \| None`                            | `String`       | Type of vehicle used for data collection           |
| `scenario_name`     | `str`                                    | `String`       | Human-readable name of the scenario scene          |
| `lidar_frames`      | `Sequence[LidarFrameDataModel]`          | `List(Struct)` | Keyframe and sweep LiDAR frame metadata per sample |
| `lidar_sources`     | `Sequence[LidarSourceDataModel] \| None` | `List(Struct)` | Per-sensor calibration metadata for LiDAR sources  |
| `category_mapping`  | `CategoryMappingDataModel \| None`       | `Struct`       | Mapping between category names and indices         |

## Nested data models

| Module                    | Schema class                   | Data model                 | Purpose                                                 |
| ------------------------- | ------------------------------ | -------------------------- | ------------------------------------------------------- |
| `lidar_frames.py`         | `LidarFrameDatasetSchema`      | `LidarFrameDataModel`      | Point cloud paths, poses, and sweep transforms          |
| `lidar_sources.py`        | `LidarSourceDatasetSchema`     | `LidarSourceDataModel`     | LiDAR sensor channel name, token, and extrinsics        |
| `category_mapping.py`     | `CategoryMappingDatasetSchema` | `CategoryMappingDataModel` | Parallel lists of category names and indices            |
| `frame_basic_metadata.py` | —                              | `FrameBasicMetadata`       | Shared per-frame metadata used during record generation |

Each nested module follows the same pattern: a `*DatasetSchema` class defines the Polars struct layout, and a matching `*DataModel` Pydantic class implements `DataModelInterface` for serialization. `DatasetRecord.to_dictionary()` delegates to these nested models when writing Parquet; `DatasetRecord.load_from_dictionary()` reconstructs them when reading back.

`DatasetTableSchema`, `DatasetRecord`, and every nested schema/data-model pair are kept in sync. When adding new columns/fields (e.g. 3D bounding boxes), add entries to the relevant `*DatasetSchema` and `*DataModel`, then wire the new column into `DatasetTableSchema` and `DatasetRecord`.

## Extending the schema

| Extension            | How                                                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| New top-level column | Add a `*DatasetSchema`/`*DataModel` pair (or extend an existing one), then wire the column into `DatasetTableSchema` and `DatasetRecord` |
| New struct field     | Add matching entries to the relevant `*DatasetSchema` and `*DataModel` classes                                                           |

See [T4Dataset](t4dataset.md) for a concrete example of how these schemas are populated from T4 annotations.

## Implementation

| Path                                                    | Description                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------- |
| `autoware_ml/databases/schemas/base_schemas.py`         | `DatasetTableColumn`, `BaseFieldSchema`, `DataModelInterface` |
| `autoware_ml/databases/schemas/dataset_schemas.py`      | `DatasetRecord` and `DatasetTableSchema`                      |
| `autoware_ml/databases/schemas/lidar_frames.py`         | LiDAR frame struct schema and data model                      |
| `autoware_ml/databases/schemas/lidar_sources.py`        | LiDAR source struct schema and data model                     |
| `autoware_ml/databases/schemas/category_mapping.py`     | Category mapping struct schema and data model                 |
| `autoware_ml/databases/schemas/frame_basic_metadata.py` | Shared per-frame metadata model                               |
