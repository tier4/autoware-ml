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
    BOXES_3D = DatasetTableColumn("boxes_3d", pl.List(pl.Struct(...)))

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
    boxes_3d: Sequence[Box3DDataModel] | None

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
| `boxes_3d`          | `Sequence[Box3DDataModel] \| None`       | `List(Struct)` | 3D box annotations for the sample/frame            |

### `lidar_frames` struct fields

Each list entry is a `LidarFrameDataModel` covering one keyframe or sweep:

| Field                                   | Polars type           | Description                                                |
| --------------------------------------- | --------------------- | ---------------------------------------------------------- |
| `lidar_frame_id`                        | `String`              | Sample-data token for this frame                           |
| `lidar_keyframe`                        | `Boolean`             | `True` for the main keyframe, `False` for sweeps           |
| `lidar_sensor_id`                       | `String`              | Calibrated-sensor token                                    |
| `lidar_sensor_channel_name`             | `String`              | LiDAR channel name (e.g. `LIDAR_TOP`)                      |
| `lidar_timestamp_seconds`               | `Float64`             | Frame timestamp in seconds                                 |
| `lidar_pointcloud_path`                 | `String`              | Absolute path to the point cloud file                      |
| `lidar_pointcloud_source_path`          | `String`              | Path to per-point metadata (or null)                       |
| `lidar_pointcloud_num_features`         | `Int32`               | Number of features per point (configured on the database)  |
| `lidar_sensor_to_ego_pose_matrix`       | `Array(Float32, 4x4)` | Sensor-to-ego transform                                    |
| `lidar_frame_ego_pose_to_global_matrix` | `Array(Float32, 4x4)` | Ego-to-global transform for this frame                     |
| `lidar_sensor_to_lidar_sweep_matrices`  | `Array(Float32, 4x4)` | Sensor-to-sweep transform                                  |
| `lidar_pointcloud_semantic_mask_path`   | `String`              | LiDAR segmentation mask path (or null)                     |

### `lidar_sources` struct fields

Each list entry is a `LidarSourceDataModel` describing one LiDAR sensor in the scene:

| Field          | Polars type           | Description               |
| -------------- | --------------------- | ------------------------- |
| `channel_name` | `String`              | Sensor channel name       |
| `sensor_token` | `String`              | Sensor token              |
| `translation`  | `Array(Float32, 3)`   | Sensor translation vector |
| `rotation`     | `Array(Float32, 3x3)` | Sensor rotation matrix    |

### `category_mapping` struct fields

| Field              | Polars type     | Description                         |
| ------------------ | --------------- | ----------------------------------- |
| `category_names`   | `List(String)`  | Ordered list of category names      |
| `category_indices` | `List(Int32)`   | Corresponding category index values |

## Nested data models

| Module                    | Schema class                   | Data model                 | Purpose                                                 |
| ------------------------- | ------------------------------ | -------------------------- | ------------------------------------------------------- |
| `lidar_frames.py`         | `LidarFrameDatasetSchema`      | `LidarFrameDataModel`      | Point cloud paths, poses, and sweep transforms          |
| `lidar_sources.py`        | `LidarSourceDatasetSchema`     | `LidarSourceDataModel`     | LiDAR sensor channel name, token, and extrinsics        |
| `category_mapping.py`     | `CategoryMappingDatasetSchema` | `CategoryMappingDataModel` | Parallel lists of category names and indices            |
| `box3d_schemas.py`        | `Box3DDatasetSchema`           | `Box3DDataModel`           | Per-object 3D box parameters and metadata               |
| `frame_basic_metadata.py` | —                              | `FrameBasicMetadata`       | Shared per-frame metadata used during record generation |

### `boxes_3d` struct fields

Each list entry is a `Box3DDataModel` with the following struct fields:

| Field                         | Polars type             | Description                                                                      |
| ----------------------------- | ----------------------- | -------------------------------------------------------------------------------- |
| `box3d_params`                | `Array(Float32, 10)`    | 3D box vector in `Box3DFieldIndex` order: `(x, y, z, l, w, h, yaw, vx, vy, vz)`  |
| `box3d_instance_id`           | `String`                | Instance identifier for the box                                                  |
| `box3d_dataset_label_name`    | `String`                | Original dataset label name                                                      |
| `box3d_label_name`            | `String`                | Normalized training/evaluation label name                                        |
| `box3d_label_index`           | `Int32`                 | Class index of `box3d_label_name`                                                |
| `box3d_num_lidar_pointclouds` | `Int32`                 | Number of LiDAR points in the box                                                |
| `box3d_num_radar_pointclouds` | `Int32`                 | Number of radar points in the box                                                |
| `box3d_valid`                 | `Boolean`               | Validity flag for this annotation                                                |
| `box3d_attributes`            | `List(String)`          | Attribute tags associated with this box                                          |
| `box3d_coordinate`            | `String`                | Coordinate frame identifier for the box representation                           |

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
| `autoware_ml/databases/schemas/box3d_schemas.py`        | 3D box struct schema and data model                           |
| `autoware_ml/databases/schemas/frame_basic_metadata.py` | Shared per-frame metadata model                               |
