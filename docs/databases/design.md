---
icon: lucide/database
---

# Database Design

The database module provides a layered architecture for describing annotation databases and generating dataset records. A shared protocol and base class sit at the top, with dataset-family-specific implementations underneath. Scenario metadata (splits, versions, sampling parameters) is modelled as immutable Pydantic objects so that every database instance is fully hashable and cacheable.

## Architecture Overview

```mermaid
classDiagram
    direction TB

    class generate_dataset {
        <<Hydra entrypoint>>
        build_database()
        main()
    }

    class DatabaseInterface {
        <<Protocol>>
        version
        scenarios
        cache_path
        load_scenario_records()
        process_scenario_records()
    }

    class BaseDatabase {
        get_polars_schema()
        get_main_database_scenario_data()
        get_unique_scenario_data()
        process_scenario_records()
    }

    class scenarios {
        DatasetParams
        ScenarioData
        Scenarios
    }

    class schemas {
        <<package>>
        DatasetRecord
        DatasetTableSchema
        DataModelInterface
        LidarFrameDataModel
        LidarSourceDataModel
        CategoryMappingDataModel
        Box3DDataModel
        Box3DDatasetSchema
    }

    class polars {
        <<external>>
        DataFrame
        Schema
    }

    class ConcreteDatabase {
        <<dataset-specific>>
        process_scenario_records()
    }

    class TrainingInference {
        <<downstream>>
        train()
        evaluate()
        predict()
    }

    generate_dataset --> DatabaseInterface : instantiates via Hydra

    DatabaseInterface ..> scenarios : uses Scenarios, ScenarioData
    DatabaseInterface --> schemas : process_scenario_records()

    BaseDatabase ..|> DatabaseInterface : satisfies
    ConcreteDatabase --|> BaseDatabase : extends

    schemas --> TrainingInference : Sequence[DatasetRecord] consumed by

    schemas ..> polars : uses pl.DataType, pl.Schema
```

## Core Components

### DatabaseInterface

`DatabaseInterface` is the protocol that every database implementation must satisfy. It defines the contract for version metadata, scenario access, and record generation:

```python
class DatabaseInterface(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def scenarios(self) -> MappingProxyType[str, Scenarios]: ...

    def get_unique_scenario_data(self) -> MappingProxyType[str, ScenarioData]: ...
    def load_scenario_records(self) -> Sequence[DatasetRecord]: ...
    def process_scenario_records(self) -> None: ...
```

All concrete databases are accessed through this protocol, ensuring downstream code (training, evaluation) never depends on a specific dataset format.

### BaseDatabase

`BaseDatabase` provides the shared implementation of `DatabaseInterface`. It handles initialization from version and paths, caching directory creation, Polars schema retrieval, and deduplicating scenario data across groups:

```python
class BaseDatabase:
    def __init__(
        self,
        version: str,
        root_path: str,
        cache_path: str,
        cache_file_prefix_name: str,
        num_workers: int,
        taxonomy: DatabaseTaxonomy,
        box3d_pipelines: Sequence[Box3DPipeline],
        lidar_intensity_scale: float,
        lidar_pointcloud_num_features: int,
    ) -> None:
        ...

    def get_polars_schema(self) -> pl.Schema: ...
    def get_unique_scenario_data(self) -> MappingProxyType[str, ScenarioData]: ...
    def process_scenario_records(self) -> None:
        raise NotImplementedError("Subclasses must implement process_scenario_records method!")
```

To add a new dataset family, subclass `BaseDatabase` and implement `process_scenario_records()`. See [T4Dataset](t4dataset.md) for a concrete example.

### Scenarios

The `scenarios` module models scenario metadata as immutable Pydantic objects. `DatasetParams` captures per-dataset preprocessing parameters, `ScenarioData` uniquely identifies a single scenario with its version and sampling settings, and `Scenarios` is the abstract base that concrete implementations extend to parse scenario configs based on a dataset:

```python
class DatasetParams(BaseModel):
    dataset_name: str
    sample_steps: int

class ScenarioData(BaseModel):
    dataset_params: DatasetParams
    scenario_id: str
    scenario_version: str
    vehicle_type: str | None = None
    location: str | None = None

class Scenarios(BaseModel):
    scenario_root_path: Path
    dataset_params: Sequence[DatasetParams]
    scenario_data: Mapping[SplitType, Sequence[ScenarioData]] | None = None

    @model_validator(mode="after")
    def build_scenarios(self) -> Scenarios:
        raise NotImplementedError("Subclasses must implement build_scenarios!")
```

### Schema

`process_scenario_records()` — Process scenarios/samples from a database to a parquet file and save it. `BaseDatabase.get_polars_schema()` delegates to `DatasetTableSchema` so records can be serialized to Parquet via `DatasetRecord.to_dictionary()`.

The schema is defined in the `autoware_ml/databases/schemas/` package and covers basic frame metadata, nested LiDAR structs, and annotation fields such as category mapping and 3D boxes. The 3D box payload is modeled by `Box3DDataModel` with its struct layout defined in `Box3DDatasetSchema`, and is stored in the top-level `boxes_3d` list column. See [Dataset Schema](schemas.md) for the full column layout, nested data models, and extension guide.

### Packed LiDAR frames

A scene may keep the frames of a LiDAR channel, `data/<channel>/*.pcd.bin`, as one
`data/<channel>.pack` file (t4pack v1: every frame compressed on its own with zstd over its
columns, plus a frame index). Packs are written by the data processing tools; autoware-ml only
reads them.

When a scene has a pack, record generation reads its index once and stores the location of every
frame and sweep in `lidar_pointcloud_t4pack_frame`. `lidar_pointcloud_path` keeps the path of the
loose file, so the pack is found next to it. `LoadPointsFromFile` and
`LoadMultiSweepPointsFromFile` choose the file format with `pcd_file_format`, a
`PCDFileFormat` from `autoware_ml.types.dataset`:

| `pcd_file_format` | Reads                                                                         |
| ----------------- | ----------------------------------------------------------------------------- |
| `auto` (default)  | the pack when the record has a pack location, else the `.pcd.bin` file        |
| `bin`             | the `.pcd.bin` file of every frame                                            |
| `t4pack`          | the pack of every frame; fails on a record without a pack location            |

Both formats return the same points. A pack is read one byte range per frame. In a Hydra config:

```yaml
- _target_: autoware_ml.transforms.multi_task.point_cloud.loading.LoadPointsFromFile
  pcd_file_format:
    _target_: autoware_ml.types.dataset.PCDFileFormat
    _args_: [t4pack]
```

See `autoware_ml/databases/t4pack/t4pack.py`.

### Dataset Generation (Hydra Entrypoint)

The `generate_dataset.py` script is the Hydra-based entrypoint that wires everything together. It reads a YAML config, instantiates the configured database class, and triggers record generation:

```python
@hydra.main(version_base=None, config_path=_CONFIG_PATH)
def main(cfg: DictConfig):
    database: DatabaseInterface = instantiate(cfg.database)
    database.process_scenario_records()
```

To run dataset generation:

```bash
python3 autoware_ml/scripts/generate_dataset.py \
    --config-name default_t4dataset_generator \
    working_dir=<working_dir> \
    data_root_path=<data_root_path> \
    database.num_workers=32
```

Configuration is done through YAML files under `autoware_ml/configs/generators/`. Override any parameter from the command line using Hydra syntax. See [Configuration Guide](../user-guide/configuration.md) for full details.

## Extending the Database

| Extension Point      | How                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| New dataset family   | Subclass `BaseDatabase`, implement `process_scenario_records()`, register in a Hydra config                  |
| New scenario format  | Subclass `Scenarios`, implement `build_scenarios()` to parse format-specific YAML                            |
| New schema columns   | See [Dataset Schema](schemas.md)                                                                             |

## Implementation

| Path                                          | Description                                           |
| --------------------------------------------- | ----------------------------------------------------- |
| `autoware_ml/databases/schemas/`              | Dataset schema package — see [schemas.md](schemas.md) |
| `autoware_ml/databases/scenarios.py`          | `ScenarioData`, `DatasetParams`, `Scenarios`          |
| `autoware_ml/databases/database_interface.py` | `DatabaseInterface` protocol                          |
| `autoware_ml/databases/base_database.py`      | Shared `BaseDatabase` implementation                  |
| `autoware_ml/scripts/generate_dataset.py`     | Hydra entrypoint for dataset generation               |
| `autoware_ml/configs/generators/`             | YAML configs for dataset generation                   |
