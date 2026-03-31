# Database Module

The database module defines how Autoware-ML describes annotation databases and generates dataset records from them. It provides a layered architecture: a shared protocol and base class at the top, with dataset-family-specific implementations (currently T4) underneath. Scenario metadata (splits, versions, sampling parameters) is modelled as immutable Pydantic objects so that every database instance is fully hashable and cacheable.

The Hydra-based entrypoint in `scripts/generate_dataset.py` composes a YAML config that selects the concrete database class and its scenario groups, instantiates the database, and triggers parallel record generation. The output is a stream of `DatasetRecord` rows that can be persisted as Parquet for downstream training or evaluation pipelines.

## Module relationships

| Module                              | Role                                                                          | Depends on                                                                              |
| ----------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `schemas.py`                        | Defines `DatasetRecord` and `DatasetTableSchema` (output row shape)           | `polars`                                                                                |
| `scenarios.py`                      | Defines `ScenarioData`, `DatabaseVersion`, and abstract `Scenarios` base      | _(none)_                                                                                |
| `database_interface.py`             | `DatabaseInterface` protocol all databases must satisfy                       | `scenarios`, `schemas`                                                                  |
| `base_database.py`                  | `BaseDatabase` shared implementation of `DatabaseInterface`                   | `scenarios`, `schemas`, `polars`                                                        |
| `t4datasets/t4scenarios.py`         | `T4Scenarios` extends `Scenarios` — reads scenario YAML and builds split data | `scenarios`                                                                             |
| `t4datasets/t4records_generator.py` | `T4RecordsGenerator` reads T4 annotations and emits `DatasetRecord`           | `scenarios`, `schemas`, `t4-devkit`                                                     |
| `t4datasets/t4database.py`          | `T4Database` extends `BaseDatabase` — orchestrates parallel record generation | `base_database`, `t4scenarios`, `t4records_generator`, `scenarios`, `schemas`, `polars` |
| `scripts/generate_dataset.py`       | Hydra entrypoint that instantiates a `DatabaseInterface` from config          | `database_interface`                                                                    |

```mermaid
classDiagram
    direction TB

    class polars {
        <<external>>
        DataFrame
        Schema
    }

    class t4_devkit {
        <<external>>
        Tier4
        Sample
        SampleData
        CalibratedSensor
    }

    class schemas {
        DatasetRecord
        DatasetTableSchema
        DatasetTableColumn
    }

    class scenarios {
        DatabaseVersion
        ScenarioData
        Scenarios
    }

    class DatabaseInterface {
        <<Protocol>>
        database_version
        database_root_path
        scenario_root_path
        scenarios
        cache_path
        load_scenario_records()
    }

    class BaseDatabase {
        get_polars_schema()
        get_main_database_scenario_data()
        get_unique_scenario_data()
        process_scenario_records()
    }

    class T4Scenarios {
        build_scenarios()
        _build_scenario_data()
        _build_scenario_splits()
    }

    class T4RecordsGenerator {
        generate_dataset_records()
        extract_t4_sample_record()
    }

    class T4Database {
        process_scenario_records()
        _multi_process_scenario_records()
    }

    class generate_dataset {
        <<Hydra entrypoint>>
        build_database()
        main()
    }

    schemas ..> polars : uses pl.DataType, pl.Schema

    DatabaseInterface ..> scenarios : uses Scenarios, ScenarioData
    DatabaseInterface ..> schemas : uses DatasetRecord

    BaseDatabase ..|> DatabaseInterface : satisfies
    BaseDatabase --> scenarios : uses Scenarios, ScenarioData
    BaseDatabase --> schemas : uses DatasetRecord, DatasetTableSchema
    BaseDatabase --> polars : DataFrame, Schema

    T4Scenarios --|> scenarios : extends Scenarios

    T4Database --|> BaseDatabase : extends
    T4Database --> T4Scenarios : scenario groups
    T4Database --> T4RecordsGenerator : creates per scenario
    T4Database --> polars : writes Parquet via DataFrame

    T4RecordsGenerator --> scenarios : reads ScenarioData
    T4RecordsGenerator --> schemas : emits DatasetRecord
    T4RecordsGenerator --> t4_devkit : reads T4 annotations

    generate_dataset --> DatabaseInterface : instantiates via Hydra
```
