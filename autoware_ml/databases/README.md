# Database Module

The database module defines how Autoware-ML describes annotation databases and generates dataset records from them. It provides a layered architecture: a shared protocol and base class at the top, with dataset-family-specific implementations underneath. Scenario metadata (splits, versions, sampling parameters) is modelled as immutable Pydantic objects so that every database instance is fully hashable and cacheable.

The Hydra-based entrypoint in `scripts/generate_dataset.py` composes a YAML config that selects the concrete database class and its scenario groups, instantiates the database, and triggers parallel record generation. The output is a stream of `DatasetRecord` rows that can be persisted as Parquet for downstream training or evaluation pipelines.

## Module relationships

| Module                        | Role                                                                     | Depends on                       |
| ----------------------------- | ------------------------------------------------------------------------ | -------------------------------- |
| `schemas.py`                  | Defines `DatasetRecord` and `DatasetTableSchema` (output row shape)      | `polars`                         |
| `scenarios.py`                | Defines `ScenarioData`, `DatabaseVersion`, and abstract `Scenarios` base | _(none)_                         |
| `database_interface.py`       | `DatabaseInterface` protocol all databases must satisfy                  | `scenarios`, `schemas`           |
| `base_database.py`            | `BaseDatabase` shared implementation of `DatabaseInterface`              | `scenarios`, `schemas`, `polars` |
| `scripts/generate_dataset.py` | Hydra entrypoint that instantiates a `DatabaseInterface` from config     | `database_interface`             |

To add a new dataset family, extend `Scenarios` with format-specific YAML parsing, extend `BaseDatabase` with record generation logic, and register the new class in a Hydra config. See `t4datasets/` for a concrete example.

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

    class scenarios {
        DatabaseVersion
        ScenarioData
        Scenarios
    }

    class schemas {
        DatasetRecord
        DatasetTableSchema
        DatasetTableColumn
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

    class Output {
        <<result>>
        Iterable~DatasetRecord~
    }

    generate_dataset --> DatabaseInterface : instantiates via Hydra

    DatabaseInterface ..> scenarios : uses Scenarios, ScenarioData
    DatabaseInterface ..> schemas : uses DatasetRecord

    BaseDatabase ..|> DatabaseInterface : satisfies
    BaseDatabase --> scenarios : uses Scenarios, ScenarioData
    BaseDatabase --> schemas : uses DatasetRecord, DatasetTableSchema
    BaseDatabase --> polars : DataFrame, Schema

    ConcreteDatabase --|> BaseDatabase : extends
    ConcreteDatabase --> Output : process_scenario_records()

    Output --> schemas : list of DatasetRecord

    schemas ..> polars : uses pl.DataType, pl.Schema
```
