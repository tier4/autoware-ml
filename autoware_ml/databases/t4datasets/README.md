# T4 Datasets

This sub-module implements the database layer for the **T4** annotation format, built on top of the abstract base classes in `databases/`.

## Module relationships

| Module                   | Role                                                                                            | Depends on                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `t4scenarios.py`         | `T4Scenarios` extends `Scenarios`: reads scenario YAML files and builds per-split scenario data | `scenarios`                                                                             |
| `t4records_generator.py` | `T4RecordsGenerator` reads T4 annotations via `t4-devkit` and emits `DatasetRecord`             | `scenarios`, `schemas`, `t4-devkit`                                                     |
| `t4database.py`          | `T4Database` extends `BaseDatabase`: orchestrates parallel record generation across scenarios   | `base_database`, `t4scenarios`, `t4records_generator`, `scenarios`, `schemas`, `polars` |

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

    class scenarios {
        <<databases>>
        Scenarios
        ScenarioData
        DatabaseVersion
    }

    class schemas {
        <<databases>>
        DatasetRecord
        DatasetTableSchema
    }

    class BaseDatabase {
        <<databases>>
        get_polars_schema()
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

    T4Scenarios --|> scenarios : extends Scenarios

    T4Database --|> BaseDatabase : extends
    T4Database --> T4Scenarios : scenario groups
    T4Database --> T4RecordsGenerator : creates per scenario
    T4Database --> polars : writes Parquet via DataFrame

    T4RecordsGenerator --> scenarios : reads ScenarioData
    T4RecordsGenerator --> schemas : emits DatasetRecord
    T4RecordsGenerator --> t4_devkit : reads T4 annotations
```
