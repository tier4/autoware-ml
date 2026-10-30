# Integrating Argoverse 2 Motion Forecasting into autoware-ml

Plan for adding the Argoverse 2 Motion Forecasting dataset (AV2 MF) to autoware-ml at the
**database + dataset layer**: raw AV2 scenarios are converted into the shared record table, a
dataset serves them, and the pipeline ends in a collated `MotionForecastingGTBatch`. No model,
loss or metric is part of this plan.

Status: approved plan, not implemented. Nothing in the repository or the pixi environment has
been changed for it yet (a feature branch `feat/av2-motion-forecasting-database` exists without
commits).

## 1. Context

autoware-ml is a perception framework today. Every database (T4dataset, nuScenes) converts
sensor annotations into one shared Parquet record table (`DatasetTableSchema`), and every
dataset and batch dataclass is lidar / camera / box / mask shaped. AV2 MF is a trajectory
prediction dataset:

- ~200k train / 25k val / 25k test scenarios, one directory per scenario:
  `<split>/<scenario_id>/scenario_<scenario_id>.parquet` and
  `<split>/<scenario_id>/log_map_archive_<scenario_id>.json`.
- Each scenario is an 11 s window at 10 Hz: 110 timesteps, steps 0-49 observed, 50-109 future.
  The test split only ships the 50 observed steps. One focal track per scenario
  (`focal_track_id`), a `city`, and `start_timestamp` / `end_timestamp` in ns.
- Parquet rows: `observed`, `track_id`, `object_type` (vehicle, pedestrian, motorcyclist,
  cyclist, bus, static, background, construction, riderless_bicycle, unknown),
  `object_category` (0 TRACK_FRAGMENT, 1 UNSCORED_TRACK, 2 SCORED_TRACK, 3 FOCAL_TRACK),
  `timestep`, `position_x/y`, `heading`, `velocity_x/y`.
- Map JSON: `drivable_areas`, `lane_segments` (id, is_intersection, lane_type, left / right
  boundary points xyz, mark types, predecessors, successors, neighbors), `pedestrian_crossings`
  (edge1, edge2).
- No sensor data.

Decisions taken with the user:

| Topic | Decision |
| --- | --- |
| Scope | Database + dataset layer only, ending in a collated batch dataclass |
| Raw file reader | Official `av2` devkit (`av2==0.3.6`) |
| Representation | Dataset-agnostic (generic agent tracks + polyline map), so T4dataset (lanelet2 maps, T4 annotation tracks) can feed the same pipeline later |
| Data | Not downloaded yet; the plan includes the download and uses synthetic fixtures for tests |
| Cache reset | Adding a nullable column to the shared schema regenerates every existing cache once; accepted |

## 2. Findings from the codebase

- There is no "convert" CLI. `DataModule.prepare_data()`
  (`autoware_ml/datamodule/data_module.py:229`) calls `database.process_scenario_records()`,
  which writes `<cache_path>/<prefix>_<database_hash>.parquet`. The hash is a SHA256 over
  `hash_fields` plus the Polars schema string (`autoware_ml/databases/base_database.py`).
- `BaseDatabase.__init__` requires `lidar_intensity_scale > 0`,
  `lidar_pointcloud_num_features >= 3`, `box3d_pipelines`, and a `DatabaseTaxonomy`.
  `DatabaseTaxonomy(detection3d, segmentation3d)` requires both heads
  (`autoware_ml/databases/taxonomy.py`). Both have to be relaxed. Head accessors are used only
  on T4 / nuScenes paths (`t4records_generator.py:346`, `nuscenes_records_generator.py:277`,
  `visualization/preview.py:268-272`,
  `configs/datamodule/frame_sampling/default_frame_sampling.yaml`).
- `DatasetRecord.lidar_frames` is required but an empty list validates; every other annotation
  column is nullable (`autoware_ml/databases/schemas/dataset_schemas.py`). The null-struct guard
  for `category_mapping` (lines 252-260) is the pattern for a new nullable struct column.
- `ModelGTSample` / `ModelGTBatch` are NamedTuples with task-specific fields. New fields can be
  appended with `None` defaults, plus one `collate_*` staticmethod and a `to_device` branch
  (`autoware_ml/dataclasses/batch/sample_batch.py`; template
  `autoware_ml/dataclasses/batch/segmentation3d.py`).
- `T4Dataset` in `autoware_ml/datamodule/t4dataset/dataset.py` indexes `LIDAR_FRAMES[0]`; do not
  subclass it. Subclass `BaseDataset` (`autoware_ml/datamodule/base_dataset.py`: abstract
  `get_data_sample`, `bind` / `_serve_records`, `collate_fn`). `DatasetSource` requires
  `dataset.database_root_path == database.root_path`.
- `ScenarioSplitter` filters by `scenario_id` only. `NuScenesScenarios`
  (`autoware_ml/databases/nuscenes/nuscenes_scenarios.py`) derives splits from the corpus
  instead of YAML lists, which is the pattern for AV2's train / val / test directories.
- Transforms: `BaseTransform(_required_keys)` + `TransformsCompose`
  (`autoware_ml/transforms/base.py`). The loading pattern is `LoadPointsFromFile`: file
  references in the sample become loaded data in the sample. Format-specific readers live under
  `databases/` (`transforms/point_cloud/loading.py` imports `databases/t4pack`).
- `av2==0.3.6` (PyPI, cp311 manylinux wheel) declares unpinned dependencies: `av`, `click`,
  `joblib`, `kornia`, `matplotlib`, `nox`, `numba`, `numpy`, `opencv-python`, `pandas`, `polars`,
  `pyarrow`, `pyproj`, `rich`, `scipy`, `torch`, `tqdm`, `universal-pathlib`. The repo pins
  `opencv-python-headless`; resolving `opencv-python` next to it installs a second `cv2`.
  Verified in the pixi env (then uninstalled again): `pip install --no-deps av2==0.3.6` works,
  `pandas`, `pyarrow`, `joblib`, `click`, `cv2` are already present, `universal-pathlib`
  (`upath`) is missing and is imported by `av2.utils.typing`, so it is required; `numba`,
  `kornia`, `av`, `pyproj` are missing and still need to be checked against the motion
  forecasting and map modules once `upath` is installed.
- 250k scenarios: `Scenarios.hash_repr` and `__str__` list every scenario (~45 MB string) and
  `database_hash` is recomputed on every access. AV2 needs a digest and memoization.

## 3. Generic representation and the AV2 mapping

### 3.1 Enums (`autoware_ml/types/motion.py`, new)

| Enum | Members | Meaning |
| --- | --- | --- |
| `TracksFormat` | `AV2_SCENARIO_PARQUET` (later `T4_ANNOTATION_TRACKS`) | Source format of the agent tracks a record points at |
| `VectorMapFormat` | `AV2_VECTOR_MAP_JSON` (later `LANELET2_OSM`) | Source format of the map a record points at |
| `AgentRole` | `TRACK_FRAGMENT=0`, `UNSCORED=1`, `SCORED=2`, `FOCAL=3` | Role of an agent, equal to AV2 `object_category` |
| `PolylineType` | `LANE_CENTERLINE`, `LANE_LEFT_BOUNDARY`, `LANE_RIGHT_BOUNDARY`, `CROSSWALK_EDGE`, `DRIVABLE_AREA_BOUNDARY` | Kind of a map polyline |

Add `MOTION_FORECASTING = "motion_forecasting"` to `autoware_ml/types/tasks.py::TaskType`.

### 3.2 Agent types

Agent types go through the existing `LabelVocabulary` -> `LabelTaxonomy` machinery with a new
`MotionForecastingTaxonomy(LabelTaxonomy)`:

| AV2 `object_type` (raw) | fine name (vocabulary) | default class (coarse level) |
| --- | --- | --- |
| vehicle | vehicle | vehicle |
| bus | bus | vehicle |
| pedestrian | pedestrian | pedestrian |
| cyclist | cyclist | cyclist |
| motorcyclist | motorcyclist | cyclist |
| riderless_bicycle | riderless_bicycle | null (ignore_index) |
| static, background, construction, unknown | null | - |

`class_groups` for the default level: `grouped_vehicle: [vehicle]`,
`grouped_two_wheeler: [cyclist]`, `grouped_pedestrian: [pedestrian]`. Agents whose class
resolves to `ignore_index` stay in the tensors as context with `agent_types == ignore_index`;
dropping them is a transform decision, not a loader one. An unlisted raw name raises, which is
the repo's mechanism for discovering a new category.

### 3.3 Map elements

| AV2 element | polylines | `PolylineType` | attributes |
| --- | --- | --- | --- |
| `LaneSegment` | 3 per lane: centerline via `av2.geometry.interpolate.compute_midpoint_line(left, right)`, left boundary, right boundary | 0, 1, 2 | `element_id = str(lane.id)`, `is_intersection` |
| `PedestrianCrossing` | 2 per crossing (`edge1`, `edge2`) | 3 | `element_id = str(crossing.id)` |
| `DrivableArea` | 1 per area (exterior boundary) | 4 | `element_id = str(area.id)` |

Every polyline is resampled to `L` points with `av2.geometry.interpolate.interp_arc`
(default `L = 20`). Lane connectivity (predecessors, successors, neighbors) is out of scope and
can be exposed later as an edge list.

### 3.4 Time axis

- `num_steps = len(timestamps_ns)` (110 for train / val, 50 for test).
- `current_step_index = 49` from the constant `OBSERVED_STEPS = 50`, which is part of the
  database hash.
- `step_seconds` from the timestamp differences (0.1).
- Record `timestamp_seconds = timestamps_ns[current_step_index] * 1e-9`.

### 3.5 Records

One record per scenario. Tracks are **not** stored in the record table: 200k scenarios x ~70
tracks x 110 steps is tens of GB, and the generators hold every record in memory before writing.
The record stores only lightweight file references, mirroring how lidar frames store paths.

## 4. Implementation steps

Each step leaves `pytest autoware_ml/ -v` green. Every new Python file carries the Apache-2.0
TIER IV header; every new config directory gets an empty `__init__.py`.

### Step 1: dependency, enums, dictionary

- `pyproject.toml`: add `av2==0.3.6`. Preferred: a pixi `dependency-overrides` entry under
  `[tool.pixi.pypi-options]` that keeps `opencv-python` out (verify the installed pixi supports
  it). Fallback: a pixi task `install-av2 = "python -m pip install --no-deps av2==0.3.6"` next
  to `install-project`, plus the runtime imports AV2 MF actually needs in
  `[project.dependencies]`: `universal-pathlib` (verified required), and `numba`, `kornia`,
  `av`, `pyproj` only if the import gate below demands them. Re-lock with pixi and mirror the
  step in `docker/Dockerfile` if it installs dependencies separately.
- Gate: `python -c "import av2.datasets.motion_forecasting.scenario_serialization, av2.map.map_api, av2.geometry.interpolate"`
  and a clean `pip check`.
- New `autoware_ml/types/motion.py`; extend `autoware_ml/types/tasks.py`.
- `.cspell.json`: `argoverse`, `av2`, `motorcyclist`, `riderless`, `crosswalk`, `polyline(s)`,
  as flagged by the checker.

### Step 2: relax `DatabaseTaxonomy` (`autoware_ml/databases/taxonomy.py`)

- Add `MotionForecastingTaxonomy(LabelTaxonomy)`.
- `DatabaseTaxonomy(detection3d=None, segmentation3d=None, motion_forecasting=None)`:
  isinstance checks only when a head is set; at least one head required; the properties raise a
  descriptive `ValueError` when the head is absent (never return `None`, so `.class_names` on a
  motion-only database fails clearly).
- `__str__` stays byte-identical for the two-head case and appends `, motion_forecasting=...`
  only when set. `str(taxonomy)` feeds `hash_fields`, so this keeps every existing cache hash
  stable in this step.
- Tests in `autoware_ml/databases/tests/test_taxonomy.py`: motion-only accepted, all-None
  rejected, absent head raises, two-head string unchanged.

### Step 3: relax `BaseDatabase.__init__` (`autoware_ml/databases/base_database.py`)

- `box3d_pipelines: Sequence[Box3DPipeline] = ()`, `lidar_intensity_scale: float | None = None`,
  `lidar_pointcloud_num_features: int | None = None`; validate only when not None.
- `hash_fields` keeps its shape (`str(5)` stays `"5"` for existing databases; AV2 renders
  `"[]"` and `"None"`). The `lidar_intensity_scale` property becomes `float | None`; update the
  `DatabaseInterface` docstrings in `autoware_ml/databases/database_interface.py`.
- Tests: `autoware_ml/databases/tests/test_database_hash.py` passes unchanged; add a
  construction test without lidar settings.

### Step 4: schema column

New `autoware_ml/databases/schemas/motion_forecasting.py`:

| Struct field | Polars type | Nullable | Meaning |
| --- | --- | --- | --- |
| `motion_tracks_relative_path` | String | no | Track source file, relative to the database root |
| `motion_tracks_format` | String | no | `TracksFormat` value |
| `motion_map_relative_path` | String | yes | Map file, relative to the database root |
| `motion_map_format` | String | yes | `VectorMapFormat` value |
| `motion_focal_track_id` | String | yes | Focal agent id |
| `motion_current_step_index` | Int32 | no | Index of the "current" step inside the window |
| `motion_num_steps` | Int32 | no | Steps in the window (110 or 50) |
| `motion_step_seconds` | Float64 | no | Step period |
| `motion_num_tracks` | Int32 | no | Tracks in the scenario |

- `MotionForecastingDatasetSchema(BaseFieldSchema)` and
  `MotionForecastingDataModel(BaseModel, DataModelInterface)` (frozen, strict; enum fields
  coerced from the stored strings; relative-path validator reusing
  `autoware_ml/databases/schemas/utils.py::check_relative_path`;
  `0 <= current_step_index < num_steps`; `step_seconds > 0`; `to_dictionary` /
  `load_from_dictionary`).
- `dataset_schemas.py`: `DatasetTableSchema.MOTION_FORECASTING =
  DatasetTableColumn("motion_forecasting", pl.Struct(...))` and
  `DatasetRecord.motion_forecasting: MotionForecastingDataModel | None = None`. The default
  means the T4 / nuScenes generators and `autoware_ml/datamodule/tests/corpus.py` need no
  edits. Use the null-struct guard as for `category_mapping`.
- Consequence to state in the PR: the schema string changes, so every existing cache is
  regenerated once on the next `prepare_data`.
- Tests: `autoware_ml/databases/schemas/tests/test_motion_forecasting.py` (round trip through
  `pl.DataFrame(..., schema=DatasetTableSchema.to_polars_schema())`, null struct, validators).
  Docs: `docs/databases/schemas.md`.

### Step 5: scenarios (`autoware_ml/databases/argoverse2/av2_motion_scenarios.py`)

- `AV2MotionScenarios(Scenarios)`: each `dataset_params[i].dataset_name` must be one of
  `train`, `val`, `test` and names a split directory under `scenario_root_path` (a `ValueError`
  lists the accepted names otherwise). List the subdirectories sorted, apply `sample_steps` as
  a stride over scenarios (one scenario is one sample, so the frame stride of the other
  databases becomes a scenario stride; document this in the docstring and the config), and
  build `ScenarioData(scenario_id=<dir>, scenario_version=<split>, vehicle_type=None,
  location=None)`.
- Override `hash_repr` to emit per split `n=<count> sha256=<digest of the sorted ids>` instead
  of listing 250k entries; `scenario_root_path` stays out, as in the base class.
- Tests: `autoware_ml/databases/argoverse2/tests/test_av2_motion_scenarios.py` with `tmp_path`
  directories (discovery, stride, unknown split name, missing directory, compact `hash_repr`
  that excludes the root path and changes with the id set).

### Step 6: AV2 reader (`autoware_ml/databases/argoverse2/av2_motion_reader.py`)

The only module importing `av2`.

- `read_scenario_metadata(tracks_path) -> ScenarioMetadata` (scenario_id, city,
  focal_track_id, timestamps_seconds, num_tracks).
- `read_agent_tracks(tracks_path, taxonomy) -> AgentTracksSample` via
  `load_argoverse_scenario_parquet`: `[A, T]` arrays filled by `timestep`, `valid` from
  presence, `agent_types = taxonomy.resolve_index(object_type.value)`, `agent_roles` from
  `category`, the focal track at row 0, z = 0 for agents.
- `read_vector_map(map_path, num_points_per_polyline, polyline_types) -> MapPolylinesSample`
  via `ArgoverseStaticMap.from_json` (it parses the log id from the
  `log_map_archive_<id>.json` filename, so keep AV2's filenames).
- Everything is returned as torch tensors (float32 / int64 / bool) in the city frame.
- Tests (`pytest.importorskip("av2")`): a fixture builder writes a 3-track, 110-step scenario
  with `serialize_argoverse_scenario_parquet` plus a minimal hand-written map JSON (one lane
  segment, one crossing, one drivable area) into `tmp_path`; assert shapes, focal first,
  type / role indices, resampling to `L`, unknown object type raises. Steps 9 and 10 reuse this
  builder.

### Step 7: records generator and database (`autoware_ml/databases/argoverse2/`)

- `AV2MotionRecordsGenerator(database_root_path, scenario_data).generate_dataset_records()`:
  locate `<root>/<split>/<id>/scenario_<id>.parquet` and `log_map_archive_<id>.json`
  (`ValueError` if missing), read the metadata, and build exactly one `DatasetRecord`:
  `scenario_id = sample_id = scenario_name = id`, `sample_index = 0`,
  `timestamp_seconds = timestamps[49]`, `location = city`, `vehicle_type = None`,
  `lidar_frames = []`, every other annotation column `None`, and
  `motion_forecasting = MotionForecastingDataModel(...)` with paths made relative through
  `autoware_ml/databases/schemas/utils.py::relative_to_database_root`.
- `AV2MotionDatabase(BaseDatabase)`: `__init__(version, root_path, scenarios, cache_path,
  cache_file_prefix_name, num_workers, taxonomy, scenarios_per_worker_task=256)`; passes
  `box3d_pipelines=()` and `None` lidar settings; requires `taxonomy.motion_forecasting`.
  `__str__` prints per-split counts, not lists. `hash_fields` adds `observed_steps`,
  `tracks_format`, `map_format`. `database_hash` is memoized (scenarios are frozen).
  `process_scenario_records()` follows `autoware_ml/databases/nuscenes/nuscenes_dataset.py`
  but the workers take chunks of scenarios (frozen worker-params dataclass plus a module-level
  `_apply_*` function) to avoid pickling 250k tasks, then
  `pl.DataFrame(dicts, schema).write_parquet`.
- Tests: `test_av2_motion_records_generator.py` (record content, relative paths, missing files)
  and a `TestAV2MotionDatabaseHash` class in `autoware_ml/databases/tests/test_database_hash.py`
  (paths ignored; version, taxonomy, stride and `OBSERVED_STEPS` via `patch.object` change it;
  `hash_repr` stays small for many scenarios).

### Step 8: batch dataclasses

New `autoware_ml/dataclasses/batch/motion_forecasting.py`:

| Class | Content |
| --- | --- |
| `MotionSourceSample` | File references resolved against the dataset root, formats, focal id, `current_step_index`, `num_steps`, `step_seconds`, `num_tracks`; the analog of `LiDARPointCloudSample` |
| `AgentTracksSample` | `positions [A,T,3]`, `headings [A,T]`, `velocities [A,T,2]`, `valid [A,T]`, `agent_types [A]`, `agent_roles [A]`, `track_ids`, `current_step_index`, `step_seconds`, `ignore_index` |
| `MapPolylinesSample` | `points [P,L,3]`, `valid [P,L]`, `polyline_types [P]`, `is_intersection [P]`, `element_ids` |
| `MotionForecastingGTSample` | `agents: AgentTracksSample`, `map_polylines: MapPolylinesSample \| None` |
| `MotionForecastingGTBatch` | Padded to the batch-max `A` and `P` with `agent_mask [B,A]` and `polyline_mask [B,P]`; requires equal `T`, `L`, `current_step_index`, `step_seconds` (`ValueError` otherwise); `collate_gt_samples`, `to_device` |

`autoware_ml/dataclasses/batch/sample_batch.py`: append `motion_source_sample=None` and
`motion_forecasting_gt_sample=None` to `ModelGTSample`; append
`motion_forecasting_gt_batch=None` to `ModelGTBatch`; add
`collate_motion_forecasting_gt_samples` (None if no sample carries it, raise if only some do,
template `collate_segmentation3d_gt_samples`), call it in `collate_gt_samples`, move it in
`to_device`. Dynamic padding keeps the `collate_fn` signatures of `BaseDataset` and
`ConcatDataset` untouched; a cap on agents or polylines belongs in a transform.

Tests: `autoware_ml/dataclasses/batch/tests/test_motion_forecasting.py` (padding, masks,
mismatched `T` rejected, `to_device` on CPU) and additions to `test_sample_batch.py`
(none / partial / all-present collation of the new field).

### Step 9: loading transforms (`autoware_ml/transforms/motion_forecasting/loading.py`)

- `LoadAgentTracks(taxonomy)`: `_required_keys = ["motion_source_sample"]`; dispatches on
  `tracks_format` with a lazy import of the AV2 reader so `av2` stays optional for lidar-only
  users; fills `motion_forecasting_gt_sample`.
- `LoadVectorMap(num_points_per_polyline=20, polyline_types=all)`: requires the GT sample;
  raises when the record has no map path.
- The taxonomy reaches the transform by Hydra interpolation
  (`taxonomy: ${database.taxonomy.motion_forecasting}`), the same mechanism
  `frame_sampling` uses for `${database.taxonomy.detection3d.class_names}`.
- Agent-centric normalization is not in scope (a model-side choice); listed as a follow-up.
- Tests: `autoware_ml/transforms/tests/test_motion_forecasting_loading.py` (importorskip av2):
  required-key errors, missing map error, shapes.

### Step 10: dataset (`autoware_ml/datamodule/motion_forecasting/dataset.py`)

- `MotionForecastingDataset(BaseDataset)`: `__init__(database_root_path, *, transforms,
  dataset_records_dataframe=None)` passing `det3d_supervised=False, seg3d_supervised=False`.
- `_serve_records` selects `SCENARIO_ID`, `TIMESTAMP_SECONDS`, `MOTION_FORECASTING` and raises
  if the struct is null for any row (the table was not built for motion forecasting).
- `get_data_sample` returns a `ModelGTSample` with every sensor / box field `None`,
  `frame_meta = FrameMetaSample(ego2global=eye(4), scene_token=scenario_id,
  timestamp=timestamp_seconds, lanelet_map_path=None)` and a `motion_source_sample` whose paths
  are joined onto `database_root_path`.
- `scene_index_groups()` works unchanged (one record per scenario); `DataModule.setup` logs
  `det3d False, seg3d False`, which is accurate.
- Tests: `autoware_ml/datamodule/motion_forecasting/tests/test_dataset.py` (2-scenario
  synthetic corpus -> generator -> `bind` -> `__getitem__` -> `collate_fn`), plus a datamodule
  end-to-end test reusing the `FakeDatabase` and splitter helpers of
  `autoware_ml/datamodule/tests/test_data_module.py` with `num_workers=0`.

### Step 11: Hydra configs

- `autoware_ml/configs/database/argoverse2/argoverse2_motion_forecasting_base.yaml`:
  `_target_: ...AV2MotionDatabase`, `version: AV2MotionForecasting-base-v1.0.0`,
  `root_path: ${data_root_path}/argoverse2/motion_forecasting/`,
  `cache_path: ${cache_root_path}/motion_forecasting/`,
  `cache_file_prefix_name: av2_motion_forecasting_database`, `num_workers: 16`, defaults for the
  taxonomy and the scenarios.
- `.../argoverse2/scenarios/motion_forecasting/db_av2_motion_forecasting.yaml`:
  `AV2MotionScenarios`, `scenario_root_path: ${data_root_path}/argoverse2/motion_forecasting/`,
  three `DatasetParams` (`train`, `val`, `test`, `sample_steps: 1`) with the stride comment.
- `.../argoverse2/taxonomy/default_motion_forecasting.yaml` (`DatabaseTaxonomy` with only
  `motion_forecasting`), `taxonomy/motion_forecasting/default_coarse.yaml`
  (`MotionForecastingTaxonomy`, section 3.2), `taxonomy/vocabulary/default_motion_forecasting.yaml`
  (`LabelVocabulary`, section 3.2).
- `autoware_ml/configs/datamodule/argoverse2/default_motion_forecasting_datamodule.yaml` (same
  three-line shape as `nuscenes/default_detection3d_datamodule.yaml`) and
  `datasets/default_motion_forecasting_{train,test}_dataset.yaml`
  (`MotionForecastingDataset`, `database_root_path: ${database.root_path}`,
  `TransformsCompose` with `LoadAgentTracks` and `LoadVectorMap`).
- Test `autoware_ml/configs/tests/test_motion_forecasting_datamodule_configs.py`: compose via
  `ConfigStore` as `test_datamodule_configs.py` does, set `AUTOWARE_ML_DATA_PATH` with
  `monkeypatch`, compare the datamodule keys to `DataModule.__init__` (compose only, since
  instantiating would list the data directories).

### Step 12: docs

- `docs/databases/argoverse2.md`: summary table, expected layout, download, mapping tables,
  install note.
- Nav entry in `zensical.toml`; column and struct section in `docs/databases/schemas.md`;
  extension-table row in `docs/databases/design.md`; one-line supported-databases note in
  `docs/index.md`.

## 5. Verification

1. Unit tests: `pytest autoware_ml/ -v`; `pre-commit run --all-files` (ruff, yamllint,
   markdownlint, cspell).
2. Dependency gate (step 1): the av2 import check and `pip check` inside the pixi env.
3. Download (not yet on disk):

   ```bash
   s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/*" \
     "${AUTOWARE_ML_DATA_PATH}/argoverse2/motion_forecasting/"
   ```

   Start with `val/` only for a smoke run and confirm the layout
   `<split>/<id>/scenario_<id>.parquet` + `log_map_archive_<id>.json`.

4. Real-data smoke run: a Python snippet that stores a `ConfigStore` node composing
   `/defaults/modules/paths`, `/database/argoverse2@database: argoverse2_motion_forecasting_base`
   and `/datamodule/argoverse2@datamodule: default_motion_forecasting_datamodule`, overrides
   `dataset_params.*.sample_steps=1000` for a ~250-scenario subset, then
   `instantiate(cfg.datamodule)`, `prepare_data()` (the cache file appears under
   `<cache_root>/motion_forecasting/`), `setup("fit")`, pulls one train batch and checks the
   `motion_forecasting_gt_batch` shapes (`[B, A, 110, 3]`, polylines `[B, P, 20, 3]`, masks),
   then `setup("test")` and checks `T == 50`.
5. Full generation timing with `sample_steps=1`: expect ~250k rows and a Parquet file of tens of
   MB, not GB.

## 6. Risks and notes

- `av2` dependency tree: `opencv-python` vs `opencv-python-headless` (double `cv2`); `nox`,
  `kornia`, `numba`, `av` are pulled as runtime dependencies if the tree is resolved normally.
  Decide override vs `--no-deps` task at step 1 using the import gate.
- One-time cache regeneration for every database (accepted); state it in the PR description.
- `sample_steps` doubles as a scenario stride for AV2 (documented). If it confuses, an
  AV2-only `DatasetParams` subclass can be added later.
- 250k-entry scenario strings are avoided by the digest `hash_repr` and the memoized
  `database_hash`; make sure nothing logs `str(scenarios)` for AV2.
- Generation time is roughly 10 minutes for 250k scenarios with 16 workers
  (`load_argoverse_scenario_parquet` is pandas plus Python objects, ~20-40 ms each). A
  metadata-only polars read is a contained fallback inside `read_scenario_metadata`.
- Per-sample load cost (parquet + map JSON + resampling hundreds of lanes) is tens of ms; fine
  for the data layer, but a radius or nearest-k polyline filter transform will be wanted before
  training.
- Mixing train (110 steps) and test (50 steps) records in one split raises at collate by
  design.
- Follow-ups outside this scope: agent-centric normalization and nearest-k polyline transforms,
  a forecasting model with minADE / minFDE / miss-rate metrics, T4dataset readers
  (`T4_ANNOTATION_TRACKS`, `LANELET2_OSM`), visualization support.

## 7. Files touched (summary)

New:

- `autoware_ml/types/motion.py`
- `autoware_ml/databases/schemas/motion_forecasting.py`
- `autoware_ml/databases/argoverse2/{__init__.py, av2_motion_scenarios.py, av2_motion_reader.py, av2_motion_records_generator.py, av2_motion_database.py, tests/*}`
- `autoware_ml/dataclasses/batch/motion_forecasting.py`
- `autoware_ml/transforms/motion_forecasting/{__init__.py, loading.py}`
- `autoware_ml/datamodule/motion_forecasting/{__init__.py, dataset.py, tests/*}`
- `autoware_ml/configs/database/argoverse2/**`, `autoware_ml/configs/datamodule/argoverse2/**`
- `autoware_ml/configs/tests/test_motion_forecasting_datamodule_configs.py`
- `docs/databases/argoverse2.md`

Modified:

- `pyproject.toml`, `pixi.lock`, `.cspell.json`, `docker/Dockerfile` (if needed)
- `autoware_ml/types/tasks.py`
- `autoware_ml/databases/taxonomy.py`, `autoware_ml/databases/base_database.py`,
  `autoware_ml/databases/database_interface.py`
- `autoware_ml/databases/schemas/dataset_schemas.py`
- `autoware_ml/dataclasses/batch/sample_batch.py`
- `autoware_ml/databases/tests/{test_taxonomy.py, test_database_hash.py}`,
  `autoware_ml/dataclasses/batch/tests/test_sample_batch.py`
- `docs/databases/{schemas.md, design.md}`, `docs/index.md`, `zensical.toml`
