# Autoware-ML Ops

`autoware_ml/ops` contains low-level operators and export bridges that are part
of the model graph at runtime or deployment time.

This package is reserved for code that does at least one of the following:

- wraps a custom CUDA/C++ extension
- exposes an ONNX symbolic or export-only operator bridge
- adapts an external runtime library for model execution
- provides graph-level helper operators used directly by model code

If a helper is only used for data preparation, transforms, or general Python
bookkeeping, it should not live in `ops/`.

## Package Layout

- `bev_pool/`
  - BEV pooling CUDA/C++ extension used by camera-lidar detection models
- `indexing/`
  - export-aware indexing operators such as `argsort` and `unique`
  - used by PTv3 export and other graph-level point-cloud utilities
- `segment/`
  - segment-reduction operators and ONNX helpers
  - includes `segment_csr` and scatter-reduce symbolic registration
- `spconv/`
  - deployment-aware sparse convolution wrappers built on top of external `spconv`
  - keeps eager execution aligned with exportable ONNX custom ops

## Design Rules

- Keep the public API in each subpackage `__init__.py`.
- Prefer model-facing wrappers over ad hoc ONNX graph patching.
- Keep task-specific export symbolics in `ops/`, not inside model wrappers.
- Only add logic here when it is part of the executed or exported graph.

## Building Extensions

After creating the contributor environment with `pixi`, rebuild the project in
editable mode to compile the local ops extensions:

```shell
cd /workspace
pixi run --environment dev setup-project
```

## Generating Compilation Database

To generate `compile_commands.json` for clangd:

```shell
bear --output /workspace/autoware_ml/ops/compile_commands.json -- \
  pixi run --environment dev python -m pip install --ignore-installed --no-build-isolation --no-cache-dir --no-deps -v -e .
```

After generation, reload clangd (`Ctrl+Shift+P` -> `clangd: Restart language server`) to pick up the new compilation database.

The `default` pixi environment is not sufficient for rebuilding ops locally;
use `dev` for any extension build or C/C++ tooling workflow.
