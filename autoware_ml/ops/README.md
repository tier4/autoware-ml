# Autoware-ML Custom Operations

This directory contains custom CUDA/C++ operations for the Autoware-ML framework.

## Building Extensions

Extensions only can be quickly built without dependencies with:

```shell
cd /workspace
pip install --no-build-isolation --no-cache-dir --no-deps -v -e .
```

## Generating Compilation Database

To generate `compile_commands.json` for clangd IDE support:

```shell
bear --output /workspace/autoware_ml/ops/compile_commands.json -- \
  pip install --ignore-installed --no-build-isolation --no-cache-dir --no-deps -v -e .
```

After generation, reload clangd (`Ctrl+Shift+P` -> `clangd: Restart language server`) to pick up the new compilation database.
