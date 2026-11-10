---
icon: lucide/package
---

# Deployment

Autoware-ML exports trained models for production use. The pipeline converts
PyTorch checkpoints to optimized inference formats, one deployable module at a
time, and stamps every exported module with its identity and provenance.

## Deployment Pipeline

```text
Checkpoint (.ckpt) -> ONNX (.onnx) -> graph modifiers -> precision -> metadata stamp -> TensorRT (.engine)
```

Each stage runs per module. The stamp is applied to the final ONNX, so the
shipped file always carries its own metadata.

## Basic Usage

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt
```

`--weights` accepts one or more checkpoint paths and is the only way to supply
parameters to the export model. For a single-task export, pass one
checkpoint. For a multi-head export, pass one `--weights` per source
checkpoint (see [Multi-head exports](#multi-head-exports)).

!!! warning
    Pass `--release vMAJOR.MINOR.PATCH` if the model is version controlled.
    Without it the artifacts are stamped `unversioned`, which is fine for
    quick iteration but never for a release build.

This generates one ONNX (`<module>.onnx`) and one TensorRT engine
(`<module>.engine`) per deployable module, when both stages are enabled and
supported by the model. The deploy command also creates a dedicated MLflow run
linked to the source training run and logs exported artifacts there.

You can disable either stage during iteration:

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

Deploy writes its outputs into the deploy MLflow run's artifact directory, under
`mlruns/<experiment_group_name>/<experiment_name>/<deploy_run_id>/artifacts/exports/`.
The same run records the resolved config and CLI overrides under `config/`, and a
`run_metadata.json` listing the source checkpoints, so an export can always be traced
back to the training run it came from. A deploy that fails after its run was created
leaves the run marked `FAILED`.

Without MLflow logging (`~logger`), outputs land in the config's `experiment_run_dir`,
`<working_dir>/tasks/<experiment_group_name>/<experiment_name>/<experiment_uid>/`.
The output location is not configurable on its own.

## Export Modules

A model exports one or more deployable modules. Models without a dedicated
export split produce a single module named `end_to_end`.

Artifact files are named after the module. Every exported module has an entry
under `deploy.onnx.modules`: `{}` exports it with the shared `deploy.onnx`
settings, and any key in the entry overrides them. A module without an entry
fails the deploy before the deploy run is created, so a module added to a model
cannot silently export without its `metainfo` or `modify_graph`, and a
misspelled entry shows up as the missing one. Entries of modules the model does
not export are left alone, so a model whose modules depend on its config, as
BEVFusion's on `camera_network`, lists the entries of every variant in one base
config.

## Multi-head exports

Multi-head export models can expose multiple deployable modules from one
configured model. `--weights` can merge checkpoints from independently
trained parts of that model into a single export:

```bash
autoware-ml deploy \
    --config-name detection3d/ptv3/nuscenes/voxel005_b16_30e_nuscenes_51m \
    --weights mlruns/ptv3_seg3d/NuScenesDataset-base-v1.0.0/nuscenes/voxel005_b16_50e_nuscenes_51m/<run_id>/artifacts/checkpoints/best.ckpt \
    --weights mlruns/ptv3/NuScenesDataset-base-v1.0.0/nuscenes/voxel005_b16_30e_nuscenes_51m/<run_id>/artifacts/checkpoints/best.ckpt
```

Checkpoints are applied in the order they appear on the command line, and
later checkpoints overwrite any keys already set by earlier ones. Each
checkpoint only contributes the state-dict keys that exist on the export
model and match its tensor shapes. Keys missing from the model are skipped.
Keys with a matching name but mismatching shape raise an error immediately.

**Full coverage is enforced.** After all checkpoints are loaded, deploy
verifies that every parameter in the export model has been covered by at
least one of the supplied `--weights`. If any parameter is left
uninitialized, the command fails up front with the list of missing keys
instead of producing an ONNX or engine that contains untrained layers. Add
or replace `--weights` entries until every key is covered.

## ONNX Metadata

Every exported module carries its identity and provenance inside the ONNX
file: producer, git commit of the export, release (encoded into
`model_version` as `major * 10000 + minor * 100 + patch`), the config name,
export date, and the linked deploy run.

Per-module inference parameters come from the deploy config's `metainfo`
block. The pipeline serializes whatever the config declares without
interpreting it:

```yaml
deploy:
  onnx:
    modules:
      <module>:
        metainfo:
          class_names: ${taxonomy.detection3d.class_names}
```

Each `metainfo` value is stamped as one compact JSON document (scalars,
strings, lists and mappings, nested as declared), while the provenance
properties above are plain strings, so a consumer reads a parameter with any
JSON parser. Non-finite floats, values JSON cannot represent, and keys that
collide with the automatically stamped properties fail at export time.

## Configuration

### ONNX Settings

Shared settings live at `deploy.onnx`. Each exported module lists an entry
under `deploy.onnx.modules.<module>`, `{}` to keep the shared settings or the
keys that override them:

```yaml
deploy:
  onnx:
    enabled: true
    dynamo: true
    opset_version: 21
    precision: fp32
    modules:
      end_to_end:
        precision: fp16
```

**Tensor names**: the export spec names the graph inputs and outputs, either
the names the deployment's graphs declare (see
[Model-Owned Export Wrappers](#model-owned-export-wrappers)) or the names a
hand-written spec lists; the export log prints them. A spec names every one of
its inputs and at least one output, or it is refused when built. `deploy.onnx`
carries no `input_names` or `output_names`; an entry fails the deploy before its
MLflow run is created or the model loads. The keys of each module's
`deploy.tensorrt.modules.<module>.input_shapes` are these names. A model that lets a config
choose its names exposes them as a model parameter, as PTv3's
`export_output_names`.

**Dynamic axes**: which axes vary between runs is a property of the model, so
the export spec declares them once (`ExportSpec.dynamic_axes`, keyed by its
input and output names) and both exporters derive their arguments from that
declaration. `deploy.onnx` carries no `dynamic_axes` or `dynamic_shapes`; an
entry fails the deploy like a configured name does. The spec refuses an axis of
a tensor it does not name. The deployment settings that do vary, the TensorRT
shape ranges, stay in `deploy.tensorrt.modules.<module>.input_shapes`.

**precision**: `fp32` exports the model unchanged. `fp16` halves the weights and
internal tensors but keeps graph inputs and outputs fp32, so consumers keep their
fp32 buffers either way. Calibrated precisions such as `int8` are out of scope here.

An fp16 export only works if the inference engine honors its dtypes instead of
reassigning them. In Autoware, that means `trt_precision: strongly-typed` on the
node.

**fp16_range_share**: before the export, the module runs once on its trace batch and every float tensor it reads or computes, a graph input or an intermediate such as an attention logit, has to stay within this share of the fp16 maximum (65504). An engine that runs a layer in fp16 overflows silently on a larger value, so the export is refused instead, and the error names the module path and the op of the largest offenders. Minus infinity masks and the float32 extremes of empty reductions are sentinels and left out, a positive infinity is an overflow. The check follows the engines, not `precision`: an fp32 ONNX still runs in fp16 when the engine is built with fp16 enabled. A module whose numerics need fp32 declares `required_onnx_precision = "fp32"` and is skipped, and a deployment with fp32 engines sets `fp16_range_share: null` for a module under `deploy.onnx.modules`.

**dynamo**: `true` exports through `torch.export`, which gets one `Dim` per
axis name the spec declares over the inputs. Set `dynamo: false` for models
that export through the TorchScript exporter and its ONNX symbolic functions;
it takes the spec's `dynamic_axes` directly.

### TensorRT Settings

```yaml
deploy:
  tensorrt:
    enabled: true
    workspace_size: 8589934592  # 8 GiB
    modules:
      end_to_end:                     # one entry per exported module, keyed like deploy.onnx.modules
        input_shapes:
          input:
            min_shape: [1, 3, 224, 224]
            opt_shape: [1, 3, 256, 256]   # Optimized for this
            max_shape: [1, 3, 512, 512]
```

!!! tip
    TensorRT optimizes most aggressively for `opt_shape`. Set this to your typical inference resolution.

Every dynamic input of a module needs its min / opt / max shapes; a static input
needs none. An entry that names no input of the module, a dynamic input without
an entry, or a table left at the old `deploy.tensorrt.input_shapes` stops the
build with an error naming the key.

Engines are always built strongly typed: the builder uses exactly the precisions
the ONNX carries and never picks its own. Choose the engine's numerics with
`deploy.onnx.precision`.

## Model-Owned Export Wrappers

Export logic stays inside the model. A model declares its deployment with two
methods:

- `deployment_graphs()` returns one `DeploymentGraph` per exported computation
  graph (one ONNX file and one TensorRT engine each): its name, its export-ready
  module, the tensor names it takes and returns, and its dynamic axes.
- `deployment_forward(multi_task_batch_inputs, executors, device)` is the
  PyTorch code between the graphs. It takes the preprocessed batch `forward()`
  takes, runs each graph as `executors[name].run(input_name=tensor)`, creates its
  own tensors on `device` and returns the model outputs by name.

```python
def deployment_graphs(self):
    return (
        DeploymentGraph("encoder", module=self.encoder, input_names=("features",),
                        output_names=("encoded",),
                        onnx_dynamic_axes={"features": {0: "num_points"}, "encoded": {0: "num_points"}}),
        DeploymentGraph("head", module=self.head, input_names=("bev",), output_names=("heatmap", "reg")),
    )

def deployment_forward(self, multi_task_batch_inputs, executors, device):
    encoded = executors["encoder"].run(features=self.featurize(multi_task_batch_inputs))["encoded"]
    return executors["head"].run(bev=self.scatter(encoded).to(device))
```

`build_export_specs()` runs `deployment_forward()` once in eval mode and returns
one export spec per graph, traced on a copy of the tensors the function passed
it. The rules the run enforces:

- Every declared graph runs exactly once: one graph is one engine with one
  optimization profile, traced on the inputs of that run. A sub-network the model
  applies several times in one inference (once per camera, say) is one graph run
  once on the batched inputs, as BEVFusion's image backbone takes every camera
  at once.
- A graph's module returns its outputs as a tensor, a tuple in `output_names`
  order, or a mapping with the keys in that order.
- A `DeploymentGraph` module is export-ready as declared (the live submodule or
  the copy its `prepare_for_export()` makes); the framework switches it to eval
  mode for the export and changes nothing else, and declares the graphs again
  after replacing a module.

`forward()` and `deployment_forward()` compose the same modules twice, and
nothing but a test keeps the two equal: every model that declares graphs runs
`assert_deployment_reproduces_forward()` (in
`autoware_ml/deployment/tests/deployment_assertions.py`) on one batch in its
tests, comparing every model output.

A model whose deployment is one graph without code around it can still override
`build_export_spec()` and return an export module plus example tensor inputs;
this single-graph path is kept until TransFusion, FRNet, CalibrationStatus and
camera-lidar BEVFusion declare `deployment_graphs()` (see the TODO in
`ModuleBaseModel`).

## Optional Graph Modification

Post-export ONNX graph modification is still available as a fallback, shared
across modules or per module:

```yaml
deploy:
  onnx:
    modules:
      <module>:
        modify_graph:
          _target_: my_module.OnnxGraphModifier
          # modifier-specific parameters
```

Use for operator replacement, shape inference fixes, or custom plugin insertion.
Modifiers run before the precision conversion and the metadata stamp, so the
shipped file reflects them.

## Overriding at Runtime

Override deployment settings from CLI:

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<experiment_group_name>/<experiment_name>/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.modules.end_to_end.input_shapes.input.opt_shape=[1,3,256,256] \
    deploy.tensorrt.workspace_size=8589934592
```
