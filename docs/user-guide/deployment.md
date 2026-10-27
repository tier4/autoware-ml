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

Every exported module must have a matching entry under `deploy.onnx.modules`,
and artifact files are named after the module. A module missing from the
config fails the deploy immediately.

## Multi-head exports

Multi-head export models can expose multiple deployable modules from one
configured model. `--weights` can merge checkpoints from independently
trained parts of that model into a single export:

```bash
autoware-ml deploy \
    --config-name detection3d/ptv3/t4dataset/voxel012_tiny_b8_30e_t4dataset_122m_j6gen2_base \
    --weights mlruns/ptv3_seg3d/T4Dataset-j6gen2_semaseg-v1.0.0/j6gen2_semaseg/voxel012_tiny_b16_50e_t4dataset_122m/<run_id>/artifacts/checkpoints/best.ckpt \
    --weights mlruns/ptv3/T4Dataset-j6gen2_base-v1.0.0/j6gen2_base/voxel012_tiny_b8_30e_t4dataset_122m/<run_id>/artifacts/checkpoints/best.ckpt
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
          class_names: ${database.taxonomy.detection3d.class_names}
```

Each `metainfo` value is stamped as one compact JSON document (scalars,
strings, lists and mappings, nested as declared), while the provenance
properties above are plain strings, so a consumer reads a parameter with any
JSON parser. Non-finite floats, values JSON cannot represent, and keys that
collide with the automatically stamped properties fail at export time.

## Configuration

### ONNX Settings

Shared settings live at `deploy.onnx`. Per-module settings under
`deploy.onnx.modules.<module>` override them:

```yaml
deploy:
  onnx:
    enabled: true
    dynamo: true
    opset_version: 21
    precision: fp32
    dynamic_shapes:
      input_tensor: { 2: height, 3: width }
    modules:
      end_to_end:
        precision: fp16
```

**Tensor names**: the export spec names the graph inputs and outputs, either
the names a stage graph declares (see
[Model-Owned Export Wrappers](#model-owned-export-wrappers)) or the names a
hand-written spec lists; the export log prints them. A spec names every one of
its inputs and at least one output, or it is refused when built. `deploy.onnx`
carries no `input_names` or `output_names`; an entry fails the deploy before its
MLflow run is created or the model loads. The keys of `dynamic_shapes`,
`dynamic_axes` and `deploy.tensorrt.input_shapes` are these names. A model that
lets a config choose its names exposes them as a model parameter, as PTv3's
`export_output_names`.

**precision**: `fp32` exports the model unchanged. `fp16` halves the weights and
internal tensors but keeps graph inputs and outputs fp32, so consumers keep their
fp32 buffers either way. Calibrated precisions such as `int8` are out of scope here.

An fp16 export only works if the inference engine honors its dtypes instead of
reassigning them. In Autoware, that means `trt_precision: strongly-typed` on the
node.

**dynamic_shapes**: Keys are exported input names, values map dimension indices
to symbolic names. For the default export path these names come from
`forward()`. Models with explicit export wrappers define their own exported
input names through `build_export_spec()`.
You can also provide symbolic bounds when export needs them:

```yaml
deploy:
  onnx:
    dynamic_shapes:
      points:
        0: { name: num_points, min: 2 }
```

Set `dynamo: false` for models that rely on legacy ONNX symbolic functions
instead of `torch.export`. In that mode, `dynamic_axes` is passed to the legacy
exporter directly, and `dynamic_shapes` can still be used as a shorthand to
derive equivalent symbolic axes. An export spec that carries its own
`dynamic_axes` needs `dynamo: false`: with `dynamo: true` the export stops with
an error instead of producing a graph with those axes static.

### TensorRT Settings

```yaml
deploy:
  tensorrt:
    enabled: true
    workspace_size: 8589934592  # 8 GiB
    input_shapes:
      input:
        min_shape: [1, 3, 224, 224]
        opt_shape: [1, 3, 256, 256]   # Optimized for this
        max_shape: [1, 3, 512, 512]
```

!!! tip
    TensorRT optimizes most aggressively for `opt_shape`. Set this to your typical inference resolution.

Engines are always built strongly typed: the builder uses exactly the precisions
the ONNX carries and never picks its own. Choose the engine's numerics with
`deploy.onnx.precision`.

## Model-Owned Export Wrappers

The preferred deployment path is to keep export logic inside the model. Models
with deployment-specific requirements should override `build_export_spec()` and
return an explicit export module plus example tensor inputs.

This keeps export-time behavior close to the model implementation and avoids
ad hoc post-processing for most cases.

A model whose deployment splits into several exported graphs with PyTorch glue
between them can declare that split once instead, by overriding
`build_stages()`: an ordered list of `GraphStage` (one exported module, named
after the stage, with the tensor names it reads and writes) and `TorchStage`
(glue that is not exported). `build_export_specs()` then checks the dataflow,
runs the stages once on the example batch in eval mode, and returns one export
spec per `GraphStage`, traced on the tensors the glue produced. The model
outputs are the graph outputs no later stage reads, so a declaration ends with
a `GraphStage`; PyTorch post-processing of the deployed outputs belongs in
`build_deployed_eval_output()`. CenterPoint declares its two modules this way.

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
    deploy.tensorrt.input_shapes.input.opt_shape=[1,3,256,256] \
    deploy.tensorrt.workspace_size=8589934592
```
