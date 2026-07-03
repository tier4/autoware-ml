---
icon: lucide/package
---

# Deployment

Autoware-ML exports trained models for production use. The pipeline converts PyTorch checkpoints to optimized inference formats.

## Deployment Pipeline

```text
Checkpoint (.ckpt) -> ONNX (.onnx) -> TensorRT (.engine)
```

## Basic Usage

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt
```

`--weights` accepts one or more checkpoint paths and is the only way to supply
parameters to the export model. For a single-task export, pass one
checkpoint. For a multi-head export, pass one `--weights` per source
checkpoint (see [Multi-head exports](#multi-head-exports)).

This generates ONNX (`.onnx`) and TensorRT (`.engine`) files when both stages
are enabled and supported by the model. The deploy command also creates a
dedicated MLflow run linked to the source training run and logs exported
artifacts there.

You can disable either stage during iteration:

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.enabled=false
```

By default, deploy writes ONNX and TensorRT outputs into the current MLflow
run artifact directory under `exports/`.

When MLflow logging is enabled, any custom `output_dir` must stay inside that
run artifact directory. Leave `output_dir` unset to use the default
`exports/` location, or disable MLflow logging if you need to export outside
the run artifact tree.

**Custom output name:**

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt \
    output_name=model_v1
```

**Custom output directory inside MLflow artifacts:**

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt \
    output_dir=mlruns/<task>/<model>/<config>/<deploy_run_id>/artifacts/custom_exports
```

## Multi-head exports

Multi-head export models can expose multiple deployable modules from one
configured model. PTv3 detection exports the backbone and detection head as
separate modules, and `--weights` can merge a pretrained backbone checkpoint
with a detection checkpoint:

```bash
autoware-ml deploy \
    --config-name detection3d/ptv3/voxel012_122m_t4dataset_j6gen2 \
    --weights mlruns/segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt \
    --weights mlruns/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2/<run_id>/artifacts/checkpoints/best.ckpt
```

Checkpoints are applied in the order they appear on the command line, and
later checkpoints overwrite any keys already set by earlier ones. Each
checkpoint only contributes the state-dict keys that exist on the export
model and match its tensor shapes. Keys missing from the model are skipped;
keys with a matching name but mismatching shape raise an error immediately.

**Full coverage is enforced.** After all checkpoints are loaded, deploy
verifies that every parameter in the export model has been covered by at
least one of the supplied `--weights`. If any parameter is left
uninitialized, the command fails up front with the list of missing keys
instead of producing an ONNX or engine that contains untrained layers. Add
or replace `--weights` entries until every key is covered.

## Configuration

### ONNX Settings

```yaml
deploy:
  onnx:
    enabled: true
    dynamo: true
    opset_version: 21
    input_names: [input]
    output_names: [output]
    dynamic_shapes:
      input_tensor: { 2: height, 3: width }
```

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
derive equivalent symbolic axes.

### TensorRT Settings

```yaml
deploy:
  tensorrt:
    enabled: true
    workspace_size: 1073741824  # 1GB
    input_shapes:
      input:
        min_shape: [1, 3, 224, 224]
        opt_shape: [1, 3, 256, 256]   # Optimized for this
        max_shape: [1, 3, 512, 512]
```

!!! tip
    TensorRT optimizes most aggressively for `opt_shape`. Set this to your typical inference resolution.

## Model-Owned Export Wrappers

The preferred deployment path is to keep export logic inside the model. Models
with deployment-specific requirements should override `build_export_spec()` and
return an explicit export module plus example tensor inputs.

This keeps export-time behavior close to the model implementation and avoids
ad hoc post-processing for most cases.

## Optional Graph Modification

Post-export ONNX graph modification is still available as a fallback:

```yaml
deploy:
  onnx:
    modify_graph:
      _target_: my_module.OnnxGraphModifier
      # modifier-specific parameters
```

Use for operator replacement, shape inference fixes, or custom plugin insertion.

## Overriding at Runtime

Override deployment settings from CLI:

```bash
autoware-ml deploy \
    --config-name <task>/<model>/<config> \
    --weights mlruns/<task>/<model>/<config>/<run_id>/artifacts/checkpoints/best.ckpt \
    deploy.tensorrt.input_shapes.input.opt_shape=[1,3,256,256] \
    deploy.tensorrt.workspace_size=2147483648
```
