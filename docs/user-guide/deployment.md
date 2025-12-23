---
icon: lucide/package
---

# Deployment

Autoware-ML exports trained models for production use. The pipeline converts PyTorch checkpoints to optimized inference formats.

## Deployment Pipeline

```text
Checkpoint (.ckpt) → ONNX (.onnx) → TensorRT (.engine)
```

## Basic Usage

```bash
autoware-ml deploy \
    --config-name my_task/my_model \
    +checkpoint=path/to/checkpoint.ckpt
```

This generates ONNX (`.onnx`) and TensorRT (`.engine`) files.

**Custom output:**

```bash
autoware-ml deploy \
    --config-name my_task/my_model \
    +checkpoint=path/to/checkpoint.ckpt \
    output_dir=./deployed \
    output_name=model_v1
```

## Configuration

### ONNX Settings

```yaml
deploy:
  onnx:
    opset_version: 21
    input_names: [input]
    output_names: [output]
    dynamic_shapes:
      input_tensor: { 2: height, 3: width }
```

**dynamic_shapes**: Keys are parameter names from `forward()`, values map dimension indices to symbolic names.

### TensorRT Settings

```yaml
deploy:
  tensorrt:
    workspace_size: 1073741824  # 1GB
    input_shapes:
      input:
        min_shape: [1, 3, 224, 224]
        opt_shape: [1, 3, 256, 256]   # Optimized for this
        max_shape: [1, 3, 512, 512]
```

!!! tip
    TensorRT optimizes most aggressively for `opt_shape`. Set this to your typical inference resolution.

## Graph Modification

Some models require post-export ONNX graph modifications for TensorRT compatibility:

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
    --config-name my_task/my_model \
    +checkpoint=path/to/checkpoint.ckpt \
    deploy.tensorrt.input_shapes.input.opt_shape=[1,3,256,256] \
    deploy.tensorrt.workspace_size=2147483648
```
