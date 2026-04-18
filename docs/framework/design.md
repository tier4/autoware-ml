---
icon: lucide/workflow
---

# Framework Design

Autoware-ML is built on a modular architecture that separates concerns into key components including configuration, data handling, model definition, training, and deployment, among others. This design makes it easy to add new models and datasets while reusing common infrastructure.

## Architecture Overview

```mermaid
flowchart TB
    subgraph Optimization [Hyperparameter Tuning]
        Optuna[Optuna]
    end

    subgraph Configuration [Configuration Layer]
        YAML[YAML Configs]
        Optuna --> Hydra[Hydra]
        YAML --> Hydra
    end

    subgraph TrainingPipeline [Training Pipeline]
        InfoFiles[Info Files]
        InfoFiles --> LightningDataModule[Lightning Data Module]
        LightningDataModule --> Transforms[Transforms]
        Transforms --> Collation[Collation]
        Collation --> BatchTransfer[Batch Transfer]
        BatchTransfer --> Preprocessing[Preprocessing]
        Preprocessing --> ForwardPass[Forward Pass]
        ForwardPass --> LossComputation[Loss Computation]
        LossComputation --> BackwardPass[Backward Pass]
    end

    subgraph ModelLayer [Model Definition]
        LightningModule[Lightning Module]
        LightningModule --> Blocks["Blocks"]
        LightningModule --> Optimizers["Optimizers"]
        LightningModule --> Schedulers["Schedulers"]
    end

    subgraph TrainingLoop [Training Orchestration]
        Trainer[Lightning Trainer]
        Trainer --> CustomCallbacks[Custom Callbacks]
        Trainer --> MLflow[MLflow Logger]
        Trainer --> Checkpoints[Checkpoints]
    end

    subgraph Deployment [Deployment Pipeline]
        ModelWeights[Model Weights]
        ModelWeights --> ONNXExport[ONNX Export]
        ONNXExport --> TensorRTEngine[TensorRT Engine]
    end

    Hydra --> LightningDataModule
    Hydra --> LightningModule
    Hydra --> Trainer
    Hydra --> ModelWeights

    style ModelWeights fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style ONNXExport fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style TensorRTEngine fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style Blocks fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style BatchTransfer fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style Preprocessing fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style ForwardPass fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style LossComputation fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style BackwardPass fill:#a5d6a7,opacity:0.2,stroke:#05bc23
```

**Legend:** <span style="display: inline-block; width: 12px; height: 12px; background-color: #42a5f5; border: 1px solid #1976d2; margin-right: 4px; vertical-align: middle;"></span> CPU operations | <span style="display: inline-block; width: 12px; height: 12px; background-color: #66bb6a; border: 1px solid #388e3c; margin-right: 4px; vertical-align: middle;"></span> GPU operations

## Core Components

### Configuration (Hydra)

Everything in Autoware-ML is configured through YAML files processed by [Hydra](https://hydra.cc/). This enables:

- **Hierarchical configs** - Inherit from base configs, override specific values
- **Runtime overrides** - Change any parameter from the command line
- **Automatic instantiation** - `_target_` keys specify Python classes to instantiate via `hydra.utils.instantiate()`

See [Configuration Guide](../user-guide/configuration.md) for full details on Hydra syntax.

### Data Module

The `DataModule` class (extending `LightningDataModule`) manages:

- Dataset creation for each split (train/val/test/predict)
- DataLoader configuration (batch size, workers, shuffling, pin_memory, etc.)
- Transforms (CPU-side augmentations per split)
- Collation (batching samples together, stacking selected keys)
- Preprocessing (GPU-side transforms after batch transfer)

```python
class DataModule(L.LightningDataModule, ABC):
    def __init__(
        self,
        stack_keys: Sequence[str] | None = None,
        train_transforms: TransformsCompose | None = None,
        val_transforms: TransformsCompose | None = None,
        test_transforms: TransformsCompose | None = None,
        predict_transforms: TransformsCompose | None = None,
        data_preprocessing: DataPreprocessing | None = None,
        train_dataloader_cfg: DataLoaderConfig | None = None,
    ):
        ...

    @abstractmethod
    def _create_dataset(
        self, split: str, transforms: TransformsCompose | None = None
    ) -> Dataset:
        ...

    def collate_fn(self, batch_inputs_dicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
        ...

    def on_after_batch_transfer(
        self, batch_inputs_dict: dict[str, Any], dataloader_idx: int
    ) -> dict[str, Any]:
        if self.data_preprocessing is not None:
            batch_inputs_dict = self.data_preprocessing(batch_inputs_dict)
        return batch_inputs_dict
```

The `Dataset` base class handles transforms application:

```python
class Dataset(TorchDataset, ABC):
    def __getitem__(self, index: int) -> dict[str, Any]:
        input_dict = self.get_data_info(index)
        context = PipelineContext(dataset=self, index=index)
        return self.apply_transforms(input_dict, self.dataset_transforms, context)

    @abstractmethod
    def get_data_info(self, index: int) -> dict[str, Any]:
        ...
```

Datasets are expected to return metadata records. File loading and sample
materialization should happen in transforms.

### Transforms

Transforms are composable data augmentations applied per-sample on CPU. They follow a dict-in/dict-out pattern where each transform receives a dictionary and returns updates to merge back.

```python
class BaseTransform(ABC):
    def __call__(
        self,
        input_dict: dict[str, Any],
        context: PipelineContext | None = None,
    ) -> dict[str, Any]:
        self._context = context           # accessible via self.context property
        self._validate_required_keys(input_dict)
        self._handle_optional_keys(input_dict)
        if not self._should_apply():
            return self.on_skip(input_dict)
        return self.transform(input_dict)

    @abstractmethod
    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        ...

class TransformsCompose:
    def __init__(self, pipeline: Sequence[BaseTransform] | None = None):
        self.pipeline = pipeline or []

    def __call__(
        self,
        input_dict: dict[str, Any],
        context: PipelineContext | None = None,
    ) -> dict[str, Any]:
        for transform in self.pipeline:
            input_dict |= transform(input_dict, context=context)
        return input_dict
```

Transforms are configured per split (train/val/test/predict) in the `DataModule` and applied during `Dataset.__getitem__()`.

Public transform targets should be referenced through their subpackage API, for
example `autoware_ml.transforms.point_cloud.loading.LoadPointsFromFile` or
`autoware_ml.transforms.point_cloud.scene.RandomFlip3D`. Implementation modules should
define the transform once, and the subpackage `__init__.py` acts as the single
public re-export layer.

### Preprocessing

Preprocessing runs on GPU after batch transfer, enabling hardware-accelerated operations. It follows the same dict-in/dict-out pattern as transforms but operates on batched tensors.

```python
class DataPreprocessing(nn.Module):
    def __init__(self, pipeline: Sequence[nn.Module] | None = None):
        super().__init__()
        self.pipeline = nn.ModuleList(pipeline or [])

    def __call__(self, batch_inputs_dict: dict[str, Any]) -> dict[str, Any]:
        for layer in self.pipeline:
            batch_inputs_dict |= layer(batch_inputs_dict)
        return batch_inputs_dict
```

The `DataModule.on_after_batch_transfer()` method automatically applies preprocessing if configured. Each preprocessing layer is an `nn.Module` that operates on batched tensors already on the target device.

### Model

All supported models inherit from `BaseModel` (extending `LightningModule`),
which provides a standard interface and a set of override hooks for
task-specific behavior:

```python
class BaseModel(L.LightningModule, ABC):
    def __init__(
        self,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
    ):
        super().__init__()
        self.forward_signature = inspect.signature(self.forward)
        self.compute_metrics_signature = inspect.signature(self.compute_metrics)
        ...

    @abstractmethod
    def forward(self, **kwargs: Any) -> torch.Tensor | Sequence[torch.Tensor]:
        ...

    @abstractmethod
    def compute_metrics(
        self, outputs: torch.Tensor | Sequence[torch.Tensor], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        ...

    def run_model(self, batch_inputs_dict: Mapping[str, Any]) -> Any:
        ...

    def predict_outputs(self, outputs: Any) -> Any:
        ...

    def build_export_spec(self, batch_inputs_dict: Mapping[str, Any]) -> ExportSpec:
        ...

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        ...
```

The base class handles:

- **Unified step logic** - All models share the same step logic for training, validation, test and predict steps
- **Automatic signature inspection** - Only passes relevant kwargs to `forward()` and `compute_metrics()` based on method signatures captured at initialization
- **Metric logging** - Logs metrics to Lightning's logger with proper prefixes
- **Predict step** - Handles prediction without computing metrics
- **Export contract** - Supports a generic forward-signature-based export path and model-owned explicit export wrappers

Models can have **any internal architecture**. The default path filters batch
inputs to match method signatures using `inspect.signature()`, while more
specialized models can override hooks such as `run_model()`,
`prepare_metric_inputs()`, `get_log_batch_size()`, or `build_export_spec()`
without leaving the shared framework contract.

!!! note
    When a model relies on the default signature-based path, `forward()` and
    `compute_metrics()` argument names must match keys in the batch dictionary.
    Models with more specialized batching or export requirements should
    override the relevant hooks instead of bypassing `BaseModel`.

### Deployment Pipeline

The deployment pipeline exports trained models to production-ready formats:

```mermaid
flowchart LR
    subgraph ONNXExport [ONNX Export]
        Checkpoint[Checkpoint] --> Load[Load Weights]
        Load --> Model[Model Eval Mode]
        Model --> Trace[Trace with Sample]
        Trace --> ONNX[ONNX File]
    end

    subgraph TensorRTBuild [TensorRT Build]
        ONNX --> Parse[Parse ONNX]
        Parse --> Optimize[Build Engine]
        Optimize --> EngineFile[Engine File]
    end

    style Checkpoint fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style Load fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style Model fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style Trace fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style ONNX fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style Parse fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style Optimize fill:#a5d6a7,opacity:0.2,stroke:#05bc23
    style EngineFile fill:#a5d6a7,opacity:0.2,stroke:#05bc23
```

The deployment process:

1. **Load checkpoint** - Instantiates model from config and loads weights from checkpoint
2. **Get input sample** - Uses the predict dataloader to obtain a preprocessed sample for deployment
3. **Resolve export spec** - Builds the effective export module and example inputs through the model's `build_export_spec()` contract
4. **Export to ONNX** - Traces the resolved export module, supporting dynamic shapes for variable input sizes
5. **Build TensorRT engine** - Optimizes the ONNX model for inference on NVIDIA GPUs with configurable optimization profile

Configuration is done through the `deploy` section in task configs.

## Extending the Framework

| Extension Point | How                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------- |
| New model       | Subclass `BaseModel`, implement `forward()` and `compute_metrics()`, override hooks as needed |
| New dataset     | Subclass `DataModule` and `Dataset`                                                           |
| New transform   | Subclass `BaseTransform`, implement `transform()`                                             |
| New task        | Create config in `configs/tasks/`                                                             |

See [Adding Models](../contributing/adding-models.md) for a detailed guide.
