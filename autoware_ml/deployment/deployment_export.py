import logging
from pathlib import Path
from typing import Sequence
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict
from omegaconf import DictConfig, OmegaConf

from autoware_ml.datamodule.data_module import DataModule
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.utils.deploy import (
    resolve_export_specs,
    merge_module_onnx_cfg,
    should_export_stage,
    supports_export_stage,
    export_to_onnx,
    should_modify_graph,
    modify_onnx_graph,
    build_tensorrt_engine,
)
from autoware_ml.utils.mlflow_helpers import get_git_sha
from autoware_ml.utils.onnx_meta import stamp_onnx_meta
from autoware_ml.utils.onnx_precision import (
    convert_onnx_precision,
    resolve_onnx_precision,
    should_convert_precision,
    validate_module_onnx_precision,
)

logger = logging.getLogger(__name__)


class DeploymentExportOutputs(BaseModel):
    """Pydantic model for deployment export outputs."""

    model_config: ConfigDict = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    onnx_exported_paths: Sequence[Path]
    tensorrt_exported_paths: Sequence[Path]


class DeploymentExport:
    """Class for exporting a trained model for deployment."""

    def __init__(
        self,
        deploy_cfg: DictConfig,
        output_dir: Path,
        datamodule: DataModule,
        model: ModuleBaseModel,
        config_name: str,
        release: str | None,
        tracker: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Initialize the exporter.

        Args:
            deploy_cfg: ``deploy`` section of the Hydra configuration.
            output_dir: Directory that receives the exported ONNX and TensorRT files.
            datamodule: Data module providing the example batch used for tracing.
            model: Trained model whose export modules are traced.
            config_name: User-facing config name stamped into every ONNX module.
            release: ``vMAJOR.MINOR.PATCH`` stamped into every ONNX module, or ``None``
                for an unversioned export.
            tracker: Active experiment tracker name recorded in the stamp, if any.
            run_id: Deploy run id recorded in the stamp, if any.
        """
        self.deploy_cfg = deploy_cfg
        self.output_dir = output_dir
        self.datamodule = datamodule
        self.model = model
        self.config_name = config_name
        self.release = release
        self.tracker = tracker
        self.run_id = run_id
        self.device = self.model.device
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_export_specs(self) -> MappingProxyType[str, dict]:
        """Prepare export specifications for the model."""
        export_specs = resolve_export_specs(
            datamodule=self.datamodule, model=self.model, device=self.device
        )
        return MappingProxyType(export_specs)

    def export(self) -> DeploymentExportOutputs:
        """Export the trained model for deployment."""
        logger.info("Starting model export...")

        export_specs = self.prepare_export_specs()
        export_git_sha = get_git_sha()

        onnx_exported_paths: Sequence[Path] = []
        tensorrt_exported_paths: Sequence[Path] = []

        for module_name, export_spec in export_specs.items():
            module_onnx_cfg = merge_module_onnx_cfg(self.deploy_cfg.onnx, module_name)
            module_onnx_path = self.output_dir / f"{module_name}.onnx"
            module_engine_path = self.output_dir / f"{module_name}.engine"

            if should_export_stage(self.deploy_cfg.onnx):
                if not supports_export_stage(export_spec, "onnx"):
                    raise RuntimeError(
                        f"Module '{module_name}' does not support ONNX export but "
                        "deploy.onnx.enabled=true. Disable the stage or use a supported model."
                    )
                else:
                    validate_module_onnx_precision(export_spec.module, module_onnx_cfg)
                    export_to_onnx(
                        export_spec.module,
                        export_spec.args,
                        module_onnx_cfg,
                        export_spec.input_param_names,
                        export_spec.output_names,
                        export_spec.dynamic_axes,
                        module_onnx_path,
                    )
                    onnx_exported_paths.append(module_onnx_path)

                    modify_graph_cfg = module_onnx_cfg.get("modify_graph", None)
                    if should_modify_graph(modify_graph_cfg):
                        module_onnx_path = modify_onnx_graph(module_onnx_path, modify_graph_cfg)

                    # Precision conversion runs after the graph modifiers so they keep operating
                    # on the fp32 export they were written against.
                    if should_convert_precision(module_onnx_cfg):
                        module_onnx_path = convert_onnx_precision(
                            module_onnx_path, resolve_onnx_precision(module_onnx_cfg)
                        )

                    # Stamp the final ONNX so the shipped file carries its own provenance.
                    metainfo_cfg = module_onnx_cfg.get("metainfo", None)
                    stamp_onnx_meta(
                        module_onnx_path,
                        config_name=self.config_name,
                        module=module_name,
                        release=self.release,
                        export_git_sha=export_git_sha,
                        metainfo=(
                            OmegaConf.to_container(metainfo_cfg, resolve=True)
                            if metainfo_cfg is not None
                            else None
                        ),
                        tracker=self.tracker,
                        run_id=self.run_id,
                    )
                    logger.info(
                        "Stamped %s metadata: release=%s, commit=%s",
                        module_name,
                        self.release or "unversioned",
                        export_git_sha,
                    )

            if should_export_stage(self.deploy_cfg.tensorrt):
                if not supports_export_stage(export_spec, "tensorrt"):
                    raise RuntimeError(
                        f"Module '{module_name}' does not support TensorRT export but "
                        "deploy.tensorrt.enabled=true. Disable the stage or use a supported model."
                    )
                else:
                    if not module_onnx_path.exists():
                        raise FileNotFoundError(
                            f"ONNX file not found: {module_onnx_path}. "
                            "TensorRT export requires a valid ONNX model."
                        )
                    build_tensorrt_engine(module_onnx_path, self.deploy_cfg, module_engine_path)
                    tensorrt_exported_paths.append(module_engine_path)

        # Implement the export logic here
        logger.info("Model export completed.")
        return DeploymentExportOutputs(
            onnx_exported_paths=onnx_exported_paths,
            tensorrt_exported_paths=tensorrt_exported_paths,
        )
