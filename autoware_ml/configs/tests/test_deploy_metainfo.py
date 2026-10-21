"""Composition tests for the per-module deploy metainfo blocks."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.models.segmentation3d.ptv3_base import PTv3BaseModel
from autoware_ml.utils.deploy import merge_module_onnx_cfg

PTV3_CONFIGS = [
    "tasks/multi/ptv3/t4dataset/voxel012_tiny_b8_100e_t4dataset_122m_j6gen2_det3d_seg3d",
    "tasks/multi/litept/t4dataset/voxel012_tiny_b8_100e_t4dataset_122m_j6gen2_det3d_seg3d",
    "tasks/segmentation3d/ptv3/t4dataset/voxel012_tiny_b16_50e_t4dataset_122m_j6gen2_semaseg",
    "tasks/segmentation3d/litept/t4dataset/voxel012_tiny_b16_50e_t4dataset_122m_j6gen2_semaseg",
    "tasks/detection3d/ptv3/t4dataset/voxel012_tiny_b8_30e_t4dataset_122m_j6gen2_base",
    "tasks/multi/ptv3/nuscenes/voxel005_base_b16_100e_nuscenes_51m",
    "tasks/segmentation3d/ptv3/nuscenes/voxel005_base_b16_50e_nuscenes_51m",
    "tasks/detection3d/ptv3/nuscenes/voxel005_base_b16_30e_nuscenes_51m",
]
DET3D_CONFIGS = [name for name in PTV3_CONFIGS if "segmentation3d" not in name]


def _compose(config_name: str):
    register_config_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return cfg


@pytest.mark.parametrize("config_name", PTV3_CONFIGS)
def test_encoder_metainfo_serialization_orders_match_export_order(config_name: str) -> None:
    # The encoder metainfo declares the orders the exported graph is built with;
    # config drift against the code constant must fail loud, never ship.
    cfg = _compose(config_name)
    module_cfg = merge_module_onnx_cfg(cfg.deploy.onnx, "ptv3_encoder")
    assert list(module_cfg.metainfo.serialization_orders) == list(PTv3BaseModel.EXPORT_ORDER)


@pytest.mark.parametrize("config_name", PTV3_CONFIGS)
def test_metainfo_interpolations_resolve_for_every_module(config_name: str) -> None:
    # Every metainfo value must resolve through the production merge path; a
    # broken interpolation must fail at composition, not at deploy time.
    cfg = _compose(config_name)
    for module_name in cfg.deploy.onnx.modules:
        module_cfg = merge_module_onnx_cfg(cfg.deploy.onnx, module_name)
        metainfo = OmegaConf.to_container(module_cfg.metainfo, resolve=True)
        assert metainfo, f"{config_name}: {module_name} has an empty metainfo block"
        if module_name in ("ptv3_seg3d_head", "ptv3_det3d_head"):
            assert len(metainfo["class_names"]) > 0


@pytest.mark.parametrize("config_name", DET3D_CONFIGS)
def test_det3d_head_has_twist_follows_use_velocity(config_name: str) -> None:
    # The runtime learns from has_twist whether a velocity output exists, so it
    # is derived from the head configuration instead of being declared twice.
    cfg = _compose(config_name)
    module_cfg = merge_module_onnx_cfg(cfg.deploy.onnx, "ptv3_det3d_head")
    assert isinstance(module_cfg.metainfo.has_twist, bool)
    assert module_cfg.metainfo.has_twist == cfg.model.bbox_head.use_velocity


@pytest.mark.parametrize("config_name", DET3D_CONFIGS)
def test_det3d_head_exports_fused_bf16_attention(config_name: str) -> None:
    # The fused TransHead path needs the head flag and the fp16 module precision
    # together; export refuses a mismatch, so every shipped config must pair them.
    cfg = _compose(config_name)
    module_cfg = merge_module_onnx_cfg(cfg.deploy.onnx, "ptv3_det3d_head")
    assert cfg.model.bbox_head.use_bf16_cross_attention is True
    assert module_cfg.precision == "fp16"
