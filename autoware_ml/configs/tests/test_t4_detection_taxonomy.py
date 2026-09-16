"""Tests for the bundled T4 detection taxonomy."""

from __future__ import annotations

import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.transforms.boxes3d.annotations import resolve_detection_class

T4_DETECTION_CONFIG = "tasks/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2"


def _t4_detection_taxonomy() -> dict:
    register_config_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        cfg = compose(config_name=T4_DETECTION_CONFIG)
    taxonomy = cfg.t4dataset.detection3d
    return {
        key: OmegaConf.to_container(taxonomy[key], resolve=True)
        for key in ("class_names", "name_mapping", "merge_objects")
    }


def test_every_mapped_target_is_a_detection_class_or_a_merge_source() -> None:
    # A mapping onto a name that is neither trained nor consumed by a merge rule
    # would drop every box of that source label without a word.
    taxonomy = _t4_detection_taxonomy()
    merge_names = {name for _, members in taxonomy["merge_objects"] for name in members}
    allowed = set(taxonomy["class_names"]) | merge_names
    targets = {name for name in taxonomy["name_mapping"].values() if name is not None}
    assert targets <= allowed, sorted(targets - allowed)


@pytest.mark.parametrize("raw_name", ["static_object.bicycle_rack", "static_object.bicycle rack"])
def test_bicycle_rack_source_labels_resolve(raw_name: str) -> None:
    # Both spellings the T4 corpora use for bicycle racks must reach the class.
    taxonomy = _t4_detection_taxonomy()
    resolved = resolve_detection_class(
        {"gt_nusc_name": raw_name},
        class_names=taxonomy["class_names"],
        name_mapping=taxonomy["name_mapping"],
    )
    assert resolved == "bicycle_rack"
