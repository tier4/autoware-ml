"""Tests for the bundled T4 detection taxonomy."""

from __future__ import annotations

import hydra
import pytest
from hydra import compose, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from autoware_ml.configs.resolvers import register_config_resolvers
from autoware_ml.databases.taxonomy import DatabaseTaxonomy

T4_DETECTION_CONFIG = "tasks/detection3d/ptv3/voxel012_122m_t4dataset_j6gen2"


def _t4_taxonomy() -> DatabaseTaxonomy:
    register_config_resolvers()
    GlobalHydra.instance().clear()
    with initialize_config_module(version_base=None, config_module="autoware_ml.configs"):
        cfg = compose(config_name=T4_DETECTION_CONFIG)
    return hydra.utils.instantiate(cfg.database.taxonomy)


def test_every_raw_name_of_the_vocabulary_resolves() -> None:
    # Every fine name is either trained or dropped on purpose, none falls to the ignore index.
    taxonomy = _t4_taxonomy().detection3d

    for fine_name in taxonomy.vocabulary.fine_names:
        assert fine_name in taxonomy.class_mapping


@pytest.mark.parametrize("raw_name", ["static_object.bicycle_rack", "static_object.bicycle rack"])
def test_bicycle_rack_source_labels_resolve(raw_name: str) -> None:
    # Both bicycle rack spellings map to one fine name. Detection drops it, segmentation keeps
    # it as manmade.
    taxonomy = _t4_taxonomy()

    assert taxonomy.detection3d.fine_name(raw_name) == "bicycle_rack"
    assert taxonomy.detection3d.class_name("bicycle_rack") is None
    assert taxonomy.segmentation3d.class_name("bicycle_rack") == "manmade"
