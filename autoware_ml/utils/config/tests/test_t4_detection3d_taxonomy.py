"""Tests for the bundled T4 taxonomies."""

from __future__ import annotations

import unittest

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.utils.config.resolvers import register_config_resolvers
from autoware_ml.utils.config.settings import CONFIGS_ROOT, Settings

T4_MULTI_CONFIG = "tasks/multi/ptv3/t4dataset/voxel012_b8_100e_t4dataset_122m"


def _t4_taxonomies() -> dict[str, LabelTaxonomy]:
    register_config_resolvers(Settings())
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS_ROOT)):
        cfg = compose(config_name=T4_MULTI_CONFIG, return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    return {task: hydra.utils.instantiate(node) for task, node in cfg.taxonomy.items()}


class TestT4Taxonomies(unittest.TestCase):
    def test_every_fine_name_of_the_vocabulary_is_mapped(self) -> None:
        # Every fine name is either trained or dropped on purpose, none falls to the ignore index.
        taxonomy = _t4_taxonomies()["detection3d"]

        for fine_name in taxonomy.vocabulary.fine_names:
            with self.subTest(fine_name=fine_name):
                self.assertIn(fine_name, taxonomy.class_mapping)

    def test_bicycle_rack_source_labels_resolve(self) -> None:
        # Both bicycle rack spellings map to one fine name. Detection drops it, segmentation keeps
        # it as manmade.
        taxonomies = _t4_taxonomies()

        for raw_name in ("static_object.bicycle_rack", "static_object.bicycle rack"):
            with self.subTest(raw_name=raw_name):
                self.assertEqual(taxonomies["detection3d"].fine_name(raw_name), "bicycle_rack")
        self.assertIsNone(taxonomies["detection3d"].class_name("bicycle_rack"))
        self.assertEqual(taxonomies["segmentation3d"].class_name("bicycle_rack"), "manmade")


if __name__ == "__main__":
    unittest.main()
