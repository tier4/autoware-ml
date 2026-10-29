# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for building the visualization preview settings from the configuration."""

from __future__ import annotations

import unittest

from omegaconf import OmegaConf

from autoware_ml.builders.visualization_builder import build_preview_config


class TestBuildPreviewConfig(unittest.TestCase):
    """The ``visualization`` group maps onto the preview and the viewer session settings."""

    def test_empty_group_gives_an_automatic_rerun_preview(self) -> None:
        """Without a ``visualization`` group every setting takes its default."""
        config = build_preview_config(OmegaConf.create({}))

        self.assertEqual(config.mode, "auto")
        self.assertEqual(config.split, "test")
        self.assertEqual(config.sample_index, 0)
        self.assertEqual(config.max_samples, 1)
        self.assertIsNone(config.prediction_frequency_hz)
        self.assertEqual(config.device, "auto")
        self.assertFalse(config.point_labels)
        self.assertIsNone(config.class_names)
        self.assertIsNone(config.segmentation_class_names)
        self.assertIsNone(config.detection_class_names)
        self.assertEqual(config.session.backend, "rerun")
        self.assertEqual(config.session.web_port, 9090)
        self.assertEqual(config.session.grpc_port, 9876)
        self.assertTrue(config.session.wait)
        self.assertEqual(config.session.timeline, "frame")
        self.assertFalse(config.session.camera_frustums_visible)
        self.assertFalse(config.session.sync_views)
        self.assertEqual(config.session.sync_port, 9091)

    def test_every_override_reaches_its_setting(self) -> None:
        """Each key of the group sets the preview or session field of the same meaning."""
        config = build_preview_config(
            OmegaConf.create(
                {
                    "visualization": {
                        "mode": "data",
                        "split": "val",
                        "sample_index": 4,
                        "max_samples": 8,
                        "prediction_frequency_hz": 5,
                        "device": "cpu",
                        "point_labels": True,
                        "point_color_mode": "intensity",
                        "class_names": ["unknown"],
                        "segmentation3d_class_names": ["road", "car"],
                        "detection3d_class_names": ["car"],
                        "backend": "noop",
                        "application_id": "preview",
                        "recording_id": "run-1",
                        "web_port": 9091,
                        "grpc_port": 9877,
                        "wait": False,
                        "server_memory_limit": "10%",
                        "timeline": "sample",
                        "camera_frustums_visible": True,
                        "sync_views": True,
                        "sync_port": 9095,
                    }
                }
            )
        )

        self.assertEqual(config.mode, "data")
        self.assertEqual(config.split, "val")
        self.assertEqual(config.sample_index, 4)
        self.assertEqual(config.max_samples, 8)
        self.assertEqual(config.prediction_frequency_hz, 5.0)
        self.assertIsInstance(config.prediction_frequency_hz, float)
        self.assertEqual(config.device, "cpu")
        self.assertTrue(config.point_labels)
        self.assertEqual(config.class_names, ("unknown",))
        self.assertEqual(config.segmentation_class_names, ("road", "car"))
        self.assertEqual(config.detection_class_names, ("car",))
        self.assertEqual(config.session.backend, "noop")
        self.assertEqual(config.session.application_id, "preview")
        self.assertEqual(config.session.recording_id, "run-1")
        self.assertEqual(config.session.web_port, 9091)
        self.assertEqual(config.session.grpc_port, 9877)
        self.assertFalse(config.session.wait)
        self.assertEqual(config.session.server_memory_limit, "10%")
        self.assertEqual(config.session.timeline, "sample")
        self.assertEqual(config.session.point_color_mode, "intensity")
        self.assertTrue(config.session.camera_frustums_visible)
        self.assertTrue(config.session.sync_views)
        self.assertEqual(config.session.sync_port, 9095)

    def test_empty_class_name_lists_keep_the_taxonomy(self) -> None:
        """An empty override is treated as unset, so the database taxonomy names stay."""
        config = build_preview_config(
            OmegaConf.create({"visualization": {"segmentation3d_class_names": []}})
        )

        self.assertIsNone(config.segmentation_class_names)


if __name__ == "__main__":
    unittest.main()
