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

"""Tests for the loading transforms of the motion task."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from autoware_ml.dataclasses.batch.motion import MotionSourceSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.databases.argoverse2.tests.av2_fixtures import (
    FOCAL_TRACK_ID,
    av2_motion_taxonomy,
    write_scenario,
)
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.transforms.motion.loading import LoadAgentTracks, LoadVectorMap
from autoware_ml.types.dataset import DatasetFormat
from autoware_ml.types.motion import PolylineType

# The transforms import the av2 devkit lazily; only the tests that read a scenario need it
_AV2_AVAILABLE = importlib.util.find_spec("av2") is not None


def _source(scenario_path: Path, with_map: bool = True) -> MotionSourceSample:
    """Source sample of the scenario directory, with or without its map file."""
    scenario_id = scenario_path.name
    return MotionSourceSample(
        dataset_format=DatasetFormat.AV2,
        tracks_path=str(scenario_path / f"scenario_{scenario_id}.parquet"),
        map_path=str(scenario_path / f"log_map_archive_{scenario_id}.json") if with_map else None,
        scenario_key=None,
        focal_track_id=FOCAL_TRACK_ID,
        first_step_index=0,
        current_step_index=49,
        num_steps=110,
        step_seconds=0.1,
        num_tracks=3,
    )


def _sample(source: MotionSourceSample | None) -> ModelGTSample:
    """GT sample that carries the source sample and nothing else."""
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=None,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=None,
        segmentation3d_gt_sample=None,
        motion_source_sample=source,
    )


class TestMotionLoadingArguments(unittest.TestCase):
    """The transforms check their arguments and inputs before any file is read."""

    def test_load_agent_tracks_requires_the_source_sample(self) -> None:
        """
        Input: a GT sample without a motion source sample.
        Expected: the transform has nothing to load from and says which field is missing.
        Check: KeyError naming motion_source_sample.
        """
        with self.assertRaisesRegex(KeyError, "motion_source_sample"):
            LoadAgentTracks(av2_motion_taxonomy())(_sample(None))

    def test_load_agent_tracks_requires_a_label_taxonomy(self) -> None:
        """
        Input: the transform built with a vocabulary instead of a label taxonomy.
        Expected: the agent types are resolved through a taxonomy, so a vocabulary alone is
        refused at construction.
        Check: TypeError naming LabelTaxonomy.
        """
        with self.assertRaisesRegex(TypeError, "must be a LabelTaxonomy"):
            LoadAgentTracks(av2_motion_taxonomy().vocabulary)

    def test_load_vector_map_rejects_too_few_points_and_unknown_types(self) -> None:
        """
        Input: the map transform built with one point per polyline, and with a polyline type
        name that does not exist.
        Expected: both settings are checked at construction.
        Check: ValueError naming the minimum and the unknown type.
        """
        with self.assertRaisesRegex(ValueError, "at least 2"):
            LoadVectorMap(num_points_per_polyline=1)
        with self.assertRaisesRegex(ValueError, "Unknown polyline type 'river'"):
            LoadVectorMap(polyline_types=["river"])


@unittest.skipUnless(_AV2_AVAILABLE, "the av2 devkit is not installed")
class TestMotionLoading(unittest.TestCase):
    """The transforms load the tracks and the map of a synthetic Argoverse 2 scenario."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.scenario_path = write_scenario(Path(self._tmp_dir.name) / "train", "abc")

    def test_load_agent_tracks_fills_the_gt_sample(self) -> None:
        """
        Input: a GT sample whose source names the synthetic scenario.
        Expected: the tracks land in the motion GT sample without a map, with the focal
        track first and the current step of the source, and the source sample is kept.
        Check: the GT sample, its map, the positions shape, the first track id, the current
        step and the source sample.
        """
        loaded = LoadAgentTracks(av2_motion_taxonomy())(_sample(_source(self.scenario_path)))

        gt_sample = loaded.motion_gt_sample
        self.assertIsNotNone(gt_sample)
        self.assertIsNone(gt_sample.map_polylines)
        self.assertEqual(gt_sample.agents.positions.shape, (3, 110, 3))
        self.assertEqual(gt_sample.agents.track_ids[0], FOCAL_TRACK_ID)
        self.assertEqual(gt_sample.agents.current_step_index, 49)
        self.assertIsNotNone(loaded.motion_source_sample)

    def test_load_vector_map_requires_the_tracks_first(self) -> None:
        """
        Input: the map transform applied to a sample whose tracks are not loaded yet.
        Expected: the map joins the GT sample the tracks create, so it is refused without it.
        Check: KeyError naming motion_gt_sample.
        """
        with self.assertRaisesRegex(KeyError, "motion_gt_sample"):
            LoadVectorMap()(_sample(_source(self.scenario_path)))

    def test_load_vector_map_rejects_a_scenario_without_a_map(self) -> None:
        """
        Input: the tracks loaded from a source without a map path.
        Expected: the map transform cannot serve a scenario that names no map.
        Check: ValueError saying the scenario has no map.
        """
        loaded = LoadAgentTracks(av2_motion_taxonomy())(
            _sample(_source(self.scenario_path, with_map=False))
        )

        with self.assertRaisesRegex(ValueError, "has no map"):
            LoadVectorMap()(loaded)

    def test_the_composed_pipeline_loads_tracks_and_map(self) -> None:
        """
        Input: the two transforms composed, the map one resampling to 5 points and keeping
        the centerline by name and the crosswalk edge by value.
        Expected: the GT sample holds the three agents and the three kept polylines in file
        order, and the pipeline names the kept types in its repr.
        Check: the agent count, the map presence and shape, the type list and the repr.
        """
        pipeline = TransformsCompose(
            [
                LoadAgentTracks(av2_motion_taxonomy()),
                LoadVectorMap(num_points_per_polyline=5, polyline_types=["lane_centerline", 3]),
            ]
        )

        loaded = pipeline(_sample(_source(self.scenario_path)))

        gt_sample = loaded.motion_gt_sample
        self.assertEqual(gt_sample.agents.num_agents, 3)
        self.assertIsNotNone(gt_sample.map_polylines)
        self.assertEqual(gt_sample.map_polylines.points.shape, (3, 5, 3))
        self.assertEqual(
            gt_sample.map_polylines.polyline_types.tolist(),
            [
                PolylineType.LANE_CENTERLINE,
                PolylineType.CROSSWALK_EDGE,
                PolylineType.CROSSWALK_EDGE,
            ],
        )
        self.assertIn("LANE_CENTERLINE", repr(pipeline))

    def test_reloading_the_tracks_keeps_a_loaded_map(self) -> None:
        """
        Input: a sample with tracks and map loaded, run through the tracks transform again.
        Expected: reloading the tracks replaces the agents only and keeps the map object.
        Check: the map of the reloaded sample is the same object as before.
        """
        load_tracks = LoadAgentTracks(av2_motion_taxonomy())
        loaded = LoadVectorMap()(load_tracks(_sample(_source(self.scenario_path))))

        reloaded = load_tracks(loaded)

        self.assertIs(
            reloaded.motion_gt_sample.map_polylines, loaded.motion_gt_sample.map_polylines
        )
