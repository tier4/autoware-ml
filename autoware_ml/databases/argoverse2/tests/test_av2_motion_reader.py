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

"""Tests for the readers of the Argoverse 2 Motion Forecasting files."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

if importlib.util.find_spec("av2") is None:
    raise unittest.SkipTest("the av2 devkit is not installed")

from autoware_ml.databases.argoverse2.av2_motion_reader import (  # noqa: E402
    read_agent_tracks,
    read_scenario_metadata,
    read_vector_map,
    step_seconds_of,
)
from autoware_ml.databases.argoverse2.tests.av2_fixtures import (  # noqa: E402
    CITY,
    FOCAL_TRACK_ID,
    av2_prediction_taxonomy,
    write_scenario,
)
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary  # noqa: E402
from autoware_ml.types.motion import AgentRole, PolylineType  # noqa: E402


def _tracks_file(scenario_path: Path) -> Path:
    """Path of the tracks parquet of the scenario directory."""
    return scenario_path / f"scenario_{scenario_path.name}.parquet"


def _map_file(scenario_path: Path) -> Path:
    """Path of the map archive of the scenario directory."""
    return scenario_path / f"log_map_archive_{scenario_path.name}.json"


class AV2MotionReaderTestCase(unittest.TestCase):
    """Base case that writes one synthetic train scenario per test."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.tmp_path = Path(self._tmp_dir.name)
        self.scenario_path = write_scenario(self.tmp_path / "train", "abc")


class TestScenarioMetadata(AV2MotionReaderTestCase):
    """The scenario metadata is read from the tracks file alone."""

    def test_scenario_metadata(self) -> None:
        """
        Input: the tracks file of the synthetic scenario "abc" with 110 steps and three tracks.
        Expected: the metadata names the scenario, its city and focal track, counts the steps
        and tracks, and derives the step period and the timestamps from the time axis.
        Check: every metadata field, the period and the first and fiftieth timestamps.
        """
        metadata = read_scenario_metadata(_tracks_file(self.scenario_path))

        self.assertEqual(metadata.scenario_id, "abc")
        self.assertEqual(metadata.city, CITY)
        self.assertEqual(metadata.focal_track_id, FOCAL_TRACK_ID)
        self.assertEqual(metadata.num_steps, 110)
        self.assertEqual(metadata.num_tracks, 3)
        self.assertAlmostEqual(metadata.step_seconds, 0.1)
        self.assertAlmostEqual(float(metadata.timestamps_seconds[0]), 1.0)
        self.assertAlmostEqual(float(metadata.timestamps_seconds[49]), 5.9)

    def test_step_seconds_rejects_a_non_uniform_axis(self) -> None:
        """
        Input: a time axis with an uneven gap and an axis of a single step.
        Expected: the period is defined for a uniform axis of at least two steps only.
        Check: ValueError naming the uneven spacing and the step count.
        """
        with self.assertRaisesRegex(ValueError, "not uniformly spaced"):
            step_seconds_of(np.array([0.0, 0.1, 0.3]))
        with self.assertRaisesRegex(ValueError, "at least two steps"):
            step_seconds_of(np.array([0.0]))


class TestAgentTracks(AV2MotionReaderTestCase):
    """The agent tracks of a scenario are read into one padded sample."""

    def test_agent_tracks_shapes_and_focal_first(self) -> None:
        """
        Input: the synthetic scenario read with the prediction taxonomy at current step 49.
        Expected: the three tracks are padded to the 110 steps in float32 / int64 / bool
        tensors, the focal track comes first, and Argoverse 2 records no box size.
        Check: the agent and step counts, the tensor shapes and dtypes, the track id order,
        the current step, the period, the ignore index and the absent sizes.
        """
        tracks = read_agent_tracks(_tracks_file(self.scenario_path), av2_prediction_taxonomy(), 49)

        self.assertEqual(tracks.num_agents, 3)
        self.assertEqual(tracks.num_steps, 110)
        self.assertEqual(tracks.positions.shape, (3, 110, 3))
        self.assertEqual(tracks.headings.shape, (3, 110))
        self.assertEqual(tracks.velocities.shape, (3, 110, 2))
        self.assertEqual(tracks.valid.shape, (3, 110))
        self.assertEqual(tracks.positions.dtype, torch.float32)
        self.assertEqual(tracks.agent_types.dtype, torch.int64)
        self.assertEqual(tracks.agent_roles.dtype, torch.int64)
        self.assertEqual(tracks.valid.dtype, torch.bool)
        self.assertEqual(tracks.track_ids[0], FOCAL_TRACK_ID)
        self.assertEqual(set(tracks.track_ids), {FOCAL_TRACK_ID, "walker", "cone"})
        self.assertEqual(tracks.current_step_index, 49)
        self.assertAlmostEqual(tracks.step_seconds, 0.1)
        self.assertEqual(tracks.ignore_index, -1)
        self.assertIsNone(tracks.sizes)

    def test_agent_tracks_values(self) -> None:
        """
        Input: the synthetic scenario, whose focal track moves along x at one meter per step,
        whose walker is observed at the even steps only and whose cone is a one step fragment.
        Expected: the observed states land at their steps with their heading and velocity,
        the unobserved steps are invalid zeros, the types are resolved at the coarse level
        with the static cone kept as context under the ignore index, and the roles follow
        the file.
        Check: the valid masks, sampled positions, velocities and headings, and the type and
        role of every track.
        """
        tracks = read_agent_tracks(_tracks_file(self.scenario_path), av2_prediction_taxonomy(), 49)
        walker = tracks.track_ids.index("walker")
        cone = tracks.track_ids.index("cone")

        self.assertTrue(tracks.valid[0].all().item())
        self.assertEqual(tracks.positions[0, 7].tolist(), [7.0, 0.0, 0.0])
        self.assertEqual(tracks.velocities[0, 7].tolist(), [10.0, 0.0])
        self.assertEqual(tracks.headings[0, 7].item(), 0.0)
        self.assertTrue(tracks.valid[walker, ::2].all().item())
        self.assertFalse(tracks.valid[walker, 1::2].any().item())
        self.assertEqual(tracks.positions[walker, 1].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(tracks.positions[walker, 8].tolist(), [0.0, 8.0, 0.0])
        self.assertAlmostEqual(tracks.headings[walker, 8].item(), np.pi / 2, places=5)
        self.assertEqual(tracks.valid[cone].sum().item(), 1)
        self.assertEqual(tracks.agent_types[0].item(), 0)
        self.assertEqual(tracks.agent_types[walker].item(), 1)
        self.assertEqual(tracks.agent_types[cone].item(), -1)
        self.assertEqual(tracks.agent_roles[0].item(), AgentRole.FOCAL)
        self.assertEqual(tracks.agent_roles[walker].item(), AgentRole.SCORED)
        self.assertEqual(tracks.agent_roles[cone].item(), AgentRole.TRACK_FRAGMENT)

    def test_agent_tracks_of_a_test_scenario(self) -> None:
        """
        Input: a test split scenario of 50 steps, the observed window only.
        Expected: the reader takes the step count from the file, so a scenario without the
        future reads as 50 fully valid steps of the focal track.
        Check: the step count and the valid mask of the focal track.
        """
        scenario_path = write_scenario(self.tmp_path / "test", "t00", num_steps=50)

        tracks = read_agent_tracks(_tracks_file(scenario_path), av2_prediction_taxonomy(), 49)

        self.assertEqual(tracks.num_steps, 50)
        self.assertTrue(tracks.valid[0].all().item())

    def test_agent_tracks_rejects_a_current_step_outside_the_window(self) -> None:
        """
        Input: the 110 step scenario read with current step 110.
        Expected: the current step must index a step of the window.
        Check: ValueError naming the window bounds.
        """
        with self.assertRaisesRegex(ValueError, "current_step_index must lie in \\[0, 110\\)"):
            read_agent_tracks(_tracks_file(self.scenario_path), av2_prediction_taxonomy(), 110)

    def test_agent_tracks_rejects_an_unlisted_object_type(self) -> None:
        """
        Input: a scenario whose focal track is a bus, read with a taxonomy whose vocabulary
        does not list buses.
        Expected: an object type the vocabulary does not know is refused rather than mapped
        to the ignore index silently.
        Check: KeyError naming the unlisted type.
        """
        scenario_path = write_scenario(
            self.tmp_path / "train", "odd", object_types={FOCAL_TRACK_ID: "bus"}
        )
        taxonomy = av2_prediction_taxonomy()
        vocabulary = dict(taxonomy.vocabulary.class_renaming)
        vocabulary.pop("bus")
        narrowed = LabelTaxonomy(
            vocabulary=LabelVocabulary(vocabulary),
            class_names=taxonomy.class_names,
            class_mapping={k: v for k, v in taxonomy.class_mapping.items() if k != "bus"},
            ignore_index=-1,
        )

        with self.assertRaisesRegex(KeyError, "'bus' is not listed in the vocabulary"):
            read_agent_tracks(_tracks_file(scenario_path), narrowed, 49)


class TestVectorMap(AV2MotionReaderTestCase):
    """The vector map of a scenario is read into resampled polylines."""

    def test_vector_map_polylines(self) -> None:
        """
        Input: the synthetic map with one lane, one crossing and one drivable area, resampled
        to 5 points per polyline.
        Expected: 3 polylines per lane, 2 per crossing and 1 per drivable area, each resampled
        to 5 equally spaced float32 points, typed and tagged with its element id and
        intersection flag, with the centerline as the mean of the resampled boundaries and
        the drivable area boundary closed.
        Check: the counts, shapes and dtype, the type, id and intersection lists, the sampled
        coordinates of the lane and the crossing, and the first and last boundary point.
        """
        polylines = read_vector_map(_map_file(self.scenario_path), num_points_per_polyline=5)

        self.assertEqual(polylines.num_polylines, 6)
        self.assertEqual(polylines.num_points_per_polyline, 5)
        self.assertEqual(polylines.points.shape, (6, 5, 3))
        self.assertEqual(polylines.points.dtype, torch.float32)
        self.assertTrue(polylines.valid.all().item())
        self.assertEqual(
            polylines.polyline_types.tolist(),
            [
                PolylineType.LANE_CENTERLINE,
                PolylineType.LANE_LEFT_BOUNDARY,
                PolylineType.LANE_RIGHT_BOUNDARY,
                PolylineType.CROSSWALK_EDGE,
                PolylineType.CROSSWALK_EDGE,
                PolylineType.DRIVABLE_AREA_BOUNDARY,
            ],
        )
        self.assertEqual(polylines.element_ids, ["10", "10", "10", "20", "20", "30"])
        self.assertEqual(
            polylines.is_intersection.tolist(), [True, True, True, False, False, False]
        )
        centerline = polylines.points[0]
        np.testing.assert_allclose(centerline[:, 0], [0.0, 25.0, 50.0, 75.0, 100.0], atol=1e-4)
        np.testing.assert_allclose(centerline[:, 1], [0.0] * 5, atol=1e-4)
        np.testing.assert_allclose(polylines.points[1][:, 1], [2.0] * 5, atol=1e-4)
        np.testing.assert_allclose(polylines.points[2][:, 1], [-2.0] * 5, atol=1e-4)
        np.testing.assert_allclose(
            polylines.points[3][:, 1], [-5.0, -2.5, 0.0, 2.5, 5.0], atol=1e-4
        )
        self.assertEqual(polylines.points[5][0].tolist(), polylines.points[5][-1].tolist())

    def test_vector_map_filters_polyline_types(self) -> None:
        """
        Input: the synthetic map read with the centerline and crosswalk edge types only.
        Expected: the other polylines are left out and the kept ones keep the file order.
        Check: the type list and the points shape.
        """
        polylines = read_vector_map(
            _map_file(self.scenario_path),
            num_points_per_polyline=4,
            polyline_types=[PolylineType.LANE_CENTERLINE, PolylineType.CROSSWALK_EDGE],
        )

        self.assertEqual(polylines.polyline_types.tolist(), [0, 3, 3])
        self.assertEqual(polylines.points.shape, (3, 4, 3))

    def test_vector_map_without_kept_polylines_is_empty(self) -> None:
        """
        Input: the synthetic map read with an empty list of polyline types.
        Expected: the sample is empty but keeps the point dimension of the default
        resampling, so a batch can still stack it.
        Check: the polyline count, the shapes of every tensor and the empty id list.
        """
        polylines = read_vector_map(_map_file(self.scenario_path), polyline_types=[])

        self.assertEqual(polylines.num_polylines, 0)
        self.assertEqual(polylines.points.shape, (0, 20, 3))
        self.assertEqual(polylines.valid.shape, (0, 20))
        self.assertEqual(polylines.polyline_types.shape, (0,))
        self.assertEqual(polylines.element_ids, [])

    def test_vector_map_rejects_too_few_points(self) -> None:
        """
        Input: the synthetic map read with one point per polyline.
        Expected: a polyline needs at least two points to be resampled.
        Check: ValueError naming the minimum.
        """
        with self.assertRaisesRegex(ValueError, "num_points_per_polyline must be at least 2"):
            read_vector_map(_map_file(self.scenario_path), num_points_per_polyline=1)
