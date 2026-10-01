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

"""Unit tests for the point cloud loading transforms and their sweep windows."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.batch.segmentation3d import Segmentation3DGTSample
from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample, LidarSourceView
from autoware_ml.types.geometry import PointFieldIndex
from autoware_ml.transforms.point_cloud.loading import (
    LoadMultiSweepPointsFromFile,
    LoadPointsFromFile,
    SweepWindow,
)

NUM_POINTS = 8


class SweepLoadingTestCase(unittest.TestCase):
    """Shared fixtures writing a record of dated lidar frames to disk."""

    # Capture offsets of the stored frames, the sample first, then two past and two future
    # sweeps, so a window has something to choose from on either side
    TIME_OFFSETS = (0.0, 0.05, 0.10, -0.05, -0.10)
    SAMPLE_TIMESTAMP = 1000.0

    def setUp(self) -> None:
        """Write one point file per stored frame and build the sample holding them."""
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.lidar_point_cloud_samples = [
            self.write_frame(index, offset) for index, offset in enumerate(self.TIME_OFFSETS)
        ]
        self.addCleanup(self._directory.cleanup)

    def write_frame(
        self, index: int, time_offset: float, num_features: int = 5
    ) -> LiDARPointCloudSample:
        """Write the point file of one stored frame and return its record row.

        Args:
            index: Position of the frame in the record.
            time_offset: Seconds the frame was captured before the sample.
            num_features: Number of float32 features stored per point.

        Returns:
            LiDARPointCloudSample: Record row of the written frame.
        """
        path = self.root / f"frame_{index}.bin"
        # Every point of a frame carries its own index, so a test can tell the frames apart
        points = np.full((NUM_POINTS, num_features), float(index), dtype=np.float32)
        points.tofile(path)
        return LiDARPointCloudSample(
            point_cloud_path=str(path),
            timestamp=self.SAMPLE_TIMESTAMP - time_offset,
            num_features=num_features,
            intensity_scale=255.0,
            sensor_to_ego_pose_matrix=torch.eye(4, dtype=torch.float32),
            lidar_to_ego_pose_to_global_matrix=torch.eye(4, dtype=torch.float32),
            lidar_sensor_to_lidar_sweep_matrix=torch.eye(4, dtype=torch.float32),
        )

    def build_sample(
        self,
        lidar_point_cloud_samples: list[LiDARPointCloudSample] | None = None,
        with_labels: bool = False,
    ) -> ModelGTSample:
        """Build a sample whose current frame has been loaded already.

        Args:
            lidar_point_cloud_samples: Stored frames of the record, the fixture ones by default.
            with_labels: Whether the current frame carries a semantic label per point.

        Returns:
            ModelGTSample: Sample ready for the multi sweep loader.
        """
        lidar_point_cloud_samples = (
            self.lidar_point_cloud_samples
            if lidar_point_cloud_samples is None
            else lidar_point_cloud_samples
        )
        segmentation3d_gt_sample = (
            Segmentation3DGTSample(
                gt_semantic_mask=torch.zeros(NUM_POINTS, dtype=torch.int64), ignore_index=-1
            )
            if with_labels
            else None
        )
        sample = ModelGTSample(
            lidar_point_cloud_samples=lidar_point_cloud_samples,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=segmentation3d_gt_sample,
        )
        return LoadPointsFromFile(use_dim=(0, 1, 2, 3))(sample)

    @staticmethod
    def window(num: int, selection: str = "nearest", time_lag_range=(0.01, 1.0)) -> SweepWindow:
        """Build a sweep window from the given bounds."""
        return SweepWindow(num=num, time_lag_range=time_lag_range, selection=selection)

    @staticmethod
    def time_lags(sample: ModelGTSample) -> list[float]:
        """Distinct time lags carried by the points of a sample, in order of appearance.

        The lags are stored as float32, so they are rounded back to the millisecond the
        fixture dated the frames with.
        """
        assert sample.point_cloud_data is not None
        lags = sample.point_cloud_data.points[:, 4]
        return [round(lag, 3) for lag in torch.unique_consecutive(lags).tolist()]


class TestSweepWindow(SweepLoadingTestCase):
    """The declared bounds of one side of the sweep window."""

    def test_rejects_a_zero_minimum_lag(self) -> None:
        """
        Input: a window whose smallest eligible distance is zero.
        Expected: lag zero marks the current frame, so a sweep may not use it.
        Check: a ValueError naming the bounds is raised.
        """
        with self.assertRaisesRegex(ValueError, "min time lag"):
            self.window(num=1, time_lag_range=(0.0, 1.0))

    def test_rejects_an_inverted_range(self) -> None:
        """
        Input: a window whose smallest distance exceeds its largest.
        Expected: the range admits nothing, so it is rejected at construction.
        Check: a ValueError naming the bounds is raised.
        """
        with self.assertRaisesRegex(ValueError, "min time lag"):
            self.window(num=1, time_lag_range=(1.0, 0.5))

    def test_rejects_an_empty_window(self) -> None:
        """
        Input: a window appending no sweeps.
        Expected: an unused side is declared by omitting it, not by asking for zero sweeps.
        Check: constructing the window raises.
        """
        with self.assertRaises(ValueError):
            self.window(num=0)


class TestLoadMultiSweepPointsFromFile(SweepLoadingTestCase):
    """Appending the stored sweeps of a record to the current frame."""

    def test_requires_a_window(self) -> None:
        """
        Input: a loader with neither a past nor a future window.
        Expected: appending nothing is the job of the single frame loader, so the
        configuration is rejected.
        Check: a ValueError pointing at LoadPointsFromFile is raised.
        """
        with self.assertRaisesRegex(ValueError, "LoadPointsFromFile"):
            LoadMultiSweepPointsFromFile()

    def test_rejects_a_window_hydra_did_not_build(self) -> None:
        """
        Input: a mapping in place of a SweepWindow, as a config missing its target yields.
        Expected: the loader reads the bounds off the window, so a mapping is rejected at
        construction instead of failing later on a missing attribute.
        Check: a TypeError naming the argument is raised.
        """
        with self.assertRaisesRegex(TypeError, "past must be a SweepWindow"):
            LoadMultiSweepPointsFromFile(past={"num": 1})

    def test_past_window_appends_the_frames_before_the_sample(self) -> None:
        """
        Input: a past window of two sweeps over a record holding two on either side.
        Expected: only the frames captured before the sample are appended, nearest first, so
        every appended point carries a positive lag.
        Check: read the distinct lags in order of appearance.
        """
        loaded = LoadMultiSweepPointsFromFile(past=self.window(num=2))(self.build_sample())

        self.assertEqual(self.time_lags(loaded), [0.0, 0.05, 0.1])

    def test_future_window_appends_the_frames_after_the_sample(self) -> None:
        """
        Input: a future window of two sweeps over the same record.
        Expected: only the frames captured after the sample are appended, nearest first, and
        their lag is negative so a consumer reads the direction off the sign.
        Check: read the distinct lags in order of appearance.
        """
        loaded = LoadMultiSweepPointsFromFile(future=self.window(num=2))(self.build_sample())

        self.assertEqual(self.time_lags(loaded), [0.0, -0.05, -0.1])

    def test_both_windows_put_the_past_before_the_future(self) -> None:
        """
        Input: a past and a future window of one sweep each.
        Expected: the current frame leads the output, then the past sweeps and then the future
        ones, so the leading block stays the labeled frame.
        Check: read the distinct lags in order of appearance.
        """
        loaded = LoadMultiSweepPointsFromFile(past=self.window(num=1), future=self.window(num=1))(
            self.build_sample()
        )

        self.assertEqual(self.time_lags(loaded), [0.0, 0.05, -0.05])

    def test_asking_for_more_sweeps_than_the_record_holds_appends_what_there_is(self) -> None:
        """
        Input: a past window of five sweeps over a record holding two.
        Expected: a scene that ends before the window is filled yields fewer sweeps, nothing
        is duplicated to pad it out.
        Check: only the two stored past frames are appended.
        """
        loaded = LoadMultiSweepPointsFromFile(past=self.window(num=5))(self.build_sample())

        self.assertEqual(self.time_lags(loaded), [0.0, 0.05, 0.1])

    def test_a_frame_outside_the_range_is_unavailable(self) -> None:
        """
        Input: a past window whose largest distance excludes the farther stored frame.
        Expected: a frame outside the range is as unavailable as one the scene does not have,
        so it contributes no points.
        Check: only the nearer past frame is appended.
        """
        loaded = LoadMultiSweepPointsFromFile(past=self.window(num=2, time_lag_range=(0.01, 0.07)))(
            self.build_sample()
        )

        self.assertEqual(self.time_lags(loaded), [0.0, 0.05])

    def test_random_selection_stays_inside_its_own_side(self) -> None:
        """
        Input: a past window of one sweep picked at random, drawn repeatedly.
        Expected: the random baseline varies over the eligible past frames and never reaches
        the future ones.
        Check: every draw appends one of the two stored past lags.
        """
        loader = LoadMultiSweepPointsFromFile(past=self.window(num=1, selection="random"))

        drawn = {self.time_lags(loader(self.build_sample()))[1] for _ in range(20)}

        self.assertTrue(drawn.issubset({0.05, 0.1}))

    def test_rejects_a_stored_sweep_dated_at_the_sample(self) -> None:
        """
        Input: a record whose stored sweep shares the timestamp of the sample.
        Expected: lag zero marks the current frame, so a sweep with that timestamp is rejected.
        Check: a ValueError naming the timestamp clash is raised.
        """
        frames = [self.lidar_point_cloud_samples[0], self.write_frame(1, time_offset=0.0)]

        with self.assertRaisesRegex(ValueError, "timestamp of its current frame"):
            LoadMultiSweepPointsFromFile(past=self.window(num=1))(self.build_sample(frames))

    def test_sweeps_are_moved_into_the_current_lidar_frame(self) -> None:
        """
        Input: a past and a future sweep, each with its own rotation around z and translation
        from the current lidar frame to the sweep frame.
        Expected: every sweep point is moved back by the inverse of its own matrix.
        Check: the coordinates of the past and of the future sweep block.
        """
        past_to_sweep = torch.eye(4, dtype=torch.float32)
        past_to_sweep[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        past_to_sweep[:3, 3] = torch.tensor([0.0, 3.0, 0.0])
        future_to_sweep = torch.eye(4, dtype=torch.float32)
        future_to_sweep[:3, :3] = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))
        future_to_sweep[:3, 3] = torch.tensor([1.0, 0.0, 0.5])
        # Every point of frame 1 sits at (1, 1, 1) and every point of frame 2 at (2, 2, 2)
        frames = [
            self.lidar_point_cloud_samples[0],
            self.write_frame(1, time_offset=0.05)._replace(
                lidar_sensor_to_lidar_sweep_matrix=past_to_sweep
            ),
            self.write_frame(2, time_offset=-0.05)._replace(
                lidar_sensor_to_lidar_sweep_matrix=future_to_sweep
            ),
        ]

        loaded = LoadMultiSweepPointsFromFile(past=self.window(num=1), future=self.window(num=1))(
            self.build_sample(frames)
        )

        assert loaded.point_cloud_data is not None
        coordinates = loaded.point_cloud_data.points[:, :3]
        past_block = coordinates[NUM_POINTS : 2 * NUM_POINTS]
        future_block = coordinates[2 * NUM_POINTS :]
        torch.testing.assert_close(
            past_block, torch.tensor([-2.0, -1.0, 1.0]).expand(NUM_POINTS, 3)
        )
        torch.testing.assert_close(
            future_block, torch.tensor([-1.0, -2.0, 1.5]).expand(NUM_POINTS, 3)
        )

    def test_the_sweeps_take_the_ignore_label(self) -> None:
        """
        Input: a labeled sample with a past and a future window of one sweep each.
        Expected: the sweeps only shape the geometry, so their points carry the ignore index
        while the labels of the current frame stay in place.
        Check: compare the label block of the current frame and of the sweeps.
        """
        loaded = LoadMultiSweepPointsFromFile(past=self.window(num=1), future=self.window(num=1))(
            self.build_sample(with_labels=True)
        )

        assert loaded.segmentation3d_gt_sample is not None
        labels = loaded.segmentation3d_gt_sample.gt_semantic_mask
        self.assertEqual(labels.shape[0], 3 * NUM_POINTS)
        self.assertTrue(bool((labels[:NUM_POINTS] == 0).all()))
        self.assertTrue(bool((labels[NUM_POINTS:] == -1).all()))


class TestLoadPointsFromFile(SweepLoadingTestCase):
    """Reading the stored feature layout of a frame off its record row."""

    def test_reads_the_feature_count_the_record_declares(self) -> None:
        """
        Input: a record row declaring seven features per point, as a sensing corpus stores.
        Expected: the loader reshapes the file by the declared count instead of a configured
        one, so a wider corpus needs no pipeline change.
        Check: the loaded cloud holds the written number of points.
        """
        frames = [self.write_frame(0, time_offset=0.0, num_features=7)]

        loaded = self.build_sample(frames)

        assert loaded.point_cloud_data is not None
        self.assertEqual(len(loaded.point_cloud_data), NUM_POINTS)

    def test_rejects_a_column_the_frame_does_not_store(self) -> None:
        """
        Input: a frame storing four features read with a pipeline asking for five.
        Expected: reading past the stored layout would shift the features, so the mismatch
        is reported.
        Check: a ValueError naming the stored count is raised.
        """
        frames = [self.write_frame(0, time_offset=0.0, num_features=4)]
        sample = ModelGTSample(
            lidar_point_cloud_samples=frames,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

        with self.assertRaisesRegex(ValueError, "features stored per"):
            LoadPointsFromFile(use_dim=(0, 1, 2, 3, 4))(sample)

    def test_normalizes_the_intensity_to_the_unit_range(self) -> None:
        """
        Input: a frame whose every stored feature holds the full scale the database declares.
        Expected: the network consumes intensity in [0, 1] while the geometry keeps its
        meters, so only the intensity column is divided by the declared scale.
        Check: intensity reads 1.0 and a coordinate is untouched.
        """
        frames = [self.write_frame(255, time_offset=0.0)]

        loaded = self.build_sample(frames)

        assert loaded.point_cloud_data is not None
        points = loaded.point_cloud_data.points
        self.assertAlmostEqual(float(points[0, PointFieldIndex.INTENSITY]), 1.0, places=6)
        self.assertAlmostEqual(float(points[0, PointFieldIndex.X]), 255.0, places=4)

    def test_the_labels_leave_with_the_points_removed_around_the_origin(self) -> None:
        """
        Input: a labeled frame with one point inside the removal radius and two outside it.
        Expected: the removed point takes its label along, so every kept point keeps its own.
        Check: the kept points and their labels.
        """
        path = self.root / "close.bin"
        points = np.zeros((3, 5), dtype=np.float32)
        points[0, :2] = 0.5
        points[1, 0] = 5.0
        points[2, 1] = 5.0
        points.tofile(path)
        frame = LiDARPointCloudSample(
            point_cloud_path=str(path),
            timestamp=self.SAMPLE_TIMESTAMP,
            num_features=5,
            intensity_scale=255.0,
            sensor_to_ego_pose_matrix=torch.eye(4, dtype=torch.float32),
            lidar_to_ego_pose_to_global_matrix=torch.eye(4, dtype=torch.float32),
            lidar_sensor_to_lidar_sweep_matrix=torch.eye(4, dtype=torch.float32),
        )
        sample = ModelGTSample(
            lidar_point_cloud_samples=[frame],
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=Segmentation3DGTSample(
                gt_semantic_mask=torch.tensor([1, 2, 3]), ignore_index=-1
            ),
        )

        loaded = LoadPointsFromFile(use_dim=(0, 1, 2, 3), bev_remove_radius=1.0)(sample)

        assert loaded.point_cloud_data is not None
        assert loaded.segmentation3d_gt_sample is not None
        kept = loaded.point_cloud_data.points
        self.assertEqual(kept[:, PointFieldIndex.X].tolist(), [5.0, 0.0])
        self.assertEqual(kept[:, PointFieldIndex.Y].tolist(), [0.0, 5.0])
        self.assertEqual(loaded.segmentation3d_gt_sample.gt_semantic_mask.tolist(), [2, 3])

    def test_serves_one_lidar_source_in_its_own_sensor_frame(self) -> None:
        """
        Input: a merged frame whose points carry their own index, and a source view naming
        points 2 to 5 of a sensor mounted 1 m ahead of the merged frame and turned by 90 deg.
        Expected: only the points of the source are served, moved into the sensor frame.
        Check: the point count and the transformed coordinates of the first served point.
        """
        path = self.root / "merged.bin"
        points = np.zeros((8, 5), dtype=np.float32)
        points[:, 0] = np.arange(8, dtype=np.float32)
        points.tofile(path)
        sensor_to_frame = torch.eye(4, dtype=torch.float32)
        sensor_to_frame[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        sensor_to_frame[0, 3] = 1.0
        sample = LiDARPointCloudSample(
            point_cloud_path=str(path),
            timestamp=self.SAMPLE_TIMESTAMP,
            num_features=5,
            intensity_scale=255.0,
            sensor_to_ego_pose_matrix=torch.eye(4, dtype=torch.float32),
            lidar_to_ego_pose_to_global_matrix=torch.eye(4, dtype=torch.float32),
            lidar_sensor_to_lidar_sweep_matrix=torch.eye(4, dtype=torch.float32),
            source_view=LidarSourceView(
                point_index_begin=2, num_points=3, sensor_to_frame_matrix=sensor_to_frame
            ),
        )

        loaded = LoadPointsFromFile(use_dim=(0, 1, 2, 3))(self.build_sample([sample]))

        assert loaded.point_cloud_data is not None
        served = loaded.point_cloud_data.points
        self.assertEqual(served.shape[0], 3)
        # Point 2 of the merged frame sits 1 m ahead of the sensor, which faces the frame y axis
        self.assertAlmostEqual(float(served[0, PointFieldIndex.X]), 0.0, places=5)
        self.assertAlmostEqual(float(served[0, PointFieldIndex.Y]), -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
