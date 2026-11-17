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

"""Tests for the agent tracks the motion task reads from the later records of a scene."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl
from jaxtyping import Float64

from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.databases.taxonomy import LabelTaxonomy, LabelVocabulary
from autoware_ml.dataclasses.batch.motion import AgentTracksSample
from autoware_ml.datamodule.common.motion import MotionTask
from autoware_ml.types.motion import AgentRole

FRAMES = 21
# The scene lies this far along the map x axis, where a float32 pose would lose millimeters
MAP_X = 65000.0
# The walker is a trained class, the parked car a label the taxonomy drops
TAXONOMY = LabelTaxonomy(
    LabelVocabulary({"pedestrian": "pedestrian", "car": "car"}),
    ["pedestrian"],
    {"pedestrian": "pedestrian", "car": None},
    -1,
)


def pose(x: float, yaw: float = 0.0) -> Float64[np.ndarray, "4 4"]:
    """Ego pose at x along the global x axis, heading yaw."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    matrix[0, 3] = x
    return matrix


def box(
    center: np.ndarray, yaw: float, velocity: tuple[float, float], instance: str, label: str
) -> dict:
    """Box record with a center, a heading and a velocity in the frame of its record."""
    params = np.zeros(10, dtype=np.float32)
    params[:3] = center
    params[3:6] = (0.5, 0.6, 1.7) if label == "pedestrian" else (4.5, 1.8, 1.5)
    params[6] = yaw
    params[7:9] = velocity
    return {
        Box3DDatasetSchema.BOX3D_PARAMS.name: params,
        Box3DDatasetSchema.BOX3D_INSTANCE_ID.name: instance,
        Box3DDatasetSchema.BOX3D_LABEL_NAME.name: label,
    }


def records(
    jump_frame: int | None = None,
    walker_id: str = "walker",
    key_every: int = 1,
    ego_yaw_rate: float = 0.0,
) -> pl.DataFrame:
    """A 2 s scene at 10 Hz, the ego driving 10 m/s along x from MAP_X and two agents.

    The walker moves 2 m/s along global y from (MAP_X + 20, 0), heading along global y, the parked
    car stays at global (MAP_X + 30, 5) heading along global x, with no annotated velocity. With
    ``jump_frame`` the walker's id names an object 100 m away from that frame on. Every
    ``key_every``-th frame is a key frame. The ego turns by ``ego_yaw_rate`` per frame.
    """
    rows = []
    for frame in range(FRAMES):
        ego_yaw = ego_yaw_rate * frame
        ego = pose(MAP_X + 1.0 * frame, ego_yaw)
        to_frame = np.linalg.inv(ego)
        walker_global = np.array([MAP_X + 20.0, 0.2 * frame, 0.0])
        if jump_frame is not None and frame >= jump_frame:
            walker_global = walker_global + np.array([100.0, 0.0, 0.0])
        car_global = np.array([MAP_X + 30.0, 5.0, 0.0])
        walker_velocity = to_frame[:2, :2] @ np.array([0.0, 2.0])
        boxes = [
            box(
                (to_frame @ np.append(walker_global, 1.0))[:3],
                np.pi / 2 - ego_yaw,
                tuple(walker_velocity),
                walker_id,
                "pedestrian",
            ),
            box(
                (to_frame @ np.append(car_global, 1.0))[:3],
                -ego_yaw,
                (np.nan, np.nan),
                "car",
                "car",
            ),
        ]
        rows.append(
            {
                DatasetTableSchema.SCENARIO_ID.name: "scene",
                DatasetTableSchema.TIMESTAMP_SECONDS.name: 100.0 + 0.1 * frame,
                DatasetTableSchema.BOXES_3D.name: boxes,
                DatasetTableSchema.LIDAR_FRAMES.name: [
                    {
                        LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name: ego,
                        LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name: np.eye(
                            4, dtype=np.float32
                        ),
                        LidarFrameDatasetSchema.lidar_keyframe.name: frame % key_every == 0,
                    }
                ],
            }
        )
    ego_pose = LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix
    mounting = LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix
    return pl.DataFrame(rows).with_columns(
        pl.col(DatasetTableSchema.LIDAR_FRAMES.name).cast(
            pl.List(
                pl.Struct(
                    {
                        ego_pose.name: ego_pose.dtype,
                        mounting.name: mounting.dtype,
                        LidarFrameDatasetSchema.lidar_keyframe.name: pl.Boolean,
                    }
                )
            )
        ),
        pl.col(DatasetTableSchema.BOXES_3D.name).cast(
            pl.List(
                pl.Struct(
                    {
                        Box3DDatasetSchema.BOX3D_PARAMS.name: pl.Array(pl.Float32, shape=(10,)),
                        Box3DDatasetSchema.BOX3D_INSTANCE_ID.name: pl.String,
                        Box3DDatasetSchema.BOX3D_LABEL_NAME.name: pl.String,
                    }
                )
            )
        ),
    )


def task(dataframe: pl.DataFrame, sparse_key_frames: bool = False) -> MotionTask:
    """Motion task over 0.5 s steps and 2 agent steps."""
    return MotionTask(
        "/unused",
        dataframe,
        taxonomy=TAXONOMY,
        agent_steps=2,
        step_seconds=0.5,
        time_tolerance_seconds=0.03,
        max_speed=50.0,
        sparse_key_frames=sparse_key_frames,
    )


def tracks(dataframe: pl.DataFrame, record: int) -> AgentTracksSample:
    """Agent tracks of one record."""
    return task(dataframe).get_data_sample(record).motion_gt_sample.agents


class TestMotionTask(unittest.TestCase):
    def test_tracks_start_at_the_current_record_in_its_frame(self) -> None:
        """
        Input: the scene at its first record, the ego at x=0 moving 10 m/s.
        Expected: the window holds the current step and 2 futures, the walker moves from
          (20, 0) to y=1 and y=2, the parked car stays at (30, 5), every step known.
        """
        agents = tracks(records(), 0)
        self.assertEqual((agents.current_step_index, agents.num_steps), (0, 3))
        self.assertEqual(agents.step_seconds, 0.5)
        self.assertEqual(list(agents.track_ids), ["walker", "car"])
        np.testing.assert_allclose(
            agents.positions[0, :, :2], [[20, 0], [20, 1], [20, 2]], atol=1e-4
        )
        np.testing.assert_allclose(agents.positions[1, :, :2], [[30, 5]] * 3, atol=1e-4)
        self.assertTrue(bool(agents.valid.all()))

    def test_headings_and_velocities_follow_the_current_frame(self) -> None:
        """
        Input: the ego turning 0.05 rad per frame, the tracks of its first record.
        Expected: the walker keeps heading along y and moving 2 m/s along y in the frame of the
          first record at every step, the car keeps heading along x.
        """
        agents = tracks(records(ego_yaw_rate=0.05), 0)
        np.testing.assert_allclose(agents.headings[0], [np.pi / 2] * 3, atol=1e-5)
        np.testing.assert_allclose(agents.velocities[0], [[0.0, 2.0]] * 3, atol=1e-5)
        np.testing.assert_allclose(agents.headings[1], [0.0] * 3, atol=1e-5)

    def test_an_unannotated_velocity_is_zero(self) -> None:
        """
        Input: the parked car, whose boxes carry no velocity.
        Expected: zero velocity at every step instead of the non finite record value.
        """
        agents = tracks(records(), 0)
        np.testing.assert_array_equal(agents.velocities[1], np.zeros((3, 2)))

    def test_types_roles_and_sizes_follow_the_boxes(self) -> None:
        """
        Input: a trained walker and a car the taxonomy drops.
        Expected: the walker typed 0, the car the ignore index, both scored, the box sizes.
        """
        agents = tracks(records(), 0)
        self.assertEqual(agents.agent_types.tolist(), [0, -1])
        self.assertEqual(agents.ignore_index, -1)
        self.assertEqual(agents.agent_roles.tolist(), [AgentRole.SCORED] * 2)
        np.testing.assert_allclose(agents.sizes, [[0.5, 0.6, 1.7], [4.5, 1.8, 1.5]])

    def test_steps_past_the_scene_end_are_invalid_and_zero(self) -> None:
        """
        Input: the record 0.7 s before the end of the scene.
        Expected: the 1.0 s step lies past the last record for every agent and carries zeros.
        """
        agents = tracks(records(), FRAMES - 8)
        self.assertTrue(agents.valid[:, 2].logical_not().all())
        self.assertTrue(bool((agents.positions[:, 2] == 0).all()))

    def test_an_id_switch_cuts_the_track(self) -> None:
        """
        Input: the walker id names an object 100 m away from frame 10 on, the tracks of frame 0.
        Expected: the walker keeps its 0.5 s step and loses the 1.0 s one, the car keeps both.
        """
        agents = tracks(records(jump_frame=10), 0)
        self.assertEqual(agents.valid.tolist(), [[True, True, False], [True, True, True]])

    def test_a_record_without_boxes_has_no_agents(self) -> None:
        """
        Input: a scene whose third record holds no boxes.
        Expected: empty tracks of the full window for that record.
        """
        boxes = DatasetTableSchema.BOXES_3D.name
        dataframe = records().with_row_index("row")
        dataframe = dataframe.with_columns(
            pl.when(pl.col("row") == 2).then(pl.col(boxes).list.head(0)).otherwise(pl.col(boxes))
        ).drop("row")
        agents = tracks(dataframe, 2)
        self.assertEqual(tuple(agents.positions.shape), (0, 3, 3))
        self.assertEqual(tuple(agents.sizes.shape), (0, 3))
        self.assertEqual(list(agents.track_ids), [])

    def test_key_frames_sparser_than_the_steps_are_rejected(self) -> None:
        """
        Input: a 10 Hz scene with a key frame every 1 s, read with 0.5 s steps.
        Expected: ValueError, every second step would land on a frame without boxes.
        """
        with self.assertRaisesRegex(ValueError, r"scenarios \['scene'\] come every up to 1.00 s"):
            task(records(key_every=10))

    def test_sparse_key_frames_leave_the_steps_between_them_invalid(self) -> None:
        """
        Input: a 10 Hz scene with a key frame every 1 s, read with 0.5 s steps and sparse key
          frames accepted, the tracks of its first frame.
        Expected: the 0.5 s step lands between key frames and is invalid for every agent, although
          its record holds boxes, the current and the 1.0 s steps are known with the walker at y=2.
        """
        agents = task(records(key_every=10), sparse_key_frames=True).get_data_sample(0)
        agents = agents.motion_gt_sample.agents
        self.assertEqual(agents.valid.tolist(), [[True, False, True], [True, False, True]])
        np.testing.assert_allclose(agents.positions[0, 2, :2], [20, 2], atol=1e-4)

    def test_a_scenario_sparser_than_the_steps_is_rejected_among_dense_ones(self) -> None:
        """
        Input: two dense 10 Hz scenes and one with a key frame every 1 s, read with 0.5 s steps.
        Expected: ValueError naming the sparse scene, a median over all scenes would let it pass.
        """
        scene = DatasetTableSchema.SCENARIO_ID.name
        mixed = pl.concat(
            [
                records().with_columns(pl.lit("dense_a").alias(scene)),
                records().with_columns(pl.lit("dense_b").alias(scene)),
                records(key_every=10).with_columns(pl.lit("sparse").alias(scene)),
            ]
        )

        with self.assertRaisesRegex(ValueError, r"scenarios \['sparse'\]"):
            task(mixed)

    def test_records_without_two_key_frames_in_a_scenario_are_rejected(self) -> None:
        """
        Input: a scene whose only key frame is its first frame.
        Expected: ValueError, no future step can ever be supervised.
        """
        with self.assertRaisesRegex(ValueError, "holds two key frames"):
            task(records(key_every=100))

    def test_boxes_sharing_an_instance_id_are_rejected(self) -> None:
        """
        Input: the walker and the car carry the same instance id.
        Expected: ValueError when the task is built, the futures of the id are ambiguous.
        """
        with self.assertRaisesRegex(ValueError, "two boxes of one instance"):
            task(records(walker_id="car"))

    def test_a_non_positive_max_speed_is_rejected(self) -> None:
        """
        Input: a largest agent speed of zero.
        Expected: ValueError, every future step would end its track.
        """
        with self.assertRaisesRegex(ValueError, "max_speed must be positive"):
            MotionTask(
                "/unused",
                records(),
                taxonomy=TAXONOMY,
                agent_steps=2,
                step_seconds=0.5,
                time_tolerance_seconds=0.03,
                max_speed=0.0,
                sparse_key_frames=False,
            )

    def test_records_without_the_track_columns_are_rejected(self) -> None:
        """
        Input: records without their lidar frames.
        Expected: ValueError naming the missing column, the poses of the records come from it.
        """
        with self.assertRaisesRegex(ValueError, DatasetTableSchema.LIDAR_FRAMES.name):
            task(records().drop(DatasetTableSchema.LIDAR_FRAMES.name))

    def test_a_tolerance_of_half_a_step_is_rejected(self) -> None:
        """
        Input: a tolerance as large as half the step.
        Expected: ValueError, one record could serve two steps.
        """
        with self.assertRaisesRegex(ValueError, "half a step"):
            MotionTask(
                "/unused",
                records(),
                taxonomy=TAXONOMY,
                agent_steps=2,
                step_seconds=0.1,
                time_tolerance_seconds=0.05,
                max_speed=50.0,
                sparse_key_frames=False,
            )
