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

"""Dataset task reading the agent tracks of a record from the boxes of the later records."""

import logging
from collections.abc import Sequence

import numpy as np
from jaxtyping import Bool, Float, Float32, Float64, Int64
import polars as pl
import torch

from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema, key_frame_flags
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.dataclasses.batch.motion import AgentTracksSample, MotionGTSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.common.detection3d import box_instance_ids, flatten_box_fields
from autoware_ml.types.geometry import Box3DFieldIndex
from autoware_ml.types.motion import AgentRole

logger = logging.getLogger(__name__)

# Box fields of an agent state: center, heading and planar velocity
_STATE_FIELDS = [
    Box3DFieldIndex.X,
    Box3DFieldIndex.Y,
    Box3DFieldIndex.Z,
    Box3DFieldIndex.YAW,
    Box3DFieldIndex.VELOCITY_X,
    Box3DFieldIndex.VELOCITY_Y,
]
# Columns of an agent state
_STATE_CENTER, _STATE_YAW, _STATE_VELOCITY = slice(0, 3), 3, slice(4, 6)


class MotionTask(BaseDatasetTask):
    """Read the tracks of the annotated agents of a record from the later records.

    Every box of the record is an agent, in the order of the boxes. Its track starts at the
    record (step 0, the current step) and follows it over the future steps. The records of a
    scene are matched by timestamp: the record at a future step is the key frame of the same
    scenario closest to the step time, within the tolerance. A step without such a key frame, or
    an agent missing from it, is invalid. The tracks are expressed in the frame of the current
    record's point cloud. A T4Dataset scene has no focal agent, so every agent is scored.
    """

    COLUMNS = (
        DatasetTableSchema.SCENARIO_ID.name,
        DatasetTableSchema.TIMESTAMP_SECONDS.name,
        DatasetTableSchema.BOXES_3D.name,
        DatasetTableSchema.LIDAR_FRAMES.name,
    )

    def __init__(
        self,
        database_root_path: str,
        dataset_records_dataframe: pl.DataFrame,
        *,
        taxonomy: LabelTaxonomy,
        agent_steps: int,
        step_seconds: float,
        time_tolerance_seconds: float,
        max_speed: float,
        sparse_key_frames: bool,
    ) -> None:
        """
        Initialize the task and index the future records of every record.

        Args:
          database_root_path: Root directory of the database.
          dataset_records_dataframe: Records of the corpus.
          taxonomy: Taxonomy that maps the fine label name of a box to the agent type.
          agent_steps: Number of future steps of the agent tracks.
          step_seconds: Time between two steps.
          time_tolerance_seconds: Largest gap between a step time and the record serving it.
          max_speed: Largest plausible agent speed in m/s. A future position farther from the
            previous known one ends the track there: the instance id then names another
            object, a pseudo label id switch.
          sparse_key_frames: Accept scenarios whose key frames come less often than the steps,
            such as human labelled 1 Hz scenes read with 0.5 s steps for an evaluation. A step
            without a key frame stays invalid. Without it such scenarios are refused, so a
            training never meets futures with regular holes.

        Raises:
          ValueError: If a setting is out of range, or the key frames come less often than the
            steps and ``sparse_key_frames`` is off.
        """
        if agent_steps < 1 or step_seconds <= 0:
            raise ValueError(
                f"Futures need positive horizons and steps, got agent_steps={agent_steps}, "
                f"step_seconds={step_seconds}."
            )
        if max_speed <= 0:
            raise ValueError(f"max_speed must be positive, got {max_speed}.")
        if not 0 < time_tolerance_seconds < step_seconds / 2:
            raise ValueError(
                f"time_tolerance_seconds={time_tolerance_seconds} must be positive and below "
                "half a step, so one record never serves two steps."
            )
        super().__init__(database_root_path, dataset_records_dataframe)
        self.taxonomy: LabelTaxonomy = taxonomy
        self.agent_steps: int = agent_steps
        self.step_seconds: float = step_seconds
        self.max_speed: float = max_speed
        records = self.dataset_records_dataframe
        # Only key frames carry boxes, so every future step of every scenario needs one
        intervals = key_frame_intervals(records)
        if len(records) and not intervals:
            raise ValueError("No scenario of the records holds two key frames to build futures.")
        sparse = {
            scenario: interval
            for scenario, interval in intervals.items()
            if interval > step_seconds + time_tolerance_seconds
        }
        if sparse and not sparse_key_frames:
            raise ValueError(
                f"The key frames of scenarios {sorted(sparse)[:5]} come every up to "
                f"{max(sparse.values()):.2f} s, the futures need one every {step_seconds} s."
            )
        self.poses: Float64[np.ndarray, "num_records 4 4"] = record_poses(records)
        future_records = records_at_offsets(
            records,
            [step * step_seconds for step in range(1, agent_steps + 1)],
            time_tolerance_seconds,
        )
        # A lidar frame between the key frames carries no boxes, so it never serves a step
        key_frames = key_frame_flags(records).to_numpy()
        self.future_records: Int64[np.ndarray, "num_records agent_steps"] = np.where(
            (future_records >= 0) & key_frames[np.maximum(future_records, 0)], future_records, -1
        )
        box_offsets, box_instance_ids, box_states = flatten_box_states(records)
        self.box_offsets: Int64[np.ndarray, " num_records_plus_1"] = box_offsets
        self.box_instance_ids: Int64[np.ndarray, " num_boxes"] = box_instance_ids
        self.box_states: Float32[np.ndarray, "num_boxes 6"] = box_states

    def select_columns(self, dataset_records_dataframe: pl.DataFrame) -> pl.DataFrame:
        """
        Keep the scenario, the time, the boxes and the lidar frames of the records.

        Args:
          dataset_records_dataframe: Records of the corpus.

        Returns:
          pl.DataFrame: The records with the columns the tracks are built from.

        Raises:
          ValueError: If a column is missing.
        """
        missing = [
            column for column in self.COLUMNS if column not in dataset_records_dataframe.columns
        ]
        if missing:
            raise ValueError(f"{self} needs the columns {list(self.COLUMNS)}, missing {missing}.")
        return dataset_records_dataframe.select(list(self.COLUMNS))

    def get_data_sample(self, idx: int) -> ModelGTSample:
        """
        Build the agent tracks of one record.

        Args:
          idx: Index of the record.

        Returns:
          ModelGTSample: Sample holding the agent tracks of the record.
        """
        begin, end = self.box_offsets[idx], self.box_offsets[idx + 1]
        agent_ids = self.box_instance_ids[begin:end]
        num_agents = len(agent_ids)
        num_steps = self.agent_steps + 1
        states = np.zeros((num_agents, num_steps, 6), dtype=np.float64)
        valid = np.zeros((num_agents, num_steps), dtype=bool)
        states[:, 0] = self.box_states[begin:end]
        valid[:, 0] = True
        to_current = np.linalg.inv(self.poses[idx])
        for step in range(1, num_steps):
            future = self.future_records[idx, step - 1]
            if future < 0:
                continue
            future_begin, future_end = self.box_offsets[future], self.box_offsets[future + 1]
            found, rows = match_ids(agent_ids, self.box_instance_ids[future_begin:future_end])
            transform = to_current @ self.poses[future]
            states[found, step] = move_states(
                self.box_states[future_begin:future_end][rows].astype(np.float64), transform
            )
            valid[found, step] = True
        valid[:, 1:] &= plausible_steps(
            states[:, 0, _STATE_CENTER],
            states[:, 1:, _STATE_CENTER],
            valid[:, 1:],
            self.max_speed * self.step_seconds,
        )
        # An unobserved step carries zeros
        states[~valid] = 0.0

        label_names, track_ids, sizes = self.current_boxes(idx)
        agents = AgentTracksSample(
            positions=torch.from_numpy(states[..., _STATE_CENTER]).float(),
            headings=torch.from_numpy(states[..., _STATE_YAW]).float(),
            velocities=torch.from_numpy(states[..., _STATE_VELOCITY]).float(),
            valid=torch.from_numpy(valid),
            agent_types=torch.tensor(
                [self.taxonomy.class_index(name) for name in label_names], dtype=torch.int64
            ),
            agent_roles=torch.full((num_agents,), AgentRole.SCORED, dtype=torch.int64),
            track_ids=track_ids,
            current_step_index=0,
            step_seconds=self.step_seconds,
            ignore_index=self.taxonomy.ignore_index,
            sizes=torch.from_numpy(sizes),
        )
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
            motion_gt_sample=MotionGTSample(agents=agents),
        )

    def current_boxes(
        self, idx: int
    ) -> tuple[list[str], list[str], Float32[np.ndarray, "num_agents 3"]]:
        """
        Read the label, the instance token and the size of every box of one record.

        Args:
          idx: Index of the record.

        Returns:
          tuple: The fine label name, the instance token and the length, width and height
            ``(A, 3)`` of every box, in the order of the boxes.
        """
        if self.box_offsets[idx] == self.box_offsets[idx + 1]:
            return [], [], np.zeros((0, 3), dtype=np.float32)
        boxes = self.dataset_records_dataframe.item(idx, DatasetTableSchema.BOXES_3D.name).struct
        box_params = np.asarray(
            boxes.field(Box3DDatasetSchema.BOX3D_PARAMS.name).to_numpy(), dtype=np.float32
        )
        return (
            boxes.field(Box3DDatasetSchema.BOX3D_LABEL_NAME.name).to_list(),
            boxes.field(Box3DDatasetSchema.BOX3D_INSTANCE_ID.name).to_list(),
            np.ascontiguousarray(
                box_params[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT + 1]
            ),
        )


def move_states(
    states: Float64[np.ndarray, "num_agents 6"], transform: Float64[np.ndarray, "4 4"]
) -> Float64[np.ndarray, "num_agents 6"]:
    """
    Express agent states of another record in the frame of the current one.

    Args:
      states: States ``(N, 6)`` of the agents in the frame of their record, see box_states.
      transform: Transform from the frame of their record to the current one ``(4, 4)``.

    Returns:
      The states in the current frame ``(N, 6)``: the center moved, the heading and the velocity
        turned by the rotation.
    """
    rotation = transform[:2, :2]
    moved = states.copy()
    moved[:, _STATE_CENTER] = states[:, _STATE_CENTER] @ transform[:3, :3].T + transform[:3, 3]
    yaw = states[:, _STATE_YAW]
    direction = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1) @ rotation.T
    moved[:, _STATE_YAW] = np.arctan2(direction[:, 1], direction[:, 0])
    moved[:, _STATE_VELOCITY] = states[:, _STATE_VELOCITY] @ rotation.T
    return moved


def record_poses(records: pl.DataFrame) -> Float64[np.ndarray, "num_records 4 4"]:
    """
    Transform of every record from the frame of its point cloud to the global frame.

    Args:
      records: Records with their lidar frames.

    Returns:
      Ego pose composed with the lidar mounting of the first lidar frame of every record.
    """
    first_frame = pl.col(DatasetTableSchema.LIDAR_FRAMES.name).list.first().struct
    poses = records.select(
        first_frame.field(LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name),
        first_frame.field(LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name),
    )
    ego_to_global = poses.to_series(0).to_numpy().astype(np.float64)
    sensor_to_ego = poses.to_series(1).to_numpy().astype(np.float64)
    return ego_to_global @ sensor_to_ego


def key_frame_intervals(records: pl.DataFrame) -> dict[str, float]:
    """
    Median time between consecutive key frames of every scenario.

    Args:
      records: Records with their scenario, timestamp and lidar frames.

    Returns:
      The median gap in seconds of every scenario holding two key frames.
    """
    scenario = DatasetTableSchema.SCENARIO_ID.name
    time = DatasetTableSchema.TIMESTAMP_SECONDS.name
    gaps = (
        records.select(scenario, time)
        .filter(key_frame_flags(records))
        .sort(scenario, time)
        .with_columns(pl.col(time).diff().over(scenario).alias("gap"))
        .drop_nulls("gap")
        .group_by(scenario)
        .agg(pl.col("gap").median())
    )
    return dict(zip(gaps[scenario].to_list(), gaps["gap"].to_list(), strict=True))


def records_at_offsets(
    records: pl.DataFrame, offsets: Sequence[float], tolerance: float
) -> Int64[np.ndarray, "num_records num_offsets"]:
    """
    Index of the record of the same scenario closest to every record's time plus each offset.

    Args:
      records: Records with their scenario and timestamp.
      offsets: Time offsets in seconds.
      tolerance: Largest gap between the offset time and the matched record.

    Returns:
      One record index per record and offset, -1 where no record lies within the tolerance.
    """
    scenario = DatasetTableSchema.SCENARIO_ID.name
    time = DatasetTableSchema.TIMESTAMP_SECONDS.name
    table = records.select(scenario, time).with_row_index("record")
    candidates = table.rename({"record": "found"}).sort(time)
    columns = []
    for offset in offsets:
        queries = table.with_columns(pl.col(time) + offset).sort(time)
        found = queries.join_asof(
            candidates,
            on=time,
            by=scenario,
            strategy="nearest",
            tolerance=tolerance,
            # Both sides are sorted by time above
            check_sortedness=False,
        )
        columns.append(found.sort("record")["found"].fill_null(-1).to_numpy().astype(np.int64))
    return np.stack(columns, axis=1)


def flatten_box_states(
    records: pl.DataFrame,
) -> tuple[
    Int64[np.ndarray, " num_records_plus_1"],
    Int64[np.ndarray, " num_boxes"],
    Float32[np.ndarray, "num_boxes 6"],
]:
    """
    Gather the instance identity and the state of every box of every record.

    Args:
      records: Records with their boxes.

    Returns:
      tuple: Offset of every record's boxes (one more than the records), the identity of every
        box and its state ``(x, y, z, yaw, vx, vy)``, in record order. A velocity the record
        does not annotate is zero.
    """
    instance_id = Box3DDatasetSchema.BOX3D_INSTANCE_ID.name
    params = Box3DDatasetSchema.BOX3D_PARAMS.name
    offsets, box_fields = flatten_box_fields(records, [instance_id, params])
    states = box_states(box_fields[params].to_numpy().reshape(-1, len(Box3DFieldIndex)))
    return offsets, box_instance_ids(box_fields[instance_id], offsets), states


def box_states(
    box_params: Float[np.ndarray, "num_boxes box_dims"],
) -> Float32[np.ndarray, "num_boxes 6"]:
    """
    State of every box as the agent tracks carry it.

    Args:
      box_params: Parameters of every box ``(N, 10)``, laid out by Box3DFieldIndex.

    Returns:
      The center, the heading and the planar velocity of every box ``(N, 6)``, a velocity the
        record does not annotate set to zero.
    """
    states = np.ascontiguousarray(box_params[:, _STATE_FIELDS], dtype=np.float32)
    velocity = states[:, _STATE_VELOCITY]
    velocity[~np.isfinite(velocity)] = 0.0
    return states


def plausible_steps(
    current: Float64[np.ndarray, "num_agents 3"],
    positions: Float64[np.ndarray, "num_agents num_steps 3"],
    valid: Bool[np.ndarray, "num_agents num_steps"],
    max_step_distance: float,
) -> Bool[np.ndarray, "num_agents num_steps"]:
    """
    Cut every trajectory at its first jump beyond the distance an agent can cover.

    Args:
      current: Current position of every agent ``(A, 3)``.
      positions: Future positions ``(A, T, 3)``.
      valid: Whether a future position is known ``(A, T)``.
      max_step_distance: Largest BEV distance covered in one step.

    Returns:
      The steps before the first implausible jump of every agent ``(A, T)``.
    """
    plausible = np.ones_like(valid)
    last = current[:, :2].copy()
    last_step = np.full(len(current), -1)
    for step in range(valid.shape[1]):
        distance = np.linalg.norm(positions[:, step, :2] - last, axis=1)
        jump = valid[:, step] & (distance > max_step_distance * (step - last_step))
        plausible[:, step:] &= ~jump[:, None]
        known = valid[:, step] & plausible[:, step]
        last[known] = positions[known, step, :2]
        last_step[known] = step
    return plausible


def match_ids(
    queries: Int64[np.ndarray, " num_queries"], candidates: Int64[np.ndarray, " num_candidates"]
) -> tuple[Bool[np.ndarray, " num_queries"], Int64[np.ndarray, " num_found"]]:
    """
    Find every query identity among the candidates.

    Args:
      queries: Identities to look up.
      candidates: Identities to search.

    Returns:
      tuple: Whether every query was found, and the candidate row of every found query.
    """
    if len(candidates) == 0:
        return np.zeros(len(queries), dtype=bool), np.zeros(0, dtype=np.int64)
    order = np.argsort(candidates)
    positions = np.clip(np.searchsorted(candidates[order], queries), 0, len(candidates) - 1)
    found = candidates[order][positions] == queries
    return found, order[positions[found]]
