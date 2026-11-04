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


"""K-means motion anchors from the futures of the training agents."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import numpy as np
from jaxtyping import Float

from autoware_ml.anchors.base import AnchorRecipe
from autoware_ml.anchors.kmeans import cluster_modes
from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import key_frame_flags
from autoware_ml.databases.taxonomy import LabelTaxonomy
from autoware_ml.datamodule.common.detection3d import flatten_box_fields
from autoware_ml.datamodule.common.motion import MotionTask
from autoware_ml.datamodule.training_split import TrainingSplit
from autoware_ml.geometry.bbox_3d.exclusion import excluded_boxes, parse_exclusion_rules
from autoware_ml.types.geometry import Box3DFieldIndex
from autoware_ml.types.tasks import TaskType

logger = logging.getLogger(__name__)


def collect_agent_futures(
    motion_task: MotionTask,
    taxonomy: LabelTaxonomy,
    record_stride: int,
    max_agent_distance: float,
    exclusion_rules: frozenset[tuple[str, str]],
) -> list[list[Float[np.ndarray, "num_steps 2"]]]:
    """Gather the complete futures of the agents per class.

    Args:
        motion_task: Agent tracks of the records, one agent per box in the order of the boxes.
        taxonomy: Detection taxonomy, boxes of a fine label it does not train are skipped.
        record_stride: Every how many key-frame records one is read.
        max_agent_distance: Largest BEV distance of a kept agent from the sensor.
        exclusion_rules: Class and attribute pairs of the boxes the training pipeline drops.

    Returns:
        Agent futures ``(T, 2)`` in the frame of the agent per class.
    """
    records = motion_task.dataset_records_dataframe
    label = Box3DDatasetSchema.BOX3D_LABEL_NAME.name
    params = Box3DDatasetSchema.BOX3D_PARAMS.name
    attributes = Box3DDatasetSchema.BOX3D_ATTRIBUTES.name
    offsets, boxes = flatten_box_fields(records, [label, params, attributes])
    names = boxes[label].to_list()
    box_params = boxes[params].to_numpy().reshape(-1, len(Box3DFieldIndex))
    excluded = excluded_boxes(names, boxes[attributes].to_list(), exclusion_rules)
    agents: list[list[Float[np.ndarray, "num_steps 2"]]] = [[] for _ in taxonomy.class_names]
    # Only key frames carry boxes, so the stride counts key frames
    key_frames = key_frame_flags(records).to_numpy()
    for record in np.flatnonzero(key_frames)[::record_stride].tolist():
        tracks = motion_task.get_data_sample(record).motion_gt_sample.agents
        future = slice(tracks.current_step_index + 1, None)
        complete = tracks.valid[:, future].all(dim=1).numpy()
        positions = tracks.positions[:, future, :2].numpy()
        for row in np.flatnonzero(complete):
            box = int(offsets[record]) + int(row)
            class_name = taxonomy.class_name(names[box])
            center = box_params[box, :2]
            if excluded[box] or class_name is None or np.hypot(*center) > max_agent_distance:
                continue
            yaw = box_params[box, Box3DFieldIndex.YAW]
            offset = positions[row] - center
            cos, sin = np.cos(yaw), np.sin(yaw)
            agents[taxonomy.class_names.index(class_name)].append(
                np.stack(
                    [
                        offset[:, 0] * cos + offset[:, 1] * sin,
                        -offset[:, 0] * sin + offset[:, 1] * cos,
                    ],
                    -1,
                )
            )
    return agents


class TrajectoryAnchorRecipe(AnchorRecipe):
    """K-means modes of the complete agent futures of every class, in the frame of the agent."""

    name = "motion_trajectories"
    task_types = (TaskType.MOTION,)

    def __init__(
        self,
        *,
        cache_path: str,
        taxonomy: LabelTaxonomy,
        num_modes: int,
        record_stride: int,
        max_agent_distance: float,
        iterations: int,
        seed: int,
        class_donors: Mapping[str, str],
        filter_attributes: Sequence[Sequence[str]],
    ) -> None:
        """Initialize the recipe.

        Args:
            cache_path: Directory of the cached anchors.
            taxonomy: Detection taxonomy, its classes get one set of modes each.
            num_modes: Modes per class.
            record_stride: Every how many key-frame training records one is read.
            max_agent_distance: Largest BEV distance of a read agent from the sensor.
            iterations: Number of k-means iterations.
            seed: Seed of the initial centroids.
            class_donors: Class with fewer complete futures than modes to the class whose
                futures it takes.
            filter_attributes: Class and attribute pairs of the boxes the training pipeline
                drops, their agents shape no mode.

        Raises:
            ValueError: If a donor pair names a class outside the taxonomy, or if a donor takes
                the futures of another donor.
        """
        unknown = {name for pair in class_donors.items() for name in pair} - set(
            taxonomy.class_names
        )
        if unknown:
            raise ValueError(f"class_donors names classes outside the taxonomy: {sorted(unknown)}.")
        chained = set(class_donors) & set(class_donors.values())
        if chained:
            raise ValueError(f"class_donors chains donors, {sorted(chained)} give and take.")
        super().__init__(cache_path)
        self.taxonomy = taxonomy
        self.num_modes = num_modes
        self.record_stride = record_stride
        self.max_agent_distance = max_agent_distance
        self.iterations = iterations
        self.seed = seed
        self.class_donors = dict(class_donors)
        self.exclusion_rules = parse_exclusion_rules(filter_attributes)

    def hash_repr(self) -> str:
        """Canonical text of the recipe settings."""
        return (
            f"{type(self).__name__}(taxonomy={self.taxonomy}, num_modes={self.num_modes}, "
            f"record_stride={self.record_stride}, max_agent_distance={self.max_agent_distance}, "
            f"iterations={self.iterations}, seed={self.seed}, "
            f"class_donors={sorted(self.class_donors.items())}, "
            f"exclusion_rules={sorted(self.exclusion_rules)})"
        )

    def build(self, split: TrainingSplit) -> dict[str, Float[np.ndarray, "..."]]:
        """Cluster the complete training futures of every class into modes.

        Args:
            split: Training sources and their records.

        Returns:
            ``anchors`` ``(num_classes, num_modes, agent_steps, 2)``.

        Raises:
            ValueError: If a class that takes the futures of a donor has enough for its modes,
                or its donor too few.
        """
        class_names = list(self.taxonomy.class_names)
        agents: list[list[Float[np.ndarray, "num_steps 2"]]] = [[] for _ in class_names]
        for training in split.datasets(self.task_types):
            motion_task = training.dataset.dataset_tasks[TaskType.MOTION]
            if not isinstance(motion_task, MotionTask):
                raise TypeError(
                    f"The motion task of {training.source.database.version} is "
                    f"{type(motion_task).__name__}, the trajectory anchors read a MotionTask."
                )
            source_agents = collect_agent_futures(
                motion_task,
                self.taxonomy,
                self.record_stride,
                self.max_agent_distance,
                self.exclusion_rules,
            )
            for class_index, futures in enumerate(source_agents):
                agents[class_index].extend(futures)
            logger.info(
                f"{training.source.database.version}: complete agent futures per class "
                f"{dict(zip(class_names, map(len, source_agents)))}"
            )
        # A class too rare to cluster takes the futures of the donor the config names for it
        for name, donor in self.class_donors.items():
            own = agents[class_names.index(name)]
            if len(own) >= self.num_modes:
                raise ValueError(
                    f"Class {name} has {len(own)} complete futures, enough for its "
                    f"{self.num_modes} modes. Remove it from class_donors."
                )
            donated = agents[class_names.index(donor)]
            if len(donated) < self.num_modes:
                raise ValueError(
                    f"Donor {donor} of class {name} has {len(donated)} complete futures, "
                    f"{self.num_modes} modes need more."
                )
            agents[class_names.index(name)] = donated
        modes = [
            cluster_modes(futures, self.num_modes, self.iterations, self.seed, name)
            for futures, name in zip(agents, class_names, strict=True)
        ]
        return {"anchors": np.stack(modes)}
