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

"""What the stream of a temporal model has observed so far, as the temporal metrics count it.

A temporal model only knows the objects it has seen in its stream. The temporal metrics count a
ground truth object from the first frame of the stream in which it holds enough lidar points,
and leave it out before. A stream is the run of frames a batch position serves of one scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from jaxtyping import Bool, Int, Int64

from autoware_ml.metrics.eval_output import Detection3DEvalFrame


@dataclass
class _LaneObservation:
    """Instances one batch position has observed in its current scene.

    Attributes:
        scene_token: Scene the position streams.
        seen: Instance ids observed so far.
    """

    scene_token: str
    seen: set[int] = field(default_factory=set)


class StreamObserver:
    """Follow the objects every batch position of a streaming evaluation has observed.

    The streaming datamodule serves every scene in time order to one batch position, so a batch
    position streams one scene after the other. An object counts as observed from the first frame
    of its scene in which its box holds at least ``min_observed_points`` lidar points.
    """

    def __init__(self, min_observed_points: int) -> None:
        """Initialize the observer.

        Args:
            min_observed_points: Lidar points a box needs in one frame for its object to count
                as observed.

        Raises:
            ValueError: If the point threshold is negative.
        """
        if min_observed_points < 0:
            raise ValueError(
                f"min_observed_points must be non-negative, got {min_observed_points}."
            )
        self.min_observed_points = min_observed_points
        self.reset()

    def reset(self) -> None:
        """Forget every stream."""
        self._lanes: dict[int, _LaneObservation] = {}

    def observe(
        self,
        lane: int,
        scene_token: str,
        instance_ids: Int64[torch.Tensor, " num_gts"],
        num_points: Int[torch.Tensor, " num_gts"],
    ) -> Bool[torch.Tensor, " num_gts"]:
        """Add the objects a frame shows and tell which of its boxes are observed.

        Args:
            lane: Batch position of the frame.
            scene_token: Scene of the frame.
            instance_ids: Object identity of every ground truth box.
            num_points: Lidar points inside every ground truth box.

        Returns:
            Whether the object of every ground truth box was observed in the stream so far.
        """
        state = self._lanes.get(lane)
        if state is None or state.scene_token != scene_token:
            state = _LaneObservation(scene_token)
            self._lanes[lane] = state
        state.seen.update(instance_ids[num_points >= self.min_observed_points].tolist())
        return torch.tensor(
            [value in state.seen for value in instance_ids.tolist()],
            dtype=torch.bool,
            device=instance_ids.device,
        )


def observe_detection_frame(
    observer: StreamObserver, lane: int, frame: Detection3DEvalFrame
) -> Bool[torch.Tensor, " num_gts"]:
    """Observe the boxes of a detection frame, see :meth:`StreamObserver.observe`.

    Args:
        observer: The observer of the suite.
        lane: Batch position of the frame.
        frame: The frame, with the identities and point counts of its boxes and its scene.

    Returns:
        Whether the object of every ground truth box was observed in the stream so far.

    Raises:
        ValueError: If the frame lacks the identities, the point counts or the scene of its boxes.
    """
    if frame.gt_instance_ids is None or frame.gt_num_points is None or frame.frame_meta is None:
        raise ValueError(
            "Observing a stream needs the instance ids and the point counts of the boxes and the "
            "scene of every frame."
        )
    return observer.observe(
        lane, frame.frame_meta.scene_token, frame.gt_instance_ids, frame.gt_num_points
    )
