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

"""Per-sample, per-class annotation masks for datasets whose annotations are partial.

A T4 scene may have been annotated without its traffic cones and barriers, which the dataset
reports as a per-sample ``gt_traffic_cone_barrier_bbox_status`` flag. The detection head must
not learn that those classes are absent in such a sample, but the head should not know about
T4 or about cones: it takes a generic ``(batch_size, num_classes)`` mask of annotated classes.
This module is where the dataset flag is turned into that mask.
"""

from collections.abc import Sequence
import torch
from jaxtyping import Bool


class PartialAnnotations:
    """Map a dataset's per-sample annotation flag to a per-sample, per-class annotated mask.

    Args:
        class_names: Detection class names, in head order.
        partial_ignore_labels: Class names whose annotations the dataset flag covers. In a
            sample flagged as not annotated, these classes are reported as unannotated so the
            head neither supervises their dense heatmap nor pushes negative queries away from
            them. Every other class is always annotated.
    """

    def __init__(self, class_names: Sequence[str], partial_ignore_labels: Sequence[str]) -> None:
        self.class_names = list(class_names)
        if len(partial_ignore_labels) == 0:
            raise ValueError(
                "PartialAnnotations needs at least one label in partial_ignore_labels, "
                "leave partial_annotations unset to supervise every class."
            )
        unknown_labels = [label for label in partial_ignore_labels if label not in self.class_names]
        if unknown_labels:
            raise ValueError(
                f"PartialAnnotations labels {unknown_labels} are not in class_names "
                f"{self.class_names}."
            )
        self.partial_ignore_class_ids: Sequence[int] = sorted(
            {self.class_names.index(label) for label in partial_ignore_labels}
        )

    @property
    def num_classes(self) -> int:
        """Number of detection classes."""
        return len(self.class_names)

    def build_annotated_class_masks(
        self,
        gt_traffic_cone_barrier_bbox_status: Bool[torch.Tensor, " batch_size"] | None,
    ) -> Bool[torch.Tensor, "batch_size num_classes"]:
        """Build the annotated mask of a batch from the dataset flag.

        Args:
            gt_traffic_cone_barrier_bbox_status: Per-sample flag, True when the partially ignored
                classes are annotated in that sample.

        Returns:
            True where a class is annotated in a sample: everywhere, except the partially ignored
            classes of the samples flagged False.

        Raises:
            ValueError: If the batch carries no flag. Partial ignoring was configured, so
                silently supervising every class would teach the head that the ignored classes
                are absent from the unannotated samples.
        """
        if gt_traffic_cone_barrier_bbox_status is None:
            raise ValueError(
                "PartialAnnotations is configured for "
                f"{[self.class_names[i] for i in self.partial_ignore_class_ids]} but the batch "
                "carries no gt_traffic_cone_barrier_bbox_status. Either the database has no "
                "annotation status for its scenarios or the flag was dropped on the way to the "
                "model; remove partial_annotations to supervise every class."
            )
        status = gt_traffic_cone_barrier_bbox_status.to(dtype=torch.bool)
        if status.ndim != 1:
            raise ValueError(
                "gt_traffic_cone_barrier_bbox_status must have shape (batch_size,), got "
                f"{tuple(status.shape)}."
            )
        batch_size = status.shape[0]
        annotated_class_masks = torch.ones(
            (batch_size, self.num_classes), dtype=torch.bool, device=status.device
        )
        # (batch_size, 1) unannotated samples broadcast over the ignored class columns.
        annotated_class_masks[:, list(self.partial_ignore_class_ids)] &= status.unsqueeze(1)
        return annotated_class_masks
