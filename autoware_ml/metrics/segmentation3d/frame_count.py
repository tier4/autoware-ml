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

"""Number of evaluated frames."""

from __future__ import annotations

from typing import Protocol

from autoware_ml.metrics.base import EvalStage, Metric
from autoware_ml.metrics.report import MetricReport


class _CountsFrames(Protocol):
    """State exposing how many frames the suite accumulated into it."""

    @property
    def num_frames(self) -> int: ...


class FrameCount(Metric[_CountsFrames]):
    """Number of frames the other metrics scored.

    Reads ``num_frames`` off the state, so it runs on both the confusion-matrix
    suite and the point-cache suite. A region-filtered component reports only
    the frames its filter covered (a scene without a lanelet map is omitted from
    that slice). Range clipping removes points, never frames, so the count is
    range-invariant and reported once per slice: a frame with no points in a
    window still counts, it is a frame the model was evaluated on.
    """

    range_invariant = True

    def evaluate(self, state: _CountsFrames, stage: EvalStage) -> MetricReport:
        """Count the accumulated frames.

        Args:
            state: Segmentation state for one (filter, range) bucket.
            stage: Evaluation stage requesting the metric.

        Returns:
            The frame count.
        """
        report = MetricReport()
        report.record("num_frames", float(state.num_frames))
        return report
