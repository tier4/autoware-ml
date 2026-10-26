"""Number of evaluated frames."""

from __future__ import annotations

from typing import Protocol

from autoware_ml.metrics.base import EvalStage, Metric


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

    def evaluate(self, state: _CountsFrames, stage: EvalStage) -> dict[str, float]:
        """Count the accumulated frames.

        Args:
            state: Segmentation state for one (filter, range) bucket.
            stage: Evaluation stage requesting the metric.

        Returns:
            Mapping with the frame count.
        """
        return {"num_frames": float(state.num_frames)}
