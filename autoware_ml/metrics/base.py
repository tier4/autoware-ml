"""Core metric framework shared across tasks.

Metrics are built on ``torchmetrics``. The design separates two concerns:

* A **suite** (:class:`MetricSuite`, a ``torchmetrics.Metric``) owns the
  accumulated state and its cross-GPU reduction via ``add_state``. It knows
  nothing about which metrics run. It only builds a task ``state`` object and
  hands it to whatever metrics were injected, once overall and once per range.
* A **metric** (:class:`Metric`) is a small, self-contained, injectable object
  that computes its own numbers from that state in ``evaluate`` and declares the
  stages it runs in.

This module holds only the task-agnostic pieces: the lifecycle ``EvalStage``,
the radial ``MetricRange`` used for distance buckets, and the two base classes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

import torchmetrics

logger = logging.getLogger(__name__)


class EvalStage(str, Enum):
    """Stage a metric reports for. Metrics run at validation and test only."""

    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class MetricRange:
    """Radial distance window in meters used to bucket metrics by range.

    Attributes:
        name: Human-readable label, kept for config clarity (not used in keys).
        min_distance: Inclusive lower bound in meters.
        max_distance: Exclusive upper bound in meters, or ``None`` for unbounded.
    """

    name: str
    min_distance: float
    max_distance: float | None


def _distance_token(distance: float) -> str:
    token = f"{float(distance):g}".replace("-", "minus").replace(".", "p")
    return f"{token}m"


def range_suffix(metric_range: MetricRange) -> str:
    """Collision-free key suffix for a range, e.g. ``0m_50m`` or ``90m_inf``."""
    lower = _distance_token(metric_range.min_distance)
    if metric_range.max_distance is None:
        return f"{lower}_inf"
    return f"{lower}_{_distance_token(metric_range.max_distance)}"


StateT = TypeVar("StateT")


class Metric(ABC, Generic[StateT]):
    """One injectable metric. Owns its computation; holds no accumulated state.

    A metric is a stateless strategy: it reads the synced ``state`` the suite
    builds and returns its slice of the report. ``stages`` declares when it runs
    and is configurable, so the same metric can be light at validation and full
    at test purely from config.
    """

    def __init__(self, stages: tuple[str, ...] | list[str] = ("val", "test")) -> None:
        """Store the stages this metric runs in.

        Args:
            stages: Stage names this metric reports for. Each must name an
                :class:`EvalStage` value (``"val"`` or ``"test"``); an unknown
                name raises ``ValueError``.
        """
        self.stages: frozenset[EvalStage] = frozenset(EvalStage(stage) for stage in stages)

    @abstractmethod
    def evaluate(self, state: StateT, stage: EvalStage) -> dict[str, float]:
        """Compute this metric's keys from the suite's synced ``state``."""


class MetricSuite(torchmetrics.Metric, ABC, Generic[StateT]):
    """Task state-engine that composes injected metrics.

    The suite owns the accumulated state and its cross-GPU sync (via
    ``add_state``), and the per-range dispatch. It does not decide which metrics
    run: that list is injected. At ``compute`` it builds the state once overall
    and once per range, then asks each stage-applicable metric to ``evaluate``.

    Subclasses implement the task-specific parts only: ``add_state`` (in their
    ``__init__``), ``update``, and ``state_for``.
    """

    prefix: str = ""
    _required_keys: tuple[str, ...] = ()

    full_state_update: bool = False

    def __init__(
        self,
        components: list[Metric[StateT]],
        ranges: tuple[MetricRange, ...] = (),
        **kwargs: Any,
    ) -> None:
        """Store the injected component metrics and ranges.

        Args:
            components: Metrics that compose this suite, run against its state.
                Empty logs nothing, which is almost always a misconfiguration.
            ranges: Radial windows. Every metric key is also emitted per range
                with a distance suffix.
            **kwargs: Forwarded to ``torchmetrics.Metric`` (for example
                ``sync_on_compute``).
        """
        super().__init__(**kwargs)
        self.components = list(components)
        self.ranges = tuple(ranges)
        self._stage = EvalStage.TEST
        if not self.components:
            logger.warning(
                "%s was constructed with no components, so it will log nothing.",
                type(self).__name__,
            )

        suffixes = [range_suffix(metric_range) for metric_range in self.ranges]
        duplicates = sorted(suffix for suffix in set(suffixes) if suffixes.count(suffix) > 1)
        if duplicates:
            raise ValueError(f"Range metric suffixes must be unique: {duplicates}")

    @abstractmethod
    def update(self, eval_out: dict[str, Any]) -> None:
        """Accumulate one batch into the suite's state."""

    @abstractmethod
    def state_for(self, metric_range: MetricRange | None) -> StateT:
        """Build the state the metrics consume, overall (``None``) or per range."""

    def compute(self) -> dict[str, float]:
        """Build state and run every stage-applicable metric, overall and per range."""
        report = self._run(self.state_for(None), suffix="")
        for metric_range in self.ranges:
            range_report = self._run(self.state_for(metric_range), range_suffix(metric_range))
            report.update(range_report)
        return report

    def _run(self, state: StateT, suffix: str) -> dict[str, float]:
        report: dict[str, float] = {}
        for component in self.components:
            if self._stage not in component.stages:
                continue
            for name, value in component.evaluate(state, self._stage).items():
                key = f"{name}_{suffix}" if suffix else name
                if key in report:
                    raise ValueError(
                        f"Two metrics emit the same key {key!r}. Give one a distinct name."
                    )
                report[key] = value
        return report

    def result(self, stage: EvalStage) -> dict[str, float]:
        """Set the reporting stage and compute. torchmetrics syncs inside compute.

        Call once per epoch on a freshly ``reset`` suite. The mixin clones a suite
        per stage, so a single instance only ever reports for one stage.
        """
        self._stage = stage
        return self.compute()
