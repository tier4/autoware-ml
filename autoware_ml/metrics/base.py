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

import numpy as np
import torch.distributed
import torchmetrics

logger = logging.getLogger(__name__)


def rank_zero() -> bool:
    """True on the single process that should emit log lines (or when not DDP).

    Returns:
        Whether this process should log.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


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


def number_token(value: float) -> str:
    """Key-safe token for a plain number, e.g. ``99p5`` for 99.5.

    Args:
        value: Number to encode.

    Returns:
        The token, with ``.`` written as ``p`` and a ``minus`` prefix for negatives.
    """
    return f"{float(value):g}".replace("-", "minus").replace(".", "p")


def _distance_token(distance: float) -> str:
    return f"{number_token(distance)}m"


def range_suffix(metric_range: MetricRange) -> str:
    """Collision-free key suffix for a range, e.g. ``0m_50m`` or ``90m_inf``.

    Args:
        metric_range: Radial distance window in meters.

    Returns:
        The ``<min>m_<max>m`` suffix, an unbounded maximum encoded as ``inf``.
    """
    lower = _distance_token(metric_range.min_distance)
    if metric_range.max_distance is None:
        return f"{lower}_inf"
    return f"{lower}_{_distance_token(metric_range.max_distance)}"


class MetricFilter(ABC):
    """An optional, per-metric selector applied before a metric runs.

    A filter is an *axis*, not a metric: it decides which accumulated elements
    (detection boxes, segmentation points) a metric sees, exactly as the range
    bins do. ``name`` prefixes the metric's keys (empty for the identity filter).
    ``cache_key`` groups components that share a filter so the suite builds the
    filtered sub-state once. ``required_eval_keys`` names the per-frame data the
    filter needs (e.g. the ego pose), folded into the suite's fail-loud check.
    """

    name: str = ""
    required_eval_keys: tuple[str, ...] = ()

    @property
    def cache_key(self) -> str:
        """Value key grouping equivalent filters. Identity groups share ``""``."""
        return self.name

    @abstractmethod
    def keep(self, xyz: Any, context: dict[str, Any]) -> Any:
        """Boolean mask over elements to retain.

        Args:
            xyz: Bare base_link points ``(N, 3)`` or full detection box rows
                ``(N, 7+)``. Concrete filters dispatch on the column count so a
                box's whole footprint (not just its center) can be tested.
            context: Per-frame values named in ``required_eval_keys`` (e.g.
                ``ego2global``, ``scene_token``), supplied by the suite from the
                model's eval-output.

        Returns:
            Boolean mask over the input elements, true for those to keep.
        """

    def available(self, context: dict[str, Any]) -> bool:
        """Whether this filter can evaluate the given frame.

        A filter that needs external data absent for some frames (e.g. a lanelet
        map missing for a scene) returns ``False`` so the suite excludes that
        frame from this filter's bucket (the unfiltered metrics still see it)
        and counts the exclusion. The default filter always applies. This is an
        explicit availability check by design: a resource that *should* exist but
        is corrupt must still fail loud inside ``keep``, never be caught here.

        Args:
            context: Per-frame values named in ``required_eval_keys``.

        Returns:
            Whether the frame enters this filter's bucket.
        """
        return True


class IdentityFilter(MetricFilter):
    """The default filter. Keeps everything, adds no key prefix."""

    name = ""

    def keep(self, xyz: Any, context: dict[str, Any]) -> Any:
        """All-true mask over the given elements.

        Args:
            xyz: Point or box rows ``(N, ...)``.
            context: Per-frame values, unused here.

        Returns:
            Boolean mask of ``N`` true values.
        """
        return np.ones((len(xyz),), dtype=bool)


IDENTITY = IdentityFilter()


StateT = TypeVar("StateT")


class Metric(ABC, Generic[StateT]):
    """One injectable metric. Owns its computation and holds no accumulated state.

    A metric is a stateless strategy: it reads the synced ``state`` the suite
    builds and returns its slice of the report. ``stages`` declares when it runs
    and is configurable, so the same metric can be light at validation and full
    at test purely from config. ``required_eval_keys`` names any extra keys the
    metric needs in the model's eval-output, folded into the suite's own
    required-key check so a misconfigured metric fails loud at the first batch.
    ``filter`` is an optional region/selection axis (default ``None`` means
    whole-scene) by which the suite groups components and prefixes their keys.
    ``needs_ttc`` declares that the metric reads the per-box collision TTC, so
    a suite only runs its collision provider at stages where such a metric is
    active (TTC is expensive and test-only in practice). ``needs_boxes``
    declares that the metric reads per-frame detection ground truth, so a suite
    can demand those annotations instead of caching empty placeholders.
    """

    required_eval_keys: tuple[str, ...] = ()
    needs_ttc: bool = False
    needs_boxes: bool = False

    def __init__(
        self,
        stages: tuple[str, ...] | list[str] = ("val", "test"),
        filter: MetricFilter | None = None,
    ) -> None:
        """Store the stages this metric runs in and its optional filter.

        Args:
            stages: Stage names this metric reports for. Each must name an
                :class:`EvalStage` value (``"val"`` or ``"test"``), an unknown
                name raises ``ValueError``.
            filter: Optional selection axis (e.g. a lanelet ``RegionFilter``).
                ``None`` means the whole-scene metric.
        """
        self.stages: frozenset[EvalStage] = frozenset(EvalStage(stage) for stage in stages)
        self.filter: MetricFilter = filter if filter is not None else IDENTITY

    @abstractmethod
    def evaluate(self, state: StateT, stage: EvalStage) -> dict[str, float]:
        """Compute this metric's keys from the suite's synced ``state``.

        Args:
            state: State for one (filter, range) bucket.
            stage: Evaluation stage being reported.

        Returns:
            Metric keys mapped to values.
        """


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
        prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Store the injected component metrics and ranges.

        Args:
            components: Metrics that compose this suite, run against its state.
                Empty logs nothing, which is almost always a misconfiguration.
            ranges: Radial windows. Every metric key is also emitted per range
                with a distance suffix.
            prefix: Overrides the class-level log prefix. Needed when two suites
                of the same class run side by side (e.g. an occlusion-split
                detection suite) so their keys land in distinct namespaces.
                The eval mixin fails loud on a prefix collision.
            **kwargs: Forwarded to ``torchmetrics.Metric`` (for example
                ``sync_on_compute``).
        """
        super().__init__(**kwargs)
        if prefix is not None:
            self.prefix = str(prefix)
        self.components = list(components)
        self.ranges = tuple(ranges)
        self._stage = EvalStage.TEST
        # Availability-gated filters (lanelet region / collision area) register here so
        # the base can log per-filter frame coverage, empty means nothing to gate.
        self._coverage_filters: list[MetricFilter] = []
        if not self.components:
            logger.warning(
                "%s was constructed with no components, so it will log nothing.",
                type(self).__name__,
            )

        suffixes = [range_suffix(metric_range) for metric_range in self.ranges]
        duplicates = sorted(suffix for suffix in set(suffixes) if suffixes.count(suffix) > 1)
        if duplicates:
            raise ValueError(f"Range metric suffixes must be unique: {duplicates}")

    def bind_stage(self, stage: EvalStage) -> None:
        """Fix the stage this suite instance serves (the mixin clones per stage).

        Accumulation and required-key checks are stage-aware: a validation clone
        must not demand data that only test-stage components consume.

        Args:
            stage: Evaluation stage this instance accumulates and reports for.
        """
        self._stage = EvalStage(stage)

    def active_components(self) -> list[Metric[StateT]]:
        """Components that report at this suite instance's stage.

        Returns:
            Components whose configured stages include the bound stage.
        """
        return [component for component in self.components if self._stage in component.stages]

    def _filterable_components(self) -> list[Metric[StateT]]:
        """Components whose filters define this suite's filter axis.

        Subclasses that serve additional component lists from the same
        accumulation (e.g. a grouped view) override this so those filters are
        registered too.
        """
        return self.components

    def _register_component_filters(self) -> None:
        """Collect the distinct non-identity filters and their coverage counters.

        Fills ``_component_filters`` (the ordered distinct filters) and
        ``_filter_indices`` (filter cache key to its position). When any filter
        exists, the per-filter frame-coverage counters are registered and wired
        into the coverage log, so an availability-gated slice reported over fewer
        scenes than the whole-scene metrics is never silent. Subclasses call this
        once from ``__init__`` and build their own mask or bucket states on top.
        """
        self._component_filters: list[MetricFilter] = []
        self._filter_indices: dict[str, int] = {}
        for component in self._filterable_components():
            metric_filter = component.filter
            if metric_filter is IDENTITY:
                continue
            if not metric_filter.name:
                raise ValueError(
                    f"{type(metric_filter).__name__} on {type(component).__name__} has an "
                    "empty name, it would silently be applied as the identity."
                )
            if metric_filter.cache_key not in self._filter_indices:
                self._filter_indices[metric_filter.cache_key] = len(self._component_filters)
                self._component_filters.append(metric_filter)
        if self._component_filters:
            self._coverage_filters = self._component_filters
            self.add_state(
                "region_frames_seen",
                default=torch.zeros(len(self._component_filters), dtype=torch.long),
                dist_reduce_fx="sum",
            )
            self.add_state(
                "region_frames_covered",
                default=torch.zeros(len(self._component_filters), dtype=torch.long),
                dist_reduce_fx="sum",
            )

    def _filter_index(self, metric_filter: MetricFilter) -> int | None:
        """Registered position of a filter, ``None`` for the identity.

        An unregistered filter is a configuration bug (its masks were never
        accumulated) and raises.
        """
        if not metric_filter.name:
            return None
        if metric_filter.cache_key not in self._filter_indices:
            raise ValueError(f"Unregistered filter {metric_filter.cache_key!r}.")
        return self._filter_indices[metric_filter.cache_key]

    def _stage_filter_keys(self) -> set[str]:
        """Cache keys of filters used by components active at this stage."""
        return {
            component.filter.cache_key
            for component in self._filterable_components()
            if self._stage in component.stages and component.filter.name
        }

    def required_keys(self) -> tuple[str, ...]:
        """Eval-output keys needed at this suite instance's stage.

        Returns:
            The required eval-output key names.
        """
        extra = tuple(
            key
            for component in self.active_components()
            for key in (*component.required_eval_keys, *component.filter.required_eval_keys)
        )
        return tuple(dict.fromkeys(tuple(type(self)._required_keys) + extra))

    @abstractmethod
    def update(self, eval_out: dict[str, Any]) -> None:
        """Accumulate one batch into the suite's state.

        Args:
            eval_out: Flat eval-output dict built by the model for one batch.
        """

    @abstractmethod
    def state_for(self, metric_range: MetricRange | None, metric_filter: MetricFilter) -> StateT:
        """Build the state the metrics consume, for a range window and a filter.

        Args:
            metric_range: Radial window in meters, or ``None`` for no clipping.
            metric_filter: Spatial filter selecting the bucket's elements.

        Returns:
            State restricted to the requested (filter, range) bucket.
        """

    def compute(self) -> dict[str, float]:
        """Build state and run metrics across the filter and range axes.

        Components are grouped by filter so each filtered sub-state (and its
        memoized matching) is built once and shared by every component using it.

        Returns:
            Metric keys mapped to values, across all components and buckets.
        """
        self._log_coverage()
        return self._compute_report(self.active_components(), self.state_for)

    def _compute_report(self, components: list[Metric[StateT]], state_builder) -> dict[str, float]:
        """Run ``components`` across the filter and range axes of ``state_builder``.

        ``state_builder(range, filter)`` supplies the state. Parameterizing it
        lets a suite serve a second taxonomy view (e.g. behaviour groups) from
        the same accumulation.
        """
        report: dict[str, float] = {}
        groups: dict[str, tuple[MetricFilter, list[Metric[StateT]]]] = {}
        for component in components:
            metric_filter = component.filter
            groups.setdefault(metric_filter.cache_key, (metric_filter, []))[1].append(component)
        for metric_filter, group_components in groups.values():
            for key, value in self._run_group(
                metric_filter, group_components, state_builder
            ).items():
                # Groups are keyed by cache_key but keys carry the filter's
                # display name: two same-name filters with different parameters
                # must not silently overwrite each other across groups.
                if key in report:
                    raise ValueError(
                        f"Two filter groups emit the same key {key!r}. Give one "
                        "filter a distinct name."
                    )
                report[key] = value
        return report

    def _note_frame_coverage(self, index: int, available: bool) -> None:
        """Tally one frame for an availability-gated filter.

        ``region_frames_seen`` / ``region_frames_covered`` are DDP-summed counters
        the suite registers alongside its filters. The caller invokes this only for
        filters active at this stage, so ``seen`` counts the frames the slice could
        have used and ``covered`` the frames it did.
        """
        self.region_frames_seen[index] += 1
        if available:
            self.region_frames_covered[index] += 1

    def _log_coverage(self) -> None:
        """Log per-filter frame coverage once per epoch.

        Availability-gated filters exclude frames whose scene has no lanelet map.
        Coverage is always reported (``covered / seen``) so it is clear how much of
        the data a slice used. When frames were dropped it escalates to a warning,
        so a slice reported over fewer scenes than the whole-scene metrics is never
        silent.
        """
        if not self._coverage_filters or not rank_zero():
            return
        seen = self.region_frames_seen.tolist()
        covered = self.region_frames_covered.tolist()
        for metric_filter, n_seen, n_covered in zip(
            self._coverage_filters, seen, covered, strict=True
        ):
            if not n_seen:
                continue
            if n_covered < n_seen:
                logger.warning(
                    "Filter %r ran on %d/%d frames, %d frame(s) in scenes without a "
                    "lanelet map were excluded from its slice (whole-scene metrics keep them).",
                    metric_filter.name,
                    n_covered,
                    n_seen,
                    n_seen - n_covered,
                )
            else:
                logger.info(
                    "Filter %r ran on %d/%d frames (full map coverage).",
                    metric_filter.name,
                    n_covered,
                    n_seen,
                )

    def _run_group(
        self, metric_filter: MetricFilter, components: list[Metric[StateT]], state_builder
    ) -> dict[str, float]:
        windows: list[tuple[MetricRange | None, str]] = [(None, "")]
        windows += [(metric_range, range_suffix(metric_range)) for metric_range in self.ranges]
        report: dict[str, float] = {}
        for metric_range, suffix in windows:
            state = state_builder(metric_range, metric_filter)
            for component in components:
                if self._stage not in component.stages:
                    continue
                for name, value in component.evaluate(state, self._stage).items():
                    # Key names come from the component's OWN filter: filters are
                    # grouped by cache_key (equal masks), but two same-mask
                    # filters may carry different display names.
                    key = self._compose_key(component.filter, name, suffix)
                    if key in report:
                        raise ValueError(
                            f"Two metrics emit the same key {key!r}. Give one a distinct name."
                        )
                    report[key] = value
        return report

    @staticmethod
    def _compose_key(metric_filter: MetricFilter, name: str, suffix: str) -> str:
        base = f"{name}_{suffix}" if suffix else name
        return f"{metric_filter.name}/{base}" if metric_filter.name else base

    def runs_at(self, stage: EvalStage) -> bool:
        """Whether any component reports at ``stage``.

        Lets the mixin skip both accumulation and reporting for a suite that is
        inert at this stage, so a heavy test-only suite never fills during
        validation.

        Args:
            stage: Evaluation stage to probe.

        Returns:
            Whether at least one component reports at ``stage``.
        """
        return any(stage in component.stages for component in self.components)

    def result(self, stage: EvalStage) -> dict[str, float]:
        """Set the reporting stage and compute. torchmetrics syncs inside compute.

        Call once per epoch on a freshly ``reset`` suite. The mixin clones a suite
        per stage, so a single instance only ever reports for one stage.

        Args:
            stage: Evaluation stage to report for.

        Returns:
            Metric keys mapped to values.
        """
        self._stage = stage
        return self.compute()
