"""Metric lifecycle for models.

``MetricEvalMixin`` is mixed into ``BaseModel`` and drives the validation and
test metric lifecycle for a list of :class:`~autoware_ml.metrics.base.MetricSuite`
objects. A model only implements ``build_eval_output``. The mixin resets each
suite at epoch start, calls ``update`` per batch, and ``result`` at epoch end,
logging under ``{split}/{prefix}/{key}``.

Each suite is cloned per stage and registered as a submodule, so Lightning moves
its state to the right device. torchmetrics owns the cross-GPU sync, which runs
inside ``result`` at epoch end.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn

from autoware_ml.metrics.base import EvalStage, MetricSuite


class MetricEvalMixin:
    """Owns the metric suites and the validation/test epoch lifecycle."""

    def __init__(
        self, *args: Any, metrics: Sequence[MetricSuite] | None = None, **kwargs: Any
    ) -> None:
        """Clone the metric suites per stage and register them as submodules.

        Args:
            metrics: Suites attached from config. Empty means only losses are
                logged.
            *args: Positional arguments forwarded to the next base.
            **kwargs: Keyword arguments forwarded to the next base.
        """
        super().__init__(*args, **kwargs)
        prototypes = list(metrics) if metrics else []
        for metric in prototypes:
            if not metric.prefix:
                raise ValueError(
                    f"{type(metric).__name__} has an empty prefix, every suite attached "
                    "to a model needs one so its log keys are namespaced."
                )
        prefixes = [metric.prefix for metric in prototypes]
        duplicates = sorted({prefix for prefix in prefixes if prefixes.count(prefix) > 1})
        if duplicates:
            raise ValueError(
                f"Two suites share the prefix(es) {duplicates}, their keys would merge "
                "into one namespace. Set a distinct prefix per suite."
            )
        # A suite is cloned only for stages it actually reports at, so a heavy
        # test-only suite (and its deep-copied providers) never exists at val.
        self._metrics_by_stage = nn.ModuleDict(
            {
                stage.value: nn.ModuleList(
                    [
                        self._stage_clone(metric, stage)
                        for metric in prototypes
                        if metric.runs_at(stage)
                    ]
                )
                for stage in (EvalStage.VAL, EvalStage.TEST)
            }
        )

    @staticmethod
    def _stage_clone(metric: MetricSuite, stage: EvalStage) -> MetricSuite:
        clone = metric.clone()
        clone.bind_stage(stage)
        return clone

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Map raw forward outputs and the batch to the flat dict metrics read.

        Override in a model that attaches metrics. The default produces nothing,
        which is correct for a model with no metrics.

        Args:
            batch: Collated batch as fed to the model.
            outputs: Raw forward outputs.

        Returns:
            Flat dict the attached metric suites read.
        """
        return {}

    def _stage_metrics(self, stage: EvalStage) -> nn.ModuleList:
        return self._metrics_by_stage[stage.value]

    def on_validation_epoch_start(self) -> None:
        """Reset the validation metric state for a fresh epoch."""
        for metric in self._stage_metrics(EvalStage.VAL):
            metric.reset()

    def on_test_epoch_start(self) -> None:
        """Reset the test metric state for a fresh epoch."""
        for metric in self._stage_metrics(EvalStage.TEST):
            metric.reset()

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Accumulate one validation batch into every metric.

        Args:
            outputs: Raw forward outputs of the batch.
            batch: Collated batch as fed to the model.
            batch_idx: Batch index within the epoch.
            dataloader_idx: Dataloader index, part of the Lightning signature.
        """
        self._update_metrics(EvalStage.VAL, outputs, batch, batch_idx)

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Accumulate one test batch into every metric.

        Args:
            outputs: Raw forward outputs of the batch.
            batch: Collated batch as fed to the model.
            batch_idx: Batch index within the epoch.
            dataloader_idx: Dataloader index, part of the Lightning signature.
        """
        self._update_metrics(EvalStage.TEST, outputs, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        """Combine, compute, and log the validation metrics."""
        self._log_metrics(EvalStage.VAL)

    def on_test_epoch_end(self) -> None:
        """Combine, compute, and log the test metrics."""
        self._log_metrics(EvalStage.TEST)

    def _update_metrics(self, stage: EvalStage, outputs: Any, batch: Any, batch_idx: int) -> None:
        metrics = self._stage_metrics(stage)
        if not len(metrics):
            return
        raw_outputs = (
            outputs["model_outputs"]
            if isinstance(outputs, Mapping) and "model_outputs" in outputs
            else outputs
        )
        eval_out = self.build_eval_output(batch, raw_outputs)
        if batch_idx == 0:
            self._check_required_keys(list(metrics), eval_out)
        for metric in metrics:
            metric.update(eval_out)

    def _check_required_keys(self, metrics: list, eval_out: Mapping[str, Any]) -> None:
        for metric in metrics:
            missing = [key for key in metric.required_keys() if key not in eval_out]
            if missing:
                raise ValueError(
                    f"Metric {type(metric).__name__!r} needs {missing}, not produced by "
                    f"{type(self).__name__}.build_eval_output."
                )

    def _log_metrics(self, stage: EvalStage) -> None:
        metrics = self._stage_metrics(stage)
        report: dict[str, float] = {}
        for metric in metrics:
            for name, value in metric.result(stage).items():
                key = f"{stage.value}/{metric.prefix}/{name}"
                if key in report:
                    raise ValueError(
                        f"Two metrics log the same key {key!r}. Set a distinct prefix."
                    )
                report[key] = value
        if not report:
            return
        # Values are already global and identical on every rank after sync, so no sync_dist.
        self.log_dict(report, on_step=False, on_epoch=True, logger=True)
