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
        self._metrics_by_stage = nn.ModuleDict(
            {
                EvalStage.VAL.value: nn.ModuleList([metric.clone() for metric in prototypes]),
                EvalStage.TEST.value: nn.ModuleList([metric.clone() for metric in prototypes]),
            }
        )

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Map raw forward outputs and the batch to the flat dict metrics read.

        Override in a model that attaches metrics. The default produces nothing,
        which is correct for a model with no metrics.
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
        """Accumulate one validation batch into every metric."""
        self._update_metrics(EvalStage.VAL, outputs, batch, batch_idx)

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Accumulate one test batch into every metric."""
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
            self._check_required_keys(metrics, eval_out)
        for metric in metrics:
            metric.update(eval_out)

    def _check_required_keys(self, metrics: nn.ModuleList, eval_out: Mapping[str, Any]) -> None:
        for metric in metrics:
            missing = [key for key in metric._required_keys if key not in eval_out]
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
