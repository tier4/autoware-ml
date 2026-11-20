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

"""Metric lifecycle for models.

``MetricEvalMixin`` is mixed into ``BaseModel`` and drives the validation and
test metric lifecycle for a list of :class:`~autoware_ml.metrics.base.MetricSuite`
objects. A model only implements ``build_eval_output``, returning the typed
:class:`~autoware_ml.metrics.eval_output.EvalOutput`. The mixin checks every
suite's requirements against the first batch, resets each suite at epoch start,
calls ``update`` per batch, and ``result`` at epoch end, logging under
``{split}/{prefix}/{key}``.

Each suite is cloned per stage and registered as a submodule, so Lightning moves
its state to the right device. torchmetrics owns the cross-GPU sync, which runs
inside ``result`` at epoch end.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import torch.nn as nn

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.metrics.base import EvalStage, MetricSuite
from autoware_ml.metrics.eval_output import EvalOutput
from autoware_ml.metrics.report import MetricReport

logger = logging.getLogger(__name__)


def resolve_metric_suites(
    metric_suites: Sequence[MetricSuite] | None, metrics: Sequence[MetricSuite] | None
) -> Sequence[MetricSuite] | None:
    """The metric suites a model was given, under their name or the deprecated one.

    Models took their suites as ``metrics``, which reads as if it held metric
    components. The argument and the config key are now ``metric_suites``;
    ``metrics`` keeps working until it is removed, and logs a warning so a config
    that still uses it shows up in the run's output.

    Args:
        metric_suites: Suites under their current name.
        metrics: Suites under the deprecated name.

    Returns:
        Whichever of the two was given, ``None`` when neither was.

    Raises:
        ValueError: If both were given.
    """
    if metrics is None:
        return metric_suites
    if metric_suites is not None:
        raise ValueError(
            "Got metric suites as both `metric_suites` and `metrics`. Pass them as "
            "`metric_suites` only, `metrics` is its deprecated name."
        )
    logger.warning(
        "Passing metric suites as `metrics` is deprecated, pass them as `metric_suites` "
        "(in a model config: `metric_suites: ${metric_suites}`)."
    )
    return metrics


class MetricEvalMixin:
    """Owns the metric suites and the validation/test epoch lifecycle."""

    def __init__(
        self,
        *args: Any,
        metric_suites: Sequence[MetricSuite] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
        **kwargs: Any,
    ) -> None:
        """Clone the metric suites per stage and register them as submodules.

        Args:
            metric_suites: Suites attached from config. Empty means only losses are
                logged.
            metrics: Deprecated name of ``metric_suites``, still accepted with a logged
                warning.
            *args: Positional arguments forwarded to the next base.
            **kwargs: Keyword arguments forwarded to the next base.
        """
        super().__init__(*args, **kwargs)
        metric_suites = resolve_metric_suites(metric_suites, metrics)
        prototypes = list(metric_suites) if metric_suites else []
        for suite in prototypes:
            if not suite.prefix:
                raise ValueError(
                    f"{type(suite).__name__} has an empty prefix, every suite attached "
                    "to a model needs one so its log keys are namespaced."
                )
        prefixes = [suite.prefix for suite in prototypes]
        duplicates = sorted({prefix for prefix in prefixes if prefixes.count(prefix) > 1})
        if duplicates:
            raise ValueError(
                f"Two suites share the prefix(es) {duplicates}, their keys would merge "
                "into one namespace. Set a distinct prefix per suite."
            )
        # A suite is cloned only for stages it actually reports at, so a heavy
        # test-only suite (and its deep-copied providers) never exists at val.
        self._suites_by_stage = nn.ModuleDict(
            {
                stage.value: nn.ModuleList(
                    [
                        self._stage_clone(suite, stage)
                        for suite in prototypes
                        if suite.runs_at(stage)
                    ]
                )
                for stage in (EvalStage.VAL, EvalStage.TEST)
            }
        )

    @staticmethod
    def _stage_clone(suite: MetricSuite, stage: EvalStage) -> MetricSuite:
        clone = suite.clone()
        clone.bind_stage(stage)
        return clone

    def build_eval_output(self, batch: ModelBatchInputs, outputs: Any) -> EvalOutput:
        """Map raw forward outputs and the batch to the typed eval output the metrics read.

        Override in a model that attaches metrics. The default produces an empty
        output, which is correct for a model with no metrics.

        Args:
            batch: Model inputs as fed to the model.
            outputs: Raw forward outputs.

        Returns:
            The eval output the attached metric suites read.
        """
        return EvalOutput()

    def _stage_suites(self, stage: EvalStage) -> nn.ModuleList:
        return self._suites_by_stage[stage.value]

    def on_validation_epoch_start(self) -> None:
        """Reset the validation metric state for a fresh epoch."""
        for suite in self._stage_suites(EvalStage.VAL):
            suite.reset()

    def on_test_epoch_start(self) -> None:
        """Reset the test metric state for a fresh epoch."""
        for suite in self._stage_suites(EvalStage.TEST):
            suite.reset()

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Accumulate one validation batch into every suite.

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
        """Accumulate one test batch into every suite.

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
        suites = self._stage_suites(stage)
        if not len(suites):
            return
        raw_outputs = (
            outputs["model_outputs"]
            if isinstance(outputs, Mapping) and "model_outputs" in outputs
            else outputs
        )
        eval_output = self.build_eval_output(batch, raw_outputs)
        if batch_idx == 0:
            self._check_requirements(list(suites), eval_output)
        for suite in suites:
            suite.update(eval_output)

    def _check_requirements(self, suites: list[MetricSuite], eval_output: EvalOutput) -> None:
        for suite in suites:
            missing = suite.requirements().check_requirements(eval_output)
            if missing:
                raise ValueError(
                    f"Suite {type(suite).__name__!r} needs {list(missing)}, not produced by "
                    f"{type(self).__name__}.build_eval_output."
                )

    def _log_metrics(self, stage: EvalStage) -> None:
        report = MetricReport()
        for suite in self._stage_suites(stage):
            report.merge(suite.result(stage), prefix=suite.prefix)
        if not len(report):
            return
        # Values are already global and identical on every rank after sync, so no sync_dist.
        self.log_dict(report.to_log_dict(stage.value), on_step=False, on_epoch=True, logger=True)
