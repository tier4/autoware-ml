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

"""Unit tests for how a model hands its metric suites to the eval mixin."""

import unittest

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite
from autoware_ml.metrics.eval_mixin import MetricEvalMixin


_LOGGER = "autoware_ml.metrics.eval_mixin"


class _Host(MetricEvalMixin):
    """The mixin on its own, without a Lightning module around it."""


def _suites() -> list[Detection3DMetricSuite]:
    return [Detection3DMetricSuite(components=[MeanAP(thresholds=(0.5,))])]


class TestMetricSuitesArgument(unittest.TestCase):
    """The metric_suites argument, and metrics as its deprecated name."""

    def test_suites_attach_without_a_warning(self) -> None:
        """
        Input: one detection suite passed as ``metric_suites``.
        Expected: the suite is attached for validation and test, and nothing is logged.
        Check: one suite with the det3d prefix at each stage, no log record.
        """
        with self.assertNoLogs(_LOGGER):
            host = _Host(metric_suites=_suites())

        for stage in (EvalStage.VAL, EvalStage.TEST):
            self.assertEqual([suite.prefix for suite in host._stage_suites(stage)], ["det3d"])

    def test_metrics_still_attach_with_a_logged_warning(self) -> None:
        """
        Input: the same suite passed under the deprecated name ``metrics``.
        Expected: it is attached exactly as under ``metric_suites``, and a logged warning names
        the new key, so a config that still uses the old one shows up in the run's output.
        Check: one suite with the det3d prefix at each stage, and the warning record.
        """
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            host = _Host(metrics=_suites())

        self.assertEqual(len(logs.records), 1)
        self.assertIn("`metric_suites`", logs.records[0].getMessage())

        for stage in (EvalStage.VAL, EvalStage.TEST):
            self.assertEqual([suite.prefix for suite in host._stage_suites(stage)], ["det3d"])

    def test_both_names_at_once_are_refused(self) -> None:
        """
        Input: suites passed as both ``metric_suites`` and ``metrics``.
        Expected: the model refuses to pick one, so a config cannot silently drop a list.
        Check: ValueError naming both arguments.
        """
        with self.assertRaisesRegex(ValueError, r"`metric_suites` and `metrics`"):
            _Host(metric_suites=_suites(), metrics=_suites())

    def test_no_suites_logs_only_losses(self) -> None:
        """
        Input: neither name given.
        Expected: no suite is attached at any stage, and nothing is logged.
        Check: empty suite lists at validation and test, no log record.
        """
        with self.assertNoLogs(_LOGGER):
            host = _Host()

        for stage in (EvalStage.VAL, EvalStage.TEST):
            self.assertEqual(len(host._stage_suites(stage)), 0)


if __name__ == "__main__":
    unittest.main()
