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

"""Unit tests for the typed metric keys and the report that collects them."""

import unittest

from autoware_ml.metrics.report import MetricKey, MetricReport


class TestMetricKey(unittest.TestCase):
    """A key renders its levels in one place and takes each level once."""

    def test_renders_the_set_levels_only(self) -> None:
        """
        Input: a bare key, a key with a range suffix, and a key with every level set.
        Expected: the rendered name joins the set levels with slashes, appends the range suffix
        with an underscore, and leaves the empty levels out.
        Check: the three rendered names.
        """
        self.assertEqual(str(MetricKey(name="mAP")), "mAP")
        self.assertEqual(str(MetricKey(name="mAP", range_suffix="0m_50m")), "mAP_0m_50m")
        key = MetricKey(
            name="mAP",
            range_suffix="0m_50m",
            filter_name="region_road",
            view="grouped",
            prefix="det3d",
        )
        self.assertEqual(str(key), "det3d/grouped/region_road/mAP_0m_50m")

    def test_scoped_sets_a_level_once(self) -> None:
        """
        Input: a bare key placed in the region_road filter slice, then in the det3d prefix, then
        in the corridor filter slice again.
        Expected: each free level is set on a copy, and a level that is already set is a layering
        bug and is refused.
        Check: the filter name and prefix of the copies, ValueError on the second filter name.
        """
        key = MetricKey(name="mAP").scoped(filter_name="region_road")
        self.assertEqual(key.filter_name, "region_road")
        self.assertEqual(key.scoped(prefix="det3d").prefix, "det3d")
        with self.assertRaisesRegex(ValueError, "already carries"):
            key.scoped(filter_name="corridor")


class TestMetricReport(unittest.TestCase):
    """A report keeps one value per key and refuses a second."""

    def test_records_and_reads_back_by_key(self) -> None:
        """
        Input: a report with mAP and mAP_car recorded in that order.
        Expected: the keys come back in record order and each key reads its value, a key that
        was never recorded is refused.
        Check: the keys, the two values, the length and KeyError on the absent key.
        """
        report = MetricReport()
        report.record("mAP", 0.5)
        report.record("mAP_car", 0.25)
        self.assertEqual(report.metric_names(), (MetricKey(name="mAP"), MetricKey(name="mAP_car")))
        self.assertEqual(report.metric_value(MetricKey(name="mAP")), 0.5)
        self.assertEqual(report.metric_value(MetricKey(name="mAP_car")), 0.25)
        self.assertEqual(len(report), 2)
        with self.assertRaisesRegex(KeyError, "mAPH is not reported"):
            report.metric_value(MetricKey(name="mAPH"))

    def test_rejects_a_duplicate_key(self) -> None:
        """
        Input: mAP recorded twice in one report.
        Expected: the second record is refused instead of overwriting the first.
        Check: ValueError naming the key.
        """
        report = MetricReport()
        report.record("mAP", 0.5)
        with self.assertRaisesRegex(ValueError, "same key 'mAP'"):
            report.record("mAP", 0.6)

    def test_log_dict_rejects_keys_that_render_alike(self) -> None:
        """
        Input: a key named mAP_0m_50m and a key named mAP in the 0m_50m range window.
        Expected: the two keys differ as objects, so both are held, but they would log under one
        name, so the log payload is refused.
        Check: the report length and ValueError naming the log name.
        """
        report = MetricReport()
        report.add(MetricKey(name="mAP_0m_50m"), 0.5)
        report.add(MetricKey(name="mAP", range_suffix="0m_50m"), 0.6)
        self.assertEqual(len(report), 2)
        with self.assertRaisesRegex(ValueError, "same log name 'val/mAP_0m_50m'"):
            report.to_log_dict("val")

    def test_merge_places_the_keys_one_level_out(self) -> None:
        """
        Input: a metric report holding mAP, merged into a suite report as is and once more in the
        region_road slice and the 0m_50m window, then the suite report merged into a model report
        under the det3d prefix.
        Expected: each merge places the keys one level further out, the merged report is left
        unchanged, and the log payload carries the split in front of every rendered name.
        Check: the suite keys and value, the untouched metric keys, and the log payload.
        """
        metric = MetricReport()
        metric.record("mAP", 0.5)
        suite = MetricReport()
        suite.merge(metric)
        suite.merge(metric, range_suffix="0m_50m", filter_name="region_road")
        placed = MetricKey(name="mAP", range_suffix="0m_50m", filter_name="region_road")
        self.assertEqual(suite.metric_names(), (MetricKey(name="mAP"), placed))
        self.assertEqual(suite.metric_value(placed), 0.5)
        self.assertEqual(metric.metric_names(), (MetricKey(name="mAP"),))

        model = MetricReport()
        model.merge(suite, prefix="det3d")
        self.assertEqual(
            model.to_log_dict("val"),
            {"val/det3d/mAP": 0.5, "val/det3d/region_road/mAP_0m_50m": 0.5},
        )

    def test_merge_rejects_a_collision_across_reports(self) -> None:
        """
        Input: two metric reports that both hold num_frames, merged into one suite report.
        Expected: the second merge is refused, two metrics must not report under one key.
        Check: ValueError naming the key.
        """
        first = MetricReport()
        first.record("num_frames", 1.0)
        second = MetricReport()
        second.record("num_frames", 2.0)
        suite = MetricReport()
        suite.merge(first)
        with self.assertRaisesRegex(ValueError, "same key 'num_frames'"):
            suite.merge(second)


if __name__ == "__main__":
    unittest.main()
