"""Tests of the typed metric keys and the report that collects them."""

from __future__ import annotations

import pytest

from autoware_ml.metrics.report import MetricKey, MetricReport


def test_key_renders_the_set_levels_only() -> None:
    assert str(MetricKey(name="mAP")) == "mAP"
    assert str(MetricKey(name="mAP", range_suffix="0m_50m")) == "mAP_0m_50m"
    key = MetricKey(
        name="mAP", range_suffix="0m_50m", filter_name="region_road", view="grouped", prefix="det3d"
    )
    assert str(key) == "det3d/grouped/region_road/mAP_0m_50m"


def test_scoped_sets_a_level_once() -> None:
    key = MetricKey(name="mAP").scoped(filter_name="region_road")
    assert key.filter_name == "region_road"
    assert key.scoped(prefix="det3d").prefix == "det3d"
    with pytest.raises(ValueError, match="already carries"):
        key.scoped(filter_name="corridor")


def test_report_records_and_looks_up_by_key_or_name() -> None:
    report = MetricReport()
    report.record("mAP", 0.5)
    report.record("mAP_car", 0.25)
    assert report["mAP"] == 0.5
    assert report[MetricKey(name="mAP_car")] == 0.25
    assert "mAP" in report
    assert "mAPH" not in report
    assert report.names() == ("mAP", "mAP_car")
    assert report.keys() == (MetricKey(name="mAP"), MetricKey(name="mAP_car"))
    assert report.values() == (0.5, 0.25)
    assert len(report) == 2


def test_report_rejects_a_duplicate_name() -> None:
    report = MetricReport()
    report.record("mAP", 0.5)
    with pytest.raises(ValueError, match="same key 'mAP'"):
        report.record("mAP", 0.6)


def test_report_rejects_keys_that_render_alike() -> None:
    report = MetricReport()
    report.add(MetricKey(name="mAP_0m_50m"), 0.5)
    with pytest.raises(ValueError, match="same key 'mAP_0m_50m'"):
        report.add(MetricKey(name="mAP", range_suffix="0m_50m"), 0.6)


def test_merge_places_the_keys_one_level_out() -> None:
    metric = MetricReport()
    metric.record("mAP", 0.5)
    suite = MetricReport()
    suite.merge(metric)
    suite.merge(metric, range_suffix="0m_50m", filter_name="region_road")
    assert suite.names() == ("mAP", "region_road/mAP_0m_50m")
    assert suite["region_road/mAP_0m_50m"] == 0.5
    assert metric.names() == ("mAP",)

    model = MetricReport()
    model.merge(suite, prefix="det3d")
    assert model.to_log_dict("val") == {
        "val/det3d/mAP": 0.5,
        "val/det3d/region_road/mAP_0m_50m": 0.5,
    }


def test_merge_rejects_a_collision_across_reports() -> None:
    first = MetricReport()
    first.record("num_frames", 1.0)
    second = MetricReport()
    second.record("num_frames", 2.0)
    suite = MetricReport()
    suite.merge(first)
    with pytest.raises(ValueError, match="same key 'num_frames'"):
        suite.merge(second)


def test_items_pair_each_key_with_its_value() -> None:
    report = MetricReport()
    report.record("acc", 1.0)
    report.record("mIoU", 0.75)
    assert list(report.items()) == [(MetricKey(name="acc"), 1.0), (MetricKey(name="mIoU"), 0.75)]
    assert [key.name for key in report] == ["acc", "mIoU"]
