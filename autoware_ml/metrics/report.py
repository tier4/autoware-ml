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

"""Typed metric keys and the report that collects them.

A metric names its values by bare names such as ``mAP`` or ``mAP_car``. The suite
reports the same metric once per filter slice and once per range window, a
grouped view adds its own level, and the model logs every suite under its own
prefix. :class:`MetricKey` holds these levels as fields and renders the log name
in one place, so no caller composes a key by string formatting.
:class:`MetricReport` collects the keyed values and rejects a duplicate at every
level, so two metrics, two filters, two views, or two suites can never silently
overwrite each other.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MetricKey(BaseModel):
    """
    Where one metric value is reported.

    Rendered as ``{prefix}/{view}/{filter_name}/{name}_{range_suffix}`` with every
    empty level left out, for example ``det3d/region_road/mAP_0m_50m``.

    Attributes:
      name: The metric's own name, for example ``mAP`` or ``mAP_car``.
      range_suffix: Radial window the value covers, for example ``0m_50m``, empty for the
        unbinned value.
      filter_name: Display name of the filter whose slice the value covers, empty for the
        whole scene.
      view: Taxonomy view the value belongs to, for example ``grouped``, empty for the
        trained classes.
      prefix: Log namespace of the suite that reported the value, empty inside the suite.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    name: str
    range_suffix: str = ""
    filter_name: str = ""
    view: str = ""
    prefix: str = ""

    def __str__(self) -> str:
        """Render the log name."""
        base = f"{self.name}_{self.range_suffix}" if self.range_suffix else self.name
        return "/".join(
            level for level in (self.prefix, self.view, self.filter_name, base) if level
        )

    def update_keys(
        self,
        range_suffix: str = "",
        filter_name: str = "",
        view: str = "",
        prefix: str = "",
    ) -> MetricKey:
        """
        Return a new :class:`MetricKey` with the given level keys.

        Empty arguments leave their level unchanged. Setting a level the key already
        carries raises, since each layer sets its own level once.

        Args:
          range_suffix: Radial window to place the key in.
          filter_name: Filter slice to place the key in.
          view: Taxonomy view to place the key in.
          prefix: Suite namespace to place the key in.

        Returns:
          MetricKey: The key with the given levels set.
        """
        for level, requested, current in (
            ("range suffix", range_suffix, self.range_suffix),
            ("filter name", filter_name, self.filter_name),
            ("view", view, self.view),
            ("prefix", prefix, self.prefix),
        ):
            if requested and current:
                raise ValueError(f"{self} already carries a {level}, a level is set once.")
        return MetricKey(
            name=self.name,
            range_suffix=range_suffix or self.range_suffix,
            filter_name=filter_name or self.filter_name,
            view=view or self.view,
            prefix=prefix or self.prefix,
        )


class MetricReport:
    """
    Metric values keyed by :class:`MetricKey`, in record order.

    A metric builds its own report with :meth:`record`. A suite merges the reports of
    its metrics with :meth:`merge`, placing their keys in a filter slice and a range
    window, and the model merges the suites under their prefixes. A key is reported
    once: a second value under the same key fails loud instead of overwriting the first,
    and :meth:`to_log_dict` rejects two keys that render to one log name.
    """

    def __init__(self) -> None:
        """Start an empty report."""
        self._metrics: dict[MetricKey, float] = {}

    def record(self, name: str, value: float) -> None:
        """
        Add one value under a bare metric name.

        Args:
          name: The metric's own name, for example ``mAP_car``.
          value: The value.
        """
        self.add(MetricKey(name=name), value)

    def add(self, key: MetricKey, value: float) -> None:
        """
        Add one value under a key.

        Args:
          key: Where the value is reported.
          value: The value.
        """
        if key in self._metrics:
            raise ValueError(
                f"Two metrics emit the same key {str(key)!r}. Give one a distinct name, "
                "filter name or suite prefix."
            )
        self._metrics[key] = float(value)

    def merge(
        self,
        other: MetricReport,
        range_suffix: str = "",
        filter_name: str = "",
        view: str = "",
        prefix: str = "",
    ) -> None:
        """
        Add every value of another report, placing its keys one level further out.

        Args:
          other: The report to fold in, left unchanged.
          range_suffix: Radial window to place the keys in.
          filter_name: Filter slice to place the keys in.
          view: Taxonomy view to place the keys in.
          prefix: Suite namespace to place the keys in.
        """
        for key in other.metric_names():
            self.add(
                key.update_keys(
                    range_suffix=range_suffix, filter_name=filter_name, view=view, prefix=prefix
                ),
                other.metric_value(key),
            )

    def metric_names(self) -> tuple[MetricKey, ...]:
        """
        The keys in record order.

        Returns:
          tuple[MetricKey, ...]: The keys.
        """
        return tuple(self._metrics)

    def metric_value(self, key: MetricKey) -> float:
        """
        The value reported under a key.

        Args:
          key: The key to read.

        Returns:
          float: The value.
        """
        if key not in self._metrics:
            raise KeyError(
                f"{key} is not reported, the report holds {list(map(str, self._metrics))}."
            )
        return self._metrics[key]

    def to_log_dict(self, split: str) -> dict[str, float]:
        """
        The payload Lightning's ``log_dict`` takes, every rendered name under the split.

        Two keys that differ as objects but render to one name would silently share a log
        entry, so that collision is rejected here, at the only place the rendered name is
        what counts.

        Args:
          split: The split being logged, for example ``val``.

        Returns:
          dict[str, float]: Rendered ``{split}/{name}`` to value.
        """
        payload: dict[str, float] = {}
        for key, value in self._metrics.items():
            name = f"{split}/{key}"
            if name in payload:
                raise ValueError(
                    f"Two metric keys render to the same log name {name!r}. Give one a "
                    "distinct name."
                )
            payload[name] = value
        return payload

    def __len__(self) -> int:
        """The number of reported values."""
        return len(self._metrics)

    def __repr__(self) -> str:
        """Render the report as its names and values."""
        rendered = {str(key): value for key, value in self._metrics.items()}
        return f"MetricReport({rendered!r})"
