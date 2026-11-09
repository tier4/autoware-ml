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

from collections.abc import Iterator

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

    def scoped(
        self,
        range_suffix: str = "",
        filter_name: str = "",
        view: str = "",
        prefix: str = "",
    ) -> MetricKey:
        """
        Place the key one level further out.

        A level is set once, on the way out from the metric to the log, so a key that
        already carries a level is a layering bug and raises.

        Args:
          range_suffix: Radial window to place the key in.
          filter_name: Filter slice to place the key in.
          view: Taxonomy view to place the key in.
          prefix: Suite namespace to place the key in.

        Returns:
          MetricKey: The key with the given levels set.
        """
        if range_suffix and self.range_suffix:
            raise ValueError(f"{self} already carries a range suffix, a level is set once.")
        if filter_name and self.filter_name:
            raise ValueError(f"{self} already carries a filter name, a level is set once.")
        if view and self.view:
            raise ValueError(f"{self} already carries a view, a level is set once.")
        if prefix and self.prefix:
            raise ValueError(f"{self} already carries a prefix, a level is set once.")
        return MetricKey(
            name=self.name,
            range_suffix=range_suffix or self.range_suffix,
            filter_name=filter_name or self.filter_name,
            view=view or self.view,
            prefix=prefix or self.prefix,
        )


class MetricReport:
    """
    Ordered metric values keyed by :class:`MetricKey`, one value per rendered name.

    A metric builds its own report with :meth:`record`. A suite merges the reports of
    its metrics with :meth:`merge`, placing their keys in a filter slice and a range
    window, and the model merges the suites under their prefixes. Every level rejects
    a duplicate rendered name, so a collision anywhere on the way to the log fails loud
    instead of overwriting a value.
    """

    def __init__(self) -> None:
        """Start an empty report."""
        # The duplicate check needs a lookup by rendered name, kept private: a report is
        # read through its keys, never as a mapping.
        self._keys: dict[str, MetricKey] = {}
        self._values: dict[str, float] = {}

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
        rendered = str(key)
        if rendered in self._values:
            raise ValueError(
                f"Two metrics emit the same key {rendered!r}. Give one a distinct name, "
                "filter name or suite prefix."
            )
        self._keys[rendered] = key
        self._values[rendered] = float(value)

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
        for key, value in other.items():
            self.add(
                key.scoped(
                    range_suffix=range_suffix, filter_name=filter_name, view=view, prefix=prefix
                ),
                value,
            )

    def keys(self) -> tuple[MetricKey, ...]:
        """
        The keys in report order.

        Returns:
          tuple[MetricKey, ...]: The keys.
        """
        return tuple(self._keys.values())

    def names(self) -> tuple[str, ...]:
        """
        The rendered names in report order.

        Returns:
          tuple[str, ...]: The rendered names.
        """
        return tuple(self._values)

    def values(self) -> tuple[float, ...]:
        """
        The values in report order.

        Returns:
          tuple[float, ...]: The values.
        """
        return tuple(self._values.values())

    def items(self) -> Iterator[tuple[MetricKey, float]]:
        """
        The keys and their values in report order.

        Yields:
          tuple[MetricKey, float]: One key and its value.
        """
        for rendered, key in self._keys.items():
            yield key, self._values[rendered]

    def to_log_dict(self, split: str) -> dict[str, float]:
        """
        The payload Lightning's ``log_dict`` takes, every name under the split.

        Args:
          split: The split being logged, for example ``val``.

        Returns:
          dict[str, float]: Rendered ``{split}/{name}`` to value.
        """
        return {f"{split}/{rendered}": value for rendered, value in self._values.items()}

    def __getitem__(self, key: MetricKey | str) -> float:
        """Look a value up by its key or its rendered name."""
        return self._values[str(key)]

    def __contains__(self, key: object) -> bool:
        """Whether a key or a rendered name is reported."""
        return str(key) in self._values

    def __iter__(self) -> Iterator[MetricKey]:
        """Iterate over the keys in report order."""
        return iter(self._keys.values())

    def __len__(self) -> int:
        """The number of reported values."""
        return len(self._values)

    def __repr__(self) -> str:
        """Render the report as its names and values."""
        return f"MetricReport({self._values!r})"
