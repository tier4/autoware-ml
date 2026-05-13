"""Minimal compatibility helpers for vendored RT-DETR modules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

GLOBAL_CONFIG: dict[str, Any] = defaultdict(dict)


def register(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    """Compatibility decorator used by upstream RT-DETR modules."""

    del args, kwargs

    def decorator(obj: Any) -> Any:
        return obj

    return decorator
