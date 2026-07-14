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

"""Multi-source annotation-file specifications.

A dataset split can mix several annotation files with different supervision
coverage - e.g. a detection+segmentation set combined with a
segmentation-only set. Each source declares explicitly which supervision its
labels provide (``det3d``/``seg3d``) and how often its frames are repeated
(``repeat``), so nothing is inferred from the annotation content.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnnotationSource:
    """One annotation file with its declared supervision coverage.

    Attributes:
        path: Resolved annotation pkl path.
        det3d: Whether the source's detection annotations supervise training
            and evaluation. When ``False``, instances are dropped; box-less
            frames contribute no detection loss and neutral detection metric
            entries.
        seg3d: Whether the source's segmentation labels supervise training
            and evaluation. When ``False``, the per-frame category mapping is
            emptied so every point maps to the ignore index. The mask file is
            still loaded, so a mask path must exist for every frame.
        repeat: How many times the source's frames appear per epoch
            (physical repetition, i.e. oversampling).
    """

    path: str
    det3d: bool
    seg3d: bool
    repeat: int


_SOURCE_KEYS = ("path", "det3d", "seg3d", "repeat")


def _resolve_ann_path(path: str, data_root: str) -> str:
    """Resolve an annotation path relative to the dataset root."""
    return path if os.path.isabs(path) else os.path.join(data_root, path)


def _coerce_source_entry(entry: Any, data_root: str) -> AnnotationSource:
    """Validate and convert one explicit source specification."""
    if not isinstance(entry, Mapping):
        raise TypeError(
            f"Annotation source entries must be mappings with keys {_SOURCE_KEYS}, "
            f"got {type(entry)!r}."
        )
    missing = [key for key in _SOURCE_KEYS if key not in entry]
    unknown = [key for key in entry if key not in _SOURCE_KEYS]
    if missing or unknown:
        raise ValueError(
            f"Annotation source entries must define exactly {_SOURCE_KEYS}; "
            f"missing {missing}, unknown {unknown}."
        )
    repeat = int(entry["repeat"])
    if repeat < 1:
        raise ValueError(f"Annotation source repeat must be >= 1, got {repeat}.")
    return AnnotationSource(
        path=_resolve_ann_path(str(entry["path"]), data_root),
        det3d=bool(entry["det3d"]),
        seg3d=bool(entry["seg3d"]),
        repeat=repeat,
    )


def coerce_annotation_sources(ann_file: Any, data_root: str) -> list[AnnotationSource]:
    """Normalize an annotation-file setting to a list of sources.

    Args:
        ann_file: Either a single path (full det3d+seg3d supervision, no
            repetition - the pre-existing single-file behavior) or a sequence
            of explicit source mappings with exactly the keys ``path``,
            ``det3d``, ``seg3d``, and ``repeat``.
        data_root: Dataset root used to resolve relative paths.

    Returns:
        Ordered list of validated annotation sources.

    Raises:
        TypeError: If the value is neither a path nor a sequence of mappings.
        ValueError: If a source entry is incomplete or invalid, or the
            sequence is empty.
    """
    if isinstance(ann_file, str):
        return [
            AnnotationSource(
                path=_resolve_ann_path(ann_file, data_root), det3d=True, seg3d=True, repeat=1
            )
        ]
    if isinstance(ann_file, Sequence):
        sources = [_coerce_source_entry(entry, data_root) for entry in ann_file]
        if not sources:
            raise ValueError("Annotation source lists must contain at least one entry.")
        return sources
    raise TypeError(
        "Annotation files must be a path or a sequence of source mappings, "
        f"got {type(ann_file)!r}."
    )
