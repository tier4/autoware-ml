"""Fork-shareable storage for large annotation sample lists.

Datasets that hold annotations as a plain ``list[dict]`` inflate multi-worker
training memory: every DataLoader worker forked from the main process touches
the objects' reference counts while reading them, which dirties the
copy-on-write pages and gradually privatizes the whole annotation list in every
worker. Storing the samples as one contiguous pickle buffer removes the
per-object refcounts, so all workers keep sharing the parent's pages.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Any

import numpy as np


class SerializedSampleList(Sequence):
    """Immutable sample list stored as one pickle buffer plus offsets.

    Drop-in replacement for a ``list[dict]`` that only needs ``len()`` and
    integer indexing. ``__getitem__`` deserializes a single record on demand
    and returns a fresh object, so mutations of returned samples do not
    persist. Build it as the last step of dataset ``__init__``, after any
    full-pass computations over the live list.
    """

    def __init__(self, samples: Sequence[dict[str, Any]]) -> None:
        """Serialize *samples* into one contiguous buffer.

        Args:
            samples: Annotation records to store.
        """
        offsets = np.zeros(len(samples) + 1, dtype=np.int64)
        parts: list[bytes] = []
        for index, sample in enumerate(samples):
            part = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
            parts.append(part)
            offsets[index + 1] = offsets[index] + len(part)
        self._offsets = offsets
        self._buffer = b"".join(parts)

    def __len__(self) -> int:
        """Return the number of stored samples."""
        return len(self._offsets) - 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Deserialize and return the sample at *index*.

        Args:
            index: Integer sample index; negative indices are supported.

        Returns:
            A fresh deserialized copy of the stored sample.
        """
        if isinstance(index, slice):
            raise TypeError("SerializedSampleList does not support slicing.")
        length = len(self)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError(f"Index {index} out of range for {length} samples.")
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        return pickle.loads(memoryview(self._buffer)[start:end])
