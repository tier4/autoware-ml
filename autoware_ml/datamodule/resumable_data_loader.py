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

"""Training dataloader that resumes a checkpoint at the next untrained batch."""

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

from torch.utils.data import DataLoader


class _SkipBatches:
    """Index sampler that starts past the batches its loader skips."""

    def __init__(self, loader: "ResumableDataLoader", index_sampler: Iterable[Any]) -> None:
        """Wrap the index sampler of one loader.

        Args:
            loader: Loader whose pending skip applies.
            index_sampler: Batch sampler, or sampler without automatic batching.
        """
        self.loader = loader
        self.index_sampler = index_sampler

    def __len__(self) -> int:
        """Full epoch length, so that Lightning counts the skipped batches as done."""
        return len(self.index_sampler)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator:
        """Draw the epoch's indices and drop the skipped batches without loading them."""
        return islice(iter(self.index_sampler), self.loader._skip, None)


class ResumableDataLoader(DataLoader):
    """DataLoader whose position inside an epoch is saved in checkpoints.

    Lightning saves ``state_dict`` with every checkpoint and passes it to ``load_state_dict``
    before a resumed run iterates. The resumed epoch then skips the batches it had already
    yielded and continues with the next one. The sampler must derive its order from
    ``set_epoch`` (every ``DistributedSampler`` does), so the skipped batches are the trained
    ones. Lightning does not read ahead of a dataloader with a length, so yielded is trained.
    """

    _skip = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the loader.

        Args:
            *args: Positional ``DataLoader`` arguments.
            **kwargs: Keyword ``DataLoader`` arguments.
        """
        super().__init__(*args, **kwargs)
        self._epoch = 0
        self._yielded = 0
        self._resume_state: dict[str, int] | None = None

    @property
    def _index_sampler(self) -> _SkipBatches:
        # DataLoader iterators draw their indices from this private property.
        return _SkipBatches(self, super()._index_sampler)

    def __iter__(self) -> Iterator[Any]:
        """Yield the batches of the current epoch, after the trained ones on resume."""
        # A generator runs from the first batch on, after Lightning's set_epoch.
        epoch = getattr(self.sampler, "epoch", 0)
        skip = 0
        if self._resume_state is not None and self._resume_state["epoch"] == epoch:
            skip = self._resume_state["batches"]
        self._resume_state = None
        self._epoch, self._yielded = epoch, skip

        self._skip = skip
        try:
            iterator = super().__iter__()
        finally:
            self._skip = 0
        for batch in iterator:
            self._yielded += 1
            yield batch

    def state_dict(self) -> dict[str, int]:
        """Return the epoch and the number of its batches yielded so far.

        Returns:
            dict[str, int]: ``epoch`` and ``batches``.
        """
        return {"epoch": self._epoch, "batches": self._yielded}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        """Skip the yielded batches when the saved epoch is iterated next.

        Args:
            state_dict: State returned by ``state_dict``.
        """
        self._resume_state = dict(state_dict)
