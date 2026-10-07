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

"""Checkpoint writer that never leaves a half-written file behind."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lightning.pytorch.plugins.io import TorchCheckpointIO


class AtomicCheckpointIO(TorchCheckpointIO):
    """Write checkpoints to a temporary sibling and rename them into place.

    Lightning's default writer relies on the filesystem layer for atomicity; on the
    network filesystems of a cluster a job killed mid-write can leave a truncated
    ``last.ckpt``. A rename within one directory is atomic on POSIX filesystems,
    NFS included, so a reader only ever sees the previous or the complete file.
    """

    TMP_SUFFIX = ".tmp"

    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: str | Path, storage_options: Any | None = None
    ) -> None:
        """Save the checkpoint atomically.

        Args:
            checkpoint: Checkpoint payload.
            path: Destination file.
            storage_options: Unsupported by the torch writer; forwarded for its error.
        """
        final_path = Path(path)
        tmp_path = final_path.with_name(final_path.name + self.TMP_SUFFIX)
        super().save_checkpoint(checkpoint, tmp_path, storage_options)
        os.replace(tmp_path, final_path)

    def remove_checkpoint(self, path: str | Path) -> None:
        """Remove a checkpoint and any leftover temporary file of it.

        Args:
            path: Checkpoint file to delete.
        """
        final_path = Path(path)
        tmp_path = final_path.with_name(final_path.name + self.TMP_SUFFIX)
        if tmp_path.exists():
            super().remove_checkpoint(tmp_path)
        if final_path.exists() or final_path.is_symlink():
            super().remove_checkpoint(final_path)
