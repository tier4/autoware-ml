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

"""Find the checkpoint a relaunched job resumes from, and the run it belongs to."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class CheckpointLocator:
    """Newest loadable checkpoint of one checkpoint directory.

    ``last.ckpt`` is tried first when it exists (following a symlink), then every
    other ``*.ckpt`` file from the newest to the oldest by modification time. Files
    that do not load are skipped, so a checkpoint truncated by a kill never blocks
    the resume.
    """

    LAST_NAME = "last.ckpt"
    REQUIRED_KEYS = ("state_dict", "optimizer_states", "loops")

    def __init__(self, checkpoints_dir: Path, last_name: str = LAST_NAME) -> None:
        """Initialize the locator.

        Args:
            checkpoints_dir: Directory to search.
            last_name: Name of the pointer checkpoint written by the periodic callback.
        """
        self.checkpoints_dir = checkpoints_dir
        self.last_name = last_name

    def candidates(self) -> list[Path]:
        """Return the checkpoint files in the order a resume tries them.

        Returns:
            Resolved paths, ``last.ckpt``'s target first and the others newest first;
            empty when the directory does not exist.
        """
        if not self.checkpoints_dir.is_dir():
            return []
        candidates: list[Path] = []
        last = self.checkpoints_dir / self.last_name
        if last.exists():
            candidates.append(last.resolve())
        others = [
            path.resolve()
            for path in self.checkpoints_dir.glob("*.ckpt")
            if path.name != self.last_name and path.is_file()
        ]
        others.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        candidates.extend(path for path in others if path not in candidates)
        return candidates

    def find_latest(self) -> Path | None:
        """Return the newest loadable checkpoint, or ``None`` when there is none."""
        return next((path for path in self.candidates() if self.is_loadable(path)), None)

    @staticmethod
    def is_loadable(checkpoint_path: Path) -> bool:
        """Return whether a checkpoint file holds a complete, readable training state.

        A job killed while a file was being written can leave a truncated checkpoint
        behind; this reads it through and checks the keys a resume needs.

        Args:
            checkpoint_path: Candidate checkpoint file.

        Returns:
            ``True`` when the file loads and carries the training state.
        """
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        except Exception as error:  # noqa: BLE001 - any read error means "not this file"
            logger.warning("Checkpoint '%s' is not loadable (%s).", checkpoint_path, error)
            return False
        missing = [key for key in CheckpointLocator.REQUIRED_KEYS if key not in payload]
        if missing:
            logger.warning("Checkpoint '%s' misses %s; skipping it.", checkpoint_path, missing)
            return False
        return True


class RunPointer:
    """File that receives the checkpoint directory of the current run.

    A slurm launch script sets :attr:`ENV` to a path derived from the stable job id;
    the training entrypoint writes the checkpoint directory of its run there, and the
    relaunch after a requeue reads it back to pass ``--resume-latest``.
    """

    ENV = "AUTOWARE_ML_RUN_POINTER_FILE"

    def __init__(self, path: Path) -> None:
        """Initialize the pointer.

        Args:
            path: Pointer file.
        """
        self.path = path

    @classmethod
    def from_env(cls) -> RunPointer | None:
        """Return the pointer named by :attr:`ENV`, or ``None`` when it is not set."""
        pointer = os.environ.get(cls.ENV)
        return cls(Path(pointer)) if pointer else None

    @staticmethod
    def is_writer_process() -> bool:
        """Return whether this process is local rank 0 of node 0."""
        return os.environ.get("LOCAL_RANK", "0") == "0" and os.environ.get("NODE_RANK", "0") == "0"

    def write(self, checkpoints_dir: Path) -> None:
        """Write the checkpoint directory of the current run to the pointer file.

        Args:
            checkpoints_dir: Directory the run's checkpoints are written to.
        """
        # Lightning's subprocess launcher runs this entrypoint once per GPU; only the first
        # process of the first node writes, and through a temp name of its own: eight ranks
        # replacing one shared ".tmp" on a network filesystem lost the file under each other.
        if not self.is_writer_process():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        tmp_path.write_text(f"{checkpoints_dir}\n")
        os.replace(tmp_path, self.path)
        logger.info("Wrote the run pointer '%s' -> '%s'.", self.path, checkpoints_dir)
