#!/usr/bin/env python3
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

"""Write the per database scenario lists of the 10 Hz pseudo label corpus.

    write_pseudo_10hz_scenario_lists.py <corpus root> [--val-every 40] \
        [--holdout <perception-devops scenario dir> ...]

Reads ``DATASET_SCENES.csv`` of the corpus and writes ``scenario_lists/<db>.yaml`` next
to it, one file per source database, in the format the T4 scenario configs read:
``train``/``val``/``test`` lists of ``<scene>/<version>``. A scene whose ``data`` link is
missing (its source scene is not on this host) is left out. About one scene in
``--val-every`` is held out for validation and another one in ``--val-every`` for test,
chosen by a hash of the scene id so that a scene recorded in several databases lands in
the same split everywhere and the two hold-outs are disjoint (the splitter rejects a
scene listed in two splits). The pseudo boxes are only a sanity signal there; the
segmentation is validated on ground truth elsewhere.

The corpus labels every frame of its scenes, and for a scene with human boxes those boxes
are the labels of its keyframes. A scene that is a validation or test scene of another
benchmark must therefore never be trained on: ``--holdout`` takes scenario list
directories (for example the ``detection3d`` and ``segmentation3d`` lists of
perception-devops) and moves every scene of their ``val`` and ``test`` splits out of the
training split, into the test split here unless its hash bucket already makes it a
validation scene.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import yaml


class Split(StrEnum):
    """Split of a scenario list."""

    TRAIN = auto()
    VAL = auto()
    TEST = auto()


@dataclass(frozen=True)
class CorpusEntries:
    """
    Scene entries of the corpus, grouped by their source database.

    Attributes:
      per_database: Source database name to its ``<scene>/<version>`` entries, in corpus order.
      skipped: Number of scenes left out because their source data is missing.
    """

    per_database: dict[str, list[str]]
    skipped: int


class PseudoScenarioListWriter:
    """Writes the per database scenario lists of the 10 Hz pseudo label corpus."""

    def __init__(self, root: Path, val_every: int, holdout_dirs: Sequence[Path]) -> None:
        """
        Initialize the writer.

        Args:
          root: Corpus root holding ``DATASET_SCENES.csv``.
          val_every: Hold out about one scene in this many for validation, and as many for test.
          holdout_dirs: Scenario list directories whose val and test scenes must not be trained
            on.
        """

        self._root = root
        self._val_every = val_every
        self._holdout_dirs = list(holdout_dirs)
        self._held_out = self.read_holdout_scenes(self._holdout_dirs)

    @property
    def out_dir(self) -> Path:
        """Directory the scenario lists are written to."""
        return self._root / "scenario_lists"

    @staticmethod
    def read_holdout_scenes(directories: Sequence[Path]) -> set[str]:
        """
        Scene ids of the val and test splits of every scenario list in the directories.

        Args:
          directories: Scenario list directories.

        Returns:
          set[str]: Scene ids that must not be trained on.
        """

        scenes: set[str] = set()
        for directory in directories:
            for path in sorted(directory.glob("*.yaml")):
                lists = yaml.safe_load(path.read_text()) or {}
                for split in (Split.VAL, Split.TEST):
                    scenes.update(entry.split("/")[0] for entry in lists.get(split) or [])
        return scenes

    def read_entries(self) -> CorpusEntries:
        """
        Read the scene entries of the corpus, leaving out the scenes without source data.

        Returns:
          CorpusEntries: Entries of every source database and the number of skipped scenes.
        """

        per_database: dict[str, list[str]] = defaultdict(list)
        skipped = 0
        with (self._root / "DATASET_SCENES.csv").open(newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                if not (self._root / row["key"] / "data").exists():
                    skipped += 1
                    continue
                per_database[row["db"]].append(f"{row['scene']}/{row['version']}")
        return CorpusEntries(per_database=dict(per_database), skipped=skipped)

    def split_of(self, entry: str) -> Split:
        """
        Split of one scene entry.

        Hash bucket 0 is validation and bucket 1 test; a scene of another bucket is a test
        scene when another benchmark holds it out, a training scene otherwise.

        Args:
          entry: Scene entry, ``<scene>/<version>``.

        Returns:
          Split: Split the entry belongs to.
        """

        scene = entry.split("/")[0]
        bucket = int(hashlib.sha1(scene.encode()).hexdigest(), 16) % self._val_every
        if bucket == 0:
            return Split.VAL
        if bucket == 1 or scene in self._held_out:
            return Split.TEST
        return Split.TRAIN

    def split(self, entries: Sequence[str]) -> dict[Split, list[str]]:
        """
        Split the entries of one database, keeping their order within every split.

        Args:
          entries: Scene entries of the database.

        Returns:
          dict[Split, list[str]]: Entries of every split, train, val and test in this order.
        """

        splits: dict[Split, list[str]] = {split: [] for split in Split}
        for entry in entries:
            splits[self.split_of(entry)].append(entry)
        return splits

    def write(self, database: str, splits: dict[Split, list[str]]) -> None:
        """
        Write the scenario list of one database.

        Args:
          database: Source database name, the stem of the written file.
          splits: Entries of every split.
        """

        holdout_dirs = [str(path) for path in self._holdout_dirs]
        lines = [
            f"# Scenario list of {database} in the pseudo_10hz corpus: "
            "<scenario_id>/<version> per split.",
            "# Point layout ['x', 'y', 'z', 'intensity', 'ring'], "
            "semantic masks: True (pseudo labels).",
            "# Written by scripts/datasets/write_pseudo_10hz_scenario_lists.py; val and test are "
            f"two disjoint 1-in-{self._val_every} hash buckets of the scene id,",
            f"# test also holds the val/test scenes of {holdout_dirs}.",
        ]
        for split, split_entries in splits.items():
            lines.append(f"{split}:" + (" []" if not split_entries else ""))
            lines.extend(f"- {entry}" for entry in split_entries)
        with (self.out_dir / f"{database}.yaml").open("w") as list_file:
            list_file.write("\n".join(lines) + "\n")

    def run(self) -> None:
        """Write the scenario list of every database of the corpus and print a summary."""
        corpus = self.read_entries()
        self.out_dir.mkdir(exist_ok=True)
        totals = {split: 0 for split in Split}
        for database, entries in sorted(corpus.per_database.items()):
            splits = self.split(entries)
            for split, split_entries in splits.items():
                totals[split] += len(split_entries)
            self.write(database, splits)
        print(
            f"{len(self._held_out)} held-out benchmark scenes; "
            f"{len(corpus.per_database)} databases, {totals[Split.TRAIN]} train, "
            f"{totals[Split.VAL]} val and {totals[Split.TEST]} test scenes written "
            f"to {self.out_dir}; {corpus.skipped} scenes without source data skipped."
        )


def main() -> int:
    """
    Parse the command line and write the scenario lists.

    Returns:
      int: Exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="corpus root holding DATASET_SCENES.csv")
    parser.add_argument("--val-every", type=int, default=40, help="hold out about one scene in n")
    parser.add_argument(
        "--holdout",
        type=Path,
        nargs="*",
        default=[],
        help="scenario list directories whose val and test scenes must not be trained on",
    )
    args = parser.parse_args()
    PseudoScenarioListWriter(args.root, args.val_every, args.holdout).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
