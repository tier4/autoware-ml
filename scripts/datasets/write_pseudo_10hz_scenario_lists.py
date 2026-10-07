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
from pathlib import Path

import yaml


def holdout_scenes(directories: list[Path]) -> set[str]:
    """Scene ids of the val and test splits of every scenario list in the directories."""
    scenes: set[str] = set()
    for directory in directories:
        for path in sorted(directory.glob("*.yaml")):
            lists = yaml.safe_load(path.read_text()) or {}
            for split in ("val", "test"):
                scenes.update(entry.split("/")[0] for entry in lists.get(split) or [])
    return scenes


def main() -> int:
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
    held_out = holdout_scenes(args.holdout)

    rows = list(csv.DictReader((args.root / "DATASET_SCENES.csv").open()))
    per_db: dict[str, list[str]] = defaultdict(list)
    skipped = 0
    for row in rows:
        entry = f"{row['scene']}/{row['version']}"
        if not (args.root / row["key"] / "data").exists():
            skipped += 1
            continue
        per_db[row["db"]].append(entry)

    out_dir = args.root / "scenario_lists"
    out_dir.mkdir(exist_ok=True)
    total = {"train": 0, "val": 0, "test": 0}

    def bucket(entry: str) -> int:
        scene = entry.split("/")[0]
        return int(hashlib.sha1(scene.encode()).hexdigest(), 16) % args.val_every

    for db, entries in sorted(per_db.items()):
        val = [entry for entry in entries if bucket(entry) == 0]
        test = [
            entry
            for entry in entries
            if bucket(entry) == 1 or (bucket(entry) > 1 and entry.split("/")[0] in held_out)
        ]
        train = [
            entry for entry in entries if bucket(entry) > 1 and entry.split("/")[0] not in held_out
        ]
        total["train"] += len(train)
        total["val"] += len(val)
        total["test"] += len(test)
        lines = [
            f"# Scenario list of {db} in the pseudo_10hz corpus: <scenario_id>/<version> per split.",
            "# Point layout ['x', 'y', 'z', 'intensity', 'ring'], semantic masks: True (pseudo labels).",
            f"# Written by scripts/datasets/write_pseudo_10hz_scenario_lists.py; val and test are two disjoint 1-in-{args.val_every} hash buckets of the scene id,",
            f"# test also holds the val/test scenes of {[str(path) for path in args.holdout]}.",
        ]
        for split, split_entries in (("train", train), ("val", val), ("test", test)):
            lines.append(f"{split}:" + (" []" if not split_entries else ""))
            lines.extend(f"- {entry}" for entry in split_entries)
        (out_dir / f"{db}.yaml").write_text("\n".join(lines) + "\n")
    print(
        f"{len(held_out)} held-out benchmark scenes; "
        f"{len(per_db)} databases, {total['train']} train, {total['val']} val and {total['test']} test scenes written "
        f"to {out_dir}; {skipped} scenes without source data skipped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
