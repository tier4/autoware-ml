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

"""CLI-facing dataset-generation command functions."""

from collections.abc import Sequence
from typing import Any

from autoware_ml.tools.dataset import generate_dataset


def main(
    dataset: str,
    tasks: Sequence[str],
    root_path: str,
    out_dir: str,
    **kwargs: Any,
) -> None:
    """Dispatch dataset generation from the CLI entrypoint."""
    generate_dataset(
        dataset=dataset,
        tasks=tasks,
        root_path=root_path,
        out_dir=out_dir,
        **kwargs,
    )
