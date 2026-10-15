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

"""Unit tests for the semantic mask paths of the nuScenes 3D segmentation task."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from autoware_ml.datamodule.nuscenes.segmentation3d import NuScenesSegmentation3DTask
from autoware_ml.datamodule.t4dataset.tests.test_segmentation3d import (
    TAXONOMY,
    build_records_dataframe,
)


class TestNuScenesSegmentation3DTask(unittest.TestCase):
    """The mask is read from the stored root-relative path joined onto the database root."""

    def test_reads_the_mask_under_roots_of_any_depth(self) -> None:
        """
        Input: nuScenes roots one and four directories deep, each holding a mask at
        <root>/lidarseg/v1.0-trainval/a.bin, recorded as lidarseg/v1.0-trainval/a.bin.
        Expected: the task reads the mask under both roots by joining the stored path onto
        the root, rather than rebuilding it from the six trailing directories a T4 scene has.
        Check: the resolved labels.
        """
        for depth in ("nuscenes", "workspace/data/sets/nuscenes"):
            with self.subTest(root=depth):
                root = Path(tempfile.mkdtemp()) / depth
                mask_path = root / "lidarseg" / "v1.0-trainval" / "a.bin"
                mask_path.parent.mkdir(parents=True)
                np.array([0, 1], dtype=np.uint8).tofile(mask_path)
                task = NuScenesSegmentation3DTask(
                    database_root_path=str(root),
                    dataset_records_dataframe=build_records_dataframe(
                        "lidarseg/v1.0-trainval/a.bin", {"car": 0, "tree": 1}
                    ),
                    taxonomy=TAXONOMY,
                )

                sample = task.get_data_sample(0)

                assert sample.segmentation3d_gt_sample is not None
                self.assertEqual(sample.segmentation3d_gt_sample.gt_semantic_mask.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
