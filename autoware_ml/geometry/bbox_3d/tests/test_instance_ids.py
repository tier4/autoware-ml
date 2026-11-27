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

"""Stable integer identities of annotation instance tokens."""

import unittest

from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids


class TestInstanceIds(unittest.TestCase):
    def test_equal_tokens_map_to_one_non_negative_id(self) -> None:
        ids = hash_instance_ids(["a3f1", "77c0", "a3f1"])

        self.assertEqual(ids.dtype.name, "int64")
        self.assertEqual(ids[0], ids[2])
        self.assertNotEqual(ids[0], ids[1])
        self.assertTrue((ids >= 0).all())
        # Independent of the process, unlike the salted built-in hash
        self.assertEqual(int(ids[0]), int(hash_instance_ids(["a3f1"])[0]))
