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

"""Integer identities of annotated objects."""

import hashlib
from collections.abc import Sequence

import numpy as np
from jaxtyping import Int64


def hash_instance_ids(instance_ids: Sequence[str]) -> Int64[np.ndarray, " num_ids"]:
    """
    Map annotation instance tokens to stable non-negative integers.

    Args:
      instance_ids: Instance token of every box.

    Returns:
      One 63-bit hash per token, equal for equal tokens in every process.
    """
    return np.asarray(
        [
            int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
            & 0x7FFF_FFFF_FFFF_FFFF
            for token in instance_ids
        ],
        dtype=np.int64,
    )
