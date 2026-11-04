"""Integer identities of annotated objects."""

import hashlib
from typing import Sequence

import numpy as np
import numpy.typing as npt


def hash_instance_ids(instance_ids: Sequence[str]) -> npt.NDArray[np.int64]:
    """
    Map annotation instance tokens to stable non-negative integers.

    Args:
      instance_ids: Instance token of every box.

    Returns:
      npt.NDArray[np.int64]: One 63-bit hash per token, equal for equal tokens in every process.
    """
    return np.asarray(
        [
            int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
            & 0x7FFF_FFFF_FFFF_FFFF
            for token in instance_ids
        ],
        dtype=np.int64,
    )
