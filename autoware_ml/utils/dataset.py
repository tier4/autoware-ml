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

"""Conversion of a quaternion and translation to a transformation matrix."""

import numpy as np
from jaxtyping import Float32, Float64
from pyquaternion import Quaternion


def convert_quaternion_to_matrix(
    rotation_quaternion: Quaternion,
    translation: Float64[np.ndarray, " 3"] | None = None,
    convert_to_float32: bool = False,
) -> Float32[np.ndarray, "4 4"] | Float64[np.ndarray, "4 4"]:
    """
    Convert a translation and quaternion to a 4x4 transformation matrix.

    Args:
        rotation_quaternion: Quaternion to represent the rotation.
        translation: Translation to represent the translation, or None.
        convert_to_float32: Whether to convert the result to float32.
    Returns:
        Float32[np.ndarray, "4 4"] | Float64[np.ndarray, "4 4"]: 4x4 transformation matrix.
    """

    if not isinstance(rotation_quaternion, Quaternion):
        raise TypeError(
            "Rotation quaternion must be a Quaternion object, "
            f"but it's {type(rotation_quaternion).__name__}"
        )

    result = np.eye(4)
    result[:3, :3] = rotation_quaternion.rotation_matrix

    if translation is not None:
        if not isinstance(translation, np.ndarray):
            raise TypeError(
                f"Translation must be a numpy array or None, but it's {type(translation).__name__}"
            )

        if translation.shape != (3,):
            raise ValueError(
                f"Translation must be an array of shape (3,), got shape {translation.shape}"
            )

        result[:3, 3] = translation

    if convert_to_float32:
        return result.astype(np.float32)

    return result
