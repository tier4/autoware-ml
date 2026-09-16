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
        rotation: Quaternion to represent the rotation.
        translation: Translation to represent the translation, or None.
        convert_to_float32: Whether to convert the result to float32.
    Returns:
        Float32[np.ndarray, "4 4"] | Float64[np.ndarray, "4 4"]: 4x4 transformation matrix.
    """

    assert isinstance(rotation_quaternion, Quaternion), (
        "Rotation quaternion must be a Quaternion object"
    )

    result = np.eye(4)
    result[:3, :3] = rotation_quaternion.rotation_matrix

    if translation is not None:
        assert isinstance(translation, np.ndarray), "Translation must be a numpy array or None"

        if translation.shape != (3,):
            raise ValueError(
                f"Translation must be an array of shape (3,), got shape {translation.shape}"
            )

        result[:3, 3] = translation

    if convert_to_float32:
        return result.astype(np.float32)

    return result
