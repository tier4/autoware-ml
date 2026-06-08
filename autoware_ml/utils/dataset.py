import numpy as np
import numpy.typing as npt
from pyquaternion import Quaternion


def convert_quaternion_to_matrix(
    rotation_quaternion: Quaternion,
    translation: npt.NDArray[np.float64] | None = None,
    convert_to_float32: bool = False,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:  # (4, 4)
    """
    Convert a translation and quaternion to a 4x4 transformation matrix.

    Args:
        rotation: Quaternion to represent the rotation.
        translation (3x1 or None): Translation to represent the translation.
        convert_to_float32: Whether to convert the result to float32.
    Returns:
        npt.NDArray[np.float32] | npt.NDArray[np.float64]: 4x4 transformation matrix.
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
