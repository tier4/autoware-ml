"""Per-frame evaluation metadata carried next to the model inputs.

The region and collision evaluation filters need two things the model inputs do not
carry: the pose of the frame in the map and a scene identifier the lanelet map provider
resolves to the scene's map. The dataset attaches them to every sample, the collation
stacks them, and the detection eval-output builders hand them to the metrics as
``ego2global`` and ``scene_token``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

from jaxtyping import Float32
import torch
from torch import Tensor

from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample


class FrameMetaSample(NamedTuple):
    """Evaluation metadata of one frame.

    Attributes:
        ego2global: 4x4 transform from the frame the points and boxes are expressed in (the
            main lidar frame) to the map frame. It is named after the ``ego2global`` key the
            metrics read; the lidar mounting is already composed in.
        scene_token: Scene identifier the lanelet map provider resolves to the scene's map.
    """

    ego2global: Float32[Tensor, "4 4"]
    scene_token: str


class FrameMetaBatch(NamedTuple):
    """Evaluation metadata of every frame of a batch.

    Attributes:
        ego2globals: Per-frame 4x4 transforms from the (augmented) lidar frame to the map frame.
        scene_tokens: Per-frame scene identifiers.
    """

    ego2globals: Float32[Tensor, "batch_size 4 4"]
    scene_tokens: Sequence[str]

    @staticmethod
    def collate_gt_samples(
        frame_meta_samples: Sequence[FrameMetaSample],
        lidar_transformation_samples: Sequence[LiDARTransformationSample | None],
    ) -> FrameMetaBatch:
        """Stack the frame metadata of a batch, following the lidar-space augmentations.

        The points and boxes of a sample live in the frame produced by its lidar-space
        augmentation, ``p_aug = T @ p_raw``, so the map transform of the augmented frame is
        ``ego2global @ inv(T)``. Samples without an augmentation keep their transform.

        Args:
            frame_meta_samples: Frame metadata of every sample, in batch order.
            lidar_transformation_samples: Lidar-space augmentation of every sample, ``None``
                for samples that were not augmented.

        Returns:
            The stacked frame metadata.

        Raises:
            ValueError: If the two sequences differ in length or are empty.
        """
        if len(frame_meta_samples) == 0:
            raise ValueError("At least one frame metadata sample is required for collating.")
        if len(frame_meta_samples) != len(lidar_transformation_samples):
            raise ValueError(
                f"Got {len(frame_meta_samples)} frame metadata samples but "
                f"{len(lidar_transformation_samples)} lidar transformation samples."
            )

        ego2globals = []
        for frame_meta, lidar_transformation in zip(
            frame_meta_samples, lidar_transformation_samples
        ):
            ego2global = frame_meta.ego2global
            if lidar_transformation is not None:
                ego2global = ego2global @ torch.linalg.inv(
                    lidar_transformation.transformation_matrix.to(ego2global)
                )
            ego2globals.append(ego2global)

        return FrameMetaBatch(
            ego2globals=torch.stack(ego2globals, dim=0),
            scene_tokens=[frame_meta.scene_token for frame_meta in frame_meta_samples],
        )

    def to_device(self, device: torch.device) -> FrameMetaBatch:
        """Move the transforms to ``device``.

        Args:
            device: Target device.

        Returns:
            The batch with its transforms on ``device``.
        """
        return FrameMetaBatch(
            ego2globals=self.ego2globals.to(device), scene_tokens=self.scene_tokens
        )
