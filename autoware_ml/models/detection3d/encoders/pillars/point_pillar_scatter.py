from typing import Sequence

from jaxtyping import Float32
from torch import nn
import torch


class PointPillarsScatter(nn.Module):
    """Scatter sparse pillar features to a dense BEV feature map.

    The scatter step converts sparse pillar descriptors into a dense 2D BEV map
    consumed by downstream convolutional backbones.
    """

    def __init__(self, in_channels: int, output_shape: Sequence[int]) -> None:
        """Initialize the dense BEV scatter module.

        Args:
            in_channels: Input feature dimension.
            output_shape: Output BEV shape as ``(height, width)``.
        """
        super().__init__()
        self.in_channels = in_channels
        self.output_shape = tuple(output_shape)

    def forward(
        self,
        pillar_features: Float32[torch.Tensor, "num_pillars num_channels"],
        coords: Float32[torch.Tensor, "num_pillars 3"],
        batch_indices: Float32[torch.Tensor, " num_pillars"],
        batch_size: int,
    ) -> Float32[torch.Tensor, "batch_size num_output_channels height width"]:
        """Scatter pillar features into a dense BEV canvas.

        Args:
            pillar_features: Encoded pillar features.
            coords: Voxel coordinates in (x, y, z).
            batch_indices: Batch indices for each pillar.
            batch_size: Batch size of the current sample set.

        Returns:
            Dense BEV feature map after scattering pillar features into their respective locations
            (height, width).
        """
        batch_indices = batch_indices.long()
        y_indices = coords[:, 1].long()
        x_indices = coords[:, 0].long()
        height, width = self.output_shape
        flat_indices = batch_indices * (height * width) + y_indices * width + x_indices

        canvas = pillar_features.new_zeros((batch_size * height * width, self.in_channels))
        scatter_indices = flat_indices.unsqueeze(1).expand(-1, self.in_channels)
        canvas = canvas.scatter(0, scatter_indices, pillar_features)
        return (
            canvas.view(batch_size, height, width, self.in_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
