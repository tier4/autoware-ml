import torch.nn as nn

from spconv.pytorch import SubMConv3d
from spconv.pytorch import SparseConvTensor
from spconv.pytorch.modules import SparseModule


class SparseBasicBlock(SparseModule):
    """Residual submanifold-convolution block (ResNet basic block, sparse).

    Inherits ``spconv``'s ``SparseModule`` so ``SparseSequential`` passes the
    full ``SparseConvTensor`` (not just its ``.features``).
    """

    def __init__(self, channels: int, indice_key: str, eps: float, momentum: float) -> None:
        super().__init__()
        self.conv1 = SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = nn.BatchNorm1d(channels, eps=eps, momentum=momentum)
        self.conv2 = SubMConv3d(
            channels, channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = nn.BatchNorm1d(channels, eps=eps, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        """Apply the residual sparse convolution block.

        Args:
            x: Input sparse convolution tensor.

        Returns:
            Output sparse tensor with the residual update applied.
        """
        identity = x.features
        out = self.conv1(x)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.relu(out.features + identity))
        return out
