"""
model.py

2-layer Graph Convolutional Network for node-level distance regression.

Architecture
------------
  GCNConv(1  -> 64) -> ReLU
  GCNConv(64 ->  1)

Input  : x (N, 1)  wetland_presence features
Output : (N, 1)    predicted shortest-path distance to goal node
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class WetlandGCN(torch.nn.Module):
    """2-layer GCN for node-level regression on the wetlands grid graph."""

    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels=1, out_channels=hidden_channels)
        self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, 1) node feature tensor
        edge_index : (2, E) adjacency in COO format

        Returns
        -------
        out : (N, 1) predicted distance-to-goal for each node
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


if __name__ == "__main__":
    import sys

    # Smoke-test with a tiny synthetic graph
    N = 10
    x = torch.zeros(N, 1)
    x[0] = 1.0  # one wetland node
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )

    model = WetlandGCN(hidden_channels=64)
    out = model(x, edge_index)
    print(f"output shape : {out.shape}")   # expect (10, 1)
    print(f"sample preds : {out[:4].squeeze().tolist()}")
    print(f"param count  : {sum(p.numel() for p in model.parameters())}")
    sys.exit(0)
