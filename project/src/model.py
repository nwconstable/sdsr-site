"""
model.py

2-layer Graph Convolutional Network for node-level distance regression.

Architecture
------------
  GCNConv(F  -> 64) -> ReLU
  GCNConv(64 ->  1)

Input  : x (N, F)  goal-conditioned node features
Output : (N, 1)    predicted shortest-path distance to goal node
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class WetlandGCN(torch.nn.Module):
    """2-layer GCN for node-level regression on the wetlands grid graph."""

    def __init__(self, in_channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        self.conv2 = GCNConv(in_channels=hidden_channels, out_channels=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, F) node feature tensor
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
    x = torch.zeros(N, 7)
    x[:, 1] = torch.linspace(0.05, 0.95, N)
    x[:, 2] = torch.linspace(0.95, 0.05, N)
    x[:, 3] = 0.5
    x[:, 4] = 0.5
    x[:, 5] = x[:, 3] - x[:, 1]
    x[:, 6] = x[:, 4] - x[:, 2]
    x[0] = 1.0  # one wetland node
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )

    model = WetlandGCN(in_channels=x.shape[1], hidden_channels=64)
    out = model(x, edge_index)
    print(f"output shape : {out.shape}")   # expect (10, 1)
    print(f"sample preds : {out[:4].squeeze().tolist()}")
    print(f"param count  : {sum(p.numel() for p in model.parameters())}")
    sys.exit(0)
