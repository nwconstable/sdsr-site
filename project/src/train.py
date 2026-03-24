"""
train.py

Training routines for the wetlands GNN experiment.

Functions
---------
train_centralized  : train WetlandGCN on the full graph (centralized baseline)
                     — FedAvg and gossip trainers will be added in Issues #5 / #6
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from model import WetlandGCN


def train_centralized(
    data: Data,
    model: WetlandGCN,
    epochs: int,
    lr: float = 1e-3,
) -> list[float]:
    """Train *model* on the full graph for *epochs* steps.

    Parameters
    ----------
    data   : PyG Data with fields x (N,1) and y (N,1)
    model  : WetlandGCN instance; updated in-place
    epochs : number of gradient steps
    lr     : Adam learning rate

    Returns
    -------
    losses : list of per-epoch MSE loss values (length == epochs)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a tiny 2x5 grid (10 nodes) with synthetic wetland features and
    # hand-crafted distance targets so no GeoPackage file is needed.

    ROWS, COLS = 2, 5
    N = ROWS * COLS

    # Node features: alternating wetland presence
    wetland = torch.tensor(
        [[float(i % 2)] for i in range(N)], dtype=torch.float
    )

    # 4-neighbor edges for a 2x5 grid
    src, dst = [], []
    for r in range(ROWS):
        for c in range(COLS):
            u = r * COLS + c
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    src.append(u)
                    dst.append(nr * COLS + nc)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Synthetic targets: distance = node index (increases away from node 0)
    targets = torch.tensor([[float(i)] for i in range(N)], dtype=torch.float)

    data = Data(x=wetland, edge_index=edge_index, y=targets)

    model = WetlandGCN(hidden_channels=64)
    initial_loss = F.mse_loss(model(data.x, data.edge_index), data.y).item()

    EPOCHS = 10
    losses = train_centralized(data, model, epochs=EPOCHS, lr=1e-2)

    print(f"Initial loss : {initial_loss:.4f}")
    for i, l in enumerate(losses, 1):
        print(f"  Epoch {i:>2} : {l:.4f}")
    print(f"Final loss   : {losses[-1]:.4f}")
    assert losses[-1] < initial_loss, "Loss did not decrease"
    print("Smoke test passed.")
    sys.exit(0)
