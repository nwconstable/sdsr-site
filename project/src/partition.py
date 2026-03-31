"""
partition.py

Drone node partitioning utilities for the wetlands GNN experiment.

Functions
---------
partition_nodes      : Split a square grid graph into K spatial column strips.
build_local_subgraph : Build a re-indexed PyG subgraph for a given node set.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph


def partition_nodes(grid_size: int, K: int, method: str = "grid") -> list[np.ndarray]:
    """Return K arrays of global node indices, one per drone.

    Parameters
    ----------
    grid_size : side length of the square grid  (total nodes = grid_size**2)
    K         : number of partitions (drones)
    method    : partitioning strategy.  Only ``"grid"`` (column-wise strips)
                is currently supported.  Pass ``method="grid"`` explicitly or
                rely on the default.

    Returns
    -------
    partitions : list of K 1-D int64 numpy arrays.  Each array contains the
                 global node indices assigned to one drone.  Strips are
                 approximately equal width; the last strip absorbs any
                 remainder so that the union equals ``range(grid_size**2)``
                 exactly.

    Notes
    -----
    Node index layout (row-major):
        node_id = row * grid_size + col
    Column-wise strip assignment:
        strip_id = min(col * K // grid_size, K - 1)
    """
    if method != "grid":
        raise ValueError(f"Unknown partition method '{method}'. Only 'grid' is supported.")
    if K < 1:
        raise ValueError(f"K must be at least 1, got {K}")
    if K > grid_size:
        raise ValueError(f"K ({K}) cannot exceed grid_size ({grid_size})")

    N = grid_size * grid_size
    all_nodes = np.arange(N, dtype=np.int64)

    # Column index for each node
    cols = all_nodes % grid_size

    # Map each column to one of K strips (integer division, clamped to K-1)
    strip_ids = np.minimum(cols * K // grid_size, K - 1)

    return [all_nodes[strip_ids == k] for k in range(K)]


def build_local_subgraph(data: Data, node_indices: np.ndarray) -> Data:
    """Return a new PyG Data restricted to *node_indices* (global).

    Edges where either endpoint is outside the given node set are dropped.
    Surviving nodes are re-indexed to ``0 .. len(node_indices)-1``.

    Parameters
    ----------
    data         : full PyG Data object with fields x, edge_index, y
    node_indices : 1-D array or tensor of global node indices for this
                   partition.  Both ``np.ndarray`` and ``torch.Tensor``
                   are accepted.

    Returns
    -------
    sub_data : Data(x, edge_index, y) with local node indexing.
    """
    global_idx = torch.as_tensor(node_indices, dtype=torch.long)
    sub_ei, _ = pyg_subgraph(
        global_idx, data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )
    return Data(x=data.x[global_idx], edge_index=sub_ei, y=data.y[global_idx])


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GRID_SIZE = 10
    K = 4

    print(f"Grid size : {GRID_SIZE}x{GRID_SIZE}  ({GRID_SIZE**2} nodes)")
    print(f"K         : {K}")
    print()

    partitions = partition_nodes(GRID_SIZE, K)

    # Verify completeness and uniqueness
    all_indices = np.concatenate(partitions)
    assert set(all_indices.tolist()) == set(range(GRID_SIZE**2)), \
        "Partition union does not cover all nodes!"
    assert len(all_indices) == GRID_SIZE**2, \
        "Duplicate nodes detected across partitions!"

    for i, part in enumerate(partitions):
        col_min = int(part.min() % GRID_SIZE)
        col_max = int(part.max() % GRID_SIZE)
        print(f"  Partition {i}: {len(part):>4} nodes  cols {col_min}..{col_max}")

    # Build a minimal synthetic 10x10 graph for the subgraph demo
    N = GRID_SIZE**2
    src, dst = [], []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            u = r * GRID_SIZE + c
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    src.append(u)
                    dst.append(nr * GRID_SIZE + nc)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.zeros(N, 1)
    y = torch.arange(N, dtype=torch.float).unsqueeze(1)
    data = Data(x=x, edge_index=edge_index, y=y)

    print()
    print("Subgraph summary (partition 0):")
    sub = build_local_subgraph(data, partitions[0])
    print(f"  Nodes        : {sub.num_nodes}")
    print(f"  Edges        : {sub.edge_index.shape[1]}")
    print(f"  Node id range: 0 .. {sub.num_nodes - 1}")
    print(f"  Max edge idx : {sub.edge_index.max().item()}")

    assert sub.num_nodes == len(partitions[0]), "Wrong node count in subgraph!"
    assert sub.edge_index.max().item() < sub.num_nodes, "Edge index out of local bounds!"
    assert sub.x.shape == (sub.num_nodes, 1), "Feature shape mismatch!"
    assert sub.y.shape == (sub.num_nodes, 1), "Target shape mismatch!"

    # Verify for multiple K values
    print()
    print("Checking K in {2, 3, 4, 5} and grid_size in {10, 50, 100} ...")
    for gs in (10, 50, 100):
        for k in (2, 3, 4, 5):
            parts = partition_nodes(gs, k)
            assert len(parts) == k, f"Expected {k} partitions, got {len(parts)}"
            merged = np.concatenate(parts)
            assert len(merged) == gs**2, f"Node count mismatch gs={gs} K={k}"
            assert set(merged.tolist()) == set(range(gs**2)), \
                f"Coverage gap gs={gs} K={k}"
            assert all(len(p) > 0 for p in parts), f"Empty partition gs={gs} K={k}"
    print("  All checks passed.")

    print()
    print("partition.py smoke test passed.")
    sys.exit(0)
