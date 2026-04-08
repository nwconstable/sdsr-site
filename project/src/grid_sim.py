"""
grid_sim.py

Spatial grid simulator for the wetlands GNN experiment.

Classes
-------
SpatialGridSimulator : Place K drones on a grid, simulate movement, visualise.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from partition import partition_nodes, build_local_subgraph


class SpatialGridSimulator:
    """Place K drones on a square grid and simulate their movement.

    Parameters
    ----------
    grid_size  : side length of the square grid  (total nodes = grid_size**2)
    partitions : list of K global node-index arrays (output of partition_nodes)
    data       : full PyG Data object (x, edge_index, y)

    Attributes
    ----------
    drone_positions() : dict[int, int] — drone_id -> current node index
    """

    def __init__(
        self, grid_size: int, partitions: list[np.ndarray], data: Data
    ) -> None:
        self.grid_size = grid_size
        self.partitions = partitions
        self.data = data

        # Precomputed partition membership sets for O(1) lookup
        self._partition_sets: list[set[int]] = [
            set(p.tolist()) for p in partitions
        ]

        # _positions: drone_id -> current node index
        # Initial position = the actual node closest to the numeric centroid
        self._positions: dict[int, int] = {}
        for drone_id, part in enumerate(partitions):
            if len(part) > 0:
                centroid = float(np.mean(part))
                closest = int(part[np.argmin(np.abs(part.astype(float) - centroid))])
                self._positions[drone_id] = closest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def drone_positions(self) -> dict[int, int]:
        """Return a copy of the current drone_id -> node_index mapping."""
        return self._positions.copy()

    def step_drones(self) -> None:
        """Move each drone to a random adjacent node within its partition.

        Uses 4-neighbour (N/S/E/W) grid adjacency and only considers nodes
        that belong to the drone's own partition.  If no valid neighbour
        exists (drone is at a partition boundary with no in-partition
        neighbours) the drone stays in place.
        """
        for drone_id, pos in list(self._positions.items()):
            row, col = divmod(pos, self.grid_size)
            candidates: list[int] = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    neighbor = nr * self.grid_size + nc
                    if neighbor in self._partition_sets[drone_id]:
                        candidates.append(neighbor)
            if candidates:
                self._positions[drone_id] = random.choice(candidates)

    def get_local_view(self, drone_id: int, radius: int = 1) -> Data:
        """Return a PyG Data subgraph of nodes within *radius* hops of the drone.

        Expands outward via BFS over the 4-neighbour grid for exactly *radius*
        hops.  No partition constraint is applied (the drone can see beyond its
        own partition boundary).
        """
        pos = self._positions[drone_id]
        visited: set[int] = {pos}
        frontier: set[int] = {pos}

        for _ in range(radius):
            next_frontier: set[int] = set()
            for node in frontier:
                r, c = divmod(node, self.grid_size)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        neighbor = nr * self.grid_size + nc
                        if neighbor not in visited:
                            next_frontier.add(neighbor)
                            visited.add(neighbor)
            frontier = next_frontier

        node_indices = np.array(sorted(visited), dtype=np.int64)
        return build_local_subgraph(self.data, node_indices)

    def visualize(self, step: int, output_dir: str | Path) -> None:
        """Save wetland heatmap + drone markers to output_dir/sim_step_NNNN.png."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        grid_data = np.zeros((self.grid_size, self.grid_size))
        n_nodes = min(self.grid_size * self.grid_size, self.data.x.shape[0])
        for node in range(n_nodes):
            r, c = divmod(node, self.grid_size)
            grid_data[r, c] = self.data.x[node].item()

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(grid_data, cmap="Blues", origin="upper")
        plt.colorbar(im, ax=ax, label="Wetland Level")

        for drone_id, pos in self._positions.items():
            r, c = divmod(pos, self.grid_size)
            ax.scatter(c, r, c="red", s=80, edgecolors="black", zorder=5,
                       label=f"Drone {drone_id}")

        ax.set_title(f"Simulation Step {step:04d}")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / f"sim_step_{step:04d}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    GRID_SIZE = 10
    K = 4
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "sim"

    # Build a synthetic 10x10 graph — no GeoPackage needed
    N = GRID_SIZE * GRID_SIZE
    src_list, dst_list = [], []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            u = r * GRID_SIZE + c
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    src_list.append(u)
                    dst_list.append(nr * GRID_SIZE + nc)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    x = torch.rand(N, 1)
    y = torch.arange(N, dtype=torch.float).unsqueeze(1)
    data = Data(x=x, edge_index=edge_index, y=y, grid_size=GRID_SIZE)

    partitions = partition_nodes(GRID_SIZE, K)
    simulator = SpatialGridSimulator(GRID_SIZE, partitions, data)

    print(f"Initial drone positions: {simulator.drone_positions()}")

    # Run 3 steps, save images
    for step in range(3):
        simulator.visualize(step, OUTPUT_DIR)
        simulator.step_drones()
        print(f"  Step {step + 1}: {simulator.drone_positions()}")

    print(f"\n3 step images saved to {OUTPUT_DIR}")

    # Verify get_local_view returns valid PyG Data
    view = simulator.get_local_view(0, radius=2)
    assert view.x is not None, "Local view missing x"
    assert view.edge_index is not None, "Local view missing edge_index"
    print(f"Local view (drone 0, radius=2): {view.num_nodes} nodes")

    # Verify partition constraints hold across K and grid_size values
    print("\nChecking K in {2,3,4,5} and grid_size in {10,50} ...")
    for gs in (10, 50):
        n = gs * gs
        # Minimal synthetic graph (edges not needed for position checks)
        test_ei = torch.zeros(2, 0, dtype=torch.long)
        test_data = Data(
            x=torch.rand(n, 1),
            edge_index=test_ei,
            y=torch.zeros(n, 1),
            grid_size=gs,
        )
        for k in (2, 3, 4, 5):
            parts = partition_nodes(gs, k)
            sim = SpatialGridSimulator(gs, parts, test_data)
            pos = sim.drone_positions()
            assert len(pos) == k, f"Expected {k} drones, got {len(pos)}"
            for did in range(k):
                assert pos[did] in set(parts[did].tolist()), (
                    f"Initial pos {pos[did]} not in partition {did}"
                )
            # Step once and verify positions stay in partition
            sim.step_drones()
            new_pos = sim.drone_positions()
            for did in range(k):
                assert new_pos[did] in set(parts[did].tolist()), (
                    f"After step, pos {new_pos[did]} escaped partition {did}"
                )
    print("  All checks passed.")

    print("\ngrid_sim.py smoke test passed.")
    sys.exit(0)
