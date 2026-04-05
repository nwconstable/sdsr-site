import sys

import numpy as np
from pathlib import Path

import torch
from partition import partition_nodes
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Assuming Data is a torch_geometric.data.Data instance with node features in data.x

class SpatialGridSimulator:
    def __init__(self, grid_size: int, partitions: list[np.ndarray], data: Data):
        self.grid_size = grid_size
        self.partitions = partitions
        self.data = data
        self.drone_positions = {}  # dict[int, int]: drone_id -> node index

        for drone_id, part in enumerate(partitions):
            if len(part) > 0:
                self.drone_positions[drone_id] = int(np.mean(part))

    def drone_positions(self) -> dict[int, int]:
        return self.drone_positions.copy()

    def step_drones(self):
        for drone_id in self.drone_positions.keys():
            # Use NeighborLoader to sample neighbors from the drone's partition
            neighbor_loader = NeighborLoader(
                data=self.data, # Sampling from each drone's partition only
                num_neighbors=4, # Sample up to 4 neighbors for each drone
                batch_size=20, # Needs adjusting?
                input_nodes=torch.tensor([self.drone_positions[drone_id]]).to_sparse() # Starting position of the drone
            )

            neighbor_data = next(iter(neighbor_loader))
            visitable_nodes = neighbor_data.n_id.numpy()

            # Get the first sampled node that is in the drone's partition
            for node in visitable_nodes:
                if node in self.partitions[drone_id]:
                    self.drone_positions[drone_id] = node
                    break

    def get_local_view(self, drone_id: int, radius: int = 1) -> Data:
        # Get all nodes within 'radius' hops of the drone's current position
        pos = self.drone_positions[drone_id]

        # Use NeighborLoader to sample neighbors from the drone's current position
        neighbor_loader = NeighborLoader(
            data=self.data,
            num_neighbors=4,
            batch_size=20, # Needs adjusting?
            input_nodes=pos # Starting from the current node
        )

        # Initialize all immediate neighbors
        neighbor_data = next(iter(neighbor_loader))

        # Iterate until we reach the desired radius
        for _ in range(radius - 1):
            neighbor_data = next(iter(neighbor_loader))

        # Return the neighbor data as the local view for the drone
        return neighbor_data

    def visualize(self, step: int, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()

        grid_data = np.zeros((self.grid_size, self.grid_size))

        # Not sure if this is citing the grid data corectly
        for node in range(self.grid_size * self.grid_size):
            i, j = divmod(node, self.grid_size)
            grid_data[i, j] = self.data.x[node].item() if (node < self.data.x.shape[0]) else 0.0

        im = ax.imshow(grid_data, cmap='Blues', origin='upper')

        for drone_id, pos in self.drone_positions.items():
            i, j = divmod(pos, self.grid_size)
            ax.scatter(j, i, c='red', s=50, edgecolors='black')

        plt.colorbar(im, label='Wetland Level')
        plt.title(f'Simulation Step {step}')
        plt.savefig(output_dir / f'sim_step_{step:04d}.png')
        plt.close()

# ---------------------------------------------------------------------------
# Grid Simulator Debugging
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    GRID_SIZE = 50
    K = 5

    debug_dir = Path("__file__").resolve().parent.parent / "data/DebugResults"
    sample_mfile = debug_dir / "sample_data.pth"

    print(f"Checking for debug directory and sample data file in \"{debug_dir}\"...")

    if not debug_dir.exists():
        print("Creating debug directory for grid simulator results...")
        debug_dir.mkdir(parents=True, exist_ok=True)

    if sample_mfile.exists():
        print(f"Loading sample data from \"{sample_mfile}\"...")
        data = torch.load(sample_mfile, weights_only=False)
    else:
        from partition import partition_nodes
        from build_graph import build_pyg_data
        from load_data import load_gdf

        gdf = load_gdf()
        data, goal = build_pyg_data(gdf, grid_size=GRID_SIZE, seed=42)

        torch.save(data, sample_mfile)
    
    for grid_size in range(10, GRID_SIZE + 1, 10):
        for k in range(2, K + 1):
            partitions = partition_nodes(grid_size=grid_size, K=k)

            # Initialize the simulator
            simulator = SpatialGridSimulator(GRID_SIZE, partitions, data)

            for step in range(5):
                simulator.visualize(step, output_dir=debug_dir)
                simulator.step_drones()