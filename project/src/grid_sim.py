import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

# Assuming Data is a dict[int, float] where key is node index (0 to grid_size**2 - 1) and value is some data (e.g., wetness level)
# If Data is a different type, adjust accordingly.

class SpatialGridSimulator:
    def __init__(self, grid_size: int, partitions: list[np.ndarray], data: dict[int, float]):
        self.grid_size = grid_size
        self.partitions = partitions
        self.data = data
        self.drone_positions = {}  # dict[int, int]: drone_id -> node index

        for drone_id, part in enumerate(partitions):
            if len(part) > 0:
                self.drone_positions[drone_id] = int(part[0])  # Start at first node in each partition

    def drone_positions(self) -> dict[int, int]:
        return self.drone_positions.copy()

    def step_drones(self):
        for drone_id, pos in self.drone_positions.items():
            part = set(self.partitions[drone_id])
            neighbors = [n for n in self._get_neighbors(pos) if n in part]
            if neighbors:
                self.drone_positions[drone_id] = np.random.choice(neighbors)

    def get_local_view(self, drone_id: int, radius: int = 1) -> dict[int, float]:
        pos = self.drone_positions[drone_id]
        visited = set()
        queue = [(pos, 0)]
        local_nodes = set()

        while queue:
            node, dist = queue.pop(0)

            if node in visited:
                continue

            visited.add(node)

            if dist <= radius:
                local_nodes.add(node)

            if dist < radius:
                for neigh in self._get_neighbors(node):
                    queue.append((neigh, dist + 1))

        return {node: self.data[node] for node in local_nodes}

    def visualize(self, step: int, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots()

        # Convert data dict to 2D array for imshow
        grid_data = np.zeros((self.grid_size, self.grid_size))

        for node, value in self.data.items():
            i, j = divmod(node, self.grid_size)
            grid_data[i, j] = value

        im = ax.imshow(grid_data, cmap='Blues', origin='upper')  # Wetland-like heatmap

        # Add drone markers
        for drone_id, pos in self.drone_positions.items():
            i, j = divmod(pos, self.grid_size)
            ax.scatter(j, i, c='red', s=50, edgecolors='black')

        plt.colorbar(im, label='Wetland Level')
        plt.title(f'Simulation Step {step}')
        plt.savefig(output_dir / f'sim_step_{step:04d}.png')
        plt.close()

    def _get_neighbors(self, node: int) -> list[int]:
        i, j = divmod(node, self.grid_size)
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                neighbors.append(ni * self.grid_size + nj)
        return neighbors
    
# ---------------------------------------------------------------------------
# Grid Simulator Debugging
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from partition import partition_nodes
    from build_graph import build_pyg_data
    from load_data import load_gdf

    GRID_SIZE = 10
    K = 6

    # Create PyG data for a 10x10 grid
    gdf = load_gdf()
    data, goal = build_pyg_data(gdf, grid_size=GRID_SIZE, seed=42)

    # Create partitions (e.g., vertical slices)
    partitions = partition_nodes(grid_size=GRID_SIZE, K=K, seed=42)

    # Create the simulator
    simulator = SpatialGridSimulator(GRID_SIZE, partitions, data)

    # Print the images to a debug directory
    debug_dir = Path("__file__").resolve().parent.parent / "data/DebugResults"

    if not debug_dir.exists():
        print("Creating debug directory for grid simulator results...")
        debug_dir.mkdir(parents=True, exist_ok=True)

    for step in range(K):
        simulator.visualize(step, output_dir=debug_dir)
        simulator.step_drones()