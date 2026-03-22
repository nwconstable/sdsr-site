"""
build_graph.py

Converts Minnesota wetlands GIS data (GeoDataFrame) into a PyTorch Geometric
Data object suitable for GNN training.

Pipeline
--------
1. Create a 2D grid over the GeoDataFrame bounding box
2. Assign wetland_presence node features (1 if cell intersects a wetland polygon)
3. Build 4-neighbor edge_index (up / down / left / right)
4. Compute Dijkstra shortest-path distance from every node to a random goal
5. Return a PyG Data object  (x, edge_index, y, pos)
"""

from __future__ import annotations

import heapq
import random
import sys
from typing import Tuple

import geopandas as gpd
import numpy as np
import torch
from shapely.geometry import box
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def create_grid(
    gdf: gpd.GeoDataFrame,
    grid_size: int = 50,
) -> Tuple[np.ndarray, float, float, float, float, float, float]:
    """
    Create a 2D grid of shapely Polygons over the bounding box of *gdf*.

    Node ordering is row-major with row 0 at the bottom (south).
    Node index = row * grid_size + col

    Returns
    -------
    cell_polys : ndarray of shapely Polygon, shape (N,)
    minx, miny, maxx, maxy : bounding box coordinates
    cell_w, cell_h : cell width and height in CRS units
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    cell_w = (maxx - minx) / grid_size
    cell_h = (maxy - miny) / grid_size

    cell_polys = np.empty(grid_size * grid_size, dtype=object)

    for row in range(grid_size):
        for col in range(grid_size):
            x0 = minx + col * cell_w
            y0 = miny + row * cell_h
            cell_polys[row * grid_size + col] = box(x0, y0, x0 + cell_w, y0 + cell_h)

    return cell_polys, minx, miny, maxx, maxy, cell_w, cell_h

def assign_wetland_features(
    gdf: gpd.GeoDataFrame,
    cell_polys: np.ndarray,
) -> np.ndarray:
    """
    Return a float32 array of shape (N,) with wetland_presence in {0.0, 1.0}.

    Uses a spatial index (STRtree) for efficient intersection queries.
    """
    sindex = gdf.sindex
    wetland_presence = np.zeros(len(cell_polys), dtype=np.float32)

    for i, cell in enumerate(cell_polys):
        candidates = list(sindex.intersection(cell.bounds))

        if candidates and gdf.iloc[candidates].intersects(cell).any():
            wetland_presence[i] = 1.0

    return wetland_presence

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_edge_index(grid_size: int) -> torch.Tensor:
    """
    Build bidirectional 4-neighbor edge_index for a *grid_size* x *grid_size* grid.

    Returns
    -------
    edge_index : LongTensor of shape (2, E)
    """
    src, dst = [], []

    def _idx(r: int, c: int) -> int:
        return r * grid_size + c

    for r in range(grid_size):
        for c in range(grid_size):
            # Right neighbor: generate once and add both directions
            if c + 1 < grid_size:
                u = _idx(r, c)
                v = _idx(r, c + 1)
                src += [u, v]
                dst += [v, u]

            # Upper neighbor (next row): generate once and add both directions
            if r + 1 < grid_size:
                u = _idx(r, c)
                v = _idx(r + 1, c)
                src += [u, v]
                dst += [v, u]
    return torch.tensor([src, dst], dtype=torch.long)

# ---------------------------------------------------------------------------
# Dijkstra labels
# ---------------------------------------------------------------------------
def compute_dijkstra_labels(
    wetland_presence: np.ndarray,
    grid_size: int,
    goal_node: int | None = None,
    wetland_cost: float = 5.0,
    land_cost: float = 1.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Shortest-path distance from every node to *goal_node*.

    Edge cost equals the traversal cost of the *destination* node:
      - wetland -> wetland_cost (default 5)
      - land    -> land_cost    (default 1)

    Because cost is symmetric, running Dijkstra *from* the goal gives the
    same distances as running it *to* the goal.

    Returns
    -------
    distances : float32 array of shape (N,)
    goal_node : int
    """
    N = grid_size * grid_size

    if goal_node is None:
        rng = random.Random(seed)
        goal_node = rng.randint(0, N - 1)

    node_cost = np.where(wetland_presence > 0, wetland_cost, land_cost).astype(np.float64)

    # Adjacency list: adj[u] = [(v, cost_of_v), ...]
    adj: list[list[tuple[int, float]]] = [[] for _ in range(N)]

    for r in range(grid_size):
        for c in range(grid_size):
            u = r * grid_size + c

            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc

                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    v = nr * grid_size + nc
                    adj[u].append((v, node_cost[v]))

    dist = np.full(N, np.inf, dtype=np.float64)
    dist[goal_node] = 0.0
    pq: list[tuple[float, int]] = [(0.0, goal_node)]

    while pq:
        d, u = heapq.heappop(pq)

        if d > dist[u]:
            continue

        for v, w in adj[u]:
            nd = d + w

            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist.astype(np.float32), goal_node

# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_pyg_data(
    gdf: gpd.GeoDataFrame,
    grid_size: int = 50,
    goal_node: int | None = None,
    seed: int | None = 42,
) -> Tuple[Data, int]:
    """
    Build a PyTorch Geometric Data object from the wetlands GeoDataFrame.

    Parameters
    ----------
    gdf       : GeoDataFrame with wetland polygon geometries
    grid_size : number of cells per side (≤ 100 recommended)
    goal_node : fixed goal index; if None a random one is chosen
    seed      : RNG seed used when goal_node is None

    Returns
    -------
    data      : PyG Data with fields: x, edge_index, y, pos
    goal_node : the goal node index that was used
    """
    print(f"Building {grid_size}x{grid_size} grid graph ({grid_size**2} nodes)...")

    # 1. Grid
    cell_polys, minx, miny, maxx, maxy, cell_w, cell_h = create_grid(gdf, grid_size)

    # 2. Wetland features
    print("  Assigning wetland_presence features (may take a moment)...")
    wetland_presence = assign_wetland_features(gdf, cell_polys)
    n_wet = int(wetland_presence.sum())

    print(f"  Wetland cells : {n_wet} / {len(cell_polys)} "
          f"({100 * n_wet / len(cell_polys):.1f}%)")

    # 3. Edge index
    edge_index = build_edge_index(grid_size)

    # 4. Dijkstra labels
    labels, goal_node = compute_dijkstra_labels(
        wetland_presence, grid_size, goal_node=goal_node, seed=seed
    )
    print(f"  Goal node     : {goal_node}  "
          f"(row={goal_node // grid_size}, col={goal_node % grid_size})")

    # 5. Normalised (x, y) cell-centre positions in [0, 1]
    rows = np.arange(len(cell_polys)) // grid_size
    cols = np.arange(len(cell_polys)) % grid_size
    pos = torch.tensor(
        np.stack([(cols + 0.5) / grid_size, (rows + 0.5) / grid_size], axis=1),
        dtype=torch.float,
    )

    data = Data(
        x=torch.tensor(wetland_presence, dtype=torch.float).unsqueeze(1),  # (N, 1)
        edge_index=edge_index,                                             # (2, E)
        y=torch.tensor(labels, dtype=torch.float).unsqueeze(1),            # (N, 1)
        pos=pos,                                                           # (N, 2)
        grid_size=grid_size,
        goal_node=goal_node,
    )

    return data, goal_node

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from load_data import load_gdf  # noqa: E402

    gdf = load_gdf()
    data, goal = build_pyg_data(gdf, grid_size=50, seed=42)
    print(data)
    print(f"x          : {data.x.shape}")
    print(f"edge_index : {data.edge_index.shape}")
    print(f"y          : {data.y.shape}")
    print(f"Goal node  : {goal}")
    sys.exit(0)
