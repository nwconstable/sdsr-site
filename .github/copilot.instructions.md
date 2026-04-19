We are implementing a simulation for decentralized training of a Graph Neural Network (GNN) for a spatial task.

High-level goal:
Compare three training strategies:
1. Centralized training
2. Federated averaging (FedAvg)
3. Fully decentralized gossip-based training

We are using PyTorch Geometric.

-----------------------------------
DATA PIPELINE
-----------------------------------

We have Minnesota wetlands GIS data loaded via GeoPandas (GeoDataFrame with polygon geometries).

We want to:

1. Convert GIS polygons into a grid-based graph:
   - Create a 2D grid over the bounding box (e.g., 50x50 or 100x100)
   - Each grid cell is a node
   - Node feature:
       wetland_presence = 1 if cell intersects any wetland polygon, else 0

2. Build edges:
   - 4-neighbor grid connectivity (up/down/left/right)

3. Create labels:
   - Select a random goal node
   - Assign traversal cost:
       wetland = high cost (e.g., 5)
       land = low cost (e.g., 1)
   - Compute shortest path distance from every node to goal (Dijkstra)
   - Use this distance as regression target

4. Convert to PyTorch Geometric Data object:
   - x: node features
   - edge_index: adjacency
   - y: distance-to-goal

-----------------------------------
MODEL
-----------------------------------

Define a simple GNN:

- 2-layer GCN
- Hidden dim ~32 or 64
- Output: scalar per node (regression)

-----------------------------------
PARTITIONING (SIMULATED DRONES)
-----------------------------------

Split nodes into K regions (e.g., 3–5 partitions):
- Each partition = one "drone"
- Each drone only trains on its local subgraph

-----------------------------------
TRAINING MODES
-----------------------------------

1. Centralized:
   - Train on full graph

2. Federated (FedAvg):
   Loop:
     - Each drone trains locally for E steps
     - Collect weights
     - Average weights globally
     - Broadcast back

3. Gossip:
   Loop:
     - Each drone trains locally
     - Randomly select pairs of drones
     - When connected:
         average their model weights
     - No central aggregation

-----------------------------------
CONSTRAINT SIMULATION
-----------------------------------

Simulate communication limits between drones in a dedicated comms.py module:

- Communication every K steps (comm_every)
- Random dropout per drone (probability p)
- Protocol Interruption Logger:
    - Track every dropout/blackout event by round and drone ID
    - A "blackout" is a dropout that exceeds the configured baseline rate
    - Log reason ("dropout" vs "blackout"), round number, and drone ID
    - Expose summary statistics and save to file

Simulate compute limits per drone (already implemented in train_gossip and
NodeAgent.train_local):

- num_threads    : PyTorch CPU thread count per drone during local training.
                   Default 4 (ARM Cortex-A72 / Jetson Nano class device).
                   Override at call: num_threads=1 (very constrained) to
                   num_threads=8 (high-end).
- time_budget_ms : Wall-clock ms cap on each drone's gradient loop.
                   Loop exits early if budget is consumed before local_steps
                   are reached.  None = no constraint (unconstrained baseline).

-----------------------------------
SPATIAL GRID SIMULATOR
-----------------------------------

A grid_sim.py module places K drones on the grid and tracks their state:

- Drone positions: dict mapping drone_id -> node_index
- Drones observe their local neighbourhood (subgraph around position)
- Optional: drones can step to adjacent nodes each round
- Visualisation: save a matplotlib snapshot per training round showing
  drone positions overlaid on the wetland grid (heatmap of wetland_presence)

-----------------------------------
FUTURE WORK (post-finalisation)
-----------------------------------

Synthetic graph construction:
- After the real-data pipeline is finalised and validated, add support for
  generating synthetic grid graphs (no GIS data required) using only a
  grid_size and a wetland_density parameter.
- Useful for ablation studies and fast iteration without the GeoPackage.
- Implement as build_synthetic_graph(grid_size, wetland_density, seed) in
  a separate build_synthetic.py file.

-----------------------------------
EVALUATION
-----------------------------------

- Compute MSE vs the ground-truth Dijkstra distance labels for each method
- Treat centralized training as a comparison reference in reports, not as the regression target
- Measure convergence over time
- Optional:
    simulate greedy path:
    from multiple random starts, move to neighbor with lowest predicted distance
  evaluate path efficiency and goal-reaching success rate vs optimal

-----------------------------------
IMPORTANT
-----------------------------------

Keep everything small and simple:
- Grid ≤ 100x100
- ≤ 5 drones
- Simple GCN only

Focus on training dynamics, not model complexity.

-----------------------------------
MAINTENANCE RULES
-----------------------------------

1. **Always update `project/README.md`** after any change to source files,
   module status, CLI arguments, or dependencies.  The README is the
   human-facing reference for running and understanding the project.

2. **`issues.md` is the canonical task tracker.**  Every new feature,
   bug, or TODO must be filed there as a numbered issue with full context,
   function signatures, and a verifiable acceptance checklist.
   `copilot-next.instructions.md` is deprecated — do not update it;
   `issues.md` supersedes it entirely.

-----------------------------------
NEXT STEP
-----------------------------------

See `.github/issues.md` for the current implementation backlog.
Issues are ordered by dependency.  The next unfinished issue is the
correct starting point.  Update the issue's acceptance checklist and
add ✓ DONE to the heading when complete.