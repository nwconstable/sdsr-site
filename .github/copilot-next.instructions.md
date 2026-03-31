-----------------------------------
STATUS
-----------------------------------

DONE: Data pipeline (project/src/build_graph.py)
  - create_grid              : 2D grid of shapely cells over the GDF bounding box
  - assign_wetland_features  : wetland_presence node features via STRtree spatial index
  - build_edge_index         : bidirectional 4-neighbor edge_index
  - compute_dijkstra_labels  : shortest-path regression targets from random goal node
  - build_pyg_data           : assembles PyG Data(x, edge_index, y, pos)

DONE: GNN model (project/src/model.py)
  - WetlandGCN               : 2-layer GCN, hidden_channels=64, scalar node regression
                               GCNConv(1->64) -> ReLU -> GCNConv(64->1)

DONE: Centralized training (project/src/train.py)
  - train_centralized        : Adam + MSE on full graph; returns per-epoch loss list

DONE: Gossip training (project/src/train.py)
  - train_gossip             : K drone models, local training + pairwise weight
                               averaging each round; evaluates via averaged state_dict
  - num_threads (default 4)  : per-drone PyTorch thread cap (edge-device simulation)
  - time_budget_ms           : optional wall-clock ms cap on each drone's gradient loop

DONE: FedAvg training (project/src/federated_agent.py)
  - train_fedavg             : NodeAgent/CentralAgent loop; channel.sample_participants
                               dropout; evaluates global model each round; returns
                               (list[float], WetlandGCN)
  - num_threads / time_budget_ms threaded through train_local (same semantics as gossip)

DONE: Drone partitioning (project/src/partition.py)
  - partition_nodes          : column-wise strip split; returns K non-empty np.ndarray arrays
  - build_local_subgraph     : canonical public subgraph builder (re-indexes nodes 0..N-1)
  - train.py / federated_agent.py now import build_local_subgraph from here

-----------------------------------
NEXT STEP
-----------------------------------

Implement the following files in project/src/ in order:

-----------------------------------
✓ DONE  partition.py  —  column-wise strip partitioning + build_local_subgraph
-----------------------------------

-----------------------------------
1. comms.py  —  communication constraints + protocol interruption
-----------------------------------

  NOTE — existing stubs:
    - train.py and federated_agent.py both import CommunicationChannel under
      a TYPE_CHECKING guard to avoid NameError at runtime (comms.py not yet
      present). train.py __main__ uses a _MockChannel stub for gossip smoke
      tests. Once comms.py exists, these guards become live imports and the
      _MockChannel stub in train.py __main__ should be replaced with a real
      CommunicationChannel(comm_every=1, dropout_p=0.0).

  class CommunicationChannel:
    - __init__(comm_every, dropout_p, baseline_p=None, seed=None)
        comm_every  : int   - steps between comm rounds
        dropout_p   : float - per-drone drop probability each round
        baseline_p  : float - expected baseline dropout rate (defaults to dropout_p)
                              events above baseline are flagged as blackouts
    - def is_comm_round(step) -> bool
        Returns True when step % comm_every == 0
    - def sample_participants(drone_ids) -> list[int]
        Draw each drone independently; drop with probability dropout_p
    - def gossip_pairs(drone_ids) -> list[tuple[int, int]]
        Randomly pair shuffled participants (unpaired drone is skipped)

  class ProtocolInterruptionLogger:
    - __init__(baseline_p)
    - def log(round_num, drone_id, reason: Literal["dropout","blackout"])
        "blackout" = unexpected dropout beyond baseline rate
    - def record_round(round_num, participants, all_drone_ids)
        Infers dropouts, classifies each as dropout vs blackout
    - def summary() -> dict
        Returns counts: total_rounds, total_dropouts, total_blackouts, per_drone
    - def save(path)
        Saves full event log to a JSON file

  Integrate ProtocolInterruptionLogger into CommunicationChannel so that
  record_round is called automatically after each sample_participants call.

-----------------------------------
2. simulator.py  —  spatial grid simulator
-----------------------------------

  class SpatialGridSimulator:
    - __init__(grid_size, partitions, data)
        partitions : list of node-index arrays (output of partition_nodes)
        Each drone starts at the centroid node of its partition
    - def drone_positions() -> dict[int, int]
        Maps drone_id -> current node index
    - def step_drones()
        Each drone moves to a random adjacent node within its partition
    - def get_local_view(drone_id, radius=1) -> Data
        Returns subgraph of nodes within `radius` hops of drone's position
    - def visualize(step, output_dir)
        Save a matplotlib figure: wetland heatmap + drone position markers
        Filename: output_dir / f"sim_step_{step:04d}.png"

-----------------------------------
3. evaluate.py  —  evaluation & plots
-----------------------------------

  def compare_convergence(centralized_losses, fedavg_losses, gossip_losses,
                          output_dir="project/results")
    - Plot all three loss curves on one figure (log-scale y-axis optional)
    - Label axes, add legend, save to output_dir/convergence.png

  def eval_greedy_path(data, model, grid_size, start_node=None, seed=None)
    -> tuple[float, float, float]
    - Run greedy navigation: from start_node move to the neighbour with
      lowest predicted distance each step; stop when goal_node is reached
      or after grid_size*2 steps (cycle guard)
    - Optimal path length = data.y[start_node].item() (true Dijkstra label)
    - Return (greedy_path_cost, optimal_path_cost, efficiency_ratio)
      where efficiency_ratio = optimal / greedy (1.0 = perfect)

  def run_full_evaluation(data, centralized_model, fedavg_model, gossip_models,
                          centralized_losses, fedavg_losses, gossip_losses,
                          output_dir="project/results")
    - Print MSE comparison table (centralized / FedAvg / gossip)
    - Call compare_convergence
    - Call eval_greedy_path for each method and print results

-----------------------------------
4. main.py  —  entry point
-----------------------------------

Wire everything together with argparse:

  Arguments:
    --grid-size      int   default=50
    --K              int   default=4   (number of drones)
    --epochs         int   default=200
    --local-steps    int   default=5
    --comm-every     int   default=10
    --dropout-p      float default=0.1
    --baseline-p     float default=0.1
    --lr             float default=1e-3
    --num-threads    int   default=4   (per-drone CPU thread cap)
    --time-budget-ms int   default=None (no limit; pass integer ms to constrain)
    --seed           int   default=42
    --output-dir     str   default="project/results"

  Steps:
    1. Load data       : from load_data import gdf
    2. Build graph     : build_pyg_data(gdf, grid_size, seed)
    3. Partition       : partition_nodes(grid_size, K)
    4. Comms channel   : CommunicationChannel(comm_every, dropout_p, baseline_p, seed)
    5. Simulator       : SpatialGridSimulator(grid_size, partitions, data)
    6. Train all three : train_centralized, train_fedavg, train_gossip
    7. Evaluate        : run_full_evaluation(...)
    8. Save interrupt log : channel.logger.save(output_dir / "interruptions.json")

-----------------------------------
FUTURE WORK (implement after finalisation)
-----------------------------------

Synthetic graph construction (build_synthetic.py):
  - No GIS data required; useful for ablation and fast iteration
  - def build_synthetic_graph(grid_size, wetland_density, seed=None) -> Data
      - Randomly assign wetland_presence = 1 with probability wetland_density
      - Reuse build_edge_index and compute_dijkstra_labels from build_graph.py
      - Return same PyG Data format as build_pyg_data
  - Implement ONLY after real-data pipeline is validated end-to-end

-----------------------------------
CONSTRAINTS (keep simple)
-----------------------------------

- Grid <= 100x100
- K <= 5 drones
- Hidden dim = 64
- Focus on training dynamics, not model complexity
- All results saved to project/results/
