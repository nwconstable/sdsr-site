# Project: Networked Edge Training for Spatial Graph Learning
## Group 2 — Noah Constable, Joe Conroy

### Purpose
Train a network of simulated edge devices (drones) using three strategies — centralized, FedAvg, and gossip — then empirically compare them on resource dimensions such as network connectivity and compute.

The graph-learning target is goal-conditioned shortest-path regression: every node predicts its Dijkstra distance to a selected goal node using fractional wetland coverage, node position, and goal-relative features.

---

### Resources

One of the datasets used is the [Minnesota National Wetlands Inventory Update](https://www.dnr.state.mn.us/wetlands/nwi_proj.html) GeoPackage, which is wrangled into a 2D grid graph via GeoPandas.

A CUDA-enabled GPU is recommended for performance but **not required**. All scripts run on a CPU-only PyTorch installation. For GPU acceleration follow the [CUDA installation guide](https://developer.nvidia.com/cuda-toolkit) and then install a CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

### Quick Start

```bash
# 1. (Optional) activate a virtual environment
# 2. Install dependencies  (see Libraries section)
# 3. From project/src/:

python main.py \
    --grid-size 50 \
    --K 4 \
    --epochs 200 \
    --local-steps 5 \
    --comm-every 10 \
    --dropout-p 0.1 \
    --baseline-p 0.1 \
    --lr 1e-3 \
    --seed 42 \
    --num-threads 4 \
    --use-simulator-integration \
    --simulator-view-radius 2 \
    --simulator-comm-radius 3 \
    --simulator-step-drones \
    --output-dir ../results
```

**Outputs written to `--output-dir` (`project/results/` by default):**

| File | Description |
|---|---|
| `convergence.png` | Loss curves for all three training methods |
| `mse_evaluation.png` | Bar charts of final and mean MSE per method |
| `greedy_path_evaluation.png` | Mean greedy-path efficiency and goal-reaching success rate across seeded starts |
| `interruptions.json` | Separate FedAvg and gossip dropout/blackout logs keyed by training method |

---

### CLI Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--grid-size` | int | 50 | Side length of the square grid (total nodes = grid_size²) |
| `--K` | int | 4 | Number of drone partitions |
| `--epochs` | int | 200 | Training rounds |
| `--local-steps` | int | 5 | Gradient steps per drone per round |
| `--comm-every` | int | 10 | Epoch interval between communication events |
| `--dropout-p` | float | 0.1 | Per-drone dropout probability each round |
| `--baseline-p` | float | 0.1 | Dropout rate above which a round is flagged as blackout |
| `--lr` | float | 1e-3 | Adam learning rate |
| `--seed` | int | 42 | Global RNG seed |
| `--output-dir` | str | project/results | Output directory for plots and logs |
| `--num-threads` | int | 4 | CPU thread cap per drone during local training |
| `--time-budget` | int | None | Wall-clock ms budget per drone per round (None = unconstrained) |
| `--use-simulator-integration` | flag | off | Refresh decentralized local training data from simulator positions and gate decentralized communication by proximity |
| `--simulator-view-radius` | int | 2 | Hop radius for simulator-derived local subgraphs during FedAvg and gossip |
| `--simulator-comm-radius` | int | 2 | Manhattan-radius communication limit for simulator-driven FedAvg uplinks and gossip exchanges |
| `--simulator-step-drones` | flag | off | Move each drone by one in-partition grid step before each decentralized round |

`main.py` validates CLI argument ranges before starting and emits six stage
status updates so long experiment runs are easier to monitor. Re-running with
the same `--seed` on the same environment is intended to reproduce the same
graph construction, communication events, and reported metrics.

---

### Module Inventory

| File | Status | Description |
|---|---|---|
| `load_data.py` | Done | Validates and loads the wetlands GeoPackage via GeoPandas |
| `build_graph.py` | Done | Builds PyG `Data` with goal-conditioned node features, Dijkstra labels, and goal metadata |
| `model.py` | Done | `WetlandGCN`: 2-layer GCN for node-level distance regression from goal-conditioned features |
| `partition.py` | Done | `partition_nodes` (column-strip) and `build_local_subgraph` |
| `comms.py` | Done | `CommunicationChannel` (schedule, dropout, gossip pairing) + `ProtocolInterruptionLogger` |
| `train.py` | Done | `train_centralized`, `train_gossip` (with compute constraints and optional simulator-driven local views / proximity gossip) |
| `federated_agent.py` | Done | `NodeAgent` / `CentralAgent` + `train_fedavg`; compute constraints and optional simulator-driven local views / base-station reachability |
| `grid_sim.py` | Done | `SpatialGridSimulator` — drone positions, movement, local-view extraction, proximity queries, and per-step visualisation |
| `evaluations.py` | Done | `compare_convergence`, multi-start greedy-path evaluation, and result plots |
| `main.py` | Done | Full experiment entry point with argparse CLI |
| `build_synthetic.py` | Future | Synthetic grid graph without GeoPackage (Issue #9, post-validation) |

---

### Individual Module CLIs

Each module has a `__main__` smoke test that can be run independently for development and debugging. All commands run from `project/src/`:

```bash
python load_data.py          # validate + load GeoPackage
python build_graph.py        # build PyG Data from GeoPackage
python partition.py          # partition_nodes + build_local_subgraph smoke test
python comms.py              # 50-round comm simulation with 4 drones
python train.py              # centralized + gossip smoke tests (synthetic data)
python federated_agent.py    # FedAvg training on real GeoPackage data
python grid_sim.py           # SpatialGridSimulator: 3 steps + PNG output
python evaluations.py        # full evaluation (loads/generates training data)
```

---

### Per-Drone Compute Simulation

`train_gossip` (`train.py`) and `NodeAgent.train_local` (`federated_agent.py`) simulate edge-device resource limits:

| Parameter | Default | Description |
|---|---|---|
| `num_threads` | `4` | Caps PyTorch CPU thread pool via `torch.set_num_threads()`. 4 reflects an ARM Cortex-A72 (Raspberry Pi 4) / Jetson Nano. Use `1–2` for heavily constrained devices, `8` for high-end boards. |
| `time_budget_ms` | `None` | Wall-clock deadline in milliseconds. The gradient loop exits after the first step that exceeds the budget. `None` = unconstrained. |

```python
# Simulate a constrained 2-core drone with a 100 ms/round compute cap
train_gossip(data, partitions, channel, epochs=50, local_steps=20,
             num_threads=2, time_budget_ms=100)
```

### Simulator-Driven Decentralized Mode

When `--use-simulator-integration` is enabled, centralized training still uses
the full graph, but FedAvg and gossip stop training on one fixed partition
subgraph for the entire run.

- Each decentralized drone reads a fresh local training graph from
    `SpatialGridSimulator.get_local_view(...)` at every round.
- If `--simulator-step-drones` is enabled, drones move by one
    partition-respecting grid step before that round's local-view extraction.
- Gossip exchanges are limited to dropout survivors that are also within the
    configured Manhattan communication radius of one another.
- FedAvg uses the same dropout model, but only drones within the configured
    communication radius of a fixed grid-centroid base station may uplink their
    updates on a communication round.
- Being out of range is treated as a topology constraint, not as a dropout or
    blackout event in `interruptions.json`.

### Modeling Notes

- `data.x` now contains 7 features per node: fractional wetland coverage, normalized `(x, y)` position, normalized goal `(x, y)`, and goal-relative deltas.
- Wetland coverage is computed per grid cell as the clipped union area of intersecting wetland polygons divided by the cell area, so the first feature channel lies in `[0, 1]` instead of being binary.
- Dijkstra traversal cost scales linearly with that coverage signal: `land_cost + coverage * (wetland_cost - land_cost)`, with the default endpoints still equal to 1 for dry land and 5 for fully wetland-covered cells.
- Centralized, FedAvg, and gossip all start from the same initialized model state and use the same hidden width so comparisons are not confounded by model capacity or initialization drift.
- `comm_every` is enforced during FedAvg and gossip training. Models still train locally every epoch, but communication dropout is only sampled on scheduled communication epochs.
- When simulator integration is enabled, FedAvg and gossip still start from matched simulator states, but their local data access and communication participation become position-dependent through simulator local views and proximity filtering.
- Greedy-path evaluation averages over multiple seeded start nodes and reports both efficiency and success rate, so identical single-start rollouts no longer dominate the table.

---

### Libraries

Install with `pip install <package>`:

| Package | Required by |
|---|---|
| `geopandas` | `load_data.py`, `build_graph.py` |
| `shapely` | included with geopandas |
| `numpy` | all modules |
| `torch` | all training modules |
| `torch_geometric` | `build_graph.py`, `partition.py`, `train.py`, `federated_agent.py` |
| `matplotlib` | `grid_sim.py`, `evaluations.py` |
