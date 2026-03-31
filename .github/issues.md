# Project Issues

Issues are ordered by implementation dependency. Each issue is self-contained
and written for an autonomous coding agent: it includes full context, the exact
file to write, expected function/class signatures, and a verifiable acceptance
checklist.

---

## Issue #1 — Implement drone node partitioning

**Labels:** `feature` `data`
**File:** `project/src/partition.py`
**Depends on:** build_graph.py (done)

### Context
The grid graph produced by `build_pyg_data` must be split into K spatial
regions, one per simulated drone. Each drone trains only on its local subgraph.
Partitioning is row-wise strip splitting (column-major strips along the x-axis).

**Note — existing implementation:** A private `_build_local_subgraph` helper
already exists in `train.py` (line ~71):
```python
def _build_local_subgraph(data: Data, node_indices: np.ndarray) -> Data:
    global_idx = torch.tensor(node_indices, dtype=torch.long)
    sub_ei, _ = pyg_subgraph(global_idx, data.edge_index,
                              relabel_nodes=True, num_nodes=data.num_nodes)
    return Data(x=data.x[global_idx], edge_index=sub_ei, y=data.y[global_idx])
```
`partition.py` must provide the **canonical public** version of this function.
Once `partition.py` is implemented, update `train.py` to `from partition import
build_local_subgraph` and remove the private duplicate. `federated_agent.py`
uses an inline equivalent — update it to import from `partition.py` as well.

### Required functions

```python
def partition_nodes(grid_size: int, K: int, method: str = "grid") -> list[np.ndarray]:
    """Return K arrays of global node indices, one per drone."""

def build_local_subgraph(data: Data, node_indices: np.ndarray) -> Data:
    """Return a PyG Data with re-indexed nodes; only edges where both
    endpoints are in node_indices are kept. Preserves x and y."""
```

### Acceptance criteria
- [ ] `partition_nodes` returns exactly K non-empty arrays
- [ ] Union of all arrays equals `range(grid_size**2)` (no node is missing or duplicated)
- [ ] `build_local_subgraph` edge_index contains only edges internal to the partition
- [ ] Re-indexed node IDs run from 0 to len(node_indices)-1
- [ ] Works for K in {2, 3, 4, 5} and grid_size in {10, 50, 100}
- [ ] `__main__` block prints partition sizes and a sample subgraph summary
- [ ] `train.py` private `_build_local_subgraph` is removed and replaced with an import from `partition.py`
- [ ] `federated_agent.py` inline subgraph construction is consolidated to use `build_local_subgraph` from `partition.py`

---

## Issue #2 — Implement communication constraints and protocol interruption logger

**Labels:** `feature` `simulation`
**File:** `project/src/comms.py`
**Depends on:** #1 (partition.py)

### Context
During FedAvg and gossip training, drones communicate only at scheduled
intervals and may randomly fail to participate (dropout). The
`CommunicationChannel` class models these constraints. The
`ProtocolInterruptionLogger` records every dropout event and flags as
"blackout" any session where the dropout rate exceeds the configured baseline,
enabling post-hoc analysis of anomalous communication failures.

**Note — existing stubs:** `train.py` and `federated_agent.py` both import
`CommunicationChannel` under a `TYPE_CHECKING` guard to avoid a `NameError`
at runtime. A `_MockChannel` stub is used in `train.py __main__` for smoke
testing `train_gossip`. Once `comms.py` is implemented:
- Remove the `_MockChannel` stub from `train.py __main__` and replace it with
  a real `CommunicationChannel(comm_every=1, dropout_p=0.0)`.
- The `TYPE_CHECKING` guard imports in both files become live runtime imports;
  no other changes to those files are needed.

`comms.py` does **not** need to import from `partition.py` directly — the
dependency on #1 is transitive: `sample_participants` and `gossip_pairs` are
consumed by `train_fedavg` / `train_gossip`, which themselves depend on
`build_local_subgraph` from `partition.py`. Implement #1 first so the full
training pipeline can be wired in `main.py` (Issue #8).

### Required classes

```python
class CommunicationChannel:
    def __init__(self, comm_every: int, dropout_p: float,
                 baseline_p: float = None, seed: int = None): ...
    def is_comm_round(self, step: int) -> bool: ...
    def sample_participants(self, drone_ids: list[int]) -> list[int]: ...
    def gossip_pairs(self, drone_ids: list[int]) -> list[tuple[int, int]]: ...
    # logger : ProtocolInterruptionLogger (attached automatically)

class ProtocolInterruptionLogger:
    def __init__(self, baseline_p: float): ...
    def log(self, round_num: int, drone_id: int,
            reason: str):  # reason in {"dropout", "blackout"}
        ...
    def record_round(self, round_num: int, participants: list[int],
                     all_drone_ids: list[int]): ...
    def summary(self) -> dict: ...
    def save(self, path): ...  # JSON output
```

### Acceptance criteria
- [ ] `is_comm_round(step)` returns True iff `step % comm_every == 0`
- [ ] `sample_participants` drops each drone independently with probability `dropout_p`
- [ ] `gossip_pairs` returns non-overlapping pairs from the participant list
- [ ] `record_round` logs a "blackout" when the per-round dropout rate > `baseline_p`
- [ ] `summary()` returns keys: `total_rounds`, `total_dropouts`, `total_blackouts`, `per_drone`
- [ ] `save(path)` writes valid JSON containing the full event list
- [ ] `CommunicationChannel` calls `record_round` automatically inside `sample_participants`
- [ ] `__main__` block simulates 50 rounds with 4 drones and prints the summary

---

## Issue #3 — Implement spatial grid simulator

**Labels:** `feature` `simulation`
**File:** `project/src/simulator.py`
**Depends on:** #1 (partition.py), build_graph.py (done)

### Context
The simulator places each drone at a node on the grid and tracks its position
across training rounds. It provides per-drone local subgraph views and
generates matplotlib visualisations showing drone positions overlaid on the
wetland heatmap.

### Required class

```python
class SpatialGridSimulator:
    def __init__(self, grid_size: int, partitions: list[np.ndarray],
                 data: Data): ...
    def drone_positions(self) -> dict[int, int]:
        """drone_id -> current node index"""
    def step_drones(self):
        """Move each drone to a random adjacent node within its partition."""
    def get_local_view(self, drone_id: int, radius: int = 1) -> Data:
        """Return subgraph of nodes within `radius` hops of drone position."""
    def visualize(self, step: int, output_dir: str | Path):
        """Save wetland heatmap + drone markers to output_dir/sim_step_NNNN.png"""
```

### Acceptance criteria
- [ ] Initial drone position is the centroid node of each drone's partition
- [ ] `step_drones` keeps each drone within its own partition's node set
- [ ] `get_local_view` returns a valid PyG Data object with x and edge_index
- [ ] `visualize` saves a `.png` file; file is created and non-empty
- [ ] Works with K in {2, 3, 4, 5} and grid_size in {10, 50}
- [ ] `__main__` block runs 3 steps and saves images to `project/results/sim/`

---

## Issue #4 — Implement centralized training ✓ DONE

**Labels:** `feature` `training`
**File:** `project/src/train.py`
**Depends on:** model.py (done), build_graph.py (done)

### Context
The centralized baseline trains the WetlandGCN on the full graph. Its loss
curve is the gold standard against which FedAvg and gossip are compared.

### Required function

```python
def train_centralized(
    data: Data,
    model: WetlandGCN,
    epochs: int,
    lr: float = 1e-3,
) -> list[float]:
    """Return per-epoch MSE loss list."""
```

### Acceptance criteria
- [x] Uses `torch.optim.Adam` and `F.mse_loss`
- [x] Returns a list of length `epochs`
- [x] Loss is monotonically decreasing on a small synthetic graph (smoke test)
- [x] Model state is updated in-place
- [x] `__main__` block in train.py runs a 10-epoch smoke test with a 10-node graph

---

## Issue #5 — Implement FedAvg training ✓ DONE

**Labels:** `feature` `training`
**File:** `project/src/federated_agent.py` (implemented via `NodeAgent`/`CentralAgent`; canonical location per spec is `train.py`)
**Depends on:** #1 (partition.py), #2 (comms.py), #4 (train_centralized)

### Context
FedAvg trains K copies of WetlandGCN (one per drone) on local subgraphs.
At each communication round, the server averages participating drones' weights
and broadcasts the result back.

### Required function

```python
def train_fedavg(
    data: Data,
    partitions: list[np.ndarray],
    channel: CommunicationChannel,
    epochs: int,
    local_steps: int,
    lr: float = 1e-3,
) -> tuple[list[float], WetlandGCN]:
    """Return (per-round MSE list, final global model)."""
```

### Acceptance criteria
- [x] Each drone trains on its local subgraph for exactly `local_steps` gradient steps per round
- [x] Only `channel.sample_participants` drones contribute to the average each round
- [x] Averaged weights are broadcast to **all** drones (not just participants)
- [x] Global model is evaluated on the **full** graph after each round
- [x] Returns a MSE list with one entry per communication round
- [x] With dropout_p=0, final MSE should be lower than an untrained model

---

## Issue #6 — Implement gossip training ✓ DONE

**Labels:** `feature` `training`
**File:** `project/src/train.py`
**Depends on:** #1 (partition.py), #2 (comms.py), #4 (train_centralized)

### Context
Gossip training is fully decentralised: drones train locally, then randomly
paired drones average their weights with no central aggregation. The
evaluation model is the element-wise average of all drone models.

### Required function

```python
def train_gossip(
    data: Data,
    partitions: list[np.ndarray],
    channel: CommunicationChannel,
    epochs: int,
    local_steps: int,
    lr: float = 1e-3,
) -> tuple[list[float], list[WetlandGCN]]:
    """Return (per-round MSE list, list of final drone models)."""
```

### Acceptance criteria
- [x] Each drone trains on its local subgraph for exactly `local_steps` gradient steps per round
- [x] `channel.gossip_pairs` determines which drones exchange weights each round
- [x] Weight exchange averages state_dicts of each pair; no central server involved
- [x] Evaluation uses the element-wise average of **all** drone state_dicts on the full graph
- [x] Returns a MSE list with one entry per communication round
- [x] Gossip loss curve is plottable alongside centralized and FedAvg curves

---

## Issue #7 — Implement evaluation and plots

**Labels:** `feature` `evaluation`
**File:** `project/src/evaluate.py`
**Depends on:** #4, #5, #6 (all train functions)

### Context
The evaluation module compares all three training methods and optionally
validates learned representations by simulating greedy navigation on the grid.

### Required functions

```python
def compare_convergence(
    centralized_losses: list[float],
    fedavg_losses: list[float],
    gossip_losses: list[float],
    output_dir: str | Path = "project/results",
) -> None:
    """Save convergence plot to output_dir/convergence.png."""

def eval_greedy_path(
    data: Data,
    model: WetlandGCN,
    grid_size: int,
    start_node: int | None = None,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return (greedy_path_cost, optimal_path_cost, efficiency_ratio)."""

def run_full_evaluation(
    data: Data,
    centralized_model: WetlandGCN,
    fedavg_model: WetlandGCN,
    gossip_models: list[WetlandGCN],
    centralized_losses: list[float],
    fedavg_losses: list[float],
    gossip_losses: list[float],
    output_dir: str | Path = "project/results",
) -> None:
    """Print MSE table, save convergence plot, print greedy path results."""
```

### Acceptance criteria
- [ ] `compare_convergence` saves a valid PNG with all three curves labelled
- [ ] `eval_greedy_path` terminates (cycle guard of grid_size*2 steps)
- [ ] `eval_greedy_path` returns efficiency_ratio in (0, 1] (greedy >= optimal cost)
- [ ] `run_full_evaluation` prints a formatted MSE table to stdout
- [ ] Output directory is created if it does not exist

---

## Issue #8 — Implement main entry point

**Labels:** `feature` `integration`
**File:** `project/src/main.py`
**Depends on:** #1, #2, #3, #4, #5, #6, #7

### Context
main.py wires all modules together behind an argparse CLI so that the full
experiment can be reproduced with a single command.

### Required CLI arguments

| Argument | Type | Default |
|---|---|---|
| `--grid-size` | int | 50 |
| `--K` | int | 4 |
| `--epochs` | int | 200 |
| `--local-steps` | int | 5 |
| `--comm-every` | int | 10 |
| `--dropout-p` | float | 0.1 |
| `--baseline-p` | float | 0.1 |
| `--lr` | float | 1e-3 |
| `--seed` | int | 42 |
| `--output-dir` | str | project/results |

### Execution order
1. Load GeoDataFrame (`load_data.gdf`)
2. `build_pyg_data` -> data
3. `partition_nodes` -> partitions
4. Construct `CommunicationChannel`
5. Construct `SpatialGridSimulator`
6. `train_centralized`
7. `train_fedavg`
8. `train_gossip`
9. `run_full_evaluation`
10. `channel.logger.save(output_dir/interruptions.json)`

### Acceptance criteria
- [ ] All arguments are parsed via argparse with documented defaults
- [ ] All steps execute without error with default arguments on a 50x50 grid
- [ ] `project/results/convergence.png` and `project/results/interruptions.json` are written
- [ ] Program exits with code 0 on success

---

## Issue #9 — Synthetic graph construction (future work)

**Labels:** `future-work` `data`
**File:** `project/src/build_synthetic.py`
**Depends on:** build_graph.py (done) — implement AFTER main.py is validated end-to-end

### Context
A synthetic graph generator enables fast ablation studies without requiring
the GeoPackage file. It mirrors the output format of `build_pyg_data` exactly
so all downstream code is reusable unchanged.

### Required function

```python
def build_synthetic_graph(
    grid_size: int,
    wetland_density: float,
    seed: int | None = None,
) -> tuple[Data, int]:
    """
    Generate a random grid graph.

    wetland_density : float in [0, 1] — fraction of nodes assigned wetland=1
    Returns same (data, goal_node) tuple as build_pyg_data.
    Reuses build_edge_index and compute_dijkstra_labels from build_graph.py.
    """
```

### Acceptance criteria
- [ ] Output `data` has identical field names and shapes as `build_pyg_data` output
- [ ] `wetland_density=0.0` produces all-zero x; `wetland_density=1.0` produces all-one x
- [ ] `seed` controls both wetland assignment and goal node selection reproducibly
- [ ] `__main__` block builds a 20x20 synthetic graph and prints a summary
- [ ] No GIS libraries (geopandas, shapely) are imported
