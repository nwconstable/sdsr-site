# Project Issues

Issues are ordered by implementation dependency. Each issue is self-contained
and written for an autonomous coding agent: it includes full context, the exact
file to write, expected function/class signatures, and a verifiable acceptance
checklist.

---

## Issue #1 — Implement drone node partitioning ✓ DONE

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
- [x] `partition_nodes` returns exactly K non-empty arrays
- [x] Union of all arrays equals `range(grid_size**2)` (no node is missing or duplicated)
- [x] `build_local_subgraph` edge_index contains only edges internal to the partition
- [x] Re-indexed node IDs run from 0 to len(node_indices)-1
- [x] Works for K in {2, 3, 4, 5} and grid_size in {10, 50, 100}
- [x] `__main__` block prints partition sizes and a sample subgraph summary
- [x] `train.py` private `_build_local_subgraph` is removed and replaced with an import from `partition.py`
- [x] `federated_agent.py` inline subgraph construction is consolidated to use `build_local_subgraph` from `partition.py`

---

## Issue #2 — Implement communication constraints and protocol interruption logger ✓ DONE

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
- [x] `is_comm_round(step)` returns True iff `step % comm_every == 0`
- [x] `sample_participants` drops each drone independently with probability `dropout_p`
- [x] `gossip_pairs` returns non-overlapping pairs from the participant list
- [x] `record_round` logs a "blackout" when the per-round dropout rate > `baseline_p`
- [x] `summary()` returns keys: `total_rounds`, `total_dropouts`, `total_blackouts`, `per_drone`
- [x] `save(path)` writes valid JSON containing the full event list
- [x] `CommunicationChannel` calls `record_round` automatically inside `sample_participants`
- [x] `__main__` block simulates 50 rounds with 4 drones and prints the summary

---

## Issue #3 — Implement spatial grid simulator ✓ DONE

**Labels:** `feature` `simulation`
**File:** `project/src/grid_sim.py` (implemented as `grid_sim.py`; `main.py` imports from this name)
**Depends on:** #1 (partition.py), build_graph.py (done)

### Context
The simulator places each drone at a node on the grid and tracks its position
across training rounds. It provides per-drone local subgraph views and
generates matplotlib visualisations showing drone positions overlaid on the
wetland heatmap.

**Note — bugs found and fixed (see Issue #11):** The initial implementation
had four runtime-crashing bugs: a `drone_positions` attribute/method naming
collision, `NeighborLoader` misuse in `step_drones`, incorrect radius logic
in `get_local_view`, and a wrong output directory in `__main__`.

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
- [x] Initial drone position is the centroid node of each drone's partition
- [x] `step_drones` keeps each drone within its own partition's node set
- [x] `get_local_view` returns a valid PyG Data object with x and edge_index
- [x] `visualize` saves a `.png` file; file is created and non-empty
- [x] Works with K in {2, 3, 4, 5} and grid_size in {10, 50}
- [x] `__main__` block runs 3 steps and saves images to `project/results/sim/`

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
- [x] With dropout_p=0, final MSE should be lower than an untrained model

---

## Issue #7 — Implement evaluation and plots ✓ DONE

**Labels:** `feature` `evaluation`
**File:** `project/src/evaluations.py`
**Depends on:** #4 (centralized), #5 (FedAvg), #6 (gossip)

### Context
We need a single evaluation module that compares the three training methods.
It should print a compact MSE table, save a convergence plot, and perform a
greedy routing-style rollout using the predicted node distances.

### Required functions

```python
def compare_convergence(
    central_losses: list[float],
    fed_losses: list[float],
    gossip_losses: list[float],
    output_path: str | Path,
) -> None:
    """Save a matplotlib line plot of MSE vs round for all methods."""

def eval_greedy_path(
    data: Data,
    model: torch.nn.Module,
    grid_size: int,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (greedy_path_cost, optimal_path_cost, efficiency_ratio)."""

def run_full_evaluation(
    data: Data,
    central_model: torch.nn.Module,
    fed_model: torch.nn.Module,
    gossip_models: list[torch.nn.Module],
    central_losses: list[float],
    fed_losses: list[float],
    gossip_losses: list[float],
    output_dir: str | Path,
    grid_size: int,
) -> None:
    """Print tables and save plots for method comparison."""
```

### Acceptance criteria
- [x] `compare_convergence` saves a non-empty PNG file
- [x] `eval_greedy_path` returns finite numeric values on a small grid
- [x] `run_full_evaluation` prints a 3-row MSE table for Centralized, FedAvg, Gossip
- [x] Greedy-path evaluation uses the element-wise average of gossip models
- [x] Output files are saved under `project/results/`
- [x] `__main__` block runs a smoke test using randomly initialized models

---

## Issue #8 — Implement main entry point

**Labels:** `feature` `integration`
**File:** `project/src/main.py`
**Depends on:** #1–#7

### Context
The project needs a single CLI entry point that loads data, builds the grid
graph, partitions the graph into K drones, runs all three training methods,
and saves evaluation outputs.

### Required behavior
- Parse CLI args for grid size, number of drones, epochs, local steps,
  comm_every, dropout_p, baseline_p, lr, seed, num_threads, and output_dir
- Build the dataset once and reuse it for all methods
- Run centralized, FedAvg, and gossip training in sequence
- Save interruption logs and evaluation plots
- Print concise status messages and final tables

### Acceptance criteria
- [ ] `python project/src/main.py --help` shows all expected arguments
- [ ] Running the CLI produces training output for all three methods
- [ ] Results directory contains convergence plot and interruption log JSON
- [ ] The run is deterministic when a seed is provided

---

## Issue #9 — Synthetic graph construction (future work)

**Labels:** `feature` `future-work`
**File:** `project/src/build_synthetic.py`
**Depends on:** none

### Context
Add a synthetic graph generator for fast ablations once the real-data pipeline
is stable. This is explicitly future work and should not block the real-data
path.

### Required function

```python
def build_synthetic_graph(
    grid_size: int,
    wetland_density: float,
    seed: int | None = None,
) -> Data:
    """Return a synthetic wetland grid graph with Dijkstra distance labels."""
```

### Acceptance criteria
- [ ] No GIS dependency required
- [ ] Deterministic with a fixed seed
- [ ] Produces a valid PyG Data object compatible with the rest of the pipeline

---

## Issue #10 — Bug: `main.py` integration errors ✓ DONE

**Labels:** `bug` `integration`
**File:** `project/src/main.py`
**Depends on:** #8

### Context
The initial main-entry integration had broken imports and inconsistent module
names that prevented the full pipeline from running.

### Acceptance criteria
- [x] `main.py` imports the implemented module names actually present in `project/src/`
- [x] The end-to-end CLI runs without import errors

---

## Issue #11 — Bug: `grid_sim.py` `SpatialGridSimulator` runtime crashes ✓ DONE

**Labels:** `bug` `simulation`
**File:** `project/src/grid_sim.py`
**Depends on:** #3

### Context
The simulator had runtime-crashing attribute and local-view bugs that made the
swarm visualization path unreliable.

### Acceptance criteria
- [x] No attribute/method naming collision remains
- [x] Local-view extraction works without NeighborLoader misuse
- [x] Smoke test saves images to the correct directory

---

## Issue #12 — Bug: `evaluations.py` `eval_greedy_path` broken ✓ DONE

**Labels:** `bug` `evaluation`
**File:** `project/src/evaluations.py`
**Depends on:** #7

### Context
The original greedy evaluator could fail due to disconnected logic and did not
produce a trustworthy path-quality summary.

### Acceptance criteria
- [x] Greedy evaluation runs to completion on the real graph
- [x] Returned metrics are finite and reproducible with a fixed seed

---

## Issue #13 — Bug: distance-regression task missing goal-conditioned features ✓ DONE

**Labels:** `bug` `modeling`
**File:** `project/src/build_graph.py`, `project/src/model.py`, `project/src/partition.py`
**Depends on:** #7, #8

### Context
The model was being asked to predict distance-to-goal labels without receiving
goal-conditioned input features, making the task under-specified and the
method-comparison results difficult to defend.

### Acceptance criteria
- [x] Node features include enough goal-conditioned information for the task to be identifiable
- [x] Local subgraphs preserve the metadata needed for consistent training and evaluation
- [x] README reflects the expanded feature set

---

## Issue #14 — Bug: training-method comparison had fairness and scheduling drift ✓ DONE

**Labels:** `bug` `training`
**File:** `project/src/main.py`, `project/src/train.py`, `project/src/federated_agent.py`, `project/src/comms.py`
**Depends on:** #5, #6, #8, #13

### Context
The three training methods were not being compared under fair conditions due
to architecture mismatch, shared communication state, and communication
scheduling that was represented but not enforced.

### Acceptance criteria
- [x] Centralized, FedAvg, and gossip use matched model capacity by default
- [x] FedAvg and gossip do not share mutable communication-channel state
- [x] Communication happens only on configured rounds in the actual training loops
- [x] README reflects the comparison rules

---

## Issue #15 — Bug: greedy-path evaluation was too weak to defend ✓ DONE

**Labels:** `bug` `evaluation`
**File:** `project/src/evaluations.py`
**Depends on:** #7, #13, #14

### Context
The prior greedy-path summary used a single start and could report identical
rows without distinguishing evaluator weakness from true behavioral similarity.

### Acceptance criteria
- [x] Greedy-path evaluation averages over multiple starts
- [x] Reports goal-reaching success rate alongside path cost
- [x] Uses the averaged gossip model for gossip evaluation
- [x] README documents the stronger evaluation protocol

---

## Issue #16 — Add reusable scientific-workflow custom agent ✓ DONE

**Labels:** `feature` `workflow` `documentation`
**File:** `.github/agents/sdsr-scientific-workflow.agent.md`
**Depends on:** `.github/copilot.instructions.md`, `.github/issues.md`

### Context
The project now has a clearer operating model than the base workspace
instructions alone capture. Work should preserve the scientific framing used in
recent debugging and comparison work: hypothesis, implementation, test,
results, with human-facing documentation updated at each step.

The custom agent should encode the project goal, method-comparison rules,
evaluation discipline, and documentation expectations so future work does not
drift back toward theory-only or under-verified changes.

### Acceptance criteria
- [x] A user-invocable custom agent exists under `.github/agents/`
- [x] The agent description includes the SDSR experiment, centralized vs FedAvg vs gossip comparison, and scientific workflow trigger phrases
- [x] The agent body requires a hypothesis → implementation → test → results loop
- [x] The agent reminds future work to update `project/README.md` and `.github/issues.md`
- [x] The agent captures the fairness and evaluation lessons learned from the recent investigation# Project Issues

Issues are ordered by implementation dependency. Each issue is self-contained
and written for an autonomous coding agent: it includes full context, the exact
file to write, expected function/class signatures, and a verifiable acceptance
checklist.

---

## Issue #1 — Implement drone node partitioning ✓ DONE

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
- [x] `partition_nodes` returns exactly K non-empty arrays
- [x] Union of all arrays equals `range(grid_size**2)` (no node is missing or duplicated)
- [x] `build_local_subgraph` edge_index contains only edges internal to the partition
- [x] Re-indexed node IDs run from 0 to len(node_indices)-1
- [x] Works for K in {2, 3, 4, 5} and grid_size in {10, 50, 100}
- [x] `__main__` block prints partition sizes and a sample subgraph summary
- [x] `train.py` private `_build_local_subgraph` is removed and replaced with an import from `partition.py`
- [x] `federated_agent.py` inline subgraph construction is consolidated to use `build_local_subgraph` from `partition.py`

---

## Issue #2 — Implement communication constraints and protocol interruption logger ✓ DONE

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
- [x] `is_comm_round(step)` returns True iff `step % comm_every == 0`
- [x] `sample_participants` drops each drone independently with probability `dropout_p`
- [x] `gossip_pairs` returns non-overlapping pairs from the participant list
- [x] `record_round` logs a "blackout" when the per-round dropout rate > `baseline_p`
- [x] `summary()` returns keys: `total_rounds`, `total_dropouts`, `total_blackouts`, `per_drone`
- [x] `save(path)` writes valid JSON containing the full event list
- [x] `CommunicationChannel` calls `record_round` automatically inside `sample_participants`
- [x] `__main__` block simulates 50 rounds with 4 drones and prints the summary

---

## Issue #3 — Implement spatial grid simulator ✓ DONE

**Labels:** `feature` `simulation`
**File:** `project/src/grid_sim.py` (implemented as `grid_sim.py`; `main.py` imports from this name)
**Depends on:** #1 (partition.py), build_graph.py (done)

### Context
The simulator places each drone at a node on the grid and tracks its position
across training rounds. It provides per-drone local subgraph views and
generates matplotlib visualisations showing drone positions overlaid on the
wetland heatmap.

**Note — bugs found and fixed (see Issue #11):** The initial implementation
had four runtime-crashing bugs: a `drone_positions` attribute/method naming
collision, `NeighborLoader` misuse in `step_drones`, incorrect radius logic
in `get_local_view`, and a wrong output directory in `__main__`.

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
- [x] Initial drone position is the centroid node of each drone's partition
- [x] `step_drones` keeps each drone within its own partition's node set
- [x] `get_local_view` returns a valid PyG Data object with x and edge_index
- [x] `visualize` saves a `.png` file; file is created and non-empty
- [x] Works with K in {2, 3, 4, 5} and grid_size in {10, 50}
- [x] `__main__` block runs 3 steps and saves images to `project/results/sim/`

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

## Issue #7 — Implement evaluation and plots ✓ DONE

**Labels:** `feature` `evaluation`
**File:** `project/src/evaluations.py` (implemented as `evaluations.py`; `main.py` imports from this name)
**Depends on:** #4, #5, #6 (all train functions)

### Context
The evaluation module compares all three training methods and optionally
validates learned representations by simulating greedy navigation on the grid.

**Note — bugs found and fixed (see Issue #12):** The initial implementation
of `eval_greedy_path` called `model(x=data.y, edge_index=n)` where `n` is an
integer (not a tensor), accumulated costs into a scalar instead of a list,
had no cycle guard, and left a debug `print`. Fixed by rewriting with proper
graph-edge traversal, correct model call, `grid_size*2` step limit, and
optional-cost traversal logic.

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
- [x] `compare_convergence` saves a valid PNG with all three curves labelled
- [x] `eval_greedy_path` terminates (cycle guard of grid_size*2 steps)
- [x] `eval_greedy_path` returns efficiency_ratio in (0, 1] (greedy >= optimal cost)
- [x] `run_full_evaluation` prints a formatted MSE table to stdout
- [x] Output directory is created if it does not exist

---

## Issue #8 — Implement main entry point

**Labels:** `feature` `integration`
**File:** `project/src/main.py`
**Depends on:** #1, #2, #3, #4, #5, #6, #7

### Context
main.py wires all modules together behind an argparse CLI so that the full
experiment can be reproduced with a single command.

**Note — bugs found and fixed (see Issue #10):** The initial implementation
had three integration errors: `from load_data import gdf` (not callable) →
fixed to `load_gdf()`; `SpatialGridSimulator(data, partitions, channel, gs)`
with wrong arg order and extra `channel` → fixed to `SpatialGridSimulator(gs,
partitions, data)`; `run_full_evaluation` missing `output_dir` argument.

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

---

## Issue #10 — Bug: `main.py` integration errors ✓ DONE

**Labels:** `bug` `integration`
**File:** `project/src/main.py`
**Depends on:** #3, #7 (fixed), #8

### Bugs

1. **`load_data` import/call mismatch** (`main.py:7,44`):
   `from load_data import gdf` imports the module-level variable `gdf`, then
   `gdf_data = gdf()` tries to call it as a function → `TypeError`.
   Fix: `from load_data import load_gdf` + `gdf_data = load_gdf()`.

2. **`SpatialGridSimulator` wrong constructor call** (`main.py:63`):
   `SpatialGridSimulator(data, partitions, channel, args.grid_size)` passes
   `data` first and adds an extra `channel` arg not in the constructor.
   Spec: `__init__(self, grid_size, partitions, data)`.
   Fix: `SpatialGridSimulator(args.grid_size, partitions, data)`.

3. **`run_full_evaluation` missing `output_dir`** (`main.py:68`):
   Call omits the `output_dir` keyword argument; plots are saved to the
   default `"project/results"` instead of `args.output_dir`.
   Fix: add `output_dir=args.output_dir`.

### Acceptance criteria
- [x] `load_gdf()` is called correctly
- [x] `SpatialGridSimulator` receives `(grid_size, partitions, data)`
- [x] `run_full_evaluation` receives `output_dir=args.output_dir`

---

## Issue #11 — Bug: `grid_sim.py` `SpatialGridSimulator` runtime crashes ✓ DONE

**Labels:** `bug` `simulation`
**File:** `project/src/grid_sim.py`
**Depends on:** #3

### Bugs

1. **`drone_positions` attribute/method name collision** (`grid_sim.py:19,25`):
   `self.drone_positions = {}` (instance dict) shadows `def drone_positions(self)`
   (method). Calling `simulator.drone_positions()` raises `TypeError: 'dict'
   object is not callable`. Fix: rename instance dict to `self._positions`.

2. **`step_drones` uses `NeighborLoader` incorrectly** (`grid_sim.py:31-45`):
   `input_nodes=torch.tensor([pos]).to_sparse()` is not a valid `NeighborLoader`
   argument and crashes. Using a DataLoader for drone movement is also
   architecturally wrong. Fix: replace with direct 4-neighbour grid arithmetic
   filtered to the drone's partition set.

3. **`get_local_view` radius expansion is wrong** (`grid_sim.py:52-67`):
   Calls `next(iter(neighbor_loader))` in a loop, which reinitialises the
   iterator each time; no actual radius expansion happens.
   Fix: BFS over the 4-neighbour grid for `radius` hops.

4. **`__main__` wrong output directory and grid_size mismatch**:
   Saves to `data/DebugResults` (not `project/results/sim/`). Passes
   `GRID_SIZE=50` as the simulator `grid_size` while running partitions for
   grids as small as 10, causing out-of-bounds node indices.

### Acceptance criteria
- [x] `drone_positions()` method returns a dict without `TypeError`
- [x] `step_drones()` moves each drone to a partition-adjacent node using grid arithmetic
- [x] `get_local_view(drone_id, radius=2)` returns a PyG Data with BFS-expanded node set
- [x] `__main__` saves to `project/results/sim/` with matching grid_size

---

## Issue #12 — Bug: `evaluations.py` `eval_greedy_path` broken ✓ DONE

**Labels:** `bug` `evaluation`
**File:** `project/src/evaluations.py`
**Depends on:** #7

### Bugs

1. **Wrong model call signature** (`evaluations.py:68`):
   `model(x=data_targets, edge_index=n)` passes `data.y` (labels) as `x` and
   an integer `n` as `edge_index` → `TypeError`. Must call
   `model(data.x, data.edge_index)` once to get predictions for all nodes.

2. **Neighbours not filtered by graph edges** (`evaluations.py:58`):
   `neighbors = [n for n in range(data.num_nodes) if n not in visited]`
   treats all unvisited nodes as reachable, ignoring the actual graph topology.
   Fix: build an adjacency list from `data.edge_index` and traverse only
   actual edges.

3. **Optimal cost accumulates into a scalar** (`evaluations.py:80-85`):
   `all_costs += abs(model(...))` sums into a `float`, then
   `np.mean(all_costs[:grid_size])` tries to slice a scalar → `TypeError`.
   Fix: optimal cost = `data.y[start_node].item()` (pre-computed Dijkstra label).

4. **No cycle guard; debug `print` left in** (`evaluations.py:56,67`):
   `while len(visited) < min(grid_size, data.num_nodes)` has no step limit;
   a disconnected or fully-visited graph loops forever. Debug `print` on
   line 67 spams stdout. Fix: cap at `grid_size * 2` steps; remove `print`.

### Acceptance criteria
- [x] `eval_greedy_path` calls `model(data.x, data.edge_index)` once, not per-node
- [x] Greedy step selects from actual graph neighbours (via `edge_index`)
- [x] Terminates within `grid_size * 2` steps
- [x] `efficiency_ratio` in `(0, 1]`; no debug output

---

## Issue #13 — Bug: distance-regression task missing goal-conditioned features ✓ DONE

**Labels:** `bug` `modeling`
**Files:** `project/src/build_graph.py`, `project/src/model.py`, `project/src/partition.py`
**Depends on:** build_graph.py (done), model.py (done)

### Context
The original graph builder exposed only a single `wetland_presence` feature per
node while asking the model to predict distance to a specific goal node. That
task is under-specified: the goal location must be part of the model input if
we want the learned distance field to correspond to a chosen destination.

### Fix

1. Expand `data.x` to include goal-conditioned features:
    - wetland presence
    - normalized node `(x, y)` position
    - normalized goal `(x, y)` position
    - goal-relative deltas `(goal_x - x, goal_y - y)`
2. Update `WetlandGCN` to accept configurable `in_channels`.
3. Preserve goal metadata when creating local partition subgraphs.

### Acceptance criteria
- [x] `build_pyg_data` stores goal-conditioned features in `data.x`
- [x] `WetlandGCN` accepts the feature dimension explicitly
- [x] Partition subgraphs preserve position and goal metadata needed for downstream debugging/evaluation

---

## Issue #14 — Bug: training-method comparison had fairness and scheduling drift ✓ DONE

**Labels:** `bug` `training` `fairness`
**Files:** `project/src/main.py`, `project/src/train.py`, `project/src/federated_agent.py`
**Depends on:** #2, #5, #6, #8

### Context
Several implementation details made the FedAvg/Gossip vs centralized comparison
hard to defend:

1. `comm_every` was not enforced in the training loops.
2. FedAvg used a narrower model than centralized/gossip.
3. Gossip and FedAvg shared the same `CommunicationChannel` instance.
4. Local models did not start from the same initialization.
5. FedAvg trained on a random local radius subset instead of the full partition.

### Fix

1. Enforce `channel.is_comm_round(epoch)` in both decentralized training loops.
2. Use a shared 64-hidden-unit architecture across all methods.
3. Instantiate separate communication channels for gossip and FedAvg.
4. Seed all methods from the same initial model state.
5. Train FedAvg nodes on the full partition subgraph for parity with gossip.

### Acceptance criteria
- [x] Gossip and FedAvg communicate only on scheduled epochs
- [x] FedAvg and gossip use the same hidden width as centralized
- [x] `main.py` builds separate channels and writes both interruption logs
- [x] All three methods start from a shared initial state
- [x] FedAvg local training consumes the full partition graph

---

## Issue #15 — Bug: greedy-path evaluation was too weak to defend ✓ DONE

**Labels:** `bug` `evaluation`
**Files:** `project/src/evaluations.py`
**Depends on:** #7, #12

### Context
The prior greedy-path report summarized a single seeded rollout and, for the
gossip row, evaluated only the first local model rather than the averaged
swarm model. That made identical rows suspicious even when the underlying
predictions differed.

### Fix

1. Average gossip model weights before path evaluation.
2. Evaluate greedy navigation over multiple seeded start nodes.
3. Report goal-reaching success rate alongside mean efficiency.
4. Save a dedicated greedy-path summary plot.

### Acceptance criteria
- [x] Gossip path evaluation uses the averaged swarm model
- [x] Greedy metrics are computed across multiple seeded start nodes
- [x] Success rate is reported in the console output
- [x] `greedy_path_evaluation.png` is written to the output directory
