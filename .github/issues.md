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
- [x] Re-indexed node IDs run from 0 to `len(node_indices)-1`
- [x] Works for K in {2, 3, 4, 5} and `grid_size` in {10, 50, 100}
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
- [x] `record_round` logs a `"blackout"` when the per-round dropout rate > `baseline_p`
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
- [x] `get_local_view` returns a valid PyG `Data` object with `x` and `edge_index`
- [x] `visualize` saves a `.png` file; file is created and non-empty
- [x] Works for K in {2, 3, 4, 5} and `grid_size` in {10, 50}
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
- [x] `__main__` block in `train.py` runs a 10-epoch smoke test with a 10-node graph

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
- [x] With `dropout_p=0`, final MSE is lower than an untrained model

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
- [x] Weight exchange averages `state_dict`s of each pair; no central server involved
- [x] Evaluation uses the element-wise average of **all** drone `state_dict`s on the full graph
- [x] Returns a MSE list with one entry per communication round
- [x] With `dropout_p=0`, final MSE is lower than an untrained model
- [x] Gossip loss curve is plottable alongside centralized and FedAvg curves

---

## Issue #7 — Implement evaluation and plots ✓ DONE

**Labels:** `feature` `evaluation`
**File:** `project/src/evaluations.py` (implemented as `evaluations.py`; `main.py` imports from this name)
**Depends on:** #4 (centralized), #5 (FedAvg), #6 (gossip)

### Context
We need a single evaluation module that compares the three training methods.
It should print a compact MSE table, save a convergence plot, and perform a
greedy routing-style rollout using the predicted node distances.

The module should support both aggregate training comparison and behavioral
validation of learned distance fields via greedy navigation on the grid.

**Note — bugs found and fixed (see Issue #12):** The initial implementation
of `eval_greedy_path` had a broken model call, invalid neighbour traversal,
incorrect optimal-cost accumulation, and no reliable cycle guard, making the
path-quality summary difficult to defend.

### Required functions

```python
def compare_convergence(
    central_losses: list[float],
    fed_losses: list[float],
    gossip_losses: list[float],
    output_dir: str | Path = "project/results",
) -> None:
    """Save a matplotlib line plot of MSE vs round for all methods."""

def eval_greedy_path(
    data: Data,
    model: torch.nn.Module,
    grid_size: int,
    start_node: int | None = None,
    seed: int | None = None,
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
    output_dir: str | Path = "project/results",
    grid_size: int,
) -> None:
    """Print tables and save plots for method comparison."""
```

### Acceptance criteria
- [x] `compare_convergence` saves a valid non-empty PNG with all three curves labelled
- [x] `eval_greedy_path` returns finite numeric values on a small grid
- [x] `eval_greedy_path` terminates within `grid_size * 2` steps
- [x] `efficiency_ratio` is in `(0, 1]`
- [x] `run_full_evaluation` prints a formatted 3-row MSE table for Centralized, FedAvg, Gossip
- [x] Greedy-path evaluation uses the element-wise average of gossip models
- [x] Output files are saved under `project/results/`, creating the output directory if needed
- [x] `__main__` block runs a smoke test using randomly initialized models

---

## Issue #8 — Implement main entry point ✓ DONE

**Labels:** `feature` `integration`
**File:** `project/src/main.py`
**Depends on:** #1–#7

### Context
The project needs a single CLI entry point that loads data, builds the grid
graph, partitions the graph into K drones, runs all three training methods,
and saves evaluation outputs.

**Note — bugs found and fixed (see Issue #10):** The initial main-entry
integration had broken imports, inconsistent constructor calls, and a missing
`output_dir` pass-through that prevented the full pipeline from running
correctly.

### Required behavior
- Parse CLI args for grid size, number of drones (`--K`), epochs, local
  steps, `comm_every`, `dropout_p`, `baseline_p`, `lr`, `seed`,
  `num_threads`, and `output_dir`
- Document defaults in the argparse help output
- Build the dataset once and reuse it for all methods
- Construct the simulator and communication channel(s) needed for the run
- Run centralized, FedAvg, and gossip training in sequence
- Save interruption logs and evaluation plots
- Print concise status messages and final tables

### Execution order
1. Load the GeoDataFrame via `load_data`
2. Build the graph dataset once
3. Partition the graph into drone-local node sets
4. Construct communication channel(s)
5. Construct the spatial simulator if the workflow uses it
6. Run centralized training
7. Run FedAvg training
8. Run gossip training
9. Run the full evaluation pass
10. Save interruption logs under the selected output directory

### Acceptance criteria
- [x] `python project/src/main.py --help` shows all expected arguments with documented defaults
- [x] Running the CLI produces training output for all three methods
- [x] All pipeline steps execute without import or runtime errors with default arguments
- [x] Results directory contains convergence plot and interruption log JSON
- [x] The run is deterministic when a seed is provided
- [x] Program exits with code 0 on success

---

## Issue #9 — Synthetic graph construction (future work)

**Labels:** `feature` `future-work` `data`
**File:** `project/src/build_synthetic.py`
**Depends on:** build_graph.py (done) — implement after `main.py` is validated end-to-end

### Context
Add a synthetic graph generator for fast ablations once the real-data pipeline
is stable. This is explicitly future work and should not block the real-data
path.

The synthetic builder should mirror the real graph-builder output closely so
that downstream training, evaluation, and comparison code can run unchanged.

### Required function

```python
def build_synthetic_graph(
    grid_size: int,
    wetland_density: float,
    seed: int | None = None,
) -> tuple[Data, int]:
    """Return a synthetic wetland grid graph and goal node compatible with build_pyg_data."""
```

### Acceptance criteria
- [ ] No GIS dependency required
- [ ] Deterministic with a fixed seed
- [ ] Produces output compatible with the rest of the pipeline and `build_pyg_data`
- [ ] `wetland_density=0.0` produces all-zero wetland indicators; `wetland_density=1.0` produces all-one indicators
- [ ] `seed` controls both wetland assignment and goal selection reproducibly
- [ ] `__main__` block builds a small synthetic graph and prints a summary

---

## Issue #10 — Bug: `main.py` integration errors ✓ DONE

**Labels:** `bug` `integration`
**File:** `project/src/main.py`
**Depends on:** #8

### Context
The initial main-entry integration had broken imports and inconsistent module
names that prevented the full pipeline from running.

### Bugs

1. **`load_data` import/call mismatch**:
   `gdf` was imported and called like a function instead of calling the real
   loader.
2. **`SpatialGridSimulator` wrong constructor call**:
   `main.py` passed arguments in the wrong order and included an extra
   argument not accepted by the constructor.
3. **`run_full_evaluation` missing `output_dir`**:
   Evaluation artifacts were written to the default location rather than the
   requested output directory.

### Acceptance criteria
- [x] `main.py` imports the implemented module names actually present in `project/src/`
- [x] `load_gdf()` is called correctly
- [x] `SpatialGridSimulator` receives `(grid_size, partitions, data)`
- [x] `run_full_evaluation` receives `output_dir=args.output_dir`
- [x] The end-to-end CLI runs without import errors

---

## Issue #11 — Bug: `grid_sim.py` `SpatialGridSimulator` runtime crashes ✓ DONE

**Labels:** `bug` `simulation`
**File:** `project/src/grid_sim.py`
**Depends on:** #3

### Context
The simulator had runtime-crashing attribute and local-view bugs that made the
swarm visualization path unreliable.

### Bugs

1. **`drone_positions` attribute/method naming collision**:
   An instance attribute shadowed the method, causing `TypeError` when the
   method was called.
2. **`step_drones` used `NeighborLoader` incorrectly**:
   Drone movement depended on an invalid loader call instead of direct grid
   adjacency logic.
3. **`get_local_view` radius expansion was incorrect**:
   The old loop reinitialized the iterator and did not perform true hop-based
   expansion.
4. **`__main__` used the wrong output directory and inconsistent grid sizes**:
   The smoke test could write to the wrong place and construct invalid node
   indices.

### Acceptance criteria
- [x] No attribute/method naming collision remains
- [x] `step_drones()` moves each drone using partition-respecting grid adjacency
- [x] Local-view extraction works without `NeighborLoader` misuse
- [x] `get_local_view(drone_id, radius=2)` returns a BFS-expanded local subgraph
- [x] Smoke test saves images to the correct directory with matching grid size

---

## Issue #12 — Bug: `evaluations.py` `eval_greedy_path` broken ✓ DONE

**Labels:** `bug` `evaluation`
**File:** `project/src/evaluations.py`
**Depends on:** #7

### Context
The original greedy evaluator could fail due to disconnected logic and did not
produce a trustworthy path-quality summary.

### Bugs

1. **Wrong model call signature**:
   The model was called with labels as features and an integer as the edge
   index.
2. **Neighbours were not filtered by actual graph edges**:
   The traversal logic treated all unvisited nodes as reachable.
3. **Optimal cost accumulation was incorrect**:
   The implementation accumulated into a scalar and then treated it like a
   sequence.
4. **No cycle guard and debug output remained**:
   The loop could fail to terminate and still emitted debug prints.

### Acceptance criteria
- [x] `eval_greedy_path` calls `model(data.x, data.edge_index)` once, not per-node
- [x] Greedy evaluation traverses actual graph neighbours derived from `edge_index`
- [x] Greedy evaluation runs to completion on the real graph
- [x] Terminates within `grid_size * 2` steps
- [x] Returned metrics are finite and reproducible with a fixed seed
- [x] `efficiency_ratio` is in `(0, 1]` and no debug output remains

---

## Issue #13 — Bug: distance-regression task missing goal-conditioned features ✓ DONE

**Labels:** `bug` `modeling`
**Files:** `project/src/build_graph.py`, `project/src/model.py`, `project/src/partition.py`
**Depends on:** #7, #8

### Context
The model was being asked to predict distance-to-goal labels without receiving
goal-conditioned input features, making the task under-specified and the
method-comparison results difficult to defend.

The original graph builder exposed only a wetland indicator while asking the
model to learn a distance field tied to a specific destination. That task is
not identifiable unless the goal location is encoded in the input features.

### Fix

1. Expand `data.x` to include goal-conditioned features such as node position,
   goal position, and goal-relative deltas.
2. Update `WetlandGCN` to accept the input feature dimension explicitly.
3. Preserve goal-related metadata when building local partition subgraphs.

### Acceptance criteria
- [x] Node features include enough goal-conditioned information for the task to be identifiable
- [x] `build_pyg_data` stores goal-conditioned features in `data.x`
- [x] `WetlandGCN` accepts the feature dimension explicitly
- [x] Local subgraphs preserve the metadata needed for consistent training and evaluation
- [x] README reflects the expanded feature set

---

## Issue #14 — Bug: training-method comparison had fairness and scheduling drift ✓ DONE

**Labels:** `bug` `training` `fairness`
**Files:** `project/src/main.py`, `project/src/train.py`, `project/src/federated_agent.py`, `project/src/comms.py`
**Depends on:** #5, #6, #8, #13

### Context
The three training methods were not being compared under fair conditions due
to architecture mismatch, shared communication state, and communication
scheduling that was represented but not enforced.

Specific drift included unscheduled communication, narrower FedAvg models,
shared channel state between methods, inconsistent initialization, and
non-comparable local training scopes.

### Fix

1. Enforce `channel.is_comm_round(...)` inside the decentralized training loops.
2. Use matched model capacity across centralized, FedAvg, and gossip.
3. Avoid shared mutable communication-channel state between methods.
4. Seed all methods from a shared initial model state.
5. Ensure FedAvg and gossip train on comparable local partitions.

### Acceptance criteria
- [x] Centralized, FedAvg, and gossip use matched model capacity by default
- [x] FedAvg and gossip do not share mutable communication-channel state
- [x] Communication happens only on configured rounds in the actual training loops
- [x] All three methods start from a shared initial state
- [x] FedAvg local training consumes the full partition graph for parity with gossip
- [x] README reflects the comparison rules

---

## Issue #15 — Bug: greedy-path evaluation was too weak to defend ✓ DONE

**Labels:** `bug` `evaluation`
**Files:** `project/src/evaluations.py`
**Depends on:** #7, #13, #14

### Context
The prior greedy-path summary used a single start and could report identical
rows without distinguishing evaluator weakness from true behavioral similarity.

The gossip row also risked evaluating a single local model instead of the
averaged swarm model, which weakened method-level conclusions.

### Fix

1. Average gossip model weights before path evaluation.
2. Evaluate greedy navigation over multiple seeded start nodes.
3. Report goal-reaching success rate alongside mean path quality.
4. Save a dedicated greedy-path summary artifact.

### Acceptance criteria
- [x] Greedy-path evaluation averages over multiple starts
- [x] Reports goal-reaching success rate alongside path cost
- [x] Uses the averaged gossip model for gossip evaluation
- [x] `greedy_path_evaluation.png` is written to the output directory
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
evaluation discipline, documentation expectations, and VVUQ guidance so future
work does not drift back toward theory-only or under-verified changes.

### Acceptance criteria
- [x] A user-invocable custom agent exists under `.github/agents/`
- [x] The agent description includes the SDSR experiment, centralized vs FedAvg vs gossip comparison, and scientific workflow trigger phrases
- [x] The agent body requires a hypothesis → implementation → test → results loop
- [x] The agent reminds future work to update `project/README.md` and `.github/issues.md`
- [x] The agent captures the fairness and evaluation lessons learned from the recent investigation
- [x] The agent includes VVUQ guidance for verification, validation, and uncertainty quantification
