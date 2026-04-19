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

---

## Issue #17 — Add issue-intake custom agent ✓ DONE

**Labels:** `feature` `workflow` `documentation`
**File:** `.github/agents/sdsr-issue-intake.agent.md`
**Depends on:** `.github/copilot.instructions.md`, `.github/issues.md`

### Context
The repository has a strong issue-writing format, but vague user requests still
need to be translated into concrete backlog entries after inspecting the code,
existing issues, and project instructions. That triage step is distinct from
the scientific workflow agent: it should focus on intake, deduplication, and
turning fuzzy requests into implementation-ready issue entries.

The custom agent should review current context before drafting, avoid duplicate
issues when work is already tracked, and write new issues in the repository's
canonical format when a fresh task is warranted.

### Acceptance criteria
- [x] A user-invocable custom agent exists under `.github/agents/`
- [x] The agent description mentions vague requests, issue drafting, repo context, and `.github/issues.md`
- [x] The agent requires reading `.github/copilot.instructions.md` and `.github/issues.md` before drafting
- [x] The agent explicitly handles duplicate or overlapping issues instead of blindly appending new ones
- [x] The agent instructs itself to write actionable acceptance criteria in the repository's issue format

---

## Issue #18 — Add persistent drone-failure fault tolerance for FedAvg and gossip

**Labels:** `feature` `training` `evaluation`
**Files:** `project/src/train.py`, `project/src/federated_agent.py`, `project/src/main.py`, `project/src/evaluations.py`, `project/README.md`
**Depends on:** #2, #5, #6, #7, #8, #14

### Context
The repository already simulates **communication dropout** via `dropout_p` in
`CommunicationChannel`, but that is a transient per-round failure mode: the
drone misses a communication event and may participate again later. The user
now wants a separate experiment for **persistent drone failure**: a drone can
drop out entirely mid-run, stop training and communicating for the remainder of
the experiment, and be treated as permanently out of action.

This should apply to the decentralized methods only (`train_fedavg` in
`federated_agent.py` and `train_gossip` in `train.py`). Centralized training
remains the reference baseline and should not simulate per-drone failure.

Smallest reasonable assumption for this issue: introduce a per-round failure
rate applied independently to each currently active drone. Once a drone fails,
it remains inactive for the rest of the run and should no longer contribute to
local training, communication, or decentralized model aggregation/evaluation.

This issue overlaps conceptually with Issue #2, but it is not mergeable with
the existing communication-dropout feature because the state transition is
persistent and changes the training population rather than a single comm round.

### Required behavior
- Add a new CLI/configurable setting in `main.py` for persistent drone failure,
    with a default of `0.0` so existing runs remain unchanged when the feature is
    disabled. Use a name that is clearly distinct from communication dropout,
    such as `--drone-failure-rate`.
- Extend `train_fedavg` and `train_gossip` so failed drones are removed from
    future local-training and communication participation for the remainder of
    the run.
- Define evaluation behavior for failed drones explicitly. The simplest
    defensible behavior is that inactive drones no longer count toward the active
    decentralized model pool after failure.
- Ensure the training loops do not crash if some drones fail early, and handle
    the edge case where all decentralized drones fail before the final epoch.
- Report persistent-failure metrics separately from communication-dropout
    metrics in the console output and structured output artifacts.
- Update `project/README.md` to document the new failure mode, CLI argument,
    and how it differs from `dropout_p` communication dropout.

### Acceptance criteria
- [ ] `main.py` accepts a persistent drone-failure rate argument with default `0.0`, and argument validation distinguishes it from communication dropout
- [ ] With persistent failure disabled, FedAvg and gossip retain their prior behavior and still run successfully
- [ ] In both `train_fedavg` and `train_gossip`, a failed drone performs no further local training and is never selected for later communication rounds
- [ ] Decentralized evaluation handles permanent failures without crashing, including the case where all drones have failed
- [ ] Console output reports persistent-failure metrics per method, such as total failed drones and active drones remaining
- [ ] Output artifacts include a structured permanent-failure summary that is separate from, or clearly nested apart from, the existing communication interruption log
- [ ] A deterministic smoke test or seeded run demonstrates at least one permanent drone failure and confirms the run still completes
- [ ] `project/README.md` explains the new setting and explicitly distinguishes permanent drone failure from transient communication dropout

---

## Issue #19 — Fix malformed output-directory rooting in `main.py`

**Labels:** `bug` `integration` `documentation`
**Files:** `project/src/main.py`, `project/README.md`
**Depends on:** #8, #10

### Context
The current CLI output-directory handling in `main.py` is rooted against the
current working directory via `Path.cwd()`. That causes malformed output paths
when the command is launched from `project/src/`: the default `--output-dir`
value of `project/results` resolves to `sdsr-site/project/src/project/results`
instead of the intended base directory `sdsr-site/project/results`.

The user requirement is stricter than a generic relative-path fix. The base
output root should always be the repository's canonical results directory:
`sdsr-site/project/results/`. Any user-specified output target should be
treated as a folder name or nested subpath **under that base root**, not as an
arbitrary filesystem path. If the user provides a new folder name, the CLI
should create the missing directories automatically.

This overlaps with Issue #10 only in that both concern `main.py` integration,
but it is not mergeable into the earlier completed issue because the bug is a
newly identified path-resolution defect with tighter CLI semantics.

### Fix
- Replace `Path.cwd()`-based output resolution in `main.py` with a resolver
    rooted at the repository's canonical results base, derived from the source
    file location.
- Treat the `--output-dir` value as a folder name or nested relative path under
    `project/results/`, for example:
    - default behavior writes to `sdsr-site/project/results`
    - `--output-dir issue8_run1` writes to `sdsr-site/project/results/issue8_run1`
    - `--output-dir experiments/faults/run_a` writes to
        `sdsr-site/project/results/experiments/faults/run_a`
- Prevent path resolution from escaping the `project/results/` base via
    absolute paths or upward traversal.
- Keep downstream evaluation and log-writing code unchanged except for using
    the corrected resolved path.
- Update `project/README.md` so the CLI documentation matches the rooted output
    behavior.

### Acceptance criteria
- [ ] Default CLI output writes to `sdsr-site/project/results`, not `sdsr-site/project/src/project/results`
- [ ] A user-specified folder name such as `--output-dir test_run` resolves to `sdsr-site/project/results/test_run`
- [ ] A user-specified nested path such as `--output-dir experiments/test_run` resolves under `sdsr-site/project/results/experiments/test_run`
- [ ] Missing directories for the requested output target are created automatically
- [ ] Path resolution cannot escape the `sdsr-site/project/results` base
- [ ] `project/README.md` documents that `--output-dir` is rooted under `project/results/`

---

## Issue #20 — Replace binary wetland occupancy with fractional cell coverage ✓ DONE

**Labels:** `feature` `data` `modeling` `documentation`
**Files:** `project/src/build_graph.py`, `project/src/evaluations.py`, `project/README.md`
**Depends on:** #13

### Context
The current graph builder marks a grid cell as wetland with a hard binary rule:
if any wetland polygon intersects the cell, `wetland_presence[i] = 1.0`.
That is easy to compute, but it collapses small edge touches and near-full-cell
coverage into the same feature value.

The user now wants a modest realism improvement without changing the overall
experiment structure. The smallest coherent change is to replace the binary
wetland indicator with **fractional wetland coverage per grid cell** while
keeping the rest of the graph pipeline intact.

Repository-specific implication: changing only the node feature would leave the
ground-truth Dijkstra labels driven by the old binary threshold in
`compute_dijkstra_labels`, which weakens the value of the realism change. For
this issue, treat fractional coverage as the canonical per-cell wetland signal
for both `data.x` and traversal-cost construction.

Smallest reasonable assumption for ambiguous geometry cases: per-cell coverage
should be computed as the fraction of the cell area covered by the **union** of
intersecting wetland geometry clipped to that cell, so overlapping polygons do
not double-count and the resulting value stays in `[0, 1]`.

This is not covered by existing issues. Issue #13 established that `data.x`
must contain sufficient task information, but it did not change the wetland
channel from binary occupancy to fractional coverage.

### Required behavior
- Update `assign_wetland_features` in `project/src/build_graph.py` so it
    returns a float32 array of per-cell wetland coverage fractions in `[0, 1]`
    instead of `{0.0, 1.0}`.
- Keep the spatial-index candidate filtering, but compute the exact per-cell
    wetland fraction from clipped polygon geometry rather than a boolean
    `intersects(...).any()` test.
- Update `compute_dijkstra_labels` so node traversal cost scales with coverage
    instead of thresholding on `wetland_presence > 0`. The simplest defensible
    rule is linear interpolation between `land_cost` and `wetland_cost`, e.g.
    `land_cost + coverage * (wetland_cost - land_cost)`.
- Preserve the existing `data.x` feature layout and tensor shape, with the
    first channel now representing fractional wetland coverage rather than binary
    occupancy.
- Update the status output in `build_pyg_data` so the printed summary remains
    meaningful under fractional features. Do not keep reporting a raw "wetland
    cells" count as though the feature were still binary.
- Keep greedy-path terrain-cost evaluation aligned with the new fractional
    coverage semantics so downstream path metrics use the same terrain model as
    the Dijkstra labels.
- Update `project/README.md` to document the new feature semantics and how
    traversal costs now relate to fractional coverage.

### Acceptance criteria
- [x] `assign_wetland_features` returns a float32 array with one value per node, and every value is in `[0, 1]`
- [x] A cell with no intersecting wetland geometry gets `0.0`, and a fully covered cell gets `1.0`
- [x] Overlapping wetland polygons within a cell do not produce values greater than `1.0`
- [x] `compute_dijkstra_labels` uses fractional coverage directly when constructing node traversal costs instead of thresholding with `wetland_presence > 0`
- [x] `build_pyg_data` still produces a valid PyG `Data` object with the same feature dimension expected by downstream modules
- [x] Console/status output in `build_pyg_data` reflects fractional coverage semantics rather than binary occupied-cell counting
- [x] Greedy-path evaluation uses coverage-scaled traversal cost rather than a binary wetland threshold
- [x] `project/README.md` explains that the first node feature is fractional wetland coverage and that Dijkstra costs scale with coverage

---

## Issue #21 — Integrate `SpatialGridSimulator` into decentralized local training and proximity-limited communication

**Labels:** `feature` `simulation` `training` `documentation`
**Files:** `project/src/grid_sim.py`, `project/src/comms.py`, `project/src/train.py`, `project/src/federated_agent.py`, `project/src/main.py`, `project/README.md`
**Depends on:** #2, #3, #5, #6, #8, #11, #14

### Context
The repository currently constructs a `SpatialGridSimulator` in `main.py`, but
the experimental pipeline still trains decentralized methods on static,
precomputed subgraphs and uses communication scheduling that ignores drone
positions. Concretely:
- `train_gossip` precomputes `local_subgraphs = [build_local_subgraph(...)]`
    once and reuses them for the full run.
- `train_fedavg` precomputes `local_partitions` once and calls
    `NodeAgent.train_local(..., use_full_graph=True)`, so the agent-level local
    subset logic and simulator position are bypassed.
- `CommunicationChannel.sample_participants` and `gossip_pairs` operate on the
    full drone-ID list without any spatial eligibility filter.

The user now wants the simulator to matter operationally, not just for initial
position reporting: each decentralized drone should train on a local view taken
from its current position, and communication eligibility should be constrained
by drone proximity in addition to the existing global dropout schedule.

This is not covered by existing issues. Issue #3 implemented the simulator
itself, but did not wire it into training or communication. Issues #2, #5, #6,
and #8 established the current static-partition and global-channel behavior;
this issue changes those semantics and must be tracked separately.

Smallest reasonable assumptions for the current codebase:
- Centralized training remains unchanged and does not use the simulator.
- Simulator-driven behavior is optional and disabled by default so existing
    seeded runs remain comparable.
- In simulator-driven mode, each decentralized drone advances at most one grid
    step per training round via `SpatialGridSimulator.step_drones()` before
    selecting its current training view.
- The local training graph for a drone is `simulator.get_local_view(drone_id,
    radius=view_radius)` rather than a one-time partition subgraph.
- Proximity is measured in grid hops / Manhattan distance between current drone
    positions.
- For gossip, only drones that are both dropout survivors and within
    `comm_radius` of each other may exchange weights.
- For FedAvg, a concrete server-reachability rule must be documented. The
    smallest defensible rule is to treat the central coordinator as a fixed
    base-station node at the grid centroid; only drones within `comm_radius` of
    that base station may uplink on a communication round.
- Spatial ineligibility is a topology constraint, not a dropout event. The
    interruption logger should continue to represent stochastic communication
    dropout, not lack of proximity.

### Required behavior
- Extend `main.py` to expose and validate simulator-integration settings for
    decentralized runs, such as a local-view radius, communication radius, and
    whether drones step each round before training/communication.
- Thread a shared `SpatialGridSimulator` instance into `train_fedavg` and
    `train_gossip` when simulator-driven mode is enabled.
- Update `train_gossip` so each drone refreshes its local training graph from
    the simulator each round instead of using a fixed precomputed subgraph for
    the entire run.
- Update `train_fedavg` so each node trains on a simulator-derived local view
    rather than the current fixed partition/full-graph bypass path.
- Extend `grid_sim.py` and/or `comms.py` with a clear API for proximity
    filtering based on current positions. The implementation may add helper
    methods such as pairwise communication-neighbour queries or base-station
    reachability helpers, but the final API must keep the training loops easy to
    audit.
- Apply stochastic dropout only after establishing the topology-eligible set
    for the current communication round, and keep dropout logging semantics
    explicit and consistent.
- Preserve existing behavior when simulator-driven mode is disabled.
- Update `project/README.md` so the experiment description matches the new
    simulator-driven local-data and communication-topology behavior.

### Acceptance criteria
- [x] With simulator integration disabled, `train_fedavg`, `train_gossip`, and `main.py` retain their current fixed-partition / global-communication behavior and still run successfully
- [x] `main.py` accepts documented simulator-integration arguments for decentralized runs, with defaults that preserve existing behavior
- [x] In simulator-driven mode, each decentralized drone's training data is refreshed from `SpatialGridSimulator.get_local_view(...)` each round instead of being fixed once at startup
- [x] Drone motion, when enabled, occurs at a documented cadence and keeps each drone inside its partition
- [x] Gossip communication rounds only form pairs that satisfy both the dropout filter and the configured proximity rule
- [x] FedAvg communication rounds only aggregate updates from drones that satisfy both the dropout filter and the documented server-reachability rule
- [x] The communication logger does not misclassify out-of-range drones as dropout or blackout events
- [ ] A seeded end-to-end run with simulator integration enabled completes without runtime errors and shows position-dependent decentralized participation over time
- [x] `project/README.md` documents the simulator-driven training mode, proximity-based communication semantics, and any new CLI arguments

Validation note: seeded synthetic in-memory validations for both the legacy and
simulator-driven decentralized paths passed after implementation. A full
real-data CLI run with simulator integration is still blocked in this
environment by an upstream GeoPackage/Shapely allocation failure during
`load_gdf()`, before the new simulator-driven training code executes.

---

## Issue #22 — Add periodic experiment-state snapshot export during training ✓ DONE

**Labels:** `feature` `simulation` `documentation`
**Files:** `project/src/main.py`, `project/src/train.py`, `project/src/federated_agent.py`, `project/src/grid_sim.py`, `project/README.md`
**Depends on:** #21

### Context
The repository already has the pieces needed for static simulator visuals: `main.py`
constructs `SpatialGridSimulator` instances for FedAvg and gossip when
`--use-simulator-integration` is enabled, and `grid_sim.py` already exposes
`SpatialGridSimulator.visualize(...)` to write PNG frames of drone positions on
top of the wetland grid. However, the end-to-end experiment never calls that
visualization path during training, so a user cannot inspect how the spatial
state evolves over time.

The new request is for visual representations of the experiment, for example a
state image every 5 epochs. This is not covered by Issue #21: that issue makes
simulator state affect decentralized training and communication, but it does not
persist periodic state artifacts for later inspection.

Smallest reasonable assumptions for this issue:
- Standardize on PNG output rather than adding both JPG and PNG.
- Scope the snapshots to simulator-driven decentralized methods (FedAvg and
    gossip), since centralized training has no per-drone spatial state to render.
- Make the feature optional and disabled by default so existing runs and output
    volume remain unchanged unless explicitly requested.

### Required behavior
- Add a CLI/configurable snapshot cadence in `main.py`, such as
    `--snapshot-every`, with `0` meaning disabled.
- When snapshot export is enabled together with `--use-simulator-integration`,
    save experiment-state PNGs for both FedAvg and gossip at the requested epoch
    cadence during the real training loops rather than only in a standalone smoke
    test.
- Save snapshots into method-specific subdirectories under the selected output
    directory so the two decentralized runs do not overwrite one another.
- Include enough identifying information in the saved artifact naming and/or
    figure title to tell which method and epoch each image corresponds to.
- Define the cadence precisely. The smallest defensible rule is: save the
    initial state at epoch 0, then save again after every `N` training rounds,
    and also save the final state if the run ends off-cadence.
- Make the rendered state informative even when drone motion is disabled. The
    smallest defensible behavior is to render the current decentralized model's
    full-grid prediction field with drone positions overlaid, rather than a
    static wetland background alone.
- Keep behavior explicit when snapshot export is requested without simulator
    integration. The simplest acceptable behavior is to reject that argument
    combination in `validate_args(...)` with a clear error.
- Update `project/README.md` so the new CLI option and output artifact layout
    are documented.

### Acceptance criteria
- [x] `main.py` accepts a documented snapshot-cadence argument whose default leaves current runs unchanged
- [x] Requesting snapshots without `--use-simulator-integration` fails fast with a clear validation error, or an equally explicit documented behavior is implemented
- [x] FedAvg training writes PNG state snapshots into a dedicated method-specific output directory at the configured cadence
- [x] Gossip training writes PNG state snapshots into a dedicated method-specific output directory at the configured cadence
- [x] Snapshot filenames and/or figure titles identify the method and epoch unambiguously
- [x] Snapshot renders reflect evolving decentralized model state even when drone positions remain fixed
- [x] A seeded smoke test or small end-to-end run with `--snapshot-every 5` produces multiple non-empty PNG files without interrupting training
- [x] `project/README.md` documents the new option and where the snapshot images are written

---

## Issue #23 — Establish v2 parallel workflow boundary and namespace ✓ DONE

**Labels:** `workflow` `documentation`
**Files:** `project/src/v2/__init__.py`, `project/README.md`
**Depends on:** none

### Context
The current repository is explicitly a v1 benchmark centered on goal-conditioned
shortest-path regression over a single graph built from the wetlands dataset.
The user now wants a **parallel v2 workflow** for decentralized mosquito-risk
mapping that should reuse generic infrastructure where possible, but must not
overwrite or repurpose the existing v1 code path.

The v2 design is materially different from v1:
- v1 target: Dijkstra distance to a selected goal node
- v2 target: hidden continuous mosquito-risk field over sampled real wetland
    windows
- v1 local training: partition-oriented / simulator-local graph views
- v2 local training: free-moving drones observing local vision bubbles on the
    full sampled map

This issue exists to define the repository boundary before any v2 benchmark
logic is added. Without that boundary, later issues risk mutating v1 code in
place and making the two experiment families hard to compare or maintain.

Smallest reasonable assumptions for this issue:
- v1 remains the canonical shortest-path benchmark and is not removed,
    renamed, or behaviorally changed.
- New benchmark-specific code lives under a dedicated `project/src/v2/`
    namespace rather than modifying existing source modules in place.
- Existing generic modules may be imported directly from v2 when their
    semantics still apply, especially `load_data.py` for GeoPackage access and
    `comms.py` for communication scheduling / dropout logging.
- v2 outputs and cached task artifacts must live under v2-specific roots so
    they do not collide with v1 results.

### Required behavior
- Create the v2 source namespace under `project/src/v2/` with a minimal
    package marker and a short package-level docstring describing the v2
    benchmark scope.
- Update `project/README.md` so the repository documents two parallel
    workflows:
    - v1 shortest-path regression (existing)
    - v2 mosquito-risk mapping (new, under active implementation)
- Document the reuse boundary explicitly: v2 may import generic utilities from
    `load_data.py` and `comms.py`, but benchmark-specific data generation,
    simulation, training, evaluation, and CLI orchestration belong in new v2
    modules.
- Define the default v2 directory conventions in the README so downstream
    issues have a stable target for new files and artifacts.

### Acceptance criteria
- [x] `project/src/v2/` exists as a documented parallel namespace for new benchmark code
- [x] `project/README.md` distinguishes the existing v1 shortest-path workflow from the new v2 mosquito-risk workflow
- [x] The README states that v1 source files remain supported and are not the implementation target for new v2 benchmark logic
- [x] The README documents that v2 should reuse `load_data.py` and `comms.py` where possible instead of duplicating them
- [x] The README defines non-conflicting v2 output and cache roots so later issues can rely on them

---

## Issue #24 — Build sampled wetland-window tasks and immutable v2 task cache

**Labels:** `feature` `data` `workflow`
**Files:** `project/src/v2/task_sampling.py`, `project/src/v2/task_cache.py`, `project/README.md`
**Depends on:** #23

### Context
The current pipeline grids the same statewide GeoPackage extent for each run.
That is appropriate for the v1 single-graph benchmark, but it is too rigid for
the v2 mosquito-risk mapping workflow, where the user wants many sampled maps
derived from the real wetlands data instead of one repeatedly reused
"mega-map."

The v2 benchmark should therefore separate **task generation** from **model
training**. A task is a frozen sampled map window plus derived graph-ready base
covariates. Multiple training runs may reuse the same task while varying drone
starting positions, vision range, communication, movement policy, and compute
constraints.

This issue covers only the real-data map-window sampling and cache layer. It
does not yet define the hidden mosquito-risk field itself.

Smallest reasonable assumptions for this issue:
- Use the existing `load_data.load_gdf()` entry point rather than duplicating
    GeoPackage-loading logic.
- Sample axis-aligned spatial windows from the statewide wetlands dataset.
- Reject or resample windows that are effectively empty after clipping, so the
    v2 task library does not fill with trivial no-wetland graphs.
- Treat the cached task artifact as immutable once written; training code may
    read it but must not modify it in place.
- The cache key must be reproducible from the task-generation specification
    rather than derived from wall-clock timestamps alone.

### Required behavior
- Implement a v2 task sampler that:
    - loads the statewide GeoDataFrame via `load_gdf()`
    - samples a bounded spatial window reproducibly from a seed / spec
    - clips the wetland geometries to that window
    - records enough metadata to reconstruct the same window later
- Implement a v2 task cache layer that persists each sampled task artifact and
    a manifest entry describing:
    - task identifier
    - source dataset / layer metadata
    - sampling seed and sampling parameters
    - window bounds
    - basic wetland summary statistics
- Ensure repeated requests for the same task specification resolve to the same
    cached artifact rather than silently regenerating a different one.
- Expose a lightweight standalone entry point in `project/src/v2/task_sampling.py`
    so users can resolve or create cached tasks outside the core training /
    experiment loop.
- Document the task-library concept and cache behavior in `project/README.md`.

### Acceptance criteria
- [x] A fixed task-generation seed and spec produce the same sampled wetland window across repeated runs
- [x] Sampled tasks are persisted under the v2 cache root defined in #23 and do not overwrite v1 outputs
- [x] Each cached task includes a manifest or metadata record with the window bounds and generation parameters needed for exact reproduction
- [x] The sampler rejects or resamples trivially empty windows according to a documented minimum-content rule
- [x] Re-requesting an existing task spec returns the same cached task identifier instead of generating a second logically identical task
- [x] `project/src/v2/task_sampling.py` can be invoked directly to create or reuse a cached task artifact outside the main experiment loop
- [x] `project/README.md` explains the difference between cached task generation and later model-training runs

---

## Issue #25 — Generate hidden continuous mosquito-risk fields for cached v2 tasks

**Labels:** `feature` `data` `evaluation`
**Files:** `project/src/v2/risk_field.py`, `project/src/v2/task_cache.py`, `project/README.md`
**Depends on:** #24

### Context
The v2 benchmark replaces the v1 Dijkstra-distance label with a hidden
continuous mosquito-risk field layered on top of each sampled real wetland
window. The user wants that field to be grounded in real wetland structure,
biologically suggestive where possible, and stochastic enough that it is not a
trivial re-encoding of the visible wetland-coverage feature.

The currently agreed v2 label components are:
- wetland-perimeter hotspot seeds with moderate probability rather than an
    overly strong deterministic boundary rule
- local risk bumps that decay over nearby cells rather than hard binary seed
    labels alone
- varying bump amplitudes across seeds
- a small heterogeneous background field
- additionally, (low) internal-marsh hotspot probability using indirect wetland-attribute
    proxies when available, specifically `COW_CLASS1 == "EM1"` and
    `SPCC_DESC == "Shallow Marsh"`

This issue covers the hidden risk-field generator only. It does not yet define
the v2 graph features or model architecture.

Smallest reasonable assumptions for this issue:
- The canonical v2 stored target is continuous risk, not binary class.
- Thresholded hotspot labels may be derived secondarily for evaluation but are
    not the primary stored benchmark target.
- The risk generator may use cached task geometry / covariates plus raw
    wetland-attribute proxies available in the sampled task metadata, but it
    should not depend on labels observed during training.
- If the proxy fields are absent or missing for a sampled task, the generator
    should fall back to a documented geometry-only behavior instead of failing.

### Required behavior
- Implement a v2 risk-field generator that operates on one cached sampled task
    and produces a continuous per-node mosquito-risk field using the agreed
    ingredients:
    - probabilistic wetland-perimeter seeds
    - spatially decaying local bumps around active seeds
    - heterogeneous bump amplitudes
    - low-amplitude background variation
    - elevated internal-marsh seed probability based on `EM1` and
        `Shallow Marsh` proxies when available
- Persist the generated field and all generator parameters into the cached task
    artifact or an associated immutable derivative artifact.
- Expose a deterministic interface so a fixed sampled task plus fixed risk
    generation seed reproduces the same hidden field exactly.
- Optionally derive a thresholded hotspot label or rank-based hotspot mask for
    later evaluation, but keep the continuous field as the canonical source.
- Document the generator inputs, fallback behavior, and stored outputs in the
    README.

### Acceptance criteria
- [ ] A fixed cached task and fixed risk-generation seed reproduce the same continuous risk field exactly
- [ ] Changing the risk-generation seed changes the field while leaving the sampled map window unchanged
- [ ] The stored v2 task artifact records the risk-generator parameters and whether the EM1 / Shallow Marsh proxy path was used
- [ ] The generated field includes nontrivial spatial variation beyond raw wetland coverage alone
- [ ] Missing proxy attributes fall back to a documented behavior instead of causing task generation to fail
- [ ] Any thresholded hotspot label is documented as a derived evaluation artifact rather than the canonical stored target
- [ ] `project/README.md` documents the continuous-field design and the use of wetland perimeter and marsh-proxy factors

---

## Issue #26 — Create v2 graph builder and risk-prediction model without goal-conditioned features

**Labels:** `feature` `data` `training`
**Files:** `project/src/v2/build_graph_v2.py`, `project/src/v2/model_v2.py`, `project/README.md`
**Depends on:** #24, #25

### Context
The current graph builder in `build_graph.py` is tightly coupled to the v1
task: it creates one graph over a wetland grid, attaches goal-conditioned node
features, and stores Dijkstra-distance labels. None of those semantics carry
forward to the v2 mosquito-risk benchmark.

v2 therefore needs a new graph/data contract that is compatible with PyTorch
Geometric training while remaining separate from the v1 builder. The user has
requested a parallel v2 workflow rather than in-place replacement.

Smallest reasonable assumptions for this issue:
- Reuse generic grid-graph ideas from the existing builder where still valid,
    but implement them in new v2 files rather than editing `build_graph.py`.
- Keep the node-level scalar prediction shape compatible with the existing
    simple GCN pattern where practical, but interpret it as continuous
    mosquito-risk rather than distance-to-goal.
- Include spatial-position features in v2.
- Include goal-specific fields only in v1; v2 must not carry `goal_node`,
    `goal_pos`, or goal-relative delta features.

### Required behavior
- Implement a v2 graph builder that transforms one cached sampled task into a
    PyG `Data` object for risk prediction.
- Define and document the v2 node feature set. The smallest defensible feature
    set includes:
    - wetland coverage
    - normalized spatial position
    - one or more wetland-structure / proxy channels useful for v2 risk
        reconstruction, such as perimeter proximity or marsh-proxy indicators
- Store the continuous risk field as the target `y` for the v2 graph.
- Implement a v2 risk-prediction GNN model in a new file that keeps the simple
    scalar node-level output contract but is semantically distinct from the v1
    distance-regression model.
- Update the README so the v2 graph contract and model target are documented.

### Acceptance criteria
- [ ] `build_graph_v2.py` builds a valid PyG `Data` object from a cached v2 task without relying on goal-node concepts
- [ ] The v2 `Data` object stores the continuous mosquito-risk field as the node-level target
- [ ] The v2 feature tensor includes documented spatial and wetland-derived channels beyond a single raw wetland-coverage scalar
- [ ] `model_v2.py` exposes a node-level risk model whose output shape is compatible with centralized and decentralized training
- [ ] No goal-conditioned fields from v1 are required by the v2 graph builder or model
- [ ] `project/README.md` documents the v2 feature contract and risk target semantics

---

## Issue #27 — Implement free-movement v2 drone simulator without fixed partitions

**Labels:** `feature` `simulation`
**Files:** `project/src/v2/grid_sim_v2.py`, `project/README.md`
**Depends on:** #24, #26

### Context
The current `SpatialGridSimulator` is designed for the v1 workflow: drones are
associated with partitions and, when movement is enabled, step only within
their own partition. That matches the v1 decentralized setup but does not fit
the v2 benchmark, where drones are intended to explore the sampled map freely
and gather local risk observations from anywhere on the task graph.

This issue adds a v2 simulator rather than modifying the existing one in place.

Smallest reasonable assumptions for this issue:
- Drones move on the same 4-neighbour grid adjacency used elsewhere in the
    repository.
- Drones are initialized at reproducible random map positions rather than
    partition centroids.
- A configurable minimum-separation rule between starting positions is useful
    when feasible, but a documented best-effort fallback is acceptable when the
    map is too small.
- The simulator should preserve the ability to export snapshots for debugging
    and evaluation.

### Required behavior
- Implement a v2 simulator that:
    - initializes drone positions randomly and reproducibly on the full sampled
        map
    - supports movement anywhere on the map via 4-neighbour adjacency
    - returns per-drone local observation bubbles based on a configured vision
        radius
    - can render the current task state and drone positions to PNG
- Keep the simulator interface easy to audit from the training loops, similar
    in spirit to the existing simulator but without partition semantics.
- Document the simulator’s initialization and movement assumptions in the
    README.

### Acceptance criteria
- [ ] `grid_sim_v2.py` initializes drones at reproducible random positions on the full map rather than partition centroids
- [ ] Drones may move across the full sampled map and are not restricted by v1 partition ownership
- [ ] The simulator exposes a local-view API based on vision radius for downstream training loops
- [ ] Snapshot or visualization output renders the sampled map and current drone positions without relying on v1 partition metadata
- [ ] `project/README.md` documents the v2 simulator’s free-movement semantics and initialization behavior

---

## Issue #28 — Add configurable v2 movement policies for exploration and mapping

**Labels:** `feature` `simulation` `training`
**Files:** `project/src/v2/movement_policy.py`, `project/src/v2/grid_sim_v2.py`, `project/README.md`
**Depends on:** #27

### Context
Pure random movement is easy to implement but too weak to carry the scientific
story of the v2 mapping benchmark on its own. The user wants drone movement to
be informed by what each drone can currently observe and what parts of the map
remain informative to sample.

This issue introduces a policy abstraction so v2 experiments can compare random
motion against at least one informed exploration strategy without baking one
hard-coded movement rule into the simulator.

Smallest reasonable assumptions for this issue:
- Keep a random-walk policy as a baseline.
- Add at least one non-random policy that uses current mapping state such as
    uncertainty, novelty, or low-observation coverage.
- The simulator remains responsible for applying valid moves; the policy only
    chooses among candidate moves.

### Required behavior
- Implement a v2 movement-policy interface in a new module.
- Provide at least two concrete policies:
    - random baseline
    - one informed exploration policy using current model or coverage state
- Thread the selected policy through the v2 simulator or training loop in a
    way that is deterministic under fixed seeds.
- Document the available policy choices and their intended behavior in the
    README.

### Acceptance criteria
- [ ] The v2 movement-policy API supports selecting among at least two named policies without editing simulator internals
- [ ] A random baseline policy is implemented and reproducible under a fixed seed
- [ ] At least one informed exploration policy uses current state beyond pure randomness when choosing the next move
- [ ] Under the same cached task and random seed, different policies can produce different trajectories
- [ ] `project/README.md` documents the available movement policies and what benchmark question they are meant to test

---

## Issue #29 — Implement v2 centralized, FedAvg, and gossip training over local observation bubbles

**Labels:** `feature` `training`
**Files:** `project/src/v2/train_v2.py`, `project/src/v2/federated_v2.py`, `project/README.md`
**Depends on:** #26, #27, #28

### Context
The v1 training code is tied to the shortest-path benchmark and, for
decentralized methods, to either partition-based local subgraphs or the v1
simulator integration. The v2 benchmark needs parallel training loops that use
the new cached tasks, the v2 graph contract, and free-moving drone observation
bubbles.

The user wants maximum reuse where sensible, but not by overwriting existing
benchmark-specific code. In particular, `comms.py` should still be reused where
its scheduling, dropout, and logging semantics remain applicable.

Smallest reasonable assumptions for this issue:
- Centralized training remains the full-task reference baseline.
- FedAvg and gossip should both train only on nodes currently visible inside
    each drone’s observation bubble.
- Existing communication-channel semantics from `comms.py` may be imported
    directly instead of rewritten.
- Existing compute-budget knobs (`num_threads`, `time_budget_ms`) should remain
    available in v2 where practical.

### Required behavior
- Implement v2 centralized training against the full cached-task graph.
- Implement v2 FedAvg and gossip training loops that:
    - read per-drone local data from the v2 simulator
    - reuse `CommunicationChannel` for communication timing and dropout where
        appropriate
    - keep communication logs method-specific
    - start from matched initialized model states for fair comparison
- Keep the v2 training loops isolated from v1 modules except for imported
    generic helpers that remain semantically valid.
- Update the README to explain how v2 local observation bubbles drive
    decentralized training.

### Acceptance criteria
- [ ] `train_v2.py` exposes a centralized reference training path for the v2 risk-prediction task
- [ ] V2 FedAvg trains from per-drone local observation bubbles instead of v1 partitions or goal-based local subsets
- [ ] V2 gossip trains from per-drone local observation bubbles instead of v1 partitions
- [ ] V2 decentralized training reuses `CommunicationChannel` from `comms.py` rather than duplicating its schedule / dropout logic
- [ ] V2 training preserves configurable compute-constraint knobs or explicitly documents any justified change in those semantics
- [ ] Centralized, FedAvg, and gossip all start from matched initial states on the same cached task for fair comparison
- [ ] `project/README.md` documents the v2 training setup and its reuse boundary with existing v1 infrastructure

---

## Issue #30 — Add v2 risk-map reconstruction evaluation and reporting

**Labels:** `feature` `evaluation` `documentation`
**Files:** `project/src/v2/evaluations_v2.py`, `project/README.md`
**Depends on:** #29

### Context
The existing evaluation module is built around the v1 shortest-path benchmark:
MSE against Dijkstra labels plus greedy-path behavior. Those outputs are not
meaningful for a v2 mosquito-risk mapping benchmark.

v2 needs its own evaluation path that measures reconstruction quality for the
continuous risk field and, optionally, hotspot-detection quality for derived
binary labels or top-risk sets.

Smallest reasonable assumptions for this issue:
- Continuous-field reconstruction remains the primary evaluation target.
- Binary hotspot metrics may be derived secondarily from thresholded labels or
    rank-based hotspot sets.
- Snapshot and map visualizations remain useful, but they should visualize
    predicted and true risk rather than greedy paths or goal-seeking behavior.

### Required behavior
- Implement a v2 evaluation module that reports full-map reconstruction quality
    for centralized, FedAvg, and gossip.
- Add at least one threshold- or ranking-based hotspot-detection summary in
    addition to continuous-risk error reporting.
- Save v2 plots and summary artifacts under v2-specific names and directories so
    they do not collide with v1 outputs.
- Document the meaning of the v2 metrics and plots in the README.

### Acceptance criteria
- [ ] `evaluations_v2.py` reports continuous-field reconstruction metrics for centralized, FedAvg, and gossip on the same cached task
- [ ] V2 evaluation includes at least one hotspot-detection summary derived from the continuous risk field
- [ ] No greedy-path or goal-reaching metrics are required by the v2 evaluation path
- [ ] V2 evaluation artifacts are written under v2-specific names or directories and do not overwrite v1 plots
- [ ] `project/README.md` documents the v2 metrics and explains how they differ from the v1 pathfinding-oriented outputs

---

## Issue #31 — Implement v2 CLI entry point and task-library workflow

**Labels:** `feature` `integration` `workflow` `documentation`
**Files:** `project/src/main_v2.py`, `project/README.md`
**Depends on:** #23, #24, #25, #26, #27, #28, #29, #30

### Context
Once the v2 task cache, latent risk generator, simulator, training loops, and
evaluation path exist, the repository still needs an end-to-end entry point for
running v2 experiments without touching the v1 `main.py` workflow.

The user has also explicitly requested the ability to reuse precomputed sampled
maps and hidden continuous hotspot fields across multiple training runs rather
than regenerating them each time.

This issue therefore closes the loop for the v2 benchmark by adding a parallel
CLI and task-library workflow.

Smallest reasonable assumptions for this issue:
- `main_v2.py` is a separate entry point; it does not replace or rename the
    existing `main.py`.
- The CLI should support both task generation / caching and experiment
    execution against an existing cached task.
- V2 outputs, logs, and snapshots should resolve under the v2 results root
    defined in #23.

### Required behavior
- Implement a v2 CLI entry point that can:
    - build or extend a cached task library
    - select an existing cached task by identifier
    - run centralized, FedAvg, and gossip on that cached task
    - persist v2 evaluation outputs and logs without colliding with v1
- Keep the v1 `main.py` behavior unchanged.
- Update the README with a v2 quick-start path covering both task generation
    and experiment execution.

### Acceptance criteria
- [ ] `main_v2.py` exists as a documented parallel CLI entry point and does not replace the existing v1 `main.py`
- [ ] The v2 CLI can generate at least one cached task and later rerun experiments on that same task without rebuilding it
- [ ] The v2 CLI can run centralized, FedAvg, and gossip against a selected cached task identifier
- [ ] V2 logs, plots, and snapshots are written under the v2 results root and do not overwrite v1 outputs
- [ ] `project/README.md` includes a v2 quick start covering task-library generation and experiment execution

---

## Issue #32 — Add actor-judge custom-agent workflow for issue-driven implementation ✓ DONE

**Labels:** `feature` `workflow` `documentation`
**Files:** `.github/agents/sdsr-coding-actor.agent.md`, `.github/agents/sdsr-code-judge.agent.md`, `.github/agents/sdsr-issue-intake.agent.md`
**Depends on:** `.github/copilot.instructions.md`, `.github/issues.md`, #16, #17

### Context
The repository already has a scientific-workflow agent and an issue-intake
agent, but the user no longer wants implementation to be routed primarily
through the scientific workflow prompt. The practical reason is repository
specific: dependency-heavy runtime testing through agent environments is often
not available, so an implementation agent that insists on local execution tends
to fail noisily rather than helpfully.

The requested replacement is an **actor-judge workflow**:
- a coding actor that takes concrete tasks from `.github/issues.md` and
    implements them according to the issue contract and repository goals
- a code judge that reviews the actor's code for correctness and completeness
    by inspection

This change should augment the current custom-agent setup without deleting the
existing scientific-workflow agent. The critical workflow change is that new
issue implementation should default to the actor first, with the judge used for
review, rather than treating execution-heavy scientific workflow as the default
handoff path.

Smallest reasonable assumptions for this issue:
- The coding actor should avoid local runtime execution and heavy dependency
    tests by design.
- The judge should be review-oriented and should not depend on runtime
    execution either.
- The issue-intake agent should hand implementation work to the actor, not the
    scientific-workflow agent.
- The scientific-workflow agent may remain available for investigations and
    experiments, but it is no longer the default implementation handoff.

### Acceptance criteria
- [x] A new user-invocable coding-actor agent exists under `.github/agents/`
- [x] The coding actor is instructed to implement scoped issues from `.github/issues.md` without relying on local runtime execution
- [x] A new user-invocable code-judge agent exists under `.github/agents/`
- [x] The code judge is instructed to review correctness and completeness by code inspection, with explicit handling of unverified runtime-dependent items
- [x] The issue-intake agent hands implementation work to `SDSR Coding Actor` instead of defaulting to `SDSR Scientific Workflow`
- [x] The new agent prompts preserve repository-specific concerns such as issue-driven implementation, reuse boundaries, and documentation discipline

---

## Issue #33 — Add detailed cached-task attribute renderer for v2 debug inspection ✓ DONE

**Labels:** `feature` `workflow` `documentation`
**Files:** `project/src/v2/task_debug_render.py`, `project/README.md`
**Depends on:** #24

### Context
The current v2 sampler debug image in `project/src/v2/task_sampling.py` is a
useful coarse check that a sampled window lands in roughly the right wetland
area, but it is not detailed enough for inspecting the actual composition of a
cached task. The user now wants a separate debug program that takes an already
created cached task and renders a more detailed local image showing wetland
boundaries and attribute distributions such as `COW_CLASS1` and `SPCC_DESC`.

This is materially different from the existing sampler debug view:
- the current view focuses on sampled-window placement and rough geometry
- the requested view focuses on within-window feature composition for one
    cached task
- the new renderer should read an existing cached task artifact rather than
    resampling a new task

This issue is intentionally scoped as a standalone debug/inspection tool rather
than part of the training or evaluation loop. It should help the user verify
that cached v2 tasks contain meaningful local wetland structure before later
issues build risk fields and graph features on top of them.

Smallest reasonable assumptions for this issue:
- Reuse `load_cached_task(...)` from `project/src/v2/task_cache.py` instead of
    duplicating task-loading logic.
- The cached `window.geojson` should already preserve wetland attributes stored
    in the sampled task; the renderer should visualize whichever relevant
    columns are present without mutating the cached artifact.
- Because cached tasks are intended to be immutable, rendered debug outputs
    should default to a v2 results/debug path rather than writing inside the
    cached task directory.
- Attribute names may appear in lowercase GeoPandas form such as
    `cow_class1` / `spcc_desc`; the renderer should handle the documented label
    intent even if source column casing differs.

### Required behavior
- Implement a standalone v2 debug renderer in a new file that can:
    - load one cached task by identifier or explicit task directory
    - render a detailed local geometry view for the sampled task
    - render wetland boundaries clearly enough to inspect feature shape and
        density within the sampled window
    - render categorical attribute views for at least:
        - `COW_CLASS1`
        - `SPCC_DESC`
- Save the rendered outputs under a non-cache debug/results location such as a
    task-specific subdirectory under `project/results/v2/`, so the immutable
    task cache is only read, not modified.
- Provide a small CLI entry point for generating the detailed render outputs
    outside the core experiment loop.
- Document the renderer's purpose, inputs, and output location in the README.

### Acceptance criteria
- [x] `project/src/v2/task_debug_render.py` exists as a standalone debug program for cached v2 tasks
- [x] The renderer loads an existing cached task via the v2 cache layer instead of resampling data
- [x] The renderer outputs at least one detailed boundary-focused image of the sampled wetland geometries
- [x] The renderer outputs categorical debug views for `COW_CLASS1` and `SPCC_DESC`, or a documented fallback when one of those attributes is absent in the cached task
- [x] Rendered debug artifacts are written outside the immutable cached task directory and do not overwrite `window.geojson` or `metadata.json`
- [x] The CLI accepts a task identifier or equivalent explicit task locator and can generate detailed debug renders without entering the training loop
- [x] `project/README.md` documents how to run the detailed cached-task renderer and where its outputs are written
