# Project: Networked Edge Training for Spatial Graph Learning
## Group 2 — Noah Constable, Joe Conroy

### Purpose
Train a network of simulated edge devices (drones) using three strategies — centralized, FedAvg, and gossip — then empirically compare them on resource dimensions such as network connectivity and compute.

The graph-learning target is goal-conditioned shortest-path regression: every node predicts its Dijkstra distance to a selected goal node using fractional wetland coverage, node position, and goal-relative features.

This repository now supports two parallel workflows:
- `v1` is the existing shortest-path regression benchmark implemented by the
    current top-level modules in `project/src/`.
- `v2` is a separate mosquito-risk mapping benchmark under active
    implementation in `project/src/v2/`.

The v1 source files remain supported and are not the implementation target for
new v2 benchmark-specific logic.

### Workflow Boundary

| Workflow | Status | Scope | Code location |
|---|---|---|---|
| `v1` | Active | Goal-conditioned shortest-path regression on the wetland grid | `project/src/*.py` |
| `v2` | In progress | Sampled wetland windows, hidden mosquito-risk fields, free-moving drones, and risk-map reconstruction | `project/src/v2/` |

The repository reuse boundary is explicit:
- `v2` should reuse generic utilities from `load_data.py` and `comms.py`
    where their semantics still fit.
- New v2 benchmark-specific data generation, simulation, training,
    evaluation, and CLI orchestration belong in `project/src/v2/` rather than
    overwriting v1 files.
- `main.py`, `build_graph.py`, `train.py`, `federated_agent.py`,
    `grid_sim.py`, and `evaluations.py` remain the v1 workflow unless a future
    issue says otherwise.

---

### Resources

One of the datasets used is the [Minnesota National Wetlands Inventory Update](https://www.dnr.state.mn.us/wetlands/nwi_proj.html) GeoPackage, which is wrangled into a 2D grid graph via GeoPandas.

A CUDA-enabled GPU is recommended for performance but **not required**. All scripts run on a CPU-only PyTorch installation. For GPU acceleration follow the [CUDA installation guide](https://developer.nvidia.com/cuda-toolkit) and then install a CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

---

### Quick Start

This quick start covers the current `v1` workflow. The parallel `v2`
workflow is being added incrementally and will receive its own entry point
once the tracked v2 issues are implemented.

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
    --snapshot-every 5 \
    --output-dir ../results
```

**Outputs written to `--output-dir` (`project/results/` by default):**

| File | Description |
|---|---|
| `convergence.png` | Loss curves for all three training methods |
| `mse_evaluation.png` | Bar charts of final and mean MSE per method |
| `greedy_path_evaluation.png` | Mean greedy-path efficiency and goal-reaching success rate across seeded starts |
| `interruptions.json` | Separate FedAvg and gossip dropout/blackout logs keyed by training method |
| `snapshots/fedavg/*.png` | Optional FedAvg PNGs showing the current predicted distance field over the grid with drone positions overlaid |
| `snapshots/gossip/*.png` | Optional gossip PNGs showing the current predicted distance field over the grid with drone positions overlaid |

### V2 Directory Conventions

The v2 workflow uses dedicated roots so its artifacts do not collide with v1:

| Path | Purpose |
|---|---|
| `project/src/v2/` | Parallel namespace for all new v2 benchmark-specific source files |
| `project/results/v2/` | Experiment outputs, plots, logs, and snapshots for v2 runs |
| `project/data/v2_tasks/` | Cached sampled wetland-window tasks and immutable derived task artifacts for v2 |

These roots are reserved even before the full v2 pipeline exists so downstream
issues can add code and artifacts without ambiguity.

### V2 Task Library

The v2 workflow separates **task generation** from **model training**.

- A `task` is a frozen sampled wetland window derived from the statewide
    GeoPackage plus immutable metadata describing how that window was produced.
- A later training run reuses one cached task while varying drone starting
    positions, vision range, communication, movement policy, and compute
    constraints.
- Task generation should be deterministic from a sampling specification rather
    than from wall-clock time.
- The default v2 sampler now thinks in projected meters and targets local
    wetland zones rather than large statewide fractions: by default it samples
    a 500 m x 500 m axis-aligned window centered on a sampled wetland feature.

Issue #24 introduces two v2 modules for this layer:

| File | Status | Description |
|---|---|---|
| `v2/task_sampling.py` | In progress | Deterministic meter-scale wetland-window sampling from the statewide GeoPackage |
| `v2/task_cache.py` | In progress | Immutable task-artifact caching and manifest management under `project/data/v2_tasks/` |
| `v2/task_debug_render.py` | In progress | Detailed cached-task boundary and attribute renderer for local v2 inspection |

The v2 cache layout is:

| Path | Contents |
|---|---|
| `project/data/v2_tasks/manifest.json` | Task-library manifest keyed by deterministic task identifier |
| `project/data/v2_tasks/task-<hash>/window.geojson` | Clipped wetland window artifact for one sampled task |
| `project/data/v2_tasks/task-<hash>/metadata.json` | Reproducibility metadata: source dataset, sampling spec, window bounds, and summary stats |

The current minimum-content rule for sampled windows is intentionally simple:
- resample until the clipped window contains at least one non-empty wetland
    feature, and
- total clipped wetland area is strictly greater than the configured minimum
    threshold.

Re-requesting the same task specification should resolve to the same cached
task identifier and should not create a second logically identical artifact.

You can generate or resolve one cached v2 task outside the experimental loop by
running the sampler module directly:

```bash
python project/src/v2/task_sampling.py --seed 7 --window-size-m 500
```

That entry point resolves or creates the matching task under
`project/data/v2_tasks/` and prints the task identifier, artifact paths,
window metadata, and wetland summary statistics so later v2 runs can reuse the
same cached map window.

The default placement mode is `wetland-feature`, which centers each candidate
window on a sampled wetland geometry so small windows remain meaningful. If you
need the older statewide-relative behavior for coarse debugging, you can still
use `--window-width-fraction`, `--window-height-fraction`, and optionally set
`--anchor-mode statewide-uniform`.

To stay within the scale you described, use values like `--window-size-m 300`
for a few hundred meters or `--window-size-m 1000` for a 1 km square upper
bound.

For visual debugging, add `--debug-image` to emit a PNG showing the sampled
window inside the statewide extent plus the clipped wetland geometries in the
accepted window:

```bash
python project/src/v2/task_sampling.py --seed 7 --window-size-m 500 --min-total-wetland-area 0.8 --debug-image
```

When `--debug-image` is provided without a path, the sampler writes
`debug_window.png` inside the cached task directory. You may also pass an
explicit PNG path.

For more detailed cached-task inspection, use the standalone renderer to read
an existing task and write rich debug images under `project/results/v2/task_debug/`:

```bash
python project/src/v2/task_debug_render.py --task-id task-abcdef1234567890
```

The detailed renderer is read-only with respect to the task cache. For each
task it writes a task-specific output directory under
`project/results/v2/task_debug/<task-id>/` containing:
- `boundaries.png` for a detailed wetland-boundary view
- `cow_class1.png` when a matching `COW_CLASS1` / `cow_class1` column exists
- `spcc_desc.png` when a matching `SPCC_DESC` / `spcc_desc` column exists
- `render_manifest.json` summarizing which attributes were rendered and any
    documented fallback when an expected attribute column was missing

You may also target a cached task by explicit directory instead of task ID:

```bash
python project/src/v2/task_debug_render.py --task-dir project/data/v2_tasks/task-abcdef1234567890
```

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
| `--snapshot-every` | int | 0 | Save simulator-state PNGs every N decentralized rounds; requires `--use-simulator-integration` |

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
| `v2/` | In progress | Parallel namespace for mosquito-risk mapping workflow modules; reuse `load_data.py` and `comms.py` where applicable |
| `v2/task_sampling.py` | In progress | Deterministic wetland-window sampler for the v2 task library |
| `v2/task_cache.py` | In progress | Immutable cache and manifest helpers for sampled v2 tasks |

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
- If `--snapshot-every N` is enabled, the experiment writes method-specific PNG
    snapshots under `snapshots/fedavg/` and `snapshots/gossip/` at epoch 0,
    after every `N` decentralized rounds, and once more at the final epoch if
    the run ends off cadence.
- Each snapshot renders the current decentralized model's full-grid predicted
    distance field, overlays the current drone positions, and includes the
    method, epoch, and full-graph MSE in the title. If drone motion is disabled,
    marker positions may stay fixed while the heatmap still changes with model
    training.

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
