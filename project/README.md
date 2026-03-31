# Project: Networked Edge Training for Spatial Graph Learning
## Group 2 - Noah Constable, Joe Conroy

### Purpose
The purpose of this project is to train a network of simulated edge devices (i.e. drones) using multiple different techniques including federated average learning and gossip learning. We want to empirically characterize each of these methods against each other to determine approximate bounds on resources such as network connectivity and compute. 

### Resources
One of the data sets we are using is the [Minnesota National Wetlands Inventory Update](https://www.dnr.state.mn.us/wetlands/nwi_proj.html) GeoPackage. This is a GIS dataset that requires 'wrangling' into an appropriate form for our purposes.

To use the programs, especially any of the PyG scripts, a CUDA-enabled GPU is recommended for performance but **not required**. All scripts should run on a CPU-only installation of PyTorch and PyTorch Geometric. If you have a compatible NVIDIA GPU and want GPU acceleration, follow the [official CUDA installation guide](https://developer.nvidia.com/cuda-toolkit) and then install a CUDA-enabled build of [PyTorch](https://pytorch.org/get-started/locally/). For CPU-only use, install the CPU-only PyTorch wheels as described in the PyTorch documentation.

### How to Use
1. (Optional) [Enable Python venv](https://www.w3schools.com/python/python_virtualenv.asp)
2. Install [dependencies](#libraries)
3. Using Python, run the scripts in the following order (unless you know what you're doing):
    a. `load_data.py`
    b. `build_graph.py`
    c. `federated_agent.py` — standalone demo of FedAvg agent interaction (requires data from step b)

> **Note:** The full experiment pipeline (`partition.py`, `comms.py`, `simulator.py`, `train.py` w/ FedAvg, `evaluate.py`, `main.py`) is under active development. See `.github/issues.md` for the remaining implementation backlog.

### Module Inventory

| File | Status | Description |
|---|---|---|
| `load_data.py` | Done | Validates and loads the wetlands GeoPackage via GeoPandas |
| `build_graph.py` | Done | Builds PyG `Data` object (grid, wetland features, Dijkstra labels) |
| `model.py` | Done | `WetlandGCN`: 2-layer GCN for node-level distance regression |
| `train.py` | Partial | `train_centralized`, `train_gossip` (with compute constraints); FedAvg in `federated_agent.py` |
| `federated_agent.py` | Done | `NodeAgent` / `CentralAgent` classes and `train_fedavg`; compute constraints supported |

### Per-drone compute simulation

Both `train_gossip` (`train.py`) and `NodeAgent.train_local` (`federated_agent.py`) accept two parameters that simulate the resource limits of an edge device:

| Parameter | Default | Description |
|---|---|---|
| `num_threads` | `4` | Caps PyTorch's CPU thread pool via `torch.set_num_threads()` for each drone's local training step. 4 reflects an ARM Cortex-A72 (Raspberry Pi 4) or Jetson Nano class board — a realistic drone compute target. Use `1–2` for heavily constrained devices, `8` for high-end boards. |
| `time_budget_ms` | `None` | Wall-clock deadline in milliseconds for the gradient loop. The loop exits after the first step that exceeds the budget, regardless of remaining `local_steps`. `None` = unconstrained. |

```python
# Example: simulate a constrained 2-core drone with a 100 ms / round compute cap
train_gossip(data, partitions, channel, epochs=50, local_steps=20,
             num_threads=2, time_budget_ms=100)
```
| `partition.py` | Pending | Spatial node partitioning into K drone regions (Issue #1) |
| `comms.py` | Pending | `CommunicationChannel` and `ProtocolInterruptionLogger` (Issue #2) |
| `simulator.py` | Pending | `SpatialGridSimulator` — drone positions and visualisation (Issue #3) |
| `evaluate.py` | Pending | Convergence plots and greedy-path evaluation (Issue #7) |
| `main.py` | Pending | Full experiment entry point with argparse CLI (Issue #8) |

### Libraries
Packages listed here can be installed by using:

`pip install {package_name}`

Python libraries used include:
- geopandas
- shapely (included with geopandas)
- numpy
- torch_geometric
- torch
- matplotlib *(required by `simulator.py` and `evaluate.py` — pending)*