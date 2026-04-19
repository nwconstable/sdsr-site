"""
train.py

Training routines for the wetlands GNN experiment.

Functions
---------
train_centralized  : train WetlandGCN on the full graph (centralized baseline)
train_gossip       : fully decentralised gossip training across K drone models
                     with per-drone CPU thread limit and optional time budget
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from comms import CommunicationChannel
from grid_sim import SpatialGridSimulator
from model import WetlandGCN
from partition import build_local_subgraph


def _clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state_dict.items()}


def _average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    avg_state: dict[str, torch.Tensor] = {}
    for key in state_dicts[0]:
        acc = state_dicts[0][key].detach().float().clone()
        for state_dict in state_dicts[1:]:
            acc += state_dict[key].detach().float()
        avg_state[key] = acc / len(state_dicts)
    return avg_state


def _evaluate_state_dict(
    data: Data,
    state_dict: dict[str, torch.Tensor],
    hidden_channels: int,
) -> float:
    eval_model = WetlandGCN(
        in_channels=int(data.x.shape[1]), hidden_channels=hidden_channels
    )
    eval_model.load_state_dict(state_dict)
    eval_model.eval()
    with torch.no_grad():
        return F.mse_loss(eval_model(data.x, data.edge_index), data.y).item()


def _predict_state_dict(
    data: Data,
    state_dict: dict[str, torch.Tensor],
    hidden_channels: int,
) -> torch.Tensor:
    eval_model = WetlandGCN(
        in_channels=int(data.x.shape[1]), hidden_channels=hidden_channels
    )
    eval_model.load_state_dict(state_dict)
    eval_model.eval()
    with torch.no_grad():
        return eval_model(data.x, data.edge_index).detach().cpu().reshape(-1)


def _should_save_snapshot(epoch: int, total_epochs: int, snapshot_every: int) -> bool:
    if snapshot_every <= 0:
        return False
    if epoch == 0 or epoch == total_epochs:
        return True
    return epoch % snapshot_every == 0


def train_centralized(
    data: Data,
    model: WetlandGCN,
    epochs: int,
    lr: float = 1e-3,
) -> list[float]:
    """Train *model* on the full graph for *epochs* steps.

    Parameters
    ----------
    data   : PyG Data with fields x (N,1) and y (N,1)
    model  : WetlandGCN instance; updated in-place
    epochs : number of gradient steps
    lr     : Adam learning rate

    Returns
    -------
    losses : list of per-epoch MSE loss values (length == epochs)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    losses: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# ---------------------------------------------------------------------------
# Gossip training
# ---------------------------------------------------------------------------

def train_gossip(
    data: Data,
    partitions: list[np.ndarray],
    channel: CommunicationChannel,
    epochs: int,
    local_steps: int,
    lr: float = 1e-3,
    num_threads: int = 4,
    time_budget_ms: int | None = None,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    hidden_channels: int = 64,
    simulator: SpatialGridSimulator | None = None,
    simulator_view_radius: int = 2,
    simulator_comm_radius: int = 2,
    move_drones: bool = False,
    snapshot_every: int = 0,
    snapshot_output_dir: str | Path | None = None,
    snapshot_method_name: str = "Gossip",
) -> tuple[list[float], list[WetlandGCN]]:
    """Fully decentralised gossip training.

     Each epoch performs local training. On scheduled communication epochs,
     ``channel.gossip_pairs`` returns random pairs of drones whose weights are
     averaged with no central server. The evaluation model is always the
     element-wise average of all drone state_dicts, evaluated on the full graph.

    Parameters
    ----------
    data           : full PyG Data object (x, edge_index, y)
    partitions     : list of K global node-index arrays (one per drone)
    channel        : CommunicationChannel — governs comm schedule and dropout
    epochs         : number of local-training epochs
    local_steps    : max gradient steps each drone takes before a possible gossip exchange
    lr             : Adam learning rate
    num_threads    : PyTorch CPU thread count per drone.  Default 4 reflects a
                     typical ARM Cortex-A72 / Jetson Nano class device (4 cores).
                     Override to simulate faster (8) or slower (1-2) hardware.
    time_budget_ms : Wall-clock time budget in ms for each drone's local training
                     step.  The gradient loop exits early if the budget runs out
                     before *local_steps* are reached.  None = no constraint.

    Returns
    -------
    losses       : per-round MSE on the full graph (length == epochs)
    drone_models : list of final WetlandGCN models, one per drone
    """
    K = len(partitions)
    drone_ids = list(range(K))

    if initial_state_dict is None:
        template_model = WetlandGCN(
            in_channels=int(data.x.shape[1]), hidden_channels=hidden_channels
        )
        initial_state_dict = _clone_state_dict(template_model.state_dict())

    drone_models: list[WetlandGCN] = [
        WetlandGCN(in_channels=int(data.x.shape[1]), hidden_channels=hidden_channels)
        for _ in range(K)
    ]
    for model in drone_models:
        model.load_state_dict(_clone_state_dict(initial_state_dict))
    optimizers = [torch.optim.Adam(m.parameters(), lr=lr) for m in drone_models]
    use_simulator = simulator is not None
    if snapshot_every > 0 and not use_simulator:
        raise ValueError("Snapshot export requires a simulator-backed gossip run.")
    if snapshot_every > 0 and snapshot_output_dir is None:
        raise ValueError("snapshot_output_dir is required when snapshot_every > 0.")
    local_subgraphs = None
    if not use_simulator:
        local_subgraphs = [build_local_subgraph(data, idx) for idx in partitions]

    losses: list[float] = []

    if use_simulator and _should_save_snapshot(0, epochs, snapshot_every):
        initial_avg_state = _average_state_dicts(
            [model.state_dict() for model in drone_models]
        )
        simulator.save_snapshot(
            epoch=0,
            output_dir=snapshot_output_dir,
            method_name=snapshot_method_name,
            node_values=_predict_state_dict(data, initial_avg_state, hidden_channels),
            metric_label="MSE",
            metric_value=_evaluate_state_dict(data, initial_avg_state, hidden_channels),
        )

    for comm_round in range(epochs):
        if use_simulator and move_drones:
            simulator.step_drones()

        round_subgraphs = (
            [
                simulator.get_local_view(drone_id, radius=simulator_view_radius)
                for drone_id in drone_ids
            ]
            if use_simulator
            else local_subgraphs
        )

        # 1. Local training — apply per-drone compute constraints
        for model, opt, sub in zip(drone_models, optimizers, round_subgraphs):
            model.train()
            _prev_threads = torch.get_num_threads()
            torch.set_num_threads(num_threads)
            _t0 = time.perf_counter()
            for _ in range(local_steps):
                opt.zero_grad()
                out = model(sub.x, sub.edge_index)
                loss = F.mse_loss(out, sub.y)
                loss.backward()
                opt.step()
                if (
                    time_budget_ms is not None
                    and (time.perf_counter() - _t0) * 1000 >= time_budget_ms
                ):
                    break
            torch.set_num_threads(_prev_threads)

        # 2. Gossip exchange — pairs determined by the channel
        if channel.is_comm_round(comm_round):
            if use_simulator:
                allowed_pairs = simulator.proximity_pairs(
                    simulator_comm_radius, drone_ids=drone_ids
                )
                pairs = channel.gossip_pairs(drone_ids, allowed_pairs=allowed_pairs)
            else:
                pairs = channel.gossip_pairs(drone_ids)
            for i, j in pairs:
                sd_i = _clone_state_dict(drone_models[i].state_dict())
                sd_j = _clone_state_dict(drone_models[j].state_dict())
                avg_sd = {k: (sd_i[k] + sd_j[k]) / 2.0 for k in sd_i}
                drone_models[i].load_state_dict(avg_sd)
                drone_models[j].load_state_dict(copy.deepcopy(avg_sd))
                # Reset optimizers so stale moment estimates don't pollute new weights
                optimizers[i] = torch.optim.Adam(drone_models[i].parameters(), lr=lr)
                optimizers[j] = torch.optim.Adam(drone_models[j].parameters(), lr=lr)

        avg_global = _average_state_dicts(
            [model.state_dict() for model in drone_models]
        )
        losses.append(_evaluate_state_dict(data, avg_global, hidden_channels))

        completed_rounds = comm_round + 1
        if use_simulator and _should_save_snapshot(
            completed_rounds, epochs, snapshot_every
        ):
            simulator.save_snapshot(
                epoch=completed_rounds,
                output_dir=snapshot_output_dir,
                method_name=snapshot_method_name,
                node_values=_predict_state_dict(data, avg_global, hidden_channels),
                metric_label="MSE",
                metric_value=losses[-1],
            )

    return losses, drone_models


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a tiny 2x5 grid (10 nodes) with synthetic wetland features and
    # hand-crafted distance targets so no GeoPackage file is needed.

    ROWS, COLS = 2, 5
    N = ROWS * COLS

    # Node features: alternating wetland presence
    wetland = torch.tensor(
        [[float(i % 2)] for i in range(N)], dtype=torch.float
    )

    # 4-neighbor edges for a 2x5 grid
    src, dst = [], []
    for r in range(ROWS):
        for c in range(COLS):
            u = r * COLS + c
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    src.append(u)
                    dst.append(nr * COLS + nc)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Synthetic targets: distance = node index (increases away from node 0)
    targets = torch.tensor([[float(i)] for i in range(N)], dtype=torch.float)

    data = Data(x=wetland, edge_index=edge_index, y=targets)

    model = WetlandGCN(in_channels=data.x.shape[1], hidden_channels=64)
    initial_loss = F.mse_loss(model(data.x, data.edge_index), data.y).item()

    EPOCHS = 10
    losses = train_centralized(data, model, epochs=EPOCHS, lr=1e-2)

    print(f"Initial loss : {initial_loss:.4f}")
    for i, l in enumerate(losses, 1):
        print(f"  Epoch {i:>2} : {l:.4f}")
    print(f"Final loss   : {losses[-1]:.4f}")
    assert losses[-1] < initial_loss, "Loss did not decrease"
    print("Centralized smoke test passed.")

    # ------------------------------------------------------------------
    # train_gossip smoke test
    # ------------------------------------------------------------------
    # Split 10-node grid into 2 partitions: left half (cols 0-2) / right half (cols 3-4)
    partitions = [np.array([0, 1, 2, 5, 6, 7]), np.array([3, 4, 8, 9])]

    channel = CommunicationChannel(comm_every=1, dropout_p=0.0)
    GOSSIP_ROUNDS = 5

    g_losses, drone_models = train_gossip(
        data, partitions, channel, epochs=GOSSIP_ROUNDS, local_steps=10,
        lr=1e-2, num_threads=4, time_budget_ms=50,
    )

    print(f"\nGossip losses over {GOSSIP_ROUNDS} rounds:")
    for i, l in enumerate(g_losses, 1):
        print(f"  Round {i} : {l:.4f}")
    assert len(g_losses) == GOSSIP_ROUNDS, "Wrong number of loss entries"
    assert len(drone_models) == len(partitions), "Wrong number of drone models"
    print("Gossip smoke test passed.")
    sys.exit(0)
