from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Union
import numpy as np
import torch
from model import WetlandGCN
from torch_geometric.data import Data

import matplotlib.pyplot as plt


def _average_model_state(models: list[WetlandGCN]) -> dict[str, torch.Tensor]:
    avg_state: dict[str, torch.Tensor] = {}
    state_dicts = [model.state_dict() for model in models]
    for key in state_dicts[0]:
        acc = state_dicts[0][key].detach().float().clone()
        for state_dict in state_dicts[1:]:
            acc += state_dict[key].detach().float()
        avg_state[key] = acc / len(state_dicts)
    return avg_state


def _build_eval_model(data: Data, state_dict: dict[str, torch.Tensor]) -> WetlandGCN:
    hidden_channels = int(state_dict["conv1.bias"].shape[0])
    model = WetlandGCN(
        in_channels=int(data.x.shape[1]), hidden_channels=hidden_channels
    )
    model.load_state_dict(state_dict)
    return model


def _rollout_greedy_path(
    data: Data,
    model: WetlandGCN,
    start_node: int,
    max_steps: int,
) -> dict[str, Any]:
    ei = data.edge_index.detach().cpu().numpy()
    adj: dict[int, list[int]] = {i: [] for i in range(data.num_nodes)}
    for src, dst in zip(ei[0], ei[1]):
        adj[int(src)].append(int(dst))

    goal_node = int(getattr(data, "goal_node", int(data.y.argmin().item())))

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).squeeze(-1)

    current = start_node
    visited: set[int] = {current}
    greedy_cost = 0.0
    steps = 0
    reached_goal = current == goal_node

    while steps < max_steps and not reached_goal:
        nbrs = [node for node in adj[current] if node not in visited]
        if not nbrs:
            break
        next_node = min(nbrs, key=lambda node: pred[node].item())
        greedy_cost += 5.0 if data.x[next_node, 0].item() > 0.5 else 1.0
        visited.add(next_node)
        current = next_node
        steps += 1
        reached_goal = current == goal_node

    optimal_cost = float(data.y[start_node].item())
    if start_node == goal_node:
        efficiency_ratio = 1.0
    elif reached_goal and greedy_cost > 0.0:
        efficiency_ratio = min(optimal_cost / greedy_cost, 1.0)
    else:
        efficiency_ratio = 0.0

    return {
        "start_node": start_node,
        "greedy_cost": greedy_cost,
        "optimal_cost": optimal_cost,
        "efficiency": efficiency_ratio,
        "steps": steps,
        "reached_goal": reached_goal,
    }


def compare_convergence(
    centralized_losses: list[float],
    fedavg_losses: list[float],
    gossip_losses: list[float],
    output_dir: str | Path = "project/results",
) -> None:
    """Save convergence plot to output_dir/convergence.png."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(centralized_losses, label="Centralized", linewidth=2)
    plt.plot(fedavg_losses, label="FedAvg", linewidth=2)
    plt.plot(gossip_losses, label="Gossip", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Convergence Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "convergence.png", dpi=300)
    plt.close()


def eval_greedy_path(
    data: Data,
    model: WetlandGCN,
    grid_size: int,
    start_node: int | None = None,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Simulate greedy pathfinding and return evaluation metrics.

    The drone starts at *start_node* and at each step moves to the unvisited
    neighbour with the lowest model-predicted distance-to-goal.  Traversal
    stops when the goal is reached or after a bounded number of steps based on
    the graph size.

    Parameters
    ----------
    data       : full PyG Data (x, edge_index, y).  data.y holds the
                 pre-computed Dijkstra optimal distances (from build_graph.py).
    model      : trained WetlandGCN
    grid_size  : side length of the grid (used for the bounded rollout cap)
    start_node : starting node index; random if None
    seed       : controls the random start node selection

    Returns
    -------
    greedy_path_cost  : total traversal cost (5 per wetland hop, 1 per land hop)
    optimal_path_cost : Dijkstra label at start_node (data.y[start_node])
    efficiency_ratio  : optimal / greedy in (0, 1]; 1.0 = greedy matches Dijkstra
    """
    rng = np.random.default_rng(seed)
    goal_node = int(getattr(data, "goal_node", int(data.y.argmin().item())))
    if start_node is None:
        candidates = [node for node in range(data.num_nodes) if node != goal_node]
        start_node = int(rng.choice(candidates))

    result = _rollout_greedy_path(
        data,
        model,
        start_node=start_node,
        max_steps=data.num_nodes - 1,  # worst case: visit all other nodes
    )
    return result["greedy_cost"], result["optimal_cost"], result["efficiency"]


def evaluate_greedy_paths(
    data: Data,
    model: WetlandGCN,
    grid_size: int,
    num_starts: int = 25,
    seed: int | None = None,
) -> dict[str, float]:
    """Evaluate greedy navigation over multiple seeded starts."""
    rng = np.random.default_rng(seed)
    goal_node = int(getattr(data, "goal_node", int(data.y.argmin().item())))
    candidates = np.array([node for node in range(data.num_nodes) if node != goal_node])
    sample_size = min(num_starts, len(candidates))
    start_nodes = rng.choice(candidates, size=sample_size, replace=False)
    results = [
        _rollout_greedy_path(
            data,
            model,
            start_node=int(start_node),
            max_steps=data.num_nodes - 1,  # worst case: visit all other nodes
        )
        for start_node in start_nodes
    ]

    return {
        "mean_greedy_cost": float(np.mean([result["greedy_cost"] for result in results])),
        "mean_optimal_cost": float(np.mean([result["optimal_cost"] for result in results])),
        "mean_efficiency": float(np.mean([result["efficiency"] for result in results])),
        "success_rate": float(np.mean([result["reached_goal"] for result in results])),
        "num_starts": float(sample_size),
    }


def plot_greedy_path_summary(
    metrics_by_model: dict[str, dict[str, float]],
    output_dir: str | Path = "project/results",
) -> None:
    """Save a summary chart for multi-start greedy-path evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(metrics_by_model.keys())
    efficiencies = [metrics_by_model[name]["mean_efficiency"] for name in models]
    success_rates = [metrics_by_model[name]["success_rate"] for name in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["blue", "green", "red"]
    ax1.bar(models, efficiencies, color=colors, alpha=0.75)
    ax1.set_ylabel("Mean Efficiency")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_title("Greedy Path Efficiency")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(models, success_rates, color=colors, alpha=0.75)
    ax2.set_ylabel("Success Rate")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("Goal-Reaching Success Rate")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "greedy_path_evaluation.png", dpi=300)
    plt.close()


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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = int(getattr(data, "grid_size", int(data.num_nodes ** 0.5)))

    # ---- Formatted MSE table ----
    print("\n" + "=" * 60)
    print("MSE EVALUATION TABLE")
    print("=" * 60)
    print(f"{'Model':<16} | {'Final MSE':>10} | {'Mean MSE':>10}")
    print("-" * 42)
    for name, losses in [
        ("Centralized", centralized_losses),
        ("FedAvg", fedavg_losses),
        ("Gossip", gossip_losses),
    ]:
        print(f"{name:<16} | {losses[-1]:>10.4f} | {np.mean(losses):>10.4f}")

    # ---- Convergence plot ----
    print("\n" + "=" * 60)
    print("CONVERGENCE COMPARISON")
    print("=" * 60)
    compare_convergence(centralized_losses, fedavg_losses, gossip_losses, output_dir)
    print(f"Convergence plot saved to {output_dir / 'convergence.png'}")

    # ---- Greedy path evaluation ----
    print("\n" + "=" * 60)
    print("GREEDY PATH EVALUATION")
    print("=" * 60)
    print(f"{'Model':<16} | {'Greedy Cost':>12} | {'Optimal Cost':>12} | {'Efficiency':>10}")
    print("-" * 58)

    if gossip_models:
        gossip_eval_model = _build_eval_model(data, _average_model_state(gossip_models))
    else:
        gossip_eval_model = WetlandGCN(in_channels=int(data.x.shape[1]), hidden_channels=64)

    greedy_metrics = {
        "Centralized": evaluate_greedy_paths(
            data, centralized_model, grid_size=grid_size, seed=42
        ),
        "FedAvg": evaluate_greedy_paths(
            data, fedavg_model, grid_size=grid_size, seed=42
        ),
        "Gossip": evaluate_greedy_paths(
            data, gossip_eval_model, grid_size=grid_size, seed=42
        ),
    }
    for name, mdl in [
        ("Centralized", centralized_model),
        ("FedAvg", fedavg_model),
        ("Gossip", gossip_eval_model),
    ]:
        metrics = greedy_metrics[name]
        print(
            f"{name:<16} | {metrics['mean_greedy_cost']:>12.2f} | "
            f"{metrics['mean_optimal_cost']:>12.2f} | {metrics['mean_efficiency']:>10.4f}"
        )
    print("\nGreedy-path success rates:")
    for name, metrics in greedy_metrics.items():
        print(
            f"  {name:<12} success={metrics['success_rate']:.2%} "
            f"over {int(metrics['num_starts'])} starts"
        )
    plot_greedy_path_summary(greedy_metrics, output_dir)

    # ---- MSE bar charts ----
    models = ["Centralized", "FedAvg", "Gossip"]
    final_losses = [centralized_losses[-1], fedavg_losses[-1], gossip_losses[-1]]
    mean_losses = [np.mean(centralized_losses), np.mean(fedavg_losses),
                   np.mean(gossip_losses)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["blue", "green", "red"]
    ax1.bar(models, final_losses, color=colors, alpha=0.7)
    ax1.set_ylabel("Final Loss")
    ax1.set_title("Final Loss by Model")
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.bar(models, mean_losses, color=colors, alpha=0.7)
    ax2.set_ylabel("Mean Loss")
    ax2.set_title("Mean Loss by Model")
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "mse_evaluation.png", dpi=300)
    plt.close()
    print(f"\nMSE bar charts saved to {output_dir / 'mse_evaluation.png'}")
    print(f"Greedy-path summary saved to {output_dir / 'greedy_path_evaluation.png'}")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from build_graph import build_pyg_data
    from load_data import load_gdf
    from comms import CommunicationChannel
    from federated_agent import train_fedavg
    from train import train_centralized, train_gossip
    from partition import partition_nodes

    base_dir = Path(__file__).resolve().parent.parent
    debug_dir = base_dir / "data" / "DebugResults"
    fedavg_file = debug_dir / "fedavg_results.txt"
    fedavg_mfile = debug_dir / "fedavg_model.pth"
    centralized_file = debug_dir / "centralized_results.txt"
    gossip_file = debug_dir / "gossip_results.txt"
    gossip_mfile = debug_dir / "gossip_model.pth"
    sample_mfile = debug_dir / "sample_data.pth"

    debug_dir.mkdir(parents=True, exist_ok=True)

    all_cached = all(p.exists() for p in [fedavg_file, fedavg_mfile,
                                            centralized_file, gossip_file,
                                            gossip_mfile, sample_mfile])

    if all_cached:
        print(f"Loading cached results from {debug_dir} ...")
        centralized_results = np.loadtxt(centralized_file).tolist()
        gossip_results = (
            np.loadtxt(gossip_file).tolist(),
            torch.load(gossip_mfile, weights_only=False),
        )
        fedavg_results = (
            np.loadtxt(fedavg_file).tolist(),
            torch.load(fedavg_mfile, weights_only=False),
        )
        sample_data = torch.load(sample_mfile, weights_only=False)
    else:
        print(f"Generating results and saving to {debug_dir} ...")
        gdf = load_gdf()
        sample_data, goal_node = build_pyg_data(gdf, grid_size=40, seed=42)
        comms = CommunicationChannel(comm_every=5, dropout_p=0.25,
                                      baseline_p=0.20, seed=42)
        training_partitions = partition_nodes(grid_size=40, K=5)
        base_model = WetlandGCN(in_channels=sample_data.x.shape[1], hidden_channels=64)

        centralized_results = train_centralized(sample_data, base_model,
                                                 epochs=5, lr=1e-3)
        gossip_results = train_gossip(sample_data, training_partitions, comms,
                                       epochs=5, local_steps=3)
        fedavg_results = train_fedavg(sample_data, training_partitions, comms,
                                       epochs=5, local_steps=3)

        np.savetxt(centralized_file, centralized_results)
        np.savetxt(gossip_file, gossip_results[0])
        torch.save(gossip_results[1], gossip_mfile)
        np.savetxt(fedavg_file, fedavg_results[0])
        torch.save(fedavg_results[1], fedavg_mfile)
        torch.save(sample_data, sample_mfile)

    base_model = WetlandGCN(in_channels=sample_data.x.shape[1], hidden_channels=64)
    print("Running full evaluation ...")
    run_full_evaluation(
        data=sample_data,
        centralized_model=base_model,
        fedavg_model=fedavg_results[1],
        gossip_models=gossip_results[1],
        centralized_losses=centralized_results,
        fedavg_losses=fedavg_results[0],
        gossip_losses=gossip_results[0],
        output_dir=debug_dir,
    )
    print("Done")
    sys.exit(0)
