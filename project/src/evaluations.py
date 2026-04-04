from pathlib import Path
import sys
from typing import Union
import numpy as np
import torch
from torch_geometric import data
from model import WetlandGCN
from torch_geometric.data import Data
import torch_geometric.transforms as T

import matplotlib.pyplot as plt

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
    plt.ylabel("Loss")
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
    """Return (greedy_path_cost, optimal_path_cost, efficiency_ratio)."""
    if seed is not None:
        np.random.seed(seed)
    
    if start_node is None:
        start_node = np.random.randint(0, data.num_nodes)
    
    # Greedy path selection using model predictions
    greedy_cost = 0.0
    current = start_node
    visited = {current}
    data_targets = data.y

    while len(visited) < min(grid_size, data.num_nodes):
        # Get predictions for unvisited neighbors
        neighbors = [n for n in range(data.num_nodes) if n not in visited]

        if not neighbors:
            break

        # Select neighbor with lowest predicted cost
        costs = {}

        for n in neighbors:
            print(f"{n}~~~~~~~~~~~{type(data)}~~~~~~~~~~~~{data_targets.shape}~~~~~~~~~~~~")
            costs[n] = abs(model(x=data_targets, edge_index=n).detach().numpy())

        next_node = min(neighbors, key=lambda n: costs[n])

        # Update greedy cost
        greedy_cost += costs[next_node]

        # Mark next_node as current node, and add to visited
        visited.add(next_node)
        current = next_node
    
    # Optimal cost (simplified as mean of all node costs)
    all_costs = 0.0

    for v in visited:
        all_costs += abs(model(x=data_targets, edge_index=v).detach().numpy())

    optimal_cost = np.mean(all_costs[:min(grid_size, data.num_nodes)])
    
    efficiency_ratio = abs((optimal_cost - greedy_cost) / optimal_cost)
    
    return greedy_cost, optimal_cost, efficiency_ratio

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
    
    # Print MSE table
    print("\n" + "="*60)
    print("MSE EVALUATION TABLE")
    print("="*60)
    models = ['Centralized', 'FedAvg', 'Gossip']
    final_losses = [centralized_losses[-1], fedavg_losses[-1], gossip_losses[-1]]
    mean_losses = [np.mean(centralized_losses), np.mean(fedavg_losses), np.mean(gossip_losses)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(models, final_losses, color=['blue', 'green', 'red'], alpha=0.7)
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Final Loss by Model')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(models, mean_losses, color=['blue', 'green', 'red'], alpha=0.7)
    ax2.set_ylabel('Mean Loss')
    ax2.set_title('Mean Loss by Model')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "mse_evaluation.png", dpi=300)
    plt.close()
    
    # Save convergence plot
    print("\n" + "="*60)
    print("CONVERGENCE COMPARISON")
    print("="*60)
    compare_convergence(centralized_losses, fedavg_losses, gossip_losses, output_dir)
    print(f"Convergence plot saved to {output_dir / 'convergence.png'}")
    
    # Evaluate greedy paths
    print("\n" + "="*60)
    print("GREEDY PATH EVALUATION")
    print("="*60)
    greedy_c, optimal_c, eff_c = eval_greedy_path(data, centralized_model, grid_size=10)
    greedy_f, optimal_f, eff_f = eval_greedy_path(data, fedavg_model, grid_size=10)
    greedy_g, optimal_g, eff_g = eval_greedy_path(data, gossip_models[0], grid_size=10)
        
    # Save greedy path results as bar charts
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    models = ['Centralized', 'FedAvg', 'Gossip']
    greedy_costs = [greedy_c, greedy_f, greedy_g]
    optimal_costs = [optimal_c, optimal_f, optimal_g]
    efficiencies = [eff_c, eff_f, eff_g]
    
    ax1.bar(models, greedy_costs, color=['blue', 'green', 'red'], alpha=0.7)
    ax1.set_ylabel('Greedy Cost')
    ax1.set_title('Greedy Path Cost by Model')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(models, optimal_costs, color=['blue', 'green', 'red'], alpha=0.7)
    ax2.set_ylabel('Optimal Cost')
    ax2.set_title('Optimal Path Cost by Model')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3.bar(models, efficiencies, color=['blue', 'green', 'red'], alpha=0.7)
    ax3.set_ylabel('Efficiency Ratio')
    ax3.set_title('Path Efficiency by Model')
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "greedy_path_evaluation.png", dpi=300)
    plt.close()
    print(f"Greedy path evaluation saved to {output_dir / 'greedy_path_evaluation.png'}")


#--------------------------------------------------------------------------------------
## Example usage in main block
#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    base_model = WetlandGCN(hidden_channels=64)

    # Check if debug results directory exists (need to run this command in same directory as this program)
    debug_dir = Path("__file__").resolve().parent.parent / "data/DebugResults"
    fedavg_file = debug_dir / "fedavg_results.txt"
    fedavg_mfile = debug_dir / "fedavg_model.pth"
    centralized_file = debug_dir / "centralized_results.txt"
    gossip_file = debug_dir / "gossip_results.txt"
    gossip_mfile = debug_dir / "gossip_model.pth"
    sample_mfile = debug_dir / "sample_data.pth"

    if (not debug_dir.exists()):
        print("Creating debug directory for citable results...")
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Checking for files in \"{debug_dir}\"...")

    # Load results to debug if they exists, otherwise make them
    if ((fedavg_file.exists()) and (centralized_file.exists()) and 
        (gossip_file.exists()) and (fedavg_mfile.exists()) and 
        (gossip_mfile.exists()) and (sample_mfile.exists())):
        # All files exist, load results and models to save time for debugging
        print(f"Loading results from \"{debug_dir}\"...")
        centralized_results = np.loadtxt(centralized_file)
        gossip_results = np.loadtxt(gossip_file), torch.load(gossip_mfile, weights_only=False)
        fedavg_results = np.loadtxt(fedavg_file), torch.load(fedavg_mfile, weights_only=False)
        sample_data = torch.load(sample_mfile, weights_only=False)
    else:
        print(f"Creating Results and saving to \"{debug_dir}\"...")
        from build_graph import build_pyg_data
        from load_data import load_gdf
        from comms import CommunicationChannel
        from federated_agent import train_fedavg
        from train import train_centralized
        from train import train_gossip
        from partition import partition_nodes

        # Test data for the evaluations doesn't exist; so make them here
        gdf = load_gdf()
        sample_data, goal_node = build_pyg_data(gdf, grid_size=40, seed=42)
        comms = CommunicationChannel(comm_every=5, dropout_p=0.25, baseline_p=0.20, seed=42)
        training_partitions = partition_nodes(grid_size=40, K=5)

        centralized_results = train_centralized(sample_data, base_model, epochs=5, lr=1e-3)
        gossip_results = train_gossip(sample_data, training_partitions, comms, epochs=5, local_steps=3)
        fedavg_results = train_fedavg(sample_data, training_partitions, comms, epochs=5, local_steps=3)

        # save results to debug directory
        np.savetxt(centralized_file, centralized_results)
        np.savetxt(gossip_file, gossip_results[0])
        torch.save(gossip_results[1], gossip_mfile)
        np.savetxt(fedavg_file, fedavg_results[0])
        torch.save(fedavg_results[1], fedavg_mfile)
        torch.save(sample_data, sample_mfile)

    # Now test the full evaluation tool
    print("Comparing convergence results...")
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