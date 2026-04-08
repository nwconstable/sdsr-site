import argparse
import copy
import json
import os
import random
import numpy as np
import torch
import sys
from load_data import load_gdf

from build_graph import build_pyg_data
from partition import partition_nodes
from comms import CommunicationChannel
from grid_sim import SpatialGridSimulator
from train import train_centralized, train_gossip
from evaluations import run_full_evaluation
from federated_agent import train_fedavg
from model import WetlandGCN

def main():
    parser = argparse.ArgumentParser(description="Run the SDSR experiment.")
    parser.add_argument('--grid-size', type=int, default=50, help='Grid size for simulation.')
    parser.add_argument('--K', type=int, default=4, help='Number of partitions or something.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--local-steps', type=int, default=5, help='Local training steps.')
    parser.add_argument('--comm-every', type=int, default=10, help='Communication frequency.')
    parser.add_argument('--dropout-p', type=float, default=0.1, help='Dropout probability.')
    parser.add_argument('--baseline-p', type=float, default=0.1, help='Baseline probability.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--output-dir', type=str, default='project/results', help='Output directory.')
    parser.add_argument('--num-threads', type=int, default=4, help="Number of threads for Gossip Training.")
    parser.add_argument('--time-budget', type=int, default=None, help="Time (ms) allowed for gossip training")

    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Data
    gdf_data = load_gdf()

    # Create training objects
    data, goal = build_pyg_data(gdf_data, grid_size=args.grid_size, seed=args.seed)
    partitions = partition_nodes(grid_size=args.grid_size, K=args.K)
    gossip_channel = CommunicationChannel(
        comm_every=args.comm_every,
        dropout_p=args.dropout_p,
        baseline_p=args.baseline_p,
        seed=args.seed,
    )
    fedavg_channel = CommunicationChannel(
        comm_every=args.comm_every,
        dropout_p=args.dropout_p,
        baseline_p=args.baseline_p,
        seed=args.seed,
    )

    # Create the models and losses
    hidden_channels = 64
    initial_model = WetlandGCN(
        in_channels=int(data.x.shape[1]), hidden_channels=hidden_channels
    )
    initial_state_dict = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }
    base_model = copy.deepcopy(initial_model)
    central_losses = train_centralized(data, base_model, args.epochs, args.lr)
    gossip_losses, gossip_models = train_gossip(
        data,
        partitions,
        gossip_channel,
        args.epochs,
        args.local_steps,
        args.lr,
        args.num_threads,
        args.time_budget,
        initial_state_dict=initial_state_dict,
        hidden_channels=hidden_channels,
    )
    fedavg_losses, fedavg_model = train_fedavg(
        data,
        partitions,
        fedavg_channel,
        args.epochs,
        args.local_steps,
        args.lr,
        args.num_threads,
        args.time_budget,
        initial_state_dict=initial_state_dict,
        hidden_channels=hidden_channels,
    )

    # Create Simulator
    simulator = SpatialGridSimulator(args.grid_size, partitions, data)

    # Evaluate and print results
    interruptions_path = os.path.join(args.output_dir, 'interruptions.json')
    with open(interruptions_path, 'w', encoding='utf-8') as fh:
        json.dump(
            {
                'gossip': gossip_channel.logger.payload(),
                'fedavg': fedavg_channel.logger.payload(),
            },
            fh,
            indent=2,
        )

    run_full_evaluation(data, base_model, fedavg_model, gossip_models, central_losses,
                         fedavg_losses, gossip_losses, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
    sys.exit(0)