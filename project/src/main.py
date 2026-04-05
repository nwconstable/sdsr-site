import argparse
import os
import random
import numpy as np
import torch
import sys
from load_data import gdf

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
    gdf_data = gdf()

    # Create training objects
    data, goal = build_pyg_data(gdf_data, grid_size=args.grid_size, seed=args.seed)
    channel = CommunicationChannel(comm_every=args.comm_every, dropout_p=args.dropout_p,
                                    baseline_p=args.baseline_p, seed=args.seed)
    partitions = partition_nodes(grid_size=args.grid_size, K=args.K)

    # Create the models and losses
    base_model = WetlandGCN()
    central_losses = train_centralized(data, base_model, args.epochs, args.lr)
    gossip_losses, gossip_models = train_gossip(data, partitions, channel, args.epochs,
                                                 args.local_steps, args.lr,
                                                 args.num_threads, args.time_budget)
    fedavg_losses, fedavg_model = train_fedavg(data, partitions, channel, args.epochs,
                                                args.local_steps, args.lr, 
                                                args.num_threads, args.time_budget)

    # Create Simulator
    simulator = SpatialGridSimulator(data, partitions, channel, args.grid_size)

    # Evaluate and print results
    channel.logger.save(os.path.join(args.output_dir, 'interruptions.json'))

    run_full_evaluation(data, simulator.centralized_model, simulator.fedavg_model,
                         simulator.gossip_models, simulator.centralized_losses,
                         simulator.fedavg_losses, simulator.gossip_losses)

if __name__ == '__main__':
    main()
    sys.exit(0)