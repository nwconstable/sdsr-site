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

    args = parser.parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Execution order
    # 1. Load GeoDataFrame
    gdf_data = gdf()

    # 2. build_pyg_data -> data
    data = build_pyg_data(gdf_data)

    # 3. partition_nodes -> partitions
    partitions = partition_nodes(data, args.K)

    # 4. Construct CommunicationChannel
    channel = CommunicationChannel(partitions, args.comm_every, args.dropout_p, args.baseline_p)

    # 5. Construct SpatialGridSimulator
    simulator = SpatialGridSimulator(data, partitions, channel, args.grid_size)

    # 6. train_centralized
    train_centralized(simulator, args.epochs, args.lr)

    # 7. train_fedavg
    train_fedavg(simulator, args.epochs, args.local_steps, args.lr)

    # 8. train_gossip
    train_gossip(simulator, args.epochs, args.local_steps, args.lr)

    # 9. run_full_evaluation
    run_full_evaluation(simulator)

    # 10. channel.logger.save(output_dir/interruptions.json)
    channel.logger.save(os.path.join(args.output_dir, 'interruptions.json'))

if __name__ == '__main__':
    main()
    sys.exit(0)