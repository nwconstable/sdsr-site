from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from build_graph import build_pyg_data
from comms import CommunicationChannel
from evaluations import run_full_evaluation
from federated_agent import train_fedavg
from grid_sim import SpatialGridSimulator
from load_data import load_gdf
from model import WetlandGCN
from partition import partition_nodes
from train import train_centralized, train_gossip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the SDSR wetland swarm-learning experiment with centralized, "
            "FedAvg, and gossip training."
        )
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=50,
        help="Side length of the square grid graph.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=4,
        help="Number of drone partitions.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training rounds for each method.",
    )
    parser.add_argument(
        "--local-steps",
        type=int,
        default=5,
        help="Maximum local gradient steps per drone per round.",
    )
    parser.add_argument(
        "--comm-every",
        type=int,
        default=10,
        help="Epoch interval between communication events.",
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=0.1,
        help="Per-drone dropout probability on communication rounds.",
    )
    parser.add_argument(
        "--baseline-p",
        type=float,
        default=0.1,
        help="Dropout rate above which a round is flagged as blackout.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducible runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="project/results",
        help="Directory where plots and interruption logs are written.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="CPU thread cap used during local drone training.",
    )
    parser.add_argument(
        "--time-budget",
        type=int,
        default=None,
        help="Optional per-drone local-training budget in milliseconds.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = {
        "grid_size": args.grid_size,
        "K": args.K,
        "epochs": args.epochs,
        "local_steps": args.local_steps,
        "comm_every": args.comm_every,
        "num_threads": args.num_threads,
    }
    for name, value in positive_int_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    for name, value in {
        "dropout_p": args.dropout_p,
        "baseline_p": args.baseline_p,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    if args.lr <= 0.0:
        raise ValueError(f"lr must be positive, got {args.lr}")

    if args.time_budget is not None and args.time_budget <= 0:
        raise ValueError(
            f"time_budget must be positive when provided, got {args.time_budget}"
        )

    if args.K > args.grid_size:
        raise ValueError(
            f"K={args.K} must not exceed grid_size={args.grid_size} for strip partitioning"
        )


def configure_reproducibility(seed: int, num_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_interruptions(
    output_dir: Path,
    gossip_channel: CommunicationChannel,
    fedavg_channel: CommunicationChannel,
) -> Path:
    interruptions_path = output_dir / "interruptions.json"
    with interruptions_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "gossip": gossip_channel.logger.payload(),
                "fedavg": fedavg_channel.logger.payload(),
            },
            fh,
            indent=2,
        )
    return interruptions_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    configure_reproducibility(args.seed, args.num_threads)
    output_dir = resolve_output_dir(args.output_dir)

    print("[1/6] Loading wetlands data...")
    gdf_data = load_gdf()

    print("[2/6] Building graph dataset and partitions...")
    data, goal_node = build_pyg_data(gdf_data, grid_size=args.grid_size, seed=args.seed)
    partitions = partition_nodes(grid_size=args.grid_size, K=args.K)
    print(
        f"  Built graph with {data.num_nodes} nodes, goal node {goal_node}, "
        f"and {len(partitions)} drone partitions."
    )

    print("[3/6] Constructing communication channels and simulator...")
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
    simulator = SpatialGridSimulator(args.grid_size, partitions, data)
    print(f"  Initial drone positions: {simulator.drone_positions()}")

    hidden_channels = 64
    initial_model = WetlandGCN(
        in_channels=int(data.x.shape[1]), hidden_channels=hidden_channels
    )
    initial_state_dict = {
        key: value.detach().clone() for key, value in initial_model.state_dict().items()
    }

    print("[4/6] Training centralized baseline...")
    central_model = copy.deepcopy(initial_model)
    central_losses = train_centralized(data, central_model, args.epochs, args.lr)
    print(f"  Centralized final MSE: {central_losses[-1]:.4f}")

    print("[5/6] Training FedAvg and gossip models...")
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
    print(
        f"  FedAvg final MSE: {fedavg_losses[-1]:.4f} | "
        f"Gossip final MSE: {gossip_losses[-1]:.4f}"
    )

    print("[6/6] Saving logs and running evaluation...")
    interruptions_path = write_interruptions(output_dir, gossip_channel, fedavg_channel)
    run_full_evaluation(
        data,
        central_model,
        fedavg_model,
        gossip_models,
        central_losses,
        fedavg_losses,
        gossip_losses,
        output_dir=output_dir,
    )
    print(f"Interruption log saved to {interruptions_path}")
    print(f"Experiment outputs written to {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())