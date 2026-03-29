from dataclasses import dataclass
import sys
from typing import Any, Dict, List, Optional, Union
import uuid
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from model import WetlandGCN

#
# Node Agent Class
#
@dataclass
class NodeAgent:
    """Agent running as an individual node in a federated environment"""
    node_id: str
    local_data: Dict[str, Any]
    model_params: Dict[str, torch.Tensor]
    current_loc: Optional[int] = None

    def __init__(self, node_id: str = None, current_loc: Optional[int] = None):
        self.node_id = node_id or str(uuid.uuid4())[:8]

        if (current_loc is not None):
            if current_loc < 0:
                raise ValueError("current_loc must be non-negative")
            self.current_loc = current_loc
        else:
            self.current_loc = None

        self.local_data = {}
        self.model_params = {}

    # Grab a subset of the graph for training
    def _select_local_subset(self, data: Data, location: Optional[int], radius: int) -> torch.Tensor:
        """
        data: Full graph data object
        location: Node index to center the local subset around (drone location)
        radius: Number of hops in the grid graph to include in the local subset 
            (e.g., radius=1 includes immediate neighbors, radius=2 includes neighbors of 
            immediate neighbors and immediate neighbors, etc.)
        """
        N = data.num_nodes

        if location is None:
            # Presume training at current_loc if location not provided
            if self.current_loc is None:
                # Random start if no current_loc provided at initialization (i.e. random start)
                self.current_loc = random.randint(0, N - 1)

            location = self.current_loc

        # infer grid size from the graph (stored metadata or sqrt on num_nodes)
        grid_size = int(getattr(data, "grid_size", int(torch.sqrt(torch.tensor(N, dtype=torch.float)).item())))
        row = location // grid_size
        col = location % grid_size

        rows = torch.arange(N, dtype=torch.long) // grid_size
        cols = torch.arange(N, dtype=torch.long) % grid_size

        mask = (rows - row).abs() + (cols - col).abs() <= radius
        subset_idx = torch.nonzero(mask, as_tuple=False).view(-1)

        return location, subset_idx

    # Train a local model on the selected subset of the graph
    def train_local(
        self,
        training_data: Union[Data, Dict[str, Any]],
        location: Optional[int] = None,
        radius: int = 2,
        epochs: int = 5,
        lr: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        training_data: Full graph data object (PyG Data or compatible dict)
        location: Node index to center the local subset around (drone location). If None, default to current_loc
        radius: Number of hops in the grid graph to include in the local subset (e.g
                radius=1 includes immediate neighbors, radius=2 includes neighbors of neighbors, etc.)
        epochs: Number of local training epochs
        lr: Learning rate for local training
        """
        if isinstance(training_data, dict):
            training_data = Data(**training_data)
        elif not isinstance(training_data, Data):
            raise ValueError(f"training_data {type(training_data)} must be a PyG Data object or compatible dict")

        if (location is not None) and ((location < 0) or (location >= training_data.num_nodes)):
            raise ValueError(f"location {location} must be between 0 and {training_data.num_nodes - 1}")

        self.local_data = {
            "location": location,
            "radius": radius,
            "num_nodes": training_data.num_nodes,
        }

        # Select local subset of the graph based on location and radius (recall default is self.current_loc)
        location, subset_idx = self._select_local_subset(training_data, location, radius)

        sub_edge_index, _ = subgraph(subset_idx, training_data.edge_index, relabel_nodes=True, num_nodes=training_data.num_nodes)
        sub_x = training_data.x[subset_idx]
        sub_y = training_data.y[subset_idx]

        model = WetlandGCN(hidden_channels=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()

        losses: List[float] = []

        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(sub_x, sub_edge_index)
            loss = F.mse_loss(out, sub_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.local_data.update(
        {
            "trained_nodes": len(subset_idx),
            "location": location,
            "last_loss": losses[-1] if losses else None,
            "loss_curve": losses,
        })

        self.model_params = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        return self.model_params
    
    def receive_global_params(self, global_params: Dict[str, float]) -> None:
        """Receive and update global parameters from central agent"""
        self.model_params = global_params.copy()
    
    def send_local_update(self) -> Dict[str, Any]:
        """Send local model update to central agent"""
        return {
            "node_id": self.node_id,
            "params": self.model_params,
            "data_size": len(self.local_data)
        }

#
# Central Agent Class
#
@dataclass
class CentralAgent:
    """Central coordinator agent in federated environment"""
    agent_id: str
    nodes: List[NodeAgent]
    global_params: Dict[str, float]
    
    def __init__(self):
        self.agent_id = "central_" + str(uuid.uuid4())[:8]
        self.nodes = []
        self.global_params = {}
    
    def register_node(self, node: NodeAgent) -> None:
        """Register a new node agent"""
        self.nodes.append(node)
    
    def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate updates from all nodes using federated averaging"""
        if not updates:
            return self.global_params
        
        total_data_size = sum(u["data_size"] for u in updates)
        aggregated = {}
        
        for update in updates:
            # Presumes weight is proportional to the amount of data used for training at that node
            weight = update["data_size"] / total_data_size

            for key, value in update["params"].items():
                aggregated[key] = aggregated.get(key, 0) + value * weight
        
        self.global_params = aggregated

        return self.global_params
    
    def broadcast_params(self) -> None:
        """Broadcast global parameters to all nodes"""
        for node in self.nodes:
            node.receive_global_params(self.global_params)

#
# Training Functinon
#
def train_fedavg(
    data: Data,
    partitions: list[np.array],
    channel: CommunicationChannel,
    epochs: int,
    local_steps: int,
    lr: float = 1e-3,
) -> tuple[list[float], WetlandGCN]:
    """
    data: Full graph data object
    partitions: List of node index arrays for each partition (e.g., [np.array([0,1,2]), np.array([3,4,5])])
    channel: The Communication channel for sending/receiving model updates between nodes and central agent
    epochs: Number of global epochs to train for
    local_steps: Number of local training steps to perform at each node before aggregation
    lr: Learning rate for local training
    """
    # Initialize central agent and node agents based on partitions
    central = CentralAgent()
    nodes = [NodeAgent(node_id=f"node_{i}") for i in range(len(partitions))]
    
    for node in nodes:
        central.register_node(node)

    # Track global loss curve
    global_loss_curve = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Each node trains locally on its partition of the data
        for i, node in enumerate(nodes):
            partition_idx = partitions[i]
            partition_data = Data(
                x=data.x[partition_idx],
                edge_index=subgraph(partition_idx, data.edge_index)[0],
                y=data.y[partition_idx]
            )
            node.train_local(partition_data, epochs=local_steps, lr=lr)

        # Central agent aggregates updates from all nodes
        updates = [node.send_local_update() for node in nodes]
        central.aggregate_updates(updates)

        # Central agent broadcasts new global parameters to all nodes
        central.broadcast_params()

        # Optionally evaluate global model on a validation set here and track loss curve

    # Return final global loss curve and trained global model (if applicable)
    return global_loss_curve, None  # Placeholder for returning trained model


#################################################################################
# Example usage (Run as script)
#################################################################################
if __name__ == "__main__":
    # Import dependent libraries
    from build_graph import build_pyg_data
    from load_data import load_gdf

    # Build the full graph from wetland data, and the goal node for the pathfinding task
    gdf = load_gdf()
    data, goal_node = build_pyg_data(gdf, grid_size=40, seed=42)

    print(f"Goal node             : {goal_node}")

    # Setup federated environment
    central = CentralAgent()
    node_a = NodeAgent("node_a")
    node_b = NodeAgent("node_b")
    central.register_node(node_a)
    central.register_node(node_b)

    print(f"Registered Nodes      : {[n.node_id for n in central.nodes]}")
    print(f"Central Agent ID      : {central.agent_id}")
    print(f"Aggregated parameters : {len(central.global_params)}\nitems                 :\n {list(central.global_params.keys())} ...")

    # Each node starts from different locations
    node_a_start = max(0, goal_node - 8)
    node_b_start = min(data.num_nodes - 1, goal_node + 8)

    print(f"Node A start          : {node_a_start}")
    print(f"Node B start          : {node_b_start}")

    # Node training commands (assuming can move up to three spaces away)
    node_a.train_local(data, location=node_a_start, radius=3, epochs=10, lr=1e-3)
    node_b.train_local(data, location=node_b_start, radius=3, epochs=10, lr=1e-3)

    print(f"Node A location       : {node_a.local_data.get('location')}")
    print(f"Node A loss           : {node_a.local_data.get('last_loss')}")
    print(f"Node B location       : {node_b.local_data.get('location')}")
    print(f"Node B loss           : {node_b.local_data.get('last_loss')}")

    # Central node collects updates, aggregates results, and broadcasts new global parameters
    updates = [node_a.send_local_update(), node_b.send_local_update()]
    central.aggregate_updates(updates)
    central.broadcast_params()

    print(f"Aggregated parameters : {len(central.global_params)}")
    print(f"items                 : {list(central.global_params.keys())}")

    # Now repeat training until the goal node is reached by either node
    while (node_a.current_loc is not goal_node) and (node_b.current_loc is not goal_node):
        # Update each node's current location
        node_a.current_loc = node_a.local_data.get("location")
        node_b.current_loc = node_b.local_data.get("location")

        # Node training
        node_a.train_local(data, radius=3, epochs=10, lr=1e-3)
        node_b.train_local(data, radius=3, epochs=10, lr=1e-3)

        print(f"Node A moved location : {node_a.local_data.get('location')}")
        print(f"Node A loss           : {node_a.local_data.get('last_loss')}")
        print(f"Node B moved location : {node_b.local_data.get('location')}")
        print(f"Node B loss           : {node_b.local_data.get('last_loss')}")

        # Central commands
        updates = [node_a.send_local_update(), node_b.send_local_update()]
        central.aggregate_updates(updates)
        central.broadcast_params()

        print(f"Aggregated parameters : {len(central.global_params)}")
        print(f"items                 : {list(central.global_params.keys())}")

    print(f"Goal node reached!")

    sys.exit(0)
