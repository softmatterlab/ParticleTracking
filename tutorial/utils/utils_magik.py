"""Utility module for MAGIK.

This module provides utility classes and functions to prepare and process
graph-based representations of particle motion, particularly for use with the
MAGIK tracking pipeline.

Key Features
------------
- Graph creation from particle trajectories.

- Dataset generation and augmentation for training.

- Trajectory reconstruction from predicted graph edges.

Module Structure
-----------------
Classes:

- `GraphFromTrajectories`: Creates a Graph representation of the motion of 
    particles.

- `GraphDataset`: Makes a dataset in torch-format for training. 

- `RandomRotation`: Rotates graph features, used in training.

- `RandomFlip`: Flips graph features, used in training.

- `NodeDropout`: Randomly removes nodes during training.

- `ComputeTrajectories`: Calculates trajectories from MAGIK output.

Functions:

- `make_list`: Converts MAGIK trajectories from graph format to a list of 
    NumPy arrays.

"""

from __future__ import annotations

from math import pi, sin, cos
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from deeplay import DeeplayModule


class GraphFromTrajectories:
    """Graph representation of the motion of particles.
    
    This class creates a graph representation of a set of trajectories, 
    where each node corresponds to a particle in a frame, and edges represent 
    the connectivity between particles in consecutive frames. The connectivity 
    is determined based on a specified radius and maximum frame distance. The 
    class also computes the ground truth connectivity based on the labels of 
    the  particles. The graph is represented using PyTorch Geometric's Data 
    class, which allows for efficient storage and manipulation of graph data.

    Parameters
    ----------
    connectivity_radius : float
        The radius within which particles are considered connected.
    max_frame_distance : int
        The maximum number of frames between two connected particles.

    Methods
    -------
    `__init__(connectivity_radius, max_frame_distance)`
        Initializes the graph from trajectories with the specified
        connectivity radius and maximum frame distance.
    `get_connectivity(node_attr, frames)`
        Computes the connectivity of the graph based on the node attributes
        and frame indices.
    `get_gt_connectivity(labels, edge_index)`
        Computes the ground truth connectivity based on the labels of the
        particles and the edge indices.
    `__call__(df)`
        Computes graphs from videos by extracting node attributes, edge
        indices, edge attributes, distances, frames, and ground truth labels
        from the input DataFrame.

    """

    def __init__(
        self: GraphFromTrajectories,
        connectivity_radius: float,
        max_frame_distance: int,
    ) -> None:
        """Initialize the graph from trajectories.

        Parameters
        ----------
        connectivity_radius : float
            The radius within which particles are considered connected.
        max_frame_distance : int
            The maximum number of frames between two connected particles.

        """

        self.connectivity_radius = connectivity_radius
        self.max_frame_distance = max_frame_distance

    def get_connectivity(
        self: GraphFromTrajectories,
        positions: np.ndarray,
        frame_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute connectivity of the graph.

        Parameters
        ----------
        positions : np.ndarray
            The attributes of the nodes in the graph, typically the coordinates
            of the particles.
        frame_indices : np.ndarray
            The frame indices corresponding to the nodes in the graph.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the edge indices and edge attributes of the 
            graph. The edge indices represent the connectivity between nodes, 
            and the edge attributes represent the distances between connected 
            nodes.

        """

        edges = []          
        edge_features = [] 
        num_nodes = len(positions)

        for node_idx in range(num_nodes):
            node_frame = frame_indices[node_idx]

            for neighbor_idx in range(node_idx + 1, num_nodes):
                neighbor_frame = frame_indices[neighbor_idx]
                frame_gap = neighbor_frame - node_frame

                if frame_gap <= 0:
                    continue
                if frame_gap > self.max_frame_distance:
                    break
                distance = np.linalg.norm(
                    positions[node_idx] - positions[neighbor_idx]
                )

                if distance < self.connectivity_radius:
                    edges.append([node_idx, neighbor_idx])
                    edge_features.append([
                        distance,
                        frame_gap / self.max_frame_distance,
                        (distance**2)/(frame_gap / self.max_frame_distance)
                    ])

        edges = np.array(edges, dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32)
            
        return edges, edge_features

    def get_gt_connectivity(
        self: "GraphFromTrajectories",
        labels: np.ndarray,
        edge_index: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute ground-truth connectivity allowing only the shortest
        valid temporal connection between consecutive localizations
        of the same particle.

        Parameters
        ----------
        labels : np.ndarray
            Array of particle identifiers for each node.
        edge_index : np.ndarray
            Array of shape (N_edges, 2) containing source and target node indices.
        times : np.ndarray
            Array of timestamps (or frame indices) for each node.

        Returns
        -------
        np.ndarray
            Boolean array of shape (N_edges,) indicating ground-truth
            connectivity. True for valid connections (same particle, minimal Δt),
            False otherwise.
        """

        # Extract source/target info
        src, tgt = edge_index[:, 0], edge_index[:, 1]
        src_label, tgt_label = labels[src], labels[tgt]
        src_time, tgt_time = times[src], times[tgt]

        # Candidate same-particle links
        same_particle_mask = src_label == tgt_label
        gt_connectivity = np.zeros_like(same_particle_mask, dtype=bool)

        # For each particle, find its consecutive detections
        unique_particles = np.unique(labels)
        for pid in unique_particles:
            # indices of detections for this particle, sorted by time
            particle_nodes = np.where(labels == pid)[0]
            particle_times = times[particle_nodes]
            sorted_idx = np.argsort(particle_times)
            sorted_nodes = particle_nodes[sorted_idx]
            sorted_times = particle_times[sorted_idx]

            # Compute minimal valid links (successive detections)
            if len(sorted_nodes) < 2:
                continue  # no link possible

            # Build all consecutive node pairs (i, i+1)
            consecutive_pairs = set(zip(sorted_nodes[:-1], sorted_nodes[1:]))

            # Mark as True only those edges that match one of these pairs
            for i, (s, t) in enumerate(zip(src, tgt)):
                if (s, t) in consecutive_pairs:
                    gt_connectivity[i] = True

        return gt_connectivity

    def __call__(
        self: GraphFromTrajectories,
        df: pd.DataFrame
    ) -> list[Data]:
        """Compute graphs from videos.

        This method generates a list of graph objects from a DataFrame of 
        particle detections. Each unique video ("set") in the DataFrame is 
        converted into a PyTorch Geometric Data object, with nodes 
        representing detections and edges representing likely temporal 
        connections based on distance and frame difference criteria.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least the columns:
            ["set", "frame", "centroid-0", "centroid-1", "label"].

        Returns
        -------
        list[Data]
            A list of torch_geometric.data.Data objects, each representing a
            graph for a video. Each graph contains:
                - x : node coordinates
                - edge_index : edge connectivity
                - edge_attr : pairwise distances
                - frames : frame indices
                - y : ground truth edge labels (connectivity)

        """

        graph_dataset = []

        videos = df["set"].unique()

        # Each set is a video, compute graphs from one video at a time.
        for current_video in videos:

            # Get a video from the dataset.
            df_video = df[df["set"] == current_video]
            df_video = df_video.sort_values(by=["frame"]).reset_index(drop=True)

            # Convert to numpy arrays.
            positions = df_video[["centroid-0","centroid-1"]].to_numpy()
            node_labels = df_video["label"].to_numpy()
            frames = df_video["frame"].to_numpy()

            # Extract graph data.
            edge_index, edge_attr = self.get_connectivity(positions, frames)
            edge_gt = self.get_gt_connectivity(node_labels, edge_index, frames)

            # Encapsulate extracted data in dictionary.
            graph = Data(
                x=torch.tensor(positions, dtype=torch.float),
                edge_index=torch.tensor(edge_index.T, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                distance=torch.tensor(edge_attr[:, 0:1], dtype=torch.float),
                frames=torch.tensor(frames, dtype=torch.float),
                y=torch.tensor(edge_gt[:, None], dtype=torch.float),
            )
            graph_dataset.append(graph)

        return graph_dataset 


class GraphDataset(torch.utils.data.Dataset):
    """GraphDataset class for training.

    This class is a PyTorch Dataset that creates a dataset of graphs for
    training. It takes a list of graphs and generates a dataset of subgraphs by
    sampling frames and edges. The dataset is designed to be used with
    PyTorch's DataLoader for efficient batch processing during training. The
    dataset can also apply transformations to the graphs during training, such
    as random rotations and flips to augment the training data.

    Parameters
    ----------
    graph_dataset : list
        list of graphs, each represented as a PyTorch Geometric Data object.
    Dt : int
        The time difference between frames to sample from the graph.
    dataset_size : int
        The size of the dataset, i.e., the number of subgraphs to sample.
    transform : callable, optional
        A function or transform to apply to each graph in the dataset.
        Default is None.

    Methods
    -------
    `__len__()`
        Returns the size of the dataset.
    `__getitem__(idx)`
        Returns a subgraph sampled from the dataset at the specified index.
    `__init__(graph_dataset, Dt, dataset_size, transform=None)`
        Initializes the dataset with the provided graph dataset, time
        difference, dataset size, and optional transform.
    `__call__(graph, predictions)`
        Computes trajectories from the graph and predictions.    
    
    """

    def __init__(
        self: GraphDataset,
        graph_dataset: list,
        Dt: int,
        dataset_size: int, 
        transform: callable = None,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        graph_dataset : list
            List of graphs, each represented as a PyTorch Geometric Data
            object.
        Dt : int
            The time difference between frames to sample from the graph.
        dataset_size : int
            The size of the dataset, i.e., the number of subgraphs to sample.
        transform : callable, optional
            A function or transform to apply to each graph in the dataset.
            Default is None.

        """

        self.graph_dataset = graph_dataset
        self.dataset_size = dataset_size
        self.Dt = Dt
        self.transform = transform

    def __len__(
        self: GraphDataset,
    ) -> int:
        """Obtain length of dataset.

        Returns
        -------
        int
            The size of the dataset, i.e., the number of subgraphs.

        """

        return self.dataset_size

    def __getitem__(
        self: GraphDataset,
        idx: int,
    ) -> Data:
        """Obtain a subgraph from the dataset.

        This method samples a random time window (of length Dt) from a 
        randomly chosen graph in the dataset. Only nodes and edges that fall 
        within the selected frame window are kept. Edge indices are reindexed 
        to ensure consistency.

        Parameters
        ----------
        idx : int
            Ignored index; sampling is random within the dataset.

        Returns
        -------
        Data
            A PyTorch Geometric Data object containing the subgraph:
                - x : node features
                - edge_index : reindexed edges
                - edge_attr : distances
                - y : ground truth labels
        
        """

        graph = self.graph_dataset[np.random.randint(0, len(self.graph_dataset) - 1)]
        frames, edge_index = graph.frames, graph.edge_index
        select_frame = np.random.randint(self.Dt, frames.max() + 1)

        start_frame = select_frame - self.Dt
        node_mask = (frames >= start_frame) & (frames < select_frame)
        node_attr = graph.x

        frame_pairs = torch.stack(
            [
                frames[edge_index[0, :]], 
                frames[edge_index[1, :]]
            ],
            axis=-1,
        )
        edge_mask = (frame_pairs >= start_frame) & (frame_pairs < select_frame)
        edge_mask = edge_mask.all(axis=-1) 

        edge_index = edge_index[:, edge_mask] - edge_index[:, edge_mask].min()

        return_graph = Data(
            x = node_attr[node_mask], 
            edge_index = edge_index, 
            edge_attr = graph.edge_attr[edge_mask],
            distance = graph.edge_attr[edge_mask,0:1], 
            y = graph.y[edge_mask],  
        )
        if self.transform: return_graph = self.transform(return_graph)
        return return_graph

class RandomRotation:
    """Random rotation to augment training data.
    
    This class applies a random rotation to the node features of a graph to
    augment the training data. The rotation is performed in the 2D plane, and
    the angle of rotation is randomly sampled from a uniform distribution. The
    rotation is applied to the x and y coordinates of the node features, which
    are assumed to be in the first two columns of the node feature matrix. The
    rotation is performed in place, and the modified graph is returned. The
    rotation is centered around the origin (0, 0) and the node features are
    restored to their original positions after the rotation.
    
    Parameters
    ----------
    graph : torch_geometric.data.Data
        The input graph object containing node features and other
        attributes.
        
    Methods
    -------
    `__call__(graph)`
        Performs the random rotation on the input graph.
    
    """

    def __call__(
        self: RandomRotation,
        graph: "torch_geometric.data.Data",
    ) -> "torch_geometric.data.Data":
        """Perform the random rotation.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            The input graph object containing node features and other
            attributes.
            
        Returns
        -------
        torch_geometric.data.Data
            The modified graph object with rotated node features.
        
        """

        graph = graph.clone()
        node_feats = graph.x[:, :2] - 0.5  # Centered positons
        angle = np.random.rand() * 2 * pi
        rotation_matrix = torch.tensor(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        ).float()
        rotated_node_attr = torch.matmul(node_feats, rotation_matrix)
        graph.x[:, :2] = rotated_node_attr + 0.5  # Restored positons
        return graph

class RandomFlip:
    """Random flip to augment training data.
    
    This class applies a random flip to the node features of a graph to 
    augment the training data. The flip is performed in the 2D plane, and the 
    flip is applied to the x and y coordinates of the node features, which are 
    assumed to be in the first two columns of the node feature matrix. The 
    flip is performed in place, and the modified graph is returned. The flip 
    is centered around the origin (0, 0) and the node features are restored to 
    their original positions after the flip.

    Methods
    -------
    `__call__(graph)`
        Performs the random flip on the input graph.
    
    """

    def __call__(
        self: RandomFlip,
        graph: "torch_geometric.data.Data",
    ) -> "torch_geometric.data.Data":
        """Perform the random flip.
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            The input graph object containing node features and other
            attributes.
        
        Returns
        -------
        torch_geometric.data.Data
            The modified graph object with flipped node features.

        """

        graph = graph.clone()
        node_feats = graph.x[:, :2] - 0.5  # Centered positons
        if np.random.randint(2): node_feats[:, 0] *= -1
        if np.random.randint(2): node_feats[:, 1] *= -1
        graph.x[:, :2] = node_feats + 0.5  # Restored positons
        return graph

class NodeDropout:
    """Removal (dropout) of random nodes to simulate missing frames.

    This class randomly removes nodes from a graph to simulate missing frames.
    The dropout is performed by randomly selecting a subset of nodes to remove
    based on a specified dropout rate. The edges, weights, labels, and
    distances connected to the removed nodes are also removed. The modified
    graph is returned with the remaining nodes and edges. The dropout is
    performed in place, and the original graph is unchanged. The dropout rate
    is specified as a parameter, and the random selection of nodes to remove is
    performed using a uniform distribution. The removed nodes are not restored
    to their original positions, and the modified graph is returned with the
    remaining nodes and edges.
  
    Parameters
    ----------
    graph : torch_geometric.data.Data
        The input graph object containing node features and other
        attributes.
  
    Methods
    -------
    `__call__(graph)`
        Performs the node dropout on the input graph.

    """

    def __call__(
        self: NodeDropout, 
        graph: "torch_geometric.data.Data",
    ) -> "torch_geometric.data.Data":
        """Perform the node dropout.

        Parameters
        ----------
        graph : torch_geometric.data.Data
            The input graph object containing node features and other
            attributes.
    
        Returns
        -------
        torch_geometric.data.Data
            The modified graph object with the remaining nodes and edges after
            the dropout.

        """

        # Ensure original graph is unchanged.
        graph = graph.clone()

        # Specify node dropout rate.
        dropout_rate = 0.05

        # Get indices of random nodes.
        idx = np.array(list(range(len(graph.x))))
        dropped_idx = idx[np.random.rand(len(graph.x)) < dropout_rate]

        # Compute connectivity matrix to dropped nodes.
        for dropped_node in dropped_idx:
            edges_connected_to_removed_node = np.any(
                np.array(graph.edge_index) == dropped_node, axis=0
            )

        # Remove edges, weights, labels connected to dropped nodes with the
        # bitwise not operator '~'.
        graph.edge_index = \
            graph.edge_index[:, ~edges_connected_to_removed_node]
        graph.edge_attr = graph.edge_attr[~edges_connected_to_removed_node]
        graph.distance = graph.distance[~edges_connected_to_removed_node]
        graph.y = graph.y[~edges_connected_to_removed_node]

        return graph


class ComputeTrajectories:
    """
    Reconstruct trajectories from Sinkhorn-based edge predictions.

    The Sinkhorn head already enforces near one-to-one assignments
    between sources and targets. This class thresholds low-probability
    links and builds connected components as trajectories.
    """

    def __init__(self, p_min: float = 0.5):
        self.p_min = p_min

    def __call__(self, graph, predictions):
        """
        Parameters
        ----------
        graph : object
            Graph-like structure with attributes such as 'frames'.
        predictions : tuple
            Output of the Sinkhorn head:
              (probs_real, edge_index, probs_all, all_src, dummy_mask)

        Returns
        -------
        list[set[int]]
            List of trajectories, each as a set of node indices.
        """
        (
            probs_real,
            edge_index,
            probs_all,
            all_src,
            dummy_mask,
        ) = predictions

        # Move to CPU numpy
        edge_index = edge_index.detach().cpu().numpy()
        probs_real = probs_real.detach().cpu().numpy().squeeze()

        # Select edges above probability threshold
        valid = probs_real >= self.p_min
        edges = edge_index[:, valid].T
        scores = probs_real[valid]

        # Optional: enforce at most one incoming edge per target
        order = np.argsort(-scores)
        used_targets = set()
        pruned_edges = []
        for i in order:
            s, t = edges[i]
            if t not in used_targets:
                pruned_edges.append((int(s), int(t)))
                used_targets.add(t)

        # Build graph from pruned edges
        pruned_graph = nx.Graph()
        pruned_graph.add_edges_from(pruned_edges)

        # Extract connected components as trajectories
        trajectories = list(nx.connected_components(pruned_graph))
        return trajectories


def make_list(
    trajs_from_graph: list[int],
    test_graph: "torch_geometric.data.Data",
    fov_size: float,
) -> list[np.ndarray]:
    """Convert MAGIK trajectories from graph format to a list of NumPy arrays.

    This function takes a list of trajectories represented as node indices
    in a graph and converts them into a list of NumPy arrays. Each array
    represents a trajectory in the format [frame, y, x]. The function
    extracts the frame indices and coordinates from the graph object and
    concatenates them into a single array for each trajectory. The
    resulting list contains arrays of shape (T, 3), where T is the number
    of frames in the trajectory.

    Parameters
    ----------
    trajs_from_graph : list of list[int]
        list of trajectories, where each trajectory is a list of node indices 
        (as returned by ComputeTrajectories).
    test_graph : torch_geometric.data.Data
        The graph object used in prediction, containing:
            - .frames: frame index for each node
            - .x: position (normalized [0,1]) for each node
    fov_size : float
        Field of view size in pixels. Multiplies normalized coordinates to get 
        real positions.

    Returns
    -------
    list of np.ndarray
        Each array is of shape (T, 3), with columns [frame, y, x], sorted by 
        frame.

    """

    trajs_list = []
    for t in trajs_from_graph:
        frames = test_graph.frames[list(t)].cpu().numpy()
        coords = (test_graph.x[list(t)].cpu().numpy()
                  * fov_size)  # Shape (T, 2), assumed [x, y]
        # Flip to [y, x] and concatenate with frames.
        traj = np.column_stack((frames, coords[:, 0], coords[:, 1]))
        # Optionally sort by frame if not ordered.
        traj = traj[np.argsort(traj[:, 0])]
        trajs_list.append(traj)
    return trajs_list


class SoftmaxHead(DeeplayModule):
    """
    Deterministic per-source softmax head with learnable per-node dummy logits.

    Output:
      probs_real : [N_edges, 1]
      edge_index : [2, N_edges]
      probs_all  : [N_edges + N_nodes]
      all_src    : [N_edges + N_nodes]
      dummy_mask : [N_edges + N_nodes]
      node_attr  : passed-through node attributes (for losses using coordinates)
    """

    def __init__(self, base_head, eps=1e-8):
        super().__init__()
        self.base_head = base_head
        self.eps = eps

        # will be lazy-initialized based on node_attr dimension
        self.dummy_mlp = None

        # optional: bias for real edges initialized negative
        self.init_real_bias = -2.0
        self.init_dummy_bias = 2.0

    def _lazy_init_dummy(self, node_attr):
        """Create dummy_mlp once we know the dimensionality of node attributes."""
        if self.dummy_mlp is not None:
            return

        dim = node_attr.shape[1]
        device = node_attr.device

        self.dummy_mlp = nn.Linear(dim, 1, bias=True).to(device)

        # Initialize: dummy should win early in training
        nn.init.constant_(self.dummy_mlp.weight, 0.0)
        nn.init.constant_(self.dummy_mlp.bias, self.init_dummy_bias)

        # Also bias the last layer of the base head (if accessible)
        if hasattr(self.base_head, "weight"):
            # it's a Linear layer
            nn.init.constant_(self.base_head.bias, self.init_real_bias)

    def forward(self, inputs):
        edge_attr, edge_index, node_attr = inputs
        src = edge_index[0]
        num_nodes = node_attr.shape[0]
        device = src.device

        # lazy create dummy MLP after dimension known
        self._lazy_init_dummy(node_attr)

        # ---- Real-edge logits ----
        logits = self.base_head(edge_attr).view(-1)  # [N_edges]
        logits = logits.clamp(-10, 10)

        # ---- Dummy logits ----
        dummy_logits = self.dummy_mlp(node_attr).view(-1)  # [N_nodes]

        # ---- Combine all ----
        all_logits = torch.cat([logits, dummy_logits], dim=0)  # [E + N]
        all_src = torch.cat([src, torch.arange(num_nodes, device=device)], dim=0)
        dummy_mask = torch.arange(all_logits.numel(), device=device) >= logits.numel()

        # ---- Per-source softmax ----

        # Scatter-reduce "amax" is not supported on MPS → manual segment max
        max_per_src = torch.full((num_nodes,), float("-inf"), device=src.device)

        # Compute maximum logit per source
        # For each entry i: max_per_src[all_src[i]] = max(old, all_logits[i])
        max_per_src = max_per_src.index_put_(
            (all_src,),
            torch.maximum(max_per_src[all_src], all_logits),
            accumulate=False
        )

        # Exponentiate shifted logits (numerically stable)
        exp_logits = torch.exp(all_logits - max_per_src[all_src])

        # Per-source sum
        sum_per_src = torch.zeros(num_nodes, device=src.device)
        sum_per_src.index_add_(0, all_src, exp_logits)

        # Softmax
        probs_all = exp_logits / (sum_per_src[all_src] + self.eps)
        probs_all = probs_all.clamp(min=self.eps, max=1.0)

        # real-edge probabilities for training and metrics
        probs_real = probs_all[~dummy_mask].unsqueeze(-1)   # [N_edges, 1]


        # ---- Extract real-edge probs ----
        probs_real = probs_all[~dummy_mask].unsqueeze(-1)  # [N_edges, 1]

        return (probs_real, edge_index, probs_all, all_src, dummy_mask)


class NodewiseNLLLoss(nn.Module):
    """
    Proper categorical cross-entropy for the SoftmaxHead.

    Input:
      y_hat = (probs_real, edge_index, probs_all, all_src, dummy_mask)
      y     = [N_edges] binary vector for *real* edges only

    Builds correct per-source targets including dummy, and computes:

        L = - Σ_i log p(correct_edge_i)

    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, y_hat, y):
        probs_real, edge_index, probs_all, all_src, dummy_mask = y_hat
        src = edge_index[0]
        device = probs_all.device

        # Prepare counts
        num_nodes = int(all_src.max().item()) + 1

        # Ground truth for real edges only: [N_edges]
        y = y.view(-1).float()

        # Build full y_all including dummy labels
        y_all = torch.zeros_like(probs_all)

        # Real edges: copy ground truth
        y_all[~dummy_mask] = y

        # Identify which source nodes have a real target (positive)
        has_real = torch.zeros(num_nodes, device=device)
        has_real.index_add_(0, src, y)
        has_real = has_real > 0

        # Dummy edges are positive only when the node has *no* real outgoing edges
        dummy_src = all_src[dummy_mask]
        y_all[dummy_mask] = (~has_real[dummy_src]).float()

        # This ensures exactly one positive per source
        # Now compute categorical cross-entropy over each source group.
        logp = torch.log(probs_all.clamp(self.eps, 1.0))

        # Reduce -log p(correct) grouped by sources
        loss_per_src = torch.zeros(num_nodes, device=device)

        pos_src = all_src[y_all > 0.5]
        pos_logp = logp[y_all > 0.5]

        loss_per_src.index_add_(0, pos_src, -pos_logp)

        # Normalize over sources that actually have predictions
        unique_sources = torch.unique(all_src)
        return loss_per_src[unique_sources].mean()
