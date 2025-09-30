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
from torch_geometric.data import Data

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
        edge_distances = [] 
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
                distances = np.linalg.norm(
                    positions[node_idx] - positions[neighbor_idx]
                )

                if distances < self.connectivity_radius:
                    edges.append([node_idx, neighbor_idx])
                    edge_distances.append(distances)

        edges = np.array(edges, dtype=np.int64)
        edge_distances = np.array(edge_distances, dtype=np.float32)
            
        return edges, edge_distances

    def get_gt_connectivity(
        self: GraphFromTrajectories,
        labels: np.ndarray,
        edge_index: np.ndarray
    ) -> np.ndarray:
        """Compute ground truth connectivity.

        Parameters
        ----------
        labels : np.ndarray
            The labels of the particles in the graph, used to determine
            the ground truth connectivity.
        edge_index : np.ndarray
            The indices of the edges in the graph, representing the
            connectivity between nodes.

        Returns
        -------
        np.ndarray
            A boolean array indicating the ground truth connectivity
            between nodes. True indicates a valid connection, while
            False indicates an invalid connection.

        """

        source_particle = labels[edge_index[:, 0]] 
        target_cell = labels[edge_index[:, 1]]
        self_connections_mask = source_particle == target_cell #source target
        gt_connectivity = self_connections_mask
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

            # Convert to numpy arrays.
            node_attr = df_video[["centroid-0","centroid-1"]].to_numpy()
            node_labels = df_video["label"].to_numpy()
            frames = df_video["frame"].to_numpy()

            # Extract graph data.
            edge_index, edge_attr = self.get_connectivity(node_attr, frames)
            edge_gt = self.get_gt_connectivity(node_labels, edge_index)

            # Encapsulate extracted data in dictionary.
            graph = Data(
                x=torch.tensor(node_attr, dtype=torch.float),
                edge_index=torch.tensor(edge_index.T, dtype=torch.long),
                edge_attr=torch.tensor(edge_attr[:, None], dtype=torch.float),
                distance=torch.tensor(edge_attr[:, None], dtype=torch.float),
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

        graph = self.graph_dataset[np.random.randint(0, self.dataset_size - 1)]
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
            x = node_attr, 
            edge_index = edge_index, 
            edge_attr = graph.edge_attr[edge_mask],
            distance = graph.edge_attr[edge_mask], 
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
    """Compute trajectories from graph predictions.

    This class computes trajectories from graph predictions by pruning edges
    based on the predicted labels. The pruning is done by removing edges that
    do not connect to the source particle or exceed a specified frame
    difference. The resulting trajectories are represented as connected
    components in the pruned graph. The class uses NetworkX to compute the
    connected components and returns a list of trajectories represented as sets
    of node indices. The trajectories are computed from the graph predictions,
    which are assumed to be binary labels indicating the presence of a
    connection between particles. The trajectories are represented as a list of
    sets, where each set contains the indices of the nodes in the trajectory.
    The class can be used to analyze the motion of particles in a video or a
    sequence of frames.

    Parameters
    ----------
    graph : torch_geometric.data.Data
        The input graph object containing node features and other attributes.
    predictions : np.ndarray
        The predicted labels for the edges in the graph, used to determine the
        connectivity between nodes. The predictions are assumed to be binary
        labels indicating the presence of a connection between particles.

    Methods
    -------
    `__call__(graph, predictions)`
        Computes trajectories from the graph and predictions. The trajectories
        are represented as connected components in the pruned graph.
    `prune_edges(graph, predictions)`
        Prunes the edges of the graph based on the predicted labels. The
        pruning is done by removing edges that do not connect to the source
        particle or exceed a specified frame difference. The resulting pruned
        edges are used to compute the trajectories from the graph predictions.

    """

    def __call__(self, graph, predictions):
        """Compute trajectories."""
        pruned_edges = self.prune_edges(graph, predictions)
        pruned_graph = nx.Graph()
        pruned_graph.add_edges_from(pruned_edges)
        trajectories = list(nx.connected_components(pruned_graph))
        return trajectories

    def prune_edges(self, graph, predictions):
        """Prune edges."""
        pruned_edges = []
        frame_pairs  = np.stack(
            [graph.frames[graph.edge_index[0]], 
             graph.frames[graph.edge_index[1]]],
             axis=1
        )
        
        # Find edges connected to the source particle and prune
        # if they exceed the frame difference.
        for source_particle in np.unique(graph.edge_index[0]): 
            source_particle_mask = graph.edge_index[0] == source_particle
            target_cell_candidates = predictions[source_particle_mask] == True
            if np.any(target_cell_candidates):
                frame_diff = (frame_pairs[source_particle_mask, 1] -
                              frame_pairs[source_particle_mask, 0])
                
                min_frame_diff = frame_diff[target_cell_candidates].min()
                target_cell_mask = (target_cell_candidates 
                                 & (frame_diff == min_frame_diff))
                
                edge = graph.edge_index[:,
                source_particle_mask][:, target_cell_mask]
                edge = edge.reshape(-1, 2)
                if len(edge) == 1:
                    pruned_edges.append(tuple(*edge.numpy()))
        return pruned_edges

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
