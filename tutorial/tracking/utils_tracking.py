"""This utility file provides functions and classes for tracking.

Key Features
------------

- **Brownian Motion Simulation**

        Simulates trajectories with Brownian characteristics,
        this is used to test the performance of LapTrack, TrackPy,
        and MAGIK.

- **Trajectory Utilities**

        Functions which utilize the Hungarian algorithm to pair a
        trajectory to its corresponding label, allowing us to track
        it in time, as well as functions to evaluate and process
        trajectories.
        
- **Classes For Graph Processing**

        In order to use MAGIK, graph representations of trajectories need 
        to be generated, as well as graph augmentations to diversify training
        data such as random rotations or flips, as well as a utility class to
        construct trajectories from the output of MAGIK.

Module Structure
----------------

Methods:

- `simulate_Brownian_trajs`: Simulates Brownian motion of particles.

- `transform_to_video`: Generates video from trajectories with DeepTrack2.

- `generate_centroids`: Initializes positions for centroids randomly.

- `traj_break`: Breaks trajectories coming in and out of the FOV.

- `play_video`: Displays a stack of images as a video.

- `convert_uint8`: Converts data to uint8 format.

- `format_image`: Converts image to format (N, C, X, Y).

- `make_video_with_trajs`: Generates video with trajectories.

- `trajectory_sqdistance`: Pairs ground truth and trajectories with squared distance.

Classes:

- `GraphFromTrajectories`: Creates a graph representation of a video.

- `GraphDataset`: Makes a dataset in torch-format for training. 

- `RandomRotation`: Rotates graph features, used in training.

- `RandomFlip`: Flips graph features, used in training.

- `NodeDropout`: Randomly removes nodes during training.

- `ComputeTrajectories`: Calculates trajectories from MAGIK output.



"""
import os
from math import cos, pi, sin

import cv2
import deeptrack as dt
import imageio
import matplotlib as mpl
import matplotlib.lines as mlines
import networkx as nx
import numpy as np
import torch

from torch_geometric.data import Data
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import linear_sum_assignment
from skimage import measure

mpl.rcParams["animation.embed_limit"] = 60

def simulate_Brownian_trajs(
        num_particles,
        num_timesteps,
        R,
        l,
):
    """Generates trajectories of particles exhibiting Brownian motion.

    Parameters
    ----------
    num_particles: int
        Number of particles you wish to simulate.
    num_timesteps: int
        How long to simulate for in number of timesteps.
    R: int
        The radius of each particle in pixels.
    l: int
        The width/height of the simulation box, set to be a square
        for simplicity.

    Returns
    -------
    np.ndarray
        A numpy array containing the trajectories of each particle
        with shape (n_timesteps, num_particles, 3), this gives the 
        x, y, t coordinates of a given particle in a given timeframe.

    """

    # Get initial positions of particles.
    start_positions = generate_centroids(
        num_particles=num_particles,
        box_size=2*l, 
        R=R,
    )
    pos = np.empty([num_particles,2])
    trajs_gt = np.empty([num_timesteps, num_particles, 2 + 1])

    # Add particle positions to array.
    pos[:, 0] = start_positions[:, 0] % (2 * l) - l
    pos[:, 1] = start_positions[:, 1] % (2 * l) - l

    # Add particle positions to array.
    for j in range(num_particles):
        trajs_gt[0, j, 0] = pos[j, 0] + l
        trajs_gt[0, j, 1] = pos[j, 1] + l
        trajs_gt[0, j, 2] = 0

    # Time simulation.
    for t in range(num_timesteps - 1):
      
        # Perform random step.
        pos[:, 0] = (pos[:, 0] +
                     np.random.normal(0,1,num_particles) + l) % (2 * l) - l
        pos[:, 1] = (pos[:, 1] +
                     np.random.normal(0,1,num_particles) + l) % (2 * l) - l

        # Add particle positions to array.
        for particle_idx in range(num_particles):
            trajs_gt[t + 1, particle_idx, 0] = pos[particle_idx, 0] + l
            trajs_gt[t + 1, particle_idx, 1] = pos[particle_idx, 1] + l
            trajs_gt[t + 1, particle_idx, 2] = t + 1
    return trajs_gt

def transform_to_video(
    trajectory_data,
    particle_props={},
    optics_props={},
    background_props={},
    image_size = [],
    save_video=False,
    maps = True,
    path="",

):
    _particle_dict = {
        "z": 0, # For particles out of focus, give z a nonzero value.
        "position_unit": "pixel",
    }
        
    _optics_dict = {
        "NA": 1.4,  # Numerical aperture.
        "wavelength": 633. * dt.units.nm, 
        "refractive_index_medium": 1.33,
        "output_region": [0, 0, image_size, image_size],
        "magnification": 1,
        "resolution": 50 * dt.units.nm,
    }

    # Background offset.
    _background_dict = {
        "background_mean": 85,  # Mean background intensity.
        "background_std": 0,  # Standard deviation of background intensity.
    }
    
    # Update the dictionaries with the user-defined values.
    _particle_dict.update(particle_props)
    _optics_dict.update(optics_props)
    _background_dict.update(background_props)


    # Reshape the trajectory.
    trajectory_data = np.moveaxis(trajectory_data, 0, 1)

    inner_sphere = dt.Sphere(
        trajectories=trajectory_data,
        replicate_index=lambda _ID: _ID,
        trajectory=lambda replicate_index, trajectories: dt.units.pixel
        * trajectories[replicate_index[-1]],
        number_of_particles=trajectory_data.shape[0],
        traj_length=trajectory_data.shape[1],
        position=lambda trajectory: trajectory[0],
        radius= lambda: np.random.uniform(40e-9, 120e-9),
        intensity= 1000,  # Change the intensities here
        **_particle_dict,
)

    # Make it sequential.
    sequential_particle = dt.Sequential(
        inner_sphere,
        position=lambda trajectory, sequence_step: trajectory[sequence_step],
    )
    background = dt.Add(
        value=80
    )  
    def background_variation(previous_values, previous_value):
        return (previous_values or [previous_value])[0]\
                + np.random.randn() * _background_dict["background_std"]

    sequential_background = dt.Sequential(background, 
                                value=background_variation)


    # Define optical setup.
    optics = dt.Fluorescence(**_optics_dict) 


    scale_factor = (
        (optics.magnification() * optics.wavelength() /
         (optics.NA() * optics.resolution())) ** 2) * (1 / np.pi)

    # The only place where a linebreak does not work for some reason...
    sample = (
        optics((sequential_particle) ^ sequential_particle.number_of_particles)         
        )   >> dt.Divide(scale_factor) >> sequential_background \
            >> dt.Gaussian(0,0.02) >> dt.NormalizeMinMax(0,1)
        
    # Sequential sample.
    sequential_sample = dt.Sequence(
        sample,
        trajectory=inner_sphere.trajectories,
        sequence_length=inner_sphere.traj_length,
    )

    # Resolve the sample.
    _video = sequential_sample.update().resolve()

    final_output = _video

    return final_output



def generate_centroids(
        num_particles,
        box_size,
        R
):
    """Generates non-overlapping particle positions in a 2D image. 
    Relies on random number generation.

    Parameters
    ----------
    box_size: int
        The width and height of the square image in pixels.

    num_particles: int
        The number of particles to place in the image.

    R: int
        The radius of each particle in pixels.

    Returns
    -------
    np.ndarray
        An array of shape (num_particles, 3), where each row contains the
        (x, y, theta) coordinates of a particle's center and orientation angle.

    """
    particles = []
    min_distance = 2 * R
    max_attempts = 10000  # Maximum attempts to place particle without overlap.

    attempts = 0

    while len(particles) < num_particles and attempts < max_attempts:
        # Randomly generate a new particle position within the valid range.
        x = np.random.uniform(R, box_size - R)
        y = np.random.uniform(R, box_size - R)
        new_particle = np.array([x, y])

        # Check if this new particle overlaps with any existing particles.
        if all(
            np.linalg.norm(new_particle - p) >= min_distance for p in particles
        ):
            particles.append(new_particle)

        attempts += 1

        if attempts == max_attempts:
            num_particles += -1
            attempts = 0
            
            
    # Assign an angle orientation after all particles have been created.
    # Create a third column with random angles.
    angle_column = np.random.uniform(-1.0, 1.0, len(particles)) * np.pi

    # Reshape z_column to be 2D.
    angle_column = angle_column.reshape(-1, 1)

    # Append the Z column to the pos_array.
    particles = np.hstack((particles, angle_column))
        
    return np.array(particles)


def traj_break(
        trajs_gt,
        _boxLength,
        n
):
    """Breaks trajectories

    The ground truth trajectories can exceed the FOV, this function will
    thus break the trajectories exiting/entering the FOV and make a list of the 
    trajectories which are only in the FOV.

    Parameters
    ----------
    trajs_gt: np.ndarray
        Ground truth trajectories.

    _boxLength: int
        Half of the FOV.

    n: int
        Number of particles.
    
    Returns
    ----------
    trajs_gt_list: 
        A list of trajectories that has entered/exited the FOV.

    """
    trajs_gt_list = []
    trajs_gt_n = trajs_gt[:, :, [2, 0, 1]] #swap axes, frame first
    for j in range(n):
        dx = np.abs(trajs_gt[1:, j, 0] - trajs_gt[:-1, j, 0])
        dy = np.abs(trajs_gt[1:, j, 1] - trajs_gt[:-1, j, 1])

        ind = np.where((dx > 1.5 * _boxLength) | (dy > 1.5 * _boxLength))[0]
        ind = list(np.unique((-1, len(dx) + 1, *ind)))

        for k in range(len(ind) - 1):
            trajs_gt_list.append(trajs_gt_n[ind[k] + 1:ind[k + 1], j, :])
    return trajs_gt_list


def play_video(
    video,
    video_name="movie",
    figsize=(5, 5),
    fps=10    
):
    """Displays a stack of images as a video inside jupyter notebooks.

    Parameters
    ----------
    video: np.ndarray
        Stack of images.

    figsize: tuple, optional
        Canvas size of the video.

    fps: int, optional
        Video frame rate.

    Returns
    -------
    None

        Instances a HTML video player with a movie.

    """
    fig = plt.figure(figsize=figsize)
    images = []
    plt.axis("off")
    plt.title(video_name, fontsize=14)

    for image in video:
        images.append([plt.imshow(image[:, :, 0], cmap="gray")])

    anim = animation.ArtistAnimation(
        fig, images, interval=1e3 / fps, blit=True, repeat_delay=0
    )

    html = HTML(anim.to_jshtml())
    display(html)
    plt.close()


def convert_uint8(movie):
    """Converts movie to uint8 format.

    Parameters
    ----------
    movie: np.ndarray 
        Movie to be converted.
        
    Returns
    -------

    converted_movie: np.ndarray
        Movie in uint8 format.

    """
    converted_movie = []
    for idx_im, im in enumerate(movie):
        im = im[:, :, 0]
        im = im / im.max()
        im = im * 255
        im = im.astype(np.uint8)
        converted_movie.append(im)
    return converted_movie

def format_image(img):
    """Expands and formats image to (N, C, X, Y), needed for LodeSTAR.

    Parameters
    ----------
    img: np.ndarray 
        Image to be converted.
        
    Returns
    -------

    np.ndarray
        image in (N, C, X, Y) format.

    """
    missing_array_dimensions = 4 - len(img.shape)
    for dim in range(missing_array_dimensions):
        img = np.expand_dims(img, axis=0)

    img = np.array(img).astype(np.float32)
    img = normalize_min_max(img)
    img = torch.from_numpy(img)

    formatted_img = np.transpose(img, (0, 3, 1, 2))
    return formatted_img


def make_video_with_trajs(
        trajs_list,
        trajs_gt_list, 
        sim_movie,
        _boxLength
):
    """Generates video with linked trajectories and ground truth.

    Overlays them with the simulated video.

    Parameters
    ----------
    trajs_list: list
        List of obtained trajectories in the FOV.

    trajs_gt_list: list
        List of ground truth trajectories in the FOV.

    sim_movie: np.ndarray 
        Simulated movie to overlay with.


    Returns
    ----------
    IPython.core.display.HTML
        HTML player with video generated from trajectories,
        Overlays movie with markers and trajectory lines for 
        ground truth and predictions.

    """
    fig, ax = plt.subplots(figsize=(5, 5))
    def update(frame):
      
      # Display the current frame.
      display_frame = sim_movie[frame]
      ax.clear()
      ax.imshow(display_frame, cmap="gray")
      ax.set_xlim([0, 2 * _boxLength])
      ax.set_ylim([0, 2 * _boxLength])


      # Plot prediction trajectories.
      for t in trajs_list:
          t = t[np.where(t[:, 0] <= frame)]
          if len(t) > 0:
              ax.plot(
                  t[:, 2],
                  t[:, 1],
                  color="w",
                  linewidth=0.5,
              )
              ax.scatter(
                  t[-1, 2],
                  t[-1, 1],
                  marker="o",
                  linewidths=1,
                  edgecolors="r",
                  s=100,
                  facecolors="none",
              )
      # Plot ground truth trajectories.
      for t in trajs_gt_list:

          # Filter points up to the current frame.
          t = t[np.where(t[:, 0] <= frame)]
          if len(t) > 0:

              # Ground truth line (might be redundant, 
              # pick a good color otherwise).
              #ax.plot(t[:, 2], t[:, 1], color="r", linewidth=1) 
              ax.scatter(
                  t[-1, 2],
                  t[-1, 1],
                  linewidths=1,
                  s=90,
                  color="c",
                  marker="+",
              )

      # Create legend handles.
      prediction_handle = mlines.Line2D(
          [],
          [],
          color="r",
          markerfacecolor="none",
          marker="o",
          linewidth=1,
          linestyle="None",
          label="Prediction",
      )

      ground_truth_handle = mlines.Line2D(
          [],
          [],
          color="c",
          marker="+",
          linestyle="None",
          label="Ground truth positions",
      )

      # Add legend to the axis.
      ax.legend(
          handles=[prediction_handle,
                   ground_truth_handle],
          loc="upper left")
      return ax

    animation = FuncAnimation(
        fig,
        update,
        frames=len(sim_movie)
    )

    video = HTML(animation.to_jshtml()); plt.close()
    return video



def trajectory_sqdistance(
        gt,
        pred,
        eps=5,
):
    """Pairs ground truth and linked trajectories.

    Parameters
    ----------
    gt: np.ndarray
        ground truth trajectory.

    pred: np.ndarray
        Predicted trajectory

    eps: int
        The radius of each particle in pixels.

    Returns
    -------
    float
        Squared distance between trajectories.

    """
    union = np.union1d(
        gt[:, 0],
        pred[:, 0]
    )
    ind = np.arange(
        union.min(),
        union.max() + 1,
        dtype=int
    )
    gt_i = (gt[:, 0] - union.min()).astype(int)
    pred_i = (pred[:, 0] - union.min()).astype(int)
 
    gt_f = np.full((*ind.shape, 2), np.Inf)
    pred_f = np.full((*ind.shape, 2), np.Inf)

    gt_f[gt_i, :] = gt[:, 1:]
    pred_f[pred_i, :] = pred[:, 1:]

    d = np.sum((gt_f - pred_f) ** 2, axis=1)

    d[np.where(d > eps ** 2)]    = eps ** 2
    d[np.isinf(d) | np.isnan(d)] = eps ** 2  

    return np.sum(d)



def trajectory_assignment(
    ground_truth,
    prediction,
    eps=5,
):
    """Compute the squared distances between all trajectories.

    This function calculates a cost matrix representing the squared distances
    between each pair of ground truth and predicted trajectories. The cost
    matrix can be used in trajectory assignment algorithms
    (e.g., Hungarian algorithm) for matching predicted trajectories to
    ground truth.

    Parameters
    ----------
    ground_truth: np.ndarray
        Array of ground truth trajectories.

    prediction: np.ndarray
        Array of predicted trajectories.
    
    eps: int, optional
        Defines the threshold for calculating squared distances and can be
        used to scale the penalty for mismatches.

    Returns
    -------
    float
        The total squared distance (or cost) between all trajectories.
        This value can be used to assess the similarity between predicted
        and ground truth trajectories.
        
    """
    dmax = 0
    cost_matrix = np.zeros((len(ground_truth), len(prediction)))

    for idxg, gt in enumerate(ground_truth):
        dmax += len(gt) * eps ** 2
        for idxp, pred in enumerate(prediction):
            cost_matrix[idxg, idxp] = trajectory_sqdistance(gt,pred,eps)

    return linear_sum_assignment(cost_matrix), cost_matrix, dmax

def trajectory_metrics(
    gt,
    pred,
    eps=5,
):
    """Function to calculate linking metrics.
    As in doi.org/10.1038/nmeth.2808.

    Parameters
    ----------

    Returns
    -------
    int, int, int, int, float, float
        True positives, false positives, false negatives, alpha, beta.

    """
    trajectory_pair, mat, dmax = trajectory_assignment(gt, pred, eps=5)

    d = sum(mat[trajectory_pair[0][:], trajectory_pair[1][:]])
    TP = len(trajectory_pair[0])
    FP = np.max([0, len(pred) - len(gt)])
    FN = np.max([0, len(gt) - len(pred)])
    dFP = 0.0
    if FP > 0:
        complement = pred
        for i in trajectory_pair[1]:
            complement = np.delete(complement,i)
        for c in complement:
            dFP += len(c) * eps**2
    alpha = 1.0 - d / dmax
    beta = (dmax - d) / (dmax + dFP)

    print(
        f""" 
        TP: {TP}
        FP: {FP}
        FN: {FN} 
        alpha: {alpha}
        beta: {beta}"""
    )
    return TP, FP, FN, alpha, beta


def normalize_min_max(
    img,
    squeeze_in_2D=False,
):
    """Normalizes image intensities to be between 0 and 1.

    Parameters
    ----------
    img: np.ndarray 
        Image to be normalized.
        
    Returns
    -------
    normalized_img: np.ndarray
        image with normalized pixel intensities.

    """

    # Eliminates an extra dimension if specified.
    if squeeze_in_2D:
        img = np.squeeze(img)

    # Set the boundary values.
    max_intensity = np.max(img)
    min_intensity = np.min(img)

    # Raise an error if max and min values are equal.
    if max_intensity == min_intensity:
        raise ValueError("Cannot normalize array with uniform intensity.")

    # Perform min-max normalization.
    normalized_img = (img - min_intensity) / (
        max_intensity - min_intensity
    )

    return normalized_img


"""CLASSES FOR MAGIK 

The code below is reworked from the following example by Jesús Piñeda:
https://github.com/DeepTrackAI/DeepTrack2/blob/develop/examples/MAGIK/Tracking_hela_cells.ipynb 

"""


class GraphFromTrajectories:
    """Graph representation of the motion of particles"""
    
    def __init__(self, connectivity_radius, max_frame_distance):

        """Initialize graph."""
        self.connectivity_radius = connectivity_radius
        self.max_frame_distance = max_frame_distance
    
    def get_connectivity(self, node_attr, frames):

        """Compute connectivity."""
        # Extract centroids.
        xy = node_attr  
        
        distances = np.linalg.norm(xy[:, None] - xy, axis=-1)
        frame_diff = (frames[:, None] - frames) * -1
        mask = ((distances < self.connectivity_radius) 
                & (frame_diff <= self.max_frame_distance)
                & (frame_diff > 0)
        )

        edge_index = np.argwhere(mask)
        edge_attr = distances[mask]
        return edge_index, edge_attr
    
    def get_gt_connectivity(self, labels, edge_index):

        """Compute ground truth connectivity."""
        source_particle = labels[edge_index[:, 0]] 
        target_cell = labels[edge_index[:, 1]]
        self_connections_mask = source_particle == target_cell #source target
        gt_connectivity = self_connections_mask
        return gt_connectivity
        
    def __call__(self, df):

        """Compute graphs from videos."""
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
    """Class to prepare the graph dataset."""

    def __init__(self, graph_dataset, Dt, dataset_size,  transform=None):
        """Initialize the dataset."""
        self.graph_dataset = graph_dataset
        self.dataset_size = dataset_size
        self.Dt = Dt
        self.transform = transform 

    def __len__(self):
        """Obtain length of dataset."""

        return self.dataset_size

    def __getitem__(self, idx):
       
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
    """Random rotations to diversify training data."""
    
    def __call__(self, graph):
        """Perform the random rotation."""

        graph = graph.clone()
        node_feats = graph.x[:, :2] - 0.5  # Centered positons.
        angle = np.random.rand() * 2 * pi
        rotation_matrix = torch.tensor(
            [[cos(angle), -sin(angle)], [sin(angle), cos(angle)]]
        ).float()
        rotated_node_attr = torch.matmul(node_feats, rotation_matrix)
        graph.x[:, :2] = rotated_node_attr + 0.5  # Restored positons.
        return graph
    
class RandomFlip:
    """Random flip to diversify training data."""
    
    def __call__(self, graph):
        """Perform the random flip."""

        graph = graph.clone()
        node_feats = graph.x[:, :2] - 0.5  # Centered positons.
        if np.random.randint(2): node_feats[:, 0] *= -1
        if np.random.randint(2): node_feats[:, 1] *= -1
        graph.x[:, :2] = node_feats + 0.5  # Restored positons.
        return graph

class NodeDropout:
  """Removal (dropout) of random nodes to simulate missing frames."""
  def __call__(self, graph):

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
    graph.edge_index = graph.edge_index[:, ~edges_connected_to_removed_node]
    graph.edge_attr = graph.edge_attr[~edges_connected_to_removed_node]
    graph.distance = graph.distance[~edges_connected_to_removed_node]
    graph.y = graph.y[~edges_connected_to_removed_node]

    return graph
  
class ComputeTrajectories:
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
