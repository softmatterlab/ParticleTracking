"""Utility functions module for Particle Tracking example notebooks.

This module provides utility functions for generating datasets, visualizing
data, and evaluating particle localization methods. It includes functions for
creating ground truth maps, generating simulated images, and evaluating
predicted positions against true positions. The module also provides
visualization functions for plotting images, masks, and ground truth maps.
as well as for plotting predicted positions. The functions are designed to
work with particle tracking data and can be used in conjunction with
DeepTrack2 and Deeplay libraries for generating and processing particle
tracking datasets.

Key Features
------------
- **Dataset generation**

    Functions provided to generate datasets with random number generation and 
    DeepTrack2 scatterers to train neural networks.

- **Data visualization and processing**

    Various functions are provided to plot, produce videos,
    and preprocess data for training, such as normalization.
    

- **Detection methods**

    Threshholding functions to infer positions from masks.


Module Structure
----------------
Methods:

    - `create_ground_truth_map`: Creates probability density.
    
    - `evaluate_locs`: Evaluate the performance of the detection method with 
        the ground truth.
    
    - `generate_centroids`: Generate non-overlapping centers of hard spheres.
    
    - `generate_particle_dataset`: Generates simulated images of particles and 
        their corresponding probability density map.
    
    -`interactive_ruler`: Displays an input image to interactively draw 
        straight lines by clicking on each side and displays the line length.
    
    - `locate_particle_center`: Determination of position from mask based on 
        radial symmetry.
    
    -`mask_to_positions`: Extract the geometric center of each region of a 
        mask.
    
    -`normalize_min_max`: Normalize a 2D array to (0,1) values.
    
    -`pad_to_square`: Pads any image to a squared size LxL.
    
    -`plot_crops`: Plot crops inside a subplot instance for easy visualization.
    
    -`plot_image_mask_ground_truth_map`: Subplots of image, mask (optional) and 
        probability map (optional).
    
    -`plot_predicted_positions`: Plots image with ground truth (optional) and 
        predictions (optional).
    
    -`transform_to_video`: Creates a video from ground truth positions.

    -`simulate_Brownian_trajs`: Simulates Brownian motion of particles.

    -`traj_break`: Breaks trajectories coming in and out of the FOV.

    -`play_video`: Displays a stack of images as a video.

    -`convert_uint8`: Converts data to uint8 format.

    -`format_image`: Converts image to format (N, C, X, Y).

    -`make_video_with_trajs`: Generates video with trajectories.

    -`trajectory_sqdistance`: Pairs ground truth and trajectories with squared 
        distance.

    -`trajectory_metrics`: Computes trajectory metrics.

    -`trajectory_assignment`: Assigns trajectories to ground truth.

    -`plot_trajectory_matches`: Plots trajectory matches.

# =============================================================================
# Spatial Quantities and Units
# =============================================================================
# All spatial quantities (e.g. radius, sigma, position) are internally expected 
# and processed in **pixels**. However, most functions provide an optional 
# `pixel_size_nm` argument (default: 100 nm) to allow input in nanometers.
# If `pixel_size_nm` is specified, physical quantities will be automatically 
# converted to pixel units. Set `pixel_size_nm=None` to disable conversion 
# and use raw pixel units directly.
# =============================================================================

"""

from __future__ import annotations
from typing import List, Tuple  # Type hinting for lists and tuples.

import math # Mathematical operations
from math import cos, pi, sin
import random  # Generate random numbers.
from itertools import cycle # Iterate over a list.
import deeptrack as dt  # DeepTrack.
from deeptrack.extras.radialcenter import radialcenter as rc 
import deeplay as dl  # Deeplay.
import imageio  # Creates images/video from data.
from itertools import cycle # Iterate over a list.
from IPython.display import HTML
import matplotlib as mpl
import matplotlib.animation as animation  # Animation package.
import matplotlib.patches as patches  # Patches for drawing shapes.
import matplotlib.pyplot as plt # Plotting package.
import matplotlib.lines as mlines
import numpy as np  # Scientific computing using arrays.
import scipy  # Optimized for linear algebra, signal processing.
import skimage  # Image analysis (scikit).
import tkinter as tk  # Package for GUI.
import torch  # Import PyTorch library for general neural network applications.
import trackpy as tp  # Particle tracking package. Crocker & Grier method.
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
import networkx as nx

def generate_centroids(
    num_particles: int, 
    image_size: int, 
    particle_radius: int = None,
    pixel_size_nm: float = 100,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Generate non-overlapping particle 
    
    This function generates non-overlapping particle centroids with random 
    orientations in a 2D image.

    Parameters
    ----------
    num_particles : int
        Number of particles to place.
    image_size : int
        Width and height of the (square) image in pixels.
    particle_radius : float, optional
        Radius of each particle. If given in nanometers, pixel_size_nm
        will convert to pixel units; if None, no overlap constraint.
    pixel_size_nm : float, optional
        Size of one pixel in nanometers. Ignored if particle_radius is None.

    Returns
    -------
    np.ndarray
        Array of shape (M, 3) with rows [x, y, theta], where M <= num_particles.
        Theta is sampled uniformly in [-pi, pi].

    """
    # Convert radius from nm to pixels if needed
    if particle_radius is not None and pixel_size_nm is not None:
        particle_radius = particle_radius / pixel_size_nm

    # Margin ensures particles stay fully inside image
    margin = float(particle_radius) if particle_radius is not None else 0.0
    min_distance = 2 * margin if particle_radius is not None else 0.0

    # Sample positions
    if particle_radius is None:
        # No-overlap: simple uniform sampling
        positions = np.random.uniform(
            low=margin,
            high=image_size - margin,
            size=(num_particles, 2)
        )
    else:
        # Rejection sampling with minimum distance
        placed = []
        attempts = 0
        while len(placed) < num_particles and attempts < max_attempts:
            candidate = np.random.uniform(
                low=margin,
                high=image_size - margin,
                size=2
            )
            if all(np.linalg.norm(candidate - p) >= min_distance for p in placed):
                placed.append(candidate)
            attempts += 1

        if not placed:
            # Return empty array if placement failed completely
            return np.empty((0, 3))

        positions = np.vstack(placed)

    # Sample random orientation angles in [-pi, pi]
    thetas = np.random.uniform(-np.pi, np.pi, size=(positions.shape[0], 1))

    # Combine into final array
    centroids = np.hstack((positions, thetas))
    return centroids


def transform_to_video(
    trajectory_data: np.ndarray,
    core_particle_props: dict = None,
    shell_particle_props: dict = None,
    optics_props: dict = None,
    background_props: dict = None,
    image_size: int = None,
    save_video: bool = False,
    path: str = "",
) -> np.ndarray:
    """Transforms trajectories into a video.

    This function generates a video of particles moving in a 2D plane. The 
    function takes trajectory data as input and generates a video that can be 
    saved to disk if desired. The function allows for the customization of 
    particle properties, background noise, and optical properties.
    
    Parameters
    ----------
    trajectory_data: np.ndarray
        Trajectory data of particles with shape (number_of_particles,
        number_of_frames, dimensions).
    core_particle_props: dict, optional
        Dictionary containing additional particle properties (e.g 'intensity',
        'radius').
    shell_particle_props: dict, optional
        Dictionary containing additional particle properties (e.g 'intensity',
        'radius').
    optics_props: dict, optional
        Dictionary containing optical properties (e.g., 'NA', 'wavelength').
    background_props: dict, optional
        Dictionary containing background properties (e.g., 'background_mean',
        'background_std').
    image_size: int, optional
        Size of the output image (square shape).
    save_video: bool, optional
        Whether to save the generated video to disk.
    path: str, optional
        File path to save the video, required if `save_video` is True.

    Returns
    -------
    np.ndarray
        The generated video frames as a NumPy array.
        
    """
    
    # Initialize defaults if not provided.
    core_particle_props = core_particle_props or {}
    shell_particle_props = shell_particle_props or {}
    background_props = background_props or {}    

     # Initialize particle dictionaries.
    _core_particle_dict = {
        "upscale_factor": 1,
    }
    _shell_particle_dict = {}

    # Default background properties.
    _background_dict = {
        "background_mean": 0,  # Mean background intensity.
        "background_std": 0,  # Std. dev. of background noise.
        "poisson_snr": 100,  # Signal-to-noise ratio for Poisson noise.
    }

    # Update the default dictionaries with user-defined properties.
    _core_particle_dict.update(core_particle_props)
    _shell_particle_dict.update(shell_particle_props)
    _background_dict.update(background_props)

    # Reshape trajectory data to fit expected input format.
    # Add third axis (frame) if not present. 
    if len(trajectory_data.shape) == 2:
        trajectory_data = trajectory_data[np.newaxis, :]  # Add a new axis.
    
    # Check if trajectory data has 3 axis (X, Y, angle).
    if trajectory_data.shape[-1] == 3:

        # Extract the orientation angle from the third column.
        angles = - trajectory_data[-1][:, 2]

        # Chop the third axis to extract only (X,Y) positions.
        trajectory_data = trajectory_data[..., :2]
    else:
        angles = np.zeros([trajectory_data[-1].shape[0], 1])

    # The desired format is (N, frames, dim), with dim the spatial dimensions.
    trajectory_data = np.moveaxis(trajectory_data, 0, 1)  # Swap axis.
    
    # Generate inner particle (core).
    inner_particle = dt.Ellipsoid(
        trajectories=trajectory_data,
        replicate_index=lambda _ID: _ID,
        trajectory=lambda replicate_index, trajectories: dt.units.pixel
        * trajectories[replicate_index],
        number_of_particles=trajectory_data.shape[0],
        traj_length=trajectory_data.shape[1],
        position=lambda trajectory: trajectory[0],
        angles_list=angles,
        rotation = lambda replicate_index, angles_list: angles_list[
            replicate_index],
        **_core_particle_dict,
    )

    # Sequential definition of particles with changing positions per frame.
    sequential_inner_particle = dt.Sequential(
        inner_particle,
        position=lambda trajectory, sequence_step: trajectory[sequence_step],
    )

#   Check if shell particle properties are provided.
    if shell_particle_props:
        # Generate outer particle (scaled radius and intensity).
        outer_particle = dt.Ellipsoid(
            trajectories=trajectory_data,
            replicate_index=lambda _ID: _ID,
            trajectory=lambda replicate_index, trajectories: dt.units.pixel
            * trajectories[replicate_index[-1]],
            number_of_particles=trajectory_data.shape[0],
            traj_length=trajectory_data.shape[1],
            position=lambda trajectory: trajectory[0],
            z = inner_particle.z or 0,
            rotation=inner_particle.rotation or 0,
            **_shell_particle_dict,
        )

        sequential_outer_particle = dt.Sequential(
            outer_particle,
            position=lambda trajectory, sequence_step: trajectory[sequence_step],
        )

        combined_particle = (
            sequential_inner_particle 
            >> sequential_outer_particle
        )  
    else:
        combined_particle = sequential_inner_particle


    # Define background intensity variation over time.
    background = dt.Add(0)#value=_background_dict["background_mean"])


    def background_variation(
        previous_values: list = None,
        previous_value: float = None,
    ) -> float:
        """Returns a new background value with random Gaussian noise.

        Parameters
        ----------
        previous_values: list, optional
            List of previous background values.

        previous_value: float, optional
            Previous background value.

        Returns
        -------
        float
            New background value with added noise.

        """
        
        return (previous_values or [previous_value])[
            0
            ] + np.random.randn() * _background_dict["background_std"]

    sequential_background = dt.Sequential(
        background,
        value=background_variation,
    )

    # Define optical setup (e.g., Fluorescence).
    optics = optics_props

    # # Compute scale factor for optics normalization.
    # scale_factor = (
    #     optics.magnification() * optics.wavelength()
    #     / (optics.NA() * optics.resolution())
    #     ) ** 2 * (1 / np.pi)
    
    # Create the sample to render: combine particles, background, and optics.
    sample = (
        dt.Upscale(optics(
            combined_particle
            ^ sequential_inner_particle.number_of_particles
            ), factor=_core_particle_dict["upscale_factor"])
        >> dt.Background(_background_dict["background_mean"])
        >> dt.Poisson(snr=_background_dict["poisson_snr"])
        >> sequential_background
        # >> dt.NormalizeMinMax()
    )

    if trajectory_data.shape[1] > 1:
        # Sequentially update and resolve the sample to produce video frames.
        sequential_sample = dt.Sequence(
            sample,
            trajectory=inner_particle.trajectories,
            sequence_length=inner_particle.traj_length,
            )

        # Resolve the sample to generate the video.
        _video = sequential_sample.update().resolve()
    else:
        _video = sample.update().resolve()
    
    return _video#.__abs__() # Ensure real-valued field.


def create_ground_truth_map(
    ground_truth_positions: np.ndarray,
    image_size: int = 128,
    sigma: float = 1.0,
    pixel_size_nm: float = 100,
) -> np.ndarray:
    """Create a 2D ground truth map with Gaussians at particle positions.

    This function generates a 2D intensity map with Gaussian blobs centered at 
    particle positions. The Gaussian blobs represent the probability density.
    The function can be used to create ground truth maps for particle tracking
    applications, where the positions of particles are known and the goal is to
    create a visual representation of their distribution. The parameter sigma 
    controls the width of the Gaussian blobs.
    
    Parameters
    ----------
    gt_pos: np.ndarray
        Ground truth positions of particles and orientation angles.
    image_size: int
        Size of the square image (image_size x image_size).
    sigma: np.ndarray or float
        Standard deviation of the Gaussian function.
        Optional dimensions 1 or 2, corresponding to an ellipsoidal Gaussian.

    Returns
    -------
    ground_truth_map: np.ndarray
        The resulting 2D intensity map.

    """

    # Initialize the empty ground truth map.
    ground_truth_map = np.zeros((image_size, image_size))

    # Ensure that sigma has non zero dimensions, even though is a scalar.
    sigma = np.atleast_1d(sigma)
    
    # Optional sanity check: warn if sigma looks suspiciously large
    if pixel_size_nm is not None:
        sigma /= pixel_size_nm
        
    # Assign Gaussian variance to each semiaxis.
    # Equal variance corresponds to a circular Gaussian.
    sigma_x = sigma[0]
    sigma_y = sigma[1] if len(sigma) == 2 else sigma[0]
    
    # Creates a grid of x and y coordinates corresponding to pixel positions in
    # the image. This grid will be used to compute the Gaussian ground truth 
    # map associated to each particle.
    x = np.linspace(0, image_size - 1, image_size)
    y = np.linspace(0, image_size - 1, image_size)
    X, Y = np.meshgrid(x, y)

    # Add a Gaussian for each particle position.
    for pos in ground_truth_positions:
        
        # Checks if positions array includes orientation angles.
        if len(pos) == 3:
            # Reverses X and Y to be consistent with matplotlib order.
            y0, x0, theta = pos 
        
        # If no angles were passed, they are all set to zero.
        else:
            y0, x0 = pos
            theta = 0
        
        # Create a 2D Gaussian centered at (x0, y0) and rotated by an angle 
        # theta, by defining the rotated semiaxis of the Gaussian, a and b.
        a = 0.5 *  (
            (np.cos(theta) / sigma_x) ** 2 + (np.sin(theta) / sigma_y) ** 2
            )

        b = 0.25 * (
            - (
                np.sin(2 * theta) / sigma_x ** 2
                ) + (np.sin(2 * theta) / sigma_y ** 2)
            )

        c = 0.5 * (
            (np.sin(theta) / sigma_x) ** 2 + (np.cos(theta) / sigma_y) ** 2
            )
        
        # Insert the rotated semiaxis into the Gaussian blob.
        gaussian = np.exp(
            -( 
                (
                    a * (X - x0) ** 2
                    ) + 2 * b * (X - x0) * (Y - y0) + c * (Y - y0) ** 2) 
            )
        
        # Add this Gaussian to the ground truth map.
        ground_truth_map += gaussian

    return ground_truth_map


def generate_particle_dataset(
    num_samples: int,
    image_size: int,
    max_num_particles: int,
    core_particle_dict: dict,
    shell_particle_dict: dict = None,
    optics_properties: dict = None,
    background_props: dict = None,
    pixel_size_nm: float = 100,
) -> tuple:
    """Simulate particles.
    
    Generates a dataset of simulated particle images and their corresponding
    ground truth maps with non-overlapping particle positions.

    Parameters
    ----------
    num_samples: int
        The number of simulated images to be generated.
    image_size: int
        The width and height of the square images in pixels.
    max_num_particles: int
        The maximum number of particles per image.
    core_particle_dict: dict
        Properties of the core particle to be passed to DeepTrack.
    shell_particle_dict: dict
        Properties of the shell to be passed to DeepTrack.
    optics_properties: dict
        Properties of the optics to be passed to DeepTrack.
    pixel_size_nm: float
        The size of each pixel in nanometers. Default is 100 nm. Set it to None
        if pixel size is not applicable.
    background_props: dict
        Background properties for the simulation.

    Returns
    -------
    images: np.ndarray
        Array of shape (num_samples, image_size, image_size) containing
        the generated simulated images.
    maps: np.ndarray
        Array of shape (num_samples, image_size, image_size, 1) containing
        the corresponding ground truth maps.
 
    """

    # Preallocate arrays to store all images and ground truth maps.
    images = np.empty(
        (num_samples, image_size, image_size, 1), 
        dtype=np.float32
        )
    
    maps = np.empty(
        (num_samples, 
         image_size, 
         image_size, 1), 
        dtype=np.float32,
        )
    
    # Generate simulated images.
    for i in range(num_samples):

        # Display progress in the terminal.
        if np.remainder(i + 1, 10):
            print(f"\rGenerating sample {i + 1}/{num_samples}", end="")

        # Generate a random particle number > 0.
        randomized_num_particles = max(
            1, np.random.randint(0, max_num_particles)
        )
        
         # Extract radius from dictionary.
        particle_radius = core_particle_dict["radius"]
        
        if shell_particle_dict is not None:
            shell_radius = shell_particle_dict["radius"]
        else:
            shell_radius = particle_radius
        
        # Calls variable as a float32 if variable is callable.
        def callable_to_value(z):
            value = z() if callable(z) else z
            value = np.asarray(value, dtype=np.float32)
            return value
        
        # Extract numerical values from callable variables in dictionaries.
        particle_radius = callable_to_value(particle_radius)
        shell_radius = callable_to_value(shell_radius)

        # Determine which semiaxis is larger.
        max_axis_shell = np.max(shell_radius)
        max_axis_particle = np.max(particle_radius)
        
        # Extract minimum radius in pixel units.
        total_particle_radius = np.maximum(
            max_axis_particle, 
            max_axis_shell
            )
        
        # Size of probability cloud set as the biggest body (pixel units).
        probability_cloud_size = (
            shell_radius if max_axis_shell
            > max_axis_particle else particle_radius
            )

        # Generate non-overlapping positions for the ground truth.
        ground_truth_positions = generate_centroids(
            num_particles=randomized_num_particles,
            image_size=image_size,
            particle_radius=total_particle_radius,
            pixel_size_nm=pixel_size_nm,
        )

        # Create the ground truth map based on the ground truth positions.
        # The radius of the gaussian is the minimum semiaxis, to minimize 
        # overlapping between near ellipses.
        _ground_truth_map = create_ground_truth_map(
            ground_truth_positions,
            image_size=image_size,
            sigma=probability_cloud_size / 3, 
            pixel_size_nm=pixel_size_nm,
        )
         
        # Convert the ground truth positions to a simulated image.
        _simulated_image = transform_to_video(
            ground_truth_positions,
            core_particle_props=core_particle_dict,
            shell_particle_props=shell_particle_dict,
            optics_props=optics_properties,
            image_size=image_size,
            background_props=background_props,
        )
        
        # Store the generated image and ground truth map.
        images[i] = _simulated_image
        maps[i] = _ground_truth_map[:, :, np.newaxis]
    
    print("\nDataset generation complete.")
    return images, maps


def plot_predicted_positions(**kwargs: dict) -> None:
    """Plot an image with predicted and ground truth particle positions.

    This function visualizes an image with overlaid particle positions. 
    The predicted positions are plotted in red, and the ground truth positions
    are plotted in blue. Both predicted_positions and ground_truth_positions
    are optional.

    Parameters
    ----------
    **kwargs: dict
        - image: 2D array-like, optional.
            The image of the experiment or simulation, displayed in grayscale.
        - predicted_positions: 2D array-like (N, 2), optional.
            Array of predicted (X, Y) coordinates of particles.
        - ground_truth_positions: 2D array-like (N, 2), optional.
            Array of true (X, Y) coordinates of particles.
        - title: str, optional.
            Title of the plot.

    Returns
    -------
    None
        The function directly plots the image with the predicted and ground
        truth positions superimposed. If neither predicted_positions nor
        ground_truth_positions is passed, the function will print a message
        indicating that no data was passed for plotting.

    Notes:
    ------
    - If both predicted_positions and ground_truth_positions are provided,
    they will be compared visually on the same image.
    - The coordinates for both predicted_positions and ground_truth_positions
    are assumed to be in (x, y) format but will be swapped to (y, x) for
    plotting since matplotlib addresses the vertical axis first
    when using imshow.

    """

    # Retrieves the image, the predicted positions, the ground truth and the
    # title. Sets to None if they were not passed to the function.
    image = kwargs.get("image", None)
    predicted_positions = kwargs.get("predicted_positions", None)
    ground_truth_positions = kwargs.get("ground_truth_positions", None)
    title = kwargs.get("title", None)

    # Ensures the image to be NumPy array.
    image = np.array(image)

    # Ensures image is a 2D array.
    image = np.squeeze(image)

    # Plot the image of the experiment/simulation.
    plt.figure()
    plt.imshow(image, cmap="gray")

    # Control flags for the existence of ground truth and predicted positions.
    if ground_truth_positions is None:
        print("No ground truth positions were passed")
    if predicted_positions is None:
        print("No predicted positions were passed")

    # Handle the case when no ground truth or prediction is passed to the
    # function.
    if predicted_positions is None and ground_truth_positions is None:
        print("No prediction or ground truth was passed")
        return

    # Plot the ground truth if provided.
    if ground_truth_positions is not None:

        # Ensures ground truth is a NumPy array.
        ground_truth_positions = np.array(ground_truth_positions)

        # Extracts only (X,Y) positions.
        ground_truth_positions = ground_truth_positions[:, :2]

        # Swap positions from (X,Y) to (Y,X) using slicing for proper plotting.
        ground_truth_positions = ground_truth_positions[:, ::-1]


        # Scatter plot of the ground truth positions
        plt.scatter(
            *ground_truth_positions.T,
            s=10,
            c="b",
            marker="x",
            label="Ground truth",
        )

    # Plot the predicted positions if provided.
    if predicted_positions is not None:

        # Ensures predicted positions is a numpy array.
        predicted_positions = np.array(predicted_positions)

        # Swap positions from (X,Y) to (Y,X) for proper plotting.
        predicted_positions = predicted_positions[:, ::-1]

        # Scatter plot of the predicted positions.
        plt.scatter(
            *predicted_positions.T,
            s=10,
            c="r",
            marker=".",
            label="Predicted positions",
        )

    # Show the title if provided.
    if title is not None:
        plt.title(title)

    # Plot settings.
    plt.axis("off")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_image_mask_ground_truth_map(**kwargs: dict) -> None:
    """Plot an image with its corresponding mask (optional), ground truth map
    (optional) and title (optional).

    Parameters
    ----------
    **kwargs: dict
        image: 2D array-like, optional.
            The image of the experiment or simulation, displayed in grayscale.

        mask: 2D array-like, optional.
            Mask of the image produced by thresholding.

        ground_truth_map: 2D array-like, optional.
            2D ground truth map. Gaussians centered at the ground truth 
            position of particles.

    Returns
    -------

    None
        The function directly plots the image side-by-side with its mask and
        ground truth map.

    Notes:
    ------
    - If both `predicted_positions` and `ground_truth_positions` are provided,
    they will be compared visually on the same image.
    - The coordinates for both `predicted_positions` and
    `ground_truth_positions` are assumed to be in (X, Y) format
    but will be swapped to (Y, X) for plotting since matplotlib addresses
    the vertical axis first when using `imshow`.

    """

    # Retrieves the image, the mask (optional), the ground truth map (optional)
    # and the title (optional).
    image = kwargs.get("image", None)
    mask = kwargs.get("mask", None)
    ground_truth_map = kwargs.get("ground_truth_map", None)
    title = kwargs.get("title", None)

    # Extract the non string-like arguments.
    list_all_arrays = [image, mask, ground_truth_map]
    arrays_to_plot = [array for array in list_all_arrays if array is not None]

    # Number of subfigures to display.
    number_of_subplots = len(arrays_to_plot)

    # Create a master figure to plot all the subplots.
    fig, axes = plt.subplots(1, number_of_subplots)

    # Subfigure counter.
    count = 0

    # Check the existence of each array and plots them accordingly.
    if image is not None:
        axes[count].imshow(image, cmap="gray")
        if title is not None:
            axes[count].set_title("Image of Particles")
        count += 1

    if ground_truth_map is not None:
        axes[count].imshow(ground_truth_map, cmap="gray")
        axes[count].set_title("Ground-Truth Map")
        count += 1

    if mask is not None:
        axes[count].imshow(mask, cmap="gray")
        axes[count].set_title("Mask")

    # Set the title.
    fig.suptitle(title, fontsize=20)

    # Adjust layout.
    fig.tight_layout()
    fig.subplots_adjust(top=1)

    # Show the plot.
    plt.show()


def evaluate_locs(
    predicted_positions: np.ndarray, 
    true_positions: np.ndarray, 
    distance_th: float = 5,
    pixel_size_nm: float = 100,
) -> tuple:
    """Evaluate metrics.
    
    This function evaluates predicted positions against ground-truth positions. 
    It calculates several metrics for performance, including true positives, 
    false positives, false negatives, F1 score, and RMS-error. The function 
    uses the Hungarian algorithm to find the best match for each predicted 
    position. The evaluation is done by using a distance threshold. 

    Parameters
    ----------
    predicted_positions: np.ndarray
        Estimated positions.

    true_positions: np.ndarray
        Ground truth positions.

    distance_th: float
        Distance threshold (in pixels) for considering a match.

    pixel_size_nm: float, optional
        The size of each pixel in nanometers. Default is 100 nm. Set it to None
        if pixel size is not applicable.

    Returns
    -------
    int, int, int, float, float
        Metrics for performance,
        (True positives, False positives, False negatives, F1 score, RMS-error)

    """

    if pixel_size_nm is not None:
        # Convert distance threshold to pixel units.
        distance_th /= pixel_size_nm

    # Checks if there is an extra axis accounting for orientation angles.
    if predicted_positions.shape[1] == 3:
        predicted_positions = predicted_positions[:, :2]
    
    if true_positions.shape[1] == 3:
        true_positions = true_positions[:, :2]    
    
    # Compute the pairwise distance matrix.
    distance_matrix = scipy.spatial.distance_matrix(
        predicted_positions, 
        true_positions,
        )

    # Solves the Linear Sum Assignment Problem to find and match the indices of
    # the pair of particles that are the closest from each other.
    row_index, column_index = scipy.optimize.linear_sum_assignment(
        distance_matrix
        )

    # Filter pairs that are within the distance threshold.
    valid_matches = distance_matrix[row_index, column_index] < distance_th
    matched_predictions = row_index[valid_matches]

    # Calculate evaluation metrics.
    TP = len(matched_predictions)
    FP = len(predicted_positions) - TP
    FN = len(true_positions) - TP
    RMSE = (
        np.sqrt(
            np.mean(distance_matrix[row_index, column_index][valid_matches])
        )
        if TP > 0
        else float("inf")
    )
    F1 = 2 * TP / (2 * TP + FP + FN) if TP > 0 else 0.0

    # Display results.
    print(
        f"True Positives: {TP}/{len(true_positions)}\n"
        f"False Positives: {FP}\n"
        f"False Negatives: {FN}\n"
        f"F1 Score: {F1:.4f}\n"
        f"RMSE: {RMSE:.4f}"
    )

    return TP, FP, FN, F1, RMSE


def normalize_min_max(
    image_array: np.ndarray, 
    squeeze_in_2D: bool = False, 
    minimum_value: float = 0.0, 
    maximum_value: float = 1.0,
) -> np.ndarray:
    """ Array normalization.
    
    This function normalizes an array using min-max normalization to scale 
    values between 0 and 1. Optionally, it squeezes the array to 2D if it has a
    single color channel.

    Parameters
    ----------
    image_array : np.ndarray
        Array representing an image to be normalized.

    squeeze_in_2D : bool
        If True, reduces the image array to 2D by removing 
        single-dimensional entries.

    minimum_value : float, optional
        Custom minimum value to use for normalization. If None, the
        minimum is computed from the array.

    maximum_value : float, optional
        Custom maximum value to use for normalization. If None, the
        maximum is computed from the array.

    Returns
    -------
    np.ndarray
        A normalized array with values scaled between 0 and 1.

    Raises
    ------
    ValueError
        If the normalization range is zero (max == min).

    """

    # Eliminates an extra dimension if specified.
    if squeeze_in_2D:
        image_array = np.squeeze(image_array)

    # Use custom or automatic min/max values.
    min_intensity = np.min(image_array)
    max_intensity = np.max(image_array)

    if max_intensity == min_intensity or max_intensity < min_intensity:
        raise ValueError("Cannot normalize array. \
            Check maximum and minimum values.")

    # Perform min-max normalization.
    normalized_image_array = (
        (image_array - min_intensity) / (max_intensity - min_intensity)
    )

    # Rescale to minimum and maximum values.
    normalized_image_array = (
        (maximum_value - minimum_value) * normalized_image_array 
        + minimum_value
    )
    
    return normalized_image_array


def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Image padding to an LxL square.
    
    This function pads an image to a square with size LxL, with L being the 
    smallest power of 2 greater or equal to the largest side of the image.

    Parameters
    ----------
    image: np.ndarray
        Image to be padded.

    Returns
    -------
    np.ndarray
        Padded image.

    """

    import math
    
    # Extract the dimensions.
    x_dimension = image.shape[0]
    y_dimension = image.shape[1]
    
    # Extract the largest dimension.
    largest_dimension = np.maximum(x_dimension, y_dimension)
    
    # Calculate the closest power of 2 equal or larger than the largest 
    # dimension. For padding up to a power of 2.
    
    # Determine the diference to up-pad to the closest power of 2.
    x_distance_to_pad = int(np.abs(x_dimension - largest_dimension))
    y_distance_to_pad = int(np.abs(y_dimension - largest_dimension))
    
    # Ensure the distances are integers.
    padded_image = np.pad(image, (
        (0, x_distance_to_pad),(0, y_distance_to_pad)
        ))

    return padded_image


def mask_to_positions(
    mask: np.ndarray, 
    intensity_image: np.ndarray=None,
) -> np.ndarray:
    """Converts a mask to a list of positions.

    This function takes a binary mask and converts it into a list of
    coordinates representing the center of particles. The function can also
    use an intensity image to calculate the weighted centroid of the particles
    if provided. The mask should be a 2D array where non-zero values indicate
    the presence of particles. The function uses the `skimage.measure` module
    to label the connected components in the mask and then calculates the
    centroids of these components. The centroids are returned as a 2D array
    
    Parameters
    ----------
    mask: np.ndarray
        A binary mask where particles are located.
        The mask should be a 2D array where non-zero values indicate the
        presence of particles.

    intensity_image: np.ndarray, optional
        An intensity image to be used for weighted centroid calculation.
        If provided, the function will calculate the weighted centroid
        of the particles based on the intensity image.

    Returns
    -------
    np.ndarray
        A 2D array containing the coordinates of the center of particles.
    
    """

    from skimage.measure import label, regionprops

    # Label the mask.
    # Determines the connectivity of pixels having the same value. 
    # 1-connectivity refers to direct connections in x and y directions. 
    # 2-connectivity refers also includes diagonal connections.
    
    # All the connected regions are associated to the same property (position). 
    labels = label(mask)
 
    
    if intensity_image is not None:
        # Instance to measure properties of previously labelled regions.
        props = regionprops(labels, intensity_image=intensity_image)
        # Extract the centroids of each labelled region.
        mask_centroids = np.array([prop.weighted_centroid for prop in props])
    else:   
        # Instance to measure properties of previously labelled regions.
        props = regionprops(labels)
        # Extract the centroids of each labelled region.
        mask_centroids = np.array([prop.centroid for prop in props])
    
    return mask_centroids

def plot_crops(
    crops_dataset: np.ndarray,
    **kwargs: dict
) -> None:
    """Plot crops.

    This function plots crops from a dataset in a grid layout. The crops are
    displayed as subplots, with a maximum of 4 columns. The function takes a
    3D array of shape (N, X, Y) as input, where N is the number of crops and
    X and Y are the dimensions of each crop. The function also accepts an
    optional title for the plot.
    
    Parameters
    ----------
    crops_dataset: np.ndarray
        A 3D array of shape (N, X, Y) where N is the number of crops (samples)
        and X, Y are the dimensions of each crop.

    **kwargs : dict (optional)
        Keyword arguments:
            - plot_title: Title of the plot.

    Returns
    -------
    None

    """

    # Extract the number of crops (N) from the dataset.
    number_of_crops = crops_dataset.shape[0]

    # Plot the first 12 crops.
    if number_of_crops > 12:
        number_of_crops = 12
    
    # Determine the layout of subplots.
    number_of_columns = 4
    number_of_rows = math.ceil(number_of_crops / number_of_columns)

    if number_of_crops <= number_of_columns:
        number_of_rows = 1
        
    # Create a grid of subplots with the specified number of rows and columns.
    fig, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(2 * number_of_columns, 2 * number_of_rows),
    )
    # Flatten the axes array for easy indexing.
    axes = axes.flatten() if number_of_crops > 1 else [axes]

    # Plot each crop.
    for i in range(number_of_crops):
        # Select the subplot for the current crop.
        axes[i].imshow(crops_dataset[i], cmap="gray", aspect="equal")
        axes[i].set_title(f"Crop {i + 1}")
        axes[i].axis("off")  # Hide axes

    # Turn off unused subplots if any.
    for j in range(number_of_crops, len(axes)):
        axes[j].axis("off")

    # Extract the title from the optional keyword argument.
    plot_title = kwargs.get("title", None)
    
    # Display plot title if available.
    if plot_title is not None:
        # Set the title
        fig.suptitle(plot_title, fontsize=20)
        
    # Adjust layout for better spacing.
    plt.tight_layout()
    plt.show()
    

def interactive_ruler(image: np.ndarray) -> None:
    """Draw lines on an image and calculate their lengths.

    This function allows the user to interactively draw lines on an image by
    clicking two points. The length of the drawn line is calculated and
    displayed on the plot. Each line is drawn in a different color, and the 
    legend shows the lengths of all lines drawn so far.

    Parameters
    ----------
        image (numpy.ndarray): A 2D numpy array representing the image 
            (intensity map).

    """
    
    # Initialize empty lists to store the coordinates of the lines and their 
    # lengths.
    line_coords = []
    line_lengths = []

    # Define a color cycle list for the lines to be drawn.
    colors = cycle([
        'red', 'blue', 'green', 'orange', 
        'purple', 'cyan', 'magenta', 'yellow'
    ])

    def onclick(event: matplotlib.backend_bases.MouseEvent) -> None:
        """ Handle mouse click events to draw lines and calculate lengths.
        
        This function is called when the user clicks on the image. It stores
        the coordinates of the clicked points and draws a line between them.
        The length of the line is calculated and displayed in the legend.
        The legend is updated with the lengths of all lines drawn so far.
        If two points have been clicked, the line is drawn and the length is
        calculated. The coordinates are cleared for the next line.

        Parameters
        ----------
            event (matplotlib.backend_bases.MouseEvent): The mouse click event.
        """
        if event.inaxes is not None:
            # Append the x, y coordinates of the click to the list.
            line_coords.append((event.xdata, event.ydata))

            # If two points have been clicked, draw the line and calculate its 
            # length.
            if len(line_coords) == 2:
                x1, y1 = line_coords[0]
                x2, y2 = line_coords[1]

                # Calculate the length of the line in pixel units.
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Save the length to the list.
                line_lengths.append(length)  

                # Draw the line with the next color in the cycle.
                color = next(colors)
                line, = ax.plot(
                    [x1, x2], [y1, y2], 
                    'o-', 
                    color=color, 
                    label=f'Segment {len(line_lengths)}: {length:.1f} pixels',
                    )
                plt.draw()

                # Update the legend with all line lengths.
                ax.legend(
                    loc='upper right', 
                    bbox_to_anchor=(1.3, 1), 
                    fontsize=10,
                    )
                plt.draw()

                # Clear the list for the next line.
                line_coords.clear()

    # Create a figure and axis.
    fig, ax = plt.subplots()

    # Display the image using imshow.
    ax.imshow(
        image, 
        cmap='gray', 
        #origin='lower', 
        #extent=[0, image.shape[1], 0, image.shape[0]],
        )
    ax.set_title("Click on two points to draw a line on the image")

    # Connect the onclick function to the figure.
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent legend from overlapping.
    plt.show()

def simulate_Brownian_trajs(
    num_particles: int,
    num_timesteps: int,
    fov_size: float,
    diffusion_std: float = 1.0,
) -> np.ndarray:
    """Simulate 2D Brownian motion trajectories in a periodic square fov.

    Parameters
    ----------
    num_particles : int
        Number of particles to simulate.
    num_timesteps : int
        Number of time steps (including t=0).
    fov_size : float
        Side length of the square simulation fov. Positions wrap modulo fov_size.
    particle_radius: float, default=None

    diffusion_std : float, default=1.0
        Standard deviation of displacement per time step.

    pixel_size_nm: float, default=100

    Returns
    -------
    np.ndarray
        Trajectories array of shape (num_timesteps, num_particles, 3).
        Each entry [t, i] = [x, y, t], where x,y are positions in [0, fov_size).
    """
    # Initial positions: uniform in [0, fov_size)
    # Reuse generate_centroids for even distribution (ignoring orientation)
    centroids = generate_centroids(
        num_particles=num_particles,
        image_size=int(fov_size),
        particle_radius = None,
    )
    # Extract x,y and ignore theta
    initial_pos = centroids[:, :2]  # shape (num_particles, 2)

    # Preallocate output: (T, N, 3)
    trajs = np.zeros((num_timesteps, num_particles, 3), dtype=float)
    # Set initial frame
    trajs[0, :, :2] = initial_pos
    trajs[0, :, 2] = 0

    # Generate all random increments at once
    increments = np.random.normal(
        loc=0.0,
        scale=diffusion_std,
        size=(num_timesteps - 1, num_particles, 2)
    )

    # Cumulative sum of increments + initial positions, modulo fov_size
    # Shape after cumsum: (T-1, N, 2)
    cum_disp = np.cumsum(increments, axis=0)
    # Broadcast initial positions and wrap
    positions = (cum_disp + initial_pos[np.newaxis, :, :]) % fov_size

    # Fill trajectories for t=1..T-1
    trajs[1:, :, :2] = positions
    # Time coordinate broadcast
    trajs[:, :, 2] = np.arange(num_timesteps)[:, None]

    return trajs


    ### FROM NOW

    ### ON, THE FUNCTIONS ARE FOR TRAJECTORY ANALYSIS ###

def traj_break(
    trajs: np.ndarray,
    fov_size: int,
    num_particles: int,
) -> list[np.ndarray]:
    """Break trajectories when particles leave and re-enter the FOV.

    This function splits each ground truth trajectory into sub-trajectories 
    that remain within the the field of view (FOV), by detecting large jumps 
    in position.

    Parameters
    ----------
    trajs : np.ndarray
        Trajectories of shape (T, N, 3), where the last dimension 
        is (x, y, frame).

    fov_size : int
        Size of the full field of view (e.g. width or height in pixels).

    num_particles : int
        Total number of particles in the simulation.

    Returns
    -------
    list of np.ndarray
        List of valid trajectory segments (with shape (t_i, 3): [frame, y, x]) 
        that stayed within FOV.
    
    """

    trajs_list = []
    trajs_n = trajs[:, :, [2, 0, 1]] #swap axes, frame first
    for j in range(num_particles):
        dx = np.abs(trajs[1:, j, 0] - trajs[:-1, j, 0])
        dy = np.abs(trajs[1:, j, 1] - trajs[:-1, j, 1])

        ind = np.where((dx > 0.75 * fov_size) | (dy > 0.75 * fov_size))[0]
        ind = list(np.unique((-1, len(dx) + 1, *ind)))

        for k in range(len(ind) - 1):
            trajs_list.append(trajs_n[ind[k] + 1:ind[k + 1], j, :])
    return trajs_list

def play_video(
    video,
    video_name="video",
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

        Instances a HTML video player with a video.

    """
    fig = plt.figure(figsize=figsize)
    images = []
    plt.axis("off")
    plt.title(video_name, fontsize=14)

    for image in video:
        images.append([plt.imshow(image[:, :, 0], cmap="gray")])

    anim = mpl.animation.ArtistAnimation(
        fig, images, interval=1e3 / fps, blit=True, repeat_delay=0
    )

    html = HTML(anim.to_jshtml())
    display(html)
    plt.close()


def convert_uint8(video):
    """Converts video to uint8 format.

    Parameters
    ----------
    video: np.ndarray 
        video to be converted.
        
    Returns
    -------

    converted_video: np.ndarray
        video in uint8 format.

    """
    converted_video = []
    for idx_im, im in enumerate(video):
        im = im[:, :, 0]
        im = im / im.max()
        im = im * 255
        im = im.astype(np.uint8)
        converted_video.append(im)
    return converted_video


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
    trajs_pred_list,
    video,
    fov_size,
    trajs_gt_list=None,
) -> HTML:
    """Generate video with predicted (and optionally ground truth) trajectories.

    Parameters
    ----------
    trajs_pred_list : list of np.ndarray
        List of predicted trajectories. Each trajectory is an array of shape 
        (T, 3): [frame, y, x].

    video : np.ndarray
        Video frames, shape (N_frames, H, W).

    fov_size : int
        The full field of view (FOV) size.

    trajs_gt_list : list of np.ndarray, optional
        List of ground truth trajectories. Each trajectory must have shape 
        (T, 3): [frame, y, x]. If None, only predictions are shown.

    Returns
    -------
    IPython.core.display.HTML
        HTML5 video displaying overlaid trajectories.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    def update(frame_idx):
        ax.clear()
        ax.imshow(video[frame_idx], cmap="gray")
        ax.set_xlim([0, fov_size])
        ax.set_ylim([fov_size, 0])  # Invert y-axis
        ax.set_xticks([]), ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

        # Plot predicted trajectories
        for traj in trajs_pred_list:
            t = traj[traj[:, 0] <= frame_idx]
            if len(t) > 0:
                ax.plot(t[:, 2], t[:, 1], color="w", linewidth=0.5)
                ax.scatter(
                    t[-1, 2], t[-1, 1],
                    s=100, facecolors="none", edgecolors="r",
                    marker="o", linewidths=1
                )

        legend_handles = [
            mlines.Line2D([], [], color="r", marker="o", linestyle="None",
                          markerfacecolor="none", label="Prediction")
        ]

        # Plot ground truth trajectories if provided
        if trajs_gt_list is not None:
            for traj in trajs_gt_list:
                t = traj[traj[:, 0] <= frame_idx]
                if len(t) > 0:
                    ax.scatter(
                        t[-1, 2], t[-1, 1],
                        color="c", s=90, marker="+", linewidths=1
                    )
            legend_handles.append(
                mlines.Line2D([], [], color="c", marker="+", linestyle="None",
                              label="Ground Truth")
            )

        ax.legend(handles=legend_handles, loc="upper left")
        return ax

    anim = animation.FuncAnimation(fig, update, frames=len(video))
    video_html = HTML(anim.to_jshtml())
    plt.close(fig)
    return video_html


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

    # d = np.sum((gt_f - pred_f) ** 2, axis=1)
    # d[np.where(d > eps ** 2)]    = eps ** 2
    # d[np.isinf(d) | np.isnan(d)] = eps ** 2  
    # return np.sum(d)

    mask = np.isfinite(gt_f[:, 0]) & np.isfinite(pred_f[:, 0])
    d2 = np.full(len(ind), eps**2)
    d2[mask] = np.sum((gt_f[mask] - pred_f[mask]) ** 2, axis=1)
    d2 = np.minimum(d2, eps**2)

    return np.sum(d2)


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
    """Computes tracking performance metrics.

    This function computes tracking performance metrics between ground truth 
    and predicted trajectories based on Chenouard et al. (Nat. Methods 2014, 
    doi.org/10.1038/nmeth.2808): TP, FP, FN, alpha, and beta.
    
    Parameters
    ----------
    gt: np.ndarray
        Array of ground truth trajectories.
    pred: np.ndarray
        Array of predicted trajectories.
    eps: int, optional
        Defines the threshold for calculating squared distances and can be
        used to scale the penalty for mismatches.

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
        alpha: {alpha:.3f}
        beta: {beta:.3f}"""
    )
    return TP, FP, FN, alpha, beta


def plot_trajectory_matches(
    trajs_gt_list: List[np.ndarray],
    trajs_pred_list: List[np.ndarray],
    matched_pairs: Tuple[np.ndarray, np.ndarray],
    figsize: Tuple[int, int] = (6, 6),
) -> None:

    """Plots matched and unmatched trajectories for visual evaluation.

    Parameters
    ----------
    trajs_gt_list : list of np.ndarray
        List of ground truth trajectories, each with shape (T, 3) [frame, y, x].

    trajs_pred_list : list of np.ndarray
        List of predicted trajectories, same format.

    matched_pairs : tuple of (np.ndarray, np.ndarray)
        Indices of matched trajectories (as returned by `linear_sum_assignment`).

    figsize : tuple
        Size of the matplotlib figure.

    """

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Plot matched trajectories.
    colors = np.random.rand(len(matched_pairs[0]), 3)
    for (gt_idx, pred_idx), color in zip(zip(*matched_pairs), colors):
        gt_traj = trajs_gt_list[gt_idx]
        pred_traj = trajs_pred_list[pred_idx]

        ax.plot(pred_traj[:, 2], pred_traj[:, 1], color=color, linewidth=3, label="_predicted")
        ax.plot(gt_traj[:, 2], gt_traj[:, 1], color="k", linewidth=1, label="_groundtruth")

    # Identify and plot unmatched ground truth (false negatives).
    unmatched_gt = set(range(len(trajs_gt_list))) - set(matched_pairs[0])
    for idx in unmatched_gt:
        traj = trajs_gt_list[idx]
        ax.plot(traj[:, 2], traj[:, 1], color="gray", linewidth=1, label="_false_negative")

    # Identify and plot unmatched predictions (false positives).
    unmatched_pred = set(range(len(trajs_pred_list))) - set(matched_pairs[1])
    for idx in unmatched_pred:
        traj = trajs_pred_list[idx]
        ax.plot(traj[:, 2], traj[:, 1], color="r", linewidth=1, label="_false_positive")

    # Set up legend with unique labels only once.
    legend_handles = [
        plt.Line2D([], [], color='k', linewidth=1, label='Ground Truth'),
        plt.Line2D([], [], color='r', linewidth=1, label='False Positive'),
        plt.Line2D([], [], color='gray', linewidth=1, label='False Negative'),
        plt.Line2D([], [], color='gray', linewidth=3, label='Prediction'),
    ]
    ax.legend(handles=legend_handles, loc='upper right')

    # Format plot
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("Trajectory Matching")

    plt.tight_layout()
    plt.show()



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


def make_list(trajs_from_graph, test_graph, fov_size):
    """
    Convert MAGIK trajectories from graph format to a list of NumPy arrays,
    where each array is shaped (T, 3) with [frame, y, x] rows.

    Parameters
    ----------
    trajs_from_graph : list
        List of trajectories, each represented by a list of node indices.

    test_graph : torch_geometric.data.Data
        Graph object with `.frames` (frame indices) and `.x` (positions) attributes.

    Returns
    -------
    trajs_list : list of np.ndarray
        Each entry is an array of shape (T, 3): [frame, y, x]
    """
    trajs_list = []
    for t in trajs_from_graph:
        frames = test_graph.frames[list(t)].cpu().numpy()
        coords = test_graph.x[list(t)].cpu().numpy() * fov_size # shape (T, 2), assumed [x, y]
        # Flip to [y, x] and concatenate with frames
        traj = np.column_stack((frames, coords[:, 0], coords[:, 1]))
        # Optionally sort by frame if not ordered
        traj = traj[np.argsort(traj[:, 0])]
        trajs_list.append(traj)
    return trajs_listxs