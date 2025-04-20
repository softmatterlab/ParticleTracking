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
    
    -`mask_to_positions`: Extract the geometric center of each region of a mask.
    
    -`normalize_min_max`: Normalize a 2D array to (0,1) values.
    
    -`pad_to_square`: Pads any image to a squared size LxL.
    
    -`plot_crops`: Plot crops inside a subplot instance for easy visualization.
    
    -`plot_image_mask_ground_truth_map`: Subplots of image, mask (optional) and 
        probability map (optional).
    
    -`plot_predicted_positions`: Plots image with ground truth (optional) and 
        predictions (optional).
    
    -`transform_to_video`: Creates a video from ground truth positions.

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
import math # Mathematical operations
import random  # Generate random numbers.
from itertools import cycle # Iterate over a list.
import deeptrack as dt  # DeepTrack.
from deeptrack.extras.radialcenter import radialcenter as rc 
import deeplay as dl  # Deeplay.
import imageio  # Creates images/video from data.
from itertools import cycle # Iterate over a list.
import matplotlib.animation as animation  # Animation package.
import matplotlib.patches as patches  # Patches for drawing shapes.
import matplotlib.pyplot as plt # Plotting package.
import numpy as np  # Scientific computing using arrays.
import scipy  # Optimized for linear algebra, signal processing.
import skimage  # Image analysis (scikit).
import tkinter as tk  # Package for GUI.
import torch  # Import PyTorch library for general neural network applications.
import trackpy as tp  # Particle tracking package. Crocker & Grier method.


def generate_centroids(
    num_particles: int, 
    image_size: int, 
    particle_radius: int,
    pixel_size_nm: float = 100,
) -> np.ndarray:
    """Generates non-overlapping particle positions in a 2D image. 
    
    Uses random number generation. Attempts to place particles at random, 
    avoiding overlapping centers by setting a minimum separation distance 
    equal to a specified particle radius.

    Parameters
    ----------
    image_size: int
        The width and height of the square image in pixels.
    num_particles: int
        The number of particles to place in the image.
    particle_radius: int
        The radius of each particle in pixels.
    pixel_size_nm: float, optional
        The size of each pixel in nanometers. Default is 100 nm. Set it to None
        for pixel units.

    Returns
    -------
    np.ndarray
        An array of shape (num_particles, 3), where each row contains the
        (x, y, theta) coordinates of a particle's center and orientation angle.

    """
    # Pre-allocate an array to store the position of particles.
    particles = []

    #
    if pixel_size_nm is not None:
        particle_radius /= pixel_size_nm

    # Condition of no overlap according to hard spheres.
    min_distance = 2 * particle_radius

    # Maximum attempts to place particle without overlap.
    max_attempts = 1000

    # Set an attempt counter to zero.
    attempts = 0
        
    # Start adding particles while keeping a separation distance.    
    while len(particles) < num_particles and attempts < max_attempts:

        # Randomly generate a new particle position within the valid range
        x = np.random.uniform(particle_radius, image_size - particle_radius)
        y = np.random.uniform(particle_radius, image_size - particle_radius)
        new_particle = np.array([x, y])

        # Check if this new particle overlaps with any existing particles
        if all(
            np.linalg.norm(new_particle - p) >= min_distance for p in particles
        ):
            particles.append(new_particle)

        attempts += 1

        # If the number of attempts has reached its maximum, decrese the number
        # of particles by 1 and reset the attempt counter to start again.
        if attempts == max_attempts:
            num_particles += -1
            attempts = 0
            """
            print(f"Simulation of {num_particles} particles with"
             "image size {image_size}x{image_size} did not converge")
            print("Number of particles target decreased by 1")
            print("Attempts counter reset to 0")
            """
    
    # Assign an angle orientation after all particles have been created.
    # Create a third column with randomized angles.
    angle_column = np.random.uniform(-1.0, 1.0, len(particles)) * np.pi

    # Reshape z_column to be 2D.
    angle_column = angle_column.reshape(-1, 1)

    # Append the Z column to the pos_array.
    particles = np.hstack((particles, angle_column))
    
    # Handle the case of only one particle centered at the middle.
    if num_particles == 1:
        angle = np.random.uniform(-1.0, 1.0) * np.pi
        particles = np.array([image_size // 2, image_size // 2, angle])       
        particles = np.array([particles])
            
    return np.array(particles)


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
    _core_particle_dict = {"upscale_factor": 1} 
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
    
    # Check if trajectory data has 3 axis (X, Y, angle).
    if trajectory_data.shape[1] == 3:

        # Extract the orientation angle from the third column.
        angles = - trajectory_data[:, 2]
        #angles = np.ones([len(trajectory_data), 1])

        # Chop the third axis to extract only (X,Y) positions.
        trajectory_data = trajectory_data[:, :2]
        
    else:
        angles = np.zeros([len(trajectory_data), 1])

    # Reshape trajectory data to fit expected input format.
    # The desired format is (N, 1, dim), with dim the spatial dimensions.
    trajectory_data = trajectory_data[np.newaxis, :]  # Add a new axis.
    trajectory_data = np.moveaxis(trajectory_data, 0, 1)  # Swap axis.
    

    # Generate inner particle
    inner_particle = dt.Ellipsoid(
        trajectories=trajectory_data,
        replicate_index=lambda _ID: _ID,
        trajectory=lambda replicate_index, trajectories: dt.units.pixel
        * trajectories[replicate_index],
        number_of_particles=trajectory_data.shape[0],
        traj_length=trajectory_data.shape[1],
        position=lambda trajectory: trajectory[0],
        # Particle can be slightly out of plane of focus at random.
        z = 0,#lambda: 100 * np.random.uniform(-1.0, 1.0) * dt.units.nm,
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
    background = dt.Add(value=_background_dict["background_mean"])


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
    #optics = dt.Darkfield(**_optics_dict)
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
        >> dt.NormalizeMinMax()
        >> dt.Poisson(snr=_background_dict["poisson_snr"])
        >> sequential_background
        >> dt.NormalizeMinMax()
    )

    if trajectory_data.shape[0] >= 1:
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
    pixel_size_nm: float = 100,
    background_props: dict = None,

) -> tuple:
    """Generates a dataset of simulated particle images and their corresponding
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
    are assumed to be in (X, Y) format but will be swapped to (Y, X) for
    plotting since matplotlib addresses the vertical axis first
    when using imshow.

    Examples:
    --------
    >>> plot_predicted_positions(image=img_array,
    ...                             predicted_positions=pred_array,
    ...                             ground_truth_positions=gt_array,
    ...                             title="Particle Localization")


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

    Example Usage:
    --------------
    >>> plot_image_mask_ground_truth_map(image=img_array,
    ...                             mask=mask_array,
    ...                             ground_truth_map=map_array,
    ...                             title="Image of experiment")

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
            axes[count].set_title("Image of particles")
        count += 1

    if ground_truth_map is not None:
        axes[count].imshow(ground_truth_map, cmap="gray")
        axes[count].set_title("Ground truth map")
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
    """Evaluate predicted positions against true positions using a distance 
    threshold.

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
    """Normalizes an array using min-max normalization to scale values
    between 0 and 1. Optionally squeezes the array to 2D if it has a
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
    """Pads any image to a squared size LxL, with L being the lowest power of 2 
    greater or equal to the largest side of the image.

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


def locate_particle_centers(
    predicted_positions: list or tuple,
    simulated_image: np.ndarray,
    estimated_radius: int,
    pixel_size_nm: float=100,
) -> np.ndarray:
    """Locates the center of particles in an image by calculating the center
    of a region of interest (ROI) around each predicted position exploiting
    radial symmetry.

    Parameters
    ----------
    predicted_positions: list or tuple 
        list or tuple of (x, y) coordinates for predicted positions.

    simulated_image: np.ndarray
        Image array in which particles are located.

    estimated_radius: int
        Estimated radius of the particles in pixels.
        This is used to define the size of the ROI.

    pixel_size_nm: float, optional
        The size of each pixel in nanometers. Default is 100 nm. Set it to None
        if pixel size is not applicable.

    Returns
    -------
    np.ndarray
        Array of corrected particle positions.

    """
    
    if pixel_size_nm is not None:
        # Convert distance threshold to pixel units.
        estimated_radius /= pixel_size_nm
     
    corrected_positions = []

    for x, y in predicted_positions:

        # Calculate bounds for the region of interest (ROI).
        x_start = int(max(0, np.floor(x) - estimated_radius // 2))
        x_end = int(min(np.floor(x)
        + estimated_radius // 2, simulated_image.shape[0]))

        y_start = int(max(0, np.floor(y) - estimated_radius // 2))
        y_end = int(min(np.floor(y)
        + estimated_radius // 2, simulated_image.shape[1]))

        # Extract the ROI from the image.
        roi = simulated_image[x_start:x_end, y_start:y_end]

        # Compute the center of the ROI.
        xc, yc = rc(roi, invert_xy=True)

        # Append the corrected position.
        corrected_positions.append([x_start + xc, y_start + yc])

    return np.array(corrected_positions)


def mask_to_positions(
    mask: np.ndarray, 
    intensity_image: np.ndarray=None,
) -> np.ndarray:
    """Converts a mask to a list of positions.
    
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
    """Plots all crops in a dataset with shape (N, X, Y) as subplots.

    Ensures no more than 4 columns in the layout.

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
    """
    Interactive function to draw lines on an image and calculate their lengths.

    This function displays an image as a background and allows the user to draw 
        lines by clicking on two points. Each line is drawn in a unique color, 
        and its length is displayed in a legend.

    Parameters
    ----------
        image (numpy.ndarray): A 2D numpy array representing the image 
            (intensity map).

    Example
    -------
        To use this function with a random image:
        >>> image = np.random.rand(100, 100)
        >>> interactive_line_plot(image)

        To use this function with a real image:
        >>> from matplotlib.image import imread
        >>> image = imread('path_to_your_image.png')
        >>> if image.ndim == 3:  # Convert RGB to grayscale
        ...     image = np.mean(image, axis=2)
        >>> interactive_line_plot(image)
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
        """
        Handle mouse click events to draw lines and calculate their lengths.

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