"""Simulation and ground truth generation utilities.

This module provides tools to simulate 2D particle motion and generate 
synthetic microscopy data for training and evaluation of particle tracking 
algorithms.

Key Features
------------
- Render synthetic microscopy images using DeepTrack with customizable optical 
  and particle properties.

- Simulate Brownian motion trajectories in a periodic field of view.

- Generate full datasets of simulated images and ground truth annotations.

Module Structure
----------------
- `generate_centroids` : Place non-overlapping particles with random angles.

- `simulate_Brownian_trajs` : Simulate Brownian motion for multiple particles.

- `transform_to_video` : Render localizations or trajectories as a video using 
    DeepTrack.

- `create_ground_truth_map` : Generate Gaussian-like ground truth maps.

- `generate_particle_dataset` : Combine centroids and DeepTrack rendering to 
    create datasets.

- `traj_break` : Split trajectories based on discontinuities or FOV exits.


=============================================================================
Spatial Quantities and Units
=============================================================================
All spatial quantities (e.g. radius, sigma, position) are internally expected 
and processed in **pixels**. However, most functions provide an optional 
`pixel_size_nm` argument (default: 100 nm) to allow input in nanometers.
If `pixel_size_nm` is specified, physical quantities will be automatically 
converted to pixel units. Set `pixel_size_nm=None` to disable conversion 
and use raw pixel units directly.
=============================================================================

"""

from __future__ import annotations

import numpy as np
import math
import matplotlib.pyplot as plt
import deeptrack as dt 

def generate_centroids(
    num_particles: int, 
    fov_size: int, 
    particle_radius: int = None,
    pixel_size_nm: float = 100,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Generate non-overlapping particles. 
    
    This function generates non-overlapping particle centroids with random 
    orientations in a 2D image.

    Parameters
    ----------
    num_particles : int
        Number of particles to place.
    fov_size : int
        Size of the square field of view, i.e., the image (in pixels).
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
            high=fov_size - margin,
            size=(num_particles, 2)
        )
    else:
        # Rejection sampling with minimum distance
        placed = []
        attempts = 0
        while len(placed) < num_particles and attempts < max_attempts:
            candidate = np.random.uniform(
                low=margin,
                high=fov_size - margin,
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
    trajs: np.ndarray,
    core_particle_props: dict = None,
    shell_particle_props: dict = None,
    optics_props: dict = None,
    background_props: dict = None,
    fov_size: int = None,
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
    trajs: np.ndarray
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
    fov_size: int, optional
        Size of the square field of view, i.e., the image (in pixels).
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
    if len(trajs.shape) == 2:
        trajs = trajs[np.newaxis, :]  # Add a new axis.
    
    # Check if trajectory data has 3 axis (X, Y, angle).
    if trajs.shape[-1] == 3:

        # Extract the orientation angle from the third column.
        angles = - trajs[-1][:, 2]

        # Chop the third axis to extract only (X,Y) positions.
        trajs = trajs[..., :2]
    else:
        angles = np.zeros([trajs[-1].shape[0], 1])

    # The desired format is (N, frames, dim), with dim the spatial dimensions.
    trajs = np.moveaxis(trajs, 0, 1)  # Swap axis.
    
    # Generate inner particle (core).
    inner_particle = dt.Ellipsoid(
        trajectories=trajs,
        replicate_index=lambda _ID: _ID,
        trajectory=lambda replicate_index, trajectories: dt.units.pixel
        * trajectories[replicate_index],
        number_of_particles=trajs.shape[0],
        traj_length=trajs.shape[1],
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
            trajectories=trajs,
            replicate_index=lambda _ID: _ID,
            trajectory=lambda replicate_index, trajectories: dt.units.pixel
            * trajectories[replicate_index[-1]],
            number_of_particles=trajs.shape[0],
            traj_length=trajs.shape[1],
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

    if trajs.shape[1] > 1:
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
    fov_size: int = 128,
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
    fov_size: int
        Size of the square field of view, i.e., the image (in pixels).
    sigma: np.ndarray or float
        Standard deviation of the Gaussian function.
        Optional dimensions 1 or 2, corresponding to an ellipsoidal Gaussian.

    Returns
    -------
    ground_truth_map: np.ndarray
        The resulting 2D intensity map.

    """

    # Initialize the empty ground truth map.
    ground_truth_map = np.zeros((fov_size, fov_size))

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
    x = np.linspace(0, fov_size - 1, fov_size)
    y = np.linspace(0, fov_size - 1, fov_size)
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
    fov_size: int,
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
    fov_size: int
        Size of the square field of view, i.e., the image (in pixels).
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
        Array of shape (num_samples, fov_size, fov_size) containing
        the generated simulated images.
    maps: np.ndarray
        Array of shape (num_samples, fov_size, fov_size, 1) containing
        the corresponding ground truth maps.
 
    """

    # Preallocate arrays to store all images and ground truth maps.
    images = np.empty(
        (num_samples, fov_size, fov_size, 1), 
        dtype=np.float32
        )
    
    maps = np.empty(
        (num_samples, 
         fov_size, 
         fov_size, 1), 
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
            fov_size=fov_size,
            particle_radius=total_particle_radius,
            pixel_size_nm=pixel_size_nm,
        )

        # Create the ground truth map based on the ground truth positions.
        # The radius of the gaussian is the minimum semiaxis, to minimize 
        # overlapping between near ellipses.
        _ground_truth_map = create_ground_truth_map(
            ground_truth_positions,
            fov_size=fov_size,
            sigma=probability_cloud_size / 3, 
            pixel_size_nm=pixel_size_nm,
        )
         
        # Convert the ground truth positions to a simulated image.
        _simulated_image = transform_to_video(
            ground_truth_positions,
            core_particle_props=core_particle_dict,
            shell_particle_props=shell_particle_dict,
            optics_props=optics_properties,
            fov_size=fov_size,
            background_props=background_props,
        )
        
        # Store the generated image and ground truth map.
        images[i] = _simulated_image
        maps[i] = _ground_truth_map[:, :, np.newaxis]
    
    print("\nDataset generation complete.")
    return images, maps


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
        Size of the square field of view, i.e., the image (in pixels). 
        Positions wrap modulo fov_size.
    diffusion_std : float, default=1.0
        Standard deviation of displacement per time step. It corresponds to 
        sqrt(2 * D * dt).
     
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
        fov_size=int(fov_size),
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
        Size of the square field of view, i.e., the image (in pixels).
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

        # Identify jumps indicating FOV exit/re-entry
        jump_indices = np.where((dx > 0.75 * fov_size) | (dy > 0.75 * fov_size))[0]
        boundaries = list(np.unique((-1, len(dx) + 1, *jump_indices)))

        # Split into segments
        for k in range(len(boundaries) - 1):
            start = boundaries[k] + 1
            end = boundaries[k + 1]
            if (end - start) >= 5:  # prevent empty slices and shorter than 5
                segment = trajs_n[start:end, j, :]
                trajs_list.append(segment)


    return trajs_list