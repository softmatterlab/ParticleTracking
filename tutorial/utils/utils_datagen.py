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

- `apply_blinking` : Introduce blinking events into trajectories.

- `trajs_array_to_list` : Split trajectories based on FOV exits.


NOTE: Spatial Quantities and Units
----------------------------------
All spatial quantities (e.g. radius, sigma, position) are internally expected 
and processed in **pixels**. However, most functions provide an optional 
`pixel_size_nm` argument (default: 100 nm) to allow input in nanometers.
If `pixel_size_nm` is specified, physical quantities will be automatically 
converted to pixel units. Set `pixel_size_nm=None` to disable conversion 
and use raw pixel units directly.

"""

from __future__ import annotations

import deeptrack as dt
import numpy as np

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
        Array of shape (M, 3) with rows [x, y, theta], where M<=num_particles
        and theta is sampled uniformly in [-pi, pi].

    """

    # Convert radius from nm to pixels if needed.
    if particle_radius is not None and pixel_size_nm is not None:
        particle_radius = particle_radius / pixel_size_nm

    # Margin ensures particles stay fully inside image.
    margin = float(particle_radius) if particle_radius is not None else 0.0
    min_distance = 2 * margin if particle_radius is not None else 0.0

    # Diffraction-limited particles cannot be placed too close to edges and each other
    if particle_radius is not None and particle_radius < 1.0:
        margin = max(5.0, margin)
        min_distance = max(5.0, min_distance)

    # Sample positions.
    if particle_radius is None:
        # No-overlap: simple uniform sampling.
        positions = np.random.uniform(
            low=margin,
            high=fov_size - margin,
            size=(num_particles, 2)
        )
    else:
        # Rejection sampling with minimum distance.
        placed = []
        attempts = 0
        while len(placed) < num_particles and attempts < max_attempts:
            candidate = np.random.uniform(
                low=margin,
                high=fov_size - margin,
                size=2
            )
            if all(np.linalg.norm(candidate - p) >= min_distance
                   for p in placed):
                placed.append(candidate)
            attempts += 1

        if not placed:
            # Return empty array if placement failed completely.
            return np.empty((0, 3))

        positions = np.vstack(placed)

    # Sample random orientation angles in [-pi, pi].
    thetas = np.random.uniform(-np.pi, np.pi, size=(positions.shape[0], 1))

    # Combine into final array.
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
        "background_mean": 0,  # Mean background intensity
        "background_std": 0,  # Std. dev. of background noise
        "poisson_snr": 100,  # Signal-to-noise ratio for Poisson noise
    }

    # Update the default dictionaries with user-defined properties.
    _core_particle_dict.update(core_particle_props)
    _shell_particle_dict.update(shell_particle_props)
    _background_dict.update(background_props)

    # Reshape trajectory data to fit expected input format.
    # Add third axis (frame) if not present.
    if len(trajs.shape) == 2:
        trajs = trajs[np.newaxis, :]  # Add a new axis

    # Check if trajectory data has 3 axis (X, Y, angle).
    if trajs.shape[-1] == 3:

        # Extract the orientation angle from the third column.
        angles = - trajs[-1][:, 2]

        # Chop the third axis to extract only (X,Y) positions.
        trajs = trajs[..., :2]
    else:
        angles = np.zeros([trajs[-1].shape[0], 1])

    # The desired format is (N, frames, dim), with dim the spatial dimensions.
    trajs = np.moveaxis(trajs, 0, 1)  # Swap axis

    # -------------------------------------------------
    # Handle blinking (NaNs) — default: no blinking
    # -------------------------------------------------

    if np.isnan(trajs).any():
        blink_mask = np.any(np.isnan(trajs), axis=-1)  # (N, T)
    else:
        blink_mask = np.zeros(trajs.shape[:2], dtype=bool)

    
    # =================================================
    # DeepTrack cannot receive NaNs → fill positions
    # =================================================
    trajs_filled = trajs.copy()

    for i in range(trajs.shape[0]):
        valid = np.where(~blink_mask[i])[0]
        if valid.size == 0:
            trajs_filled[i, :, :2] = 0.0
            continue

        first, last = valid[0], valid[-1]

        # Back-fill before first valid frame
        trajs_filled[i, :first, :2] = trajs_filled[i, first, :2]

        # Forward-fill after last valid frame
        trajs_filled[i, last + 1 :, :2] = trajs_filled[i, last, :2]

        # Forward-fill internal NaNs
        for t in range(first + 1, last + 1):
            if blink_mask[i, t]:
                trajs_filled[i, t, :2] = trajs_filled[i, t - 1, :2]


    trajs = trajs_filled

    # -----------------------------
    # Inner particle (core) with blink-aware intensity
    # -----------------------------
    
    _base_intensity = _core_particle_dict.get("intensity", 1.0)

    inner_particle = dt.Ellipsoid(
        trajectories=trajs,  # keep as ndarray (N, T, 2) for dt.Sequence compatibility
        replicate_index=lambda _ID: _ID,
        trajectory=lambda replicate_index, trajectories: np.concatenate(
            [
                trajectories[replicate_index],  # (T, 2) in pixel units
                blink_mask[replicate_index][:, None].astype(float),  # (T, 1) blink flag
            ],
            axis=1,
        ),  # returns (T, 3): [x, y, blink_flag]
        number_of_particles=trajs.shape[0],
        traj_length=trajs.shape[1],
        # initial position (x,y) only, with pixel units
        # position=lambda trajectory: dt.units.pixel * trajectory[0, :2],
        position=lambda trajectory: (
            dt.units.pixel * np.array([-10, -10])
            if trajectory[0, 2] > 0.5
            else dt.units.pixel * trajectory[0, :2]
        ),
        angles_list=angles,
        rotation=lambda replicate_index, angles_list: angles_list[replicate_index],
        **_core_particle_dict,
    )

    # def _seq_position(trajectory, sequence_step):
    #     # trajectory is (T, 3) = [x, y, blink_flag]
    #     return dt.units.pixel * trajectory[sequence_step, :2]

    def _seq_position(trajectory, sequence_step):
        # trajectory = [x, y, invisible_flag]
        if trajectory[sequence_step, 2] > 0.5:
            # Move particle far outside the image
            return dt.units.pixel * np.array([-10, -10])
        return dt.units.pixel * trajectory[sequence_step, :2]

    def _seq_intensity(trajectory, sequence_step):
        # blink_flag == 1.0 => blinking => intensity 0
        if trajectory[sequence_step, 2] > 0.5:
            return 0.0
        if callable(_base_intensity):
            return float(_base_intensity())
        return float(_base_intensity)

    sequential_inner_particle = inner_particle.to_sequential(
        position=_seq_position,
        intensity=_seq_intensity,
    )


    # # Generate inner particle (core).
    # inner_particle = dt.Ellipsoid(
    #     trajectories=trajs,
    #     replicate_index=lambda _ID: _ID,
    #     trajectory=lambda replicate_index, trajectories: dt.units.pixel
    #     * trajectories[replicate_index],
    #     number_of_particles=trajs.shape[0],
    #     traj_length=trajs.shape[1],
    #     position=lambda trajectory: trajectory[0],
    #     angles_list=angles,
    #     rotation=\
    #         lambda replicate_index, angles_list: angles_list[replicate_index],
    #     **_core_particle_dict,
    # )



    # # Sequential definition of particles with changing positions per frame.
    # sequential_inner_particle = dt.Sequential(
    #     inner_particle,
    #     position=lambda trajectory, sequence_step: trajectory[sequence_step],
    # )

    # Check if shell particle properties are provided.
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
            position=\
                lambda trajectory, sequence_step: trajectory[sequence_step],
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
    
    if save_video:
        if not path:
            raise ValueError("Path must be provided to save the video.")
        np.save(path, _video)

    return _video

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

    # Optional sanity check: warn if sigma looks suspiciously large.
    if pixel_size_nm is not None:
        sigma = sigma / pixel_size_nm

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
            - (np.sin(2 * theta) / sigma_x ** 2)
            + (np.sin(2 * theta) / sigma_y ** 2)
        )

        c = 0.5 * (
            (np.sin(theta) / sigma_x) ** 2 + (np.cos(theta) / sigma_y) ** 2
        )

        # Insert the rotated semiaxis into the Gaussian blob.
        gaussian = np.exp(
            - (
                (a * (X - x0) ** 2)
                + 2 * b * (X - x0) * (Y - y0)
                + c * (Y - y0) ** 2
            )
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
    images = np.empty((num_samples, fov_size, fov_size, 1), dtype=np.float32)
    maps = np.empty((num_samples, fov_size, fov_size, 1), dtype=np.float32)

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

        # Extract maximum radius in nanometers.
        total_particle_radius = np.maximum(
            max_axis_particle, 
            max_axis_shell
            )

        # Size of probability cloud set as the biggest body (nanometers).
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
    diffusion_std: float | tuple[float, float] = 1.0,
    min_length: int = 5,
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
    diffusion_std : float or tuple(float, float), default=1.0
        - If a single float, all trajectories share the same diffusion 
        coefficient.
        - If a tuple (low, high), a different diffusion standard deviation is 
        drawn uniformly from [low, high) for each trajectory.
        It corresponds to sqrt(2 * D * dt).
    min_length : int, default=5
        Minimum length of trajectory segments to keep after breaking at
        periodic-boundary crossings.
     
    Returns
    -------
    np.ndarray
        Trajectories array of shape (num_timesteps, num_particles, 3). Each
        entry [t, i] = [x, y, t], where x, y are positions in [0, fov_size).

    """

    # Initial positions: uniform in [0, fov_size).
    # Reuse generate_centroids for even distribution (ignoring orientation).
    centroids = generate_centroids(
        num_particles=num_particles,
        fov_size=int(fov_size),
        particle_radius = 10, # Small margin to avoid edge issues, not actual radius
    )
    # Extract x,y and ignore theta.
    initial_pos = centroids[:, :2]  # Shape (num_particles, 2)

    # Preallocate output: (T, N, 3).
    trajs = np.zeros((num_timesteps, num_particles, 3), dtype=float)
    # Set initial frame
    trajs[0, :, :2] = initial_pos
    # Third axis corresponds to rotation angle, which is set to zero due to
    # spherical symmetry of particles.
    trajs[0, :, 2] = 0

    # Determine per-particle diffusion std
    if isinstance(diffusion_std, (tuple, list, np.ndarray)) and len(diffusion_std) == 2:
        diffusion_stds = np.random.uniform(low=diffusion_std[0], high=diffusion_std[1], size=num_particles)
    else:
        diffusion_stds = np.full(num_particles, diffusion_std)

    # Generate displacements for each trajectory
    increments = np.zeros((num_timesteps - 1, num_particles, 2), dtype=float)
    for i in range(num_particles):
        increments[:, i, :] = np.random.normal(
            loc=0.0,
            scale=diffusion_stds[i],
            size=(num_timesteps - 1, 2)
        )

    # Cumulative sum of increments + initial positions, modulo fov_size.
    # Shape after cumsum: (T-1, N, 2).
    cum_disp = np.cumsum(increments, axis=0)
    # Broadcast initial positions and wrap.
    positions = (cum_disp + initial_pos[np.newaxis, :, :]) % fov_size

    # Fill trajectories for t = 1 .. T-1.
    trajs[1:, :, :2] = positions
    # # Time coordinate broadcast.
    trajs[:, :, 2] = np.arange(num_timesteps)[:, None]

    # Break trajectories at periodic-boundary crossings
    T = num_timesteps
    segments = []

    for j in range(num_particles):
        x = trajs[:, j, 0]
        y = trajs[:, j, 1]
        frames = trajs[:, j, 2]

        dx = np.abs(x[1:] - x[:-1])
        dy = np.abs(y[1:] - y[:-1])

        # Correct wrap detection for periodic boundaries
        wrap = (dx > fov_size / 2) | (dy > fov_size / 2)
        cut_idx = np.where(wrap)[0]

        # Segment boundaries
        boundaries = [-1, *cut_idx, T - 1]

        for k in range(len(boundaries) - 1):
            start = boundaries[k] + 1
            end = boundaries[k + 1] + 1  # end is exclusive

            if end - start < min_length:
                continue

            seg = np.full((T, 3), np.nan)
            seg[start:end, 0] = x[start:end]
            seg[start:end, 1] = y[start:end]
            seg[start:end, 2] = frames[start:end]

            segments.append(seg)

    if len(segments) == 0:
        return np.zeros((T, 0, 3))

    return np.stack(segments, axis=1)


def apply_blinking(
    trajs: np.ndarray,
    trim_max: int = 5,
    p: float = 0.5,
    drop_prob: float = 0.05,
    max_gap: int = 2,
) -> np.ndarray:
    """
    Apply per-particle birth/death and blinking to Brownian trajectories.

    Dropped detections are represented by NaNs in x,y.
    """
    trajs = trajs.copy()
    T, N, _ = trajs.shape

    for i in range(N):
        keep_mask = np.zeros(T, dtype=bool)

        # Birth and death (same p)
        t_birth = min(np.random.geometric(p) - 1, trim_max)
        t_death = T - 1 - min(np.random.geometric(p) - 1, trim_max)
        if t_birth > t_death:
            trajs[:, i, 0:2] = np.nan
            continue

        keep_mask[t_birth : t_death + 1] = True

        # Blinking
        drop_attempt = np.zeros(T, dtype=bool)
        drop_attempt[t_birth : t_death + 1] = (
            np.random.rand(t_death - t_birth + 1) < drop_prob
        )

        gap = 0
        for t in range(t_birth, t_death + 1):
            if drop_attempt[t]:
                if gap < max_gap:
                    keep_mask[t] = False
                    gap += 1
                else:
                    gap = 0
            else:
                gap = 0

        # Apply mask
        trajs[~keep_mask, i, 0:2] = np.nan

    return trajs


def trajs_array_to_list(
    trajs: np.ndarray,
    min_length: int = 5,
) -> list[np.ndarray]:
    """
    Convert a (T, N, 3) trajectory array into a list of trajectories
    by removing NaNs.

    Parameters
    ----------
    trajs : np.ndarray
        Array of shape (T, N, 3) with [x, y, frame].
        NaNs indicate trajectory breaks.
    min_length : int
        Minimum number of points to keep a trajectory.

    Returns
    -------
    list of np.ndarray
        List of trajectories with shape (t_i, 3): [frame, y, x].
    """

    traj_list: list[np.ndarray] = []

    T, N, _ = trajs.shape

    for j in range(N):
        # Extract one trajectory column
        traj = trajs[:, j, :]

        # Keep only rows where x is not NaN
        valid = ~np.isnan(traj[:, 0])
        traj_valid = traj[valid]

        if len(traj_valid) < min_length:
            continue

        # Reorder to [frame, y, x] if desired
        traj_list.append(traj_valid[:, [2, 0, 1]])

    return traj_list
