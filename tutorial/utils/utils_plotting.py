"""Visualization and analysis utilities for particle tracking data.

This module provides a collection of utility functions for visualizing and 
analyzing particle tracking data. It supports operations such as interactive 
image inspection, trajectory overlay, trajectory comparison, and 
quantitative trajectory analysis.

Key Features
------------
- Display video data interactively in Jupyter notebooks.

- Visualize matched/unmatched trajectories and TAMSD curves.

Module Structure
----------------
Functions:

- `play_video`: Display a video stack in Jupyter notebooks.

- `convert_uint8`: Convert video frames to uint8 format.

- `plot_crops`: Plot crops from a dataset in a grid layout.

- `interactive_ruler`: Draw lines on an image and calculate their lengths.

- `plot_image_mask_ground_truth_map`: Plot an image with its mask and
    ground-truth map.

- `plot_predicted_positions`: Overlay predicted and ground-truth positions on
    an image.

- `make_video_with_trajs`: Generate a video with overlaid trajectories.

- `plot_trajectory_matches`: Visualize matched and unmatched trajectories.

- `plot_TAMSDs`: Plot the TAMSD curves for predicted and ground-truth
    trajectories.

"""

from __future__ import annotations

from IPython.display import HTML, display
from itertools import cycle
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.backend_bases
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import math
import numpy as np

from .utils_evaluation import compute_TAMSD

def play_video(
    video: np.ndarray,
    video_name="video",
    figsize=(5, 5),
    fps=10,
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

    vmin, vmax = np.percentile(video, [1, 99])

    for image in video:
        images.append(
            [plt.imshow(image[:, :, 0], cmap="gray", vmin=vmin, vmax=vmax)]
        )

    anim = mpl.animation.ArtistAnimation(
        fig, images, interval=1e3 / fps, blit=True, repeat_delay=0
    )

    html = HTML(anim.to_jshtml())
    display(html)
    plt.close()

def convert_uint8(video: np.ndarray) -> np.ndarray:
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

def plot_crops(
    crops_dataset: np.ndarray,
    **kwargs: dict,
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
        gt_map: 2D array-like, optional.
            2D ground truth map. Gaussians centered at the ground truth 
            position of particles.

    Returns
    -------
    None
        The function directly plots the image side-by-side with its mask and
        ground truth map.

    """

    # Retrieves the image, the mask (optional), the ground truth map (optional)
    # and the title (optional).
    image = kwargs.get("image", None)
    mask = kwargs.get("mask", None)
    gt_map = kwargs.get("gt_map", None)
    title = kwargs.get("title", None)

    # Extract the non string-like arguments.
    list_all_arrays = [image, mask, gt_map]
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

    if gt_map is not None:
        axes[count].imshow(gt_map, cmap="gray")
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

def plot_predicted_positions(**kwargs: dict) -> None:
    """Plot an image with predicted and ground truth particle positions.

    This function visualizes an image with overlaid particle positions. 
    The predicted positions are plotted in red, and the ground truth positions
    are plotted in blue. Both pred_positions and gt_positions
    are optional.

    Parameters
    ----------
    **kwargs: dict
        - image: 2D array-like, optional.
            The image of the experiment or simulation, displayed in grayscale.
        - pred_positions: 2D array-like (N, 2), optional.
            Array of predicted (X, Y) coordinates of particles.
        - gt_positions: 2D array-like (N, 2), optional.
            Array of true (X, Y) coordinates of particles.
        - title: str, optional.
            Title of the plot.

    Returns
    -------
    None
        The function directly plots the image with the predicted and ground
        truth positions superimposed. If neither pred_positions nor
        gt_positions is passed, the function will print a message
        indicating that no data was passed for plotting.

    Notes:
    ------
    - If both pred_positions and gt_positions are provided,
    they will be compared visually on the same image.
    - The coordinates for both pred_positions and gt_positions
    are assumed to be in (x, y) format but will be swapped to (y, x) for
    plotting since matplotlib addresses the vertical axis first
    when using imshow.

    """

    # Retrieves the image, the predicted positions, the ground truth and the
    # title. Sets to None if they were not passed to the function.
    image = kwargs.get("image", None)
    pred_positions = kwargs.get("pred_positions", None)
    gt_positions = kwargs.get("gt_positions", None)
    title = kwargs.get("title", None)

    # Ensures the image to be NumPy array.
    image = np.array(image)

    # Ensures image is a 2D array.
    image = np.squeeze(image)

    # Plot the image of the experiment/simulation.
    plt.figure()

    vmin, vmax = np.percentile(image, [1, 99])

    plt.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)

    # Control flags for the existence of ground truth and predicted positions.
    if gt_positions is None:
        print("No ground truth positions were passed")
    if pred_positions is None:
        print("No predicted positions were passed")

    # Handle the case when no ground truth or prediction is passed to the
    # function.
    if pred_positions is None and gt_positions is None:
        print("No prediction or ground truth was passed")
        return

    # Plot the ground truth if provided.
    if gt_positions is not None:

        # Ensures ground truth is a NumPy array.
        gt_positions = np.array(gt_positions)

        # Extracts only (X,Y) positions.
        gt_positions = gt_positions[:, :2]

        # Swap positions from (X,Y) to (Y,X) using slicing for proper plotting.
        gt_positions = gt_positions[:, ::-1]


        # Scatter plot of the ground truth positions
        plt.scatter(
            *gt_positions.T,
            s=14,
            c="b",
            marker="x",
            label="Ground truth",
        )

    # Plot the predicted positions if provided.
    if pred_positions is not None:

        # Ensures predicted positions is a numpy array.
        pred_positions = np.array(pred_positions)

        # Swap positions from (X,Y) to (Y,X) for proper plotting.
        pred_positions = pred_positions[:, ::-1]

        # Scatter plot of the predicted positions.
        plt.scatter(
            *pred_positions.T,
            s=14,
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

def make_video_with_trajs(
    trajs_pred_list,
    video,
    fov_size,
    trajs_gt_list=None,
    figure_title=None
) -> HTML:
    """Generate video with predicted (and optionally ground truth) trajectories.

    Parameters
    ----------
    trajs_pred_list : list of np.ndarray
        list of predicted trajectories. Each trajectory is an array of shape 
        (T, 3): [frame, y, x].

    video : np.ndarray
        Video frames, shape (N_frames, H, W).

    fov_size : int
        The full field of view (FOV) size.

    trajs_gt_list : list of np.ndarray, optional
        list of ground truth trajectories. Each trajectory must have shape 
        (T, 3): [frame, y, x]. If None, only predictions are shown.

    Returns
    -------
    IPython.core.display.HTML
        HTML5 video displaying overlaid trajectories.

    """

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim([0, fov_size])
    ax.set_ylim([fov_size, 0])  # Invert y-axis
    ax.set_xticks([]), ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    if figure_title is not None:
        ax.set_title(figure_title)

    # Image artist (static background per frame).
    im = ax.imshow(video[0], cmap="gray", animated=True)

    # Predicted trajectories: one line + one scatter (last point) per traj.
    pred_lines, pred_scatters = [], []
    for _ in trajs_pred_list:
        line, = ax.plot([], [], color="w", linewidth=0.5, animated=True)
        scatter = ax.scatter([], [], s=100, facecolors="none", edgecolors="r",
                             marker="o", linewidths=1, animated=True)
        pred_lines.append(line)
        pred_scatters.append(scatter)

    # Ground truth (optional).
    gt_scatters = []
    if trajs_gt_list is not None:
        for _ in trajs_gt_list:
            scatter = ax.scatter([], [], color="c", s=90, marker="+",
                                 linewidths=1, animated=True)
            gt_scatters.append(scatter)

    # Legend (static, only drawn once).
    legend_handles = [
        mlines.Line2D([], [], color="r", marker="o", linestyle="None",
                      markerfacecolor="none", label="Prediction")
    ]
    if trajs_gt_list is not None:
        legend_handles.append(
            mlines.Line2D([], [], color="c", marker="+", linestyle="None",
                          label="Ground Truth")
        )
    ax.legend(handles=legend_handles, loc="upper left")

    def update(frame_idx):
        # Update video frame.
        im.set_array(video[frame_idx])

        # Update predicted trajectories.
        for traj, line, scatter in zip(
            trajs_pred_list, pred_lines, pred_scatters
        ):
            t = traj[traj[:, 0] <= frame_idx]
            if len(t) > 0:
                line.set_data(t[:, 2], t[:, 1])
                scatter.set_offsets([[t[-1, 2], t[-1, 1]]])
            else:
                line.set_data([], [])
                scatter.set_offsets(np.empty((0, 2))) 

        # Update GT trajectories
        if trajs_gt_list is not None:
            for traj, scatter in zip(trajs_gt_list, gt_scatters):
                t = traj[traj[:, 0] <= frame_idx]
                if len(t) > 0:
                    scatter.set_offsets([[t[-1, 2], t[-1, 1]]])
                else:
                    scatter.set_offsets(np.empty((0, 2))) 

        return [im] + pred_lines + pred_scatters + gt_scatters


    anim = animation.FuncAnimation(
        fig, update, frames=len(video), blit=True, interval=50
    )
    video_html = HTML(anim.to_jshtml())
    plt.close(fig)
    return video_html

def plot_trajectory_matches(
    trajs_gt_list: list[np.ndarray],
    trajs_pred_list: list[np.ndarray],
    matched_pairs: tuple[np.ndarray, np.ndarray],
    figsize: tuple[int, int] = (6, 6),
) -> None:
    """Plots matched and unmatched trajectories for visual evaluation.

    Parameters
    ----------
    trajs_gt_list : list of np.ndarray
        List of ground truth trajectories, each with shape (T, 3)[frame, y, x].

    trajs_pred_list : list of np.ndarray
        List of predicted trajectories, same format.

    matched_pairs : tuple of (np.ndarray, np.ndarray)
        Indices of matched trajectories (returned by `linear_sum_assignment`).

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

        ax.plot(pred_traj[:, 2], pred_traj[:, 1], color=color, linewidth=3,
                label="_predicted")
        ax.plot(gt_traj[:, 2], gt_traj[:, 1], color="k", linewidth=1,
                label="_groundtruth")

    # Identify and plot unmatched ground truth (false negatives).
    unmatched_gt = set(range(len(trajs_gt_list))) - set(matched_pairs[0])
    for idx in unmatched_gt:
        traj = trajs_gt_list[idx]
        ax.plot(traj[:, 2], traj[:, 1], color="gray", linewidth=1,
                label="_false_negative")

    # Identify and plot unmatched predictions (false positives).
    unmatched_pred = set(range(len(trajs_pred_list))) - set(matched_pairs[1])
    for idx in unmatched_pred:
        traj = trajs_pred_list[idx]
        ax.plot(traj[:, 2], traj[:, 1], color="r", linewidth=1,
                label="_false_positive")

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

def plot_TAMSDs(
    trajs_pred: list[np.ndarray],
    trajs_gt: list[np.ndarray] | None = None,
    matched_pairs: tuple[np.ndarray, np.ndarray] | None = None,
    max_lag: int | None = None,
) -> None:
    """
    Plots the TAMSDs for predicted and optionally ground truth trajectories.

    Parameters
    ----------
    trajs_pred : list of np.ndarray
        Predicted trajectories.
    trajs_gt : list of np.ndarray, optional
        Ground truth trajectories.
    matched_pairs : tuple of arrays, optional
        tuple (gt_indices, pred_indices) for matched trajectories.
    max_lag : int, optional
        Maximum time lag to compute TAMSD. If None, it's estimated from
        trajectory length.

    """
    fig, axes = plt.subplots(
        1, 
        2 if trajs_gt is not None else 1, 
        figsize=(10, 4) if trajs_gt is not None else (5, 4),
        sharey=True
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    if matched_pairs is not None:
        colors = plt.cm.tab20(np.linspace(0, 1, len(matched_pairs[0])))
        for color, (gt_idx, pr_idx) in zip(colors, zip(*matched_pairs)):
            lag_gt, tamsd_gt = compute_TAMSD(trajs_gt[gt_idx], max_lag)
            lag_pr, tamsd_pr = compute_TAMSD(trajs_pred[pr_idx], max_lag)
            axes[0].plot(lag_pr, tamsd_pr, color=color)
            axes[1].plot(lag_gt, tamsd_gt, color=color)
    else:
        for traj in trajs_pred:
            lag, tamsd = compute_TAMSD(traj, max_lag)
            axes[0 if trajs_gt is not None else 0].plot(lag, tamsd, alpha=0.7)

        if trajs_gt is not None:
            for traj in trajs_gt:
                lag, tamsd = compute_TAMSD(traj, max_lag)
                axes[1].plot(lag, tamsd, alpha=0.7)

    axes[0].set_title("Predicted Trajectories")
    axes[0].set_xlabel("Lag time (frames)")
    axes[0].set_ylabel("TAMSD (px²)")
    # axes[0].set_xscale("log")
    # axes[0].set_yscale("log")

    if trajs_gt is not None:
        axes[1].set_title("Ground Truth Trajectories")
        axes[1].set_xlabel("Lag time (frames)")
        # axes[1].set_xscale("log")
        # axes[1].set_yscale("log")

    plt.tight_layout()
    plt.show()
