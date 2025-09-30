"""Image preprocessing utilities.

This module provides utility functions for image normalization, padding, 
mask processing, and formatting for use in machine learning pipelines, 
especially particle tracking and localization tasks.

Key Features
------------
- Formatting of images for deep learning models.

- Extraction of particle positions from binary or intensity-weighted masks.

- Formatting of images for deep learning models.

Module Structure
----------------
Functions:

- `normalize_min_max`: Normalize an image array using min-max scaling.

- `pad_to_square`: Pad a 2D image to make it square.

- `mask_to_positions`: Extract particle coordinates from a binary mask.

- `format_image`: Reformat image into (N, C, X, Y) format for neural networks.

"""

from __future__ import annotations

import numpy as np
import torch

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

def format_image(img: np.ndarray) -> np.ndarray:
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
