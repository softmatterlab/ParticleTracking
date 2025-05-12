"""Performance evaluation utilities.

This module provides tools for evaluating particle tracking performance by 
comparing predicted and ground truth positions or trajectories. 

Key Features
------------
- Position-based evaluation with linear assignment (Hungarian algorithm).

- Trajectory-level comparison and assignment.

- Tracking metrics including TP, FP, FN, F1, alpha, beta.

- Time-Averaged Mean Squared Displacement (TAMSD) computation.

Module Structure
----------------
Functions:

- `evaluate_locs`: Computes RMS error and F1 score between predicted and true 
    positions.

- `trajectory_sqdistance`: Computes squared distance between two trajectories.

- `trajectory_assignment`: Matches predicted to ground truth trajectories.

- `trajectory_metrics`: Computes Chenouard-style tracking metrics 
    (Nat Methods, 2014).

- `compute_TAMSD`: Computes time-averaged MSD of a single trajectory.

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
import scipy.spatial
# import scipy.optimize
from scipy.optimize import linear_sum_assignment


def evaluate_locs(
    pred_positions: np.ndarray, 
    gt_positions: np.ndarray, 
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
    pred_positions: np.ndarray
        Estimated positions.

    gt_positions: np.ndarray
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
    if pred_positions.shape[1] == 3:
        pred_positions = pred_positions[:, :2]
    
    if gt_positions.shape[1] == 3:
        gt_positions = gt_positions[:, :2]    
    
    # Compute the pairwise distance matrix.
    distance_matrix = scipy.spatial.distance_matrix(
        pred_positions, 
        gt_positions,
        )

    # Solves the Linear Sum Assignment Problem to find and match the indices of
    # the pair of particles that are the closest from each other.
    row_index, column_index = scipy.optimize.linear_sum_assignment(
        distance_matrix
        )

    # Filter pairs that are within the distance threshold.
    valid_matches = distance_matrix[row_index, column_index] < distance_th
    matched_preds = row_index[valid_matches]

    # Calculate evaluation metrics.
    TP = len(matched_preds)
    FP = len(pred_positions) - TP
    FN = len(gt_positions) - TP
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
        f"True Positives: {TP}/{len(gt_positions)}\n"
        f"False Positives: {FP}\n"
        f"False Negatives: {FN}\n"
        f"F1 Score: {F1:.4f}\n"
        f"RMSE: {RMSE:.4f}"
    )

    return TP, FP, FN, F1, RMSE

def trajectory_sqdistance(
        gt: np.ndarray,
        pred: np.ndarray,
        eps: int = 5,
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

    mask = np.isfinite(gt_f[:, 0]) & np.isfinite(pred_f[:, 0])
    d2 = np.full(len(ind), eps**2)
    d2[mask] = np.sum((gt_f[mask] - pred_f[mask]) ** 2, axis=1)
    d2 = np.minimum(d2, eps**2)

    return np.sum(d2), len(ind)


def trajectory_assignment(
    gt: np.ndarray,
    pred: np.ndarray,
    eps: int = 5,
)-> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, float]:
    """Compute the squared distances between all trajectories.

    This function calculates a cost matrix representing the squared distances
    between each pair of ground truth and predicted trajectories. The cost
    matrix can be used in trajectory assignment algorithms
    (e.g., Hungarian algorithm) for matching predicted trajectories to
    ground truth.

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
    float
        The total squared distance (or cost) between all trajectories.
        This value can be used to assess the similarity between predicted
        and ground truth trajectories.
        
    """

    dmax = 0
    cost_matrix = np.zeros((len(gt), len(pred)))
    len_matrix = np.zeros((len(gt), len(pred)))

    for idxg, gt in enumerate(gt):
        dmax += len(gt) * eps ** 2
        for idxp, pred in enumerate(pred):
            cost_matrix[idxg, idxp], len_matrix[idxg, idxp] = trajectory_sqdistance(gt,pred,eps)
    
    # cost_matrix computed earlier
    row_ind, col_ind = linear_sum_assignment(cost_matrix/len_matrix)

    # Filter matches by cost threshold
    valid_matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c]/len_matrix[r,c] < eps**2:
            valid_matches.append((r, c))

    gt_indices, pred_indices = zip(*valid_matches) if valid_matches else ([], [])
    return (np.array(gt_indices), np.array(pred_indices)), cost_matrix, dmax


def trajectory_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
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

    trajectory_pair, mat, dmax = trajectory_assignment(gt, pred, eps=eps)

    d = sum(mat[trajectory_pair[0][:], trajectory_pair[1][:]])
    TP = len(trajectory_pair[0])
    FP = np.max([0, len(pred) - TP])
    FN = np.max([0, len(gt) - TP])
    dFP = 0.0
    if FP > 0:
        matched_indices = set(trajectory_pair[1])
        complement = [pred[i] for i in range(len(pred)) if i not in matched_indices]
        for c in complement:
            dFP += len(c) * eps**2
        # complement = pred
        # for i in trajectory_pair[1]:
        #     complement = np.delete(complement,i)
        # for c in complement:
        #     dFP += len(c) * eps**2
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

def compute_TAMSD(
    traj: np.ndarray, 
    max_lag: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the Time-Averaged Mean Squared Displacement (TAMSD).
    
    Parameters
    ----------
    traj : np.ndarray
        Trajectory of shape (T, 3) in the form [frame, x, y].
    
    max_lag : int, optional
        Maximum time lag to compute TAMSD for. Defaults to T // 2.
    
    Returns
    -------
    lags : np.ndarray
        Time lags used.
    tamsd : np.ndarray
        TAMSD values corresponding to each lag.

    """

    if traj.shape[0] < 4:
        return np.array([]), np.array([])


    frames = traj[:, 0].astype(int)
    coords = traj[:, 1:3]
    max_frame = int(frames.max()) + 1
    if max_lag is None:
        max_lag = int((frames.max() - frames.min()) // 2)
    traj_full = np.full((max_frame, 2), np.nan)
    traj_full[frames] = coords

    tamsd = []
    lags = np.arange(1, max_lag + 1)
    for lag in lags:
        displacements = traj_full[:-lag] - traj_full[lag:]
        valid = np.isfinite(displacements).all(axis=1)
        squared = np.sum(displacements[valid]**2, axis=1)
        if len(squared) > 0:
            tamsd.append(np.mean(squared))
        else:
            tamsd.append(np.nan)
    return lags, np.array(tamsd)