from scipy.ndimage import zoom, filters
import numpy as np
from . import cp
from . import interp
from . import calculate
from . import visualization
from . import cupy_ndimage
from .imresize import imresize

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

####################################################################################################
## The same as before
def correctMotionGrid(data_raw, coords_new):
    """
    Correct the motion using 3D interpolation for GPU arrays.

    Args:
        data_raw (cupy.ndarray): The raw 3D data (GPU array), shape (H, W, D).
        coords_new (cupy.ndarray): New grid coordinates for interpolation (GPU array), shape (H, W, D, 3).

    Returns:
        data_corrected (cupy.ndarray): The corrected 3D data after interpolation (GPU array), shape (H, W, D).
    """
    # Extract dimensions from input data
    x, y, z = data_raw.shape  # data_raw shape: (H, W, D)

    # Ensure the data is on GPU and convert to float32 for precision
    data_raw = cp.asarray(
        data_raw, dtype=cp.float32
    )  # Convert to GPU array, shape: (H, W, D)
    coords_new = cp.asarray(coords_new)  # Convert to GPU array, shape: (H, W, D, 3)

    # Transpose coordinates from (H, W, D, 3) to (3, H, W, D) for interpolation function
    # This reorders dimensions to match the expected input format for interp3Grid
    coords_new = cp.transpose(coords_new, (3, 0, 1, 2))  # Shape: (3, H, W, D)

    # Perform 3D interpolation using the deformed coordinates
    # This warps the original data according to the motion field
    data1_tran = interp.interp3Grid(
        data_raw, coords_new, method="linear"
    )  # Shape: (H, W, D)

    # Reshape the interpolated data back to original dimensions
    data_corrected = cp.reshape(data1_tran, (x, y, z))  # Shape: (H, W, D)

    return data_corrected


def getNeiDiff(phi_current, r):
    """
    Calculate the neighbor difference using a filter to enforce smoothness.

    Args:
        phi_current (cupy.ndarray): The 3D motion field data (GPU array), shape (H, W, D, 3).
        r (int): The radius of the filter (size = 2*r+1).

    Returns:
        neiDiff (cupy.ndarray): The filtered 3D data after applying the neighbor difference filter, shape (H, W, D, 3).
    """
    # Create the neighbor filter with size (2*r+1) x (2*r+1) x 1 x 1
    # This filter will be used to compute the difference between each point and its neighbors
    NeiFltr = cp.ones(
        (r * 2 + 1, r * 2 + 1, 1, 1), dtype=cp.float32
    )  # Shape: (2*r+1, 2*r+1, 1, 1)

    # Normalize the filter by dividing by the number of neighbors (excluding center)
    # This ensures the filter sums to zero, making it a difference operator
    NeiFltr = NeiFltr / (
        (r * 2 + 1) ** 2 - 1
    )  # Normalize by number of neighbors minus center

    # Set the center element to -1 to create a difference filter
    # This makes the filter compute: (sum of neighbors) - center_value
    NeiFltr[r, r] = -1  # Center element becomes negative

    # Apply the filter to compute neighbor differences
    # This enforces smoothness by penalizing large differences between neighboring motion vectors
    neiDiff = calculate.imfilter(
        phi_current, NeiFltr, boundary="replicate", output="same", functionality="corr"
    )  # Shape: (H, W, D, 3)

    return neiDiff


def calError(It, penaltyRaw, smoothPenaltySum):
    """
    Calculate the error and penalty terms for the given 3D data.

    Args:
        It (cupy.ndarray): The temporal difference (GPU array), shape (H, W, D).
        penaltyRaw (cupy.ndarray): The 4D penalty raw values (GPU array), shape (H, W, D, 3).
        smoothPenaltySum (float): The smoothing penalty sum value.

    Returns:
        tuple: diffError (float), penaltyError (float)
    """
    # Get the shape of the temporal difference image
    x, y, z = It.shape  # It shape: (H, W, D)

    # Calculate the intensity difference error (mean squared error)
    # This measures how well the warped moving image matches the reference image
    diffError = cp.mean(It**2)  # Scalar value

    # Calculate the smoothness penalty error
    # Square the penalty values and sum across the 4th dimension (x,y,z motion components)
    penaltyCorrected = (
        cp.sum(penaltyRaw**2, axis=3) * smoothPenaltySum
    )  # Shape: (H, W, D)

    # Normalize the penalty error by the total number of voxels
    penaltyError = cp.sum(penaltyCorrected) / (x * y * z)  # Scalar value

    # Handle both CuPy and NumPy arrays by converting to CPU if needed
    if hasattr(diffError, "get"):
        return diffError.get(), penaltyError.get()  # Convert GPU arrays to CPU
    else:
        return float(diffError), float(penaltyError)  # Already CPU arrays


def getSpatialGradientInOrgGrid(data_raw, coords_new):
    """
    Calculate the spatial gradient on deformed coordinates using 3D interpolation.

    Args:
        data_raw (cupy.ndarray): The raw 3D data (GPU array), shape (H, W, D).
        coords_new (cupy.ndarray): Deformed coordinates, shape (H, W, D, 3)
                                   where coords_new[...,0]=x, coords_new[...,1]=y, coords_new[...,2]=z.

    Returns:
        Ix (cupy.ndarray): Gradient along x-axis, shape (H, W, D).
        Iy (cupy.ndarray): Gradient along y-axis, shape (H, W, D).
        Iz (cupy.ndarray): Gradient along z-axis, shape (H, W, D).
    """
    step = 1.0  # Step size for finite differences
    x, y, z = data_raw.shape  # data_raw shape: (H, W, D)

    # Extract deformed coordinates for each dimension
    x_coords, y_coords, z_coords = (
        coords_new[..., 0],
        coords_new[..., 1],
        coords_new[..., 2],
    )  # Each shape: (H, W, D)

    # --- Compute gradient along x direction (Ix) ---
    # Perturb x-coordinate by adding and subtracting step
    x_coords_incre = cp.clip(x_coords + step, 0, x - 1)  # Shape: (H, W, D)
    x_coords_decre = cp.clip(x_coords - step, 0, x - 1)  # Shape: (H, W, D)

    # Interpolate at (x+step, y, z) and (x-step, y, z) to get intensity values
    data_incre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords_incre, y_coords, z_coords))
    )  # Shape: (H, W, D)
    data_decre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords_decre, y_coords, z_coords))
    )  # Shape: (H, W, D)

    # Compute x-gradient using finite differences
    Ix = (data_incre - data_decre) / (2 * step)  # Shape: (H, W, D)

    # --- Compute gradient along y direction (Iy) ---
    # Perturb y-coordinate by adding and subtracting step
    y_coords_incre = cp.clip(y_coords + step, 0, y - 1)  # Shape: (H, W, D)
    y_coords_decre = cp.clip(y_coords - step, 0, y - 1)  # Shape: (H, W, D)

    # Interpolate at (x, y+step, z) and (x, y-step, z) to get intensity values
    data_incre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords, y_coords_incre, z_coords))
    )  # Shape: (H, W, D)
    data_decre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords, y_coords_decre, z_coords))
    )  # Shape: (H, W, D)

    # Compute y-gradient using finite differences
    Iy = (data_incre - data_decre) / (2 * step)  # Shape: (H, W, D)

    # --- Compute gradient along z direction (Iz) ---
    # Perturb z-coordinate by adding and subtracting step
    z_coords_incre = cp.clip(z_coords + step, 0, z - 1)  # Shape: (H, W, D)
    z_coords_decre = cp.clip(z_coords - step, 0, z - 1)  # Shape: (H, W, D)

    # Interpolate at (x, y, z+step) and (x, y, z-step) to get intensity values
    data_incre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords, y_coords, z_coords_incre))
    )  # Shape: (H, W, D)
    data_decre = interp.interp3Grid(
        data_raw, cp.asarray((x_coords, y_coords, z_coords_decre))
    )  # Shape: (H, W, D)

    # Compute z-gradient using finite differences
    Iz = (data_incre - data_decre) / (2 * step)  # Shape: (H, W, D)

    return Ix, Iy, Iz


def getFlow3_withPenalty6(
    Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt, smoothPenaltySum, neiSum
):
    """
    Compute the flow with penalty and 3x3 matrix determinant using Lucas-Kanade method.

    Args:
        Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt (cupy.ndarray): The components for flow calculation, each shape (H, W, D).
        smoothPenaltySum (float): The smooth penalty sum.
        neiSum (cupy.ndarray): The neighbor sum, shape (H, W, D, 3).

    Returns:
        cupy.ndarray: The computed phi gradient flow, shape (H, W, D, 3).
    """
    # Add smoothness penalty to diagonal elements of the structure tensor
    # This regularizes the solution and prevents singular matrices
    Ixx += smoothPenaltySum  # Add penalty to x-x component
    Iyy += smoothPenaltySum  # Add penalty to y-y component
    Izz += smoothPenaltySum  # Add penalty to z-z component

    # Add neighbor sum to the temporal gradient terms
    # This incorporates the smoothness constraint into the optical flow equation
    Ixt += neiSum[:, :, :, 0]  # Add x-component of neighbor sum
    Iyt += neiSum[:, :, :, 1]  # Add y-component of neighbor sum
    Izt += neiSum[:, :, :, 2]  # Add z-component of neighbor sum

    # Calculate the determinant of the 3x3 structure tensor matrix
    # This is used to check if the matrix is invertible
    DET = calculate.getDet3(Ixx, Ixy, Ixz, Iyy, Iyz, Izz)  # Shape: (H, W, D)

    # Calculate the minors (2x2 determinants) for the adjugate matrix
    # These are used to compute the inverse of the structure tensor
    M11 = calculate.getDet2(Iyy, Iyz, Iyz, Izz)  # Minor for (1,1) element
    M12 = -calculate.getDet2(Ixy, Iyz, Ixz, Izz)  # Minor for (1,2) element (with sign)
    M13 = calculate.getDet2(Ixy, Iyy, Ixz, Iyz)  # Minor for (1,3) element
    M22 = calculate.getDet2(Ixx, Ixz, Ixz, Izz)  # Minor for (2,2) element
    M23 = -calculate.getDet2(Ixx, Ixy, Ixz, Iyz)  # Minor for (2,3) element (with sign)
    M33 = calculate.getDet2(Ixx, Ixy, Ixy, Iyy)  # Minor for (3,3) element

    # Compute the optical flow using the inverse of the structure tensor
    # This is the Lucas-Kanade solution: v = -A^(-1) * b
    Vx = (M11 * Ixt + M12 * Iyt + M13 * Izt) / DET  # x-component of motion
    Vy = (M12 * Ixt + M22 * Iyt + M23 * Izt) / DET  # y-component of motion
    Vz = (M13 * Ixt + M23 * Iyt + M33 * Izt) / DET  # z-component of motion

    # Stack the motion components into a single array
    phi_gradient = cp.stack((Vx, Vy, Vz), axis=-1)  # Shape: (H, W, D, 3)

    # Replace NaN values with 0 to handle singular cases
    # This prevents numerical issues when the determinant is very small

    num_nans = cp.isnan(phi_gradient)
    if cp.sum(num_nans) > 0:
        print(f"number of nans: {cp.sum(num_nans)}")
        phi_gradient[cp.isnan(phi_gradient)] = 0
    return phi_gradient


def compute_new_grid(grid, r, motion_shape):
    """
    Normalize the original grid to let the coordinates of control points be integer.

    Args:
        grid (tuple): Original grid coordinates (x_coord, y_coord, z_coord), each shape (H, W, D).
        r (int): Filter radius.
        motion_shape (tuple): Shape of the motion field (H, W, D).

    Returns:
        cupy.ndarray: Normalized grid coordinates, shape (3, H, W, D).
    """
    x_coord, y_coord, z_coord = grid  # Each shape: (H, W, D)

    # Normalize x and y coordinates to control point grid
    # The factor (2*r+1) represents the spacing between control points
    x_new = (x_coord - r) / (2 * r + 1)  # Shape: (H, W, D)
    y_new = (y_coord - r) / (2 * r + 1)  # Shape: (H, W, D)

    # Clamp the normalized coordinates to valid range
    x_new = cp.minimum(
        cp.maximum(x_new, 0.0), motion_shape[0]
    )  # Clamp to [0, motion_shape[0]]
    y_new = cp.minimum(
        cp.maximum(y_new, 0.0), motion_shape[1]
    )  # Clamp to [0, motion_shape[1]]

    # Keep z coordinate unchanged (no normalization needed)
    z_new = z_coord  # Shape: (H, W, D)

    # Stack the normalized coordinates
    return cp.stack([x_new, y_new, z_new], axis=0)  # Shape: (3, H, W, D)


## The same as before
####################################################################################################
def _softmax_stable(x, axis=-1, eps=1e-8):
    """
    Numerically stable softmax.
    """
    x_max = cp.max(x, axis=axis, keepdims=True)
    ex = cp.exp(x - x_max)
    return ex / (cp.sum(ex, axis=axis, keepdims=True) + eps)


def _patch_grid_regularization(
    mu_patch,
    conf_patch,
    lam=1.0,
    num_iters=40,
    eps=1e-6,
):
    """
    Solve a simple confidence-weighted spatial regularization on the patch grid:

        min_z  sum_ij conf_ij * (z_ij - mu_ij)^2
             + lam * sum_(neighbors) (z_ij - z_kl)^2

    by Jacobi-style iterations.

    Parameters
    ----------
    mu_patch : cp.ndarray, shape (Ny, Nx)
        Local patch-wise soft depth estimate.
    conf_patch : cp.ndarray, shape (Ny, Nx)
        Confidence per patch, in [0, 1].
    lam : float
        Spatial regularization strength.
    num_iters : int
        Number of Jacobi iterations.
    eps : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    z_patch : cp.ndarray, shape (Ny, Nx)
        Regularized patch-wise depth field.
    """
    z = mu_patch.astype(cp.float32, copy=True)
    conf = conf_patch.astype(cp.float32, copy=False)

    for _ in range(num_iters):
        up = cp.pad(z[:-1, :], ((1, 0), (0, 0)), mode="edge")
        down = cp.pad(z[1:, :], ((0, 1), (0, 0)), mode="edge")
        left = cp.pad(z[:, :-1], ((0, 0), (1, 0)), mode="edge")
        right = cp.pad(z[:, 1:], ((0, 0), (0, 1)), mode="edge")

        neighbor_sum = up + down + left + right

        z = (conf * mu_patch + lam * neighbor_sum) / (conf + 4.0 * lam + eps)

    return z


def FindInitPhase_robust(
    data_mov,
    data_ref,
    patch_size,
    overlap=0.5,
    smooth_sigma=20.0,
    use_gradient=True,
    weight_eps=1e-6,
    return_debug=False,
    #  robust-Z parameters
    z_curve_sigma=1.0,
    posterior_beta=12.0,
    local_radius=3,
    smoothness_alpha=6.0,
    regularization_lambda=1.0,
    regularization_iters=40,
    min_confidence=1e-3,
):
    """
    Robust initialization of phase for a single moving slice.

    Improvements over the original version
    --------------------------------------
    1) Use the whole ZNCC curve instead of hard argmax only.
    2) Convert the smoothed ZNCC curve into a soft posterior over z.
    3) Use posterior mean as patch-wise depth estimate.
    4) Define confidence from multiple whole-curve statistics:
       - evidence strength
       - local posterior mass near the main mode
       - curve smoothness
    5) Apply confidence-weighted patch-grid regularization so that
       low-confidence patches follow neighboring reliable patches.
    6) Fuse the regularized patch-wise depth into a dense z-map.

    Parameters
    ----------
    data_mov : cp.ndarray or np.ndarray
        Moving slice, shape (H, W) or (H, W, 1).
    data_ref : cp.ndarray or np.ndarray
        Reference volume, shape (H, W, Z).
    patch_size : int
        Patch size along Y and X.
    overlap : float, default=0.5
        Patch overlap ratio in [0, 1).
    smooth_sigma : float, default=20.0
        Gaussian smoothing sigma for final dense z-map in XY.
    use_gradient : bool, default=True
        If True, use gradient magnitude for matching.
    weight_eps : float, default=1e-6
        Small epsilon to avoid division by zero.
    return_debug : bool, default=False
        If True, return detailed intermediate results.

    New parameters
    --------------
    z_curve_sigma : float, default=1.0
        Gaussian smoothing sigma along z for each patch's score curve.
        This suppresses jagged local noise before posterior construction.
    posterior_beta : float, default=12.0
        Inverse temperature for converting scores into posterior weights.
        Larger values make the posterior sharper.
    local_radius : int, default=3
        Radius around the dominant peak used to measure local posterior mass.
        This helps distinguish broad/smooth peaks from separated multi-peaks.
    smoothness_alpha : float, default=6.0
        Controls how strongly rough score curves are penalized in confidence.
    regularization_lambda : float, default=1.0
        Strength of patch-grid spatial regularization.
    regularization_iters : int, default=40
        Number of iterations for patch-grid regularization.
    min_confidence : float, default=1e-3
        Lower bound on patch confidence.

    Returns
    -------
    phase_init : np.ndarray
        Initial phase, shape (H, W, 1, 3).
    z_map_smooth : np.ndarray
        Dense smoothed z map, shape (H, W).
    debug : dict, optional
        Returned only if return_debug=True.
    """
    # ------------------------------------------------------------------
    # Normalize input shapes
    # data_mov -> (H, W, 1)
    # data_ref -> (H, W, Z)
    # ------------------------------------------------------------------
    mov = cp.asarray(calculate.to_3d(data_mov), dtype=cp.float32)
    ref = cp.asarray(calculate.to_3d(data_ref), dtype=cp.float32)

    H, W, Dm = mov.shape
    Hr, Wr, Z = ref.shape

    if Dm != 1:
        raise ValueError(f"data_mov must represent a single slice, got shape {mov.shape}")
    if (H, W) != (Hr, Wr):
        raise ValueError(
            f"XY size mismatch: data_mov has {(H, W)}, data_ref has {(Hr, Wr)}"
        )
    if not (0 <= overlap < 1):
        raise ValueError("overlap must be in [0, 1).")
    if patch_size > H or patch_size > W:
        raise ValueError(
            f"patch_size={patch_size} is larger than input size {(H, W)}"
        )
    if Z < 1:
        raise ValueError("Reference volume must have at least one z slice.")

    # ------------------------------------------------------------------
    # Feature transform
    # ------------------------------------------------------------------
    mov2d = mov[:, :, 0]

    if use_gradient:
        mov_feat = calculate.grad_mag_2d(mov2d)

        ref_feat = cp.stack(
            [calculate.grad_mag_2d(ref[:, :, z]) for z in range(Z)],
            axis=2
        )
    else:
        mov_feat = mov2d.astype(cp.float32, copy=False)
        ref_feat = ref.astype(cp.float32, copy=False)

    # ------------------------------------------------------------------
    # Patch grid
    # ------------------------------------------------------------------
    step = max(int(round(patch_size * (1.0 - overlap))), 1)

    ys = cp.arange(0, H - patch_size + 1, step, dtype=cp.int32)
    xs = cp.arange(0, W - patch_size + 1, step, dtype=cp.int32)

    if ys.size == 0 or int(ys[-1]) != H - patch_size:
        ys = cp.unique(cp.concatenate([ys, cp.asarray([H - patch_size], dtype=cp.int32)]))
    if xs.size == 0 or int(xs[-1]) != W - patch_size:
        xs = cp.unique(cp.concatenate([xs, cp.asarray([W - patch_size], dtype=cp.int32)]))

    Ny = ys.size
    Nx = xs.size
    P = patch_size

    # ------------------------------------------------------------------
    # Extract patch tensors
    # mov_windows: (H-P+1, W-P+1, P, P)
    # ref_windows: typically (H-P+1, W-P+1, Z, P, P)
    # ------------------------------------------------------------------
    mov_windows = cp.lib.stride_tricks.sliding_window_view(mov_feat, (P, P))
    ref_windows = cp.lib.stride_tricks.sliding_window_view(ref_feat, (P, P), axis=(0, 1))

    mov_patches = mov_windows[ys[:, None], xs[None, :], :, :]        # (Ny, Nx, P, P)
    ref_patches = ref_windows[ys[:, None], xs[None, :], ...]         # (Ny, Nx, Z, P, P)

    if ref_patches.ndim != 5:
        raise RuntimeError(
            f"Unexpected ref_patches shape {ref_patches.shape}; "
            "please inspect sliding_window_view output layout."
        )
    if ref_patches.shape[-2:] != (P, P):
        raise RuntimeError(
            f"Unexpected ref_patches patch dims {ref_patches.shape[-2:]}, expected {(P, P)}"
        )

    # ------------------------------------------------------------------
    # Weighted ZNCC across patch_y, patch_x, candidate_z
    # ------------------------------------------------------------------
    weight_patch = cp.asarray(calculate.hann2d(P, P), dtype=cp.float32)
    weight_patch = weight_patch[None, None, None, :, :]  # (1,1,1,P,P)

    mov_patches_exp = mov_patches[:, :, None, :, :]      # (Ny,Nx,1,P,P)

    wsum = cp.sum(weight_patch, axis=(-2, -1), keepdims=True) + weight_eps

    mov_mean = cp.sum(weight_patch * mov_patches_exp, axis=(-2, -1), keepdims=True) / wsum
    ref_mean = cp.sum(weight_patch * ref_patches, axis=(-2, -1), keepdims=True) / wsum

    mov_centered = mov_patches_exp - mov_mean
    ref_centered = ref_patches - ref_mean

    numerator = cp.sum(weight_patch * mov_centered * ref_centered, axis=(-2, -1))
    mov_denom = cp.sum(weight_patch * mov_centered * mov_centered, axis=(-2, -1))
    ref_denom = cp.sum(weight_patch * ref_centered * ref_centered, axis=(-2, -1))

    scores_raw = numerator / (cp.sqrt(mov_denom * ref_denom) + weight_eps)  # (Ny, Nx, Z)
    scores_raw = cp.clip(scores_raw, -1.0, 1.0)

    # ------------------------------------------------------------------
    # 1D smoothing along z: suppress jagged noise while preserving
    # broader peak / slope structure.
    # ------------------------------------------------------------------
    if Z >= 3 and z_curve_sigma > 0:
        scores_smooth = cupy_ndimage.gaussian_filter1d(
            scores_raw,
            sigma=z_curve_sigma,
            axis=2,
            mode="nearest"
        )
    else:
        scores_smooth = scores_raw

    # ------------------------------------------------------------------
    # Convert whole ZNCC curve into a soft posterior over z
    # posterior(z) ∝ exp(beta * score(z))
    # ------------------------------------------------------------------
    posterior_logits = posterior_beta * scores_smooth
    posterior = _softmax_stable(posterior_logits, axis=2, eps=weight_eps)   # (Ny, Nx, Z)

    z_axis = cp.arange(Z, dtype=cp.float32)[None, None, :]

    # Soft patch-wise depth estimate: posterior mean
    mu_patch = cp.sum(posterior * z_axis, axis=2)                            # (Ny, Nx)

    # Posterior variance: optional diagnostic
    var_patch = cp.sum(posterior * (z_axis - mu_patch[:, :, None]) ** 2, axis=2)

    # Dominant mode index from smoothed curve
    peak_idx = cp.argmax(scores_smooth, axis=2).astype(cp.int32)
    peak_val = cp.max(scores_smooth, axis=2)
    median_val = cp.median(scores_smooth, axis=2)
    min_val = cp.min(scores_smooth, axis=2)

    # ------------------------------------------------------------------
    # Confidence from whole-curve features
    #
    # (1) Evidence strength:
    #     whether the curve has a meaningful elevation above its baseline.
    # ------------------------------------------------------------------
    evidence = (peak_val - median_val) / (1.0 - median_val + weight_eps)
    evidence = cp.clip(evidence, 0.0, 1.0)

    # ------------------------------------------------------------------
    # (2) Local posterior mass around the dominant mode:
    #     high for a single sharp peak or a smooth broad peak,
    #     lower for separated multi-modal ambiguity.
    # ------------------------------------------------------------------
    if Z == 1:
        local_mass = cp.ones((Ny, Nx), dtype=cp.float32)
    else:
        z_idx = cp.arange(Z, dtype=cp.int32)[None, None, :]
        mask_local = cp.abs(z_idx - peak_idx[:, :, None]) <= int(local_radius)
        local_mass = cp.sum(posterior * mask_local.astype(cp.float32), axis=2)
        local_mass = cp.clip(local_mass, 0.0, 1.0)

    # ------------------------------------------------------------------
    # (3) Smoothness quality:
    #     penalize curves with strong jagged second-order oscillation.
    #     A smooth broad hill should remain high-quality.
    # ------------------------------------------------------------------
    if Z >= 3:
        second_diff = scores_smooth[:, :, 2:] - 2.0 * scores_smooth[:, :, 1:-1] + scores_smooth[:, :, :-2]
        roughness = cp.mean(cp.abs(second_diff), axis=2)
        amplitude = peak_val - min_val
        roughness_norm = roughness / (amplitude + weight_eps)
        smoothness_quality = cp.exp(-smoothness_alpha * roughness_norm)
    else:
        smoothness_quality = cp.ones((Ny, Nx), dtype=cp.float32)

    # ------------------------------------------------------------------
    # Final confidence:
    #   evidence × averaged shape quality
    # This keeps:
    #   - sharp/high peaks -> high confidence
    #   - broad/smooth peaks -> still high confidence
    #   - globally weak curves -> low confidence
    #   - jagged/noisy spikes -> suppressed confidence
    # ------------------------------------------------------------------
    shape_quality = 0.5 * local_mass + 0.5 * smoothness_quality
    confidence = evidence * shape_quality
    confidence = cp.clip(confidence, min_confidence, 1.0).astype(cp.float32)

    # ------------------------------------------------------------------
    # Patch-grid spatial regularization
    # Low-confidence patches follow neighbors more strongly.
    # ------------------------------------------------------------------
    z_patch_reg = _patch_grid_regularization(
        mu_patch=mu_patch,
        conf_patch=confidence,
        lam=regularization_lambda,
        num_iters=regularization_iters,
        eps=weight_eps,
    )
    z_patch_reg = cp.clip(z_patch_reg, 0.0, float(Z - 1)).astype(cp.float32)

    # ------------------------------------------------------------------
    # Dense fusion by overlap-weighted voting
    # Each patch now votes with its regularized soft depth estimate.
    # ------------------------------------------------------------------
    yy_local = cp.arange(P, dtype=cp.int32)
    xx_local = cp.arange(P, dtype=cp.int32)

    y_idx = ys[:, None, None, None] + yy_local[None, None, :, None]
    x_idx = xs[None, :, None, None] + xx_local[None, None, None, :]

    y_idx = cp.broadcast_to(y_idx, (Ny, Nx, P, P))
    x_idx = cp.broadcast_to(x_idx, (Ny, Nx, P, P))

    linear_idx = (y_idx * W + x_idx).reshape(-1)

    vote_weight = cp.asarray(calculate.hann2d(P, P), dtype=cp.float32)[None, None, :, :]
    vote_weight = vote_weight * confidence[:, :, None, None]

    z_vote = vote_weight * z_patch_reg[:, :, None, None]

    z_accum = cp.zeros((H * W,), dtype=cp.float32)
    w_accum = cp.zeros((H * W,), dtype=cp.float32)

    cp.add.at(z_accum, linear_idx, z_vote.reshape(-1))
    cp.add.at(w_accum, linear_idx, vote_weight.reshape(-1))

    z_map = (z_accum / (w_accum + weight_eps)).reshape(H, W)

    global_fill = cp.sum(z_accum) / (cp.sum(w_accum) + weight_eps)
    z_map = cp.where(w_accum.reshape(H, W) > weight_eps, z_map, global_fill)

    # ------------------------------------------------------------------
    # Final XY smoothing on dense z-map
    # ------------------------------------------------------------------
    z_map_smooth = cupy_ndimage.gaussian_filter(z_map, sigma=smooth_sigma)
    z_map_smooth = cp.clip(z_map_smooth, 0.0, float(Z - 1)).astype(cp.float32)

    # ------------------------------------------------------------------
    # Build phase_init
    # ------------------------------------------------------------------
    Y, X = cp.meshgrid(
        cp.arange(H, dtype=cp.float32),
        cp.arange(W, dtype=cp.float32),
        indexing="ij"
    )

    phase_init = cp.zeros((H, W, 1, 3), dtype=cp.float32)
    phase_init[:, :, 0, 0] = Y
    phase_init[:, :, 0, 1] = X
    phase_init[:, :, 0, 2] = z_map_smooth

    if return_debug:
        debug = {
            "ys": ys.get(),
            "xs": xs.get(),
            "scores_raw": scores_raw.get(),
            "scores_smooth": scores_smooth.get(),
            "posterior": posterior.get(),
            "mu_patch": mu_patch.get(),
            "var_patch": var_patch.get(),
            "peak_idx": peak_idx.get(),
            "peak_val": peak_val.get(),
            "median_val": median_val.get(),
            "evidence": evidence.get(),
            "local_mass": local_mass.get(),
            "smoothness_quality": smoothness_quality.get(),
            "confidence": confidence.get(),
            "z_patch_reg": z_patch_reg.get(),
            "z_map_raw": z_map.get(),
        }
        return phase_init.get(), z_map_smooth.get(), debug

    return phase_init.get(), z_map_smooth.get()

def generate_continuous_H_gpu(stack, zRatio):
    """
    stack: cp.ndarray, shape (X,Y,Z)
    """
    stack_gpu = cp.asarray(stack)

    def H(coords_phys):
        coords = cp.asarray(coords_phys)
        coords_idx = coords.copy()
        coords_idx[..., 0] = coords[..., 0] / zRatio
        shape = coords_idx.shape
        coords_flat = coords_idx.reshape(-1, 3).T
        values = cupy_ndimage.map_coordinates(
            stack_gpu, coords_flat, order=3, mode="nearest"
        )
        return values.reshape(*shape[:-1])

    return H


def apply_H_to_matrix_gpu(A, H):
    """
    A: cp.ndarray, shape (X,Y,Z,3)
    H: interpolate function
    """
    coords = cp.asarray(A)
    shape = coords.shape
    coords_flat = coords.reshape(-1, 3)
    R = H(coords_flat)
    return R.reshape(shape[:-1])

def correctMotion(data_raw, motion_field):
    """
    Apply motion correction to raw data using the computed motion field.

    Args:
        data_raw (numpy.ndarray): The raw 3D data, shape (H, W, D).
        motion_field (numpy.ndarray): The computed motion field, shape (H, W, D, 3).

    Returns:
        data_tran (numpy.ndarray): The motion-corrected data, shape (H, W, D).
    """
    # Generate coordinate grid for the data
    grid = np.meshgrid(
        *[
            np.arange(n, dtype=np.float32) for n in data_raw.shape
        ],  # Create coordinate arrays
        indexing="ij",  # Use matrix indexing
        sparse=False,  # Return full grid
    )  # Returns tuple of 3 arrays, each shape: (H, W, D)

    # Compute corrected coordinates using motion field
    coords_new = interp.correctGrid(motion_field, grid)  # Shape: (H, W, D, 3)

    # Apply motion correction using 3D interpolation
    data_tran = correctMotionGrid(data_raw, coords_new)  # Shape: (H, W, D)

    # Convert to CPU if needed
    if hasattr(data_tran, "get"):
        data_tran = cp.asnumpy(data_tran)  # Convert GPU array to CPU
    else:
        data_tran = np.asarray(data_tran)  # Already CPU array

    return data_tran



# 1. Wrong-region detection helpers
def get_local_error_on_control_points(It, xG_grid, yG_grid, zG_grid,
                                      kernelsize=11, metric="mse"):
    """
    Compute local residual error on control points from a dense residual map It.

    Parameters
    ----------
    It : cp.ndarray, shape (x, y, z)
        Residual on moving grid.
    xG_grid, yG_grid, zG_grid : cp.ndarray
        Control-point meshgrid.
    kernelsize : int
        Patch size in xy plane.
    metric : str
        'mse' or 'mae'.

    Returns
    -------
    err_cp : cp.ndarray
        Local averaged error sampled on control points.
    err_dense : cp.ndarray
        Dense pixel-wise error map before averaging.
    """
    if metric.lower() == "mse":
        err_dense = It ** 2
    elif metric.lower() == "mae":
        err_dense = cp.abs(It)
    else:
        raise ValueError("metric must be 'mse' or 'mae'")

    kernel = cp.ones((kernelsize, kernelsize, 1), dtype=cp.float32) / (kernelsize ** 2)
    err_avg = calculate.imfilter(err_dense, kernel, "replicate", "same", "corr")
    err_cp = err_avg[xG_grid, yG_grid, zG_grid]
    return err_cp, err_dense


def detect_significant_mad(values, threshold=3.0):
    """
    MAD-based outlier detection on a flattened array.
    Returns indices into values.ravel().
    """
    arr = cp.asarray(values).ravel()
    if arr.size == 0:
        return cp.array([], dtype=cp.int64)

    med = cp.median(arr)
    mad = cp.median(cp.abs(arr - med))
    denom = 1.4826 * mad

    if denom == 0:
        if cp.any(arr != med):
            idx = cp.where(arr != med)[0]
        else:
            idx = cp.array([], dtype=cp.int64)
    else:
        z = cp.abs(arr - med) / denom
        idx = cp.where(z > threshold)[0]

    return idx.astype(cp.int64)
import cupy as cp
import cupyx.scipy.ndimage as ndi


def _normalize_vec_field(vx, vy, vz, eps=1e-6):
    norm = cp.sqrt(vx**2 + vy**2 + vz**2) + eps
    return vx / norm, vy / norm, vz / norm


def build_reference_trap_mask_from_bad_moving(
    bad_mask,
    phase_new,
    data_ref_layer,
    z_ratio_ref,
    seed_quantile=0.0,
    grow_radius_xy=7,
    grow_radius_z=11,
    sigma_intensity=0.15,
    sigma_grad=1.5,
    cos_thresh=0.5,
):
    """
    Build a structure-aware trap mask in reference space from bad moving locations.

    Parameters
    ----------
    bad_mask : cp.ndarray, bool, shape (x, y, z_mov)
        Bad region on moving grid.
    phase_new : cp.ndarray, shape (x, y, z_mov, 3)
        Current mapping from moving grid to reference coordinates.
    data_ref_layer : cp.ndarray, shape (x_ref, y_ref, z_ref)
        Reference image on current pyramid layer.
    z_ratio_ref : float
        Physical z anisotropy of reference on this layer.
    seed_quantile : float
        Optional threshold for seed sparsification, 0 means keep all bad points.
    grow_radius_xy : int
        Local growth radius in x/y on reference grid.
    grow_radius_z : int
        Local growth radius in z on reference grid.
    sigma_intensity : float
        Intensity similarity tolerance.
    sigma_grad : float
        Gaussian smoothing sigma before computing gradients.
    cos_thresh : float
        Gradient-direction similarity threshold.
    """
    x_ref, y_ref, z_ref = data_ref_layer.shape
    trap_mask_ref = cp.zeros((x_ref, y_ref, z_ref), dtype=cp.bool_)

    if not cp.any(bad_mask):
        return trap_mask_ref

    # 1) reference gradients
    ref_sm = ndi.gaussian_filter(data_ref_layer.astype(cp.float32), sigma=(sigma_grad, sigma_grad, sigma_grad))
    gx = ndi.sobel(ref_sm, axis=0) / 8.0
    gy = ndi.sobel(ref_sm, axis=1) / 8.0
    gz = ndi.sobel(ref_sm, axis=2) / 8.0
    gz = gz / max(z_ratio_ref, 1e-6)

    gx, gy, gz = _normalize_vec_field(gx, gy, gz)

    # 2) seeds from bad moving voxels -> reference coords
    seed_coords = phase_new[bad_mask]   # (N, 3)
    if seed_coords.shape[0] == 0:
        return trap_mask_ref

    seed_x = cp.rint(seed_coords[:, 0]).astype(cp.int32)
    seed_y = cp.rint(seed_coords[:, 1]).astype(cp.int32)
    seed_z = cp.rint(seed_coords[:, 2]).astype(cp.int32)

    keep = (
        (seed_x >= 0) & (seed_x < x_ref) &
        (seed_y >= 0) & (seed_y < y_ref) &
        (seed_z >= 0) & (seed_z < z_ref)
    )
    seed_x = seed_x[keep]
    seed_y = seed_y[keep]
    seed_z = seed_z[keep]

    if seed_x.size == 0:
        return trap_mask_ref

    
    seed_lin = seed_x * (y_ref * z_ref) + seed_y * z_ref + seed_z
    seed_lin = cp.unique(seed_lin)
    seed_x = seed_lin // (y_ref * z_ref)
    rem = seed_lin % (y_ref * z_ref)
    seed_y = rem // z_ref
    seed_z = rem % z_ref

    # 3) local structure-aware growth around each seed
    for i in range(seed_x.size):
        cx = int(seed_x[i])
        cy = int(seed_y[i])
        cz = int(seed_z[i])

        x0 = max(0, cx - grow_radius_xy)
        x1 = min(x_ref, cx + grow_radius_xy + 1)
        y0 = max(0, cy - grow_radius_xy)
        y1 = min(y_ref, cy + grow_radius_xy + 1)
        z0 = max(0, cz - grow_radius_z)
        z1 = min(z_ref, cz + grow_radius_z + 1)

        patch = data_ref_layer[x0:x1, y0:y1, z0:z1]

        # seed intensity
        I0 = data_ref_layer[cx, cy, cz]

        # seed gradient direction
        g0x = gx[cx, cy, cz]
        g0y = gy[cx, cy, cz]
        g0z = gz[cx, cy, cz]

        # local gradient directions
        pgx = gx[x0:x1, y0:y1, z0:z1]
        pgy = gy[x0:x1, y0:y1, z0:z1]
        pgz = gz[x0:x1, y0:y1, z0:z1]

        # intensity similarity
        sim_I = cp.exp(-((patch - I0) ** 2) / (2 * sigma_intensity ** 2 + 1e-6))

        # gradient direction similarity
        cos_sim = cp.abs(pgx * g0x + pgy * g0y + pgz * g0z)

        # physical anisotropic distance
        XX, YY, ZZ = cp.meshgrid(
            cp.arange(x0, x1),
            cp.arange(y0, y1),
            cp.arange(z0, z1),
            indexing="ij"
        )
        d2 = (
            (XX - cx) ** 2 +
            (YY - cy) ** 2 +
            ((ZZ - cz) / max(z_ratio_ref, 1e-6)) ** 2
        )

        # candidate region:
        # 1) intensity close to seed
        # 2) gradient direction reasonably aligned
        # 3) distance not too large
        local_mask = (sim_I > cp.exp(-0.5)) & (cos_sim > cos_thresh)

        local_mask = local_mask & (d2 <= (grow_radius_xy ** 2))

        trap_mask_ref[x0:x1, y0:y1, z0:z1] |= local_mask

    return trap_mask_ref

def get_wrong_regions_2d(err2d, r, xG, yG, x_threshold, y_threshold,
                         mad_threshold=3.0, min_component_size=2):
    """
    Detect connected high-error regions on one z slice of control-point error map.

    Parameters
    ----------
    err2d : cp.ndarray, shape (nx_cp, ny_cp)
        Error map on control points for one z slice.
    r : int
        Control-point radius.
    xG, yG : cp.ndarray
        1D control-point coordinates in image grid.
    x_threshold, y_threshold : int
        Image size bounds.
    mad_threshold : float
        MAD threshold for abnormal control points.
    min_component_size : int
        Minimum connected component size to keep.

    Returns
    -------
    region_coor : list[cp.ndarray]
        Each element = [min_y, max_y, min_x, max_x] in moving-image coordinates.
    connected_components : list[cp.ndarray]
        Connected abnormal control-point indices.
    """
    region_coor = []
    connected_components = []

    if err2d.size == 0 or cp.all(err2d == 0):
        return region_coor, connected_components

    rows, cols = err2d.shape

    if rows * cols > 144 and rows > 2 and cols > 2:
        inner_rows = cp.arange(1, rows - 1)
        inner_cols = cp.arange(1, cols - 1)
        R, C = cp.meshgrid(inner_rows, inner_cols, indexing="ij")
        flat_idx = detect_significant_mad(err2d[R, C], threshold=mad_threshold)
        sig_pts = cp.stack([R.ravel()[flat_idx], C.ravel()[flat_idx]], axis=1)
    else:
        flat_idx = detect_significant_mad(err2d, threshold=mad_threshold)
        if flat_idx.size == 0:
            return region_coor, connected_components
        sig_pts = cp.stack(cp.unravel_index(flat_idx, err2d.shape), axis=1)

    npts = int(sig_pts.shape[0])
    if npts == 0:
        return region_coor, connected_components

    # Build 4-neighbor adjacency
    adj = [[] for _ in range(npts)]
    for i in range(npts):
        for j in range(i + 1, npts):
            p1, p2 = sig_pts[i], sig_pts[j]
            if abs(int(p1[0] - p2[0])) + abs(int(p1[1] - p2[1])) == 1:
                adj[i].append(j)
                adj[j].append(i)

    visited = np.zeros(npts, dtype=bool)

    for i in range(npts):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        while stack:
            cur = stack.pop()
            if visited[cur]:
                continue
            visited[cur] = True
            comp.append(sig_pts[cur])
            for nb in adj[cur]:
                if not visited[nb]:
                    stack.append(nb)

        if len(comp) >= min_component_size:
            connected_components.append(cp.asarray(comp))

    dx = int(cp.ceil(1.5*r))

    for comp in connected_components:
        min_row, max_row = int(comp[:, 0].min()), int(comp[:, 0].max())
        min_col, max_col = int(comp[:, 1].min()), int(comp[:, 1].max())

        cp_minx = int(xG[min_row])
        cp_miny = int(yG[min_col])
        cp_maxx = int(xG[max_row])
        cp_maxy = int(yG[max_col])

        min_x = max(0, cp_minx - dx)
        min_y = max(0, cp_miny - dx)
        max_x = min(cp_maxx + dx, x_threshold - 1)
        max_y = min(cp_maxy + dx, y_threshold - 1)

        region_coor.append(cp.asarray([min_y, max_y, min_x, max_x], dtype=cp.int32))

    return region_coor, connected_components


def build_bad_region_mask_from_cp_error(err_cp, r, xG, yG, x_size, y_size,
                                        mad_threshold=3.0, min_component_size=2):
    """
    Convert control-point error volume into a dense bad-region mask on moving grid.

    Parameters
    ----------
    err_cp : cp.ndarray, shape (nx_cp, ny_cp, z)
        Error on control points.
    r, xG, yG : control-point geometry
    x_size, y_size : int
        Image size.
    mad_threshold, min_component_size : region-detection params

    Returns
    -------
    bad_mask : cp.ndarray, shape (x, y, z), bool
        Dense moving-grid mask: True = bad / exclude from data term.
    num_regions_total : int
        Total number of detected regions across z.
    """
    z_size = err_cp.shape[2]
    bad_mask = cp.zeros((x_size, y_size, z_size), dtype=cp.bool_)
    num_regions_total = 0

    for z_i in range(z_size):
        err2d = err_cp[:, :, z_i]
        regions, _ = get_wrong_regions_2d(
            err2d, r, xG, yG, x_size, y_size,
            mad_threshold=mad_threshold,
            min_component_size=min_component_size
        )
        num_regions_total += len(regions)

        for reg in regions:
            min_y, max_y, min_x, max_x = map(int, reg.tolist())
            bad_mask[min_x:max_x + 1, min_y:max_y + 1, z_i] = True

    return bad_mask, num_regions_total


def get_quantile_threshold(arr, q=0.95):
    """
    Robust quantile threshold on CuPy array.
    """
    flat = cp.asarray(arr).ravel()
    if flat.size == 0:
        return cp.asarray(0, dtype=cp.float32)
    k = int(cp.clip(cp.floor((flat.size - 1) * q), 0, flat.size - 1))
    part = cp.partition(flat, k)
    return part[k]


# =========================================================
# 2. Cross-resolution registration helpers
# =========================================================

def compose_phase_from_motion(phase_current, motion_current, zRatio, zRatio_hr):
    """
    Compose phase_current and motion_current into the final mapping phase_new.
    motion_current is defined in moving-grid coordinates.
    phase_new is defined in reference-grid coordinates.
    """
    phase_update = motion_current.copy()
    phase_update[..., 2] = phase_update[..., 2] * (zRatio_hr / zRatio)
    phase_new = phase_current + phase_update
    return phase_new


def resample_exclude_mask(mask, output_shape, threshold=0.5):
    """
    Resize exclusion mask.

    Parameters
    ----------
    mask : array
        Original exclusion mask, where:
        1 / True = exclude
        0 / False = keep
    output_shape : tuple
        Target shape.
    threshold : float
        Threshold after interpolation.

    Returns
    -------
    mask_out : cp.ndarray, bool
        True = exclude, False = keep
    """
    mask_rs = imresize(cp.asarray(mask, dtype=cp.float32), output_shape=output_shape)
    return mask_rs > threshold

def initialize_motion_for_layer(option, layer, layer_num, SZ, x, y, z, prev_motion=None):
    """
    Initialize motion field for current layer.
    """
    if layer == layer_num:
        if "motion" in option and option["motion"] is not None:
            motion_init = cp.asarray(option["motion"], dtype=cp.float32)
            motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
            motion_current[..., 0] = imresize(motion_init[..., 0], output_shape=(x, y, z)) / (SZ[0] / x)
            motion_current[..., 1] = imresize(motion_init[..., 1], output_shape=(x, y, z)) / (SZ[1] / y)
            motion_current[..., 2] = imresize(motion_init[..., 2], output_shape=(x, y, z)) / (SZ[2] / z)
        else:
            motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
    else:
        motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
        motion_current[..., 0] = imresize(prev_motion[..., 0], output_shape=(x, y, z), method="bilinear") * 2
        motion_current[..., 1] = imresize(prev_motion[..., 1], output_shape=(x, y, z), method="bilinear") * 2
        motion_current[..., 2] = imresize(prev_motion[..., 2], output_shape=(x, y, z), method="bilinear")
        motion_current = cp.asarray(motion_current, dtype=cp.float32)

    return motion_current


def initialize_phase_for_layer(option, SZ, x, y, z, zRatio, zRatio_hr):
    """
    Initialize phase field for current layer.
    """
    if "phase" in option and option["phase"] is not None:
        phase_init = cp.asarray(option["phase"], dtype=cp.float32)
        phase_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
        phase_current[..., 0] = imresize(phase_init[..., 0], output_shape=(x, y, z)) / (SZ[0] / x)
        phase_current[..., 1] = imresize(phase_init[..., 1], output_shape=(x, y, z)) / (SZ[1] / y)
        phase_current[..., 2] = imresize(phase_init[..., 2], output_shape=(x, y, z)) / (SZ[2] / z)
    else:
        X, Y, Z = cp.indices((x, y, z))
        phase_current = cp.stack([X, Y, Z * zRatio / zRatio_hr], axis=-1).astype(cp.float32)

    return phase_current


def make_control_point_grid(x, y, z, r):
    """
    Build control-point grid for LK update.
    """
    xG = cp.arange(r, x - 1, step=2 * r + 1)
    yG = cp.arange(r, y - 1, step=2 * r + 1)
    zG = cp.arange(0, z)
    xG_grid, yG_grid, zG_grid = cp.meshgrid(xG, yG, zG, indexing="ij")
    return xG, yG, zG, xG_grid, yG_grid, zG_grid


def compute_valid_mask_on_moving_grid(phase_new, H_mask_ref_layer, mask_mov_layer):
    """
    Parameters
    ----------
    phase_new : mapping from moving grid to reference continuous coordinates
    H_mask_ref_layer : continuous interpolator of reference exclusion mask
    mask_mov_layer : bool mask on moving grid, True means exclude

    Returns
    -------
    valid_mask : bool
        True means this moving-grid voxel is valid for data term.
    """
    # reference exclusion mask sampled onto moving grid
    if H_mask_ref_layer is None:
        ref_excluded = cp.zeros(mask_mov_layer.shape, dtype=cp.bool_)
    else:
        ref_excluded = apply_H_to_matrix_gpu(phase_new, H_mask_ref_layer) > 0.5

    mov_excluded = mask_mov_layer > 0
    valid_mask = (~mov_excluded) & (~ref_excluded)
    return valid_mask


# =========================================================
# 3. Core optimizer for one layer
# =========================================================

def optimize_layer_cross_resolution(
    data_mov_layer,
    data_ref_layer,
    H_ref_layer,
    H_mask_ref_layer,
    mask_mov_layer,
    phase_current,
    motion_init,
    xG_grid,
    yG_grid,
    zG_grid,
    r,
    iterNum,
    movRange,
    smoothPenalty,
    smoothPenaltySum,
    zRatio,
    zRatio_hr,
    tol=1e-3,
    verbose=False,
    layer=None,
):
    """
    Optimize one pyramid layer in the cross-resolution setting.

    Returns
    -------
    result : dict
        {
            "motion": ...,
            "phase_new": ...,
            "data_ref_sampled": ...,
            "residual": ...,
            "error_init": ...,
            "error_last": ...,
            "current_error": ...,
            "diff_error": ...,
            "penalty_error": ...,
            "valid_mask": ...
        }
    """
    x, y, z = data_mov_layer.shape
    motion_current = motion_init.copy()
    oldError = cp.inf * cp.ones(3, dtype=cp.float32)

    error_init = None
    error_last = None
    phase_new = None
    data_ref_sampled = None
    valid_mask = None
    diffError = None
    penaltyError = None
    currentError = None

    AverageFilter = cp.ones((2 * r + 1, 2 * r + 1, 1), dtype=cp.float32)

    for it in range(iterNum):
        old_motion = motion_current.copy()

        phase_new = compose_phase_from_motion(phase_current, motion_current, zRatio, zRatio_hr)
        data_ref_sampled = apply_H_to_matrix_gpu(phase_new, H_ref_layer)
        valid_mask = compute_valid_mask_on_moving_grid(phase_new, H_mask_ref_layer, mask_mov_layer)

        residual = data_mov_layer - data_ref_sampled
        residual = calculate.imfilter(
            residual, cp.ones((3, 3, 1), dtype=cp.float32) / 9,
            "replicate", "same", "corr"
        )
        residual[~valid_mask] = 0

        if it == 0:
            error_init = residual ** 2

        error_last = residual ** 2

        neiDiff = getNeiDiff(motion_current[xG_grid, yG_grid, zG_grid, :], 1)
        neiSum = smoothPenaltySum * neiDiff

        diffError, penaltyError = calError(residual, neiDiff, smoothPenaltySum)
        currentError = diffError + penaltyError

        if verbose:
            print(
                f"[layer={layer}] iter={it:02d} "
                f"error={float(currentError):.4f} "
                f"diff={float(diffError):.4f} "
                f"penalty={float(penaltyError):.4f}"
            )

        # stopping
        if it == iterNum - 1:
            break
        if cp.sum(oldError <= currentError) > 1:
            if verbose:
                print(f"[layer={layer}] stop: error increased repeatedly")
            break
        if cp.abs(oldError[-1] - currentError) < tol:
            if verbose:
                print(f"[layer={layer}] stop: error change below tol")
            break

        oldError[:-1] = oldError[1:]
        oldError[-1] = currentError

        # spatial gradient on reference, evaluated at phase_new
        Ix, Iy, Iz = getSpatialGradientInOrgGrid(data_ref_layer, phase_new)
        Ix[~valid_mask] = 0
        Iy[~valid_mask] = 0
        Iz[~valid_mask] = 0
        Iz /= zRatio_hr

        Ixx = calculate.imfilter(Ix ** 2, AverageFilter, "replicate", "same", "corr")
        Ixy = calculate.imfilter(Ix * Iy, AverageFilter, "replicate", "same", "corr")
        Ixz = calculate.imfilter(Ix * Iz, AverageFilter, "replicate", "same", "corr")
        Iyy = calculate.imfilter(Iy ** 2, AverageFilter, "replicate", "same", "corr")
        Iyz = calculate.imfilter(Iy * Iz, AverageFilter, "replicate", "same", "corr")
        Izz = calculate.imfilter(Iz ** 2, AverageFilter, "replicate", "same", "corr")
        Ixt = calculate.imfilter(Ix * residual, AverageFilter, "replicate", "same", "corr")
        Iyt = calculate.imfilter(Iy * residual, AverageFilter, "replicate", "same", "corr")
        Izt = calculate.imfilter(Iz * residual, AverageFilter, "replicate", "same", "corr")

        Ixx = Ixx[xG_grid, yG_grid, zG_grid]
        Ixy = Ixy[xG_grid, yG_grid, zG_grid]
        Ixz = Ixz[xG_grid, yG_grid, zG_grid]
        Iyy = Iyy[xG_grid, yG_grid, zG_grid]
        Iyz = Iyz[xG_grid, yG_grid, zG_grid]
        Izz = Izz[xG_grid, yG_grid, zG_grid]
        Ixt = Ixt[xG_grid, yG_grid, zG_grid]
        Iyt = Iyt[xG_grid, yG_grid, zG_grid]
        Izt = Izt[xG_grid, yG_grid, zG_grid]

        motion_update = getFlow3_withPenalty6(
            Ixx, Ixy, Ixz, Iyy, Iyz, Izz,
            Ixt, Iyt, Izt,
            smoothPenaltySum, neiSum
        )

        # step clipping
        motion_norm = cp.sqrt(cp.sum(motion_update ** 2, axis=3))
        motion_norm = cp.maximum(motion_norm / movRange, 1.0)
        motion_update = motion_update / motion_norm[..., cp.newaxis]

        motion_current_CP = motion_current[xG_grid, yG_grid, zG_grid, :] + motion_update

        grid_dense = cp.meshgrid(
            *[cp.arange(n, dtype=cp.float32) for n in data_mov_layer.shape],
            indexing="ij",
            sparse=False,
        )
        coords_new = compute_new_grid(grid_dense, r, motion_current_CP.shape)

        for d in range(3):
            temp_phi = cp.asarray(motion_current_CP[..., d])
            motion_current[..., d] = interp.interp3Grid(temp_phi, coords_new).reshape(x, y, z)

        max_diff_motion = cp.max(cp.abs(motion_current - old_motion))
        if max_diff_motion < tol:
            if verbose:
                print(f"[layer={layer}] stop: motion update below tol")
            break
        # visualization.visualize_2d_image(data_ref_sampled.get(),title = "[iter {iter}]: mapping image")
            
    # final recompute
    phase_new = compose_phase_from_motion(phase_current, motion_current, zRatio, zRatio_hr)
    data_ref_sampled = apply_H_to_matrix_gpu(phase_new, H_ref_layer)
    valid_mask = compute_valid_mask_on_moving_grid(phase_new, H_mask_ref_layer, mask_mov_layer)

    ##################################################################################################
    # visualization.visualize_2d_image(valid_mask.get(),autocontrast=False,title = f"[layer:{layer}] valid mask")
    ##################################################################################################

    residual = data_mov_layer - data_ref_sampled
    residual = calculate.imfilter(
        residual, cp.ones((3, 3, 1), dtype=cp.float32) / 9,
        "replicate", "same", "corr"
    )
    residual[~valid_mask] = 0
    error_last = residual ** 2

    neiDiff = getNeiDiff(motion_current[xG_grid, yG_grid, zG_grid, :], 1)
    diffError, penaltyError = calError(residual, neiDiff, smoothPenaltySum)
    currentError = diffError + penaltyError

    return {
        "motion": motion_current,
        "phase_new": phase_new,
        "data_ref_sampled": data_ref_sampled,
        "residual": residual,
        "error_init": error_init if error_init is not None else error_last.copy(),
        "error_last": error_last,
        "current_error": currentError,
        "diff_error": diffError,
        "penalty_error": penaltyError,
        "valid_mask": valid_mask,
    }


# =========================================================
# 4. Wrong-region correction for one layer
# =========================================================

def correct_wrong_regions_one_layer(
    data_mov_layer,
    data_ref_layer,
    H_ref_layer,
    H_mask_ref_layer,
    mask_mov_layer,
    mask_ref_layer,
    phase_current,
    motion_init_layer,
    xG,
    yG,
    xG_grid,
    yG_grid,
    zG_grid,
    r,
    iterNum,
    movRange,
    smoothPenalty,
    smoothPenaltySum,
    zRatio,
    zRatio_hr,
    error_metric="mse",
    mad_threshold=3.0,
    min_component_size=2,
    bad_region_exclude_mode="direct",
    verbose=False,
    layer=None,
):
    """
    Run normal optimization, detect bad regions, mask them out, rerun, and accept
    correction only if final objective improves.

    bad_region_exclude_mode:
        - "direct": exclude all detected bad regions
        - "highresidual": exclude only the top residual fraction inside bad regions
    """
    # -----------------------------------------------------
    # Pass 1: normal optimization
    # -----------------------------------------------------
    res0 = optimize_layer_cross_resolution(
        data_mov_layer=data_mov_layer,
        data_ref_layer=data_ref_layer,
        H_ref_layer=H_ref_layer,
        H_mask_ref_layer=H_mask_ref_layer,
        mask_mov_layer=mask_mov_layer,
        phase_current=phase_current,
        motion_init=motion_init_layer,
        xG_grid=xG_grid,
        yG_grid=yG_grid,
        zG_grid=zG_grid,
        r=r,
        iterNum=iterNum,
        movRange=movRange,
        smoothPenalty=smoothPenalty,
        smoothPenaltySum=smoothPenaltySum,
        zRatio=zRatio,
        zRatio_hr=zRatio_hr,
        verbose=verbose,
        layer=layer,
    )
    motion0 = res0["motion"]
    current_error0 = res0["current_error"]
    residual0 = res0["residual"]
    error_last0 = res0["error_last"]
    valid_mask0 = res0["valid_mask"]

    # -----------------------------------------------------
    # Detect bad regions on moving grid from residual
    # -----------------------------------------------------
    err_cp, _ = get_local_error_on_control_points(
        residual0, xG_grid, yG_grid, zG_grid,
        kernelsize=2 * r + 1,
        metric=error_metric
    )

    bad_mask, num_regions = build_bad_region_mask_from_cp_error(
        err_cp, r, xG, yG,
        x_size=data_mov_layer.shape[0],
        y_size=data_mov_layer.shape[1],
        mad_threshold=mad_threshold,
        min_component_size=min_component_size
    )
    # visualization.visualize_2d_image(bad_mask.get(),autocontrast=False)
    if verbose:
        print(f"[layer={layer}] detected wrong regions: {num_regions}")

    if num_regions == 0:
        return res0

    # Optionally refine bad-mask with dense residual threshold inside detected regions
    if bad_region_exclude_mode == "highresidual":
        dense_err = error_last0.copy()
        dense_err[~bad_mask] = 0
        th = get_quantile_threshold(dense_err[bad_mask], q=0.7) if cp.any(bad_mask) else 0
        bad_mask = bad_mask & (dense_err >= th)
        
    # visualization.visualize_2d_image(bad_mask.get(),autocontrast=False)
    # Only exclude regions that were originally valid in moving mask
    bad_mask = bad_mask & valid_mask0

    trap_mask_ref = build_reference_trap_mask_from_bad_moving(
        bad_mask=bad_mask,
        phase_new=res0["phase_new"],
        data_ref_layer=data_ref_layer,
        z_ratio_ref=zRatio_hr,
        grow_radius_xy=7,
        grow_radius_z=11,
        sigma_intensity=0.15,
        sigma_grad=1.5,
        cos_thresh=0.5,
    )

    corrected_mask_ref = mask_ref_layer | trap_mask_ref

    H_mask_ref_layer_corrected = generate_continuous_H_gpu(
        corrected_mask_ref.astype(cp.float32),
        zRatio=1
    )

    n_valid_old = int(cp.sum(valid_mask0).item())
    valid_mask1 = compute_valid_mask_on_moving_grid(
        res0["phase_new"],
        H_mask_ref_layer_corrected,
        mask_mov_layer
    )
    n_valid_new = int(cp.sum(valid_mask1).item())
    valid_ratio = n_valid_new / max(n_valid_old, 1)
    if verbose:
        print(f"[layer={layer}] mask kept ratio after correction: {valid_ratio:.3f}")

    if valid_ratio < 0.4:
        if verbose:
            print(f"[layer={layer}] skip correction: corrected mask too small")
        return res0

    # -----------------------------------------------------
    # Pass 2: robust rerun using corrected mask
    # -----------------------------------------------------
    res1 = optimize_layer_cross_resolution(
        data_mov_layer=data_mov_layer,
        data_ref_layer=data_ref_layer,
        H_ref_layer=H_ref_layer,
        H_mask_ref_layer=H_mask_ref_layer_corrected,
        mask_mov_layer=mask_mov_layer,
        phase_current=phase_current,
        motion_init=motion_init_layer,
        xG_grid=xG_grid,
        yG_grid=yG_grid,
        zG_grid=zG_grid,
        r=r,
        iterNum=iterNum,
        movRange=movRange,
        smoothPenalty=smoothPenalty,
        smoothPenaltySum=smoothPenaltySum,
        zRatio=zRatio,
        zRatio_hr=zRatio_hr,
        verbose=verbose,
        layer=layer,
    )
    
    # -----------------------------------------------------
    # Pass 3: refinement with original moving mask
    # -----------------------------------------------------
    res2 = optimize_layer_cross_resolution(
        data_mov_layer=data_mov_layer,
        data_ref_layer=data_ref_layer,
        H_ref_layer=H_ref_layer,
        H_mask_ref_layer=H_mask_ref_layer,
        mask_mov_layer=mask_mov_layer,
        phase_current=phase_current,
        motion_init=res1["motion"],
        xG_grid=xG_grid,
        yG_grid=yG_grid,
        zG_grid=zG_grid,
        r=r,
        iterNum=iterNum,
        movRange=movRange,
        smoothPenalty=smoothPenalty,
        smoothPenaltySum=smoothPenaltySum,
        zRatio=zRatio,
        zRatio_hr=zRatio_hr,
        verbose=verbose,
        layer=layer,
    )
    # visualization.visualize_2d_image(res0["data_ref_sampled"].get(),title="first processed")
    # visualization.visualize_2d_image(res1["data_ref_sampled"].get(),title="after mask")
    # visualization.visualize_2d_image(res2["data_ref_sampled"].get(),title="removed mask")
    current_error2 = res2["current_error"]

    if verbose:
        print(
            f"[layer={layer}] compare original vs corrected: "
            f"{float(current_error0):.4f} -> {float(current_error2):.4f}"
        )

    if current_error2 < current_error0:
        if verbose:
            print(f"[layer={layer}] accept corrected result")
        return res2
    else:
        if verbose:
            print(f"[layer={layer}] keep original result")
        return res0


# =========================================================
# 5. Main public function
# =========================================================

def getMotion_v2(data_mov, data_ref, option, verbose=False):
    """
    Cross-resolution registration with wrong-region correction.

    Parameters
    ----------
    data_mov : np.ndarray or cp.ndarray
        Moving image, shape (H, W, D_mov)
    data_ref : np.ndarray or cp.ndarray
        Reference image, shape (H_hr, W_hr, D_ref)
    option : dict
        Required fields:
            mask_ref, mask_mov, layer, iter, r, zRatio, zRatio_HR,
            smoothPenalty, movRange
        Optional:
            motion, phase, tol,
            wrong_region_enable,
            wrong_region_metric,
            wrong_region_mad_threshold,
            wrong_region_min_component_size,
            wrong_region_exclude_mode
    verbose : bool

    Returns
    -------
    phase_new : np.ndarray
    motion_current : np.ndarray
    data_ref_sampled : np.ndarray
    """
    # -----------------------------------------------------
    # Options
    # -----------------------------------------------------
    option["mask_ref"] = cp.asarray(option["mask_ref"], dtype=cp.float32)
    option["mask_mov"] = cp.asarray(option["mask_mov"], dtype=cp.float32)

    layer_num = int(option["layer"])
    iterNum = int(option["iter"])
    r = int(option["r"])
    zRatio_raw = float(option["zRatio"])
    zRatio_HR = float(option["zRatio_HR"])
    smoothPenalty = float(option["smoothPenalty"])
    movRange = float(option.get("movRange", 5.0))
    tol = float(option.get("tol", 1e-3))

    wrong_region_enable = bool(option.get("wrong_region_enable", True))
    wrong_region_metric = option.get("wrong_region_metric", "mse")
    wrong_region_mad_threshold = float(option.get("wrong_region_mad_threshold", 3.0))
    wrong_region_min_component_size = int(option.get("wrong_region_min_component_size", 2))
    wrong_region_exclude_mode = option.get("wrong_region_exclude_mode", "direct")

    data_mov = cp.asarray(data_mov, dtype=cp.float32)
    data_ref = cp.asarray(data_ref, dtype=cp.float32)

    SZ = data_mov.shape
    SZ_HR = data_ref.shape

    motion_current = None
    phase_new = None
    data_ref_sampled = None

    # -----------------------------------------------------
    # Pyramid: coarse -> fine
    # -----------------------------------------------------
    for layer in range(layer_num, -1, -1):
        if verbose:
            print(f"\n========== start layer {layer}/{layer_num} ==========")

        # moving resolution
        x = int(SZ[0] / (2 ** layer))
        y = int(SZ[1] / (2 ** layer))
        z = SZ[2]

        # reference resolution
        x_hr = int(SZ_HR[0] / (2 ** layer))
        y_hr = int(SZ_HR[1] / (2 ** layer))
        z_hr = SZ_HR[2]

        data_mov_layer = imresize(data_mov, output_shape=(x, y, z))
        data_ref_layer = imresize(data_ref, output_shape=(x_hr, y_hr, z_hr))

        mask_mov_layer = resample_exclude_mask(option["mask_mov"], output_shape=(x, y, z))
        mask_ref_layer = resample_exclude_mask(option["mask_ref"], output_shape=(x_hr, y_hr, z_hr))

        zRatio = zRatio_raw / (2 ** layer)
        zRatio_hr = zRatio_HR / (2 ** layer)

        H_ref_layer = generate_continuous_H_gpu(data_ref_layer, zRatio=1)
        H_mask_ref_layer = generate_continuous_H_gpu(mask_ref_layer.astype(cp.float32), zRatio=1)

        # init motion / phase
        motion_init_layer = initialize_motion_for_layer(
            option=option,
            layer=layer,
            layer_num=layer_num,
            SZ=SZ,
            x=x, y=y, z=z,
            prev_motion=motion_current
        )
        phase_current = initialize_phase_for_layer(
            option=option,
            SZ=SZ,
            x=x, y=y, z=z,
            zRatio=zRatio,
            zRatio_hr=zRatio_hr
        )

        # keep a copy for possible correction restart
        motion_init_this_layer = motion_init_layer.copy()

        patchConnectNum = (2 * r + 1) ** 2
        smoothPenaltySum = smoothPenalty * patchConnectNum

        xG, yG, zG, xG_grid, yG_grid, zG_grid = make_control_point_grid(x, y, z, r)

        if wrong_region_enable:
            result = correct_wrong_regions_one_layer(
                data_mov_layer=data_mov_layer,
                data_ref_layer=data_ref_layer,
                H_ref_layer=H_ref_layer,
                H_mask_ref_layer=H_mask_ref_layer,
                mask_mov_layer=mask_mov_layer,
                mask_ref_layer=mask_ref_layer,
                phase_current=phase_current,
                motion_init_layer=motion_init_this_layer,
                xG=xG,
                yG=yG,
                xG_grid=xG_grid,
                yG_grid=yG_grid,
                zG_grid=zG_grid,
                r=r,
                iterNum=iterNum,
                movRange=movRange,
                smoothPenalty=smoothPenalty,
                smoothPenaltySum=smoothPenaltySum,
                zRatio=zRatio,
                zRatio_hr=zRatio_hr,
                error_metric=wrong_region_metric,
                mad_threshold=wrong_region_mad_threshold,
                min_component_size=wrong_region_min_component_size,
                bad_region_exclude_mode=wrong_region_exclude_mode,
                verbose=verbose,
                layer=layer,
            )
        else:
            result = optimize_layer_cross_resolution(
                data_mov_layer=data_mov_layer,
                data_ref_layer=data_ref_layer,
                H_ref_layer=H_ref_layer,
                H_mask_ref_layer=H_mask_ref_layer,
                mask_mov_layer=mask_mov_layer,
                phase_current=phase_current,
                motion_init=motion_init_this_layer,
                xG_grid=xG_grid,
                yG_grid=yG_grid,
                zG_grid=zG_grid,
                r=r,
                iterNum=iterNum,
                movRange=movRange,
                smoothPenalty=smoothPenalty,
                smoothPenaltySum=smoothPenaltySum,
                zRatio=zRatio,
                zRatio_hr=zRatio_hr,
                tol=tol,
                verbose=verbose,
                layer=layer,
            )

        motion_current = result["motion"]
        phase_new = result["phase_new"]
        data_ref_sampled = result["data_ref_sampled"]

    # -----------------------------------------------------
    # Return as numpy
    # -----------------------------------------------------
    if hasattr(motion_current, "get"):
        motion_current_np = cp.asnumpy(motion_current)
    else:
        motion_current_np = np.asarray(motion_current)

    if hasattr(phase_new, "get"):
        phase_new_np = cp.asnumpy(phase_new)
    else:
        phase_new_np = np.asarray(phase_new)

    if hasattr(data_ref_sampled, "get"):
        data_ref_sampled_np = cp.asnumpy(data_ref_sampled)
    else:
        data_ref_sampled_np = np.asarray(data_ref_sampled)

    return phase_new_np, motion_current_np, data_ref_sampled_np
