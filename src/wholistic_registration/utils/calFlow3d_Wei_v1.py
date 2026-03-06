"""

version : 0.3
file name : calFlow3d_Wei_v1.py

Alghothm Author : Wei Zheng (Vigirnia Tech) , Virginia M.S.(HHMI)
Code Author : Wei Zheng for matlab and Yunfeng Chi (Tsinghua University) for python
Last Update Date : 2025/4/10


Overview:
    This script implements motion correction and related opreations using a multi-scale approach.
    The primary goal is to registrate the moving image to a reference image using a GPU-accelerated method.
    Our alghothm is based on optical flow methods.We use a Anisotropic(only x and y directions have scaling change) Pyramid method to capture the large displayment between the moving image and the reference image.
    At the same time,we assume the motion field is smooth and continuous. So we calculate the motion of control points with the LK method. At the same time we add a smoothness penalty to the objective function,so that we can get a smooth motion field on each voxel.
    Eventually,we construct an iterative method to get the final motion field.

    
Functions:
    - correctMotionGrid: Correct the motion using 3D interpolation for GPU arrays. Given raw data and the motion field, we can get the data at the new coordinates.
    - getNeiDiff: Calculate the neighbor difference using a filter. This is used to add a smoothness penalty to the objective function.
    - calError: Calculate the error and penalty terms for the given 3D data. This is used to check the convergence of the algorithm.
    - getSpatialGradientInOrgGrid: Calculate the spatial gradient in the original image using 3D interpolation.
    - getFlow3_withPenalty6: Compute the flow with penalty and 3x3 matrix determinant.
    - compute_new_grid:do the normalization to the origin grid to let the coordinates of control points are integer.
    - getMotion: Function to compute motion correction using multi-scale approach.


"""

from scipy.ndimage import zoom, filters
import numpy as np
from . import cp
from . import interp
from . import calculate
from . import visualization
from .imresize import imresize


def correctMotionGrid(data_raw, coords_new):
    """
    Correct the motion using 3D interpolation for GPU arrays.

    Args:
        data_raw (cupy.ndarray): The raw 3D data (GPU array), shape (H, W, D).
        coords_new (cupy.ndarray): New grid coordinates for interpolation (GPU array), shape (H, W, D, 3).

    Returns:
        dat_corrected (cupy.ndarray): The corrected 3D data after interpolation (GPU array), shape (H, W, D).
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
    dat_corrected = cp.reshape(data1_tran, (x, y, z))  # Shape: (H, W, D)

    return dat_corrected


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


def getMotion(dat_mov, dat_ref, option, verbose=False):
    """
    Function to compute motion correction using multi-scale approach.

    Args:
        dat_mov (cupy.ndarray): The moving image data, shape (H, W, D).
        dat_ref (cupy.ndarray): The reference image data, shape (H, W, D).
        smoothPenalty_raw (float): The smooth penalty parameter.
        option (dict): Dictionary containing parameters for the method.

    Returns:
        motion_current (cupy.ndarray): The computed motion fields, shape (H, W, D, 3).
        currentError (float): The final error.
        coordinate_new (cupy.ndarray): The corrected indices for each layer, shape (H, W, D, 3).
    """
    # Initialize error logging dictionary to track convergence
    error_log = {}

    # Convert masks to GPU arrays and ensure float32 precision
    option["mask_ref"] = cp.asarray(
        option["mask_ref"], dtype=cp.float32
    )  # Shape: (H, W, D)
    option["mask_mov"] = cp.asarray(
        option["mask_mov"], dtype=cp.float32
    )  # Shape: (H, W, D)

    # Extract parameters from option dictionary
    layer_num = option["layer"]  # Number of pyramid layers
    iterNum = option["iter"]  # Number of iterations per layer
    r = option["r"]  # Filter radius
    zRatio_raw = option["zRatio"]  # Z-axis scaling ratio

    SZ = dat_mov.shape  # Original image dimensions
    movRange = option["movRange"]  # Maximum motion range for regularization

    # Multi-scale processing: start from coarsest level and refine
    for layer in range(layer_num, -1, -1):
        # Initialize error tracking for this layer
        error_log[f"layer_{layer}"] = {}
        error_log[f"layer_{layer}"]["diffError"] = []
        error_log[f"layer_{layer}"]["penaltyError"] = []
        error_log[f"layer_{layer}"]["currentError"] = []
        error_log[f"layer_{layer}"]["motion_current"] = []
        error_log[f"layer_{layer}"]["data_mov_corrected"] = []
        error_log[f"layer_{layer}"]["max_diff_motion"] = []
        

        if verbose:
            print(f"starting layer {layer} out of {layer_num}")

        # Calculate dimensions for current pyramid level
        x = int(SZ[0] / (2**layer))  # Downsampled width
        y = int(SZ[1] / (2**layer))  # Downsampled height
        z = SZ[2]  # Keep original depth

        # Downsample images to current pyramid level
        data_mov_layer = imresize(
            cp.asarray(dat_mov), output_shape=(x, y, z)
        )  # Shape: (x, y, z) # bicubic by default
        data_reference_layer = imresize(
            cp.asarray(dat_ref), output_shape=(x, y, z)
        )  # Shape: (x, y, z)
        error_log[f"layer_{layer}"]["data_ref"] = data_reference_layer
        error_log[f"layer_{layer}"]["data_mov"] = data_mov_layer

        # Update dimensions after downsampling
        x, y, z = data_mov_layer.shape  # Updated dimensions for current level

        # Scale z-ratio for current pyramid level
        zRatio = zRatio_raw / (2**layer)

        # Initialize motion field for current layer
        if layer == layer_num:  # If at the coarsest level
            if "motion" in option and option["motion"] is not None:
                # Use provided initial motion field
                motion_current = cp.zeros(
                    (x, y, z, 3), dtype=cp.float32
                )  # Shape: (x, y, z, 3)
                motion_init = cp.array(
                    option["motion"], dtype=cp.float32
                )  # Shape: (H, W, D, 3)

                # Downsample and scale the initial motion field
                motion_current[:, :, :, 0] = cp.asarray(
                    imresize(motion_init[:, :, :, 0], output_shape=(x, y, z))
                    / (SZ[0] / x)
                )  # Scale x-component
                motion_current[:, :, :, 1] = cp.asarray(
                    imresize(motion_init[:, :, :, 1], output_shape=(x, y, z))
                    / (SZ[1] / y)
                )  # Scale y-component
                motion_current[:, :, :, 2] = cp.asarray(
                    imresize(motion_init[:, :, :, 2], output_shape=(x, y, z))
                    / (SZ[2] / z)
                )  # Scale z-component
            else:
                # Start with zero motion field
                motion_current = cp.zeros(
                    (x, y, z, 3), dtype=cp.float32
                )  # Shape: (x, y, z, 3)
        else:
            # Upsample motion field from previous (finer) level
            motion_current_temp = cp.asarray(
                motion_current
            )  # Shape: (prev_x, prev_y, prev_z, 3)
            motion_current = cp.zeros(
                (x, y, z, 3), dtype=cp.float32
            )  # Shape: (x, y, z, 3)

            # Upsample and scale motion components (x2 for x,y due to pyramid structure)
            motion_current[:, :, :, 0] = cp.asarray(
                imresize(
                    motion_current_temp[:, :, :, 0],
                    output_shape=(x, y, z),
                    method="bilinear",
                )
                * 2
            )  # Scale x-component
            motion_current[:, :, :, 1] = cp.asarray(
                imresize(
                    motion_current_temp[:, :, :, 1],
                    output_shape=(x, y, z),
                    method="bilinear",
                )
                * 2
            )  # Scale y-component
            motion_current[:, :, :, 2] = cp.asarray(
                imresize(
                    motion_current_temp[:, :, :, 2],
                    output_shape=(x, y, z),
                    method="bilinear",
                )
            )  # Scale z-component
            motion_current = cp.asarray(
                motion_current, dtype=cp.float32
            )  # Shape: (x, y, z, 3)

        # Generate coordinate grid for current level
        grid = cp.meshgrid(
            *[
                cp.arange(n, dtype=cp.float32) for n in data_mov_layer.shape
            ],  # Create coordinate arrays
            indexing="ij",  # Use matrix indexing (row, column)
            sparse=False,  # Return full grid
        )  # Returns tuple of 3 arrays, each shape: (x, y, z)

        # Downsample masks to current pyramid level
        mask_ref = (
            imresize(option["mask_ref"], output_shape=(x, y, z)) > 0
        )  # Shape: (x, y, z)
        mask_mov = imresize(
            option["mask_mov"], output_shape=(x, y, z)
        )  # Shape: (x, y, z)

        # Initialize error tracking array for convergence checking
        oldError = cp.inf * cp.ones(3)  # Shape: (3,) - track last 3 errors

        # Set up penalty parameters
        smoothPenalty = option["smoothPenalty"]  # Smoothness penalty weight
        patchConnectNum = (r * 2 + 1) ** 2  # Number of connected patches
        smoothPenaltySum = smoothPenalty * patchConnectNum  # Total penalty weight

        # Define control point grid (sparse grid for motion computation)
        xG = cp.arange(r, x - 1, step=2 * r + 1)  # Control points in x direction
        yG = cp.arange(r, y - 1, step=2 * r + 1)  # Control points in y direction
        zG = cp.arange(0, z)  # All z positions
        xG_grid, yG_grid, zG_grid = cp.meshgrid(
            xG, yG, zG, indexing="ij"
        )  # Shape: (len(xG), len(yG), len(zG))

        # Main iteration loop for motion estimation
        for iter in range(iterNum):
            old_motion = motion_current.copy()
            # Apply current motion field to get warped moving image
            coords_new = interp.correctGrid(motion_current, grid)  # Shape: (x, y, z, 3) # add grid coordinates to motion
            data_mov_corrected = correctMotionGrid(data_mov_layer, coords_new)  # Shape: (x, y, z) # get data1 at new coordinates/motion control points
            
            # Save motion field periodically for logging
            if iter % option["save_ite"] == 0:
                # Convert GPU arrays to CPU before storing to avoid GPU memory accumulation
                if hasattr(data_mov_corrected, "get"):
                    error_log[f"layer_{layer}"]["data_mov_corrected"].append(
                        cp.asnumpy(data_mov_corrected).copy()
                    )
                else:
                    error_log[f"layer_{layer}"]["data_mov_corrected"].append(
                        np.asarray(data_mov_corrected).copy()
                    )
                if hasattr(motion_current, "get"):
                    error_log[f"layer_{layer}"]["motion_current"].append(
                        cp.asnumpy(motion_current).copy()
                    )  # Convert to CPU
                else:
                    error_log[f"layer_{layer}"]["motion_current"].append(
                        np.asarray(motion_current).copy()
                    )  # Already CPU

            # Apply motion to moving mask and combine with reference mask
            mask_mov_current = (
                correctMotionGrid(mask_mov, coords_new) > 0
            )  # Shape: (x, y, z)
            mask = mask_mov_current | mask_ref  # Shape: (x, y, z) - combined mask

            # Compute temporal difference between reference and warped moving image
            It = data_reference_layer - data_mov_corrected  # Shape: (x, y, z) # temporal difference

            # Apply spatial smoothing to temporal difference
            It = calculate.imfilter(
                It, cp.ones((3, 3, 1)) / 9, "replicate", "same", "corr"
            )  # Shape: (x, y, z)

            # Zero out masked regions (where we don't want to compute motion)
            It[mask] = 0  # Shape: (x, y, z)

            # Compute neighbor motion differences for smoothness constraint
            neiDiff = getNeiDiff(
                motion_current[xG_grid, yG_grid, zG_grid, :], 1
            )  # Shape: (len(xG), len(yG), len(zG), 3)

            # Scale z-component by z-ratio to account for anisotropic voxels
            neiDiff[:, :, :, 2] = (
                neiDiff[:, :, :, 2] * zRatio
            )  # Shape: (len(xG), len(yG), len(zG), 3)

            # Compute smoothness penalty term
            neiSum = smoothPenaltySum * neiDiff  # Shape: (len(xG), len(yG), len(zG), 3)

            # Calculate error terms for convergence checking
            diffError, penaltyError = calError(
                It, neiDiff, smoothPenaltySum
            )  # Both scalar values
            currentError = diffError + penaltyError  # Total error

            # Log errors for this iteration
            error_log[f"layer_{layer}"]["diffError"].append(diffError)
            error_log[f"layer_{layer}"]["penaltyError"].append(penaltyError)
            error_log[f"layer_{layer}"]["currentError"].append(currentError)
            

            if verbose:
                print(
                    f"Downsample layer: {layer}\tIter: {iter}\tError: {currentError:.3f}, Diff Error: {diffError:.3f}, Penalty Error: {penaltyError:.3f}"
                )

            # Check convergence: stop if error increases for multiple iterations
            if iter == iterNum - 1:
                if verbose:
                    print("Reached the maximum number of iterations")
                break
            elif cp.sum(oldError <= currentError) > 1:
                if verbose:
                    print("Error increased for multiple iterations")
                break
            elif np.abs(oldError[-1] - currentError) < option["tol"]:
                if verbose:
                    print("Absolute difference between old and new error is less than 1e-3")
                break
            else:
                # Update error history (shift and add new error)
                oldError[:-1] = oldError[1:]  # Shift left
                oldError[-1] = currentError  # Add new error

            # Compute spatial gradients of the moving image at deformed coordinates
            Ix, Iy, Iz = getSpatialGradientInOrgGrid(
                data_mov_layer, coords_new
            )  # Each shape: (x, y, z) # dimensionality of the data

            # Zero out masked regions in gradients
            Ix[mask] = 0  # Shape: (x, y, z)
            Iy[mask] = 0  # Shape: (x, y, z)
            Iz[mask] = 0  # Shape: (x, y, z)

            # Scale z-gradient by z-ratio for anisotropic voxels
            Iz = Iz / zRatio  # Shape: (x, y, z)

            # Compute structure tensor components using spatial averaging
            AverageFilter = cp.ones(
                (r * 2 + 1, r * 2 + 1, 1)
            )  # Shape: (2*r+1, 2*r+1, 1)

            # Compute all components of the structure tensor and smooth
            Ixx = calculate.imfilter(
                Ix**2, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Ixy = calculate.imfilter(
                Ix * Iy, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Ixz = calculate.imfilter(
                Ix * Iz, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Iyy = calculate.imfilter(
                Iy**2, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Iyz = calculate.imfilter(
                Iy * Iz, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Izz = calculate.imfilter(
                Iz**2, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)

            # Compute temporal gradient components
            Ixt = calculate.imfilter(
                Ix * It, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Iyt = calculate.imfilter(
                Iy * It, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Izt = calculate.imfilter(
                Iz * It, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)

            # Extract values at control points only (sparse grid)
            Ixx = Ixx[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Ixy = Ixy[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Ixz = Ixz[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Iyy = Iyy[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Iyz = Iyz[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Izz = Izz[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Ixt = Ixt[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Iyt = Iyt[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Izt = Izt[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))

            # Compute motion update using Lucas-Kanade method with smoothness penalty
            motion_update_normalized = getFlow3_withPenalty6(
                Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt, smoothPenaltySum, neiSum
            )  # Shape: (len(xG), len(yG), len(zG), 3) # solve the linear system of equations

            # Limit motion update magnitude for stability
            motion_update_dist = cp.sqrt(
                cp.sum(motion_update_normalized**2, axis=3)
            )  # Shape: (len(xG), len(yG), len(zG))
            motion_update_dist = cp.maximum(
                motion_update_dist / movRange, 1.0
            )  # Shape: (len(xG), len(yG), len(zG))

            # print(f"motion_update_dist: {motion_update_dist}")
            # check if any motion_udpted this is not 1.0
            if verbose:
                if cp.sum(motion_update_dist != 1) > 0:
                    print(f"motion_update_dist is not 1: {cp.sum(motion_update_dist != 1)}")
                
            motion_update_normalized = (
                motion_update_normalized / motion_update_dist[..., cp.newaxis]
            )  # Shape: (len(xG), len(yG), len(zG), 3)

            # Final motion update (unnormalized)
            motion_update = (
                motion_update_normalized  # Shape: (len(xG), len(yG), len(zG), 3)
            )

            # Scale z-component by z-ratio
            motion_update[:, :, :, 2] = (
                motion_update[:, :, :, 2] / zRatio
            )  # Shape: (len(xG), len(yG), len(zG), 3)

            # Update motion at control points
            motion_current_CP = (
                motion_current[xG_grid, yG_grid, zG_grid, :] + motion_update
            )  # Shape: (len(xG), len(yG), len(zG), 3)

            # Interpolate motion update from control points to full grid
            coords_new = compute_new_grid(
                grid, r, motion_current_CP.shape # is integer at each control point # unclear why this is needed at every iteration - does CP number ever change?
            )  # Shape: (3, x, y, z) # 


            # Interpolate each motion component separately
            for dirNum in range(3):
                temp_phi = cp.asarray(
                    motion_current_CP[:, :, :, dirNum]
                )  # Shape: (len(xG), len(yG), len(zG))
                motion_current[:, :, :, dirNum] = interp.interp3Grid(
                    temp_phi, coords_new
                ).reshape(
                    x, y, z
                )  # Shape: (x, y, z)

            diff_motion = np.abs(motion_current - old_motion)
            max_diff_motion = np.max(diff_motion)
            error_log[f"layer_{layer}"]["max_diff_motion"].append(max_diff_motion)
            
            max_motion = np.max(np.abs(motion_current))
            if verbose:
                print(
                    f"Downsample layer: {layer}\tIter: {iter}\tMax motion: {max_motion:.2f}\tMax diff. old vs new motion: {max_diff_motion:.4f}"
                )

            if max_diff_motion < 1e-3:
                if verbose:
                    print("Max diff. old vs new motion is less than 1e-3")
                break


    # Final output processing
    # Generate final coordinate grid
    grid = cp.meshgrid(
        *[
            cp.arange(n, dtype=cp.float32) for n in data_mov_layer.shape
        ],  # Create coordinate arrays
        indexing="ij",  # Use matrix indexing
        sparse=False,  # Return full grid
    )  # Returns tuple of 3 arrays, each shape: (x, y, z)

    # Compute final corrected coordinates
    coords_new = interp.correctGrid(motion_current, grid)  # Shape: (x, y, z, 3)

    # Convert motion field to CPU if needed
    if hasattr(motion_current, "get"):
        motion_current = cp.asnumpy(motion_current)  # Convert GPU array to CPU
    else:
        motion_current = np.asarray(motion_current)  # Already CPU array

    return motion_current, currentError, coords_new, error_log


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


######################################################################################################################
## TO BE IMPROVED
def getMapping(data_mov, data_ref, option, verbose=False):
    option["mask_ref"] = cp.asarray(
        option["mask_ref"], dtype=cp.float32
    )  # Shape: (H, W, D)
    # moving中的某些点扔出定义域，不去做映射
    option["mask_mov"] = cp.asarray(
        option["mask_mov"], dtype=cp.float32
    )  # Shape: (H, W, D)
    layer_num = option["layer"]  # Number of pyramid layers
    iterNum = option["iter"]  # Number of iterations per layer
    r = option["r"]  # Filter radius
    zRatio_raw = option["zRatio"]  # Z-axis scaling ratio
    zRatio_HR = option["zRatio_HR"]  # Z-axis scaling ratio for high-res reference
    SZ = data_mov.shape  # Original image dimensions
    SZ_HR = data_ref.shape
    movRange=option['movRange']
    for layer in range(layer_num, -1, -1):
        if verbose:
            print(f"starting layer {layer} out of {layer_num}")
        x = int(SZ[0] / (2**layer))  # Downsampled width
        y = int(SZ[1] / (2**layer))  # Downsampled height
        z = SZ[2]  # Keep original depth
        x_hr = int(SZ_HR[0] / (2**layer))  # Downsampled width
        y_hr = int(SZ_HR[1] / (2**layer))  # Downsampled height
        z_hr = SZ_HR[2]  # Keep original depth
        data_mov_layer = imresize(
            cp.asarray(data_mov), output_shape=(x, y, z)
        )  # Shape: (x, y, z) # bicubic by default
        data_reference_layer = imresize(
            cp.asarray(data_ref), output_shape=(x_hr, y_hr, z_hr)
        )  # Shape: (x_hr, y_hr, z_hr)
        x, y, z = data_mov_layer.shape  # Updated dimensions for current level
        zRatio = zRatio_raw / (2**layer)
        zRatio_hr = zRatio_HR / (2**layer)
        H_layer = generate_continuous_H_gpu(
            data_reference_layer,
            zRatio=1
        )

        # Initialize motion field for current layer
        if layer == layer_num:  # If at the coarsest level
            if "motion" in option and option["motion"] is not None:
                # Use provided initial motion field
                motion_current = cp.zeros(
                    (x, y, z, 3), dtype=cp.float32
                )  # Shape: (x, y, z, 3)
                motion_init = cp.array(
                    option["motion"], dtype=cp.float32
                )  # Shape: (H, W, D, 3)
                # Downsample and scale the initial motion field
                motion_current[:, :, :, 0] = cp.asarray(
                    imresize(motion_init[:, :, :, 0], output_shape=(x, y, z))
                    / (SZ[0] / x)
                )  # Scale x-component
                motion_current[:, :, :, 1] = cp.asarray(
                    imresize(motion_init[:, :, :, 1], output_shape=(x, y, z))
                    / (SZ[1] / y)
                )  # Scale y-component
                motion_current[:, :, :, 2] = cp.asarray(
                    imresize(motion_init[:, :, :, 2], output_shape=(x, y, z))
                    / (SZ[2] / z)
                )  # Scale z-component
            else:
                # Start with zero motion field
                motion_current = cp.zeros(
                    (x, y, z, 3), dtype=cp.float32
                )  # Shape: (x, y, z, 3)

        else:
            # Upsample motion field from previous (finer) level
            motion_current_temp = cp.asarray(
                motion_current
            )  # Shape: (prev_x, prev_y, prev_z, 3)
            motion_current = cp.zeros(
                (x, y, z, 3), dtype=cp.float32
            )  # Shape: (x, y, z, 3)
            # Upsample and scale motion components (x2 for x,y due to pyramid structure)
            motion_current[:, :, :, 0] = cp.asarray(
                imresize(
                    motion_current_temp[:, :, :, 0],
                    output_shape=(x, y, z),
                    method="bilinear",
                )
                * 2
            )  # Scale x-component
            motion_current[:, :, :, 1] = cp.asarray(
                imresize(
                    motion_current_temp[:, :, :, 1],
                    output_shape=(x, y, z),
                    method="bilinear",
                )
                * 2
            )  # Scale y-component
            motion_current[:, :, :, 2] = cp.asarray(
                imresize(
                    motion_current_temp[:, :, :, 2],
                    output_shape=(x, y, z),
                    method="bilinear",
                )
            )  # Scale z-component
            motion_current = cp.asarray(
                motion_current, dtype=cp.float32
            )  # Shape: (x, y, z, 3)

        if "phase" in option and option["phase"] is not None:
            phase_init=cp.array(
                option["phase"], dtype=cp.float32
            )
            phase_current[:,:,:,0]=cp.asarray(
                imresize(phase_init[:,:,:,0], output_shape=(x, y, z))
                / (SZ[0] / x)
            )  # Scale x-component
            phase_current[:,:,:,1]=cp.asarray(
                imresize(phase_init[:,:,:,1], output_shape=(x, y, z))
                / (SZ[1] / y)
            )  # Scale y-component
            phase_current[:,:,:,2]=cp.asarray(
                imresize(phase_init[:,:,:,2], output_shape=(x, y, z))
                / (SZ[2] / z)
            )  # Scale z-component
        else:
            X, Y, Z = cp.indices((x, y, z))
            phase_current = cp.stack(
                [X, Y, Z*zRatio/zRatio_hr], axis=-1
            ).astype(cp.float32)
        oldError = cp.inf * cp.ones(3)  # Shape: (3,) - track last 3 errors
        smoothPenalty = option["smoothPenalty"]  # Smoothness penalty weight
        patchConnectNum = (r * 2 + 1) ** 2  # Number of connected patches
        smoothPenaltySum = smoothPenalty * patchConnectNum  # Total penalty weight
        xG = cp.arange(r, x - 1, step=2 * r + 1)  # Control points in x direction
        yG = cp.arange(r, y - 1, step=2 * r + 1)  # Control points in y direction
        zG = cp.arange(0, z)  # All z positions
        xG_grid, yG_grid, zG_grid = cp.meshgrid(
            xG, yG, zG, indexing="ij"
        )  # Shape: (len(xG), len(yG), len(zG))
        for iter in range(iterNum):
            old_motion = motion_current.copy()
            # Apply current motion field to get warped moving image
            phase_update = motion_current.copy()
            ####### attention
            phase_update[...,2] = phase_update[...,2]*zRatio_hr/zRatio
            phase_new = phase_current + phase_update
            data_mov_mapped= apply_H_to_matrix_gpu(phase_new,H_layer)
            It = data_mov_layer - data_mov_mapped  # Shape: (x, y, z) # temporal difference
            It = calculate.imfilter(
                It, cp.ones((3, 3, 1)) / 9, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            # Compute neighbor motion differences for smoothness constraint
            neiDiff = getNeiDiff(
                motion_current[xG_grid, yG_grid, zG_grid, :], 1
            )  # Shape: (len(xG), len(yG), len(zG), 3)
            # Scale z-component by z-ratio to account for anisotropic voxels
            # neiDiff[:, :, :, 2] = (
            #     neiDiff[:, :, :, 2] * zRatio
            # )  # Shape: (len(xG), len(yG), len(zG), 3)
            neiSum = smoothPenaltySum * neiDiff 
            diffError, penaltyError = calError(
                It, neiDiff, smoothPenaltySum
            )  # Both scalar values
            currentError = diffError + penaltyError  # Total error
            if verbose:
                print(
                    f"Downsample layer: {layer}\tIter: {iter}\tError: {currentError:.3f}, Diff Error: {diffError:.3f}, Penalty Error: {penaltyError:.3f}"
                )
            if iter == iterNum - 1:
                if verbose:
                    print("Reached the maximum number of iterations")
                break
            elif cp.sum(oldError <= currentError) > 1:
                if verbose:
                    print("Error increased for multiple iterations")
                break
            elif np.abs(oldError[-1] - currentError) < option["tol"]:
                if verbose:
                    print("Absolute difference between old and new error is less than 1e-3")
                break
            else:
                # Update error history (shift and add new error)
                oldError[:-1] = oldError[1:]  # Shift left
                oldError[-1] = currentError  # Add new error
            Ix, Iy, Iz = getSpatialGradientInOrgGrid(
                data_reference_layer, phase_new
            )  # Each shape: (x, y, z) # dimensionality of the data
            Iz /= zRatio_hr
            AverageFilter = cp.ones(
                (r * 2 + 1, r * 2 + 1, 1)
            )  # Shape: (2*r+1, 2*r+1, 1)
            Ixx = calculate.imfilter(
                Ix**2, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Ixy = calculate.imfilter(
                Ix * Iy, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Ixz = calculate.imfilter(
                Ix * Iz, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Iyy = calculate.imfilter(
                Iy**2, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Iyz = calculate.imfilter(
                Iy * Iz, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Izz = calculate.imfilter(
                Iz**2, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)


            #test the rank of the matrix
            # for i in range(Ixx.shape[0]):
            #     for j in range(Ixx.shape[1]):
            #         for k in range(Ixx.shape[2]):
            #             A=cp.array([[Ixx[i][j][k],Ixy[i][j][k],Ixz[i][j][k]],[Ixy[i][j][k],Iyy[i][j][k],Iyz[i][j][k]],[Ixz[i][j][k],Iyz[i][j][k],Izz[i][j][k]]])
            #             print(f"layer:{layer}, iter:{iter} control point[{i},{j},{k}] eigenvalues of structure tensor:{cp.linalg.eigvalsh(A)}")

            Ixt = calculate.imfilter(
                Ix * It, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Iyt = calculate.imfilter(
                Iy * It, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Izt = calculate.imfilter(
                Iz * It, AverageFilter, "replicate", "same", "corr"
            )  # Shape: (x, y, z)
            Ixx = Ixx[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Ixy = Ixy[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Ixz = Ixz[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Iyy = Iyy[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Iyz = Iyz[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Izz = Izz[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Ixt = Ixt[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Iyt = Iyt[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            Izt = Izt[xG_grid, yG_grid, zG_grid]  # Shape: (len(xG), len(yG), len(zG))
            motion_update_normalized = getFlow3_withPenalty6(
                Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt, smoothPenaltySum, neiSum
            )  # Shape: (len(xG), len(yG), len(zG), 3) # solve the linear system of equations
            motion_update_dist = cp.sqrt(
                cp.sum(motion_update_normalized**2, axis=3)
            )  # Shape: (len(xG), len(yG), len(zG))
            motion_update_dist = cp.maximum(
                motion_update_dist / movRange, 1.0
            )  # Shape: (len(xG), len(yG), len(zG))
            motion_update_normalized = (
                motion_update_normalized / motion_update_dist[..., cp.newaxis]
            )  # Shape: (len(xG), len(yG), len(zG), 3)
            motion_update = (
                motion_update_normalized  # Shape: (len(xG), len(yG), len(zG), 3)
            )
            # motion_update[:, :, :, 2] = (
            #     motion_update[:, :, :, 2]
            # )  # Shape: (len(xG), len(yG), len(zG), 3)
            motion_current_CP = (
                motion_current[xG_grid, yG_grid, zG_grid, :] + motion_update
            ) 
            ######
            grid = cp.meshgrid(
                *[
                    cp.arange(n, dtype=cp.float32) for n in data_mov_layer.shape
                ],
                indexing="ij",  
                sparse=False, 
            )  

            coords_new = compute_new_grid(
                grid, r, motion_current_CP.shape 
            ) 
            for dirNum in range(3):
                temp_phi = cp.asarray(
                    motion_current_CP[:, :, :, dirNum]
                )  # Shape: (len(xG), len(yG), len(zG))
                motion_current[:, :, :, dirNum] = interp.interp3Grid(
                    temp_phi, coords_new
                ).reshape(
                    x, y, z
                ) 
            diff_motion = np.abs(motion_current - old_motion)
            max_diff_motion = np.max(diff_motion)
            if max_diff_motion < 1e-3:
                break

    phase_update = motion_current.copy()
    ####### attention
    phase_update[...,2] = phase_update[...,2]*zRatio_hr/zRatio
    phase_new = phase_current + phase_update
    data_mov_mapped= apply_H_to_matrix_gpu(phase_new,H_layer)
    if hasattr(motion_current, "get"):
        motion_current = cp.asnumpy(motion_current) 
    else:
        motion_current = np.asarray(motion_current) 
    if hasattr(phase_new, "get"):
        phase_new = cp.asnumpy(phase_new) 
    else:
        phase_new = np.asarray(phase_new)  
    
    return phase_new,motion_current, data_mov_mapped


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
    H: 插值函数
    """
    coords = cp.asarray(A)
    shape = coords.shape
    coords_flat = coords.reshape(-1, 3)
    R = H(coords_flat)
    return R.reshape(shape[:-1])
