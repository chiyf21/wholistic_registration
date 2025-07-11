"""

version : 0.1
file name: interp.py

Alghothm Author : Wei Zheng (Vigirnia Tech) , Virginia M.S.(HHMI)
Code Author : Wei Zheng for matlab and Yunfeng Chi (Tsinghua University) for python
Last Update Date : 2025/4/19


Overview:
    This file contains functions for 3D interpolation  of volumetric images and some supported functions.
    It includes GPU-accelerated trilinear interpolation, resizing functions, and index correction based on motion data.

    
functions:
    - interp3Grid: Performs trilinear interpolation on 3D data using GPU.
    - ind2sub: Converts linear indices to subscripts for a multidimensional array.
    - correctIdx: Corrects indices based on motion data.
    - correctGrid: Corrects coordinates based on motion data.
    - rangeConstrain: Ensures a value is within a specified range.
    
"""

import numpy as np
from scipy.ndimage import zoom
import warnings
from . import cp, Gimage, RegularGridInterpolator


def interp3Grid(image, coords_new, method="linear", mode="nearest", extrapval=0):
    if method == "nearest":
        order = 0
    elif method == "linear":
        order = 1
    elif method == "cubic":
        order = 3
    elif method == "lanczos2":
        order = 4
    elif method == "lanczos3":
        order = 5
    elif method == "box":
        order = 0
    else:
        raise ValueError(f"Unsupported interpolation method: {method}")
    image_warped = Gimage.map_coordinates(
        image, coords_new, order=order, mode=mode, cval=extrapval
    )

    return image_warped


def ind2sub(siz, ndx):
    """
    Convert a linear index to multiple subscripts for a multidimensional array.

    Parameters:
        siz (list or tuple): The size of each dimension of the array.
        ndx (int or array-like): The linear index (or indices) to convert.

    Returns:
        A tuple containing the subscripts for each dimension.

    2025/4/16 we don't use this function on version 1.0 now
    """
    siz = cp.array(siz, dtype=int)
    lensiz = len(siz)

    if lensiz < 2:
        raise ValueError("Size must have at least 2 dimensions.")

    if not cp.issubdtype(type(ndx), cp.integer) and not cp.issubdtype(
        type(ndx), cp.floating
    ):
        raise ValueError("Index should be an integer or floating-point number.")

    # Flatten ndx if it's an array-like object
    ndx = cp.atleast_1d(ndx)

    # Prepare for the result subscripts
    subscripts = []

    # Adjust size for the case where there are more outputs than dimensions
    if len(subscripts) > 2:
        if lensiz + 1 < len(subscripts):
            siz = cp.concatenate([siz, cp.ones(len(subscripts) - lensiz)])

    # Initialize the index
    k = cp.cumprod(siz)[::-1]

    for i in range(len(subscripts), 2, -1):
        vi, ndx = cp.divmod(ndx - 1, k[i - 2])  # equivalent to rem and floor
        subscripts.insert(0, vi + 1)  # Convert to 1-based indexing

    # Calculate the final 2D indices
    vi, ndx = cp.divmod(ndx - 1, siz[0])
    subscripts.insert(0, vi + 1)  # Convert to 1-based indexing
    subscripts.insert(1, ndx + 1)

    return tuple(subscripts)


def correctGrid(phi_current, grid):
    out = phi_current.copy()
    for idx, g in enumerate(grid):
        out[..., idx] += g
    return out


def correctIdx(data_raw, phi_current, x_ind, y_ind, z_ind):
    """
    Correct the indices based on the motion data.

    Parameters:
        data_raw (ndarray): The raw data array with shape (x, y, z).
        phi_current (ndarray): The motion field with shape (x, y, z, 3).
        x_ind (ndarray): The x-indices of the data.
        y_ind (ndarray): The y-indices of the data.
        z_ind (ndarray): The z-indices of the data.

    Returns:
        x_new (ndarray): Corrected x-indices.
        y_new (ndarray): Corrected y-indices.
        z_new (ndarray): Corrected z-indices.

    2025/4/16 we don't use this function on version 1.0 now
    """
    # Get the size of the data
    x, y, z = data_raw.shape

    # Reshape phi_current to get the motion vectors for each index
    x_bias = phi_current[:, :, :, 0].reshape(-1)
    y_bias = phi_current[:, :, :, 1].reshape(-1)
    z_bias = phi_current[:, :, :, 2].reshape(-1)

    # Apply the bias and make sure indices are within bounds
    x_new = x_ind + x_bias
    y_new = y_ind + y_bias
    z_new = z_ind + z_bias

    # Ensure the indices are within the bounds of the data dimensions
    x_new = cp.clip(x_new, 0, x - 1)  # Clipping to ensure indices are within [0, x-1]
    y_new = cp.clip(y_new, 0, y - 1)  # Clipping to ensure indices are within [0, y-1]
    z_new = cp.clip(z_new, 0, z - 1)  # Clipping to ensure indices are within [0, z-1]

    x_new = cp.asarray(x_new, dtype=cp.float32)
    y_new = cp.asarray(y_new, dtype=cp.float32)
    z_new = cp.asarray(z_new, dtype=cp.float32)

    return x_new, y_new, z_new


def rangeConstrain(value, min_val, max_val):
    """
    Ensure the value is within the range [min_val, max_val].

    Parameters:
        value (cupy.ndarray): The input value to be constrained.
        min_val (int): The minimum allowable value.
        max_val (int): The maximum allowable value.

    Returns:
        cupy.ndarray: The constrained value.
    """
    return cp.clip(value, min_val, max_val)
