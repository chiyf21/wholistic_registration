"""

version : 0.1
file name : calFlow3d_Wei.py

Alghothm Author : Wei Zheng (Vigirnia Tech) , Virginia M.S.(HHMI)
Code Author : Wei Zheng for matlab and Yunfeng Chi (Tsinghua University) for python
Date : 2025/4/9


Overview:
    This file contains functions for calculating 3D flow fields and image filtering using GPU acceleration.
    It includes functions for filtering images, computing determinants of matrices, and correcting indices based on motion data.
    The functions are designed to work with cupy arrays for GPU processing.

funtions:
    - imfilter: Perform filtering on a GPU array using either correlation or convolution with FFT.
    - getDet3: Compute the determinant of a 3x3 matrix formed by given components.
    - getDet2: Compute the determinant of a 2x2 matrix formed by given components.

    
"""
import numpy as np
from . import cp, cupy_ndimage
convolve = cupy_ndimage.convolve
correlate = cupy_ndimage.correlate

def imfilter(A, H, boundary='reflect', output='same', functionality='corr'):
    """
    Perform filtering on a GPU array using either correlation or convolution.
    Now implemented with cupyx.scipy.ndimage functions for better performance.

    Parameters:
    - A (cupy.ndarray): The input data (image).
    - H (cupy.ndarray): The filter (kernel).
    - boundary (str): How to handle boundaries ('reflect', 'constant', 'nearest', 'mirror', 'wrap').
    - output (str): 'same' or 'valid', determines the size of the output.
    - functionality (str): 'corr' for correlation or 'conv' for convolution.

    Returns:
    - cupy.ndarray: The filtered result.
    """
    # Convert boundary mode to match scipy's conventions
    mode_mapping = {
        'replicate': 'nearest',
        'reflect': 'reflect',
        'constant': 'constant'
    }
    boundary = mode_mapping.get(boundary, boundary)
    

    # Perform the operation
    if functionality == 'corr':
        result = correlate(A, H, mode=boundary)
    else:  # convolution
        result = convolve(A, H, mode=boundary)
    
    # Handle output size
    if output == 'same':
        return result
    elif output == 'valid':
        # Calculate valid region
        pad_y = H.shape[0] // 2
        pad_x = H.shape[1] // 2
        return result[pad_y:-pad_y, pad_x:-pad_x]
    else:
        raise ValueError("Invalid output mode. Use 'same' or 'valid'.")
    
def getDet3(Ixx, Ixy, Ixz, Iyy, Iyz, Izz):
    """
    Compute the determinant of the 3x3 matrix formed by Ixx, Ixy, Ixz, Iyy, Iyz, Izz.

    Parameters:
        Ixx, Ixy, Ixz, Iyy, Iyz, Izz (cupy.ndarray): The components of the 3x3 matrix.

    Returns:
        cupy.ndarray: The determinant of the 3x3 matrix.
    """
    return Ixx * (Iyy * Izz - Iyz ** 2) - Ixy * (Ixy * Izz - Iyz * Ixz) + Ixz * (Ixy * Iyz - Iyy * Ixz)

def getDet2(A, B, C, D):
    """
    Compute the determinant of the 2x2 matrix.

    Parameters:
        A, B, C, D (cupy.ndarray): The components of the 2x2 matrix.

    Returns:
        cupy.ndarray: The determinant of the 2x2 matrix.
    """
    return A * D - B * C