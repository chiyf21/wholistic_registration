"""

Author: Yunfeng Chi
Date: 2025/4/10

Overview:
    This script provides functions for visualizing 2D and 3D images, as well as overlaying motion fields on 2D images.
    
Functions:
    1. visualize_2d_image:
        Visualizes a single 2D image.
    
    2. visualize_3d_image:
        Visualizes a 3D image or volume using a specific slice along one axis (e.g., z-axis).
    
    3. overlay_motion_on_2d:
        Overlays a motion field on a 2D image and displays the result.
    
Usage:
    - Import this script and use the functions to visualize image data.
    
    Example:
        import visualization
        img_2d = np.random.rand(256, 256)
        motion_field = np.random.rand(256, 256, 2)  # Example motion field with u and v components
        visualization.visualize_2d_image(img_2d)
        visualization.overlay_motion_on_2d(img_2d, motion_field)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D

def visualize_2d_image(image, cmap='gray', title='2D Image',threshold=None):
    """
    Visualizes a single 2D image.
    
    Args:
        image (ndarray): The 2D image to display.
        cmap (str): The colormap to use for visualization.
        title (str): Title of the image plot.
    """
    if threshold is None:
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')  # Hide axes for a cleaner visualization
        plt.colorbar()
        plt.show()
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap=cmap,vmin=threshold[0], vmax=threshold[1])
        plt.title(title)
        plt.axis('off')  # Hide axes for a cleaner visualization
        plt.colorbar()
        plt.show()

def visualize_3d_image(image, slice_axis=2, slice_index=None, cmap='gray', title='3D Image Slice'):
    """
    Visualizes a 3D image by displaying a slice along a given axis (x, y, or z).
    
    Args:
        image (ndarray): The 3D image (volume) to display.
        slice_axis (int): The axis along which to slice the 3D image (0: x-axis, 1: y-axis, 2: z-axis).
        slice_index (int): The index of the slice to display along the specified axis.
        cmap (str): The colormap to use for visualization.
        title (str): Title of the slice plot.
    """
    if slice_index is None:
        slice_index = image.shape[slice_axis] // 2  # Default to middle slice
        
    if slice_axis == 0:
        image_slice = image[slice_index, :, :]
    elif slice_axis == 1:
        image_slice = image[:, slice_index, :]
    elif slice_axis == 2:
        image_slice = image[:, :, slice_index]
    else:
        raise ValueError("slice_axis must be 0 (x-axis), 1 (y-axis), or 2 (z-axis)")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image_slice, cmap=cmap)
    plt.title(f'{title} (Slice {slice_index})')
    plt.axis('off')
    plt.colorbar()
    plt.show()

def overlay_motion_on_2d(image, motion_field, quiver_scale=5, cmap='jet', title='Motion Overlay'):
    """
    Overlays a motion field on a 2D image and displays the result.
    
    Args:
        image (ndarray): The 2D image to display.
        motion_field (ndarray): A motion field with shape (height, width, 2), where the third dimension 
                                 contains the motion vectors in x (u) and y (v).
        quiver_scale (float): Scaling factor for the motion vectors.
        cmap (str): The colormap to use for visualization.
        title (str): Title of the image with motion overlay.
    """

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    
    u = motion_field[:, :, 0]  
    v = motion_field[:, :, 1]  
    
    Y, X = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    
    plt.quiver(X, Y, u, v, color='g',scale=quiver_scale)

    plt.colorbar(label='Motion Vector Magnitude')
    plt.show()

