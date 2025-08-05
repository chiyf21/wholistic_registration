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
import os

def auto_contrast(img, low_percentile=1, high_percentile=99):
    img = img.astype(np.float32)
    p_low, p_high = np.percentile(img, [low_percentile, high_percentile])
    img_clipped = np.clip(img, p_low, p_high)
    return (img_clipped - p_low) / (p_high - p_low + 1e-8)


def visualize_2d_image(image, cmap='gray', title='2D Image',threshold=None,autocontrast=True,figsize=(8,8)):
    """
    Visualizes a single 2D image.
    
    Args:
        image (ndarray): The 2D image to display.
        cmap (str): The colormap to use for visualization.
        title (str): Title of the image plot.
    """
    if threshold is None:
        plt.figure(figsize=figsize)
        if autocontrast:
            plt.imshow(auto_contrast(image), cmap=cmap)
        else:
            plt.imshow(image, cmap=cmap)

        plt.title(title)
        plt.axis('off')  # Hide axes for a cleaner visualization
        plt.colorbar()
        plt.show()
    else:
        plt.figure(figsize=figsize)
        if autocontrast:
            plt.imshow(auto_contrast(image), cmap=cmap)
        else:
            plt.imshow(image, cmap=cmap)
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

def quivermotion_py(template, r, motion_field, save_path=None, file_name=None):
    """
    Display and optionally save an image with an overlay of the motion field (similar to MATLAB's quivermotion_Chi).
    
    Parameters:
        template (ndarray): Original image(H, W) or (H, W, C)
        r (int): Subsampling step size
        motion_field (ndarray): the flow or displacement field with shape (H, W, 2)
        save_path (str): optional, directory to save the image
        file_name (str): optional, name of the file to save the image (.png extension)
    """
    H, W = template.shape[:2]
    
    # sample coordinates for quiver
    x_indices = np.arange(r, W, 2*r + 1)
    y_indices = np.arange(r, H, 2*r + 1)
    x_sub, y_sub = np.meshgrid(x_indices, y_indices)

    # extract u, v components (note the order [v, u])
    u = motion_field[..., 0]
    v = motion_field[..., 1]
    u_sub = u[y_indices[:, None], x_indices]
    v_sub = v[y_indices[:, None], x_indices]

    # display the image with motion field overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(template, cmap='gray', origin='upper')
    plt.quiver(x_sub, y_sub, u_sub, -v_sub, color='g', angles='xy', scale_units='xy', alpha=1,scale=0.7, linewidth=2.0)
    plt.title("Motion Field Overlay on Image")
    plt.axis('off')

    # save the figure
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, file_name)
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_file}")

    plt.show()