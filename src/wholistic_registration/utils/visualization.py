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

# FIX: removed duplicate numpy/matplotlib imports and unused Axes3D import.
# Made plotly and ipywidgets lazy imports so the module doesn't fail if they
# aren't installed (they're only needed by specific functions).
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def visualize_3d_image(image, slice_axis=2, cmap='gray', title='3D Image Slice',autocontrast = False):
    """
    Interactive viewer for a 3D image stack using ipywidgets.

    Args:
        image (ndarray): 3D image, shape (D0, D1, D2)
        slice_axis (int): axis to slice along (0, 1, or 2)
        cmap (str): matplotlib colormap
        title (str): plot title prefix
    """
    from ipywidgets import interact, IntSlider

    if image.ndim != 3:
        raise ValueError("image must be a 3D ndarray")
    if slice_axis not in [0, 1, 2]:
        raise ValueError("slice_axis must be 0, 1, or 2")

    max_index = image.shape[slice_axis] - 1

    def _show_slice(slice_index):
        if slice_axis == 0:
            image_slice = image[slice_index, :, :]
        elif slice_axis == 1:
            image_slice = image[:, slice_index, :]
        else:
            image_slice = image[:, :, slice_index]

        plt.figure(figsize=(6, 6))
        if autocontrast:
            plt.imshow(auto_contrast(image_slice), cmap=cmap)
        else:
            plt.imshow(image_slice, cmap=cmap)
        plt.title(f"{title} (axis={slice_axis}, slice={slice_index})")
        plt.axis("off")
        plt.colorbar()
        plt.show()

    interact(
        _show_slice,
        slice_index=IntSlider(
            min=0,
            max=max_index,
            step=1,
            value=max_index // 2,
            description='Slice'
        )
    )

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
    # plt.quiver(x_sub, y_sub, u_sub, -v_sub, color='g', angles='xy', scale_units='xy', alpha=1,scale=0.7, linewidth=2.0)
    plt.quiver(x_sub, y_sub, u_sub, -v_sub, color='g', angles='xy', scale_units='xy', alpha=1,scale=1.2, linewidth=2.0)
    plt.title("Motion Field Overlay on Image")
    plt.axis('off')

    # save the figure
    if save_path and file_name:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, file_name)
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to: {save_file}")

    plt.show()
    
def plot_deformed_grid_plotly(
    phase=None,
    motion=None,
    z0=None,
    step=20,
    scale_z=1.0,
    spacing=(1.0, 1.0, 1.0),
    xlim=None,
    ylim=None,
    zlim=None,
    upsample=1,
    interp_order=3,
    smooth_sigma=0.0,
    show_surface=False,
    surface_opacity=0.35,
    line_width=3,
    title="Deformed grid in 3D",
    return_fig=False,
    slice_indices=None,
    slice_stride=1,
):
    """
    Visualize deformed grid(s) as 3D wireframe / surface using Plotly.

    Supports both:
    1) Single 2D plane:
       - phase  shape (H, W, 3)
       - motion shape (H, W, 3)

    2) 3D stack of planes:
       - phase  shape (H, W, K, 3)
       - motion shape (H, W, K, 3)

    Parameters
    ----------
    phase : ndarray, optional
        If single plane:
            shape (H, W, 3)
        If stack:
            shape (H, W, K, 3)

        Mapping coordinates:
            phase[..., 0] = mapped X
            phase[..., 1] = mapped Y
            phase[..., 2] = mapped Z

    motion : ndarray, optional
        If single plane:
            shape (H, W, 3)
        If stack:
            shape (H, W, K, 3)

        Displacement field:
            motion[..., 0] = dX
            motion[..., 1] = dY
            motion[..., 2] = dZ

    z0 : float, sequence, optional
        Used only when motion is provided.

        - If motion is single-plane (H, W, 3):
            z0 should be a scalar.

        - If motion is stack (H, W, K, 3):
            z0 can be:
              * scalar: base z for slice 0, then slice k uses z0 + k
              * array-like of length K: per-slice initial z

    step : int
        Grid interval for plotting lines.

    scale_z : float
        Additional visualization scaling for z axis only.

    spacing : tuple of 3 floats
        Physical spacing for (x, y, z), e.g. (sx, sy, sz).

    xlim, ylim, zlim : tuple or None
        Axis display ranges.

    upsample : int
        Interpolation factor for denser/smoother visualization.

    interp_order : int
        Interpolation order used by scipy.ndimage.zoom.
        1 = linear, 3 = cubic.

    smooth_sigma : float
        Gaussian smoothing sigma applied to X/Y/Z maps before plotting.
        0 means no smoothing.

    show_surface : bool
        Whether to overlay a semi-transparent surface for each slice.

    surface_opacity : float
        Surface opacity if show_surface=True.

    line_width : float
        Width of wireframe lines.

    title : str
        Figure title.

    return_fig : bool
        If True, return the Plotly figure object.

    slice_indices : list[int] or None
        Which slices to show when input is a stack.
        If None, uses all slices with the given slice_stride.

    slice_stride : int
        Used only when slice_indices is None and input is a stack.
        Example: slice_stride=2 plots every other slice.

    Returns
    -------
    fig : plotly.graph_objects.Figure, optional
        Returned only if return_fig=True.
    """
    import numpy as np
    import plotly.graph_objects as go
    from scipy.ndimage import zoom, gaussian_filter

    # ------------------------------------------------------------
    # Helper: convert single plane to stack with K=1
    # ------------------------------------------------------------
    def _ensure_stack_lastdim3(arr, name):
        arr = np.asarray(arr)

        if arr.ndim == 3 and arr.shape[-1] >= 3:
            # (H, W, 3) -> (H, W, 1, 3)
            arr = arr[:, :, None, :]

        elif arr.ndim == 4 and arr.shape[-1] >= 3:
            # (H, W, K, 3) -> unchanged
            pass

        else:
            raise ValueError(
                f"{name} must have shape (H, W, 3) or (H, W, K, 3), got {arr.shape}"
            )

        return arr

    # ------------------------------------------------------------
    # Build Xmap / Ymap / Zmap stack: shape (H, W, K)
    # ------------------------------------------------------------
    if phase is None:
        if motion is None:
            raise ValueError("Provide either phase or motion.")
        motion = _ensure_stack_lastdim3(motion, "motion")

        H, W, K, C = motion.shape
        if C < 3:
            raise ValueError("motion last dimension must be at least 3.")

        # Build z0 list
        if z0 is None:
            raise ValueError("When using motion input, z0 must be provided.")

        if np.isscalar(z0):
            z0_arr = np.asarray([float(z0) + k for k in range(K)], dtype=np.float32)
        else:
            z0_arr = np.asarray(z0, dtype=np.float32).reshape(-1)
            if z0_arr.size != K:
                raise ValueError(
                    f"For stack motion, z0 must be scalar or length-K array. "
                    f"Expected K={K}, got {z0_arr.size}"
                )

        xx, yy = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        Xmap = np.empty((H, W, K), dtype=np.float32)
        Ymap = np.empty((H, W, K), dtype=np.float32)
        Zmap = np.empty((H, W, K), dtype=np.float32)

        for k in range(K):
            Xmap[:, :, k] = xx + motion[:, :, k, 0]
            Ymap[:, :, k] = yy + motion[:, :, k, 1]
            Zmap[:, :, k] = z0_arr[k] + motion[:, :, k, 2]

    else:
        phase = _ensure_stack_lastdim3(phase, "phase")

        H, W, K, C = phase.shape
        if C < 3:
            raise ValueError("phase last dimension must be at least 3.")

        Xmap = phase[:, :, :, 0].astype(np.float32, copy=False)
        Ymap = phase[:, :, :, 1].astype(np.float32, copy=False)
        Zmap = phase[:, :, :, 2].astype(np.float32, copy=False)

    # ------------------------------------------------------------
    # Decide which slices to display
    # ------------------------------------------------------------
    if slice_indices is None:
        slice_indices = list(range(0, K, max(int(slice_stride), 1)))
    else:
        slice_indices = list(slice_indices)

    if len(slice_indices) == 0:
        raise ValueError("No slice selected for plotting.")

    for k in slice_indices:
        if k < 0 or k >= K:
            raise ValueError(f"slice index {k} out of range for K={K}")

    # ------------------------------------------------------------
    # Physical spacing
    # ------------------------------------------------------------
    sx, sy, sz = spacing

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------
    fig = go.Figure()

    # ------------------------------------------------------------
    # Plot each selected slice
    # ------------------------------------------------------------
    for k in slice_indices:
        Xk = Xmap[:, :, k]
        Yk = Ymap[:, :, k]
        Zk = Zmap[:, :, k]

        # Optional smoothing
        if smooth_sigma is not None and smooth_sigma > 0:
            Xk = gaussian_filter(Xk, sigma=smooth_sigma)
            Yk = gaussian_filter(Yk, sigma=smooth_sigma)
            Zk = gaussian_filter(Zk, sigma=smooth_sigma)

        # Optional upsampling
        if upsample is not None and upsample > 1:
            Xk = zoom(Xk, upsample, order=interp_order)
            Yk = zoom(Yk, upsample, order=interp_order)
            Zk = zoom(Zk, upsample, order=interp_order)

        H2, W2 = Xk.shape

        # Apply physical spacing
        Xplot = Xk * sx
        Yplot = Yk * sy
        Zplot = Zk * sz * scale_z

        # Optional surface
        if show_surface:
            fig.add_trace(go.Surface(
                x=Xplot,
                y=Yplot,
                z=Zplot,
                opacity=surface_opacity,
                showscale=False,
                showlegend=False
            ))

        # Horizontal grid lines
        for i in range(0, H2, step):
            fig.add_trace(go.Scatter3d(
                x=Xplot[i, :],
                y=Yplot[i, :],
                z=Zplot[i, :],
                mode='lines',
                line=dict(width=line_width),
                name=f"slice {k}" if i == 0 else None,
                showlegend=(i == 0)
            ))

        # Vertical grid lines
        for j in range(0, W2, step):
            fig.add_trace(go.Scatter3d(
                x=Xplot[:, j],
                y=Yplot[:, j],
                z=Zplot[:, j],
                mode='lines',
                line=dict(width=line_width),
                showlegend=False
            ))

    # ------------------------------------------------------------
    # Axis settings
    # ------------------------------------------------------------
    xaxis_dict = dict(title='Ref X')
    yaxis_dict = dict(title='Ref Y')
    zaxis_dict = dict(title='Ref Z')

    if xlim is not None:
        xaxis_dict["range"] = list(xlim)
    if ylim is not None:
        yaxis_dict["range"] = list(ylim)
    if zlim is not None:
        zaxis_dict["range"] = list(zlim)

    fig.update_layout(
        scene=dict(
            xaxis=xaxis_dict,
            yaxis=yaxis_dict,
            zaxis=zaxis_dict,
            aspectmode='data'
        ),
        title=title
    )

    fig.show()

    if return_fig:
        return fig

def plot_sequence(sequence, title='Sequence Plot', xlabel='Index', ylabel='Value',
                  marker=None, figsize=(8, 4)):
    """
    Plot a 1D sequence as a line chart.

    Args:
        sequence: 1D list / tuple / numpy array
        title (str): figure title
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        marker (str or None): marker style, e.g. 'o', '.', None
        figsize (tuple): figure size
    """
    seq = np.asarray(sequence)

    if seq.ndim != 1:
        raise ValueError(f"sequence must be 1D, but got shape={seq.shape}")

    x = np.arange(len(seq))

    plt.figure(figsize=figsize)
    plt.plot(x, seq, marker=marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
