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
    return_fig=False
):
    """
    Visualize a deformed 2D plane as a 3D wireframe/surface using Plotly.

    Parameters
    ----------
    phase : ndarray, optional
        Shape (H, W, 3). Direct mapping coordinates:
            phase[..., 0] = mapped X
            phase[..., 1] = mapped Y
            phase[..., 2] = mapped Z

    motion : ndarray, optional
        Shape (H, W, 3). Displacement field:
            motion[..., 0] = dX
            motion[..., 1] = dY
            motion[..., 2] = dZ

    z0 : float, optional
        Initial plane z coordinate when using motion input.

    step : int
        Grid interval for plotting lines. Smaller -> denser grid.

    scale_z : float
        Additional visualization scaling for z axis only.
        If you want x/y/z to represent the same unit, keep scale_z=1.0.

    spacing : tuple of 3 floats
        Physical spacing for (x, y, z), e.g. (sx, sy, sz).
        Final plotted coordinates are:
            Xplot = Xmap * sx
            Yplot = Ymap * sy
            Zplot = Zmap * sz * scale_z

    xlim, ylim, zlim : tuple or None
        Axis display ranges, e.g. xlim=(0, 500)

    upsample : int
        Interpolation factor for denser/smoother visualization.
        upsample=1 means no upsampling.

    interp_order : int
        Interpolation order used by scipy.ndimage.zoom.
        Common values:
            1 = linear
            3 = cubic

    smooth_sigma : float
        Gaussian smoothing sigma applied to X/Y/Z maps before plotting.
        0 means no smoothing.

    show_surface : bool
        Whether to overlay a semi-transparent surface.

    surface_opacity : float
        Opacity of the surface if show_surface=True.

    line_width : float
        Width of wireframe lines.

    title : str
        Figure title.

    return_fig : bool
        If True, return the Plotly figure object.

    Returns
    -------
    fig : plotly.graph_objects.Figure, optional
        Returned only if return_fig=True.
    """

    import plotly.graph_objects as go
    from scipy.ndimage import zoom, gaussian_filter

    if phase is None:
        if motion is None or z0 is None:
            raise ValueError("Provide either phase, or motion together with z0.")

        motion = np.asarray(motion)
        if motion.ndim != 3 or motion.shape[2] < 3:
            raise ValueError("motion must have shape (H, W, 3).")

        H, W, _ = motion.shape

        # FIX: translated Chinese comment — use indexing='ij' to align x/y with array row/col semantics
        xx, yy = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        Xmap = xx + motion[:, :, 0]
        Ymap = yy + motion[:, :, 1]
        Zmap = z0 + motion[:, :, 2]

    else:
        phase = np.asarray(phase)
        if phase.ndim != 3 or phase.shape[2] < 3:
            raise ValueError("phase must have shape (H, W, 3).")

        Xmap = phase[:, :, 0]
        Ymap = phase[:, :, 1]
        Zmap = phase[:, :, 2]
        H, W = Xmap.shape

    # -----------------------------
    # 2) Optional smoothing
    # -----------------------------
    if smooth_sigma is not None and smooth_sigma > 0:
        Xmap = gaussian_filter(Xmap, sigma=smooth_sigma)
        Ymap = gaussian_filter(Ymap, sigma=smooth_sigma)
        Zmap = gaussian_filter(Zmap, sigma=smooth_sigma)

    # -----------------------------
    # 3) Optional upsampling
    # -----------------------------
    if upsample is not None and upsample > 1:
        Xmap = zoom(Xmap, upsample, order=interp_order)
        Ymap = zoom(Ymap, upsample, order=interp_order)
        Zmap = zoom(Zmap, upsample, order=interp_order)

    # Updated size after upsampling
    H2, W2 = Xmap.shape

    # -----------------------------
    # 4) Apply physical spacing
    # -----------------------------
    sx, sy, sz = spacing

    Xplot = Xmap * sx
    Yplot = Ymap * sy
    Zplot = Zmap * sz * scale_z

    # -----------------------------
    # 5) Create figure
    # -----------------------------
    fig = go.Figure()

    # Optional surface
    if show_surface:
        fig.add_trace(go.Surface(
            x=Xplot,
            y=Yplot,
            z=Zplot,
            opacity=surface_opacity,
            showscale=False
        ))

    # Horizontal lines
    for i in range(0, H2, step):
        fig.add_trace(go.Scatter3d(
            x=Xplot[i, :],
            y=Yplot[i, :],
            z=Zplot[i, :],
            mode='lines',
            line=dict(width=line_width),
            showlegend=False
        ))

    # Vertical lines
    for j in range(0, W2, step):
        fig.add_trace(go.Scatter3d(
            x=Xplot[:, j],
            y=Yplot[:, j],
            z=Zplot[:, j],
            mode='lines',
            line=dict(width=line_width),
            showlegend=False
        ))

    # -----------------------------
    # 6) Axis settings
    # -----------------------------
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
