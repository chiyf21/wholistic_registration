# reliablemask.py
from . import cp
from . import cupy_ndimage
from . import IO
import numpy as np
import tifffile
import os
from skimage.metrics import structural_similarity  as ssim
from scipy.ndimage import gaussian_filter
import re


def write_multichannel_volume_as_ome_tiff(vol3d_list, out_dir, frame_idx, configPath,
                                          spacing_x=1.0, spacing_y=1.0):
    """
    vol3d_list: list of 3 arrays, each (Z,Y,X)
        ch0, ch1, ch2

    output OME TIFF shape: (1, Z, 3, Y, X)
    """

    assert len(vol3d_list) == 3, "vol3d_list must contain exactly 3 volumes."

    processed = []
    for v in vol3d_list:
        if v.ndim == 2:
            v = v[np.newaxis, :, :]
        if v.dtype == bool:
            v = v.astype(np.uint8)
        elif v.dtype not in [np.uint8, np.float32]:
            v = v.astype(np.float32)
        processed.append(v)

    Z, Y, X = processed[0].shape
    img5d = np.stack(processed, axis=0)     # (3, Z, Y, X)
    img5d = img5d[np.newaxis, :, :, :, :]   # (1,3,Z,Y,X)
    img5d = np.transpose(img5d, (0, 2, 1, 3, 4))  # → (1,Z,3,Y,X)

    fname = os.path.join(out_dir, f"vol_{frame_idx:06d}_masked.tif")

    metadata = {
        'spacing_x': spacing_x,
        'spacing_y': spacing_y,
        'data_shape': img5d.shape
    }

    IO.saveTiff_new(
        img5d,
        fname,
        config_path=configPath,
        metadata=metadata,
        verbose=False
    )
def gradient_amplitude(volume: cp.ndarray) -> cp.ndarray:
    """
    Compute gradient amplitude of a 2D/3D image.
    volume: shape = (X,Y) or (X,Y,Z)
    """
    if volume.ndim == 2:  # 2D
        gx = cupy_ndimage.sobel(volume, axis=0)
        gy = cupy_ndimage.sobel(volume, axis=1)
        return cp.sqrt(gx**2 + gy**2)
    elif volume.ndim == 3:  # 3D
        gx = cupy_ndimage.sobel(volume, axis=0)
        gy = cupy_ndimage.sobel(volume, axis=1)
        gz = cupy_ndimage.sobel(volume, axis=2)
        return cp.sqrt(gx**2 + gy**2 + gz**2)
    else:
        raise ValueError("Only 2D or 3D data are supported")

def local_ssim_difference(I_ref,I_mov,win_size=11,use_3d=False,sigma_3d=1.5):
    """
    Compute local SSIM difference map (0~1) between two images.
    Supports 2D and 3D images. 
    
    Parameters
    ----------
    I_ref : np.ndarray
        Reference image (2D or 3D).
    I_mov : np.ndarray
        Registered/moving image (must match shape of I_ref).
    win_size : int
        SSIM window size for 2D (odd number like 11, 21).
    use_3d : bool
        If True, compute a true 3D SSIM approximation using Gaussian smoothing.
        If False, compute 2D SSIM slice-by-slice for 3D volumes (recommended for microscopy).
    sigma_3d : float or tuple
        Gaussian sigma used in 3D SSIM approximation mode.

    Returns
    -------
    D : np.ndarray, float32
        Difference map in [0,1], same shape as input.
    """
    I_ref=I_ref.astype(np.float32)
    I_mov=I_mov.astype(np.float32)
    
    if I_ref.ndim not in [2, 3]:
        raise ValueError("Input images must be 2D or 3D numpy arrays.")
    
    if I_ref.shape != I_mov.shape:
        raise ValueError("Input images must have the same shape.")
    # -----------------------
    # Case 1: 2D Image
    # -----------------------
    if I_ref.ndim == 2:
        _, ssim_map = ssim(
            I_ref, I_mov,
            win_size=win_size,
            gaussian_weights=True,
            data_range=I_ref.max() - I_ref.min(),
            full=True
        )
        D = (1 - ssim_map) / 2.0
        return np.clip(D.astype(np.float32), 0, 1)

    # -----------------------
    # Case 2: 3D Image
    # -----------------------
    if not use_3d:
        # --- Slice-wise 2D SSIM (recommended for microscopy with low Z-resolution)
        Z = I_ref.shape[0]
        D = np.zeros_like(I_ref, dtype=np.float32)

        for z in range(Z):
            _, ssim_map = ssim(
                I_ref[z], I_mov[z],
                win_size=win_size,
                gaussian_weights=True,
                data_range=I_ref[z].max() - I_ref[z].min(),
                full=True
            )
            D[z] = (1 - ssim_map) / 2.0
        
        return np.clip(D, 0, 1)

    else:
        # --- True 3D SSIM approximation using Gaussian filters
        #     (Useful only if Z-resolution is comparable to XY)
        
        C1 = (0.01 * (I_ref.max() - I_ref.min())) ** 2
        C2 = (0.03 * (I_ref.max() - I_ref.min())) ** 2

        mu_x = gaussian_filter(I_ref, sigma=sigma_3d)
        mu_y = gaussian_filter(I_mov, sigma=sigma_3d)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = gaussian_filter(I_ref * I_ref, sigma=sigma_3d) - mu_x2
        sigma_y2 = gaussian_filter(I_mov * I_mov, sigma=sigma_3d) - mu_y2
        sigma_xy = gaussian_filter(I_ref * I_mov, sigma=sigma_3d) - mu_xy

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

        ssim_map = numerator / (denominator + 1e-12)
        D = (1 - ssim_map) / 2

        return np.clip(D.astype(np.float32), 0, 1)

def local_mind_difference(
    I_ref,
    I_mov,
    patch_sigma=3,
    offset_radius=5,
    structure_thresh=0.6,
    eps=1e-6,
):
    """
    GPU MIND-based local misalignment map with explicit masking of background.

    Output semantics:
        - 0 → background or perfectly aligned
        - larger → worse local registration (misalignment)
        - completely ignores areas with no structure

    Supports 2D or 3D (slice-wise) images.
    """

    # ---------------------------
    # helpers
    # ---------------------------
    def _mind_offsets_2d(radius):
        return [
            ( radius, 0),
            (-radius, 0),
            (0,  radius),
            (0, -radius),
            ( radius,  radius),
            ( radius, -radius),
            (-radius,  radius),
            (-radius, -radius),
        ]

    def _mind_descriptor_2d(I):
        I = cp.asarray(I, dtype=cp.float32)
        offsets = _mind_offsets_2d(offset_radius)
        H, W = I.shape
        K = len(offsets)

        # smooth image to suppress noise
        I_s = cupy_ndimage.gaussian_filter(I, patch_sigma)
        D = cp.empty((K, H, W), dtype=cp.float32)

        for k, (dy, dx) in enumerate(offsets):
            I_shift = cp.roll(I_s, shift=(dy, dx), axis=(0, 1))
            diff2 = (I_s - I_shift) ** 2
            D[k] = cupy_ndimage.gaussian_filter(diff2, patch_sigma)

        V = cp.mean(D, axis=0) + eps
        MIND = cp.exp(-D / V[None])
        return MIND

    def _mind_diff_2d(Ir, Im):
        M_ref = _mind_descriptor_2d(Ir)
        M_mov = _mind_descriptor_2d(Im)

        diff = cp.mean(cp.abs(M_ref - M_mov), axis=0)

        # --------- structure mask ---------
        structure = cp.mean(M_ref, axis=0)
        mask = structure > structure_thresh

        # scale using only structured regions
        # apply hard mask: background = 0
        diff = diff * mask.astype(cp.float32)

        return cp.clip(diff**2, 0, 1)

    # ---------------------------
    # main
    # ---------------------------
    I_ref = cp.asarray(I_ref)
    I_mov = cp.asarray(I_mov)

    if I_ref.shape != I_mov.shape:
        raise ValueError("[ERROR] I_ref and I_mov must have the same shape")

    if I_ref.ndim == 2:
        return _mind_diff_2d(I_ref, I_mov)

    elif I_ref.ndim == 3:
        Z, H, W = I_ref.shape
        diff = cp.zeros((Z, H, W), dtype=cp.float32)
        for z in range(Z):
            diff[z] = _mind_diff_2d(I_ref[z], I_mov[z])
        return diff

    else:
        raise ValueError("Only 2D or 3D images are supported")


def build_reference_index(ref_dir):
    """
    scan reference folder, construct frame -> filepath
    return:
        ref_map: dict(frame -> filepath)
        ref_files: list of all files
    """
    ref_files = [f for f in os.listdir(ref_dir) if f.endswith(".tif")]
    ref_map = {}

    for f in ref_files:
        m_multi = re.match(r"vol_ref_(\d{6})_(\d{6})\.tif", f)
        m_single = re.match(r"vol_ref_(\d{6})\.tif", f)

        if m_multi:
            a = int(m_multi.group(1))
            b = int(m_multi.group(2))
            for t in range(a, b + 1):
                ref_map[t] = os.path.join(ref_dir, f)

        elif m_single:
            a = int(m_single.group(1))
            ref_map[a] = os.path.join(ref_dir, f)

        else:
            print(f"[WARNING] Unknown reference filename format: {f}")

    return ref_map, ref_files

def ComputMask(
                mem_dir,
                ca_dir,
                ref_dir,
                out_dir,
                dual_channel,
                frames,
                config,
                compute_cor_fn,
                configPath,
                T,
):
    """
    Computes spatial, temporal, and accumulative reliability masks.

    For each frame:
        1. Reads registered membrane and calcium images
        2. Computes correlation map
        3. Compares with reference image using SSIM
        4. Saves resulting mask as OME-TIFF

    Parameters:
    -----------
    mem_dir : str
        Directory containing registered membrane channel images
    ca_dir : str
        Directory containing registered calcium channel images
    ref_dir : str
        Directory containing reference images
    out_dir : str
        Directory where output masks will be saved
    config : dict
        Configuration parameters for reliable analysis
    compute_cor_fn : callable
        Function to compute correlation map from membrane and calcium channels
    configPath : str
        Path to the main configuration file
    T : int
        Total number of frames to process
    downsampleXY : int, optional
        Downsampling factor for XY dimensions (default: 1)
    downsampleT : int, optional
        Downsampling factor for temporal dimension (default: 1)

    Returns:
    --------
    None

    Output:
    -------
    Creates the following directory structure under out_dir:
    """

    # Create output directories
    mask_ds_dir = out_dir # Downsampled masks directory
    
    os.makedirs(mask_ds_dir, exist_ok=True)
    
    # Build reference image index
    ref_map, _ = build_reference_index(ref_dir)
    
    # Process each frame
    for i in frames:
        if i % 100 == 0:
            print(f"Processed {i}/{T} frames")
        
        # Read registered images
        mem_i = IO.read_reg_tiff(mem_dir, i, 1)  # Channel 1: membrane
        if dual_channel:
            ca_i = IO.read_reg_tiff(ca_dir, i, 0)    # Channel 0: calcium
        else:
            ca_i = np.zeros_like(mem_i)
        # Compute correlation map
        cor_i = compute_cor_fn(mem_i, ca_i)
        
        # Read corresponding reference image
        ref_i = tifffile.imread(ref_map[i])

        # Extract configuration parameters
        # win_size = config['win_size']      # Window size for SSIM computation
        # use_3d = config['use_3d']          # Whether to use 3D SSIM
        # sigma_3d = config['sigma_3d']      # Sigma for 3D Gaussian blur
        
        # Compute reliability mask using SSIM difference
        # mask_map = local_gradient_misalignment(cor_i, ref_i)
        mask_map = local_mind_difference(ref_i,
                                        cor_i,
                                        config['patch_sigma'],
                                        config['offset_radius'],
                                        config['structure_thresh'],
                                        config['eps'])
        if isinstance(mask_map,np.ndarray):
            # Save downsampled mask
            IO.write_multichannel_volume_as_ome_tiff(
                volume=[mask_map],      # single channel
                out_dir=out_dir,
                frame_idx=i,
                configPath=configPath,
                label='mask'
            )
        else:
            IO.write_multichannel_volume_as_ome_tiff(
                volume=[mask_map.get()],      # single channel
                out_dir=out_dir,
                frame_idx=i,
                configPath=configPath,
                label='mask'
            )