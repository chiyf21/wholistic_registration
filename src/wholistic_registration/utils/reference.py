
import numpy as np
from . import cp
def transform(image,k=1,method="raw"):
    if method=="raw":
        return k*image
    elif method=="sqrt":
        return k*np.sqrt(image)
    elif method=="log2":
        return k*np.log2(1+image)
    elif method=="log10":
        return k*np.log10(1+image)
    else:
        raise ValueError(f"Unknown method to process the image:{method}")
def pick_initial_reference(frames: cp.ndarray, max_corr_frames: int = 20, downsample=1):
    """
    Compute the initial reference image/volume from a 2D or 3D movie sequence.
    For 3D data, only the central 3 slices are used to compute correlations.
    """

    ndim = frames.ndim
    if ndim == 3:
        # 2D data: [T, H, W]
        T, H, W = frames.shape
        frame_shape = (H, W)
        frames_used = frames[:, ::downsample, ::downsample]

    elif ndim == 4:
        # 3D data: [T, H, W, Z]
        T, H, W, Z = frames.shape
        frame_shape = (H, W, Z)

        # Determine which slices to use
        if Z >= 3:
            z_mid = Z // 2
            start = max(0, z_mid - 1)
            end = min(Z, z_mid + 2)
            z_indices = slice(start, end)  # 3 slices
        else:
            # If fewer than 3 slices, use all
            z_indices = slice(0, Z)

        frames_used = frames[:, ::downsample, ::downsample, z_indices]

    else:
        raise ValueError("Input 'frames' must be [T, H, W] or [T, H, W, Z].")

    # -------- Flatten --------
    frames_flat = cp.reshape(frames_used, (T, -1)).astype(cp.float32)

    # -------- Correlation matrix --------
    frames_mean = frames_flat.mean(axis=1, keepdims=True)
    frames_demeaned = frames_flat - frames_mean

    cc = cp.matmul(frames_demeaned, frames_demeaned.T)
    diag = cp.sqrt(cp.diag(cc)) + 1e-12
    cc = cc / cp.outer(diag, diag)

    # -------- Find most correlated frame --------
    ncorr_frames = min(max_corr_frames, T - 1)
    CCsort = -cp.sort(-cc, axis=1)
    bestCC = cp.mean(CCsort[:, 1:ncorr_frames], axis=1)
    imax = cp.argmax(bestCC)

    indsort = cp.argsort(-cc[imax, :])
    top_indices = indsort[:ncorr_frames]

    # -------- Average top correlated frames (using original resolution frames) --------
    refImg = cp.mean(frames[top_indices], axis=0)

    return refImg, top_indices


def compute_reference_from_block(mem_block, ca_block,config):
    """Generate a reference image from a block of frames"""
    k=config['channels']['k']
    function=config['channels']['function']
    dual_channels=config['channels']['dual_channel']
    frames=max(len(mem_block)//2,50)
    mem_block = cp.asarray(mem_block)
    mem_ref, indsort = pick_initial_reference(mem_block,max_corr_frames=frames)
    if dual_channels:
        ca_block = cp.asarray(ca_block)
        Ca_ref = cp.mean(ca_block[indsort, :], axis=0)
        Ca_ref_transform =transform(Ca_ref,k,function)

        return (mem_ref + Ca_ref_transform).get()
    else:
        return mem_ref.get()
