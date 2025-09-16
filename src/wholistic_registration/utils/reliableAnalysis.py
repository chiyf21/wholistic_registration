# reliablemask.py
from . import cp
from . import cupy_ndimage


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


def get_temporal_and_accumula_mask(dat_cor_lst: cp.ndarray,
                                   dat_ref_lst: cp.ndarray,
                                   option: dict):
    """
    Compute temporal mask and accumulative mask based on dat_cor and dat_ref.

    Args:
        dat_cor_lst: shape = (T,X,Y) or (T,X,Y,Z)
        dat_ref_lst: shape = (T,X,Y) or (T,X,Y,Z)
        option: dictionary, should contain at least:
            - n: sliding window size
            - n2, n3: accumulation threshold control
            - sigma2: Gaussian smoothing sigma
            - temporalMaskThres: Z-score threshold

    Returns:
        dat_temporalMask, dat_accumulaMask
    """
    T = dat_cor_lst.shape[0]
    spatial_shape = dat_cor_lst.shape[1:]

    dat_diff_lst = cp.zeros((option["n"],) + spatial_shape, dtype=cp.float32)
    dat_temporalMask = cp.zeros((T,) + spatial_shape, dtype=cp.bool_)
    dat_accumulaMask = cp.zeros((T,) + spatial_shape, dtype=cp.bool_)
    distCnt = cp.zeros(spatial_shape, dtype=cp.int32)

    for t in range(T):
        if t % 100 == 0:
            print("Processed ", t)
        dat_cor = cp.array(dat_cor_lst[t], dtype=cp.float32)
        dat_ref = cp.array(dat_ref_lst[t], dtype=cp.float32)

        dat_dif = cp.abs(dat_cor - dat_ref)
        dat_dif = cupy_ndimage.gaussian_filter(dat_dif, sigma=option["sigma2"])

        if t < option["n"]:
            dat_diff_lst[t % option["n"]] = dat_dif
            continue
        else:
            diff_mu = cp.mean(dat_diff_lst, axis=0)
            diff_sigma = cp.std(dat_diff_lst, axis=0) + 1e-6
            dat_z = (dat_dif - diff_mu) / diff_sigma

            temp = dat_z > option["temporalMaskThres"]
            dat_temporalMask[t] = temp

            distCnt += temp
            if t >= option["n3"]:
                distCnt -= dat_temporalMask[t - option["n3"]]

            dat_accumulaMask[t] = (dat_accumulaMask[t - 1] | (distCnt > option["n2"]))

            dat_diff_lst[t % option["n"]] = dat_dif

    return dat_temporalMask, dat_accumulaMask


def get_spatial_mask(dat_cor_lst: cp.ndarray, option: dict):
    """
    Estimate spatial mask (spatially unreliable regions) based on registration results.

    Args:
        dat_cor_lst: shape = (T,X,Y) or (T,X,Y,Z)
        option: dictionary, should contain at least:
            - sigma: smoothing factor
            - spatialMaskThres: threshold

    Returns:
        dat_spatialMask (bool array)
    """
    # Use the mean of the first 10 frames as template
    n_template = min(10, dat_cor_lst.shape[0])
    template = cp.mean(dat_cor_lst[:n_template], axis=0)

    Iamp = gradient_amplitude(template)
    Iamp_mu = cp.median(Iamp)
    Iamp_sigma = cp.std(Iamp) + 1e-6
    Iz = (Iamp - Iamp_mu) / Iamp_sigma

    L = option["sigma"] * 2 + 1
    Iz_sm = cupy_ndimage.gaussian_filter(Iz, sigma=L)

    dat_intMask = cupy_ndimage.median_filter(template < cp.median(template),
                                             size=(5, 5) if template.ndim == 2 else (5, 5, 1))
    if template.ndim == 2:
        size = (option["sigma"] * 6 + 1, option["sigma"] * 6 + 1)
    else:
        size = (option["sigma"] * 6 + 1, option["sigma"] * 6 + 1, 1)
    dat_spatialMask = cupy_ndimage.median_filter(Iz_sm < option["spatialMaskThres"], size=size)

    return (dat_spatialMask | dat_intMask).astype(cp.bool_)


def get_reliable_mask(dat_cor_lst: cp.ndarray,
                      dat_ref_lst: cp.ndarray,
                      option: dict):
    """
    Main entry: compute temporalMask, accumulaMask, spatialMask.

    Args:
        dat_cor_lst: (T,X,Y) or (T,X,Y,Z)
        dat_ref_lst: (T,X,Y) or (T,X,Y,Z)
        option: dictionary

    Returns:
        dat_temporalMask, dat_accumulaMask, dat_spatialMask
    """
    dat_temporalMask, dat_accumulaMask = get_temporal_and_accumula_mask(dat_cor_lst, dat_ref_lst, option)
    dat_spatialMask = get_spatial_mask(dat_cor_lst, option)
    return dat_temporalMask, dat_accumulaMask, dat_spatialMask
