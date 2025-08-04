import numpy as np
def pick_initial_reference(frames: np.ndarray, max_corr_frames: int = 20): 
    """ computes the initial reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    Parameters
    ----------
    frames : 3D array, int16
        size [frames x Ly x Lx], frames from binary

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """

    nimg, Ly, Lx = frames.shape
    frames = np.reshape(frames, (nimg, -1)).astype("float32") # Nimg x (Ly*Lx)
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1)) # Nimg x (Ly*Lx)
    cc = np.matmul(frames, frames.T) # Nimg x Nimg
    ndiag = np.sqrt(np.diag(cc)) # Nimg
    cc = cc / np.outer(ndiag, ndiag) # Nimg x Nimg
    CCsort = -np.sort(-cc, axis=1) # Nimg x Nimg
    ncorr_frames = min(max_corr_frames, nimg-1)
    bestCC = np.mean(CCsort[:, 1:ncorr_frames], axis=1) # Nimg
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    refImg = np.mean(frames[indsort[0:ncorr_frames], :], axis=0)
    refImg = np.reshape(refImg, (Ly, Lx))
    return refImg