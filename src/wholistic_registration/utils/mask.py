import numpy as np
from skimage.morphology import binary_opening, binary_dilation, disk
from skimage.measure import label, regionprops

def bwareafilt3_wei(vol, size_range):
    vol = vol.astype(bool)
    
    labeled_vol = label(vol, connectivity=3) 
    
    regions = regionprops(labeled_vol)
    
    filtered_vol = np.zeros_like(vol, dtype=bool)
    
    for region in regions:
        area = region.area
        if size_range[0] <= area <= size_range[1]:
            filtered_vol[tuple(region.coords.T)] = True
    return filtered_vol

def getMask(dat_mov, thres_factor):
    dat_mov = dat_mov.astype(np.float32)
    mu = np.mean(dat_mov)
    sigma = np.std(dat_mov)
    normalized = np.abs((dat_mov - mu) / sigma)
    mask = normalized > thres_factor
    selem = np.ones((3, 3,1), dtype=bool)
    mask = binary_opening(mask, selem)
    mask = binary_dilation(mask, selem)

    return mask
