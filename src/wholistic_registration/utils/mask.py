import numpy as np
from skimage.morphology import binary_opening, binary_dilation, disk
from skimage.measure import label, regionprops

def bwareafilt3_wei(vol, size_range):
    """
    3D binary area filtering function that keeps only connected components 
    within a specified size range.
    
    Parameters:
    -----------
    vol : numpy.ndarray
        Input 3D volume (binary or numeric). Non-zero values are treated as foreground.
    size_range : list or tuple
        Two-element container [min_size, max_size] specifying the acceptable 
        size range for connected components (in number of voxels).
    
    Returns:
    --------
    numpy.ndarray
        Filtered 3D boolean volume containing only components within size range.
    """
    # Convert input volume to boolean array
    # Any non-zero values become True, zeros become False
    vol = vol.astype(bool)
    
    # Label connected components in 3D using 26-connectivity
    # Each separate connected component gets a unique integer label
    # Background (False voxels) remains labeled as 0
    labeled_vol = label(vol, connectivity=3) 
    
    # Extract properties of each labeled region
    # This gives us access to area, coordinates, and other properties
    # for each connected component
    regions = regionprops(labeled_vol)
    
    # Initialize output volume as all False (background)
    # This will store only the components that pass the size filter
    filtered_vol = np.zeros_like(vol, dtype=bool)
    
    # Iterate through each connected component and apply size filter
    for region in regions:
        # Get the area (number of voxels) of this connected component
        area = region.area
        
        # Check if component size falls within acceptable range (inclusive)
        if size_range[0] <= area <= size_range[1]:
            # If size is acceptable, set all voxels of this component to True
            # region.coords contains [row, col, depth] coordinates of all voxels
            # .T transposes to get separate arrays for each dimension
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
