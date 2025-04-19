import numpy as np

def getSmPnltNormFctr(dat_ref, option):
    """
    Calculate the smoothness penalty factor based on the gradients in the X and Y directions.

    Args:
        dat_ref (np.ndarray): The reference data with shape (z, x, y), where z, x, y are the dimensions.
        option (dict): Contains various options, including the mask_ref.

    Returns:
        float: The smoothness penalty factor.
    """
    # Calculate the X and Y gradients using central difference
    Ix = (dat_ref[2:, :, :] - dat_ref[:-2, :, :]) / 2
    Iy = (dat_ref[:, 2:, :] - dat_ref[:, :-2, :]) / 2

    # Apply the mask to exclude certain areas from the calculation
    mask_ref = option['mask_ref']
    
    # Calculate the mean squared gradients while excluding the masked areas
    Ix_squared = np.mean(Ix[~mask_ref[1:-1, :, :]]**2)
    Iy_squared = np.mean(Iy[~mask_ref[:, 1:-1, :]]**2)

    # Return the average of the squared gradients in both directions
    factor = (Ix_squared + Iy_squared) / 2
    return factor
