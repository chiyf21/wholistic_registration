import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from . import calFlow3d_Wei_v1
from . import IO
import os
def generateMotion(raw_data, art_R, amp_art,zRatio=1):
    """
    generate simulation(amp and r)
    
    Parameters:
        res_path_name: Path to result directory
        reader: Metadata reader (must implement read_meta method)
        art_R: Artifact-related parameter
        amp_art: Amplitude parameter
        *args: Optional arguments, used to provide zRatio
        
    Returns:
        motion_X, motion_Y, motion_Z: Motion data arrays
        cp_art: Indices of randomly selected points
    """

    
    # Calculate filter sigma values for 3D Gaussian filtering
    filter_sigma = np.array([art_R, art_R, art_R / zRatio])
    
    [X,Y,Z]=raw_data.shape
    N=X*Y*Z  # Total number of points in the 3D volume

    # Calculate number of points to select
    ratio_art = 1 / ((art_R * 2 + 1) **2)
    L = round(N * ratio_art)  # Number of points to select
    
    # Randomly select L points using permutation
    cp_art = np.random.permutation(N)[:L]
    
    # Initialize motion arrays with single precision
    motion_X = np.zeros((X, Y, Z), dtype=np.float32)
    motion_Y = np.zeros((X, Y, Z), dtype=np.float32)
    motion_Z = np.zeros((X, Y, Z), dtype=np.float32)
    
    # Add Gaussian noise to randomly selected points
    motion_X.flat[cp_art] = np.random.randn(L)
    motion_Y.flat[cp_art] = np.random.randn(L)
    
    # Apply 3D Gaussian filtering
    motion_X = gaussian_filter(motion_X, sigma=filter_sigma)
    motion_Y = gaussian_filter(motion_Y, sigma=filter_sigma)
    # motion_Z = gaussian_filter(motion_Z, sigma=filter_sigma)  # Commented out as in original code
    
    # Calculate scaling factors based on standard deviation
    # Note: Using motion_Y's standard deviation for both X and Y scaling
    # Preserving original behavior even if potentially a typo
    factor_X = np.std(motion_Y.flat[cp_art])
    factor_Y = np.std(motion_Y.flat[cp_art])
    
    # Apply amplitude scaling
    motion_X = motion_X / factor_X * amp_art
    motion_Y = motion_Y / factor_Y * amp_art
    
    return motion_X, motion_Y, motion_Z, cp_art



def generate_single_simulated_data(original_data_path, frame,crop_region, r_value, amp_value, noise_level,
                                  ):
    # crop region
    x_start, y_start, z_start, x_size, y_size, z_size = crop_region
    crop_range_x = slice(x_start, x_start + x_size)
    crop_range_y = slice(y_start, y_start + y_size)
    crop_range_z = slice(z_start, z_start + z_size)
    
    
    #load the data
    print("Loading raw data...")
    meta = IO.readMeta(original_data_path)
    dat_org = IO.readFrame(original_data_path,frame,1)
    
    # zRatio
    channels=meta.channels[0]
    axesCalibration=channels.volume.axesCalibration
    zRatio=axesCalibration[2]/axesCalibration[0]

    # generate motion(amp and r)
    print(f"generation motion (R={r_value}, Amp={amp_value})...")
    motion_x, motion_y, motion_z, _ = generateMotion(
        dat_org ,r_value, amp_value, zRatio
    )
    
    motion_current_real = np.stack([motion_x, motion_y, motion_z], axis=3)

    print("apply motion")
    dat_mov_raw = calFlow3d_Wei_v1.correctMotion(dat_org, -motion_current_real)  # 应用负向运动
    dat_ref_raw = calFlow3d_Wei_v1.correctMotion(dat_mov_raw, motion_current_real)  # 校正回参考位置
    
    print(f"add noise (noise level: {noise_level})")
    dat_mov = dat_mov_raw + np.random.randn(*dat_org.shape) * noise_level
    dat_ref = dat_ref_raw + np.random.randn(*dat_org.shape) * noise_level
    

    print("crop the data to the given region")
    dat_mov_cropped = dat_mov[crop_range_x, crop_range_y, crop_range_z]
    dat_ref_cropped = dat_ref[crop_range_x, crop_range_y, crop_range_z]
    motion_cropped = motion_current_real[crop_range_x, crop_range_y, crop_range_z, :]
    
    crop_info = {
        'crop_range_x': np.arange(x_start, x_start + x_size),
        'crop_range_y': np.arange(y_start, y_start + y_size),
        'crop_range_z': np.arange(z_start, z_start + z_size)
    }
    
    return dat_mov_cropped, dat_ref_cropped, motion_cropped, crop_info

