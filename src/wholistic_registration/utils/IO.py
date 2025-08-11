'''

version : 0.1
file name: IO.py

Code Author : Wei Zheng for matlab and Yunfeng Chi (Tsinghua University) for python
Last Update Date : 2025/8/05

Overview:
    This module provides functions to read metadata and frames from ND2 files, specifically designed for handling 3D imaging data. It includes functionality to read metadata, extract specific frames, and handle both single and multiple frame requests efficiently.

        
functions:
    - readMeta(filePath, Ifprint=True): Reads metadata from an ND2 file and optionally prints Z ratio and data size.
    - readFrame(filePath, frame, channel=0, to_memory=True): Reads specified frames from an ND2 file, supporting both single and multiple frame requests, and handles 5D data structures.
    
'''
import nd2
import numpy as np

def readMeta(filePath,Ifprint=True):
    """
    Reads metadata from an ND2 file and optionally prints Z ratio and data size.

    Parameters:
        filePath (str): Path to the ND2 file.
        Ifprint (bool): Whether to print Z ratio and data size. Default is True.

    Returns:
        metadata (nd2.Metadata): Metadata object containing information about the ND2 file.

    """
    with nd2.ND2File(filePath) as f:
        metadata=f.metadata
        channels=metadata.channels[0]
        if Ifprint:
            #get Zratio
            axesCalibration=channels.volume.axesCalibration
            zRatio=axesCalibration[2]/axesCalibration[0]
            print("Z ratio is", zRatio)

            #get size
            print("Data size is",channels.volume.voxelCount)

            #get total frames
            print("Total frames is",f.sizes['T'])

    return metadata


def readFrame(filePath, frame, channel=0, to_memory=True):
    """
    Reads specified frames from an ND2 file, supporting both single and multiple frame requests, and handles 5D data structures.

    Parameters:
        filePath (str): Path to the ND2 file.
        frame (int or list/array): Frame index or indices to read. If an integer, reads a single frame; if a list/array, reads multiple frames.
        channel (int): Channel index to read, default is 0.
        to_memory (bool): Whether to load data into memory. Default is True.

    Returns:
        frame_data(np.ndarray): If a single frame is requested, returns the data for that frame. If multiple frames are requested, returns an array containing all requested frames.
    
    """
    with nd2.ND2File(filePath) as f:
        sizes = f.sizes
        dims = list(sizes.keys())
        
        # Check if the data is 5D (T, Z, C, Y, X)
        is_5d = len(dims) == 5
        
        # Ensure frame is iterable; if it’s a single frame, convert it to a list
        if isinstance(frame, (int, np.integer)):
            frames_to_read = [frame]
            is_single_frame = True
        else:
            frames_to_read = list(frame)
            is_single_frame = False
        
        # Initialize a list to hold the data for each frame
        frame_data = []
        
        for t in frames_to_read:
            # Check if the frame index is within the valid range
            if t < 0 or t >= sizes['T']:
                raise ValueError(f"Time error")
            
            # read the data according to the data structure
            if is_5d:
                # if 5D structure: T,Z,C,Y,X; read the data from Z dimension
                data = f.to_dask()[t, :, channel, :, :].transpose(1,2,0)
            else:
                # if 4D structure: T,C,Y,X; read the data from T dimension
                data = f.to_dask()[t, channel, :, :]
            
            # if needed, record the data to memory
            if to_memory:
                data = data.compute()
            
            frame_data.append(data)
        
        # if only one frame is requested, return the single frame data
        # otherwise, stack the frames into a single array
        if is_single_frame:
            return frame_data[0]
        else:
            return np.stack(frame_data)


def saveTiff(image_list, config_path, save_path):
    """
    Saves a list of image frames as a multi-page TIFF file and embeds configuration
    data from a TOML file into the TIFF metadata.

    Parameters:
        image_list (list[np.ndarray]): A list where each element is a 2D or 3D NumPy array
                                       representing one image frame (H, W) or (H, W, C).
        config_path (str): Path to the TOML configuration file.
        save_path (str): Path to save the resulting TIFF file.

    Returns:
        None

    Notes:
        - The content of the TOML file will be serialized into a JSON string and stored
          in the TIFF ImageDescription tag for later retrieval.
        - All images will be converted to uint8 before saving if they are not already.
        - The function uses the tifffile library for writing multi-page TIFF files.
    """
    config_data = toml.load(config_path)

    import json
    config_str = json.dumps(config_data, ensure_ascii=False)

    # check the list
    for i, img in enumerate(image_list):
        if not isinstance(img, np.ndarray):
            raise ValueError(f"element{i} is not a image")


    tifffile.imwrite(
        save_path,
        image_list,
        description=config_str
    )

def readTifff(tiff_path):
    #haven't tested
    import json

    with tifffile.TiffFile(tiff_path) as tif:
        images = [page.asarray() for page in tif.pages]
        desc = tif.pages[0].tags["ImageDescription"].value
        config = json.loads(desc)

    return images, config
