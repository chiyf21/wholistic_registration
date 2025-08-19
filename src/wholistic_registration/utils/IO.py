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
import toml
import tifffile
import json
import zarr

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
        description=config_str,
        bigtiff=True
    )
def saveZarr(mem_data, ca_data, reference, config_path, save_path,
             chunks=(1, 512, 512)):
    """
    Save membrane channel, calcium channel, and reference image into one Zarr store.

    Parameters:
        mem_data (np.ndarray): Membrane channel data, shape (T, H, W) or (T, H, W, C).
        ca_data  (np.ndarray): Calcium channel data, shape (T, H, W) or (T, H, W, C).
        reference (np.ndarray): Reference image (2D).
        config_path (str): Path to the TOML configuration file.
        save_path (str): Path to save the resulting Zarr dataset (directory).
        chunks (tuple): Chunk size for Zarr storage, default (1, 512, 512).

    Returns:
        None

    """
    config_data = toml.load(config_path)
    config_str = json.dumps(config_data, ensure_ascii=False)

    # ensure numpy array
    mem_data = np.asarray(mem_data,dtype=np.float32)
    ca_data = np.asarray(ca_data,dtype=np.float32)
    reference = np.asarray(reference,dtype=np.float32)



    # open zarr root
    root = zarr.open(save_path, mode='w')

    # create datasets
    root.create_dataset("membrane", data=mem_data, chunks=chunks, overwrite=True)
    root.create_dataset("calcium", data=ca_data, chunks=chunks, overwrite=True)
    root.create_dataset("reference", data=reference, overwrite=True)  # usually 2D, so no chunks needed

    # save config
    root.attrs["config"] = config_str

    print(f"Saved Zarr dataset at {save_path}")
    print(f"  - membrane: {mem_data.shape}")
    print(f"  - calcium : {ca_data.shape}")
    print(f"  - reference: {reference.shape}")

def saveZarr_fast(mem_data, ca_data, reference, config_path, save_path,
                  chunks=(16, 512, 512), compressor=None, single_file=False):
    """
    Fast Zarr saving for membrane, calcium and reference data.

    Parameters:
        mem_data (np.ndarray): Membrane channel, shape (T,H,W) or (T,H,W,C)
        ca_data (np.ndarray): Calcium channel, shape (T,H,W) or (T,H,W,C)
        reference (np.ndarray): Reference image, shape (H,W)
        config_path (str): TOML configuration file path
        save_path (str): Output Zarr directory or file (if single_file=True)
        chunks (tuple): Chunk size for Zarr
        compressor: Zarr compressor (default: fast Blosc zstd)
        single_file (bool): Whether to save as single file (ZipStore)

    Returns:
        None
    """
    import zarr
    import json
    import toml
    from numcodecs import Blosc

    # default compressor
    if compressor is None:
        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)

    # load config
    config_data = toml.load(config_path)
    config_str = json.dumps(config_data, ensure_ascii=False)

    # switch to numpy
    mem_data = np.asarray(mem_data, dtype=np.float32)
    ca_data  = np.asarray(ca_data,dtype=np.float32)
    reference = np.asarray(reference,dtype=np.float32)

    # open Zarr store
    if single_file:
        store = zarr.ZipStore(save_path + ".zip", mode='w')
    else:
        store = save_path  
    root = zarr.open(store, mode='w')

    # create datasets
    root.create_dataset("membrane", data=mem_data, chunks=chunks, compressor=compressor, overwrite=True)
    root.create_dataset("calcium", data=ca_data, chunks=chunks, compressor=compressor, overwrite=True)
    root.create_dataset("reference", data=reference, compressor=compressor, overwrite=True)

    # save config
    root.attrs["config"] = config_str

    print(f"Saved fast Zarr at {save_path}")
    print(f"  - membrane: {mem_data.shape}")
    print(f"  - calcium : {ca_data.shape}")
    print(f"  - reference: {reference.shape}")
                      
def readTifff(tiff_path):
    #haven't tested
    import json

    with tifffile.TiffFile(tiff_path) as tif:
        images = [page.asarray() for page in tif.pages]
        desc = tif.pages[0].tags["ImageDescription"].value
        config = json.loads(desc)

    return images, config
