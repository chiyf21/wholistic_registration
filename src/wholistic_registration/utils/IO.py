'''

version : 0.1
file name: IO.py

Code Author : Wei Zheng for matlab and Yunfeng Chi (Tsinghua University) and Virginia Ruetten (Janelia) for python
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
    

def readMeta_new(filePath,Ifprint=True):
    """
    Reads metadata from an ND2 file and optionally prints Z ratio and data size.

    Parameters:
        filePath (str): Path to the ND2 file.
        Ifprint (bool): Whether to print Z ratio and data size. Default is True.

    Returns:
        metadata (nd2.Metadata): Metadata object containing information about the ND2 file.

    """
    with nd2.ND2File(filePath) as ndf:
        data = ndf.to_dask()
        data_shape = data.shape

        if hasattr(ndf.metadata, "channels"):
            resolutionxyz = ndf.metadata.channels[0].volume.axesCalibration
            spacing_x = resolutionxyz[0]
            spacing_y = resolutionxyz[1]
            
            nchannels = len(ndf.metadata.channels)
            voxelCount = ndf.metadata.channels[0].volume.voxelCount
            nframes = ndf.shape[0]
            
            nxpix = voxelCount[0]
            nypix = voxelCount[1]
            if len(voxelCount) > 2:
                nzpix = voxelCount[2]
                spacing_z = resolutionxyz[2]
                zRatio = spacing_z/spacing_x
                
            else:
                spacing_z = 1
                zRatio = 1
                nzpix = 1


        metadata=ndf.metadata
        channels=metadata.channels[0]

        multichannel = True if nchannels > 1 else False
        multiz = True if nzpix > 1 else False
        multiframe = True if nframes > 1 else False

        try:
            avgdiff = ndf.experiment[0].parameters.periodDiff.avg/1000
            framerate = 1 / avgdiff
        except:
            t0 = ndf.frame_metadata(0).channels[0].time.relativeTimeMs
            t1 = ndf.frame_metadata(1).channels[0].time.relativeTimeMs
            dt_ms = t1 - t0
            if dt_ms <= 0:
                raise ValueError("Invalid timestamps: Δt <= 0")
            framerate = 1000.0 / dt_ms 

        if Ifprint:
            #get Zratio
            zRatio=spacing_z/spacing_x
            print("Z ratio is", zRatio)
            #get size
            print("Data size is",[nxpix,nypix,nzpix])
            #get total frames
            print("Total frames is",ndf.sizes['T'])


    metadata_dict = { # needed for ImageJ
    'Pixels': {
        'PhysicalSizeX': spacing_x,
        'PhysicalSizeXUnit': 'um',
        'PhysicalSizeY': spacing_y,
        'PhysicalSizeYUnit': 'um',
        'PhysicalSizeZ': spacing_z,
        'PhysicalSizeZUnit': 'um',
    },
    'loop': True,
    'fps': framerate,
    'zRatio': zRatio,
    'nframes': nframes,
    'nchannels': nchannels,
    'resolutionxyz': resolutionxyz,
    'data_shape': data_shape,
    'spacing_x': spacing_x,
    'spacing_y': spacing_y,
    'spacing_z': spacing_z,
    'axes': 'TZCYX',
    'SizeC': nchannels,
    'SizeT': nframes,
    'SizeZ': nzpix,
    'SizeX': nxpix,
    'SizeY': nypix,
    'multichannel': multichannel,
    'multiz': multiz,
    'multiframe': multiframe,
    'nzpix': nzpix,
    'nxpix': nxpix,
    'nypix': nypix,
    }

    return metadata_dict

def get_framerate(filePath):
    with nd2.ND2File(filePath) as f:
        t0 = f.frame_metadata(0).channels[0].time.relativeTimeMs
        t1 = f.frame_metadata(1).channels[0].time.relativeTimeMs
        dt_ms = t1 - t0
        if dt_ms <= 0:
            raise ValueError("Invalid timestamps: Δt <= 0")
        framerate = 1000.0 / dt_ms  
        return framerate, dt_ms
        
def getTotalFrames(filePath):
    with nd2.ND2File(filePath) as f:
        frame=f.sizes['T']
    return frame


def readND2Frame(filePath, frames, slices=None, channel=0, xy_down=1, to_memory=True):
    """
    ND2 reader with high-quality XY downsampling:
      - Z slice selection (via `slices`)
      - XY resampling using skimage.resize with anti-aliasing
    """
    from skimage.transform import resize
    from skimage.transform import downscale_local_mean

    if channel is None:
        channel = slice(None)
    # Normalize indexing to preserve dimensions
    def normalize_index(idx, name):
        """Convert various index types to dimension-preserving format"""
        if idx is None:
            return slice(None)
        elif isinstance(idx, (int, np.integer)):
            return slice(idx, idx + 1)  # Convert int to slice
        elif isinstance(idx, (list, tuple, np.ndarray)):
            # For lists, ensure at least 2 elements to preserve dimensions
            idx_list = list(idx)
            if len(idx_list) == 1:
                # Duplicate the single element to preserve dimension
                return slice(idx_list[0], idx_list[0] + 1)
            return idx_list
        elif isinstance(idx, slice):
            return idx
        else:
            raise ValueError(f"Invalid {name} index type: {type(idx)}")
    
    frames = normalize_index(frames, "frames")
    slices = normalize_index(slices, "slices") 
    channel = normalize_index(channel, "channel")


    with nd2.ND2File(filePath) as f:
        sizes = f.sizes
        dims = list(sizes.keys())
        is_5d = len(dims) == 5
        metadata = readMeta_new(filePath)
        nframes = metadata['nframes']
        nchannels = metadata['nchannels']
        nzpix = metadata['nzpix']


        # ------------------------------------
        # Read ND2 data (lazy dask tensor) - ALL FRAMES AT ONCE
        # ------------------------------------
        dask_data = f.to_dask()
        multiframe = True if nframes > 1 else False
        multiz = True if nzpix > 1 else False
        multichannel = True if nchannels > 1 else False

        print(f"multiframe is {multiframe}")
        print(f"multiz is {multiz}")
        print(f"multichannel is {multichannel}")
        print(f"nframes is {nframes}")
        print(f"nzpix is {nzpix}")
        print(f"nchannels is {nchannels}")
        
        # Add dimensions if they don't exist to ensure 5D structure
        if not multiframe:
            dask_data = dask_data[None]  # Add T dimension
        
        if not multiz:
            dask_data = dask_data[:, None]  # Add Z dimension
        
        if not multichannel:
            dask_data = dask_data[:, :, None]  # Add C dimension
        print(f"After adding dimensions: dask_data.shape = {dask_data.shape}, len(data.shape) = {len(dask_data.shape)}")

        # Apply indexing while preserving dimensions
        dask_data = dask_data[frames, slices, channel]

        # check if 5D
        if len(dask_data.shape) != 5:
            raise ValueError(f"After slicing: dask_data.shape = {dask_data.shape}, len(data.shape) = {len(dask_data.shape)}")
        
        
        print(f"After slicing: dask_data.shape = {dask_data.shape}")

        print(f"dask_data.shape is {dask_data.shape}")
        

        T, Z, C, Y, X = dask_data.shape

        # ------------------------------------
        # XY downsample using binning/averaging (dask-native)
        # ------------------------------------
        if xy_down > 1:
            print(f"Downsampling data by {xy_down}x")
            dask_data = downsample(dask_data, xy_down)

        # Only compute at the very end if requested
        data = dask_data.compute().astype(np.float32) if to_memory else dask_data


        return data   

def downsample(data_tzcyx, xy_down=4):
    T, Z, C, Y, X = data_tzcyx.shape
    newY = (Y // xy_down) * xy_down
    newX = (X // xy_down) * xy_down
    data_tzcyx = data_tzcyx[:, :, :, :newY, :newX]
    data_tzcyx = data_tzcyx.reshape(T, Z, C, newY // xy_down, xy_down, newX // xy_down, xy_down)
    data_tzcyx = data_tzcyx.mean(axis=(4, 6))
    return data_tzcyx

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

def saveTiff_new(image, save_path, config_path =None, metadata = None, verbose=True):
    # check dimension of image - should always by TZCYX
    if image.ndim == 2:
        image = image[None, None, None, :, :]
    if image.ndim != 5:
        raise ValueError("All saved should be 5D (TZCYX)")

    config_str = None
    # if config_path is not None:
    #     import json
    #     config_data = toml.load(config_path)
    #     config_str = json.dumps(config_data, ensure_ascii=False)
    # else:
    #     config_str = None

    if metadata is not None:
        spacing_x = metadata['spacing_x']
        spacing_y = metadata['spacing_y']
        metadata['data_shape'] = image.shape
    else:
        metadata = {}
        spacing_x = 1
        spacing_y = 1
        metadata['spacing_x'] = spacing_x
        metadata['spacing_y'] = spacing_y
        metadata['data_shape'] = image.shape

    if verbose:
        print(f"Saving TIFF file to {save_path}")
        print(f"  - shape: {image.shape}")
        print(f"  - spacing: {spacing_x}, {spacing_y}")
        print(f"  - config: {config_str}")

    with tifffile.TiffWriter(save_path, imagej=True) as tif:
        tif.write(image, metadata=metadata, resolution=(1.0/spacing_x, 1.0/spacing_y), description=config_str)


    
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
