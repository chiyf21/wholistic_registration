import os
import dask
import tifffile as tf


def save_tiff_series_parallel(data_array, output_dir, channel=0, prefix="vol", verbose=True, metadata=None, xy_downsample=1):
    """
    Save a time series as individual TIFF files in parallel using dask.
    
    Parameters:
        data_array: dask or numpy array with shape (T, Z, Y, X) or (T, Y, X)
        output_dir: directory to save TIFF files
        channel: channel number for filename (default: 0)
        prefix: filename prefix (default: "vol")
        verbose: print progress (default: True)
        metadata: metadata dictionary with spacing info (default: None)
        xy_downsample: XY downsampling factor (default: 1, no downsampling)
    
    Returns:
        list of delayed objects (call compute() to execute)
    """
    os.makedirs(output_dir, exist_ok=True)

    if metadata is not None:
        spacing_x = metadata['spacing_x']
        spacing_y = metadata['spacing_y']
    else:
        spacing_x = 1
        spacing_y = 1
        metadata = {}
        metadata['spacing_x'] = spacing_x
        metadata['spacing_y'] = spacing_y
        metadata['data_shape'] = data_array.shape
    
    # Adjust spacing for downsampling
    effective_spacing_x = spacing_x * xy_downsample
    effective_spacing_y = spacing_y * xy_downsample
    metadata['spacing_x'] = effective_spacing_x
    metadata['spacing_y'] = effective_spacing_y
    
    def downsample_xy(data, ds_factor):
        """Downsample XY dimensions by reshaping and taking mean"""
        if ds_factor == 1:
            return data
        
        if len(data.shape) == 3:  # (Z, Y, X)
            Z, Y, X = data.shape
            # Crop to make dimensions divisible by ds_factor
            new_Y = (Y // ds_factor) * ds_factor
            new_X = (X // ds_factor) * ds_factor
            data_cropped = data[:, :new_Y, :new_X]
            # Reshape and take mean
            data_ds = data_cropped.reshape(Z, new_Y // ds_factor, ds_factor, new_X // ds_factor, ds_factor)
            return data_ds.mean(axis=(2, 4))
        else:  # (Y, X)
            Y, X = data.shape
            # Crop to make dimensions divisible by ds_factor
            new_Y = (Y // ds_factor) * ds_factor
            new_X = (X // ds_factor) * ds_factor
            data_cropped = data[:new_Y, :new_X]
            # Reshape and take mean
            data_ds = data_cropped.reshape(new_Y // ds_factor, ds_factor, new_X // ds_factor, ds_factor)
            return data_ds.mean(axis=(1, 3))
    
    @dask.delayed
    def save_single_tiff(timepoint_data, filepath):
        """Save a single timepoint as TIFF with optional downsampling"""
        # Apply downsampling if requested
        if xy_downsample > 1:
            timepoint_data = downsample_xy(timepoint_data, xy_downsample)
        
        tf.imwrite(filepath, timepoint_data, compression='lzw', metadata=metadata, 
                  resolution=(1.0/effective_spacing_x, 1.0/effective_spacing_y))
        return filepath
    
    # Create delayed tasks for each timepoint
    delayed_tasks = []
    for t in range(data_array.shape[0]):
        filename = f"{prefix}_ch{channel}_{t:06d}.tif"
        filepath = os.path.join(output_dir, filename)
        
        # Extract timepoint data (handle both 3D and 4D cases)
        if len(data_array.shape) == 4:  # (T, Z, Y, X)
            timepoint_data = data_array[t]  # (Z, Y, X)
        else:  # (T, Y, X)
            timepoint_data = data_array[t]  # (Y, X)
        
        delayed_tasks.append(save_single_tiff(timepoint_data, filepath))
    
    if verbose:
        print(f"Created {len(delayed_tasks)} delayed save tasks")
        print(f"Files will be saved to: {output_dir}")
        print(f"Filename pattern: {prefix}_ch{channel}_XXXXXX.tif")
        if xy_downsample > 1:
            print(f"XY downsampling factor: {xy_downsample}x")
            if len(data_array.shape) == 4:  # (T, Z, Y, X)
                orig_Y, orig_X = data_array.shape[2], data_array.shape[3]
            else:  # (T, Y, X)
                orig_Y, orig_X = data_array.shape[1], data_array.shape[2]
            new_Y = (orig_Y // xy_downsample) * xy_downsample // xy_downsample
            new_X = (orig_X // xy_downsample) * xy_downsample // xy_downsample
            print(f"Output size: {orig_Y}x{orig_X} -> {new_Y}x{new_X}")
    
    return delayed_tasks
