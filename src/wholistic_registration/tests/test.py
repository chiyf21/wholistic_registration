# %%
import nd2
import os
from glob import glob
from wholistic_registration.utils import IO
from wholistic_registration.core import main_function
from importlib import reload
import numpy as np
import tifffile as tf
import zarr
import dask.array as da
import time

reload(IO)
reload(main_function)

tiffolder = "/nrs/ahrens/Virginia_nrs/wVT/pipeline_result_1218/f338_registrated_1218/"
subfolders = glob(tiffolder + "*")
downsample_folder = "/nrs/ahrens/Virginia_nrs/wVT/pipeline_result_1218/f338_registrated_1218/downsampled/"
os.makedirs(downsample_folder, exist_ok=True)

# each subfolder contains a series of tiffs, each tiff is a timepoint
# using dask to read files, save a downsampled version of the tiffs
print(len(subfolders))
for subfolder in subfolders[:1]:
    print(f"Processing subfolder: {subfolder}")
    downsample_subfolder = os.path.join(downsample_folder, os.path.basename(subfolder))
    os.makedirs(downsample_subfolder, exist_ok=True)
    
    t0 = time.time()
    
    # Use the new IO.downsample_tiff_series function
    data_ds = IO.downsample_tiff_series(subfolder, xy_down=4, verbose=True)
    
    t1 = time.time()
    print(f"Setup time: {(t1 - t0):.2f} seconds")
    print(f"Downsampled dask array shape: {data_ds.shape}")
    print(f"Downsampled dask array chunks: {data_ds.chunks}")
    
    
    
    
    # Optional: compute a small subset to test
    result_sample = data_ds[:2].compute()  # Compute first 2 timepoints
    print(f"Sample result shape: {result_sample.shape}")
    save_path = os.path.join(downsample_subfolder, "test.tif")
    IO.saveTiff_new(result_sample, save_path, verbose=True)

    
    # For very large datasets, you can use batch processing:
    # data_ds = IO.downsample_tiff_series(subfolder, xy_down=4, 
    #                                    batch_processing=True, batch_size=50, verbose=True)
    
    t2 = time.time()
    print(f"Total processing time: {(t2 - t0):.2f} seconds")

#%%
t0 = time.time()
data_ds = IO.downsample(data_tzcyx, xy_down=4).compute()
t1 = time.time()
print(f"Time taken: {(t1 - t0):.2f} seconds")
IO.saveTiff_new(data_ds, save_path, config_path=None, metadata=metadata, verbose=True)




#%%

nd2folder = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/exp0/anat/end/"

tiffolder = nd2folder + "tif/"
os.makedirs(tiffolder, exist_ok=True)

nd2files = glob(os.path.join(nd2folder, "*.nd2"))
nd2file = nd2files[0]
configfilePath = nd2file.replace(".nd2", "2.toml")

#%%
reload(IO)
reload(main_function)
metadata = IO.readMeta_new(nd2file)
config = main_function.DefineParams(inputFile=nd2file, configFile=configfilePath)

# %%
with nd2.ND2File(nd2file) as ndf:
    metadata = ndf.metadata
    data = ndf.to_dask()
    print(data.shape)
    tmp = data[0, 0, 0, 0, 0].compute()
    print(tmp.dtype)
# %%
reload(IO)
metadata = IO.readMeta_new(nd2file)
config = main_function.DefineParams(inputFile=nd2file, configFile=configfilePath)


sl = slice(0, 3)
ts = np.array([0, 2])
ts = None
sl = None
channel = None
data = IO.readND2Frame(
    nd2file, ts, slices=sl, channel=channel, xy_down=4, to_memory=True
)


image = data
save_path = os.path.join(tiffolder, "test.tif")
# configfilePath,
IO.saveTiff_new(image, metadata=metadata, save_path=save_path, verbose=True)


tmp = tf.imread(save_path)
print(tmp.shape)
# %%


zarr_path = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/f338_1023registrated_test_allframes.zarr"
z = zarr.open(zarr_path, mode="r")
mem = z["membrane"]
cal = z["calcium"]
ref = z["reference"]

save_path = os.path.join(tiffolder, "test_ds_from_zarr.tif")
data_tzcyx = da.from_array(cal[:][:, :, None], chunks=(1, 1, 1, "auto", "auto"))


t0 = time.time()
data_ds = IO.downsample(data_tzcyx, xy_down=4).compute()
t1 = time.time()
print(f"Time taken: {(t1 - t0):.2f} seconds")
IO.saveTiff_new(data_ds, save_path, config_path=None, metadata=metadata, verbose=True)
