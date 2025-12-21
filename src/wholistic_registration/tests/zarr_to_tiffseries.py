#%%
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
import dask
import dask.distributed
import dask.array as da
import dask.delayed
import tifffile as tf
from wholistic_registration.utils import converters

reload(converters)


zarr_path = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/f338_1023registrated_test_allframes.zarr"
zarr_path = "/nrs/ahrens/Virginia_nrs/wVT/fast_imaging/registration/f2013_0912registered_full.zarr"

nd2_path = "/nrs/ahrens/Virginia_nrs/wVT/fast_imaging/250705_f2013_ubi_gcamp7f_bactin_mcherry_6dpf_15842/exp1/nd2/*.nd2"
tiffolder = "/nrs/ahrens/Virginia_nrs/wVT/fast_imaging/250705_f2013_ubi_gcamp7f_bactin_mcherry_6dpf_15842/exp1/reg/"
os.makedirs(tiffolder, exist_ok=True)
nd2_files = glob(nd2_path)
nd2_file = nd2_files[0]
metadata = IO.readMeta_new(nd2_file)


z = zarr.open(zarr_path, mode="r")
mem = z["membrane"]
cal = z["calcium"]
ref = z["reference"]

# Practical example with your calcium data (with 2x downsampling):
output_tiff_dir = os.path.join(os.path.dirname(tiffolder), "calcium_tiffs")
delayed_saves = converters.save_tiff_series_parallel(cal[:10], output_tiff_dir, channel=0, verbose=True, 
                                        metadata=metadata, xy_downsample=2)

# Execute the parallel saves (uncomment to run):
print("Starting parallel TIFF saves...")
t0 = time.time()
results = dask.compute(*delayed_saves)
t1 = time.time()
print(f"Saved {len(results)} TIFFs in {(t1-t0):.2f} seconds")

#%%