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

# Start Dask client with dashboard for progress monitoring
from dask.distributed import Client, progress
import dask

# Try to connect to existing cluster first, or create a new one
try:
    # Try to connect to existing cluster
    client = Client('localhost:8786', timeout='2s')
    print(f"Connected to existing Dask cluster at: {client.dashboard_link}")
except:
    # If no existing cluster, create an optimized configuration for your system
    # Your system: 80 CPUs, 376GB RAM, NUMA architecture
    
    # Option 1: High-performance threaded (recommended for TIFF I/O)
    client = Client(processes=False, threads_per_worker=8, n_workers=8, memory_limit='40GB')
    print(f"Created optimized threaded Dask client at: {client.dashboard_link}")
    print(f"Using 64 threads (8 workers × 8 threads) with 320GB total memory")
    
    # Option 2: Process-based (uncomment if you need true parallelism)
    # client = Client(n_workers=16, threads_per_worker=4, memory_limit='20GB')
    # print(f"Created process-based Dask client with 64 cores and 320GB memory")
    
    # Option 3: NUMA-aware configuration (uncomment for maximum performance)
    # from dask.distributed import LocalCluster
    # cluster = LocalCluster(n_workers=16, threads_per_worker=4, memory_limit='20GB',
    #                       processes=True, dashboard_address=':8787')
    # client = Client(cluster)

# Alternative: Use synchronous scheduler (no dashboard but no multiprocessing issues)
# dask.config.set(scheduler='synchronous')
# print("Using synchronous scheduler (no dashboard)")
# client = None

# Simple approach: Save each zarr frame as a TIFF with downsampling
output_tiff_dir = os.path.join(os.path.dirname(tiffolder), "calcium_tiffs")
os.makedirs(output_tiff_dir, exist_ok=True)

reload(converters)
print("Creating simple delayed tasks - one per frame...")
delayed_tasks = converters.save_zarr_as_tiffs_simple(cal, output_tiff_dir, n_frames=None, xy_downsample=2, channel=0, metadata=metadata)

print(f"Created {len(delayed_tasks)} simple tasks. Processing...")

# Option 1: Use threaded scheduler (more stable for I/O)
print("Using threaded scheduler for stability...")
t0 = time.time()
with dask.config.set(scheduler='threads', num_workers=16):
    results = dask.compute(*delayed_tasks)
print(f"Saved {len(results)} TIFF files")
t1 = time.time()
print(f"Time taken: {t1-t0:.2f} seconds")

# Option 2: Process in smaller chunks to avoid overwhelming scheduler (uncomment if needed)
# chunk_size = 100
# all_results = []
# for i in range(0, len(delayed_tasks), chunk_size):
#     chunk = delayed_tasks[i:i+chunk_size]
#     print(f"Processing chunk {i//chunk_size + 1}/{(len(delayed_tasks) + chunk_size - 1)//chunk_size}")
#     chunk_results = dask.compute(*chunk)
#     all_results.extend(chunk_results)
# print(f"Saved {len(all_results)} TIFF files")

# Close client when done (if it exists)
try:
    if client is not None:
        client.close()
except:
    pass

#%%