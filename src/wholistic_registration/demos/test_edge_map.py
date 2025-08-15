# %%

import wholistic_registration
from wholistic_registration.utils import preprocess, calFlow3d_Wei_v1
import numpy as np
import nd2
from glob import glob
import os
from wholistic_registration import utils
from wholistic_registration.utils import reference
from wholistic_registration.utils import mask
from importlib import reload
import tifffile as tf
from time import time
import cupy as cp
import matplotlib.pyplot as pl

t0 = time()

base_dir = "/nrs/ahrens/Virginia_nrs/wVT/fast_imaging/250705_f2013_ubi_gcamp7f_bactin_mcherry_6dpf_15842/exp1/"

save_dir = base_dir + "registration_result/"
tif_dir = save_dir + "tif/"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(tif_dir, exist_ok=True)

nd2_files = sorted(glob(os.path.join(base_dir, "nd2/*.nd2")))
nd2_file = nd2_files[0]

print(f"nd2_file is {nd2_file}")


with nd2.ND2File(nd2_file) as f:
    metadata = f.metadata
    nframes = metadata.contents.frameCount

midframe_ind = nframes // 2

max_corr_frames = 20
nRefFramesBuffer = 200
refFramesBuffer_inds = np.arange(
    midframe_ind - nRefFramesBuffer // 2, midframe_ind + nRefFramesBuffer // 2 + 1
)
ca_channel_ind = 0
ref_channel_ind = 1
# %%
slx = slice(None, None)
sly = slice(None, None)
# slx = slice(350, 1000)
# sly = slice(50, 400)
with nd2.ND2File(nd2_file) as f:
    dask_data_moving = f.to_dask()[refFramesBuffer_inds, :, sly, slx].compute()

    ref_frames = dask_data_moving[
        :,
        ref_channel_ind,
    ]
    ref_frame, ref_frame_inds = reference.pick_initial_reference(
        ref_frames, max_corr_frames
    )
    ref_ca_frames = dask_data_moving[ref_frame_inds, ca_channel_ind, :, :].mean(0)
    dask_data_calcium = f.to_dask()[:,  ca_channel_ind, sly, slx]

# make figure with two subplots with ref_frame and ref_ca_frames
clim = (400, 800)
fig, axs = pl.subplots(1, 2, figsize=(10, 5))
im0 = axs[0].imshow(ref_frame, cmap="gray")
im1 = axs[1].imshow(ref_ca_frames, cmap="gray")
pl.colorbar(im0, ax=axs[0], shrink=0.5)
pl.colorbar(im1, ax=axs[1], shrink=0.5)
im0.set_clim(clim)
im1.set_clim(clim)
pl.show()
pl.savefig(os.path.join(save_dir, "ref_frame_ca_frames.pdf"))



#%%

from skimage.filters import gaussian, sobel
from scipy.ndimage import gaussian_filter
from wholistic_registration.utils import preprocess
reload(preprocess)
trange = np.arange(0, 40000, 10000)
tmp_u16 = dask_data_calcium[trange].compute()   # (T, C, Y, X), dtype=uint16
# scale data between 10000 and 100000
# Convert to float32 in [0,1] no matter the input dtype
tmp = tmp_u16.astype(np.float32)
background = 400
tmp = tmp-background
tmp[tmp<0] = 0
tmp = tmp

sigma_xy = 6
r = sigma_xy*2
eps = 1e-6
edges = np.array([preprocess.michelson_edge_map(frame[:,:], sigma_xy=sigma_xy, r=r, eps=eps)[None] for frame in tmp])

fig, axs = pl.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(edges[0,0], cmap="gray")
axs[0].axis("off")
axs[1].imshow(edges[-1,0], cmap="gray")
axs[1].axis("off")
pl.tight_layout()
#%%
vmin = 000
vmax = 500
fig, axs = pl.subplots(1, 2, figsize=(10, 5))
im = axs[0].imshow(tmp[0], cmap="gray",vmin = vmin, vmax = vmax)
axs[0].axis("off")
axs[1].imshow(tmp[-1], cmap="gray",vmin = vmin, vmax = vmax )
axs[1].axis("off")
# pl.colorbar(im, ax=axs[0], shrink=0.5)
pl.tight_layout()
#%%
fig, axs = pl.subplots(1, 1, figsize=(10, 5))
axs = [axs]
axs[0].imshow(edges[0,0], cmap="gray")
axs[0].axis("off")
# axs[1].imshow(edges[-1,0], cmap="gray")
# axs[1].axis("off")
# pl.tight_layout()

