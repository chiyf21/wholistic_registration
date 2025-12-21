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
reload(mask)
reload(preprocess)
reload(calFlow3d_Wei_v1)
reload(wholistic_registration)





reload(reference)

t0 = time()
reload(utils)
reload(preprocess)
reload(calFlow3d_Wei_v1)
reload(wholistic_registration)

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

# take the 2D fft of the ref_frame and plit

fft_ref_frame = np.fft.fft2(ref_frame)
fft_ref_frame_shifted = np.abs(np.fft.fftshift(fft_ref_frame))
vmin = np.percentile(fft_ref_frame_shifted, 1)
vmax = np.percentile(fft_ref_frame_shifted, 99)
pl.figure()
pl.imshow(fft_ref_frame_shifted, cmap="gray", vmin=vmin, vmax=vmax)
pl.colorbar()
pl.axis("off")
pl.show()

# %%

test_frame_ind = midframe_ind - 3000
with nd2.ND2File(nd2_file) as f:
    test_data = f.to_dask()[test_frame_ind : test_frame_ind + 1, :, sly, slx].compute()
    test_frame = test_data[:, ref_channel_ind, :, :].mean(0)
    test_ca_frame = test_data[:, ca_channel_ind, :, :].mean(0)

# %% ######## PLOT TEST AND REFERENCE ########
clim = (400, 800)
fig, axs = pl.subplots(1, 2, figsize=(10, 5))
im0 = axs[0].imshow(test_frame, cmap="gray")
im1 = axs[1].imshow(test_ca_frame, cmap="gray")
pl.colorbar(im0, ax=axs[0], shrink=0.5)
pl.colorbar(im1, ax=axs[1], shrink=0.5)
im0.set_clim(clim)
im1.set_clim(clim)
pl.tight_layout()
pl.savefig(os.path.join(save_dir, "test_frame_ca_frames.pdf"))
# %% ######## PLOT DIFFERENCE BETWEEN TEST AND REFERENCE ########
clim = (-250, 250)
fig, axs = pl.subplots(1, 2, figsize=(10, 5))
im0 = axs[0].imshow(test_frame - ref_frame, cmap="gray")
im1 = axs[1].imshow(test_ca_frame - ref_ca_frames, cmap="gray")
pl.colorbar(im0, ax=axs[0], shrink=0.5)
pl.colorbar(im1, ax=axs[1], shrink=0.5)
im0.set_clim(clim)
im1.set_clim(clim)
pl.tight_layout()
pl.savefig(os.path.join(save_dir, "test_frame_ca_frames_diff.pdf"))

# %%
reload(calFlow3d_Wei_v1)

frameJump = 30000
# frameJump=1
refLength = 5
refJump = 20 / frameJump
initialLength = 5

# %%

option = {
    "layer": 2,  # number of pyramid layers
    "iter": 10,  # number of iterations per layer
    "r": 5,  # filter radius
    "motion": 0,  # initial motion field
    "mask_ref": 0,  # initial mask for reference
    "mask_mov": 0,  # initial mask for moving
    "save_ite": 5000,  # save every 50 iterations
    "movRange": 10,  # larger the less penalized
    "thresFactor": 5,
    "mask_size_range": [5, 500],
    "smoothPenalty_raw": 0.05,
    "tol": 1e-4,
}
reload(mask)

step = 200
nsteps = 30
tRange = np.arange(midframe_ind, midframe_ind + nsteps * step, step)
T = len(tRange)
print(f"tRange is {tRange}, length is {len(tRange)}")
with nd2.ND2File(nd2_file) as f:
    metadata = f.metadata
    channels = metadata.channels[0]

    # get Zratio
    axesCalibration = channels.volume.axesCalibration
    zRatio = axesCalibration[2] / axesCalibration[0]
    option["zRatio"] = zRatio

    print("Z ratio is", zRatio)
    [X, Y, Z] = channels.volume.voxelCount

    if slx.stop is not None and sly.stop is not None:
        X = slx.stop - slx.start
        Y = sly.stop - sly.start

    print("Data size is", [X, Y, Z])

    # get total frames
    frames = metadata.contents.frameCount
    print("Total frames is", frames)

    option["motion"] = np.zeros([X, Y, Z, 3])

    if Z > 1:
        # load all the data(virtual)
        dask_data_moving = f.to_dask()[:, :, ref_channel_ind, sly, slx]
        dask_data_calcium = f.to_dask()[:, :, ca_channel_ind, sly, slx]
    else:
        dask_data_moving = f.to_dask()[:, ref_channel_ind, sly, slx][:, None]
        dask_data_calcium = f.to_dask()[:, ca_channel_ind, sly, slx][:, None]

    dat_ref = ref_frame[None, :, :].transpose(2, 1, 0)
    print(f"dat_ref.shape is {dat_ref.shape}")

    option["mask_ref"] = mask.getMask(dat_ref, option["thresFactor"])
    option["mask_ref"] = mask.bwareafilt3_wei(option["mask_ref"], option["mask_size_range"])

    Pnltfactor = preprocess.getSmPnltNormFctr(
        dat_ref, option
    )  # average of the squared gradients in both x-y directions
    smoothPenalty = Pnltfactor * option["smoothPenalty_raw"]
    option["smoothPenalty"] = smoothPenalty
    print(f"smoothPenalty is {smoothPenalty}")
    results = {}
    results["tinds"] = []
    results["dat_mov"] = []
    results["dat_corrected"] = []
    results["dat_calcium_corrected"] = []
    results["motion"] = []
    results["error"] = []

    # start registration
    for tCnt in range(len(tRange)):
        t = tRange[tCnt]
        print(f"reading data... \n frame number {tCnt}, time point {t}")
        dat_mov = dask_data_moving[t].compute().transpose(2, 1, 0)

        option['mask_mov'] = mask.getMask(dat_mov,option["thresFactor"])
        option['mask_mov'] = mask.bwareafilt3_wei(option['mask_mov'],option["mask_size_range"])
        print("generate reference...")
        # if (tCnt - 1) % refJump == 0:
        #     if tCnt > refJump:
        #         refPossible = np.int32(np.min([tCnt//refJump, refLength]))
        #         # print(f"refPossible is {refPossible}")
        #         ref_range = np.arange(tCnt - refPossible * refJump, tCnt, refJump)
        #         # print(f"ref_range is {ref_range}")
        #         # Compute median along time axis (axis=3 for 4D array)
        #         dat_ref = np.median(dat_moving[:, :, :, ref_range], axis=3).astype(np.float32)


        # Update penalty factor
        # pnlt_factor = preprocess.getSmPnltNormFctr(dat_ref, option)
        # smoothPenalty=Pnltfactor*smoothPenalty_raw
        verbose = True
        motion_current, currentError, coords_new, error_log = (
            utils.calFlow3d_Wei_v1.getMotion(
                dat_mov, dat_ref, option, verbose=verbose
            )
        )

        dat_calcium_corrected = calFlow3d_Wei_v1.correctMotion(dask_data_calcium[t].compute().transpose(2,1,0),motion_current).transpose(2,1,0)[None,:,None] # shape: (1, Z, 1)
        dat_mov_corrected = calFlow3d_Wei_v1.correctMotion(dat_mov, motion_current).transpose(2,1,0)[None,:,None] # shape: (1, Z, 1)

        results["tinds"].append(t)
        results["dat_mov"].append(dat_mov)
        results["dat_corrected"].append(dat_mov_corrected)
        results["dat_calcium_corrected"].append(dat_calcium_corrected)
        results["motion"].append(motion_current)
        results["error"].append(currentError)


    results["dat_corrected"] = np.concatenate(results["dat_corrected"], axis=0)
    results["dat_calcium_corrected"] = np.concatenate(results["dat_calcium_corrected"], axis=0)
    results["dat_mov"] = np.concatenate(results["dat_mov"], axis=0)
    results["motion"] = np.array(results["motion"])
    results["error"] = np.array(results["error"])
    results["tinds"] = np.array(results["tinds"])

    nregframes = len(tRange)
    data_refs = np.repeat(dat_ref.transpose(2, 1, 0)[None, :, None], nregframes, axis=0)
    data_refs = data_refs.astype("float32")
    dat_agglomerated = np.concatenate([results["dat_mov"], data_refs, results["dat_corrected"], results["dat_calcium_corrected"]], axis=2)
    metadata = {}
    spacing_x = 1.0 / axesCalibration[0]
    spacing_y = 1.0 / axesCalibration[1]
    with tf.TiffWriter(
        os.path.join(
            tif_dir, f"registered_data_all.ome.tif"
        ),
        imagej=True,
    ) as tif:
        tif.write(
            dat_agglomerated,
            metadata=metadata,
            resolution=(1.0 / spacing_x, 1.0 / spacing_y),
        )



#%%

from skimage.filters import gaussian, sobel
from scipy.ndimage import gaussian_filter
trange = np.arange(0, 40000, 10000)
tmp_u16 = dask_data_calcium[trange].compute()  # (T, C, Y, X), dtype=uint16
# scale data between 10000 and 100000
# Convert to float32 in [0,1] no matter the input dtype
tmp = tmp_u16.astype(np.float32)
background = 400
tmp = tmp-background
tmp[tmp<0] = 0
scale = 10000
tmp = tmp*scale




sigma_px = 3
sig = (0.0, 0.0, sigma_px, sigma_px)
Ft = sobel(gaussian_filter(tmp, sigma=sig, mode="nearest"), axis = (-1, -2)).astype(np.float32)


# --- LOCAL DIVISIVE NORMALIZATION ---
# choose spatial sigma ~ (cell_diameter_px / 2) as a starting point

sigma_px = 3
# compute local RMS of the gradient magnitude (spatial only)
# broadcast sigma across (T,C,Y,X): no smoothing in T,C; only Y,X
sig = (0.0, 0.0, sigma_px, sigma_px)
local_rms = np.sqrt(gaussian_filter(Ft**2, sigma=sig, mode="nearest") + 1e-8)

Ft_norm = Ft / (local_rms + 1e-10)   # magnitude becomes contrast-invariant at cell scale

# --- SIGMOID in normalized space (fixed, data-stable) ---
low = np.percentile(Ft_norm, 50)
high = np.percentile(Ft_norm, 99)
mid = 0.5 * (low + high)
k = 9.21 / (high - low)              # ~0.01 at low, ~0.99 at high
sigmoid_map = 1.0 / (1.0 + np.exp(-k * (Ft_norm - mid)))
print(low, high, mid, k)
#%%
vmin = np.percentile(Ft, 0.1)
vmax = np.percentile(Ft, 99)    
pl.imshow(Ft[-1,0,:,:], cmap="gray", vmin=vmin, vmax=vmax)
pl.colorbar()
pl.figure()
vmin = np.percentile(Ft_norm, 1)
vmax = np.percentile(Ft_norm, 99)    
pl.imshow(Ft_norm[-1,0,:,:], cmap="gray", vmin=vmin, vmax=vmax)
pl.colorbar()
#%%
pl.figure()
vmin = np.percentile(sigmoid_map, 0)
vmax = np.percentile(sigmoid_map, 99)    
pl.imshow(sigmoid_map[0,0,:,:], cmap="gray")
# , vmin=vmin, vmax=vma
# x)
pl.colorbar()

#%%
for i in range(10):
    tmp = local_rms[i,0,:,:]
    pl.figure()
    vmin = np.percentile(tmp, 2)
    vmax = np.percentile(tmp, 90)    
    pl.imshow(tmp, cmap="gray")
    # , vmin=vmin, vmax=vmax)
    pl.colorbar()
#%%
with tf.TiffWriter(
    os.path.join(
        tif_dir, f"calcium_filtered_Ft_norm_sigmoid.ome.tif"
    ),
    imagej=True,
) as tif:
    tif.write(
        sigmoid_map[:,:,None].astype(np.float32),
        metadata=metadata,
        resolution=(1.0 / spacing_x, 1.0 / spacing_y),
    )

with tf.TiffWriter(
    os.path.join(
        tif_dir, f"calcium_filtered_Ft.ome.tif"
    ),
    imagej=True,
) as tif:
    tif.write(
        Ft[:,:,None].astype(np.float32),
        metadata=metadata,
        resolution=(1.0 / spacing_x, 1.0 / spacing_y),
    )

with tf.TiffWriter(
    os.path.join(
        tif_dir, f"calcium_filtered_Ft_norm.ome.tif"
    ),
    imagej=True,
) as tif:
    tif.write(
        Ft_norm[:,:,None].astype(np.float32),
        metadata=metadata,
        resolution=(1.0 / spacing_x, 1.0 / spacing_y),
    )

with tf.TiffWriter(
    os.path.join(
        tif_dir, f"calcium_filtered_local_rms.ome.tif"
    ),
    imagej=True,
) as tif:
    tif.write(
        local_rms[:,:,None].astype(np.float32),
        metadata=metadata,
        resolution=(1.0 / spacing_x, 1.0 / spacing_y),
    )

#%%

from skimage.filters import gaussian, sobel
from scipy.ndimage import gaussian_filter
trange = np.arange(0, 40000, 30000)
tmp_u16 = dask_data_calcium[trange].compute()  # (T, C, Y, X), dtype=uint16
# scale data between 10000 and 100000
# Convert to float32 in [0,1] no matter the input dtype
tmp = tmp_u16.astype(np.float32)
background = 400
tmp = tmp-background
tmp[tmp<0] = 0
tmp = tmp

from wholistic_registration.utils import preprocess
reload(preprocess)
sigma_xy = 6
r = sigma_xy*2
eps = 1e-6
edges = np.array([preprocess.michelson_edge_map(frame[0,:,:], sigma_xy=sigma_xy, r=r, eps=eps)[None] for frame in tmp])

fig, axs = pl.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(edges[0,0], cmap="gray")
axs[0].axis("off")
axs[1].imshow(edges[-1,0], cmap="gray")
axs[1].axis("off")
pl.tight_layout()
#%%
fig, axs = pl.subplots(1, 1, figsize=(10, 5))
axs = [axs]
axs[0].imshow(edges[0,0], cmap="gray")
axs[0].axis("off")
# axs[1].imshow(edges[-1,0], cmap="gray")
# axs[1].axis("off")
# pl.tight_layout()
#%%
fig, axs = pl.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(tmp[0,0], cmap="gray")
axs[0].axis("off")
axs[1].imshow(tmp[-1,0], cmap="gray")
axs[1].axis("off")
pl.tight_layout()
# pl.savefig(os.path.join(tif_dir, f"calcium_fmichelson_edge_map.pdf"))
#%%
pl.figure()
pl.imshow(edges[0,0]-edges[-1,0], cmap="gray")
pl.colorbar()
pl.clim([-1,1])
pl.axis("off")
#%%

with tf.TiffWriter(
    os.path.join(
        tif_dir, f"calcium_fmichelson_edge_map.ome.tif"
    ),
    imagej=True,
) as tif:
    tif.write(
        edges[:,:,None].astype(np.float32),
        metadata=metadata,
        resolution=(1.0 / spacing_x, 1.0 / spacing_y),
    )

#%%

def f(a, b):
    return np.abs((a-b)/(a+b+1e-2))

a = 1
b = np.linspace(-1.1, 10, 1000)
pl.plot(b, f(b, a))





#%%






#%%
        ####### SAVE MOTION RESULTS AND DATA #######
        num_layers = len(list(error_log.keys()))
        for ind, key in enumerate(list(error_log.keys())):  # iterate over layers
            ncorrections = len(
                error_log[key]["motion_current"]
            )  # number of iterations in this layer
            motions = np.array(
                [mc for mc in error_log[key]["motion_current"]]
            )  # motion field for this layer
            if ind == num_layers - 1:
                dat_calcium_corrected = [
                    calFlow3d_Wei_v1.correctMotion(
                        dask_data_calcium[t].compute().transpose(2, 1, 0),
                        motion_current,
                    )
                    for motion_current in motions
                ]
                dat_calcium_corrected = np.array(dat_calcium_corrected).transpose(
                    0, 3, 2, 1
                )

            if hasattr(error_log[key]["data_mov_corrected"][0], "get"):
                dat_corrected = np.array(
                    [mc.get() for mc in error_log[key]["data_mov_corrected"]]
                ).transpose(0, 3, 2, 1)
                data_mov = np.array(error_log[key]["data_mov"].get())[
                    None, :
                ].transpose(0, 3, 2, 1)
                data_ref = np.array(error_log[key]["data_ref"].get())[
                    None, :
                ].transpose(0, 3, 2, 1)
            else:
                dat_corrected = np.array(
                    [mc for mc in error_log[key]["data_mov_corrected"]]
                ).transpose(0, 3, 2, 1)
                data_mov = np.array(error_log[key]["data_mov"])[None, :].transpose(
                    0, 3, 2, 1
                )
                data_ref = np.array(error_log[key]["data_ref"])[None, :].transpose(
                    0, 3, 2, 1
                )
            if ind == num_layers - 1:
                dat_agglomerated = np.concatenate(
                    [data_mov, data_ref, dat_corrected, dat_calcium_corrected], axis=0
                )
            else:
                dat_agglomerated = np.concatenate(
                    [data_mov, data_ref, dat_corrected], axis=0
                )
            dat_agglomerated = dat_agglomerated[:, :, None].astype("float32")

            n_agglomerated = dat_agglomerated.shape[0]

            data_refs = np.repeat(data_ref, n_agglomerated, axis=0)
            data_refs = data_refs[:, :, None].astype("float32")

            dat_agglomerated_refs = np.concatenate(
                [dat_agglomerated, data_refs], axis=2
            )  # shape: (n_agglomerated, Z, C, Y, X)
            metadata = {}
            spacing_x = 1.0 / axesCalibration[0]
            spacing_y = 1.0 / axesCalibration[1]
            with tf.TiffWriter(
                os.path.join(
                    tif_dir, f"registered_data_iterations_timepoint_{t}_{key}.ome.tif"
                ),
                imagej=True,
            ) as tif:
                tif.write(
                    dat_agglomerated_refs,
                    metadata=metadata,
                    resolution=(1.0 / spacing_x, 1.0 / spacing_y),
                )

            motions = motions.transpose(0, 4, 3, 2, 1).astype("float32")
            with tf.TiffWriter(
                os.path.join(
                    tif_dir, f"registered_motion_iterations_timepoint_{t}_{key}.ome.tif"
                ),
                imagej=True,
            ) as tif:
                tif.write(
                    motions,
                    metadata=metadata,
                    resolution=(1.0 / spacing_x, 1.0 / spacing_y),
                )
print(f"Time taken: {(time() - t0)/60:.2f} minutes")
# %%
# check if the backend is Tkagg
import matplotlib
import matplotlib.pyplot as pl

print(matplotlib.get_backend())
if matplotlib.get_backend() != "TkAgg":
    print("TkAgg is not the backend")
    pl.figure()
    offset = 0
    for ind, key in enumerate(list(error_log.keys())):
        xax = np.arange(len(error_log[key]["currentError"])) + offset
        pl.plot(xax, error_log[key]["currentError"], "o-", label=key)
        offset += len(xax)
        pl.xlabel("Iteration")
        pl.ylabel("Error")
    pl.legend()
    pl.savefig(os.path.join(save_dir, "error_log.png"))
else:
    print("TkAgg is the backend")
# %%


# %%


# %%

# tRange=np.arange(0,T,frameJump)
# print(f"tRange is {tRange}, length is {len(tRange)}")

# with nd2.ND2File(nd2_file) as f:
#     metadata=f.metadata
#     channels=metadata.channels[0]
#     #get Zratio
#     axesCalibration=channels.volume.axesCalibration
#     dask_data_moving=f.to_dask()[:,1,sly,slx]
#     data_ref = dask_data_moving[tRange].compute()

# metadata = {}
# spacing_x = 1.0/axesCalibration[0]
# spacing_y = 1.0/axesCalibration[1]

# with tf.TiffWriter(os.path.join(tif_dir, f'reference_data.ome.tif'), imagej=True) as tif:
#     tif.write(data_ref[:,None,None], metadata=metadata, resolution=(1.0/spacing_x, 1.0/spacing_y))
