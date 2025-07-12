#%%

import wholistic_registration
from wholistic_registration.utils import preprocess, calFlow3d_Wei_v1
import numpy as np
import nd2
from glob import glob
import os
from wholistic_registration import utils
from importlib import reload
import tifffile as tf
from time import time
import cupy as cp

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


option={
    'layer': 3,
    'iter': 3000,
    'r': 5,
    'zRatio': 27.693,
    'motion': 0,
    'mask_ref': 0,
    'mask_mov': 0,
    'save_ite':5,
    'movRange': 10 # larger the less penalized
}

frameJump=30000
# frameJump=1
refLength=5
refJump =20/frameJump
initialLength=5
thresFactor=5
smFactor=50
maskRange=[5,500]
smoothPenalty_raw=1e-6
tol=1e-5

T=30001

slx = slice(350, 700)
sly = slice(350, 700)
slx = slice(None, None)
sly = slice(None, None)

# tRange=np.arange(0,T,frameJump)
# print(f"tRange is {tRange}, length is {len(tRange)}")

# with nd2.ND2File(nd2_file) as f:
#     metadata=f.metadata
#     channels=metadata.channels[0]
#     #get Zratio
#     axesCalibration=channels.volume.axesCalibration
#     dask_data=f.to_dask()[:,1,sly,slx]
#     data_ref = dask_data[tRange].compute()

# metadata = {}
# spacing_x = 1.0/axesCalibration[0]
# spacing_y = 1.0/axesCalibration[1]    

# with tf.TiffWriter(os.path.join(tif_dir, f'reference_data.ome.tif'), imagej=True) as tif:
#     tif.write(data_ref[:,None,None], metadata=metadata, resolution=(1.0/spacing_x, 1.0/spacing_y))


tRange=np.arange(frameJump,T,frameJump)
print(f"tRange is {tRange}, length is {len(tRange)}")
with nd2.ND2File(nd2_file) as f:
    metadata=f.metadata
    channels=metadata.channels[0]
    #get Zratio
    axesCalibration=channels.volume.axesCalibration
    zRatio=axesCalibration[2]/axesCalibration[0]
    print("Z ratio is", zRatio)
    [X,Y,Z]=channels.volume.voxelCount

    if slx.stop is not None and sly.stop is not None:
        X = slx.stop - slx.start
        Y = sly.stop - sly.start
    
    print("Data size is",[X,Y,Z])

    #get total frames
    frames=metadata.contents.frameCount
    print("Total frames is",frames)


    #initial the var
    dat_channel2_raw = np.zeros([X,Y,Z,len(tRange)],dtype=np.int16)
    dat_channel2 = np.zeros([X,Y,Z,len(tRange)],dtype=np.int16)
    dat_channel1 = np.zeros([X,Y,Z,len(tRange)],dtype=np.int16)
    dat_refs = np.zeros([X,Y,Z, len(tRange)],dtype=np.int16)

    motion_history=np.zeros([X,Y,Z,3,initialLength],dtype=np.float32)
    option['motion']=np.zeros([X,Y,Z,3])

    if Z>1:
        #load all the data(virtual)
        dask_data=f.to_dask()[:,:,1,sly,slx]
        Ca_data=f.to_dask()[:,:,0,sly,slx]
    else:
        dask_data=f.to_dask()[:,1,sly,slx][:,None]
        Ca_data=f.to_dask()[:,0,slx,sly][:,None]

    # dat_ref=np.roll(dask_data[0].compute().transpose(2,1,0),10,axis=1).copy()
    dat_ref = dask_data[0].compute().transpose(2,1,0)
    print(f"dat_ref.shape is {dat_ref.shape}")
    # option['mask_ref']=mask.getMask(dat_ref,thresFactor)
    # option['mask_ref']=mask.bwareafilt3_wei(option['mask_ref'],maskRange)
    option['mask_ref']=np.full(dat_ref.shape,False,dtype=bool)
    option['mask_mov']=np.full(dat_ref.shape,False,dtype=bool)
    Pnltfactor=preprocess.getSmPnltNormFctr(dat_ref,option)
    smoothPenalty=Pnltfactor*smoothPenalty_raw

    #start to registration
    for tCnt in range(len(tRange)):
        t=tRange[tCnt]
        print(f"reading data... \n tCnt is {tCnt}, time point {t}")
        dat_mov=dask_data[t].compute().transpose(2,1,0)
        option['mask_ref']=np.full(dat_ref.shape,False,dtype=bool)
        option['mask_mov']=np.full(dat_ref.shape,False,dtype=bool)
        # option['mask_mov']=mask.getMask(dat_ref,thresFactor)
        # option['mask_mov']=mask.bwareafilt3_wei(option['mask_mov'],maskRange)
        # print("generate reference...")
        # if (tCnt - 1) % refJump == 0:
        #     if tCnt > refJump:
        #         refPossible = np.int32(np.min([tCnt//refJump, refLength]))
        #         # print(f"refPossible is {refPossible}")
        #         ref_range = np.arange(tCnt - refPossible * refJump, tCnt, refJump)
        #         # print(f"ref_range is {ref_range}")
        #         # Compute median along time axis (axis=3 for 4D array)
        #         dat_ref = np.median(dat_channel2[:, :, :, ref_range], axis=3).astype(np.float32)
            
            # Generate and filter mask
            # option['mask_ref'] = mask.getMask(dat_ref, thresFactor)
            # option['mask_ref'] = mask.bwareafilt3_wei(option['mask_ref'], maskRange)
            
            # Update penalty factor
            # pnlt_factor = preprocess.getSmPnltNormFctr(dat_ref, option)
            # smoothPenalty=Pnltfactor*smoothPenalty_raw
        verbose = True
        motion_current, currentError, coords_new, error_log = utils.calFlow3d_Wei_v1.getMotion(dat_mov,dat_ref,smoothPenalty,option,tol=tol,verbose=verbose)

        
        for ind, key in enumerate(list(error_log.keys())):
            ncorrections = len(error_log[key]['motion_current'])
            motions = np.array([mc for mc in error_log[key]['motion_current']])
            motions = motions.transpose(0,4,3,2,1).astype('float32')
            if hasattr(error_log[key]['data_trans'][0], "get"):
                corr_dat = np.array([mc.get() for mc in error_log[key]['data_trans']])
                data_mov = np.array(error_log[key]['data_mov'].get())[None,:]
                data_ref = np.array(error_log[key]['data_ref'].get())[None,:]                
            else:
                corr_dat = np.array([mc for mc in error_log[key]['data_trans']])
                data_mov = np.array(error_log[key]['data_mov'])[None,:]
                data_ref = np.array(error_log[key]['data_ref'])[None,:]
            

            corr_dat = np.concatenate([data_mov,data_ref, corr_dat],axis=0)
            data_refs = np.repeat(data_ref,ncorrections+2,axis=0)

            corr_dat = corr_dat.transpose(0,3,2,1)[:,:,None].astype('float32')
            data_refs = data_refs.transpose(0,3,2,1)[:,:,None].astype('float32')
            corr_dat_refs = np.concatenate([corr_dat,data_refs],axis=2)
            metadata = {}
            spacing_x = 1.0/axesCalibration[0]
            spacing_y = 1.0/axesCalibration[1]
            # with tf.TiffWriter(os.path.join(tif_dir, f'registered_data_iterations_{t}_{key}.ome.tif'), imagej=True) as tif:
            #     tif.write(corr_dat_refs, metadata=metadata, resolution=(1.0/spacing_x, 1.0/spacing_y))
            # with tf.TiffWriter(os.path.join(tif_dir, f'registered_motion_iterations_{t}_{key}.ome.tif'), imagej=True) as tif:
            #     tif.write(motions, metadata=metadata, resolution=(1.0/spacing_x, 1.0/spacing_y))
print(f"Time taken: {(time() - t0)/60:.2f} minutes")
#%%
# check if the backend is Tkagg
import matplotlib
import matplotlib.pyplot as pl
print(matplotlib.get_backend())
if matplotlib.get_backend() != 'TkAgg':
    print("TkAgg is not the backend")
    pl.figure()
    offset = 0
    for ind, key in enumerate(list(error_log.keys())):
        xax = np.arange(len(error_log[key]['currentError'])) + offset
        pl.plot(xax,error_log[key]['currentError'],'o-',label=key)
        offset += len(xax)
        pl.xlabel('Iteration')
        pl.ylabel('Error')
    pl.legend()
    pl.savefig(os.path.join(save_dir, 'error_log.png'))
else:
    print("TkAgg is the backend")
#%%




#%%


#%%