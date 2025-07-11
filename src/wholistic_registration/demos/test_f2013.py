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
    'layer': 0,
    'iter': 10,
    'r': 2,
    'zRatio': 27.693,
    'motion': 0,
    'mask_ref': 0,
    'mask_mov': 0,
    'save_ite':1,
    'movRange': 100 # larger the less penalized
}

frameJump=20000
refLength=1
refJump =40/frameJump
initialLength=5
thresFactor=5
smFactor=50
maskRange=[5,500]
smoothPenalty_raw=0.01
smoothPenalty_raw=100 # larger the less penalized

T=19000
# idx = np.arange(0,T,frameJump)
# print(f"idx is {idx}")
# with nd2.ND2File(nd2_file) as f:
#     dask_data=f.to_dask()[idx].compute()
#     print(f"dask_data.shape is {dask_data.shape}")


# tf.imwrite(tif_dir + f"demo_data.tif",dask_data[:,None].astype('float32'), imagej=True)

#%%
reload(utils)
reload(preprocess)
reload(calFlow3d_Wei_v1)
reload(wholistic_registration)

slx = slice(500, 750)
sly = slice(500, 750)

with nd2.ND2File(nd2_file) as f:
    metadata=f.metadata
    channels=metadata.channels[0]
    
    #get Zratio
    axesCalibration=channels.volume.axesCalibration
    zRatio=axesCalibration[2]/axesCalibration[0]
    print("Z ratio is", zRatio)
    [X,Y,Z]=channels.volume.voxelCount

    if slx is not None and sly is not None:
        X = slx.stop - slx.start
        Y = sly.stop - sly.start
    
    print("Data size is",[X,Y,Z])

    #get total frames
    frames=metadata.contents.frameCount
    print("Total frames is",frames)

    tRange=range(0,T,frameJump)
    print(f"tRange is {tRange}, length is {len(tRange)}")

    #initial the var
    dat_channel2_raw = np.zeros([X,Y,Z,len(tRange)],dtype=np.int16)
    dat_channel2 = np.zeros([X,Y,Z,len(tRange)],dtype=np.int16)
    dat_channel1 = np.zeros([X,Y,Z,len(tRange)],dtype=np.int16)
    dat_refs = np.zeros([X,Y,Z, len(tRange)],dtype=np.int16)

    motion_history=np.zeros([X,Y,Z,3,initialLength],dtype=np.float32)
    # option['motion']=np.zeros([X,Y,Z,len(tRange)])
    option['motion']=np.zeros([X,Y,Z,3])

    if Z>1:
        #load all the data(virtual)
        dask_data=f.to_dask()[:,:,1,sly,slx]
        Ca_data=f.to_dask()[:,:,0,sly,slx]
    else:
        dask_data=f.to_dask()[:,1,sly,slx][:,None]
        Ca_data=f.to_dask()[:,0,slx,sly][:,None]


    dat_ref=np.roll(dask_data[0].compute().transpose(2,1,0),10,axis=1)
    print(f"dat_ref.shape is {dat_ref.shape}")
    # option['mask_ref']=mask.getMask(dat_ref,thresFactor)
    # option['mask_ref']=mask.bwareafilt3_wei(option['mask_ref'],maskRange)
    option['mask_ref']=np.full(dat_ref.shape,False,dtype=bool)
    option['mask_mov']=np.full(dat_ref.shape,False,dtype=bool)
    Pnltfactor=preprocess.getSmPnltNormFctr(dat_ref,option)
    smoothPenalty=Pnltfactor*smoothPenalty_raw

    #start to registration
    for tCnt in range(len(tRange)):
    # for tCnt in np.arange(1,2):
        t=tRange[tCnt]
        print(f"reading data... \n tCnt is {tCnt}, time point {t}")
        dat_mov=dask_data[t].compute().transpose(2,1,0)
        option['mask_ref']=np.full(dat_ref.shape,False,dtype=bool)
        option['mask_mov']=np.full(dat_ref.shape,False,dtype=bool)
        # option['mask_mov']=mask.getMask(dat_ref,thresFactor)
        # option['mask_mov']=mask.bwareafilt3_wei(option['mask_mov'],maskRange)
        print("generate reference...")
        # if (tCnt - 1) % refJump == 0:
        #     if tCnt > refLength * refJump:
        #         ref_range = np.arange(tCnt - refLength * refJump, tCnt, refJump)
        #         print(f"ref_range is {ref_range}")
        #         # Compute median along time axis (axis=3 for 4D array)
        #         dat_ref = np.median(dat_channel2[:, :, :, ref_range], axis=3).astype(np.float32)
            
            # Generate and filter mask
            # option['mask_ref'] = mask.getMask(dat_ref, thresFactor)
            # option['mask_ref'] = mask.bwareafilt3_wei(option['mask_ref'], maskRange)

            
            # Update penalty factor
            # pnlt_factor = preprocess.getSmPnltNormFctr(dat_ref, option)
            # smoothPenalty=Pnltfactor*smoothPenalty_raw
    
        # print("calculating the motion")
        # print(f"dat_ref.shape is {dat_ref.shape}")
        # print(f"dat_mov.shape is {dat_mov.shape}")
        # print(f"smoothPenalty is {smoothPenalty}")
        # print(f"option is {option}")
        # print(f"shape of option['motion'] is {option['motion'].shape}")
        motion_current, currentError, coords_new, error_log = utils.calFlow3d_Wei_v1.getMotion(dat_mov,dat_ref,smoothPenalty,option)

        for ind, key in enumerate(list(error_log.keys())[-1:]):
            ncorrections = len(error_log[key]['motion_current'])
            motions = np.array([mc for mc in error_log[key]['motion_current']])
            corr_dat = [utils.calFlow3d_Wei_v1.correctMotion(dat_mov,mc) for mc in error_log[key]['motion_current']]
            corr_dat = np.array(corr_dat)
        corr_dat = np.concatenate([dat_ref[None],dat_mov[None],corr_dat],axis=0)
        corr_dat = corr_dat.transpose(0,3,2,1)[:,None].astype('float32')
        motions = motions.transpose(0,4,3,2,1).astype('float32')
        tf.imwrite(tif_dir + f'registered_motion_iterations_{t}.tif', motions, imagej=True)
        tf.imwrite(tif_dir + f'registered_data_iterations_{t}.tif', corr_dat, imagej=True)

        dat_channel1[:,:,:,tCnt]=utils.calFlow3d_Wei_v1.correctMotion(Ca_data[t].compute().transpose(2,1,0),motion_current)
        dat_channel2[:,:,:,tCnt]=utils.calFlow3d_Wei_v1.correctMotion(dat_mov,motion_current)
        dat_channel2_raw[:,:,:,tCnt]=dat_mov
        dat_refs[:,:,:,tCnt]=dat_ref



                

    dat_channel = np.concatenate([dat_channel1[None],dat_channel2[None]],axis=0).transpose(4,3,0,2,1)
    dat_refs_ = np.concatenate([dat_refs[None],dat_channel2_raw[None]],axis=0).transpose(4,3,0,2,1)
    all_data = np.concatenate([dat_channel2_raw[None], dat_channel2[None],dat_refs[None]],axis=0).transpose(4,3,0,2,1)


    # tt = dat_channel.shape[0]
    # ref_channel = np.repeat(dat_ref[None],tt,axis=0).transpose(0, 3, 2, 1)[:,:,None]
    # all_channel = np.concatenate([dat_channel, ref_channel],axis=2)
    # tf.imwrite(tif_dir + f"dat_channel_reg_{t}.tif",dat_channel.astype('float32'), imagej=True)
    # tf.imwrite(tif_dir + f"all_channel_reg_{t}.tif",all_data.astype('float32'), imagej=True)

#%%
import matplotlib.pyplot as pl
pl.imshow(corr_dat[0,0,0]-corr_dat[2,0,0])


#%%


#%%