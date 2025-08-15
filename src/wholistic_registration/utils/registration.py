"""

version : 0.1
file name : registration.py

Last Update Date : 2025/8/11

Overview:
    The whole pipeline of registration
    
Functions:
    - wbi_registration
"""
import numpy as np
from . import cp
from . import preprocess as prep
from .imresize import imresize
from . import reference
from . import mask
from . import calFlow3d_Wei_v1
import toml

from . import option
#TODO
#need to write a 3D version
#2025/8/11  14:22

def transform(image,k=1,method="raw"):
    if method=="raw":
        return k*image
    elif method=="sqrt":
        return k*np.sqrt(image)
    elif method=="log2":
        return k*np.log2(1+image)
    elif method=="log10":
        return k*np.log10(1+image)
    else:
        raise ValueError(f"Unknown method to process the image:{method}")
def wbi_registration_2d(moving_membrane_image,moving_Ca_image,config_file,reference_image=None):

    '''Load the config file'''
    config=toml.load(config_file)


    '''Get frames'''
    nimg, Ly, Lx = moving_membrane_image.shape
    frames=range(0,nimg)


    '''Get reference image'''
    #if we pick reference image from moving_image
    pick_reference_auto=config["reference"]
    if pick_reference_auto:
        membrane_ref_1plane,indsort=reference.pick_initial_reference(moving_membrane_image)
    else:
        membrane_ref_1plane=reference_image

    #If to use two channel data
    channels=config["channels"]
    if channels["dual_channel"]:
        # get the corresponding planes in Ca channel
        Ca_data_reshape=np.reshape(moving_Ca_image, (len(frames), -1))
        Ca_average=np.mean(Ca_data_reshape[indsort,:],axis=0)
        Ca_ref_1plane=np.reshape(Ca_average,(Ly,Lx))
        Ca_ref_1plane_transform=transform(Ca_ref_1plane,channels["k"],channels["function"])

        # record the mean and std of Ca channel(we will use these two values to normalize the moving Ca image)
        ref_mean=np.mean(Ca_ref_1plane_transform)
        ref_std=np.std(Ca_ref_1plane_transform)

        # get the reference data(1 plane)
        dat_ref_1plane=membrane_ref_1plane+Ca_ref_1plane_transform

    else:
        dat_ref_1plane=membrane_ref_1plane

    # stack the refence to get a fake 3D image
    dat_ref=np.stack([dat_ref_1plane] * 3, axis=2)


    '''Get mask'''
    maskConfig=config["mask"]
    maskRange=maskConfig["maskRange"]
    thresFactor=maskConfig["thresFactor"]

    option['mask_ref']=mask.getMask(dat_ref,thresFactor)
    option['mask_ref']=mask.bwareafilt3_wei(option['mask_ref'],maskRange)


    '''Do registration'''
    #initial list to store the result
    mem_channel = []
    Ca_channel = []
    errors = []
    concat_images_with_checks = []

    #inital the motion
    option['motion']=np.zeros([dat_ref.shape[0],dat_ref.shape[1],2,3])

    #initial the pyramid parameters
    pyramid=config["pyramid"]
    option['r']=pyramid["r"]
    option['layer']=pyramid["layer"]
    option['iter']=pyramid["iter"]
    option['movRange']=5.
    smoothPenalty_raw=pyramid["smoothPenalty"]

    #get smoothPenalty
    Pnltfactor = prep.getSmPnltNormFctr(dat_ref, option)
    smoothPenalty=Pnltfactor*smoothPenalty_raw

    #do registration
    for i in frames:
        #get dat_mov
        mem_1plane=moving_membrane_image[i,:,:]
        Ca_1plane=moving_Ca_image[i,:,:]
        dat_mem=np.stack([mem_1plane] * 3, axis=2)
        dat_ca=np.stack([Ca_1plane] * 3, axis=2)

        if channels["dual_channel"]:
            dat_ca_tran=transform(dat_ca,channels["k"],channels["function"])
            #normalize to the mean and std of the reference
            dat_ca_tran=prep.normalize_std(ref_mean,ref_std,dat_ca_tran)
            dat_mov=dat_mem+dat_ca_tran
        else:
            dat_mov=dat_mem

        #get mask_mov
        option['mask_mov'] = mask.getMask(dat_mov, thresFactor)
        option['mask_mov'] = mask.bwareafilt3_wei(option['mask_mov'], maskRange)

        #get motion
        motion_current, _ , new_coords,error_logs = calFlow3d_Wei_v1.getMotion(dat_mov, dat_ref, smoothPenalty, option)
        corrected_ca = calFlow3d_Wei_v1.correctMotion(dat_ca, motion_current)
        corrected_mem = calFlow3d_Wei_v1.correctMotion(dat_mem, motion_current)
        corrected_mov = calFlow3d_Wei_v1.correctMotion(dat_mov, motion_current)
        initial_error=np.mean((dat_mov-dat_ref)**2)
        eventual_error=np.mean((corrected_mov-dat_ref)**2)

        #print error
        print(f"Frame: {i+1}\tInitial Error is:{initial_error}\tEventual Error: {eventual_error}")
        error=dict[
            "initial_error":initial_error,
            "eventual_error":eventual_error
        ]

        #store the result
        diff_check = np.abs(corrected_mov[:, :, 0] - dat_ref[:, :, 0])
        concat = np.concatenate([
            mem_1plane, 
            corrected_mov[:, :, 0].astype(np.float32),
            dat_ref[:, :, 0].astype(np.float32),
            diff_check.astype(np.float32)
            ], axis=1)
        
        errors.append(error)
        mem_channel.append(corrected_mem[:,:,0])
        Ca_channel.append(corrected_ca[:,:,0])
        concat_images_with_checks.append(concat)
    
    return mem_channel,Ca_channel,concat_images_with_checks,errors
