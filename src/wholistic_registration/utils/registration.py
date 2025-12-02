"""

version : 0.3
file name : registration.py

Last Update Date : 2025/12/2

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
from . import visualization
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
def wbi_registration_2d(moving_membrane_image,moving_Ca_image,config_file,reference_image=None,motion_init=None,verbose=True,frame=None):

    '''Load the config file'''
    config=toml.load(config_file)


    '''Get frames'''
    nimg, Ly, Lx = moving_membrane_image.shape
    frames=range(0,nimg)


    '''Get reference image'''
    #if we pick reference image from moving_image
    refer=config["reference"]
    if refer["pick_reference_auto"]:
        membrane_ref_1plane,indsort=reference.pick_initial_reference(moving_membrane_image)
    else:
        membrane_ref_1plane=reference_image

    #If to use two channel data
    channels=config["channels"]
    if channels["dual_channel"] and  refer["pick_reference_auto"]:
        # get the corresponding planes in Ca channel
        Ca_data_reshape=np.reshape(moving_Ca_image, (len(frames), -1))
        Ca_average=np.mean(Ca_data_reshape[indsort,:],axis=0)
        Ca_ref_1plane=np.reshape(Ca_average,moving_membrane_image.shape[1:])
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
    motions =[]

    #inital the motion
    if motion_init is None:
        option['motion']=np.zeros([dat_ref.shape[0],dat_ref.shape[1],2,3])
    else:
        option['motion']=motion_init

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
        dat_mem=np.stack([mem_1plane] * 3, axis=2)

        if channels["dual_channel"]:
            Ca_1plane=moving_Ca_image[i,:,:]
            dat_ca=np.stack([Ca_1plane] * 3, axis=2)
        
    
    
        if channels["dual_channel"]:
            dat_ca_tran=transform(dat_ca,channels["k"],channels["function"])
            #normalize to the mean and std of the reference
            if refer["pick_reference_auto"]:
                dat_ca_tran=prep.normalize_std(ref_mean,ref_std,dat_ca_tran)
            dat_mov=dat_mem+dat_ca_tran
        else:
            dat_mov=dat_mem

        #get mask_mov
        option['mask_mov'] = mask.getMask(dat_mov, thresFactor)
        option['mask_mov'] = mask.bwareafilt3_wei(option['mask_mov'], maskRange)

        #get motion
        motion_current, _ , new_coords,error_logs = calFlow3d_Wei_v1.getMotion(dat_mov, dat_ref, smoothPenalty, option)
        if channels["dual_channel"]:
            corrected_ca = calFlow3d_Wei_v1.correctMotion(dat_ca, motion_current)        
        corrected_mem = calFlow3d_Wei_v1.correctMotion(dat_mem, motion_current)
        corrected_mov = calFlow3d_Wei_v1.correctMotion(dat_mov, motion_current)
        initial_error=np.mean((dat_mov-dat_ref)**2)
        eventual_error=np.mean((corrected_mov-dat_ref)**2)

        #print error
        if verbose==True:
            if frame is not None:
                print(f"        Frame: {i+1}\tInitial Error is:{initial_error}\tEventual Error: {eventual_error}")
            else:
                print(f"        Frame: {frame}\tInitial Error is:{initial_error}\tEventual Error: {eventual_error}")
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
        if channels["dual_channel"]:
            Ca_channel.append(corrected_ca[:,:,0])
        motions.append(motion_current[:,:,0,:])
        
    return cp.asarray(mem_channel).get(),cp.asarray(Ca_channel).get(),dat_ref,errors,cp.asarray(motions).get()
def wbi_registration_3d(moving_membrane_image,moving_Ca_image,config_file,reference_image=None,motion_init=None,verbose=True,frame=None):
    '''Load the config file'''
    config=toml.load(config_file)


    '''Get frames'''
    nimg, Lz, Ly, Lx = moving_membrane_image.shape
    frames=range(0,nimg)


    '''Get reference image'''
    #if we pick reference image from moving_image
    refer=config["reference"]
    if refer["pick_reference_auto"]:
        membrane_ref,indsort=reference.pick_initial_reference(moving_membrane_image,max_corr_frames=refer['chunk_size'])
    else:
        membrane_ref=reference_image

    #If to use two channel data
    channels=config["channels"]
    if channels["dual_channel"] and  refer["pick_reference_auto"]:
        # get the corresponding planes in Ca channel
        Ca_data_reshape=np.reshape(moving_Ca_image, (len(frames), -1))
        Ca_average=np.mean(Ca_data_reshape[indsort,:],axis=0)
        Ca_ref=np.reshape(Ca_average,moving_membrane_image.shape[1:])
        Ca_ref_transform=transform(Ca_ref,channels["k"],channels["function"])

        # record the mean and std of Ca channel(we will use these two values to normalize the moving Ca image)
        ref_mean=np.mean(Ca_ref_transform)
        ref_std=np.std(Ca_ref_transform)

        # get the reference data(1 plane)
        dat_ref=membrane_ref+Ca_ref_transform

    else:
        dat_ref=membrane_ref

    # visualization.visualize_2d_image(dat_ref_1plane,title="Reference Image")
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
    motions =[]

    #inital the motion
    if motion_init is None:
        option['motion']=np.zeros([dat_ref.shape[0],dat_ref.shape[1],2,3])
    else:
        option['motion']=motion_init
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

    for i in frames:
        #get dat_mov
        dat_mem=moving_membrane_image[i,...]

        if channels["dual_channel"]:
            dat_ca=moving_Ca_image[i,...]
    
        if channels["dual_channel"]:
            dat_ca_tran=transform(dat_ca,channels["k"],channels["function"])
            #normalize to the mean and std of the reference
            if refer["pick_reference_auto"]:
                dat_ca_tran=prep.normalize_std(ref_mean,ref_std,dat_ca_tran)
            dat_mov=dat_mem+dat_ca_tran
        else:
            dat_mov=dat_mem

        #get mask_mov
        option['mask_mov'] = mask.getMask(dat_mov, thresFactor)
        option['mask_mov'] = mask.bwareafilt3_wei(option['mask_mov'], maskRange)

        #get motion
        motion_current, _ , new_coords,error_logs = calFlow3d_Wei_v1.getMotion(dat_mov, dat_ref, smoothPenalty, option)
        if channels["dual_channel"]:
            corrected_ca = calFlow3d_Wei_v1.correctMotion(dat_ca, motion_current)
        corrected_mem = calFlow3d_Wei_v1.correctMotion(dat_mem, motion_current)
        corrected_mov = calFlow3d_Wei_v1.correctMotion(dat_mov, motion_current)
        initial_error=np.mean((dat_mov-dat_ref)**2)
        eventual_error=np.mean((corrected_mov-dat_ref)**2)
        #print error
        if verbose==True:
            if frame is None:
                print(f"        Frame: {i+1}\tInitial Error is:{initial_error}\tEventual Error: {eventual_error}")
            else:
                print(f"        Frame: {frame}\tInitial Error is:{initial_error}\tEventual Error: {eventual_error}")
                frame=frame+1
        error=dict[
            "initial_error":initial_error,
            "eventual_error":eventual_error
        ]

        #store the result
        diff_check = np.abs(corrected_mov - dat_ref)
        concat = np.concatenate([
            dat_mem, 
            corrected_mov.astype(np.float32),
            dat_ref.astype(np.float32),
            diff_check.astype(np.float32)
            ], axis=1)
        
        errors.append(error)
        mem_channel.append(corrected_mem)
        if channels["dual_channel"]:
            Ca_channel.append(corrected_ca)
        motions.append(motion_current)

    return cp.asarray(mem_channel).get(),cp.asarray(Ca_channel).get(),dat_ref,errors,cp.asarray(motions).get()

def register_one_frame(configFilePath, mem_img, ca_img, ref_pool,idx,verbose=True):
    """Register one mem+Ca frame to the reference generated from the pool"""
    config=toml.load(configFilePath)
    ref_img = reference.compute_reference_from_block(
        np.array([m for m in ref_pool["mem"]]),
        np.array([c for c in ref_pool["ca"]]),
        config
    )
    if config['MetaData']['Dim']==3:
        mem_reg, ca_reg, _, _, motion_reg = wbi_registration_3d(
            np.expand_dims(mem_img, axis=0),
            np.expand_dims(ca_img, axis=0),
            configFilePath,
            ref_img,
            verbose=verbose,
            frame=idx
        )
    else:
        mem_reg, ca_reg, _, _, motion_reg = wbi_registration_2d(
            np.expand_dims(mem_img, axis=0),
            np.expand_dims(ca_img, axis=0),
            configFilePath,
            ref_img,
            verbose=verbose,
            frame=idx
        )
    return mem_reg[0], ca_reg[0], ref_img, motion_reg[0]

