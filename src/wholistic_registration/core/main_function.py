from ..utils import IO, reference, registration
import zarr
import toml
import numpy as np
from skimage.transform import resize
from numcodecs import Blosc
from ..utils.reliableAnalysis import get_reliable_mask, get_spatial_mask, get_temporal_and_accumula_mask
import  os

def DefineParams(
                configFile='./configs/config.toml',
                inputFile=None,
                outputFile=None,
                downsampleXY=1,
                downsampleT=1,
                downsampleZ=1,
                dual_channel=False,
                function=None,
                k=0.,
                chunk_size=50,
                mid_chunk_size=100,
                preprocess=False,
                thresFactor=5,
                maskRange=[5,4000],
                layer=3,
                r=5,
                iter=10,
                smoothPenalty=0.08,
                save_ref=True,
                save_motion=False,
                verbose=True
):
    '''
    [downsample]
    Params:
    -downsampleXY
        The coefficient of the downsampling on X or Y dimension.
        If downsampleX = 4 & origin shape of 1 slice is (X,Y), then after downsampleing the shape should be (X/4,Y/4)
    -downsampleT
        Sometimes we need to process the time downsample data first to overcome the large displacement when the time rate is small
        After processing the time downsample data, we can registrate the left data to them which has been registrated.
        If downsampleT = 10, we will pick frames like 1,11,21,31,etc first, and then we will registrate frame 2~5 to frame 1 ,frame 6~15 to frame 11 and so on.
    -downsample Z
        An list of the z silces we will use.
        If downsampleZ=[4,5,6,7], we will use the 5th, 6th, 7th, 8th silces of the whole data.
        If downsampleZ=None, we will use all of the z slices 

    [channels]
    Params:
    -dual_channel
        Whether use two channels to do registration
        If true,we will use the "membrane_channel+k*function(Ca_channel)" to do registration
    -function
        The method we process the Ca_channel.It can be "sqrt","log2","log10" or "raw"
        "sqrt": square root
        "log2": log2(1+x)
        "log10": log10(1+x)
        "raw": x
    -k
        The coefficient multiplied after the transformation of Ca channel data:
        the larger this value, the greater the proportion of the calcium channel, making it more susceptible to changes in the Ca channel.
    Example:
    1.
        dual_channel=true
        function="raw"
        k=0.3
    2.
        dual_channel=true
        function="log10"
        k=300

    [reference]
    Params:
    -pick_reference_auto
        Whether pick the reference image from moving image
        If true, it will pick the reference image from the moving video each several frames
        If false, you need to give a reference image
    -chunk_size
        The size of frames we will use to update the reference
    -mid_chunk_size
        The size of frames we will use to pick the initial reference from the middle block

    
    [mask]
    Params:
    -thresFactor
        The threshold value
        pixels greater than thresFactor times the standard deviation are regarded as outliers, and we will initially mask these points.
    -maskRange
        The pixel range of the mask region
        Only the pixels in the maskRange will be masked.
        Remark: This is the absolute value of the pixels, so it depends on the bit depth of the image's pixels. 
        For example, if yo image range is [0, 255], then your reasonable maskRange should be at least a subset of [0, 255]. 
        If you don't want to filter out any mask points, you can set this range to be very large.
    
    [pyramid]
    Params:
    -layer
        The layer of pyramid
        A larger layer means that the algorithm is better at capturing large-scale displacements
        and correspondingly, its ability to capture some small deformations will be slightly diminished.
    -r
        The radius of the patch
        The size of each patch is (2r+1)*(2r+1), and each patch has one control point.
        A smaller r means more control points and more easier to capture noise
    -iter
        The num of maximum iterations of each layer
    -smoothPenalty
        The coefficient of the smoothness penalty term
        A larger smoothPenalty means more smooth motion we will get and correspondingly the error of intensity will increase
    '''
    ## read the metadata
    print("Reading meta data")
    if inputFile is not None:
        meta=IO.readMeta_new(inputFile,Ifprint=verbose)
        nchannels=meta['nchannels']
        nframes=meta['nframes']
        data_shape=meta['data_shape']
        resolutionxyz=meta['resolutionxyz']
        spacing_x=meta['spacing_x']
        spacing_y=meta['spacing_y']
        spacing_z=meta['spacing_z']
        framerate=meta['fps']
        zRatio=meta['zRatio']

    print("Saving the config")
    ## load the default config file
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(current_dir), 'configs', 'config_default.toml')
    config=toml.load(config_path)

    #change the meta data 
    config['MetaData']['zRatio']=zRatio
    config['MetaData']['SIZE']=data_shape
    config['MetaData']['frames']=nframes
    config['MetaData']['Dim']=len(data_shape)
    config['MetaData']['voxelsize']=resolutionxyz
    config['MetaData']['frame_rate']=framerate

    #change the downsample config
    config['downsample']['downsampleXY']=downsampleXY
    config['downsample']['downsampleZ']=downsampleZ
    config['downsample']['downsampleT']=downsampleT
    
    #change the filepath
    config['file_path']['input_path']=inputFile
    config['file_path']['registrated_path']=outputFile

    #change the dual_channels config
    config['channels']['dual_channel']=dual_channel
    config['channels']['function']=function
    config['channels']['k']=k

    #change the reference config
    config['reference']['chunk_size']=chunk_size
    config['reference']['mid_chunk_size']=mid_chunk_size
    config['reference']['pick_reference_auto']=False
    #change the preprocess config
    config['preprocess']['normailize_to_0_255']=preprocess

    #change the mask config
    config['mask']['thresFactor']=thresFactor
    config['mask']['maskRange']=maskRange

    #change the pyramid config
    config['pyramid']['layer']=layer
    config['pyramid']['r']=r
    config['pyramid']['iter']=iter
    config['pyramid']['smoothPenalty']=smoothPenalty

    #change the pyramid config

    config['save_config']['save_ref']=save_ref
    config['save_config']['save_motion']=save_motion
    with open(configFile,'w') as f:
        toml.dump(config,f)

    if verbose == True:
        print("\nConfiguration summary:")
        print("--------------------------------------------------")

        print("[MetaData]")
        print(f"  zRatio: {config['MetaData']['zRatio']}")
        print(f"  SIZE:   {config['MetaData']['SIZE']}")
        print(f"  frames: {config['MetaData']['frames']}")
        print(f"  Dim:    {config['MetaData']['Dim']}")
        print(f"  voxelsize: {config['MetaData']['voxelsize']}")
        print(f"  frame rate: {config['MetaData']['frame_rate']} fps")
 
        print("\n[Downsample]")
        print(f"  XY: {config['downsample']['downsampleXY']}")
        if config['downsample']['downsampleZ']==-1:
            print("  Z : All the z slices")
        else:
            print(f"  Z : {config['downsample']['downsampleZ']}")
        print(f"  T : {config['downsample']['downsampleT']}")

        print("\n[File Path]")
        print(f"  input_path :  {config['file_path']['input_path']}")
        print(f"  registrated_path: {config['file_path']['registrated_path']}")

        print("\n[Channels]")
        print(f"  dual_channel : {config['channels']['dual_channel']}")
        print(f"  function     : {config['channels']['function']}")
        print(f"  k            : {config['channels']['k']}")

        print("\n[Reference]")
        print(f"  chunk_size          : {config['reference']['chunk_size']}")

        print("\n[Preprocess]")
        print(f"  normailize_to_0_255 : {config['preprocess']['normailize_to_0_255']}")

        print("\n[Mask]")
        print(f"  thresFactor : {config['mask']['thresFactor']}")
        print(f"  maskRange   : {config['mask']['maskRange']}")

        print("\n[Pyramid]")
        print(f"  layer         : {config['pyramid']['layer']}")
        print(f"  r             : {config['pyramid']['r']}")
        print(f"  iter          : {config['pyramid']['iter']}")
        print(f"  smoothPenalty : {config['pyramid']['smoothPenalty']}")

        print("--------------------------------------------------")
        print("Configuration loaded successfully.\n")

def Registration(configPath='./configs/config.toml'):
    config=toml.load(configPath)
    
    ##################################################################################################################################
    ## create the output File(zarr)
    print("Creating the zarr file...")
    outputfilePath=config['file_path']['registrated_path']
    movingFilePath=config['file_path']['input_path']
    # We need to know the shape of data first (X,Y,Z,T)
    Dim=config['MetaData']['Dim']
    downsampleXY=config['downsample']['downsampleXY']
    downsampleZ=config['downsample']['downsampleZ']
    downsampleT=config['downsample']['downsampleT']
    total_frames=config['MetaData']['frames']

    X=config['MetaData']['SIZE'][0]/downsampleXY
    Y=config['MetaData']['SIZE'][1]/downsampleXY
    

    if Dim==3:
        if downsampleZ ==-1:
            Z=config['MetaData']['SIZE'][2]
            downsampleZ=list(range(Z))
        else:
            Z=len(downsampleZ)

    #create the dataset
    root=zarr.open(outputfilePath,mode='w')
    if Dim==3:
        mem_z=root.create_dataset(
            'membrane',shape=(total_frames,Z,X,Y),
            chunk=(1,Z,X,Y),dtype="f4",compressor=None
        )
        cal_z=root.create_dataset(
            'calcium',shape=(total_frames,Z,X,Y),
            chunk=(1,Z,X,Y),dtype="f4",compressor=None
        )
        if config['save_config']['save_ref']:
            ref_z=root.create_dataset(
                'reference',shape=(total_frames,Z,X,Y),
                chunk=(1,Z,X,Y),dtype="f4",compressor=None
            )
        if config['save_config']['save_motion']:
            motion_z=root.create_dataset(
                'motion',shape=(total_frames,Z,X,Y,3),
                chunk=(1,Z,X,Y,3),dtype="f4",compressor=None
            )
    elif Dim==2:
        mem_z=root.create_dataset(
            'membrane',shape=(total_frames,X,Y),
            chunk=(1,X,Y),dtype="f4",compressor=None
        )
        cal_z=root.create_dataset(
            'calcium',shape=(total_frames,X,Y),
            chunk=(1,X,Y),dtype="f4",compressor=None
        )
        if config['save_config']['save_ref']:
            ref_z=root.create_dataset(
                'reference',shape=(total_frames,X,Y),
                chunk=(1,X,Y),dtype="f4",compressor=None
            )
        if config['save_config']['save_motion']:
            motion_z=root.create_dataset(
                'motion',shape=(total_frames,X,Y,3),
                chunk=(1,X,Y,3),dtype="f4",compressor=None
            )
    ##################################################################################################################################
    ## save the metadata
    root.attrs['Metadata']={
        'zRatio':config['MetaData']['zRatio'],
        'SIZE':config['MetaData']['SIZE'],
        'frames':config['MetaData']['frames'],
        'Dim':config['MetaData']['Dim'],
        'frames_rate':config['MetaData']['frame_rate'],
        'voxelsize':config['MetaData']['voxelsize'],
        'datatype':'f4'
    }
    ##################################################################################################################################
    ### Do registration
    print("Beginning to registrate all the frames...")
    save_motion=config["save_config"]["save_motion"]
    save_ref=config["save_config"]["save_ref"]

    ## 1.process the downsample data
    # ===================================== Step 1: process the middle block =====================================
    # basic defination
    base_transpose_axes = (0, 3, 2, 1) if Dim == 3 else (0, 2, 1)
    motion_transpose_axes = base_transpose_axes + (4,)  
    print(" Processing the middle chunk")
    chunk_size=config['reference']['chunk_size']
    mid_chunk_size=config['reference']['mid_chunk_size']
    half_chunk = mid_chunk_size // 2
    total_mid = total_frames // 2
    mid_start = total_mid - half_chunk
    mid_end   = mid_start + mid_chunk_size
    frames_mid = list(range(mid_start, mid_end))

    #read the data and compute the reference
    mem_mid = IO.readFrame(movingFilePath, frames_mid, downsampleZ, channel=1,xy_down=downsampleXY)
    ca_mid  = IO.readFrame(movingFilePath, frames_mid, downsampleZ, channel=0,xy_down=downsampleXY)
    print("     Finish loading the data of the middle block")        
    ref_img= reference.compute_reference_from_block(mem_mid,ca_mid,config)
    print("     Finish picking the initial reference from the middle block")
    
    #registrate the frames
    if Dim==3:
        mem_mid, ca_mid, _, _, motion_mid = registration.wbi_registration_3d(
            mem_mid, ca_mid, configPath, ref_img,frame=mid_start
        )

    elif  Dim==2:
        mem_mid, ca_mid, _, _, motion_mid = registration.wbi_registration_2d(
            mem_mid, ca_mid, configPath, ref_img,frame=mid_start
        )
        
    #  save the result
    batch_size = 10 # write 10 frames each time
    for i in range(0, len(motion_mid), batch_size):
        j = min(i + batch_size, len(motion_mid))
        mem_batch=mem_mid[i:j].astype("f4")
        cal_batch=ca_mid[i:j].astype("f4")
        if save_motion:
            motion_batch=motion_mid[i:j].astype("f4")
        if save_ref:
            ref_batch=np.repeat(ref_img[None, ...], j - i, axis=0).astype("f4")

        mem_z[mid_start + i:mid_start + j] = mem_batch.transpose(base_transpose_axes)
        cal_z[mid_start + i:mid_start + j] = cal_batch.transpose(base_transpose_axes)

        if save_motion:
            motion_z[mid_start + i:mid_start + j] = motion_batch.transpose(motion_transpose_axes)

        if save_ref:
            ref_z[mid_start + i:mid_start + j] = ref_batch.transpose(base_transpose_axes)

    print("     Finish processing the middle block")

    archors=[]
    # ===================================== Step 2: process backward =====================================
    base_transpose_axes_1frame = (2, 1, 0) if Dim == 3 else (1, 0)
    motion_transpose_axes_1frame = base_transpose_axes + (3,)  
    ref_windows_mem=np.array(mem_mid[0:chunk_size])
    ref_windows_ca=np.array(ca_mid[0:chunk_size])
    print(f" Processing Backward: frame {mid_start - downsampleT} to 0 (every {downsampleT} frames)")
    for idx in range(mid_start - downsampleT, -1, -downsampleT):
        archors.append(idx)
        mem_ref_block = ref_windows_mem
        ca_ref_block  = ref_windows_ca

        #read the data
        mem_img = IO.readFrame(movingFilePath, idx, downsampleZ,channel=1,xy_down=downsampleXY)
        ca_img  = IO.readFrame(movingFilePath, idx, downsampleZ,channel=0,xy_down=downsampleXY)

        #registrate this frames
        mem_reg, ca_reg, ref_img,motion_reg = registration.register_one_frame(configPath, mem_img, ca_img,
                                                    {"mem": mem_ref_block, "ca": ca_ref_block},verbose=True,idx=idx
                                                    )
        # save the result
        mem_z[idx]=mem_reg.astype("f4").transpose(base_transpose_axes_1frame)
        cal_z[idx]=ca_reg.astype("f4").transpose(base_transpose_axes_1frame)
        if save_ref:
            ref_z[idx] = ref_img.astype("f4").transpose(base_transpose_axes_1frame)
        if save_motion:
            motion_z[idx]=motion_reg.transpose(motion_transpose_axes_1frame)

        #updata the window
        mem_ref_block[1:chunk_size]=mem_ref_block[0:chunk_size-1]
        mem_ref_block[0]=mem_reg
        ca_ref_block[1:chunk_size]=ca_ref_block[0:chunk_size-1]
        ca_ref_block[0]=ca_reg

    # ===================================== Step 3: process forward =====================================
    ref_windows_mem=np.array(mem_mid[-chunk_size:])
    ref_windows_ca=np.array(ca_mid[-chunk_size:])
    print(f" Processing Backward: frame {mid_end} to {total_frames} (every {downsampleT} frames)")
    for idx in range(mid_end, total_frames, downsampleT):
        archors.append(idx)
        mem_ref_block = ref_windows_mem
        ca_ref_block  = ref_windows_ca

        mem_img = IO.readFrame(movingFilePath, idx, downsampleZ,channel=1,xy_down=downsampleXY)
        ca_img  = IO.readFrame(movingFilePath, idx, downsampleZ,channel=0,xy_down=downsampleXY)

        mem_reg, ca_reg, ref_img,motion_reg = registration.register_one_frame(configPath, mem_img, ca_img,
                                                    {"mem": mem_ref_block, "ca": ca_ref_block},verbose=True,idx=idx
                                                    )
        mem_z[idx]=mem_reg.astype("f4").transpose(base_transpose_axes_1frame)
        cal_z[idx]=ca_reg.astype("f4").transpose(base_transpose_axes_1frame)
        if save_ref:
            ref_z[idx] = ref_img.astype("f4").transpose(base_transpose_axes_1frame)
        if save_motion:
            motion_z[idx]=motion_reg.transpose(motion_transpose_axes_1frame)

        #updata the window
        mem_ref_block[0:chunk_size-1]=mem_ref_block[1:chunk_size]
        mem_ref_block[-1]=mem_reg
        ca_ref_block[0:chunk_size-1]=ca_ref_block[1:chunk_size]
        ca_ref_block[-1]=ca_reg
        #motion[z_idx] = motion_reg.astype("f4")
    
    ## 2.process the whole data
    if downsampleT!=1:
        for a in archors:
            start_idx=max(0,a - downsampleT//2 + 1)
            end_idx=min(total_frames, a + downsampleT//2)
            print(f" Processing the frames {start_idx} to {end_idx} with archor frame {a}")
            #read the reference data
            if Dim==3:
                ref_mem=mem_z[a].transpose((2,1,0))
                ref_ca=cal_z[a].transpose((2,1,0))
            elif Dim==2:
                ref_mem=mem_z[a].transpose((1,0))
                ref_ca=cal_z[a].transpose((1,0))
            if config['channels']['dual_channel']:
                ref_img=ref_mem+registration.transform(ref_ca,config['channels']['k'],config['channels']['function'])
            else:
                ref_img=ref_mem
            frames_processing=list(range(start_idx,end_idx))

            mem_img = IO.readFrame(movingFilePath, frames_processing, downsampleZ,channel=1,xy_down=downsampleXY)
            ca_img  = IO.readFrame(movingFilePath, frames_processing, downsampleZ,channel=0,xy_down=downsampleXY)

            if Dim==3:
                mem_reg, ca_reg, _, _, motion_mid = registration.wbi_registration_3d(
                    mem_img, ca_img, configPath, ref_img,frame=start_idx
                )

            elif  Dim==2:
                mem_reg, ca_reg, _, _, motion_mid = registration.wbi_registration_2d(
                    mem_img, ca_img, configPath, ref_img,frame=start_idx
                )
            #save the result
            mem_z[start_idx:end_idx] = mem_reg.transpose(base_transpose_axes)
            cal_z[start_idx:end_idx] = ca_reg.transpose(base_transpose_axes)

            if save_motion:
                motion_z[start_idx:end_idx] = motion_mid.transpose(motion_transpose_axes)
            if save_ref:
                ref_z[start_idx:end_idx] = np.repeat(ref_img[None, ...], end_idx - start_idx, axis=0).astype("f4").transpose(base_transpose_axes)
    ##################################################################################################################################


def create_downsample_dataset(
    configPath='./configs/config.toml',
    downsampleFilePath='./registrated_downsample.zarr',
    ds_XY=4,
    ds_T=2
):

    config = toml.load(configPath)

    raw_path = config['file_path']['input_path']             
    reg_path = config['file_path']['registrated_path']       

    Dim = config['MetaData']['Dim']
    total_frames = config['MetaData']['frames']

    base_dsXY = config['downsample']['downsampleXY']
    base_dsT  = config['downsample']['downsampleT']

    raw_dsXY = base_dsXY * ds_XY     
    reg_dsXY = ds_XY               
    raw_dsT  = base_dsT * ds_T
    reg_dsT  = ds_T

    raw_z = zarr.open(raw_path, mode='r')
    reg_z = zarr.open(reg_path, mode='r')

    has_ref    = 'reference' in reg_z
    has_motion = 'motion' in reg_z

    if os.path.exists(downsampleFilePath):
        os.system(f"rm -rf {downsampleFilePath}")
    root_out = zarr.open(downsampleFilePath, mode='w')

    compressor = Blosc(cname="zstd", clevel=3, shuffle=1)

    
    if Dim == 3:
        _, Z_raw, X_raw, Y_raw = raw_z['membrane'].shape
    else:
        _, X_raw, Y_raw = raw_z['membrane'].shape
        Z_raw = None

    def compute_resize_shape(orig_shape, k, is3d):
        if k == 1:
            return orig_shape
        if is3d:
            # orig = (Z, X, Y), want (Z, X//k, Y//k)
            Z, X, Y = orig_shape
            return (Z, X // k, Y // k)
        else:
            X, Y = orig_shape
            return (X // k, Y // k)

    if Dim == 3:
        raw_spatial_ds = compute_resize_shape((Z_raw, X_raw, Y_raw), raw_dsXY, is3d=True)
        reg_spatial_ds = compute_resize_shape((Z_raw, X_raw, Y_raw), reg_dsXY, is3d=True)
        assert raw_spatial_ds == reg_spatial_ds
        Z_ds, X_ds, Y_ds = raw_spatial_ds
    else:
        raw_spatial_ds = compute_resize_shape((X_raw, Y_raw), raw_dsXY, is3d=False)
        reg_spatial_ds = compute_resize_shape((X_raw, Y_raw), reg_dsXY, is3d=False)
        assert raw_spatial_ds == reg_spatial_ds
        X_ds, Y_ds = raw_spatial_ds

    T_raw_ds = total_frames // raw_dsT
    T_reg_ds = total_frames // reg_dsT
    assert T_raw_ds == T_reg_ds
    T_ds = T_raw_ds

    raw_index = np.arange(0, total_frames, raw_dsT)
    reg_index = np.arange(0, total_frames, reg_dsT)
    raw_index = raw_index[:T_ds]
    reg_index = reg_index[:T_ds]

    if Dim == 3:
        out_shape = (T_ds, Z_ds, X_ds, Y_ds * 2)
        out_chunk = (1, Z_ds, X_ds, Y_ds * 2)
    else:
        out_shape = (T_ds, X_ds, Y_ds * 2)
        out_chunk = (1, X_ds, Y_ds * 2)

    mem_out = root_out.create_dataset('membrane', shape=out_shape, chunk=out_chunk, dtype='f4', compressor=compressor)
    cal_out = root_out.create_dataset('calcium', shape=out_shape, chunk=out_chunk, dtype='f4', compressor=compressor)

    if has_ref:
        ref_out = root_out.create_dataset('reference', shape=out_shape, chunk=out_chunk, dtype='f4', compressor=compressor)

    if has_motion:
        if Dim == 3:
            out_shape_m = (T_ds, Z_ds, X_ds, Y_ds * 2, 3)
            out_chunk_m = (1, Z_ds, X_ds, Y_ds * 2, 3)
        else:
            out_shape_m = (T_ds, X_ds, Y_ds * 2, 3)
            out_chunk_m = (1, X_ds, Y_ds * 2, 3)
        motion_out = root_out.create_dataset('motion', shape=out_shape_m, chunk=out_chunk_m, dtype='f4', compressor=compressor)


    def resize_xy(data, ds_factor, is3d):
        if ds_factor == 1:
            return data.astype(np.float32)

        if is3d:
            # data: (Z, X, Y)
            Z, X, Y = data.shape
            newZ = Z
            newX = X // ds_factor
            newY = Y // ds_factor
            out = resize(
                data,
                (newZ, newX, newY),
                order=1,
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.float32)
            return out
        else:
            # data: (X, Y)
            X, Y = data.shape
            newX = X // ds_factor
            newY = Y // ds_factor
            out = resize(
                data,
                (newX, newY),
                order=1,
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.float32)
            return out


    def get_concat_frame(z_raw, z_reg, key, t_raw, t_reg):
        raw_f = z_raw[key][t_raw]
        reg_f = z_reg[key][t_reg]

        is3d = (Dim == 3)

        raw_f_ds = resize_xy(raw_f, raw_dsXY, is3d=is3d)
        reg_f_ds = resize_xy(reg_f, reg_dsXY, is3d=is3d)

        return np.concatenate([raw_f_ds, reg_f_ds], axis=-1)


    for ti, (tr, tg) in enumerate(zip(raw_index, reg_index)):
        mem_out[ti] = get_concat_frame(raw_z, reg_z, 'membrane', tr, tg)
        cal_out[ti] = get_concat_frame(raw_z, reg_z, 'calcium', tr, tg)

        if has_ref:
            ref_out[ti] = get_concat_frame(raw_z, reg_z, 'reference', tr, tg)

        if has_motion:
            # motion: shape (..., 3)
            raw_m = raw_z['motion'][tr]
            reg_m = reg_z['motion'][tg]

            if Dim == 3:
                raw_m_ds = np.stack([resize_xy(raw_m[..., c], raw_dsXY, is3d=True) for c in range(3)], axis=-1)
                reg_m_ds = np.stack([resize_xy(reg_m[..., c], reg_dsXY, is3d=True) for c in range(3)], axis=-1)
            else:
                raw_m_ds = np.stack([resize_xy(raw_m[..., c], raw_dsXY, is3d=False) for c in range(3)], axis=-1)
                reg_m_ds = np.stack([resize_xy(reg_m[..., c], reg_dsXY, is3d=False) for c in range(3)], axis=-1)

            motion_out[ti] = np.concatenate([raw_m_ds, reg_m_ds], axis=-2)

    print(f"Done. Visualization zarr saved to: {downsampleFilePath}")

# def getMask(config,
#             maskPath,
#             n=20,
#             n2=20,
#             n3=40,
#             sigma=3,
#             sigma2=10.0,
#             decay=0.99,
#             temporalMaskThres=3,
#             spatialMaskThres=-1
#             ):
#     option={
#         'n':n,
#         'n2':n2,
#         'n3':n3,
#         'sigma':sigma,
#         'sigma2':sigma2,
#         'decay':decay,
#         'temporalMaskThres':temporalMaskThres,
#         'spatialMaskThres':spatialMaskThres
#     }
#     config=toml.load(config)
#     Dim=config['MetaData']['Dim']
#     root=zarr.open(config['file_path']['registrated_path'],mode='r')
#     mem = root["membrane"]
#     ca = root["calcium"]
#     if "reference" not in root:
#         raise ValueError("Input zarr must contain 'reference' dataset.")
#     ref = root["reference"]

#       # 假设 3D 情况 (T,X,Y)；如果是 Z 维可自行扩展

#     # 创建输出 zarr
#     out = zarr.open(maskPath, mode='a')
#     if Dim==3:
#         T, Z, X, Y = mem.shape
#         mask_temporal = out.require_dataset(
#             "mask_temporal", shape=(T, Z, X, Y), dtype=float, overwrite=True
#         )
#         mask_accumula = out.require_dataset(
#             "mask_accumula", shape=(T, Z, X, Y), dtype=float, overwrite=True
#         )
#         mask_spatial = out.require_dataset(
#             "mask_spatial", shape=(Z, X, Y), dtype=bool, overwrite=True
#         )

#     if Dim==2:
#         T, X, Y = mem.shape
#         mask_temporal = out.require_dataset(
#             "mask_temporal", shape=(T, X, Y), dtype=float, overwrite=True
#         )
#         mask_accumula = out.require_dataset(
#             "mask_accumula", shape=(T, X, Y), dtype=float, overwrite=True
#         )
#         mask_spatial = out.require_dataset(
#             "mask_spatial", shape=(X, Y), dtype=bool, overwrite=True
#         )

