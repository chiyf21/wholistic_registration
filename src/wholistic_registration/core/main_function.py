from utils import IO,reference,registration
import zarr
import toml
import numpy as np
from skimage.transform import resize
from numcodecs import Blosc
from utils.reliableAnalysis import gradient_amplitude,compute_spatial_mask,compute_temporal_and_accumula_masks
import  os
from utils import imresize
import tifffile

def DefineParams(
                configFile='./configs/config.toml',
                inputFile=None,
                outputFile=None,
                maskFile=None,
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
                n=20,
                n2=20,
                n3=40,
                sigma=3,
                sigma2=10.0,
                decay=0.99,
                temporalMaskThres=3,
                spatialMaskThres=-1,
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

    [Reliable Analysis]
    Params:
    - n
        The number of temporal samples used in the first-stage temporal smoothing.
        A larger value offers more robust estimation but may oversmooth fast-changing signals.
    - n2
        The number of temporal samples used in the second, stronger temporal aggregation.
        Typically greater than n, producing a more stable reference for cumulative mask generation.
    - n3
        The number of samples or the effective range for spatial neighborhood evaluation.
        Controls the spatial robustness of the spatial mask by aggregating statistics over a wider window.
    - sigma
        Standard deviation of the Gaussian kernel for the first-pass spatial smoothing.
        Determines the strength of spatial denoising during spatial mask template creation.
    - sigma2
        Standard deviation of the second, stronger Gaussian smoothing operation.
        Used for temporal and accumulative mask estimation to enhance long-term structural stability.
    - decay
        Exponential decay factor governing how historical temporal evidence influences the cumulative mask.
        Values close to 1 emphasize long-term statistics; smaller values focus on recent frames.
    - temporalMaskThres
        Threshold for generating the temporal mask.
        Voxels with temporal stability metrics below this value are marked as unreliable.
    - spatialMaskThres
        Threshold for generating the spatial mask.
        A higher threshold produces a more conservative spatial mask; a lower threshold retains more voxels.
    '''
    ## read the metadata
    print("Reading meta data")
    if inputFile is None:
        raise ValueError("inputFile must not be None. A valid file path is required.")
    
    elif inputFile is not None:
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

    
    ## load the default config file
    import os
    print("Loading the default config file")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(current_dir), 'configs', 'config_default.toml')
    config=toml.load(config_path)

    #change the meta data 
    config['MetaData']['nchannels']=nchannels
    config['MetaData']['zRatio']=zRatio
    config['MetaData']['SIZE']=data_shape
    config['MetaData']['frames']=nframes
    config['MetaData']['Dim']=len(data_shape)
    config['MetaData']['voxelsize']=resolutionxyz
    config['MetaData']['fps']=framerate
    config['MetaData']['spacing_x']=spacing_x
    config['MetaData']['spacing_y']=spacing_y
    config['MetaData']['spacing_z']=spacing_z

    #change the downsample config
    config['downsample']['downsampleXY']=downsampleXY
    config['downsample']['downsampleZ']=downsampleZ
    config['downsample']['downsampleT']=downsampleT
    
    #change the filepath
    config['file_path']['input_path']=inputFile
    config['file_path']['registrated_path']=outputFile
    config['file_path']['mask_path']=maskFile
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

    #change the Reliable analysis config
    config['Reliable_Analysis']['n']=n
    config['Reliable_Analysis']['n2']=n2
    config['Reliable_Analysis']['n3']=n3
    config['Reliable_Analysis']['sigma']=sigma
    config['Reliable_Analysis']['sigma2']=sigma2
    config['Reliable_Analysis']['decay']=decay
    config['Reliable_Analysis']['temporalMaskThres']=temporalMaskThres
    config['Reliable_Analysis']['spatialMaskThres']=-spatialMaskThres

    # change the pyramid config
    config['save_config']['save_ref'] = save_ref
    config['save_config']['save_motion'] = save_motion

    print("Config created =====> Saving the config")
    with open(configFile, "w") as f:
        toml.dump(config, f)
    print("Config saved =====> Done")
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

        print("\n[Reliable Analysis]")
        print(f"  n            : {config['Reliable Analysis']['n']}")
        print(f"  n2           : {config['Reliable Analysis']['n2']}")
        print(f"  n3           : {config['Reliable Analysis']['n3']}")
        print(f"  sigma        : {config['Reliable Analysis']['sigma']}")
        print(f"  sigma2       : {config['Reliable Analysis']['sigma2']}")
        print(f"  decay        : {config['Reliable Analysis']['decay']}")
        print(f"  temporalMaskThres : {config['Reliable Analysis']['temporalMaskThres']}")
        print(f"  spatialMaskThres : {config['Reliable Analysis']['spatialMaskThres']}")

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
def Registration_v2(configPath='./configs/config.toml',
                    parallel=False):
    """
    Full registration pipeline that writes outputs as OME-TIFF per-volume per-channel
    (no zarr). Naming convention: vol_chN_XXXXXX.tif (6-digit frame index).
    Motion is saved as per-frame HDF5 under 'motion/'.

    Assumptions:
      - IO.readND2Frame, registration.*, reference.compute_reference_from_block,
        saveTiff_new are available in the import scope.
      - saveTiff_new(image, save_path, config_path=None, metadata=None, verbose=True)
        expects image shape TZCYX (5D).
    """
    import os
    import toml
    import numpy as np
    import h5py
    # user-provided save function (assumed imported already)
    # from your_module import saveTiff_new
    # load config
    config = toml.load(configPath)

    # basic params
    output_root = config['file_path']['registrated_path']
    movingFilePath = config['file_path']['input_path']

    Dim = config['MetaData']['Dim']
    total_frames = int(config['MetaData']['frames'])
    SIZE = config['MetaData']['SIZE']

    downsampleXY = config['downsample']['downsampleXY']
    downsampleZ = config['downsample']['downsampleZ']
    downsampleT = config['downsample']['downsampleT']

    if Dim == 3:
        if downsampleZ == -1:
            Z_full = int(SIZE[1])
            downsampleZ = list(range(Z_full))
            Z = Z_full
        else:
            Z = len(downsampleZ)
    else:
        Z = 1  # 2D treated as Z=1 for TIFF saving
    # print(downsampleZ)
    # prepare output directories (one folder per channel)
    print("Preparing output directories...")
    out_mem = os.path.join(output_root, "membrane")
    out_ca  = os.path.join(output_root, "calcium")
    out_ref = os.path.join(output_root, "reference")
    out_mot = os.path.join(output_root, "motion")

    os.makedirs(out_mem, exist_ok=True)
    os.makedirs(out_ca, exist_ok=True)

    save_ref = bool(config['save_config']['save_ref'])
    save_motion = bool(config['save_config']['save_motion'])

    if save_ref:
        os.makedirs(out_ref, exist_ok=True)
    if save_motion:
        os.makedirs(out_mot, exist_ok=True)

    # mapping for file naming: use the channel indices you read with IO.readND2Frame
    # In your original code you used channel=1 for membrane and channel=0 for calcium.
    # We'll use those indices in filenames: membrane -> ch1, calcium -> ch0
    channel_index_map = {'membrane': 1, 'calcium': 0}

    # helper: write one volume (Z,Y,X) or (Y,X) image to OME-TIFF using saveTiff_new

        # print(f"Saved {fname} (shape {img5d.shape})")

    # ---------------------------
    # Step A: process the middle chunk (same as your original code)
    # ---------------------------
    print("Processing middle chunk to build initial reference...")
    chunk_size = int(config['reference']['chunk_size'])
    mid_chunk_size = int(config['reference']['mid_chunk_size'])
    half_chunk = mid_chunk_size // 2
    total_mid = total_frames // 2
    mid_start = total_mid - half_chunk
    mid_end   = mid_start + mid_chunk_size
    frames_mid = list(range(mid_start, mid_end))

    # read middle block from nd2
    mem_mid = IO.readND2Frame(movingFilePath, frames_mid, downsampleZ, channel=1, xy_down=downsampleXY, verbose=False)
    ca_mid  = IO.readND2Frame(movingFilePath, frames_mid, downsampleZ, channel=0, xy_down=downsampleXY, verbose=False)
    print(f"Loaded middle block frames {frames_mid[0]} to {frames_mid[-1]}.")
    mem_mid=np.squeeze(mem_mid)
    ca_mid =np.squeeze(ca_mid )
    # compute reference from block
    ref_img = reference.compute_reference_from_block(mem_mid, ca_mid, config)
    print("Computed initial reference image.")

    # registration on middle block
    if Dim == 3:
        mem_mid_reg, ca_mid_reg, _, _, motion_mid = registration.wbi_registration_3d(
            mem_mid, ca_mid, configPath, ref_img, frame=mid_start
        )
    else:
        mem_mid_reg, ca_mid_reg, _, _, motion_mid = registration.wbi_registration_2d(
            mem_mid, ca_mid, configPath, ref_img, frame=mid_start
        )

    # save middle block results (iterate through frames_mid)
    print("Saving middle block results to OME-TIFF...")
    for k, frame_id in enumerate(frames_mid):
        # membrane
        mem_frame = mem_mid_reg[k]  # shape (Z,Y,X) or (Y,X)
        IO.write_volume_as_ome_tiff(mem_frame, out_mem, channel_index_map['membrane'], frame_id,configPath)

        # calcium
        ca_frame = ca_mid_reg[k]
        IO.write_volume_as_ome_tiff(ca_frame, out_ca, channel_index_map['calcium'], frame_id,configPath)
        # reference (optional): ref_img is single 3D or 2D volume; save same ref for each frame in middle block


        # motion (optional)
        if save_motion:
            mot_frame = motion_mid[k]  # shape (Z,Y,X,3) or (Y,X,3)
            mot_fname = os.path.join(out_mot, f"motion_{frame_id:06d}.h5")
            with h5py.File(mot_fname, 'w') as hf:
                hf.create_dataset('motion', data=mot_frame, compression='gzip')
            print(f"Saved {mot_fname} (motion field)")
    if save_ref:
        if Dim == 3:
            ref_frame = ref_img.copy()
        else:
            ref_frame = ref_img.copy()
        IO.write_volume_as_ome_tiff(ref_frame, out_ref, 'ref', f'{mid_start}~{mid_end}',configPath)  # filename will be vol_chref_xxx
    n_gpu = 0

    try:
        import cupy as cp
        try:
            n_gpu = cp.cuda.runtime.getDeviceCount()
            print(f"Detected {n_gpu} GPU(s).")
        except Exception as e:
            print("Failed to query GPU devices via CuPy.")
            print("Falling back to serial mode.")
            n_gpu = 0

    except ImportError:
        print("CuPy is not installed or failed to import.")
        print("Falling back to serial mode.")
        n_gpu = 0

    if parallel == False or n_gpu<=1:
        # ---------------------------
        # Step B: process backward (from mid_start - 1 down to 0)
        # ---------------------------
        
        print(f"Processing backward: frames {mid_start - 1} to 0 with chunk size {downsampleT} ...")
        archors = []
        # initialize rolling window for reference windows (first chunk_size frames from mem_mid/ca_mid)
        ref_windows_mem = np.array(mem_mid_reg[0:chunk_size])
        ref_windows_ca  = np.array(ca_mid_reg[0:chunk_size])

        ref_img = reference.compute_reference_from_block( ref_windows_mem, ref_windows_ca, config)
        
        # iterate backwards in downsampleT steps
        for idx in range(mid_start , -1, -downsampleT):
            end_idx=idx
            start_idx= max(0, end_idx-downsampleT)
            print(f" Processing from frame {start_idx} to frame {end_idx} with reference picked from {end_idx+1} to {end_idx+1+chunk_size}")
            frames_backward=list(range(start_idx,end_idx))
            mem_img = IO.readND2Frame(movingFilePath, frames_backward, downsampleZ, channel=1, xy_down=downsampleXY, verbose=False)
            ca_img  = IO.readND2Frame(movingFilePath, frames_backward, downsampleZ, channel=0, xy_down=downsampleXY, verbose=False)
            mem_img=np.squeeze(mem_img)
            ca_img =np.squeeze(ca_img)

            # pick reference
            ref_img=reference.compute_reference_from_block(ref_windows_mem,ref_windows_ca,config)

            if Dim == 3:
                mem_backward_reg, ca_backward_reg, _, _, motion_mid = registration.wbi_registration_3d(
                    mem_img, ca_img, configPath, ref_img, frame=start_idx
                )
            else:
                mem_backward_reg, ca_backward_reg, _, _, motion_mid = registration.wbi_registration_2d(
                    mem_img, ca_img, configPath, ref_img, frame=start_idx
                )
            for k, frame_id in enumerate(frames_backward):
                # membrane
                mem_frame = mem_backward_reg[k]  # shape (Z,Y,X) or (Y,X)
                IO.write_volume_as_ome_tiff(mem_frame, out_mem, channel_index_map['membrane'], frame_id,configPath)
                # calcium
                ca_frame = ca_backward_reg[k]
                IO.write_volume_as_ome_tiff(ca_frame, out_ca, channel_index_map['calcium'], frame_id,configPath)
                # reference (optional): ref_img is single 3D or 2D volume; save same ref for each frame in middle block

                # motion (optional)
                if save_motion:
                    mot_frame = motion_mid[k]  # shape (Z,Y,X,3) or (Y,X,3)
                    mot_fname = os.path.join(out_mot, f"motion_{frame_id:06d}.h5")
                    with h5py.File(mot_fname, 'w') as hf:
                        hf.create_dataset('motion', data=mot_frame, compression='gzip')
                    print(f"Saved {mot_fname} (motion field)")
            # save reference
            if save_ref:
                if Dim == 3:
                    ref_frame = ref_img.copy()
                else:
                    ref_frame = ref_img.copy()
                IO.write_volume_as_ome_tiff(ref_frame, out_ref, 'ref', f'{start_idx}~{end_idx}',configPath)  # filename will be vol_chref_xxx

            # update rolling window
            ref_windows_mem=np.array(mem_backward_reg[0:chunk_size])
            ref_windows_ca = np.array(ca_backward_reg[0:chunk_size])

        # ---------------------------
        # Step C: process forward (from mid_end + 1 down to total_frames)
        # ---------------------------
        print(f"Processing forward: frames {mid_end+1} to {total_frames-1} with chunk size {downsampleT} ...")

        # initialize rolling window from end of middle block
        ref_windows_mem = np.array(mem_mid_reg[-chunk_size:])
        ref_windows_ca  = np.array(ca_mid_reg[-chunk_size:])

        # iterate forward in steps of downsampleT
        for idx in range(mid_end, total_frames, downsampleT):
            end_idx = min(idx + downsampleT - 1, total_frames - 1)
            start_idx = idx
            print(f" Processing from frame {start_idx} to frame {end_idx} with reference picked from {idx - chunk_size} to {idx}")

            # pick frame indices for this forward chunk
            frames_forward = list(range(start_idx, min(start_idx + downsampleT, total_frames)))
            
            # read moving images
            mem_img = IO.readND2Frame(movingFilePath, frames_forward, downsampleZ, channel=1, xy_down=downsampleXY, verbose=False)
            ca_img  = IO.readND2Frame(movingFilePath, frames_forward, downsampleZ, channel=0, xy_down=downsampleXY, verbose=False)
            mem_img = np.squeeze(mem_img)
            ca_img  = np.squeeze(ca_img)

            # compute reference from current rolling window
            ref_img = reference.compute_reference_from_block(ref_windows_mem, ref_windows_ca, config)

            # registration
            if Dim == 3:
                mem_forward_reg, ca_forward_reg, _, _, motion_forward = registration.wbi_registration_3d(
                    mem_img, ca_img, configPath, ref_img, frame=start_idx
                )
            else:
                mem_forward_reg, ca_forward_reg, _, _, motion_forward = registration.wbi_registration_2d(
                    mem_img, ca_img, configPath, ref_img, frame=start_idx
                )

            # save results
            for k, frame_id in enumerate(frames_forward):
                # membrane
                IO.write_volume_as_ome_tiff(mem_forward_reg[k], out_mem, channel_index_map['membrane'], frame_id, configPath)
                # calcium
                IO.write_volume_as_ome_tiff(ca_forward_reg[k], out_ca, channel_index_map['calcium'], frame_id, configPath)
                # motion
                if save_motion:
                    mot_fname = os.path.join(out_mot, f"motion_{frame_id:06d}.h5")
                    with h5py.File(mot_fname, 'w') as hf:
                        hf.create_dataset('motion', data=motion_forward[k], compression='gzip')
                    print(f"Saved {mot_fname} (motion field)")

            # save reference
            if save_ref:
                if Dim == 3:
                    ref_frame = ref_img.copy()
                else:
                    ref_frame = ref_img.copy()
                IO.write_volume_as_ome_tiff(ref_frame, out_ref, 'ref', f'{start_idx}~{end_idx}',configPath)  # filename will be vol_chref_xxx

            # update rolling window
            ref_windows_mem=np.array(mem_forward_reg[-chunk_size:])
            ref_windows_ca = np.array(ca_forward_reg[-chunk_size:])

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
def read_reg_tiff(folder, frame_idx, ch_idx):
    fname = os.path.join(folder, f"vol_ch{ch_idx}_{frame_idx:06d}.tif")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Cannot find {fname}")
    vol = tifffile.imread(fname)  # (Z,Y,X)
    return vol
def write_volume_as_ome_tiff(volume, out_dir, ch_idx, frame_idx,configPath, spacing_x=1.0, spacing_y=1.0):
    """
    volume: np.ndarray, shape (Z,Y,X) for 3D or (Y,X) for 2D
    out_dir: target directory
    ch_idx: integer channel id used in filename
    frame_idx: integer frame index
    """
    if volume.ndim == 2:
        # make (Z=1, Y, X)
        zvol = volume[np.newaxis, :, :]
    elif volume.ndim == 3:
        zvol = volume
    else:
        raise ValueError("volume must be 2D or 3D (Z,Y,X)")

    # convert to TZCYX: T=1, Z, C=1, Y, X
    t = 1
    Zv, Yv, Xv = zvol.shape
    img5d = zvol[np.newaxis, :, np.newaxis, :, :]  # shape (1,Z,1,Y,X)

    fname = os.path.join(out_dir, f"vol_ch{ch_idx}_{frame_idx:06d}.tif")
    # optional metadata: include spacing if available
    metadata = {'spacing_x': spacing_x, 'spacing_y': spacing_y, 'data_shape': img5d.shape}
    # call the provided save function
    IO.saveTiff_new(img5d, fname, config_path=configPath, metadata=metadata, verbose=False)
def create_downsample_dataset_v2(
    configPath='./configs/config.toml',
    downsampleFilePath='./registrated_downsample',
    ds_XY=4,
    ds_T=2,
    block_size=50
):
    config = toml.load(configPath)

    raw_path = config['file_path']['input_path']          
    reg_path = config['file_path']['registrated_path']    
    Dim = config['MetaData']['Dim']
    total_frames = config['MetaData']['frames']

    base_dsXY = config['downsample']['downsampleXY']
    base_dsT  = config['downsample']['downsampleT']
    base_dsZ  = config['downsample']['downsampleZ']

    if base_dsZ == -1:
        base_dsZ = list(range(config['MetaData']['SIZE'][2]))
    else:
        base_dsZ = base_dsZ

    raw_dsXY = base_dsXY * ds_XY
    reg_dsXY = ds_XY


    if os.path.exists(downsampleFilePath):
        os.system(f"rm -rf {downsampleFilePath}")

    mem_out_dir = os.path.join(downsampleFilePath, "membrane")
    cal_out_dir = os.path.join(downsampleFilePath, "calcium")
    os.makedirs(mem_out_dir)
    os.makedirs(cal_out_dir)

    # Time point indices
    raw_index = np.arange(0, total_frames, ds_T)
    reg_index = np.arange(0, total_frames, ds_T)
    T_ds = min(len(raw_index), len(reg_index))
    raw_index = raw_index[:T_ds]
    reg_index = reg_index[:T_ds]

    # ------------ Get reg dimensions ------------
    reg_mem_path = os.path.join(reg_path, "membrane")
    reg_cal_path = os.path.join(reg_path, "calcium")

    first_reg = IO.read_reg_tiff(reg_mem_path, reg_index[0], ch_idx=1)  # (Z,Y,X)
    Z_raw, Y_raw, X_raw = first_reg.shape
    Y_ds = Y_raw // ds_XY
    X_ds = X_raw // ds_XY

    print(f"[INFO] Downsampled shape: Z={Z_raw}, Y={Y_ds}, X={X_ds}")

    #
    num_blocks = (T_ds + block_size - 1) // block_size
    print(f"[INFO] Total frames after T-downsample: {T_ds}")
    print(f"[INFO] Block count: {num_blocks}")

    for b in range(num_blocks):
        print(f"Processing block {b+1}/{num_blocks}...")
        start = b * block_size
        end   = min((b+1)*block_size, T_ds)
        size  = end - start

        raw_frames = raw_index[start:end]

        # raw_block_mem shape (T, Yd, Xd, Z) after transpose
        raw_block_mem = IO.readND2Frame(
            raw_path, frames=raw_frames, slices=base_dsZ,
            channel=1, xy_down=raw_dsXY, verbose=False
        )

        raw_block_cal = IO.readND2Frame(
            raw_path, frames=raw_frames, slices=base_dsZ,
            channel=0, xy_down=raw_dsXY, verbose=False
        )
        raw_block_mem = np.squeeze(raw_block_mem)
        raw_block_cal = np.squeeze(raw_block_cal)
        ti_reg_list = reg_index[start:end]
        Gm_block_list, Gc_block_list = [], []

        for idx in ti_reg_list:
            gm = IO.read_reg_tiff(reg_mem_path, idx, ch_idx=1)  # (Z,Y,X)
            gc = IO.read_reg_tiff(reg_cal_path, idx, ch_idx=0)
            Gm_block_list.append(gm)
            Gc_block_list.append(gc)

        Gm_block = np.array(Gm_block_list)  # (T,Z,Y,X)
        Gc_block = np.array(Gc_block_list)



        Gm_block_ds = IO.downsample(np.expand_dims(Gm_block, axis=2), xy_down=ds_XY)[:, :, 0]
        Gc_block_ds = IO.downsample(np.expand_dims(Gc_block, axis=2), xy_down=ds_XY)[:, :, 0]

        # -------- concatenate raw + reg (Z,Y,X_raw + X_reg) --------

        for i in range(size):
            frame_id = reg_index[start + i]  
            mem_list=[raw_block_mem[i],Gm_block_ds[i]]
            ca_list=[raw_block_cal[i],Gc_block_ds[i]]

            IO.write_multichannel_volume_as_ome_tiff(
                volume=mem_list, out_dir=mem_out_dir,
                frame_idx=frame_id,
                configPath=configPath,
                label='membrane_downsample'
            )

            IO.write_multichannel_volume_as_ome_tiff(
                volume=mem_list, out_dir=cal_out_dir,
                frame_idx=frame_id,
                configPath=configPath,
                label='calcium_downsample'
            )

        print(f"Block {b+1}/{num_blocks} finished.")

    print("[ALL DONE] Downsampled dataset created successfully.")

def ReliableAnalysis(
    configPath: str = None,
    ds_XY: int = 4,
    ds_T: int = 2,
):
    """
    Main entry function for computing spatial, temporal, and accumulative
    reliability masks.

    Directory structure under `registrated_path` must be:
        root_dir/
            ├── membrane/
            ├── calcium/
            └── reference/

    The user must provide a correlation function:

        def compute_cor_fn(membrane_frame, calcium_frame):
            return dat_cor   # shape (Z, Y, X)

    All output masks will be stored under the directory specified by
    `mask_path` in the config file.
    """
    config= toml.load(configPath)
    root_dir = config['file_path']['registrated_path']
    out_dir  = config['file_path']['mask_path']

    mem_dir = os.path.join(root_dir, "membrane")
    ca_dir  = os.path.join(root_dir, "calcium")
    ref_dir = os.path.join(root_dir, "reference")


    # temporal_dir = os.path.join(out_dir, "temporal_Mask")
    # accumula_dir = os.path.join(out_dir, "accumula_Mask")
    # spatial_dir  = os.path.join(out_dir, "spatial_Mask")
    # os.makedirs(spatial_dir, exist_ok=True)
    # os.makedirs(accumula_dir, exist_ok=True)
    # os.makedirs(temporal_dir, exist_ok=True)
    frames = sorted(os.listdir(mem_dir))
    T = len(frames)

    def compute_cor_fn(mem,ca):
        if config['channels']['dual_channel']:
            k = config['channels']['k']
            function = config['channels']['function']
            ca_transformed = registration.transform(ca, k, function)
            cor = mem + ca_transformed
        else:
            cor = mem
        return cor
    ComputMask(
        mem_dir,
        ca_dir,
        ref_dir,
        out_dir,
        config['Reliable_Analysis'],
        compute_cor_fn,
        configPath,
        T,
        ds_XY,
        ds_T
    )

def ReferenceComparation(configPath: str = None) :
    import re
    config=toml.load(configPath)
    ref_dir=config['file_path']['registrated_path']+'reference/'
    frames = sorted(os.listdir(ref_dir))
    prev_frame=None
    for frame in frames:
        if prev_frame is None:
            prev_frame=tifffile.imread(ref_dir+frame)
            prev_group=re.match(r"vol_chref_(\d+)_(\d+).tif",frame)
            prev_start=int(prev_group.group(1))
            prev_end=int(prev_group.group(2))

        else:
            this_frame=tifffile.imread(ref_dir+frame)
            this_group=re.match(r"vol_chref_(\d+)_(\d+).tif",frame)
            this_start=int(this_group.group(1))
            this_end=int(this_group.group(2))
            difference_map=local_ssim_difference(prev_frame,this_frame)
            
            IO.write_volume_as_ome_tiff(difference_map,
                                        os.path.join(config['file_path']['mask_path'],'Diff_in_reference'),
                                        "Reference_diff",
                                        f'{prev_start}~{prev_end}_vs_{this_start}~{this_end}',
                                        configPath
                                        )
            prev_frame=this_frame
            prev_start=this_start
            prev_end=this_end




