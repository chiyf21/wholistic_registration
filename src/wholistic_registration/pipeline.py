from core import main_function
from utils import cp
## Since CUDA requires the spawn start method, we must wrap the code in if __name__ == "__main__": if you want to parallelize the code.

if __name__ == "__main__":
    #configFile='/home/cyf/wbi/Virginia/code/wbi_0123/wholistic_registration/src/wholistic_registration/configs/config_f2013_0225.toml'
    configFile = '/home/cyf/wbi/Virginia/code/wbi_0123/wholistic_registration/src/wholistic_registration/configs/config_f338.toml'
    # Define data path and the normal config
    main_function.DefineParams(
        configFile=configFile, 
        inputFile="/home/cyf/wbi/Virginia/raw_data/f338/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_7dpf002.nd2",
        outputFile='/home/cyf/wbi/Virginia/registrated_data/f338/f338_registrated_0126/', 
        downsampleXY=1,
        frame_downsample=1,# new params
        dual_channel=True,
        time_measurement='frame', # 'frame' or 'minute'
        downsampleZ=[5,6,7], #choose which planes to use
        window_size=235, ##  minutes or frames (depends on time_measurement)
        mid_window_size=236, ## minutes or frames (depends on time_measurement)
        reference_chunk=50, ## minutes or frames (depends on time_measurement)
        mid_stride=1, #frames
        offset_radius=5,
        patch_sigma=2,
        structure_tau=0.5, # details are in the main_function.py
        structure_beta=0.05,
        layer=3,
        verbose=True
    )

# # #     # ##########################################################################################
# # #     # ## main process 
# # #     ## Do registration
    # main_function.Registration_v3(
    #     configFile,
    #     parallel=True
    # )

# #     # # # reliable analysis
    main_function.ReliableAnalysis(
        configFile,
    )

    # ###########################################################################################
    # ### visualization
    # #consistent of the mask
    # # # ## create downsample data
    # main_function.create_downsample_dataset_v3(
    #     configFile,
    #     downsampleFilePath='/home/cyf/wbi/Virginia/registrated_data/f2013/f2013_registrated_0204_downsample/',
    #     ds_XY=4,
    #     ds_T=1,
    #     block_size=50
    # )

    # main_function.create_downsample_dataset_v4(
    #     configFile,
    #     downsampleFilePath='/home/cyf/wbi/Virginia/registrated_data/f2013/f2013_registrated_0224_downsample_v4/',
    #     ds_XY=4,
    #     ds_T=5,
    #     n_workers=16
    # )
