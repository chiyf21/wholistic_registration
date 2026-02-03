from core import main_function

## Since CUDA requires the spawn start method, we must wrap the code in if __name__ == "__main__": if you want to parallelize the code.

if __name__ == "__main__":
    # configFile='Z:/wholistic_registration/src/wholistic_registration/configs/config_f2013.toml'
    configFile='Z:/wholistic_registration/src/wholistic_registration/configs/config_0203.toml'
    ## Define data path and the normal config
    main_function.DefineParams(
        configFile=configFile, 
        inputFile='Z:/Virginia_data/f338/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_7dpf002.nd2',
        outputFile='Z:/Virginia_data/f338/f338_registrated_0203/', 
        downsampleXY=4,
        frame_downsample=2,# new paras
        frames=(0, 500),# new params
        dual_channel=False,
        time_measurement='minute', # 'frame' or 'minute'
        downsampleZ=[5,6,7], #choose which planes to use
        window_size=5, ##  minutes or frames (depends on time_measurement)
        mid_window_size=5, ## minutes or frames (depends on time_measurement)
        reference_chunk=2, ## minutes or frames (depends on time_measurement)
        mid_stride=5, #frames
        verbose=True
    )

    # ##########################################################################################
    # ## main process 
    ## Do registration
    # main_function.Registration_v3(
    #     configFile,
    #     parallel=True
    # )

    # # # reliable analysis
    main_function.ReliableAnalysis(
        configFile,
    )

    
    # # ###########################################################################################
    # # ### visualization
    # # #consistent of the mask
    # # # # ## create downsample data
    main_function.create_downsample_dataset_v4(
        configFile,
        downsampleFilePath='Z:/Virginia_data/f338/f338_registrated_0203_downsample/',
        ds_XY=2,
        ds_T=1
    )

