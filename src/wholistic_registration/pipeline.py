from core import main_function

## Since CUDA requires the spawn start method, we must wrap the code in if __name__ == "__main__": if you want to parallelize the code.

if __name__ == "__main__":
    configFile='./code/wholistic_registration/configs/config_0120.toml'
    # Define data path and the normal config
    main_function.DefineParams(
        configFile=configFile, 
        inputFile='/home/cyf/wbi/Virginia/raw_data/f338/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_7dpf002.nd2',
        outputFile='/home/cyf/wbi/Virginia/registrated_data/f338/f338_registrated_0120', 
        downsampleXY=4,
        dual_channel=True,
        downsampleZ=[4,5,6], #choose which planes to use
        window_size=15, ##  minutes
        mid_window_size=15, ## minutes
        verbose=True
    )

    ###########################################################################################
    ### main process 
    # Do registration
    main_function.Registration_v3(
        configFile,
        parallel=True
    )

    # reliable analysis
    main_function.ReliableAnalysis(
        configFile,
    )

    
    ###########################################################################################
    ### visualization
    #consistent of the mask
    # # ## create downsample data
    main_function.create_downsample_dataset_v3(
        configFile,
        downsampleFilePath='./registrated_data/f338/f338_registrated_0121_downsample/',
        ds_XY=1,
        ds_T=1,
        block_size=50
    )

