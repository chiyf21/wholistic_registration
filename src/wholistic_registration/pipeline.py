from core import main_function

## Define data path and the normal config
main_function.DefineParams(
    configFile='./configs/config.toml',
    inputFile='/home/cyf/wbi/Virginia/f338/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_7dpf002.nd2',
    outputFile='./registrated_data/f338_registrated.zarr',
    downsampleT=20, 
    downsampleXY=4,  #choose downsample rate for registration(indenpendent to the dowsample below,just for accurate my debug)
    downsampleZ=[4,5,6], #choose which planes to use
    chunk_size=20,
    mid_chunk_size=40,
    k=50,
    layer=1,
    function='log10',
    dual_channel=True,
    verbose=False
)

## read in parameters from the config file
# config = toml.load('./configs/config.toml') # TBD

## Do registration
main_function.Registration(
    './configs/config.toml'
)

# # create downsample data
# main_function.create_downsample_dataset_v2(
#     './wbi_1201/configs/config.toml',
#     downsampleFilePath='./registrated_data/f338_1218registrated_downsample/',
#     ds_XY=4,
#     ds_T=5,
#     block_size=10
# )
## create downsample data
main_function.create_downsample_dataset(
    './configs/config.toml',
    downsampleFilePath='./registrated_data/f338_registrated_downsample.zarr',
    ds_XY=4,
    ds_T=4
)



# ## reliable analysis
main_function.ReliableAnalysis(
    './wbi_1201/configs/config.toml',
    ds_XY=4,
    ds_T=5
)
##reference comparation
#main_function.ReferenceComparation( './wbi_1201/configs/config.toml')

