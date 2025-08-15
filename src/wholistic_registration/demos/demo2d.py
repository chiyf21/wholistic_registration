import h5py
from utils import preprocess, calFlow3d_Wei_v1, visualization,mask,option,IO,reference
#By Yunfeng  Chi
#2025/8/11
#add a new package:registration
from utils import registration
import numpy as np
import nd2
import tifffile as tiff
from matplotlib import pyplot as plt
import os

#filePath
movingFilePath="/home/cyf/wbi/Virginia/f2013/250705_f2013_ubi_gcamp7f_bactin_mcherry_6dpf_15849.nd2"
refenceFilePath_start='/home/cyf/wbi/Virginia/f2013/250705_f2013_ubi_gcamp7f_bactin_mcherry_6dpf_15846.nd2'
refenceFilePath_end='/home/cyf/wbi/Virginia/f2013/250705_f2013_ubi_gcamp7f_bactin_mcherry_6dpf_15850.nd2'
config_file='/home/cyf/wbi/Virginia/wbi_0811/wholistic_registration/config.toml'
base_folder="/home/cyf/wbi/Virginia/f2013_0811registraed/"
os.makedirs(base_folder, exist_ok=True)

mem_folder = os.path.join(base_folder, "membrane")
ca_folder = os.path.join(base_folder, "ca")
concat_folder= os.path.join(base_folder, "concat")

os.makedirs(mem_folder, exist_ok=True)
os.makedirs(ca_folder, exist_ok=True)
os.makedirs(concat_folder, exist_ok=True)
#read meta data
meta_start=IO.readMeta(refenceFilePath_start)
meta_end=IO.readMeta(refenceFilePath_end)
meta_moving=IO.readMeta(movingFilePath)

#process 2000 frames in 1 iteration
total_frames = 47997
chunk_size = 2000


for start_frame in range(0, total_frames, chunk_size):
    end_frame = min(start_frame + chunk_size, total_frames)
    print(f"Processing frames {start_frame} ~ {end_frame-1}")

    # read moving images
    frames = range(start_frame, end_frame)
    mem_data = IO.readFrame(movingFilePath, frames, channel=1)
    Ca_data  = IO.readFrame(movingFilePath, frames, channel=0)

    # do regitration
    mem_channel, Ca_channel, concat_images, errors = registration.wbi_registration_2d(
        mem_data,
        Ca_data,
        config_file
    )

    # generate the name of the files
    memPath     = os.path.join(mem_folder, f"membrane_corrected_{start_frame}~{end_frame}.tiff")
    CaPath      = os.path.join(ca_folder, f"Ca_corrected_{start_frame}~{end_frame}.tiff")
    concatPath  = os.path.join(concat_folder, f"Concat_{start_frame}~{end_frame}.tiff")

    # save results
    IO.saveTiff(mem_channel,   config_file, memPath)
    IO.saveTiff(Ca_channel,    config_file, CaPath)
    IO.saveTiff(concat_images, config_file, concatPath)

    print(f"Saved: {memPath}, {CaPath}, {concatPath}")

print("All frames processed.")
