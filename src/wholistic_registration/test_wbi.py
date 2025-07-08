#%%
import numpy as np
import os
import tifffile
import demo_data
import matplotlib.pyplot as plt
from importlib import reload
reload(demo_data)

# Generate synthetic data
image_size = (20, 30)
frames, true_motion = demo_data.generate_cell_movement(
    num_frames=2,
    image_size=image_size,
    num_cells=5,
    max_displacement=25.0,
    radius=3,
    displacement=(0,3),
    seed=10
)
frames = np.array(frames)

# Create output directory if it doesn't exist
output_dir = '../results/'
os.makedirs(output_dir, exist_ok=True)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(frames[0])
axs[1].imshow(frames[1])
plt.savefig(os.path.join(output_dir, 'frames.png'))

#%%
from wholistic_registration.utils import preprocess, calFlow3d_Wei_v1, visulization
from wholistic_registration.utils import option
from importlib import reload
reload(calFlow3d_Wei_v1)
reload(preprocess)
reload(visulization)
# import h5py
# import wbi_0491
# import preprocess,calFlow3d_Wei_v1,visulization

#change the data path
# data_path="C:/Users/admin/Desktop/optical_flow/simulate_v4/simulate_v4/Amp/9.mat"
# with h5py.File(data_path, 'r') as f:
#     dat_ref=np.array(f['dat_ref']).transpose(2,1,0)
#     dat_mov=np.array(f['dat_mov']).transpose(2,1,0)
    
#     motion_current=np.array(f['motion_current_real']).transpose(3,2,1,0)
# dat_ref=dat_ref[500:756,1000:1256 , :]
# dat_mov=dat_mov[500:756,1000:1256 , :]


data_ref=frames[0][:,:,None]
data_mov=frames[1][:,:,None]

smoothPenalty_raw=0.01
[X,Y,Z]=data_ref.shape
option['motion']= None
option['mask_ref']=np.full(data_ref.shape,False,dtype=bool)
option['mask_mov']=np.full(data_ref.shape,False,dtype=bool)

Pnltfactor=preprocess.getSmPnltNormFctr(data_ref,option)
smoothPenalty=Pnltfactor*smoothPenalty_raw
import time
start = time.time()
motion_current = calFlow3d_Wei_v1.getMotion(
    data_mov,
    data_ref,
	smoothPenalty,
	option
)
end = time.time()
print("time:",end-start)
#%%

plt.figure()
plt.imshow(motion_current[0][:,:,0,0])
plt.colorbar()
plt.show()
#%%