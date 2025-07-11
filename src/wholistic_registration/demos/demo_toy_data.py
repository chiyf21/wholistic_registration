#%%
import numpy as np
import os
import tifffile
from wholistic_registration.utils import preprocess, calFlow3d_Wei_v1, generate_demo_data
from importlib import reload
from wholistic_registration import utils
import matplotlib.pyplot as pl
import tifffile as tf
reload(generate_demo_data)

# Generate synthetic data
image_size = (40, 50)
frames, true_motion = generate_demo_data.generate_cell_movement(
    num_frames=2,
    image_size=image_size,
    num_cells=8,
    max_displacement=30.0,
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

reload(calFlow3d_Wei_v1)

data_ref=frames[0][:,:,None]
data_mov=frames[1][:,:,None]

smoothPenalty_raw=0.01
[X,Y,Z]=data_ref.shape

option={
    'layer':1, # pyramid layer number?  # if 256 x 256 - if layer is 2, then 128 x 128, and 256 x 256 2^layer
    'iter':15, # number of iterations of fitting
    'r':2, # radius of the patch - 2*r + !
    'zRatio':27.693, # how much the z-axis is bigger than x and y
    'motion':0,
    'mask_ref':0,
    'mask_mov':0,
    'movRange': 100,
    'save_ite':1
}

option['motion']= None
option['mask_ref']=np.full(data_ref.shape,False,dtype=bool)
option['mask_mov']=np.full(data_ref.shape,False,dtype=bool)

Pnltfactor=preprocess.getSmPnltNormFctr(data_ref,option)
smoothPenalty=Pnltfactor*smoothPenalty_raw
import time
start = time.time()
motion_current, currentError, coords_new, error_log = calFlow3d_Wei_v1.getMotion(
    data_mov,
    data_ref,
	smoothPenalty,
	option
)
end = time.time()
print("time:",end-start)
#%%
for ind, key in enumerate(list(error_log.keys())[-1:]):
    ncorrections = len(error_log[key]['motion_current'])
    motions = np.array([mc for mc in error_log[key]['motion_current']])
    corr_dat = [utils.calFlow3d_Wei_v1.correctMotion(data_mov,mc) for mc in error_log[key]['motion_current']]
    corr_dat = np.array(corr_dat)
    corr_dat = np.concatenate([data_mov[None],corr_dat],axis=0)
    corr_dat = corr_dat.transpose(0,3,2,1)[:,None].astype('float32')
    motions = motions.transpose(0,4,3,2,1).astype('float32')

#%%

pl.figure()
offset = 0
for ind, key in enumerate(list(error_log.keys())):
    xax = np.arange(len(error_log[key]['currentError'])) + offset
    pl.plot(xax,error_log[key]['currentError'],'o-',label=key)
    offset += len(xax)
    pl.xlabel('Iteration')
    pl.ylabel('Error')
pl.legend()
pl.savefig(os.path.join(output_dir, 'error_log.png'))
tf.imwrite(os.path.join(output_dir, 'registered_data_iterations.tif'), corr_dat, imagej=True)
tf.imwrite(os.path.join(output_dir, 'registered_motion_iterations.tif'), motions, imagej=True)
    
#%%


#%%

reload(calFlow3d_Wei_v1)
data_mov_corrected = calFlow3d_Wei_v1.correctMotion(data_mov,motion_current)

#%%

plt.figure()
plt.imshow(motion_current[:,:,0,0])
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(motion_current[:,:,0,1])
plt.colorbar()
plt.show()

#%%

plt.figure()
plt.imshow(data_mov_corrected[:,:,0]-data_mov[:,:,0])
plt.colorbar()
plt.show()

#%%
plt.figure()
plt.imshow(data_mov_corrected[:,:,0]-data_mov[:,:,0])
plt.colorbar()
plt.show()
#%%
plt.figure()
plt.imshow(data_mov_corrected[:,:,0]-data_ref[:,:,0])
plt.colorbar()
plt.clim(-1,1)
#%%
plt.figure()
plt.imshow(data_mov[:,:,0]-data_ref[:,:,0])
plt.colorbar()
plt.clim(-1,1)
plt.show()
#%%
