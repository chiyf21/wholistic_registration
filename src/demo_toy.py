#%%
"""
Demo script for testing the motion estimation algorithm with synthetic data.
"""
import numpy as np
import os
import tifffile
from registration import motion_estimation
import registration.demo_data as demo_data
import matplotlib.pyplot as plt
from importlib import reload
reload(demo_data)
reload(motion_estimation)


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

# Create output directory if it doesn't exist
output_dir = '../results/'
os.makedirs(output_dir, exist_ok=True)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(frames[0])
axs[1].imshow(frames[1])
plt.savefig(os.path.join(output_dir, 'frames.png'))
#%%


# Initialize motion estimator and core
options = motion_estimation.MotionEstimationOptions(
    layer_num=0,
    iterations=5,
    patch_radius=0,
    smooth_penalty=0.1,
    max_motion_range=10.0,
    save_intermediate_results=True
)
core = motion_estimation.MotionEstimationCore(options)

# Compute initial gradients
Ix, Iy = core._compute_spatial_gradients(frames[1])
# For initial temporal gradient, just compute difference without warping
It = frames[0] - frames[1]
initial_motion = np.zeros((image_size[0], image_size[1], 2))

# Process each frame
reference = frames[0]
estimated_motions = []
estimate_motions_raw = []
warped_frames = []
Its = []

for i in range(1, len(frames)):
    # Initialize motion field
    motion_field = np.zeros((*image_size, 2), dtype=np.float32)
    # print(f"0  sum motion field: {np.sum(np.abs(motion_field) )}")
    
    # Process each layer
    for layer in range(options.layer_num, -1, -1):
        print(f"layer: {layer}")
        # Downsample images for current scale
        scale_factor = 1 
        if scale_factor > 1:
            print(f"downsampling {frames[i].shape} to {core._downsample_image(frames[i], layer).shape}")
            # moving_scaled = core._downsample_image(frames[i], layer)
            # reference_scaled = core._downsample_image(reference, layer)
            # motion_field = core._scale_motion_field(motion_field, moving_scaled.shape)
        else:
            moving_scaled = frames[i]
            reference_scaled = reference
        
        # Process current scale
        for iteration in range(options.iterations):
            print(f"iteration: {iteration}")
            # Warp moving image using current motion field

            # print(f"1  sum motion field: {np.sum(motion_field)}")
            # Compute motion update
            motion_update, motion_update_raw = core.compute_motion_update(
                motion_field=motion_field,
                reference_image=reference_scaled,
                moving_image=moving_scaled,
                layer=layer,
                iteration=iteration,
                output_dir=output_dir
            )
            # motion_update[:,:,1] = 0
            # print max abs motion update
            print(f"max abs motion update: {np.max(np.abs(motion_update))}")
            # motion_update[:,:,1] = 0
            
            motion_field = motion_field  + motion_update
            # print(f" sum motion field: {np.sum(np.abs(motion_field))}")
            
            warped_image = core._warp_image(moving_scaled, motion_field)
            It = reference_scaled - warped_image
            
            # Compute spatial gradients (MATLAB: [Ix, Iy] = getSpatialGradientInOrg2D_Wei)
            Ix, Iy = core._compute_spatial_gradients(warped_image)
            

            estimated_motions.append(motion_field)
            estimate_motions_raw.append(motion_update_raw)
            warped_frames.append(warped_image)
            Its.append(It)                 
            


# Save original reference and moving images in one file
tifffile.imwrite(
    os.path.join(output_dir, 'warped_frames.tif'),
    np.array(warped_frames).astype(np.float32)
)

######### TEST THE WARPING #######
# motion_update = np.zeros_like(motion_update)

# # motion_update[:, :, 1] = -7.0  # 5 pixels in y direction
# motion_update[:, :, 0] = -15.0  # 5 pixels in y direction
# warped = core._warp_image(frames[1], motion_update)

# plt.figure(figsize=(18, 12))
# plt.subplot(1, 3, 1)
# plt.imshow(frames[1], cmap='gray')
# plt.title('Test Warped')
# plt.colorbar(shrink=0.8)

# plt.subplot(1, 3, 2)
# plt.imshow(warped, cmap='gray')
# plt.title('Warped')
# plt.colorbar(shrink=0.8)

####### PLOT THE RESULTS #######

# plot the warped image and the difference and the motion update
plt.figure(figsize=(16, 16))

# Compute spatial gradients
Ix, Iy = core._compute_spatial_gradients(frames[1])

# Find global min/max for motion components for consistent colorscale
motion_vmin = min(motion_field[:, :, 0].min(), motion_field[:, :, 1].min())
motion_vmax = max(motion_field[:, :, 0].max(), motion_field[:, :, 1].max())

# Find min/max for gradients
grad_vmin = min(Ix.min(), Iy.min())
grad_vmax = max(Ix.max(), Iy.max())

# Find min/max for difference
diff = frames[1] - frames[0]
diff_vmin = diff.min()
diff_vmax = diff.max()

diff_reg = warped_image - frames[0]
diff_reg_vmin = diff_reg.min()
diff_reg_vmax = diff_reg.max()

# Row 1: Input Images
plt.subplot(4, 4, 1)
plt.imshow(frames[0], cmap='gray')
plt.title('Reference Image')
plt.colorbar(shrink=0.8)

plt.subplot(4, 4, 2)
plt.imshow(frames[1], cmap='gray')
plt.title('Moving Image')
plt.colorbar(shrink=0.8)

plt.subplot(4, 4, 3)
plt.imshow(diff, cmap='coolwarm', vmin=diff_vmin, vmax=diff_vmax)
plt.title('Original Difference')
plt.colorbar(shrink=0.8)

# Row 2: Spatial Gradients
plt.subplot(4, 4, 4)
plt.imshow(Ix, cmap='coolwarm', vmin=grad_vmin, vmax=grad_vmax)
plt.title('Spatial Gradient (x)')
plt.colorbar(shrink=0.8)

plt.subplot(4, 4, 8)
plt.imshow(Iy, cmap='coolwarm', vmin=grad_vmin, vmax=grad_vmax)
plt.title('Spatial Gradient (y)')
plt.colorbar(shrink=0.8)

plt.subplot(4, 4, 5)
plt.imshow(warped_image, cmap='gray')
plt.title('Warped Image')
plt.colorbar(shrink=0.8)
plt.subplot(4, 4, 6)

plt.imshow(warped_image, cmap='gray')
plt.title('Warped Image')
plt.colorbar(shrink=0.8)

# Row 3: Motion Updates and Results
plt.subplot(4, 4, 9)
im3 = plt.imshow(motion_field[:, :, 0], cmap='coolwarm', 
                 vmin=motion_vmin, vmax=motion_vmax)
plt.title('Motion Update (x) - Horizontal')
plt.colorbar(im3, shrink=0.5)

plt.subplot(4, 4, 10)
im4 = plt.imshow(motion_field[:, :, 1], cmap='coolwarm', 
                 vmin=motion_vmin, vmax=motion_vmax)
plt.title('Motion Update (y) - Vertical')
plt.colorbar(im4, shrink=0.5)

plt.subplot(4, 4, 13)
plt.imshow(motion_update_raw[:,:,0], cmap='coolwarm')
plt.title('Motion Update Raw (x) - Horizontal')
plt.subplot(4, 4, 14)
plt.imshow(motion_update_raw[:,:,1], cmap='coolwarm')
plt.title('Motion Update Raw (y) - Vertical')

plt.subplot(4, 4, 11)
plt.imshow(diff_reg, cmap='coolwarm', 
                 vmin=diff_reg_vmin, vmax=diff_reg_vmax)
plt.title('Difference (reference - warped)')

plt.subplot(4, 4, 12)
im5 = plt.imshow(frames[1]-warped_image, cmap='coolwarm', 
                 vmin=diff_reg_vmin, vmax=diff_reg_vmax)
plt.title('Difference (moving - warped)')
plt.colorbar(im5, shrink=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'motion_estimation.png'))

#%%
vmin = np.min(np.array(warped_frames)-frames[0][None])
vmax = np.max(np.array(warped_frames)-frames[0][None])
# plot all warped_frames
plt.figure(figsize=(18, 12))
for i in range(len(warped_frames)):
    plt.subplot(1, len(warped_frames), i+1)
    plt.imshow(warped_frames[i]-frames[0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.title(f'Warped Frame {i+1}')

#%%
# plot all warped_frames
vmin = np.min(warped_frames[0]-frames[0])
vmax = np.max(warped_frames[-1]-frames[0])
plt.figure(figsize=(18, 12))
for i  in range(1, len(warped_frames)):
    plt.subplot(1, len(warped_frames), i+1)
    plt.imshow(warped_frames[i]-warped_frames[i-1], cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.title(f'Change in warped frame {i+1}')
plt.savefig(os.path.join(output_dir, 'change_in_warped_frame.png'))

vmin = np.min(np.array(warped_frames))
vmax = np.max(np.array(warped_frames))
# plot all warped_frames
plt.figure(figsize=(18, 12))
for i in range(0, len(warped_frames)):
    plt.subplot(1, len(warped_frames), i+1)
    plt.imshow(warped_frames[i], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f'Warped frame {i+1}')
plt.savefig(os.path.join(output_dir, 'warped_frame.png'))

#%%

vmin = np.min(np.abs(estimate_motions_raw[0][:,:,0]-estimate_motions_raw[-1][:,:,1]))
vmax = np.max(np.abs(estimate_motions_raw[-1][:,:,0]-estimate_motions_raw[0][:,:,1]))

# plot all warped_frames
plt.figure(figsize=(18, 12))
for i in range(1, len(warped_frames)):
    plt.subplot(1, len(warped_frames), i+1)
    plt.imshow(estimate_motions_raw[i][:,:,0]-estimate_motions_raw[i-1][:,:,0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.title(f'Changes in warped frame {i+1}')    
plt.savefig(os.path.join(output_dir, 'change_in_motion_update_raw.png'))

vmin = np.min(estimate_motions_raw[0][:,:,0])
vmax = np.max(estimate_motions_raw[-1][:,:,0])
# plot all warped_frames
plt.figure(figsize=(18, 12))
for i in range(0, len(warped_frames)):
    plt.subplot(1, len(warped_frames), i+1)
    plt.imshow(estimate_motions_raw[i][:,:,0], cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.title(f'Motion update frame {i}')    

plt.savefig(os.path.join(output_dir, 'motion_update_raw.png'))