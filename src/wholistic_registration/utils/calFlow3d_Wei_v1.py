"""

version : 0.3
file name : calFlow3d_Wei_v1.py

Alghothm Author : Wei Zheng (Vigirnia Tech) , Virginia M.S.(HHMI)
Code Author : Wei Zheng for matlab and Yunfeng Chi (Tsinghua University) for python
Last Update Date : 2025/4/10


Overview:
    This script implements motion correction and related opreations using a multi-scale approach.
    The primary goal is to registrate the moving image to a reference image using a GPU-accelerated method.
    Our alghothm is based on optical flow methods.We use a Anisotropic(only x and y directions have scaling change) Pyramid method to capture the large displayment between the moving image and the reference image.
    At the same time,we assumpt the motion field is smooth and continuous.So we calculate the motion of control points with the LK method.At the same time we add a smoothness penalty to the objective function,so that we can get a smooth motion field on each voxel.
    Eventually,we construct an iterative method to get the final motion field.

    
Functions:
    - correctMotionGrid: Correct the motion using 3D interpolation for GPU arrays.
    - getNeiDiff: Calculate the neighbor difference using a filter.
    - calError: Calculate the error and penalty terms for the given 3D data.
    - getSpatialGradientInOrgGrid: Calculate the spatial gradient in the original image using 3D interpolation.
    - getFlow3_withPenalty6: Compute the flow with penalty and 3x3 matrix determinant.
    - compute_new_grid:do the normalization to the origin grid to let the coordinates of control points are integer.
    - getMotion: Function to compute motion correction using multi-scale approach.


"""
from scipy.ndimage import zoom, filters
import numpy as np
from . import cp
from . import interp
from . import calculate
from . import visulization
from .imresize import imresize
def correctMotionGrid(data_raw,coords_new):
    """
    Correct the motion using 3D interpolation for GPU arrays.

    Args:
        data_raw (cupy.ndarray): The raw 3D data (GPU array).
        grid_new (cupy.ndarray): New grid coordinates for interpolation (GPU array).

    Returns:
        dat_corrected (cupy.ndarray): The corrected 3D data after interpolation (GPU array).
    """

    x, y, z = data_raw.shape
    #ensure the data is on GPU
    data_raw = cp.asarray(data_raw, dtype=cp.float32)
    coords_new = cp.asarray(coords_new)
    # print("data_raw.shape is:",data_raw.shape)
    # print("grid_new.shape is:", grid_new.shape)
    coords_new=cp.transpose(coords_new,(3,0,1,2))
    data1_tran = interp.interp3Grid(data_raw, coords_new, method='linear')
    # print("data1_tran.shape is:", data1_tran.shape)
    dat_corrected = cp.reshape(data1_tran, (x, y, z))

    return dat_corrected

def getNeiDiff(phi_current,r):
    """
    Calculate the neighbor difference using a filter.

    Args:
        phi_current (cupy.ndarray): The 3D data (GPU array).
        r (int): The size of the filter.

    Returns:
        neiDiff (cupy.ndarray): The filtered 3D data after applying the neighbor difference filter.
    """
    # Create the neighbor filter (r*2+1 is the size of the filter)
    NeiFltr = cp.ones((r*2+1, r*2+1,1,1),dtype=cp.float32)

    # Normalize the filter, excluding the center
    NeiFltr = NeiFltr / ((r*2+1)**2 - 1)
    NeiFltr[r, r] = -1
    neiDiff = calculate.imfilter(phi_current, NeiFltr, boundary='replicate', output='same', functionality='corr')
    return neiDiff



def calError(It, penaltyRaw, smoothPenaltySum):
    """
    Calculate the error and penalty terms for the given 3D data.

    Args:
        It (cupy.ndarray): The 3D data (GPU array).
        penaltyRaw (cupy.ndarray): The 4D penalty raw values (GPU array).
        smoothPenaltySum (float): The smoothing penalty sum value.

    Returns:
        tuple: diffError (float), penaltyError (float)
    """

    # Get the shape of It
    x, y, z = It.shape

    # diffError: Mean of squared It
    diffError = cp.mean(It**2)
    # penaltyCorrected: Square the penaltyRaw and sum across the 4th dimension
    penaltyCorrected = cp.sum(penaltyRaw**2, axis=3) * smoothPenaltySum
    penaltyError = cp.sum(penaltyCorrected) / (x * y * z)
    # Handle both CuPy and NumPy arrays
    if hasattr(diffError, 'get'):
        return diffError.get(), penaltyError.get()
    else:
        return float(diffError), float(penaltyError)



def getSpatialGradientInOrgGrid(data_raw, coords_new):
    """
    Calculate the spatial gradient on deformed coordinates (out) using 3D interpolation.

    Args:
        data_raw (cupy.ndarray): The raw 3D data (GPU array), shape (H, W, D).
        out (cupy.ndarray): Deformed coordinates, shape (3, H, W, D) 
                           where out[0]=x, out[1]=y, out[2]=z.

    Returns:
        Ix (cupy.ndarray): Gradient along x-axis.
        Iy (cupy.ndarray): Gradient along y-axis.
        Iz (cupy.ndarray): Gradient along z-axis.
    """
    step = 1.0  # Step size for finite differences
    x, y, z = data_raw.shape

    # Extract deformed coordinates
    x_coords, y_coords, z_coords = coords_new[...,0], coords_new[...,1], coords_new[...,2]

    
    # --- Compute gradient along y direction ---
    # Perturb x-coordinate (since y-gradient depends on x changes)
    x_coords_incre = cp.clip(x_coords + step, 0, x-1)
    x_coords_decre = cp.clip(x_coords - step, 0, x-1)
    # Interpolate at (x+step, y, z) and (x-step, y, z)
    data_incre = interp.interp3Grid(data_raw, cp.asarray((x_coords_incre, y_coords, z_coords)))
    data_decre = interp.interp3Grid(data_raw, cp.asarray((x_coords_decre, y_coords, z_coords)))
    Ix = (data_incre - data_decre) / (2 * step)


    # --- Compute gradient along x direction ---
    # Perturb y-coordinate (since x-gradient depends on y changes)
    y_coords_incre = cp.clip(y_coords + step, 0, y-1)
    y_coords_decre = cp.clip(y_coords - step, 0, y-1)
    # Interpolate at (x, y+step, z) and (x, y-step, z)
    data_incre = interp.interp3Grid(data_raw, cp.asarray((x_coords, y_coords_incre, z_coords)))
    data_decre = interp.interp3Grid(data_raw, cp.asarray((x_coords, y_coords_decre, z_coords)))
    Iy = (data_incre - data_decre) / (2 * step)



    # --- Compute gradient along z direction ---
    # Perturb z-coordinate
    z_coords_incre = cp.clip(z_coords + step, 0, z-1)
    z_coords_decre = cp.clip(z_coords - step, 0, z-1)
    # Interpolate at (x, y, z+step) and (x, y, z-step)
    data_incre = interp.interp3Grid(data_raw, cp.asarray((x_coords, y_coords, z_coords_incre)))
    data_decre = interp.interp3Grid(data_raw, cp.asarray((x_coords, y_coords, z_coords_decre)))
    Iz = (data_incre - data_decre) / (2 * step)

    return Ix, Iy, Iz
def getFlow3_withPenalty6(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt, smoothPenaltySum, neiSum):
    """
    Compute the flow with penalty and 3x3 matrix determinant.

    Args:
        Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt (cupy.ndarray): The components for flow calculation.
        smoothPenaltySum (cupy.ndarray): The smooth penalty sum.
        neiSum (cupy.ndarray): The neighbor sum.

    Returns:
        cupy.ndarray: The computed phi gradient flow.
    """
    # Add penalty
    Ixx += smoothPenaltySum
    Iyy += smoothPenaltySum
    Izz += smoothPenaltySum
    Ixt += neiSum[:, :, :, 0]
    Iyt += neiSum[:, :, :, 1]
    Izt += neiSum[:, :, :, 2]
    
    # Get determinant 3x3
    DET = calculate.getDet3(Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
    
    # Get minors
    M11 = calculate.getDet2(Iyy, Iyz, Iyz, Izz)
    M12 = -calculate.getDet2(Ixy, Iyz, Ixz, Izz)
    M13 = calculate.getDet2(Ixy, Iyy, Ixz, Iyz)
    M22 = calculate.getDet2(Ixx, Ixz, Ixz, Izz)
    M23 = -calculate.getDet2(Ixx, Ixy, Ixz, Iyz)
    M33 = calculate.getDet2(Ixx, Ixy, Ixy, Iyy)
    
    # Get flow
    Vx = (M11 * Ixt + M12 * Iyt + M13 * Izt) / DET
    Vy = (M12 * Ixt + M22 * Iyt + M23 * Izt) / DET
    Vz = (M13 * Ixt + M23 * Iyt + M33 * Izt) / DET
    
    # When DET == 0, handle invalid cases (optional)
    # validIdx = DET == 0
    # Vx[validIdx] = -Ixt[validIdx] / Ixx[validIdx]
    # Vy[validIdx] = -Iyt[validIdx] / Iyy[validIdx]
    # Vz[validIdx] = -Izt[validIdx] / Izz[validIdx]
    
    # Merge the gradients into one array
    phi_gradient = cp.stack((Vx, Vy, Vz), axis=-1)
    
    # Replace NaN values with 0
    phi_gradient[cp.isnan(phi_gradient)] = 0

    return phi_gradient

def compute_new_grid(grid, r, motion_shape):
    x_coord, y_coord, z_coord = grid
    #2025/4/19 in matlab the index is begin with 1,but the transform is not a linear function,so we should do some correct to the calculation of the x_new,y_new
    x_new = (x_coord - r ) / (2 * r + 1) 
    y_new = (y_coord - r ) / (2 * r + 1) 
    
    x_new = cp.minimum(cp.maximum(x_new, 0.), motion_shape[0])
    y_new = cp.minimum(cp.maximum(y_new, 0.), motion_shape[1])

    z_new = z_coord
    
    return cp.stack([x_new, y_new, z_new], axis=0)
def getMotion(dat_mov, dat_ref, smoothPenalty_raw, option):
    """
    Function to compute motion correction using multi-scale approach.

    Args:
        dat_mov (cupy.ndarray): The moving image data.
        dat_ref (cupy.ndarray): The reference image data.
        smoothPenalty_raw (float): The smooth penalty parameter.
        option (dict): Dictionary containing parameters for the method, such as layer, iteration number, masks, etc.

    Returns:
        motion_current (cupy.ndarray): The computed motion fields.
        currentError (float): The final error.
        coordinate_new (cupy.ndarray): The corrected indices for each layer.
    """
    # Convert inputs to single precision (float32)
    option['mask_ref'] = cp.asarray(option['mask_ref'], dtype=cp.float32)
    option['mask_mov'] = cp.asarray(option['mask_mov'], dtype=cp.float32)

    # Extract parameters from option
    layer_num = option['layer']              # pyramid layer num
    iterNum = option['iter']
    r = option['r']
    zRatio_raw = option['zRatio']
    iterNum = 10  # Set the number of iterations (override option)

    SZ = dat_mov.shape
    movRange = 5.
    # Multi-scale loop
    for layer in range(layer_num, -1, -1):
        x=int(SZ[0]/(2 ** layer))
        y=int(SZ[1]/(2 ** layer))
        z=SZ[2]
        data1=imresize(cp.asarray(dat_mov),output_shape=(x,y,z))
        data2=imresize(cp.asarray(dat_ref),output_shape=(x,y,z))
        # visulization.visualize_2d_image(data1[:,:,4].get(),title=f"layer {layer} moving image")
        # visulization.visualize_2d_image(data2[:,:,4].get(),title=f"layer {layer} reference image")
        x, y, z = data1.shape
        zRatio = zRatio_raw / (2 ** layer)
        # visulization.visualize_2d_image(data1[:, :, 4].get(), title=f"layer {layer} moving image")
        # visulization.visualize_2d_image(data2[:, :, 4].get(), title=f"layer {layer} reference image")
        # Initialize motion field for each layer
        if layer == layer_num:
            if 'motion' in option and option['motion'] is not None:
                motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
                motion_init=cp.array(option['motion'], dtype=cp.float32)
                motion_current[:, :, :, 0] = cp.asarray(imresize(motion_init[:, :, :, 0], output_shape=(x,y,z))/(SZ[0]/x))
                motion_current[:, :, :, 1] = cp.asarray(imresize(motion_init[:, :, :, 1], output_shape=(x,y,z))/(SZ[1]/y))
                motion_current[:, :, :, 2] = cp.asarray(imresize(motion_init[:, :, :, 2], output_shape=(x,y,z))/(SZ[2]/z))
            else:
                motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
        else:
            motion_current_temp = cp.asarray(motion_current)
            motion_current = cp.zeros((x, y, z, 3), dtype=cp.float32)
            motion_current[:, :, :, 0] = cp.asarray(imresize(motion_current_temp[:, :, :, 0],output_shape=(x,y,z),method='bilinear')*2)
            motion_current[:, :, :, 1] = cp.asarray(imresize(motion_current_temp[:, :, :, 1],output_shape=(x,y,z),method='bilinear')*2)
            motion_current[:, :, :, 2] = cp.asarray(imresize(motion_current_temp[:, :, :, 2],output_shape=(x,y,z),method='bilinear'))
            motion_current = cp.asarray(motion_current, dtype=cp.float32)
        # Generate index arrays
        #################################################################################################################################
        #2025/4/17 attemp to use grid to speed up

        grid = cp.meshgrid(
            *[cp.arange(n, dtype=cp.float32) for n in data1.shape],
            indexing='ij',
            sparse=False,
        )
        # Initialize mask   
        mask_ref = imresize(option['mask_ref'],output_shape=(x,y,z))>0
        mask_mov = imresize(option['mask_mov'],output_shape=(x,y,z))
        
        # Initial old error
        oldError = cp.inf * cp.ones(3)
        
        # Penalty parameters
        smoothPenalty = smoothPenalty_raw
        patchConnectNum = (r * 2 + 1) ** 2
        smoothPenaltySum = smoothPenalty * patchConnectNum

        # Get patch
        xG = cp.arange(r , x-1, step=2*r+1)
        yG = cp.arange(r , y-1, step=2*r+1)
        zG = cp.arange(0, z)
        xG_grid,yG_grid,zG_grid=cp.meshgrid(xG,yG,zG,indexing='ij')

        # Update motion loop
        for iter in range(iterNum):
            # Get corrected data
            coords_new=interp.correctGrid(motion_current,grid)
            # if iter==0 and layer==3:
            #      print("x_new[0:50]:",x_new[0:50])
            #      print("y_new[0:50]:",y_new[0:50])
            #      print("z_new[0:50]:",z_new[0:50])
            data1_tran = correctMotionGrid(data1,coords_new)
            #print("some points of data1_tran:",data1_tran[10:15,10:15,5])
            mask_mov_current = correctMotionGrid(mask_mov,coords_new) > 0
            mask = mask_mov_current | mask_ref
            # Temporal difference
            #print(cp.array_equal(data1_tran[5:-5,5:-5,5],data1[5:-5,5:-5,5]))
            It = data2 - data1_tran
            It = calculate.imfilter(It,cp.ones((3,3,1))/9,'replicate','same','corr')
            It[mask] = 0
            # print(It.shape)
            # Get neighbor motion difference
            neiDiff = getNeiDiff(motion_current[xG_grid, yG_grid, zG_grid, :], 1)
            neiDiff[:, :, :, 2] = neiDiff[:, :, :, 2] * zRatio
        
            neiSum = smoothPenaltySum * neiDiff

            # Calculate error and decide to stop
            diffError, penaltyError = calError(It, neiDiff, smoothPenaltySum)
            currentError = diffError + penaltyError
            print(f"Downsample: {layer}\tIter: {iter}\tError: {currentError}\tDiff: {diffError}")
            # if (iter==0 or iter==1 or iter==2) and layer==3:
            #     visulization.visualize_2d_image(data1_tran[:,:,5].get(),threshold=(140,225))

            if iter == iterNum - 1 or cp.sum(oldError <= currentError) > 1:
                break
            else:
                oldError[:-1] = oldError[1:]
                oldError[-1] = currentError
            
            # Motion update
            
            # Ix,Iy,Iz=cp.gradient(data1_tran)

            # if iter==0 and layer==3:
            #      print("Ix[20:25,5,5]:",Ix[20:25,5,5])
            #      print("Iy[20:25,5,5]:",Iy[20:25,5,5])
            #      print("Iz[20:25,5,5]:",Iz[20:25,5,5])
            #2025/4/18 20:47 there is no problem of calculating gradient 
            Ix, Iy, Iz = getSpatialGradientInOrgGrid(data1, coords_new)
            # if iter==0 and layer==3:
            #      print(neiDiff)
            #      print("Ix[20:25,5,5]:",Ix[20:25,5,5])
            #      print("Iy[20:25,5,5]:",Iy[20:25,5,5])
            #      print("Iz[20:25,5,5]:",Iz[20:25, 5,5])
            # if iter==0 and layer==3:
            #     print(data1[:,0,0])
            #     print(data2[:,0,0])
            #     print("Ix[20:25,5,5]:",Ix[20:25,5,5])
            #     print("Iy[20:25,5,5]:",Iy[20:25,5,5])
            #     print("Iz[20:25,5,5]:",Iz[20:25,5,5])
            Ix[mask] = 0
            Iy[mask] = 0
            Iz[mask] = 0
            Iz = Iz / zRatio

            # Compute gradients
            AverageFilter=cp.ones((r*2+1,r*2+1,1))
            Ixx = calculate.imfilter(Ix**2 ,AverageFilter,'replicate','same','corr')
            Ixy = calculate.imfilter(Ix*Iy,AverageFilter,'replicate','same','corr')
            Ixz = calculate.imfilter(Ix*Iz,AverageFilter,'replicate','same','corr')
            Iyy = calculate.imfilter(Iy**2 ,AverageFilter,'replicate','same','corr')
            Iyz = calculate.imfilter(Iy*Iz,AverageFilter,'replicate','same','corr')
            Izz = calculate.imfilter(Iz**2 ,AverageFilter,'replicate','same','corr')
            Ixt = calculate.imfilter(Ix*It,AverageFilter,'replicate','same','corr')
            Iyt = calculate.imfilter(Iy*It,AverageFilter,'replicate','same','corr')
            Izt = calculate.imfilter(Iz*It,AverageFilter,'replicate','same','corr')

            Ixx = Ixx[xG_grid, yG_grid, zG_grid]
            Ixy = Ixy[xG_grid, yG_grid, zG_grid]
            Ixz = Ixz[xG_grid, yG_grid, zG_grid]
            Iyy = Iyy[xG_grid, yG_grid, zG_grid]
            Iyz = Iyz[xG_grid, yG_grid, zG_grid]
            Izz = Izz[xG_grid, yG_grid, zG_grid]
            Ixt = Ixt[xG_grid, yG_grid, zG_grid]
            Iyt = Iyt[xG_grid, yG_grid, zG_grid]
            Izt = Izt[xG_grid, yG_grid, zG_grid]
            # if iter==0 and layer==3:
            #     print("Ixx:",Ixx[0,:,1])
            #     print("Ixy:",Ixy[0,:,1])
            #     print("Ixz:",Ixz[0,:,1])
                # print(Ixx.shape)
            # Compute motion update
            motion_update_normalized = getFlow3_withPenalty6(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, Ixt, Iyt, Izt, smoothPenaltySum, neiSum)


            # if layer==layer_num and iter==0:
            #     print(motion_update_normalized)
            # Control points can't move far away
            motion_update_dist = cp.sqrt(cp.sum(motion_update_normalized ** 2, axis=3))
            motion_update_dist = cp.maximum(motion_update_dist / movRange, 1.0)
            motion_update_normalized = motion_update_normalized / motion_update_dist[..., cp.newaxis]

            # Unnormalized motion update
            motion_update = motion_update_normalized

            motion_update[:, :, :, 2] = motion_update[:, :, :, 2] / zRatio

            # Update current motion
            motion_current_CP = motion_current[xG_grid, yG_grid, zG_grid, :] + motion_update
            # if iter==0 and layer==3:
            #     print("motion_current_CP[0,:,0,:]: ",motion_current_CP[0,:,0,:])


            # Interpolate motion update
            coords_new=compute_new_grid(grid,r,motion_current_CP.shape)

            # if layer==layer_num and iter==0:
            #     print("coords_new[0,0,0,:]: ", coords_new[0,0,0,:])

            for dirNum in range(3):
                temp_phi = cp.asarray(motion_current_CP[:, :, :, dirNum])

                motion_current[:, :, :, dirNum] = interp.interp3Grid(temp_phi, coords_new).reshape(x,y,z)
            # if iter==0 and layer==3:
            #     print("motion_current[0,:,0,0]: ",motion_current[0,:,0,0])
    # Final output

    grid = cp.meshgrid(
            *[cp.arange(n, dtype=cp.float32) for n in data1.shape],
            indexing='ij',
            sparse=False,
        )
    coords_new = interp.correctGrid(motion_current,grid)
    # Gather the results
    if hasattr(motion_current, 'get'):
        motion_current = cp.asnumpy(motion_current)
    else:
        motion_current = np.asarray(motion_current)

    return motion_current, currentError, coords_new
    
def correctMotion(data_raw,motion_field):
    grid = np.meshgrid(
            *[np.arange(n, dtype=np.float32) for n in data_raw.shape],
            indexing='ij',
            sparse=False,
        )
    coords_new=interp.correctGrid(motion_field,grid)
    data_tran = correctMotionGrid(data_raw,coords_new)
    if hasattr(data_tran, 'get'):
        data_tran = cp.asnumpy(data_tran)
    else:
        data_tran = np.asarray(data_tran)
    return data_tran


