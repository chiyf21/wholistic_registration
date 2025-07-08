"""
Core motion estimation implementation with parallel processing capabilities.
This module implements the motion estimation algorithm using a multi-scale
approach with patch-based processing and parallel computation.
"""

from typing import Tuple, Optional, List
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
from tifffile import imwrite
from scipy.signal import convolve2d
from scipy import ndimage
from scipy.ndimage import map_coordinates

@dataclass
class MotionEstimationOptions:
    """Options for motion estimation.
    
    Attributes:
        layer_num: Number of pyramid layers for multi-scale processing
        iterations: Number of iterations per pyramid layer
        patch_radius: Radius of patch for local motion estimation
        z_ratio: Ratio of z to xy resolution
        max_motion_range: Maximum allowed motion magnitude
        smooth_penalty: Weight for smoothness penalty term
        save_intermediate_results: Whether to save intermediate results
    """
    layer_num: int = 3
    iterations: int = 5
    patch_radius: int = 1
    z_ratio: float = 3.0
    max_motion_range: float = 20.0
    smooth_penalty: float = 0.1
    save_intermediate_results: bool = False


class MotionEstimationCore:
    """Core implementation of motion estimation algorithms.
    
    This class implements the core motion estimation algorithm using:
    - Multi-scale pyramid processing
    - Patch-based motion computation
    - Parallel processing for efficiency
    - Smoothness constraints for regularization
    """
    
    def __init__(self, options: MotionEstimationOptions):
        """Initialize motion estimation core with configuration options.
        
        Args:
            options: Configuration options for motion estimation
        """
        self.options = options
        self.patch_size = 2 * options.patch_radius + 1
        self.patch_connect_num = self.patch_size ** 2
        
    def compute_motion_update(
        self,
        motion_field: np.ndarray,
        reference_image: np.ndarray,
        moving_image: np.ndarray,
        layer: int,
        iteration: int,
        output_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute motion update for each patch.
        
        This function:
        1. Warps moving image using current motion field
        2. Computes spatial gradients (Ix, Iy)
        3. Averages gradients over patches using imfilter
        4. Samples at control points
        5. Solves for motion update at control points
        6. Interpolates to full resolution
        
        Args:
            motion_field: Current motion field
            reference_image: Reference image
            moving_image: Moving image
            layer: Current pyramid layer
            iteration: Current iteration
            output_dir: Directory to save intermediate results
            
        Returns:
            Tuple of (motion_update, motion_update_raw)
        """
        # Warp moving image using current motion field
        warped = self._warp_image(moving_image, motion_field)
        print(f"sum motion field: {np.sum(np.abs(motion_field))}")
        # Compute spatial gradients at full resolution
        Ix, Iy = self._compute_spatial_gradients(warped)
        # Ix, Iy = self._compute_spatial_gradients(moving_image)
        # print(f'Ix: {Ix.shape}, Iy: {Iy.shape}')
        # Compute temporal gradient
        It = self._compute_temporal_gradient(moving_image, reference_image, motion_field)
        
        # Average gradients over patches (like MATLAB's imfilter)
        patch_size = 2 * self.options.patch_radius + 1
        kernel = np.ones((patch_size, patch_size)) / (patch_size * patch_size)
        # print(f"shape of Ix Ix :{(Ix*Ix).shape}")
        
        # Apply averaging filter to each gradient component
        
        Ixx = Ix * Ix
        Ixy = Ix * Iy
        Iyy = Iy * Iy
        Ixt = Ix * It
        Iyt = Iy * It
        
        # Ixx = convolve2d(Ixx, kernel, mode='same', boundary='symm')
        # Ixy = convolve2d(Ixy, kernel, mode='same', boundary='symm')
        # Iyy = convolve2d(Iyy, kernel, mode='same', boundary='symm')
        # Ixt = convolve2d(Ixt, kernel, mode='same', boundary='symm')
        # Iyt = convolve2d(Iyt, kernel, mode='same', boundary='symm')
        
        # Get control points (like MATLAB's xG, yG)
        r = self.options.patch_radius
        if r == 0:
            xG = np.arange(0, motion_field.shape[1])
            yG = np.arange(0, motion_field.shape[0])
        else:
            # Match MATLAB: xG=r+1:2*r+1:x; -> Python: indices should be 1 less
            xG = np.arange(r, motion_field.shape[1], 2*r + 1)
            yG = np.arange(r, motion_field.shape[0], 2*r + 1)
        
        # Sample at control points
        Ixx_CP = Ixx[yG, :][:, xG]  # Shape: (len(yG), len(xG))
        Ixy_CP = Ixy[yG, :][:, xG]
        Iyy_CP = Iyy[yG, :][:, xG]
        Ixt_CP = Ixt[yG, :][:, xG]
        Iyt_CP = Iyt[yG, :][:, xG]
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(Ix, cmap='gray')
        # plt.title('Ix')
        # plt.colorbar()
        # plt.show()
        
        # plt.figure(figsize=(10, 10))

        # plt.imshow(Ixx_CP, cmap='gray')
        # plt.title('Ixx_CP')
        # plt.colorbar()
        # plt.show()
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(Ixy_CP, cmap='gray')
        # plt.title('Ixy_CP')
        # plt.colorbar()
        # plt.show()
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(Iyy_CP, cmap='gray')
        # plt.title('Iyy_CP')
        # plt.colorbar()
        # plt.show()
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(Ixt_CP, cmap='gray')
        # plt.title('Ixt_CP')
        # plt.colorbar()
        # plt.show()
        
        # Compute neighbor differences using a filter approach
        motion_CP = np.zeros((len(yG), len(xG), 2))
        for i, y in enumerate(yG):
            for j, x in enumerate(xG):
                motion_CP[i, j, 0] = motion_field[y, x, 0]
                motion_CP[i, j, 1] = motion_field[y, x, 1]
        
        # Create neighbor filter exactly like MATLAB's getNeiDiff function
        patchConnectNum = (2*r + 1)**2
        
        # MATLAB: one/(patchConnectNum-1)-eye(patchConnectNum)/patchConnectNum
        nei_filter = np.ones((2*r + 1, 2*r + 1)) / (patchConnectNum - 1)
        nei_filter[r, r] = 0  # Center point gets -1
        
        # Apply filter to each component
        nei_diff = np.zeros((len(yG), len(xG), 2))
        for i in range(2):
            motion_component = motion_CP[..., i]
            # MATLAB: imfilter(motionC(:,:,:,dirNum),neiFilter,'replicate')
            nei_diff[..., i] = convolve2d(
                motion_component, 
                nei_filter, 
                mode='same', 
                boundary='symm'  # equivalent to MATLAB's 'replicate'
            )
        
        # Calculate the smoothness penalty sum - like MATLAB's smoothPenaltySum
        smooth_penalty_sum = self.options.smooth_penalty 
        # * (2*r + 1)**2
        print(f"smooth_penalty_sum: {smooth_penalty_sum}")
        
        # CRITICAL FIX: Apply smoothness penalty to matrix diagonals like in MATLAB
        # MATLAB applies penalties to the matrix diagonals:
        # Matches MATLAB: solutM = [Ixx(j,i,k)+smoothPenaltySum, Ixy(j,i,k);...
        #                           Ixy(j,i,k), Iyy(j,i,k)+smoothPenaltySum];
        Ixx_CP = Ixx_CP + self.options.smooth_penalty
        Iyy_CP = Iyy_CP + self.options.smooth_penalty
        
        # And also applies penalties to the temporal terms
        Ixt_CP = Ixt_CP + self.options.smooth_penalty * nei_diff[..., 0]  # For x component
        Iyt_CP = Iyt_CP + self.options.smooth_penalty * nei_diff[..., 1]  # For y component
        
        # Initialize output arrays for dx and dy (one value per control point)
        dx = np.zeros(Ixx_CP.shape)
        dy = np.zeros(Ixx_CP.shape)

        # print(f"max of Ixx_CP: {np.max(Ixx_CP)}")
        # print(f"max of Ixy_CP: {np.max(Ixy_CP)}")
        # print(f"max of Iyy_CP: {np.max(Iyy_CP)}")
        # print(f"max of Ixt_CP: {np.max(Ixt_CP)}")
        # print(f"max of Iyt_CP: {np.max(Iyt_CP)}")
        # maxx = np.max(Ixx_CP)


        # Loop over each control point and solve the 2x2 system independently
        for i in range(Ixx_CP.shape[0]):
            for j in range(Ixx_CP.shape[1]):
                # Extract local values at the control point (i, j)
                a = Ixx_CP[i, j]       # (Ixx_CP + smoothness penalty)
                b = Ixy_CP[i, j]
                c = Iyy_CP[i, j]       # (Iyy_CP + smoothness penalty)
                d_x = Ixt_CP[i, j]     # Should come from Ix * It
                d_y = Iyt_CP[i, j]     # Should come from Iy * It

                # Compute the determinant of the 2x2 matrix
                det = a * c - b * b

                # Check if the determinant is sufficiently large
                if abs(det) > 1e-6:
                    # Solve using Cramer's rule:
                    # a*dx + b*dy = -d_x
                    # b*dx + c*dy = -d_y
                    dx[i, j] = (-c * d_x + b * d_y) / det
                    dy[i, j] = (b * d_x - a * d_y) / det
                else:
                    # For a nearly singular matrix, use the pseudo-inverse to get the minimum-norm solution.
                    # Create the local 2x2 system matrix:
                    A_local = np.array([[a, b],
                                        [b, c]])
                    # Right-hand side of the equation is -[d_x, d_y]:
                    d_local = np.array([-d_x, -d_y])
                    # Compute the solution using the pseudo-inverse
                    sol = np.linalg.pinv(A_local) @ d_local
                    dx[i, j] = sol[0]
                    dy[i, j] = sol[1]

                # For debugging: Print details when a equals maxx
                # if a == maxx:
                #     print(f"a: {a}, b: {b}, c: {c}, d_x: {d_x}, d_y: {d_y}")
                #     print(f"det: {det}")
                #     print(f"dx: {dx[i, j]}, dy: {dy[i, j]}")


        # Clip the computed displacements to the maximum allowed motion range
        dx = np.clip(dx, -self.options.max_motion_range, self.options.max_motion_range)
        dy = np.clip(dy, -self.options.max_motion_range, self.options.max_motion_range)
    
        # Add critical MATLAB normalization step that was missing
        # MATLAB: motion_update_dist=sqrt(sum(motion_update_normalized.^2,4));
        # MATLAB: motion_update_dist=max(motion_update_dist./movRange,1);
        # MATLAB: motion_update_normalized=motion_update_normalized./motion_update_dist;
        movRange = 5.0  # Like in MATLAB implementation
        # motion_update_dist = np.sqrt(dx**2 + dy**2)
        # motion_update_dist = np.maximum(motion_update_dist / movRange, 1.0)
        # Apply normalization
        # dx = dx / motion_update_dist
        # dy = dy / motion_update_dist
        
        dx = dx[:,:,np.newaxis]
        dy = dy[:,:,np.newaxis]
        reduced_motion_field = np.concatenate([dx, dy], axis=-1)
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(reduced_motion_field[:,:,0], cmap='gray')
        # plt.colorbar()
        # plt.show()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(reduced_motion_field[:,:,1], cmap='gray')
        # plt.colorbar()
        # plt.show()
        # print(f" max x displacement: {np.max(np.abs(dx))}, max y displacement: {np.max(np.abs(dy))}")
        
        # Interpolate motion field to full resolution
        x_ind, y_ind = np.meshgrid(
            np.arange(motion_field.shape[1]),
            np.arange(motion_field.shape[0])
        )
        # MATLAB: x_new = (x_ind-r-1)/(2*r+1)+1; 
        # For Python, we need to adjust this formula:
        # 1. MATLAB's x_ind is 1-indexed, so in Python x_ind is already 1 less
        # 2. MATLAB adds +1 at the end for 1-based indexing, which we drop
        # So the equivalent becomes:
        x_new = (x_ind-r)/(2*r+1)
        y_new = (y_ind-r)/(2*r+1)
        x_new = np.clip(x_new, 0, len(xG) - 1)
        y_new = np.clip(y_new, 0, len(yG) - 1)
        
        # Interpolate motion field
        # print(f"motion_field.shape: {motion_field.shape}")
        # print(f"dx.shape: {dx.shape}")
        # print(f"x_new.shape: {x_new.shape}")
        # print(f"y_new.shape: {y_new.shape}")
        tmp = np.zeros_like(motion_field)
        for i in range(2):
            tmp[..., i] = self._interpolate_image(
                reduced_motion_field[:,:, i],
                x_new,
                y_new
            )
        
        return tmp, reduced_motion_field
    
    def _combine_masks(
        self,
        mask_moving: Optional[np.ndarray],
        mask_reference: Optional[np.ndarray],
        motion: np.ndarray
    ) -> np.ndarray:
        """Combine moving and reference masks, accounting for motion.
        
        This function:
        1. Warps the moving mask according to current motion
        2. Combines it with the reference mask
        3. Returns the combined mask for gradient computation
        
        Args:
            mask_moving: Mask for moving image
            mask_reference: Mask for reference image
            motion: Current motion field
            
        Returns:
            Combined mask
        """
        if mask_moving is None and mask_reference is None:
            return np.zeros_like(motion[..., 0], dtype=bool)
        
        combined = np.zeros_like(motion[..., 0], dtype=bool)
        
        if mask_moving is not None:
            # Warp moving mask according to motion
            warped_mask = self._warp_image(mask_moving, motion)
            combined |= warped_mask
        
        if mask_reference is not None:
            combined |= mask_reference
        
        return combined
    
    def _compute_spatial_gradients(
        self,
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spatial gradients using central differences.
        
        This function:
        1. Computes x and y gradients using central differences
        2. Matches MATLAB's gradient function behavior
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (grad_x, grad_y) gradients
        """
        # Initialize gradients
        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        
        # Compute x gradient (MATLAB: [gx, gy] = gradient(I))
        # Forward difference at edges
        grad_x[:, 0] = image[:, 1] - image[:, 0]
        grad_x[:, -1] = image[:, -1] - image[:, -2]
        # Central difference in interior
        grad_x[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2
        
        # Compute y gradient
        # Forward difference at edges
        grad_y[0, :] = image[1, :] - image[0, :]
        grad_y[-1, :] = image[-1, :] - image[-2, :]
        # Central difference in interior
        grad_y[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2
        
        return grad_x, grad_y
    
    def _compute_temporal_gradient(
        self,
        moving_image: np.ndarray,
        reference_image: np.ndarray,
        motion_field: np.ndarray
    ) -> np.ndarray:
        """Compute temporal gradient between images.
        
        This function:
        1. Warps the moving image using current motion
        2. Computes temporal difference with reference
        3. No smoothing to preserve details
        
        Args:
            moving_image: Current moving image
            reference_image: Reference image
            motion_field: Current motion field
            
        Returns:
            Temporal gradient
        """
        # Warp moving image using current motion
        warped_image = self._warp_image(moving_image, motion_field)
        
        # Compute temporal difference (MATLAB: It = I2 - I1)
        temporal_diff = reference_image - warped_image
        
        return temporal_diff
    
    def _warp_image(
        self,
        image: np.ndarray,
        motion: np.ndarray
    ) -> np.ndarray:
        """Warp image using motion field.
        
        This function warps an image according to a motion field,
        following Python's 0-based indexing convention.
        
        Args:
            image: Input image
            motion: Motion field with shape (H, W, 2), where [y, x] order
                   motion[..., 0] = y-direction motion (vertical)
                   motion[..., 1] = x-direction motion (horizontal)
            
        Returns:
            Warped image
        """
        # # Ensure motion field has same shape as image
        # if motion.shape[:2] != image.shape[:2]:
        #     print(f"Resizing motion field from {motion.shape} to {image.shape[:2]}")
        #     resized_motion = np.zeros((*image.shape[:2], 2), dtype=motion.dtype)
        #     print(f"resized_motion.shape: {resized_motion.shape}")
        #     for i in range(2):
        #         # Resize each component separately using skimage.transform.resize
        #         resized_motion[..., i] = resize(
        #             motion[..., i], 
        #             image.shape[:2], 
        #             order=1, 
        #             mode='reflect',
        #             anti_aliasing=False
        #         )
        #     motion = resized_motion
            
        # Create coordinate grid (Python 0-based)
        y, x = np.mgrid[:image.shape[0], :image.shape[1]]
        
        
        # Apply motion in reverse direction for backward warping
        # Note: In backward warping, we're finding where each target pixel
        # comes from in the source image
        # motion[..., 0] is y-direction (vertical)
        # motion[..., 1] is x-direction (horizontal)
        x_warped = x - motion[..., 0]  # Subtract x-direction motion
        y_warped = y - motion[..., 1]  # Subtract y-direction motion
        
        # Ensure coordinates are within bounds (0-based indexing)
        x_warped = np.clip(x_warped, 0, image.shape[1] - 1)
        y_warped = np.clip(y_warped, 0, image.shape[0] - 1)
        
        # Interpolate warped image using properly ordered coordinates
        warped = self._interpolate_image(image, x_warped, y_warped)
        # using scipy.ndimage.interpolation.map_coordinates
        
        # warped = map_coordinates(image, np.array([y_warped, x_warped]), order=1)
        
        return warped
    
    def _interpolate_image(
        self,
        image: np.ndarray,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Interpolate image at non-integer coordinates.
        
        This function implements bilinear interpolation using Python's 0-based
        indexing convention with special handling for image boundaries.
        
        Args:
            image: Input image
            x: X coordinates for sampling (horizontal)
            y: Y coordinates for sampling (vertical)
            
        Returns:
            Interpolated image
        """
        # Create output array

        output = map_coordinates(image, np.array([y, x]), order=1)
        # output = np.zeros_like(x, dtype=image.dtype)
        
        # # Get integer coordinates
        # x0 = np.floor(x).astype(int)
        # y0 = np.floor(y).astype(int)
        
        # # Handle boundary conditions - identify points at the edge
        # x_at_edge = (x0 >= image.shape[1] - 1)
        # y_at_edge = (y0 >= image.shape[0] - 1)
        
        # # General case - points not at the edge (most points)
        # mask_interior = ~(x_at_edge | y_at_edge)
        # if np.any(mask_interior):
        #     x0_int = x0[mask_interior]
        #     y0_int = y0[mask_interior]
        #     x1_int = x0_int + 1
        #     y1_int = y0_int + 1
            
        #     # Compute weights for interior points
        #     x_int = x[mask_interior]
        #     y_int = y[mask_interior]
        #     wx1 = (x1_int - x_int)
        #     wx0 = (x_int - x0_int)
        #     wy1 = (y1_int - y_int)
        #     wy0 = (y_int - y0_int)
            
        #     # Get pixel values
        #     v00 = image[y0_int, x0_int]
        #     v01 = image[y0_int, x1_int]
        #     v10 = image[y1_int, x0_int]
        #     v11 = image[y1_int, x1_int]
            
        #     # Interpolate
        #     output[mask_interior] = (wx1 * wy1 * v00 + 
        #                             wx1 * wy0 * v10 + 
        #                             wx0 * wy1 * v01 + 
        #                             wx0 * wy0 * v11)
        
        # # Handle edge cases - use nearest neighbor for last row/column
        # if np.any(x_at_edge | y_at_edge):
        #     mask_edge = (x_at_edge | y_at_edge)
        #     x0_edge = np.clip(x0[mask_edge], 0, image.shape[1] - 1)
        #     y0_edge = np.clip(y0[mask_edge], 0, image.shape[0] - 1)
        #     output[mask_edge] = image[y0_edge, x0_edge]
            
        return output
    
    def _compute_motion_update_parallel(
        self,
        Ix: np.ndarray,
        Iy: np.ndarray,
        It: np.ndarray,
        smooth_penalty: float
    ) -> np.ndarray:
        """Compute motion update using parallel processing.
        
        This function:
        1. Computes patch indices and connectivity
        2. Processes each patch in parallel
        3. Solves motion equation for each patch
        4. Updates motion field
        
        Args:
            Ix: X gradient
            Iy: Y gradient
            It: Temporal gradient
            smooth_penalty: Smoothness penalty weight
            
        Returns:
            Motion update field
            
        Raises:
            ValueError: If input arrays have incompatible shapes
            ValueError: If smooth_penalty is negative
            RuntimeError: If motion computation fails
        """
        # Input validation and type conversion
        if not all(isinstance(arr, np.ndarray) for arr in [Ix, Iy, It]):
            raise TypeError("Inputs must be numpy arrays")
            
        # Convert inputs to float32 if needed
        Ix = Ix.astype(np.float32, copy=False)
        Iy = Iy.astype(np.float32, copy=False)
        It = It.astype(np.float32, copy=False)
            
        if not all(arr.shape == Ix.shape for arr in [Iy, It]):
            raise ValueError("Input arrays must have same shape")
            
        if smooth_penalty < 0:
            raise ValueError("Smooth penalty must be non-negative")
            
        if len(Ix.shape) not in [2, 3]:
            raise ValueError("Input arrays must be 2D or 3D")
            
        # Check for NaN or Inf values
        if np.any(np.isnan(Ix)) or np.any(np.isnan(Iy)) or np.any(np.isnan(It)):
            raise ValueError("Input arrays contain NaN values")
            
        if np.any(np.isinf(Ix)) or np.any(np.isinf(Iy)) or np.any(np.isinf(It)):
            raise ValueError("Input arrays contain infinite values")
            
        # Debug: Check gradient magnitudes
        print(f"Gradient stats - Ix: min={np.min(Ix):.3f}, max={np.max(Ix):.3f}, mean={np.mean(Ix):.3f}")
        print(f"Gradient stats - Iy: min={np.min(Iy):.3f}, max={np.max(Iy):.3f}, mean={np.mean(Iy):.3f}")
        print(f"Gradient stats - It: min={np.min(It):.3f}, max={np.max(It):.3f}, mean={np.mean(It):.3f}")
        
        try:
            # Get patch parameters
            r = self.options.patch_radius
            if r <= 0:
                raise ValueError("Patch radius must be positive")
                
            rz = round(r / self.options.z_ratio)
            if rz <= 0:
                raise ValueError("Z-direction patch radius must be positive")
            
            # Handle both 2D and 3D images
            is_3d = len(Ix.shape) == 3
            
            # Get patch indices (Python 0-based indexing)
            # MATLAB: xG = r+1:2*r+1:Ix.shape[0]
            # For Python, we need to adjust: MATLAB (r+1) → Python (r)
            xG = np.arange(r, Ix.shape[0], 2 * r + 1)
            yG = np.arange(r, Ix.shape[1], 2 * r + 1)
            if is_3d:
                zG = np.arange(rz, Ix.shape[2], 2 * rz + 1)
            else:
                zG = [0]  # Single slice for 2D
            
            # Compute patch connectivity
            if is_3d:
                patch_connect_num = (2 * r + 1) ** 2 * (2 * rz + 1)
            else:
                patch_connect_num = (2 * r + 1) ** 2
                
            # Scale smooth penalty by average gradient magnitude
            avg_grad_mag = np.mean(np.sqrt(Ix**2 + Iy**2))
            smooth_penalty_sum = smooth_penalty * patch_connect_num * avg_grad_mag
            
            # Debug: Print patch parameters
            print(f"Patch parameters - r={r}, rz={rz}, patch_connect_num={patch_connect_num}")
            print(f"Smooth penalty - base={smooth_penalty}, sum={smooth_penalty_sum}")
            print(f"Average gradient magnitude: {avg_grad_mag:.3f}")
            
            # Initialize motion update
            if is_3d:
                motion_update = np.zeros((*Ix.shape, 3), dtype=np.float32)
            else:
                motion_update = np.zeros((*Ix.shape, 2), dtype=np.float32)
            
            # Process each patch
            for i in xG:
                for j in yG:
                    for k in zG:
                        try:
                            # Get patch indices (Python 0-based indexing)
                            i_start = max(0, i - r)
                            i_end = min(Ix.shape[0], i + r + 1)
                            j_start = max(0, j - r)
                            j_end = min(Ix.shape[1], j + r + 1)
                            
                            if is_3d:
                                k_start = max(0, k - rz)
                                k_end = min(Ix.shape[2], k + rz + 1)
                                # Extract 3D patch
                                Ix_patch = Ix[i_start:i_end, j_start:j_end, k_start:k_end]
                                Iy_patch = Iy[i_start:i_end, j_start:j_end, k_start:k_end]
                                It_patch = It[i_start:i_end, j_start:j_end, k_start:k_end]
                            else:
                                # Extract 2D patch
                                Ix_patch = Ix[i_start:i_end, j_start:j_end]
                                Iy_patch = Iy[i_start:i_end, j_start:j_end]
                                It_patch = It[i_start:i_end, j_start:j_end]
                            
                            # Skip empty patches
                            if Ix_patch.size == 0 or Iy_patch.size == 0 or It_patch.size == 0:
                                continue
                            
                            # Check for NaN or Inf in patches
                            if (np.any(np.isnan(Ix_patch)) or np.any(np.isnan(Iy_patch)) or 
                                np.any(np.isnan(It_patch))):
                                continue
                                
                            if (np.any(np.isinf(Ix_patch)) or np.any(np.isinf(Iy_patch)) or 
                                np.any(np.isinf(It_patch))):
                                continue
                            
                            # Compute terms for motion equation
                            Ixx = np.sum(Ix_patch.ravel() ** 2)
                            Ixy = np.sum(Ix_patch.ravel() * Iy_patch.ravel())
                            Iyy = np.sum(Iy_patch.ravel() ** 2)
                            Ixt = np.sum(Ix_patch.ravel() * It_patch.ravel())
                            Iyt = np.sum(Iy_patch.ravel() * It_patch.ravel())
                            
                            # Scale temporal gradient terms
                            scale = np.sqrt(Ixx + Iyy) / (np.abs(Ixt) + np.abs(Iyt) + 1e-6)
                            Ixt *= scale
                            Iyt *= scale
                            
                            # Debug: Print patch statistics
                            print(f"Patch at ({i}, {j}, {k}) - Ixx={Ixx:.3f}, Ixy={Ixy:.3f}, Iyy={Iyy:.3f}")
                            print(f"Patch at ({i}, {j}, {k}) - Ixt={Ixt:.3f}, Iyt={Iyt:.3f}")
                            print(f"Patch scale factor: {scale:.3f}")
                            
                            # Solve motion equation for this patch
                            A = np.array([
                                [Ixx + smooth_penalty_sum, Ixy],
                                [Ixy, Iyy + smooth_penalty_sum]
                            ])
                            b = np.array([-Ixt, -Iyt])
                            
                            # Debug: Check matrix condition
                            det = np.linalg.det(A)
                            print(f"Matrix determinant: {det:.3f}")
                            
                            try:
                                # Use direct solve to match MATLAB's backslash operator
                                motion = np.linalg.solve(A, b)
                                print(f"Computed motion: {motion}")
                            except np.linalg.LinAlgError:
                                # Handle singular matrix case
                                print("Singular matrix detected, using zero motion")
                                motion = np.zeros(2)
                            
                            # Update motion field in patch
                            if is_3d:
                                motion_update[i_start:i_end, j_start:j_end, k_start:k_end, 0] = motion[0]
                                motion_update[i_start:i_end, j_start:j_end, k_start:k_end, 1] = motion[1]
                            else:
                                motion_update[i_start:i_end, j_start:j_end, 0] = motion[0]
                                motion_update[i_start:i_end, j_start:j_end, 1] = motion[1]
                                
                        except Exception as e:
                            raise RuntimeError(f"Error processing patch at ({i}, {j}, {k}): {str(e)}")
            
            return motion_update
            
        except Exception as e:
            raise RuntimeError(f"Error in motion update computation: {str(e)}")

    def estimate_motion(
        self,
        moving_image: np.ndarray,
        reference_image: np.ndarray,
        initial_motion: Optional[np.ndarray] = None,
        mask_moving: Optional[np.ndarray] = None,
        mask_reference: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """Estimate motion between moving and reference images.
        
        This function:
        1. Processes images at multiple scales
        2. Computes motion updates at each scale
        3. Accumulates motion updates into final field
        4. Returns final motion field and loss history
        
        Args:
            moving_image: Moving image to register
            reference_image: Reference image
            initial_motion: Optional initial motion field
            mask_moving: Optional mask for moving image
            mask_reference: Optional mask for reference image
            output_dir: Directory to save intermediate results
            
        Returns:
            Tuple of (final motion field, loss history)
        """
        print("Starting motion estimation...")
        
        # Initialize motion field
        if initial_motion is not None:
            motion_field = initial_motion.copy()
        else:
            motion_field = np.zeros((*moving_image.shape, 2), dtype=np.float32)
        
        # Store loss history
        loss_history = []
        
        # Process each pyramid layer
        for layer in range(self.options.layer_num, -1, -1):
            print(f"\nProcessing layer {layer}")
            
            # Downsample images for current scale
            scale_factor = 2 ** layer
            if scale_factor > 1:
                moving_scaled = self._downsample_image(moving_image, layer)
                reference_scaled = self._downsample_image(reference_image, layer)
                motion_field = self._scale_motion_field(motion_field, moving_scaled.shape)
            else:
                moving_scaled = moving_image
                reference_scaled = reference_image
            
            # Process current scale
            for iteration in range(self.options.iterations):
                print(f"Iteration {iteration + 1}/{self.options.iterations}")
                
                # Compute motion update
                motion_update, motion_update_raw = self.compute_motion_update(
                    motion_field,
                    reference_scaled,
                    moving_scaled,
                    layer,
                    iteration,
                    output_dir
                )
                
                # Update motion field
                motion_field = motion_field + motion_update
                
                # Compute loss
                loss = self._compute_loss(
                    moving_scaled,
                    reference_scaled,
                    motion_field,
                    mask_moving,
                    mask_reference
                )
                loss_history.append(loss)
                
                print(f"Loss: {loss:.6f}")
                print(f"Motion field range: [{np.min(motion_field)}, {np.max(motion_field)}]")
                
                # Save intermediate results
                print(f"\n=== Saving intermediate results for layer {layer}, iteration {iteration} ===")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Output directory: {os.path.join(os.getcwd(), 'registration_results_scale_0')}")
                print(f"Moving image shape: {moving_scaled.shape}")
                print(f"Reference image shape: {reference_scaled.shape}")
                print(f"Motion field shape: {motion_field.shape}")
                
                # Warp the moving image using current motion field
                warped_image = self._warp_image(moving_scaled, motion_field)
                
                # Compute gradients for saving
                Ix, Iy = self._compute_spatial_gradients(warped_image)
                It = self._compute_temporal_gradient(warped_image, reference_scaled, motion_field)
                
                # Save results
                self._save_intermediate_results(
                    Ix=Ix,
                    Iy=Iy,
                    It=It,
                    layer=layer,
                    iteration=iteration,
                    output_dir=output_dir,
                    motion_field=motion_field,
                    warped_image=warped_image,
                    reference_image=reference_scaled
                )
            
            # Scale motion field for next layer
            if layer > 0:
                motion_field = self._scale_motion_field(motion_field, moving_image.shape)
                motion_field = motion_field * 2  # Scale by 2 for next layer
        
        return motion_field, loss_history
    
    def _scale_motion_field(
        self,
        motion: np.ndarray,
        target_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Scale motion field to target shape.
        
        This function:
        1. Resizes motion field to target shape
        2. Scales motion values appropriately (MATLAB: *2)
        
        Args:
            motion: Input motion field
            target_shape: Target shape
            
        Returns:
            Scaled motion field
        """
        # Compute scale factors
        scale_y = target_shape[0] / motion.shape[0]
        scale_x = target_shape[1] / motion.shape[1]
        
        # Resize motion field (MATLAB: imresize3 with "linear" interpolation)
        scaled_motion = np.zeros((*target_shape, 2), dtype=np.float32)
        scaled_motion[..., 0] = np.resize(motion[..., 0], target_shape) * scale_x * 2  # Scale by 2 like MATLAB
        scaled_motion[..., 1] = np.resize(motion[..., 1], target_shape) * scale_y * 2
        
        return scaled_motion
    
    def _save_intermediate_results(
        self,
        Ix: Optional[np.ndarray] = None,
        Iy: Optional[np.ndarray] = None,
        It: Optional[np.ndarray] = None,
        layer: int = 0,
        iteration: int = 0,
        output_dir: Optional[str] = None,
        motion_field: Optional[np.ndarray] = None,
        warped_image: Optional[np.ndarray] = None,
        reference_image: Optional[np.ndarray] = None
    ) -> None:
        """Save intermediate results during motion estimation.
        
        Args:
            Ix: Spatial gradient in x direction
            Iy: Spatial gradient in y direction
            It: Temporal gradient
            layer: Current pyramid layer
            iteration: Current iteration number
            output_dir: Directory to save results
            motion_field: Current motion field
            warped_image: Current warped image
            reference_image: Reference image
        """
        if output_dir is None:
            print("Warning: output_dir is None, skipping saving intermediate results")
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"\nSaving intermediate results to: {os.path.abspath(output_dir)}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return
        
        # Print gradient statistics if available
        if Ix is not None and Iy is not None:
            try:
                # print("\nGradient statistics:")
                # print(f"Spatial gradients - Ix: min={np.min(Ix):.3f}, max={np.max(Ix):.3f}, mean={np.mean(Ix):.3f}")
                # print(f"Spatial gradients - Iy: min={np.min(Iy):.3f}, max={np.max(Iy):.3f}, mean={np.mean(Iy):.3f}")
                
                # Save spatial gradients
                spatial_gradients = np.stack([Ix, Iy], axis=0).astype(np.float32)
                spatial_gradients = np.expand_dims(spatial_gradients, (0, 1))  # Add T, Z dimensions
                spatial_gradients_path = f"{output_dir}/layer{layer}_iter{iteration}_spatial_gradients.tif"
                imwrite(
                    spatial_gradients_path,
                    spatial_gradients,
                    metadata={'axes': 'TZCYX'}
                )
                # print(f"Successfully saved spatial gradients to: {spatial_gradients_path}")
                # print(f"Spatial gradients shape: {spatial_gradients.shape}")
            except Exception as e:
                print(f"Error saving spatial gradients: {e}")
        
        if It is not None:
            try:
                print(f"Temporal gradient - It: min={np.min(It):.3f}, max={np.max(It):.3f}, mean={np.mean(It):.3f}\n")
                
                # Save temporal gradient
                temporal_gradient = np.expand_dims(It.astype(np.float32), (0, 1, 2))  # Add T, Z, C dimensions
                temporal_gradient_path = f"{output_dir}/layer{layer}_iter{iteration}_temporal_gradient.tif"
                imwrite(
                    temporal_gradient_path,
                    temporal_gradient,
                    metadata={'axes': 'TZCYX'}
                )
                print(f"Successfully saved temporal gradient to: {temporal_gradient_path}")
                print(f"Temporal gradient shape: {temporal_gradient.shape}")
            except Exception as e:
                print(f"Error saving temporal gradient: {e}")
        
        if motion_field is not None:
            try:
                # Save motion field
                print(f"\nMotion field before processing - shape: {motion_field.shape}, dtype: {motion_field.dtype}")
                print(f"Motion field range: [{np.min(motion_field)}, {np.max(motion_field)}]")
                
                motion_field_save = motion_field.copy().astype(np.float32)  # Keep original shape
                print(f"Motion field after copy - shape: {motion_field_save.shape}")
                
                motion_field_save = np.moveaxis(motion_field_save, -1, 0)  # Move channels to front
                print(f"Motion field after moveaxis - shape: {motion_field_save.shape}")
                
                motion_field_save = np.expand_dims(motion_field_save, (0, 1))  # Add T, Z dimensions
                print(f"Motion field after expand_dims - shape: {motion_field_save.shape}")
                
                motion_field_path = f"{output_dir}/layer{layer}_iter{iteration}_motion_field.tif"
                imwrite(
                    motion_field_path,
                    motion_field_save,
                    metadata={'axes': 'TZCYX'}
                )
                print(f"Successfully saved motion field to: {motion_field_path}")
            except Exception as e:
                print(f"Error saving motion field: {e}")
                print(f"Motion field details:")
                print(f"Shape: {motion_field.shape if motion_field is not None else 'None'}")
                print(f"dtype: {motion_field.dtype if motion_field is not None else 'None'}")
                if motion_field is not None:
                    print(f"min/max: {np.min(motion_field)}, {np.max(motion_field)}")
                    print(f"Contains NaN: {np.any(np.isnan(motion_field))}")
                    print(f"Contains Inf: {np.any(np.isinf(motion_field))}")
        
        if warped_image is not None and reference_image is not None:
            try:
                print("\nProcessing warped and reference images")
                print(f"Warped image before processing - shape: {warped_image.shape}, dtype: {warped_image.dtype}")
                print(f"Reference image before processing - shape: {reference_image.shape}, dtype: {reference_image.dtype}")
                
                # Normalize images to [0, 1]
                warped_norm = (warped_image - warped_image.min()) / (warped_image.max() - warped_image.min())
                ref_norm = (reference_image - reference_image.min()) / (reference_image.max() - reference_image.min())
                
                # Save individual images
                warped_save = np.expand_dims(warped_norm.astype(np.float32), (0, 1, 2))  # Add T, Z, C dimensions
                warped_path = f"{output_dir}/layer{layer}_iter{iteration}_warped.tif"
                imwrite(
                    warped_path,
                    warped_save,
                    metadata={'axes': 'TZCYX'}
                )
                print(f"Successfully saved warped image to: {warped_path}")
                print(f"Warped image final shape: {warped_save.shape}")
                
                ref_save = np.expand_dims(ref_norm.astype(np.float32), (0, 1, 2))  # Add T, Z, C dimensions
                ref_path = f"{output_dir}/layer{layer}_iter{iteration}_reference.tif"
                imwrite(
                    ref_path,
                    ref_save,
                    metadata={'axes': 'TZCYX'}
                )
                print(f"Successfully saved reference image to: {ref_path}")
                print(f"Reference image final shape: {ref_save.shape}")
                
                # Create and save RGB overlay
                try:
                    print("\nCreating RGB overlay")
                    overlay = np.zeros((*warped_norm.shape, 3), dtype=np.float32)
                    overlay[..., 0] = ref_norm  # Red channel: reference image
                    overlay[..., 1] = warped_norm  # Green channel: warped image
                    
                    overlay_save = np.expand_dims(overlay, (0, 1))  # Add T, Z dimensions
                    overlay_path = f"{output_dir}/layer{layer}_iter{iteration}_overlay.tif"
                    imwrite(
                        overlay_path,
                        overlay_save,
                        metadata={'axes': 'TZCYX'}
                    )
                    print(f"Successfully saved overlay to: {overlay_path}")
                    print(f"Overlay final shape: {overlay_save.shape}")
                except Exception as e:
                    print(f"Error saving overlay: {e}")
                    print(f"Overlay shape: {overlay.shape if 'overlay' in locals() else 'Not created'}")
            except Exception as e:
                print(f"Error saving warped/reference images: {e}")
                print("Image details:")
                print(f"Warped image shape: {warped_image.shape if warped_image is not None else 'None'}")
                print(f"Reference image shape: {reference_image.shape if reference_image is not None else 'None'}")
                print(f"Warped image dtype: {warped_image.dtype if warped_image is not None else 'None'}")
                print(f"Reference image dtype: {reference_image.dtype if reference_image is not None else 'None'}")
                if warped_image is not None and reference_image is not None:
                    print(f"Warped image range: [{np.min(warped_image)}, {np.max(warped_image)}]")
                    print(f"Reference image range: [{np.min(reference_image)}, {np.max(reference_image)}]")
                    print(f"Contains NaN - Warped: {np.any(np.isnan(warped_image))}, Reference: {np.any(np.isnan(reference_image))}")
                    print(f"Contains Inf - Warped: {np.any(np.isinf(warped_image))}, Reference: {np.any(np.isinf(reference_image))}")
    
    def _plot_loss_curve(
        self,
        loss_history: List[float],
        layer: int,
        iteration: int
    ) -> None:
        """Plot and save loss curve.
        
        This function:
        1. Creates a plot of loss history
        2. Saves it as a PNG file
        
        Args:
            loss_history: List of loss values
            layer: Current pyramid layer
            iteration: Current iteration
        """
        plt.figure(figsize=(10, 5))
        plt.plot(loss_history, 'b-', label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Loss Curve (Layer {layer}, Iter {iteration})')
        plt.grid(True)
        plt.legend()
        
        # Save plot
        output_dir = "registration_results_scale_0"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/layer{layer}_iter{iteration}_loss.png")
        plt.close()

    def _downsample_image(self, image, factor):
        """Downsample the image using mean averaging.
        
        Args:
            image: Input image as numpy array
            factor: Downsampling factor
            
        Returns:
            Downsampled image where each pixel is the mean of a 
            factor x factor region
        """
        if factor == 1:
            return image
            
        # Ensure the image dimensions are divisible by the factor
        new_h = (image.shape[0] // factor) * factor
        new_w = (image.shape[1] // factor) * factor
        image = image[:new_h, :new_w]
        
        # Reshape and compute means
        return image.reshape(
            new_h // factor, factor,
            new_w // factor, factor
        ).mean(axis=(1, 3)) 