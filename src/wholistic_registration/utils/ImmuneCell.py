"""

version : 0.1
file name : ImmuneCell.py

Algorithm Author : Wei Zheng (Virginia Tech)
Code Author : Wei Zheng for Matlab and Yunfeng Chi (Tsinghua University) for Python
Date : 2025/4/9

Overview:
    This script implements region growing and statistical analysis on 3D image data.
    The primary goal is to identify significant regions within the 3D space based on 
    thresholds and segment-wise order statistics. The script utilizes a multi-scale approach 
    for region growing and calculates Z-scores for the identified regions. The flow of the algorithm 
    includes 3D image processing, Gaussian smoothing, and the application of k-segment order statistics 
    for calculating the expected values and standard deviations, followed by the computation of the Z-scores 
    to assess the significance of each region.

    
Functions:
    - ksegments_orderstatistics_fin: Calculates the expected value (mu) and standard deviation (sigma) 
      based on segment-wise order statistics for two data groups.
    - f1: Helper function to calculate the f1 statistic for a normal distribution.
    - f2: Helper function to calculate the f2 statistic for a normal distribution.
    - regionGrowxx_3D: Performs 3D region growing by shifting points in 3D space for iterative region expansion.
    - synquant_org: Main function for generating a Z-score map for a 3D input image using region growing and 
      statistical analysis, including the application of the k-segment order statistics function.

      
"""
import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter, label

def ksegments_orderstatistics_fin(largeGroup, smallGroup):
    ''' 
    func:
        Calculates the parameters `mu` (mean) and `sigma` (standard deviation) based on segment-wise order statistics,
        which are used to analyze the distribution characteristics of two groups of data.
    Args:
        largeGroup: Data from the large group (array or list of numerical values).
        smallGroup: Data from the small group (array or list of numerical values).
    returns:
        mu: The calculated expected value (mean).
        sigma: The calculated standard deviation.
    '''
    
    # Convert the input groups to NumPy arrays with floating point type
    fg = np.array(largeGroup, dtype=float)
    bg = np.array(smallGroup, dtype=float)
    
    # M and N represent the sizes of the large and small groups respectively
    M = len(fg)
    N = len(bg)
    n = M + N  # Total number of data points

    # delta represents the normalization factor for the data
    delta = 1 / n
    
    # Concatenate the small and large group data and label them (0 for small group, 1 for large group)
    all_data = np.concatenate([bg, fg])
    labels = np.concatenate([np.full_like(bg, -1), np.full_like(fg, 1)])

    # Sort all data and get the sorted indices
    od = np.argsort(all_data)
    labels = labels[od]
    
    # Find the breakpoints where the label changes (from -1 to 1)
    bkpts = np.where(labels[1:] != labels[:-1])[0]

    # Calculate segment statistics J_seg
    J_seg = np.concatenate([labels[bkpts], -labels[bkpts[-1]]])
    
    # Adjust J_seg based on the proportion of large and small groups
    J_seg[J_seg > 0] = J_seg[J_seg > 0] * n / M
    J_seg[J_seg < 0] = J_seg[J_seg < 0] * n / N

    # Calculate the segment boundaries, xx and yy, which are the normalized boundaries of the data
    xx = np.concatenate([np.array([0]), bkpts * delta])
    yy = np.concatenate([bkpts * delta, np.array([1])])

    # Calculate the inverse cumulative distribution function values for xx and yy (i.e., the quantiles of the normal distribution)
    invxx = norm.ppf(xx)
    invxx[0] = -1e5  # Handle boundary condition
    invyy = norm.ppf(yy)
    invyy[-1] = 1e5  # Handle boundary condition

    # Calculate the expected value (mu)
    mu = np.sum(J_seg * (-norm.pdf(invyy) + norm.pdf(invxx)))

    # Calculate A1, A2, and B for the standard deviation calculation
    A1 = 0
    f1invy = f1(invyy)  # Calculate f1 for invyy
    f1invx = f1(invxx)  # Calculate f1 for invxx
    for i in range(1, len(xx)):
        # Accumulate A1
        invyyf1J = f1invy[:i-1]
        invxxf1J = f1invx[:i-1]
        A1 += np.sum(J_seg[:i-1] * (invyyf1J - invxxf1J)) * J_seg[i] * (invyy[i] - invxx[i])

    # Calculate A2 and B
    A2 = np.sum(J_seg**2 * (f2(invyy) - f2(invxx) + (yy - xx) - (invyy - invxx) * f1invx))
    B = np.sum(J_seg * (f1invy - f1invx))

    # Calculate the total variance S_all
    S_all = 2 * np.sum(A1 + A2) - B**2
    
    # Calculate the standard deviation (sigma)
    sigma = np.sqrt(S_all) / np.sqrt(n)

    return mu, sigma


def f1(t):
    ''' 
    func:
        Calculates and returns the f1 function for a normal distribution.
    Args:
        t: A numerical value or an array of values.
    returns:
        f1(t): The corresponding values of the f1 function for a normal distribution.
    '''
    return t * norm.cdf(t) + norm.pdf(t)


def f2(t):
    ''' 
    func:
        Calculates and returns the f2 function for a normal distribution.
    Args:
        t: A numerical value or an array of values.
    returns:
        f2(t): The corresponding values of the f2 function for a normal distribution.
    '''
    return 0.5 * (t**2 * norm.cdf(t) - norm.cdf(t) + t * norm.pdf(t))


def regionGrowxx_3D(inputID, iters, lx, ly, lz, xxshift, yyshift):
    '''
    Function to perform 3D region growing by shifting points in 3D space.
    
    Args:
        inputID (np.array): Initial indices (list of points) for the region to grow.
        iters (int): Number of iterations to grow the region.
        lx (int): The size of the grid along the x-axis.
        ly (int): The size of the grid along the y-axis.
        lz (int): The size of the grid along the z-axis.
        xxshift (np.array): The shift values along the x-axis.
        yyshift (np.array): The shift values along the y-axis.
        
    Returns:
        np.array: The output IDs of the region after growing.
    '''
    # Convert input indices to x, y, z coordinates
    idX, idY, idZ = np.unravel_index(inputID, (lx, ly, lz))
    
    # Initialize zzshift as zeros, assuming no shift along the z-axis in this function
    zzshift = np.zeros_like(yyshift)
    
    for _ in range(iters):
        # Apply shifts and ensure indices are within bounds
        idX = np.clip(idX + xxshift, 0, lx - 1)  # Clip within the x-bound
        idY = np.clip(idY + yyshift, 0, ly - 1)  # Clip within the y-bound
        idZ = idZ + zzshift  # No shift along z-axis as per original code
        
        # Convert back to linear indices
        outputID = np.ravel_multi_index((idX, idY, idZ), (lx, ly, lz))
        
        # Get unique output IDs
        outputID = np.unique(outputID)
        
        # Update idX, idY, idZ for the next iteration
        idX, idY, idZ = np.unravel_index(outputID, (lx, ly, lz))
    
    return outputID

def synquant_org(input, minSZ, smoothingfactor, Mu, Sigma):
    ''' 
    Function to calculate a Z-map for an input 3D image using region growing and statistical analysis.
    
    Args:
        input: 3D input image (numpy array).
        minSZ: Minimum size of a region to be considered valid.
        smoothingfactor: The standard deviation for the Gaussian filter.
        Mu: Mean value used in statistical analysis.
        Sigma: Standard deviation used in statistical analysis.
    
    Returns:
        ZMap: The Z-score map for the input image.
    '''
    
    # Convert input to a double type and apply Gaussian smoothing
    imregion1 = np.array(input, dtype=float)
    imregion1G = gaussian_filter(imregion1, sigma=smoothingfactor)
    
    # Initialize variables for region growing
    iters1 = 1
    xxshift1 = np.zeros((2 * iters1 + 1, 2 * iters1 + 1))
    yyshift1 = np.zeros((2 * iters1 + 1, 2 * iters1 + 1))
    
    for i in range(-iters1, iters1 + 1):
        for j in range(-iters1, iters1 + 1):
            xxshift1[i + iters1, j + iters1] = i
            yyshift1[i + iters1, j + iters1] = j
    
    # Get the size of the image
    lenx, leny, lenz = imregion1G.shape
    ZMap = np.zeros_like(imregion1G)

    # Set thresholds for segmentation
    maxThres = 500
    minThres = 150
    stepThres = 2

    for i in range(maxThres, minThres - 1, -stepThres):
        print(f"Threshold: {i}")
        
        # Create binary mask based on threshold
        mask = imregion1G > i
        maskroi, num_features = label(mask)
        
        # Find connected components in the binary mask
        maskroiIDx = [np.where(maskroi == label_idx)[0] for label_idx in range(1, num_features + 1)]
        maskroiIDx = [idx for idx in maskroiIDx if len(idx) >= 10]  # Remove regions smaller than 10 pixels
        
        # Process each valid region
        for j in range(len(maskroiIDx)):
            idxtmp = maskroiIDx[j]
            N = len(idxtmp)
            if N > minSZ and N < 10000:
                idxtmpneiL1 = regionGrowxx_3D(idxtmp, smoothingfactor, lenx, leny, lenz, xxshift1, yyshift1)
                idxtmpnei = np.setdiff1d(idxtmpneiL1, idxtmp)  # Find neighboring points
                
                if idxtmpnei.size > 0:
                    # Extract signal values for current region and its neighbors
                    signal = imregion1[idxtmp]
                    signalNei = imregion1[idxtmpnei]
                    
                    # Calculate mean and sigma using the order statistics function
                    mutmp, sigmatmp = ksegments_orderstatistics_fin(signal, signalNei)
                    
                    # Calculate mean difference and z-score
                    meanDiff = np.mean(signal) - np.mean(signalNei)
                    zscore = (meanDiff - mutmp * Sigma) / (sigmatmp * Sigma)
                    
                    # Update ZMap with the calculated Z-score if it's larger
                    if zscore > np.max(ZMap[idxtmp]):
                        ZMap[idxtmp] = zscore
    
    return ZMap
