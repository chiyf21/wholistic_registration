#!/usr/bin/env python
"""
Educational Script: Vectorization Comparison
=============================================

This script compares two implementations of solving 2x2 linear systems
at each point in a grid:

    | a  b | | dx |   | -d_x |
    | b  c | | dy | = | -d_y |

Method 1: Nested Python loops (SLOW)
Method 2: Vectorized NumPy operations (FAST)

Both methods should produce identical results, but vectorized is ~100x faster.
"""

import numpy as np
import time

# =============================================================================
# TEST DATA GENERATION
# =============================================================================

def generate_test_data(height=100, width=100, seed=42):
    """Generate random test data for the 2x2 system solver."""
    np.random.seed(seed)
    
    # Simulate gradient products (Ixx, Ixy, Iyy) and temporal terms (Ixt, Iyt)
    Ixx_CP = np.random.rand(height, width) * 100 + 1  # Ensure positive
    Ixy_CP = np.random.rand(height, width) * 50
    Iyy_CP = np.random.rand(height, width) * 100 + 1  # Ensure positive
    Ixt_CP = np.random.rand(height, width) * 10 - 5   # Can be negative
    Iyt_CP = np.random.rand(height, width) * 10 - 5   # Can be negative
    
    # Add some near-singular cases to test the fallback
    # Make some determinants very small
    singular_mask = np.random.rand(height, width) < 0.05  # 5% singular
    Ixy_CP[singular_mask] = np.sqrt(Ixx_CP[singular_mask] * Iyy_CP[singular_mask]) * 0.9999
    
    return Ixx_CP, Ixy_CP, Iyy_CP, Ixt_CP, Iyt_CP


# =============================================================================
# METHOD 1: NESTED LOOP (ORIGINAL - SLOW)
# =============================================================================

def solve_2x2_loop(Ixx_CP, Ixy_CP, Iyy_CP, Ixt_CP, Iyt_CP):
    """
    Original implementation using nested Python loops.
    
    Solves the 2x2 system at each control point:
        | a  b | | dx |   | -d_x |
        | b  c | | dy | = | -d_y |
    
    Uses Cramer's rule for non-singular matrices,
    pseudo-inverse for near-singular matrices.
    """
    dx = np.zeros(Ixx_CP.shape)
    dy = np.zeros(Ixx_CP.shape)
    
    for i in range(Ixx_CP.shape[0]):
        for j in range(Ixx_CP.shape[1]):
            # Extract local values at the control point (i, j)
            a = Ixx_CP[i, j]
            b = Ixy_CP[i, j]
            c = Iyy_CP[i, j]
            d_x = Ixt_CP[i, j]
            d_y = Iyt_CP[i, j]
            
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
                # For a nearly singular matrix, use pseudo-inverse
                A_local = np.array([[a, b], [b, c]])
                d_local = np.array([-d_x, -d_y])
                sol = np.linalg.pinv(A_local) @ d_local
                dx[i, j] = sol[0]
                dy[i, j] = sol[1]
    
    return dx, dy


# =============================================================================
# METHOD 2: VECTORIZED (NEW - FAST)
# =============================================================================

def solve_2x2_vectorized(Ixx_CP, Ixy_CP, Iyy_CP, Ixt_CP, Iyt_CP):
    """
    Vectorized implementation using NumPy broadcasting.
    
    Same math as the loop version, but all operations are done
    on entire arrays at once.
    """
    # Rename for clarity (matching the loop variable names)
    a = Ixx_CP
    b = Ixy_CP
    c = Iyy_CP
    d_x = Ixt_CP
    d_y = Iyt_CP
    
    # Compute all determinants at once
    det = a * c - b * b
    
    # Initialize output arrays
    dx = np.zeros_like(a)
    dy = np.zeros_like(a)
    
    # Mask for non-singular matrices (|det| > 1e-6)
    non_singular = np.abs(det) > 1e-6
    
    # -------------------------------------------------------------------------
    # CASE 1: Non-singular matrices (use Cramer's rule)
    # -------------------------------------------------------------------------
    # Cramer's rule:
    #   dx = (-c * d_x + b * d_y) / det
    #   dy = (b * d_x - a * d_y) / det
    
    dx[non_singular] = (
        -c[non_singular] * d_x[non_singular] + b[non_singular] * d_y[non_singular]
    ) / det[non_singular]
    
    dy[non_singular] = (
        b[non_singular] * d_x[non_singular] - a[non_singular] * d_y[non_singular]
    ) / det[non_singular]
    
    # -------------------------------------------------------------------------
    # CASE 2: Near-singular matrices (use pseudo-inverse)
    # -------------------------------------------------------------------------
    singular = ~non_singular
    if np.any(singular):
        # Get indices of singular points
        singular_indices = np.where(singular)
        
        for idx in range(len(singular_indices[0])):
            i, j = singular_indices[0][idx], singular_indices[1][idx]
            A_local = np.array([[a[i, j], b[i, j]], 
                                [b[i, j], c[i, j]]])
            d_local = np.array([-d_x[i, j], -d_y[i, j]])
            sol = np.linalg.pinv(A_local) @ d_local
            dx[i, j] = sol[0]
            dy[i, j] = sol[1]
    
    return dx, dy


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    print("=" * 70)
    print("VECTORIZATION COMPARISON: 2x2 System Solver")
    print("=" * 70)
    
    # Test with different sizes
    sizes = [(50, 50), (100, 100), (200, 200), (500, 500)]
    
    for height, width in sizes:
        print(f"\n{'='*70}")
        print(f"Grid Size: {height} x {width} = {height * width:,} control points")
        print("=" * 70)
        
        # Generate test data
        Ixx_CP, Ixy_CP, Iyy_CP, Ixt_CP, Iyt_CP = generate_test_data(height, width)
        
        # Count singular points
        det = Ixx_CP * Iyy_CP - Ixy_CP ** 2
        n_singular = np.sum(np.abs(det) <= 1e-6)
        print(f"Singular points: {n_singular} ({100*n_singular/(height*width):.1f}%)")
        
        # ---------------------------------------------------------------------
        # METHOD 1: Loop
        # ---------------------------------------------------------------------
        print("\n[Method 1] Nested Loop...")
        start = time.time()
        dx_loop, dy_loop = solve_2x2_loop(Ixx_CP, Ixy_CP, Iyy_CP, Ixt_CP, Iyt_CP)
        time_loop = time.time() - start
        print(f"  Time: {time_loop:.4f} seconds")
        
        # ---------------------------------------------------------------------
        # METHOD 2: Vectorized
        # ---------------------------------------------------------------------
        print("\n[Method 2] Vectorized...")
        start = time.time()
        dx_vec, dy_vec = solve_2x2_vectorized(Ixx_CP, Ixy_CP, Iyy_CP, Ixt_CP, Iyt_CP)
        time_vec = time.time() - start
        print(f"  Time: {time_vec:.4f} seconds")
        
        # ---------------------------------------------------------------------
        # COMPARISON
        # ---------------------------------------------------------------------
        print("\n[Comparison]")
        
        # Check if results match
        dx_diff = np.abs(dx_loop - dx_vec)
        dy_diff = np.abs(dy_loop - dy_vec)
        max_diff = max(np.max(dx_diff), np.max(dy_diff))
        mean_diff = (np.mean(dx_diff) + np.mean(dy_diff)) / 2
        
        print(f"  Max difference:  {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        if max_diff < 1e-10:
            print(f"  ✅ Results MATCH perfectly!")
        elif max_diff < 1e-6:
            print(f"  ✅ Results match within numerical precision")
        else:
            print(f"  ⚠️  Results differ - check implementation!")
        
        # Speedup
        speedup = time_loop / time_vec if time_vec > 0 else float('inf')
        print(f"\n  🚀 Speedup: {speedup:.1f}x faster")
        
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The vectorized version is significantly faster because:
1. NumPy operations are implemented in C, not Python
2. No Python loop overhead (function calls, type checking)
3. Better CPU cache utilization (contiguous memory access)
4. Potential SIMD vectorization by the compiler

For large grids (500x500 = 250,000 points), the speedup can be 50-200x!
""")


if __name__ == "__main__":
    main()
