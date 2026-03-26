## WHOLISTIC Registration Pipeline (High Resolution)

# Remodel

Instead of searching for a motion field to warp the moving image, we find a mapping $\phi$ from each voxel in the moving image to a coordinate in the reference image. The reference is treated as a continuous function (via interpolation), so we can sample it at any coordinate.

## Setup

- $I_{\text{mov}}(X)$: moving image intensity at voxel $X$
- $I_{\text{ref}}(Y)$: reference image intensity at coordinate $Y$ (continuous via interpolation)
- $\phi(X)$: mapping — for each moving-image voxel $X$, returns the coordinate in the reference where it corresponds

The loss function has two terms:

$$
L = \underbrace{\sum_X \bigl[I_{\text{ref}}\!\bigl(\phi(X)\bigr) - I_{\text{mov}}(X)\bigr]^2}_{\text{data term}} + \underbrace{\lambda_s \sum_i \|\mathbf{m}_i - \overline{\mathbf{m}}_i\|^2}_{\text{smoothness}} \tag{1}
$$

- **Data term**: summed over all voxels $X$. The reference image $I_{\text{ref}}$ is fixed — only the sampling coordinates $\phi(X)$ change during optimization.
- **Smoothness term**: summed over control points $i$. Each control point's motion $\mathbf{m}_i$ is penalized for deviating from its neighbors' average $\overline{\mathbf{m}}_i$.

## Decomposition

The mapping is:

$$
\phi(X) = \phi_0(X) + \mathbf{m}(X) \tag{2}
$$

- $\phi_0$: initial mapping (identity, or from the init-phase estimator)
- $\mathbf{m}$: motion field we optimize. Can be initialized from a previous time frame's estimate (the "accumulated" displacement)

## Iterative optimization

We optimize $\mathbf{m}$ iteratively. At each iteration, given current motion $\mathbf{m}$ and mapping $\phi = \phi_0 + \mathbf{m}$, we seek a small update $\delta\mathbf{m}$.

The new reference sample would be $I_{\text{ref}}\!\bigl(\phi(X) + \delta\mathbf{m}\bigr)$. The reference image is fixed, but we're evaluating it at a shifted location, so we Taylor-expand in the sampling coordinate:

$$
I_{\text{ref}}\!\bigl(\phi(X) + \delta\mathbf{m}\bigr) \approx I_{\text{ref}}\!\bigl(\phi(X)\bigr) + \mathbf{g}(X) \cdot \delta\mathbf{m} \tag{3}
$$

where $\mathbf{g}(X) = \nabla I_{\text{ref}}\big|_{\phi(X)} = \bigl[\frac{\partial I_{\text{ref}}}{\partial x},\, \frac{\partial I_{\text{ref}}}{\partial y},\, \frac{\partial I_{\text{ref}}}{\partial z}\bigr]$ is the spatial gradient of the reference evaluated at $\phi(X)$. (In the code: `getSpatialGradientInOrgGrid(data_ref, phase_new)`)

Define the residual at the current iterate:

$$
I_t(X) = I_{\text{mov}}(X) - I_{\text{ref}}\!\bigl(\phi(X)\bigr) \tag{4}
$$

(In the code: `residual = data_mov - data_ref_sampled`)

Now substitute the Taylor expansion (3) into the data term of (1). The data term after the update is:

$$
\bigl[I_{\text{ref}}\!\bigl(\phi(X) + \delta\mathbf{m}\bigr) - I_{\text{mov}}(X)\bigr]^2
\approx \bigl[\underbrace{I_{\text{ref}}\!\bigl(\phi(X)\bigr) - I_{\text{mov}}(X)}_{= -I_t(X)} + \mathbf{g}(X) \cdot \delta\mathbf{m}\bigr]^2
= \bigl[\mathbf{g}(X) \cdot \delta\mathbf{m} - I_t(X)\bigr]^2 \tag{5}
$$

So $I_t$ captures the current mismatch, and $\mathbf{g} \cdot \delta\mathbf{m}$ is how much the update reduces it.

## Normal equations at control points

The update $\delta\mathbf{m}$ is solved at control points, not at every voxel:

1. Compute per-voxel spatial gradients $\mathbf{g}(X)$ on the reference, evaluated at the current deformed coordinates (`getSpatialGradientInOrgGrid`).
2. Compute the per-voxel residual $I_t(X) = I_{\text{mov}}(X) - I_{\text{ref}}(\phi(X))$ and **smooth it** with a $3 \times 3$ mean filter.
3. Form per-voxel products ($g_x^2$, $g_x g_y$, $g_x I_t$, etc.) and **sum** them over a $(2r{+}1) \times (2r{+}1)$ patch using a box filter of ones (a sum, not an average — the kernel `AverageFilter` is `cp.ones`, not normalized).
4. Sample the summed results at the control-point grid to get $\mathbf{S}_i$ and $\mathbf{b}_i$.

In short: voxel-wise gradients and smoothed residuals are combined into outer products, patch-summed, and sampled at control points to form the per-control-point linear system.

where:

- $\mathbf{m}_i$ — the current motion vector at control point $i$ (before the update).
- $\delta\mathbf{m}_i$ — the update we are solving for, so the new motion will be $\mathbf{m}_i + \delta\mathbf{m}_i$.
- $\overline{\mathbf{m}}_i$ — the **mean of the motion vectors of the neighboring control points** of $i$ (excluding $i$ itself). In code: `getNeiDiff` builds a filter with $\frac{1}{(2r{+}1)^2 - 1}$ for all neighbors and $-1$ at the center, so `neiDiff` $= \overline{\mathbf{m}}_i - \mathbf{m}_i$.

The smoothness penalty $\|\mathbf{m}_i + \delta\mathbf{m}_i - \overline{\mathbf{m}}_i\|^2$ pulls the updated motion toward its neighbors' average.

The objective for control point $i$ is:

$$
E_i = \sum_{X \in \text{patch}_i} \bigl[\mathbf{g}(X) \cdot \delta\mathbf{m}_i - I_t(X)\bigr]^2 + \lambda_s \|\mathbf{m}_i + \delta\mathbf{m}_i - \overline{\mathbf{m}}_i\|^2 \tag{6}
$$

Define the $3 \times 3$ **structure tensor** and the $3 \times 1$ **gradient-residual vector** for patch $i$:

$$
\mathbf{S}_i = \sum_{X \in \text{patch}_i} \mathbf{g}(X)\,\mathbf{g}(X)^T \in \mathbb{R}^{3 \times 3}, \qquad \mathbf{b}_i = \sum_{X \in \text{patch}_i} \mathbf{g}(X)\,I_t(X) \in \mathbb{R}^{3} \tag{7}
$$

where $\mathbf{g}(X) = (g_x, g_y, g_z)^T$ is the $3 \times 1$ spatial gradient, so $\mathbf{g}\,\mathbf{g}^T$ is the $3 \times 3$ outer product and the patch sum pools $(2r{+}1)^2$ such outer products into a single matrix. Explicitly:

$$
\mathbf{S}_i = \begin{pmatrix} \sum g_x^2 & \sum g_x g_y & \sum g_x g_z \\ \sum g_x g_y & \sum g_y^2 & \sum g_y g_z \\ \sum g_x g_z & \sum g_y g_z & \sum g_z^2 \end{pmatrix}, \qquad \mathbf{b}_i = \begin{pmatrix} \sum g_x\, I_t \\ \sum g_y\, I_t \\ \sum g_z\, I_t \end{pmatrix}
$$

(In the code: `Ixx, Ixy, Ixz, Iyy, Iyz, Izz` are the 6 unique entries of $\mathbf{S}$; `Ixt, Iyt, Izt` are the entries of $\mathbf{b}$. Note the confusing name `AverageFilter` — it sums, not averages.)

Setting $\frac{\partial E_i}{\partial\,\delta\mathbf{m}_i} = 0$ gives:

$$
\bigl(\mathbf{S}_i + \lambda_s\,\mathbf{I}\bigr)\;\delta\mathbf{m}_i = \mathbf{b}_i + \lambda_s\bigl(\overline{\mathbf{m}}_i - \mathbf{m}_i\bigr) \tag{8}
$$

This is a $3 \times 3$ linear system solved by Cramer's rule. (In the code: `getFlow3_withPenalty6`)

After solving at all control points, $\delta\mathbf{m}$ is interpolated to the dense grid and added: $\mathbf{m} \leftarrow \mathbf{m} + \delta\mathbf{m}$.


# Simulation

We build motion fields in a more continuous and biophysically motivated way. The method first generates dense random fields in the lateral directions ($x$ and $y$), then smooths them with anisotropic Gaussian filtering to obtain spatially coherent lateral deformations. The axial motion ($z$) is not generated independently; instead, it is derived from the depth-wise gradients of the lateral motion fields. This couples the $z$-direction displacement to the variation of lateral deformation across depth, making the simulated motion more consistent with realistic volumetric tissue deformation. The amplitudes of lateral and axial motion are normalized separately, allowing independent control of in-plane and out-of-plane motion strength.

Previously, we randomly selected some control points and generated motion for them, then smoothed the motion field. This caused extreme motion in some places that cannot be seen in real bio-images.

We did 10 repeats with various motion smoothness scale, deformation amplitude, and noise level. Here is the result:

![result](images/Simulation.png)

# Pre-process and post-process

## 1. How to get the initial mapping

In slice-to-volume registration, we first need to find where each 2D moving slice sits along the $z$-axis of the 3D reference volume. A naive approach would pick the $z$-slice with the highest correlation globally — but this fails when the field of view contains multiple disconnected structures, each needing its own $z$-match. So instead, we estimate the initial $z$-position **per patch**.

For each patch, rather than just picking the single best-matching $z$-slice, we use the full similarity curve across all candidate $z$-positions as a soft probability distribution.

![result](images/Different_map_in_one_image.png)

### Weighted ZNCC

For each patch $i$ (of size $P \times P$), we compute its similarity to every $z$-slice in the reference volume using **weighted zero-normalized cross-correlation (ZNCC)**.

Let $M(x,y)$ be the moving patch and $R(x,y,z)$ be the reference patch at candidate depth $z$. A Hann window $w(x,y)$ down-weights patch edges. Define the weighted means:

$$
\bar{M} = \frac{\sum_{x,y} w(x,y)\, M(x,y)}{\sum_{x,y} w(x,y)}, \qquad \bar{R}(z) = \frac{\sum_{x,y} w(x,y)\, R(x,y,z)}{\sum_{x,y} w(x,y)}
$$

and the centered versions $\tilde{M} = M - \bar{M}$, $\tilde{R}(z) = R(z) - \bar{R}(z)$. The ZNCC score at depth $z$ is:

$$
s(z) = \frac{\sum_{x,y} w\, \tilde{M}\, \tilde{R}(z)}{\sqrt{\sum_{x,y} w\, \tilde{M}^2} \;\sqrt{\sum_{x,y} w\, \tilde{R}(z)^2} + \epsilon}
$$

This gives $s(z) \in [-1, 1]$ for each candidate $z$, producing a similarity curve per patch. The score is optionally smoothed along $z$ with a 1D Gaussian ($\sigma_z$) to suppress jagged noise.

### Softmax posterior and depth estimate

Rather than picking the $z$ with the highest score (hard argmax), we convert the curve into a soft probability distribution (softmax):

$$
p(z) = \frac{\exp\bigl(\beta \cdot s(z)\bigr)}{\sum_{z'} \exp\bigl(\beta \cdot s(z')\bigr)}
$$

where $\beta$ (inverse temperature) controls sharpness: large $\beta$ concentrates mass on the peak (approaching argmax), small $\beta$ spreads it out. The patch's estimated depth is the **posterior mean**:

$$
\mu = \sum_z z \cdot p(z)
$$

This is a **soft argmax** — it returns a continuous $z$-estimate that naturally handles broad or multi-modal peaks without unstable jumps.

### Confidence

In addition to depth, we define a per-patch confidence $C$ that captures how reliable the matching information is. It has two components.

<!-- FIX: all equations below were originally plain text mixed with partial LaTeX; rewritten as proper LaTeX -->
**Evidence strength** — how much the best match stands out above the baseline:

$$
C_{\text{evidence}} = \frac{\max_z s(z) - \text{median}_z\, s(z)}{1 - \text{median}_z\, s(z) + \varepsilon}
$$

**Shape quality** — composed of two factors:

1. *Local posterior mass* around the dominant peak $z_{\text{peak}}$:

$$
C_{\text{local}} = \sum_{|z - z_{\text{peak}}| \le r} p(z)
$$

This is high when the distribution is concentrated in a contiguous region (sharp or broad peak), and low when it is split across distant peaks.

2. *Smoothness quality*, penalizing high-frequency fluctuations measured by second-order differences $R$:

$$
C_{\text{smooth}} = \exp(-\alpha \cdot R)
$$

These combine as:

$$
C_{\text{shape}} = 0.5 \cdot C_{\text{local}} + 0.5 \cdot C_{\text{smooth}}
$$

$$
C = \text{clip}\bigl(C_{\text{evidence}} \cdot C_{\text{shape}},\; 0,\; 1\bigr)
$$

This ensures a patch is considered reliable only if it has both strong evidence and a well-structured curve:

- Sharp, high peak → high confidence
- Smooth, broad peak → still high confidence
- Multiple competing peaks or noisy fluctuations → lower confidence
- Uniformly low scores → very low confidence

### Spatial regularization

After obtaining a depth estimate $\mu$ and confidence $C$ for each patch, we enforce spatial consistency by solving a regularized optimization problem:

<!-- FIX: original had plain text "min_{z_ij} sum_{i,j} C_ij * (z_ij - μ_ij)^2 + ..." instead of LaTeX -->
$$
\min_{z_{ij}} \sum_{i,j} C_{ij} \bigl(z_{ij} - \mu_{ij}\bigr)^2 + \lambda \sum_{(i,j) \sim (k,l)} \bigl(z_{ij} - z_{kl}\bigr)^2
$$

The first term keeps high-confidence patches close to their own estimates; the second encourages neighboring patches to agree. This allows unreliable patches to be guided by more reliable neighbors, producing a globally consistent depth map.

Overall, this framework replaces a brittle "winner-takes-all" strategy with a probabilistic and context-aware approach, making it significantly more robust to noise, ambiguous matches, and biological variability in the data.

## 2. How to escape from local optima

This is **not** limited to initialization — it runs at **every layer of the coarse-to-fine pyramid** inside `getMotion_v2` (enabled by `wrong_region_enable=True`). At each layer, the function `correct_wrong_regions_one_layer` wraps the normal Lucas-Kanade optimization in a three-pass procedure:

**Pass 1 — Normal optimization** (`optimize_layer_cross_resolution`): Run the standard LK iteration to convergence.

**Detect bad regions**: Compute the per-control-point residual error:

$$
r(X) = I_{\text{mov}}(X) - I_{\text{ref}}\bigl(\phi(X)\bigr)
$$

aggregate it over each control point's patch (`get_local_error_on_control_points`), then flag outlier control points using MAD (median absolute deviation) thresholding (`detect_significant_mad`). Connected components of flagged control points are expanded to a dense bad-region mask (`build_bad_region_mask_from_cp_error`).

Optionally (mode `"highresidual"`), within these bad regions only the voxels with the *lowest* residual are selected — these are the "local attractors" where the optimization has locked onto a locally consistent but globally wrong match.

**Build reference-domain trap mask**: The bad-region voxels are projected into reference space via the current mapping $\phi$, and a mask is grown around them (spatially and by intensity/gradient similarity) using `build_reference_trap_mask_from_bad_moving`. This mask is applied to the **reference** domain because that is where the false correspondence lives.

**Pass 2 — Re-optimize with mask** (`optimize_layer_cross_resolution`): Rerun LK from the same initialization, but with the trap mask suppressing the bad reference regions. This forces the algorithm to find alternative correspondences.

**Pass 3 — Refine with original mask** (`optimize_layer_cross_resolution`): Take the motion from Pass 2 as initialization and rerun LK with the **original** (unmasked) reference. This lets the solution settle freely now that it has escaped the local optimum.

**Accept or reject**: Compare the total error (data + smoothness penalty) of Pass 3 vs Pass 1. Accept the corrected result only if it improves; otherwise keep the original.

### Where in the pipeline

- **Code**: `correct_wrong_regions_one_layer` in `calFlowCrossResolution.py` (line ~1456)
- **Called by**: `getMotion_v2` (line ~1779), which loops over pyramid layers coarse-to-fine
- **Integration**: `getMotion_v2` is currently used in `test_HR.ipynb`. The main production pipeline (`main_function.py` → `registration.py`) still calls the v1 `calFlow3d_Wei_v1.getMotion`, which does **not** include wrong-region correction.

# Remaining Questions

## 1. What is the true zRatio of the image?

From my current experiments, setting `zRatio=1` yields good results. The true zRatio values I read from the annotation and raw data are different (and the $x$ and $y$ ratios also seem to differ; did I read them incorrectly?).

## 2. The distribution is quite different, and I'm not sure what the exact reason is.

Now we use histogram mapping to correct the pixels, but it depends on whether we have two sample-shape images. Or we just use a simple transform from the same plane to correct all the pixels.

![result](images/Histgram.png)

## 3. What I find strange from a biological perspective

I'm now unable to find correspondences between the two images for some structures. We achieve relatively good results on dorsal and ventral data because there isn't significant motion in those datasets. However, it has become challenging for us to properly register all frames for the gut data.

In my observation, the last three frames of the gut/raw data (slice 79) look quite similar. Currently, we use the 3rd frame as the reference, and we can register the 4th and 5th frames very well, but the first two frames cannot be properly aligned.

I think the main issue is that the first two frames appear to contain an extra structure, as I circled in the figure. The last three frames all show a structure like two fingers pressed together. In contrast, the first frame shows a closed structure, which then breaks in the second frame, and splits into two lobes in the last three frames.

![result](images/5frames_S79_gut_raw.png)

From other slices, it seems that in the later frames, this tissue has moved to a very superior position (e.g., slice 69 of the 3rd frame). Given such a large deformation, I believe it will be very difficult for us to handle this registration.

<img src="images/frame3_S69_gut_raw.png" alt="result" width="269" height="543">
