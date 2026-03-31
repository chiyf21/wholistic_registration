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

<img src="images/Different_map_in_one_image.png" alt="result" width="320" height="543">

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
### Parameter Explanation 

#### `overlap` (default = `0.5`)
This controls how much neighboring patches overlap with each other when the image is divided into small local regions. If `overlap` is larger, adjacent patches share more area, so the final depth estimation is usually smoother and more stable, because each pixel is influenced by more local measurements. If `overlap` is smaller, the computation is faster, but the patch grid becomes sparser and the result may look more blocky or less robust in difficult regions. Intuitively, this parameter controls how densely we sample the image with local patches.

#### `smooth_sigma` (default = `20.0`)
This is the Gaussian smoothing strength applied to the final dense z-map in the XY plane. A larger value means stronger spatial smoothing, so the final initialization becomes more spatially coherent and less noisy, but very large values may oversmooth real local structure. A smaller value keeps more local detail, but may also preserve noise or patch inconsistency. Intuitively, this parameter controls how smooth the final depth surface looks across the image.

#### `weight_eps` (default = `1e-6`)
This is a very small positive number added in divisions and normalizations to avoid numerical instability, especially division by zero. It does not change the method conceptually, but it makes the computation safe when local variance or weight sums are extremely small. Intuitively, this is just a numerical safeguard.

#### `z_curve_sigma` (default = `1.0`)
This controls how strongly the matching score curve along the z direction is smoothed before building the posterior. Each patch has a score curve across all candidate z slices, and this parameter reduces small jagged fluctuations in that curve. A larger value gives a smoother curve and suppresses noisy local spikes, but if it is too large it may blur away meaningful peaks. A smaller value keeps the original curve shape more faithfully, but may leave too much noise. Intuitively, this parameter controls how much we trust the overall trend of the z-score curve rather than tiny local fluctuations.

#### `posterior_beta` (default = `12.0`)
This is the inverse temperature used when converting the smoothed z-score curve into a posterior distribution over z. If `posterior_beta` is larger, the posterior becomes sharper and concentrates more strongly around the best-scoring z positions. If it is smaller, the posterior becomes flatter and spreads probability over a wider range of z. Intuitively, this parameter controls how “decisive” the algorithm is when turning matching scores into a soft depth estimate: large values behave more like a hard choice, while small values behave more cautiously.

#### `local_radius` (default = `3`)
This defines the radius around the dominant peak when measuring how much posterior mass is concentrated near the main mode. It is used to judge whether the z-curve has one main plausible region or whether the probability is spread across multiple separated depths. A larger radius is more tolerant and counts a broader neighborhood as belonging to the main peak. A smaller radius is stricter and only rewards very concentrated peaks. Intuitively, this parameter controls how locally we define “most of the probability is around one main depth”.

#### `smoothness_alpha` (default = `6.0`)
This controls how strongly rough or jagged z-score curves are penalized when computing confidence. If this value is larger, curves with strong second-order oscillation will lose confidence more aggressively. If it is smaller, even somewhat irregular curves will still retain confidence. Intuitively, this parameter controls how much the algorithm dislikes noisy, unstable z-curves.

#### `regularization_lambda` (default = `1.0`)
This is the strength of spatial regularization on the patch grid. After each patch gets its own soft depth estimate, neighboring patches are encouraged to agree with each other, especially when some patches are unreliable. A larger value means stronger spatial coupling, so low-confidence patches will follow neighboring reliable patches more strongly. A smaller value means each patch keeps its own estimate more independently. Intuitively, this parameter controls how much neighboring patches are encouraged to be consistent.

#### `regularization_iters` (default = `40`)
This is the number of iterations used in the patch-grid regularization step. More iterations allow the spatial regularization effect to propagate further across the patch grid, leading to a more stabilized patch-wise depth field. Fewer iterations make the process weaker and more local. Intuitively, this parameter controls how fully the neighboring patches are allowed to “communicate” and smooth each other.

#### `min_confidence` (default = `1e-3`)
This is the lower bound on patch confidence. Even if a patch looks very unreliable, its confidence will not drop below this value. This prevents completely zeroing out a patch and avoids numerical problems or overly extreme behavior during fusion and regularization. Intuitively, this parameter ensures that every patch still contributes at least a tiny amount, even when the algorithm is not confident about it.

#### Overall intuition

These parameters mainly affect three stages of the method. First, `overlap` and `smooth_sigma` control how the final dense z initialization is spatially sampled and smoothed. Second, `z_curve_sigma`, `posterior_beta`, `local_radius`, and `smoothness_alpha` control how the algorithm interprets the z-score curve for each patch, including how much it smooths the curve, how sharply it forms a posterior, and how it evaluates whether the curve is reliable. Third, `regularization_lambda`, `regularization_iters`, and `min_confidence` control how patch-wise depth estimates are regularized and fused together, especially in uncertain regions. In a very intuitive sense, these parameters together determine how much the algorithm trusts local evidence, how sharply it chooses depth, and how strongly it asks neighboring patches to agree.

## 2. How to escape from local optima

This is **not** limited to initialization — it runs at **every layer of the coarse-to-fine pyramid** inside `getMotion_v2` (enabled by `wrong_region_enable=True`). At each layer, the function `correct_wrong_regions_one_layer` wraps the normal Lucas-Kanade optimization in a three-pass procedure:

**Pass 1 — Normal optimization** (`optimize_layer_cross_resolution`): Run the standard LK iteration to convergence.

**Detect bad regions**: Compute the per-control-point residual error:

$$
r(X) = I_{\text{mov}}(X) - I_{\text{ref}}\bigl(\phi(X)\bigr)
$$

aggregate it over each control point's patch (`get_local_error_on_control_points`), then flag outlier control points using MAD (median absolute deviation) thresholding (`detect_significant_mad`). Connected components of flagged control points are expanded to a dense bad-region mask (`build_bad_region_mask_from_cp_error`).

Optionally (mode `"highresidual"`), within these bad regions only the voxels with the most significant residual reduction after iteration compared to the initial motion residual are selected — these are the "local attractors" where the optimization has locked onto a locally consistent but globally wrong match.

**Build reference-domain trap mask**: The bad-region voxels are projected into reference space via the current mapping $\phi$, and a mask is grown around them (spatially and by intensity/gradient similarity) using `build_reference_trap_mask_from_bad_moving`. This mask is applied to the **reference** domain because that is where the false correspondence lives.

**Pass 2 — Re-optimize with mask** (`optimize_layer_cross_resolution`): Rerun LK from the same initialization, but with the trap mask suppressing the bad reference regions. This forces the algorithm to find alternative correspondences.

**Pass 3 — Refine with original mask** (`optimize_layer_cross_resolution`): Take the motion from Pass 2 as initialization and rerun LK with the **original** (unmasked) reference. This lets the solution settle freely now that it has escaped the local optimum.

**Accept or reject**: Compare the total error (data + smoothness penalty) of Pass 3 vs Pass 1. Accept the corrected result only if it improves; otherwise keep the original.

### Where in the pipeline

- **Code**: `correct_wrong_regions_one_layer` in `calFlowCrossResolution.py` (line ~1456)
- **Called by**: `getMotion_v2` (line ~1779), which loops over pyramid layers coarse-to-fine
- **Integration**: `getMotion_v2` is currently used in `test_HR.ipynb`. The main production pipeline (`main_function.py` → `registration.py`) still calls the v1 `calFlow3d_Wei_v1.getMotion`, which does **not** include wrong-region correction.
### Parameter Explanation 

#### `error_metric` (`"mse"` or `"mae"`)
This specifies which error metric is used when computing the local residual/error on control points. The default value `"mse"` means mean squared error, which gives larger penalties to larger mismatches. 

#### `mad_threshold` (default = `3.0`)
This controls the threshold for detecting bad regions from the control-point error map using a robust MAD-based criterion. A larger value means the algorithm is more conservative and only marks very abnormal regions as bad. A smaller value means it is more aggressive and will flag more regions as suspicious. Intuitively, this parameter controls how “strict” the algorithm is when deciding that a region is problematic.

#### `min_component_size` (default = `2`)
After bad regions are detected, very small connected components can be removed as likely noise. This parameter sets the minimum size of a connected bad region that is allowed to remain. A larger value removes more tiny isolated detections, while a smaller value keeps even very small suspicious regions. Intuitively, this parameter controls whether the algorithm ignores tiny scattered artifacts or keeps them as possible wrong regions.

#### `bad_region_exclude_mode` (default = `"direct"`)
This determines how the detected bad regions are excluded before rerunning the optimization. In `"direct"` mode, all detected bad regions are excluded directly. In `"highresidual"` mode, the algorithm is more selective: it first detects bad regions, and then only keeps the highest-residual part inside those regions. Intuitively, this parameter controls whether the correction step is broad and simple, or more focused on the worst part of each problematic region.

#### `expand_radius_xy` (default = `2.0`)
This controls how much the detected problematic region is spatially expanded in the XY plane when constructing the trap mask in reference space. A larger value means the masked region becomes broader, so the algorithm blocks out a wider neighborhood around suspicious matches. A smaller value keeps the mask tighter and more localized. Intuitively, this parameter controls how much surrounding context is also treated as potentially unreliable once a bad region is found.

#### `sigma_grad` (default = `1.0`)
This controls the smoothing scale used when computing or using gradient-related information during trap-mask construction in reference space. A larger value makes the gradient information smoother and less sensitive to small local fluctuations, while a smaller value makes it more sensitive to local detail. Intuitively, this parameter controls how locally or globally the image structure is interpreted when deciding which reference regions should be masked out.

#### `intensity_k` (default = `2.0`)
This controls the strength of the intensity-based criterion when constructing the trap mask in reference space. A larger value generally means the intensity condition is stricter or more strongly emphasized, while a smaller value makes the intensity effect weaker. Intuitively, this parameter controls how strongly image intensity characteristics contribute to deciding whether a reference region looks like a potential trapping region.

#### `quantile_threshold_q` (default = `0.8`)
This parameter is used in the `"highresidual"` mode to keep only the highest-residual portion inside already detected bad regions. The default value `0.8` means the threshold is set at the 80th percentile, so only the most severe part of the residual values is retained. A larger value makes the selection more focused on only the worst points, while a smaller value includes a broader portion of the suspicious region. Intuitively, this parameter controls how concentrated the correction mask is inside each bad region.

#### Overall intuition
These parameters mainly control how the algorithm identifies suspicious regions, how aggressively it filters or refines them, and how it expands them into a trap mask in reference space. The first group, such as `error_metric`, `mad_threshold`, `min_component_size`, and `bad_region_exclude_mode`, determines how wrong regions are detected and refined. The second group, including `expand_radius_xy`, `sigma_grad`, `intensity_k`, and `quantile_threshold_q`, determines how those detected regions are turned into a mask in reference space that prevents the optimization from repeatedly falling into the same local trap.
 
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
