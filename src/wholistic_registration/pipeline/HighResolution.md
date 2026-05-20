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

## 1. How to get the initial mapping: Learning-based Coarse Registration(still training, haven't been used for the data) (NEW)

In slice-to-volume registration, we first need to find where each 2D moving slice sits along the $z$-axis of the 3D reference volume. A naive approach would pick the $z$-slice with the highest correlation globally — but this fails when the field of view contains multiple disconnected structures, each needing its own $z$-match. So instead, we estimate the initial $z$-position **per patch**.


<img src="images/Different_map_in_one_image.png" alt="result" width="320" height="543">


Before the iterative optimization stage, we first estimate a coarse correspondence between the sparse moving stack and the dense reference volume. The updated preprocessing stage consists of two steps. First, we estimate the initial axial locations of the moving slices using a global fixed-spacing ZNCC search. Second, we use a learning-based coarse matching model to predict a sparse 3D coordinate field on the moving control grid. The predicted coarse mapping is then used as the initialization for the subsequent iterative registration.

The overall coarse registration process is

$$
I^{mov}, I^{ref}
\rightarrow
\mathbf{z}^{init}
\rightarrow
\phi^{coarse},
$$

where $$I^{mov}$$ is the moving sparse stack, $$I^{ref}$$ is the reference volume, $$\mathbf{z}^{init}$$ denotes the initial reference z-indices of the moving slices, and $$\phi^{coarse}$$ is the coarse 3D coordinate field predicted by the model.

---

### Global Fixed-spacing Z Initialization

The moving image is a sparse stack with $$K$$ slices, while the reference image is a dense 3D volume with $$Z$$ slices. Since the moving slices are acquired with approximately fixed axial spacing, their corresponding reference z-locations can be represented as

$$
z_k = z_0 + k \Delta z_{ref} d,
$$

where $$z_0$$ is the unknown starting reference slice, $$\Delta z_{ref}$$ is the fixed spacing in the reference z-index system, and $$d$$ is the z-direction.

To estimate $$z_0$$, we compute a global ZNCC score between each moving slice and each reference slice. By default, the ZNCC is computed on the 2D gradient magnitude images rather than raw intensities, making the initialization more sensitive to structural similarity and less sensitive to absolute intensity changes. A Hann window is used as a spatial weight to reduce the influence of boundary artifacts.

Let $$S(k,z)$$ be the weighted ZNCC score between the $$k$$-th moving slice and the $$z$$-th reference slice. We enumerate all valid starting positions $$z_0$$ and maximize the total fixed-spacing alignment score:

$$
z_0^*
=
\arg\max_{z_0}
\sum_{k=0}^{K-1}
S
\left(
k,
\operatorname{round}
\left(
z_0 + k \Delta z_{ref} d
\right)
\right).
$$

The resulting initial z positions are

$$
z_k^{init}
=
z_0^* + k \Delta z_{ref} d.
$$

This step produces a globally consistent axial initialization for the entire sparse stack, instead of matching each moving slice independently.

---

### Learning-based Coarse Matching Model

After obtaining $$\mathbf{z}^{init}$$, we use a learning-based coarse matching model to estimate the 3D reference coordinate of each moving control point. The model takes three inputs:

$$
I^{mov}, \quad I^{ref}, \quad \mathbf{z}^{init},
$$

and outputs a coarse coordinate field

$$
\phi^{coarse}(k,i,j)
=
\left(
z^{ref}_{kij},
y^{ref}_{kij},
x^{ref}_{kij}
\right),
$$

where $$k$$ indexes the moving slice and $$(i,j)$$ indexes the spatial control grid on that slice.

The model does not perform dense full-volume matching. Instead, it uses the initial z positions and the moving control-grid coordinates to define an initial reference coordinate for each query:

$$
\mathbf{c}_0(k,i,j)
=
\left(
z_k^{init},
y_{ij},
x_{ij}
\right).
$$

For each query, the model searches only within a local 3D reference window centered at $$\mathbf{c}_0$$ or at the coordinate predicted by the previous refinement iteration. This design avoids the prohibitive memory cost of full global matching over the entire reference volume.

---

## Model Architecture

The coarse matching model contains three main components: a moving encoder, a reference encoder, and a local query-to-volume matching module.

---

### Moving Encoder

The moving encoder extracts features from the sparse moving stack. Each moving slice is first processed by a 2D convolutional feature extractor. The extracted slice-wise features are then enhanced by window-based self-attention. This allows each moving slice to aggregate local contextual information while keeping the computation manageable.

The output of the moving encoder is a set of moving feature maps at a reduced spatial resolution. Control-grid query features are then sampled or pooled from these moving features. Each query represents a control point on the moving image and will later be matched against a local candidate region in the reference volume.

The moving encoder therefore serves two purposes:

1. It extracts discriminative local image features from each moving slice.
2. It provides query tokens for the subsequent local matching stage.

---

### Reference Encoder

The reference encoder extracts a 3D feature volume from the dense reference stack. The reference image is encoded into a lower-resolution feature representation in the XY dimensions, while the z dimension is preserved to maintain axial localization accuracy.

After convolutional feature extraction, the reference feature volume is further enhanced by window-based 3D self-attention. This allows local volumetric context to be incorporated into the reference representation without requiring global attention over the entire volume.

The reference encoder provides the memory features used during query-to-volume matching. For each moving query, local candidate features are sampled from this reference feature volume around the current estimated reference coordinate.

The reference encoder therefore serves two purposes:

1. It transforms the dense reference volume into a compact and informative 3D feature memory.
2. It provides local candidate features for each moving query during matching.

---

### Local Query-to-volume Matching

The matching module estimates the reference coordinate of each moving query. Given a query feature and its current reference coordinate estimate, the model samples a local 3D candidate window from the reference feature volume.

Let the local candidate coordinates be

$$
\left\{
\mathbf{c}_{m}
\right\}_{m=1}^{M},
$$

where $$M$$ is the number of candidate points in the local search window. The matcher computes a score $$s_m$$ for each candidate using feature similarity, learned pairwise matching, relative offset encoding, and local cross-attention between the moving query and the sampled reference candidates.

The candidate scores are converted into a probability distribution:

$$
p_m
=
\frac{
\exp(s_m)
}{
\sum_{m'=1}^{M}
\exp(s_{m'})
}.
$$

The predicted coordinate is then obtained as the expectation over the local candidate coordinates:

$$
\hat{\mathbf{c}}
=
\sum_{m=1}^{M}
p_m \mathbf{c}_m.
$$

This local matching process can be repeated for multiple refinement iterations. At each iteration, the previous predicted coordinate becomes the new local search center. In this way, the model progressively refines the coarse correspondence while keeping the search local and memory-efficient.

---

## Coarse-to-fine Registration Pipeline

The final output of the learning-based model is the coarse coordinate field

$$
\phi^{coarse}(k,i,j)
=
\hat{\mathbf{c}}_{kij}.
$$

This coarse field is used as the initial mapping for the subsequent iterative optimization stage:

$$
\phi^0 = \phi^{coarse}.
$$

The complete pipeline is therefore

$$
I^{mov}, I^{ref}
\rightarrow
\mathbf{z}^{init}
\rightarrow
\phi^{coarse}
\rightarrow
\phi^{refined}
\rightarrow
\phi^{final}.
$$

The global fixed-spacing ZNCC provides a robust axial initialization, while the learning-based coarse model predicts a spatially varying 3D correspondence field. This initialization substantially reduces the search space for the following iterative registration and makes the subsequent refinement less sensitive to local minima.

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
 
# Pipeline

The goal of the long-term registration pipeline is to repeatedly match pixels between a moving image sequence and a reference image. In our data, the reference image and the moving images do not always have the same intensity distribution, even when they describe the same biological structures. Therefore, direct intensity comparison is not always reliable. To make the registration more stable, we first adjust the reference image to better match the intensity distribution of the current moving image, and then perform pixel-level registration.

The overall pipeline contains five main steps:

1. split the input data into registration and biological-marker channels;
2. estimate the initial axial correspondence between the sparse moving stack and the reference stack;
3. adjust the reference intensity distribution to match the moving image;
4. register each time point to the adjusted reference;
5. apply the estimated mapping to the biological-marker channel and save the registered result.


## 1. Initial Z-position estimation(compute once)

The moving image is a sparse z-stack, while the reference image is a denser 3D stack. Before estimating the full spatial deformation, we first identify which reference slices correspond to the moving slices.

Because the moving slices are acquired with a fixed spacing, we do not match each slice independently. Instead, we search for a globally consistent starting z-position. Once the first moving slice is assigned to a reference slice, the remaining moving slices are placed according to the known spacing.

In simple terms, this step answers the question:

> Where is this sparse moving stack located inside the dense reference stack?

The output of this step is a vector of initial z positions:

$$
\mathbf{z}^{init}
=
\left(
z_0^{init},
z_1^{init},
\ldots,
z_{K-1}^{init}
\right).
$$

These positions provide the initial 3D coordinate system for the following registration.

---

## 2. Reference intensity adjustment (most important)

A key observation in our long-term data is that the reference image and the moving images can have different pixel intensity distributions. This difference may come from imaging condition changes, signal attenuation, background variation, or biological changes over time.

If we directly compare the raw reference image with the moving image, the optimizer may interpret intensity differences as spatial mismatch. To reduce this problem, we adjust the reference intensity distribution before registration.

Specifically, we compare the reference slices corresponding to the initial z positions with the current moving stack. We then learn a quantile-based intensity mapping from the reference image to the moving image. This mapping aligns the intensity statistics of the reference with those of the moving image.

Intuitively, this step makes the reference image look more like the moving image in terms of pixel intensity, while keeping the reference geometry unchanged.

After this adjustment, we obtain an intensity-adapted reference image:

$$
I^{ref}_{adj}.
$$

This adjusted reference is used for registration instead of the original reference image.

---

## 4. Frame-by-frame registration

After the reference intensity adjustment, we register each moving frame to the adjusted reference image.

For each time point, the pipeline performs the following operations:

1. extract the current moving registration channel;
2. build a foreground mask for the moving image;
3. estimate the spatial mapping between the moving image and the adjusted reference;
4. use the current mapping as a motion prior for the next frame.

The registration estimates a mapping field:

$$
\phi_t(x,y,k)
=
\left(
x^{ref},
y^{ref},
z^{ref}
\right),
$$

which describes where each pixel in the moving sparse stack should be located in the reference coordinate system.

The mapped reference image is then sampled using this mapping and compared with the current moving image. This produces the registered image for the current time point.

Because the data are time-lapse images, adjacent frames usually have related motion. Therefore, after each frame is registered, part of the estimated motion is passed to the next frame as an initialization. This makes the registration more stable across time and reduces frame-to-frame inconsistency.

---

## 5. Applying the mapping to the biological-marker channel

The deformation field is estimated using the registration channel, but the same spatial mapping can be applied to other channels.

After the mapping $$\phi_t$$ is estimated, we apply it to the biological-marker reference channel. This produces a registered biological-marker image in the same coordinate system as the current moving frame.

This is important because the biological-marker channel may be sparse, discontinuous, or less suitable for direct registration. By estimating the deformation from a more structural channel and applying it to the marker channel, we obtain a more reliable registered marker signal.

For each frame, we save four images for inspection:

1. the raw moving registration channel;
2. the mapped registration channel;
3. the raw moving biological-marker channel;
4. the mapped biological-marker channel.

This allows us to visually compare both the registration quality and the transformed biological signal.

---

## 6. Periodic reference intensity update

During long-term imaging, the intensity distribution of the moving image may gradually change over time. Therefore, a single reference intensity adjustment at the first frame may not remain optimal for the entire sequence.

To handle this, the pipeline periodically updates the reference intensity mapping. For example, after every fixed number of frames, the reference intensity distribution is re-mapped using either the current moving frame or the average of several recent frames.

This update does not change the reference geometry. It only updates the intensity relationship between the reference image and the moving image.

The purpose is to keep the registration comparison fair over time:

> the optimizer should focus on spatial mismatch, not on artificial intensity mismatch.

---

## Summary

The long-term registration pipeline repeatedly performs pixel matching between each moving frame and a reference image. Since the reference and moving images may have different intensity distributions, we first adjust the reference intensity to better match the moving image. Then we estimate a spatial mapping using the structural registration channel and apply the same mapping to the biological-marker channel.

The complete process can be summarized as:

$$
I^{mov}_t
\rightarrow
I^{ref}_{adj}
\rightarrow
\phi_t
\rightarrow
I^{mapped}_t.
$$

Here, $$I^{mov}_t$$ is the moving image at time point $$t$$, $$I^{ref}_{adj}$$ is the intensity-adjusted reference image, $$\phi_t$$ is the estimated spatial mapping, and $$I^{mapped}_t$$ is the registered output.

This design makes the pipeline robust to both spatial motion and intensity distribution changes, which are common in long-term biological imaging data.


# Remaining Questions

