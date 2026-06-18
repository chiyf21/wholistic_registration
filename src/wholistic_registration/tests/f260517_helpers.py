"""Helper functions for the F260517 cross-resolution registration test pipeline.

Split out of test_F260517_v2.py so they can be imported and hot-reloaded:

    import f260517_helpers as fh
    from importlib import reload
    reload(fh)            # picks up edits without restarting the kernel

These are the pure / reusable pieces (IO, intensity mapping, z estimation,
supersurface upsampling, projection, diagnostics). The run-specific orchestration
(process_single_frame, the warmup/forward loops) stays in test_F260517_v2.py
because it closes over the loaded data and the `option` dict.
"""

import os

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import distance_transform_edt, map_coordinates

from wholistic_registration.utils import IO
from wholistic_registration.utils import calFlowCrossResolution
from wholistic_registration.utils import preprocess as prep
from wholistic_registration.utils import mask


def read_ome_tiff_timepoints(tiff_path, n_timepoints=None):
    """Read an OME-TIFF, optionally only the first `n_timepoints` along axis 0.

    The moving (exp) file is ~15 GB; loading every frame just to register a few
    is wasteful and can blow up RAM. `tifffile.aszarr()` exposes the stack as a
    lazy zarr array, so slicing `[:n]` reads only those pages off disk.

    n_timepoints=None loads the whole stack (equivalent to IO.readTiff).

    Returns (array, ome_metadata_str).
    """
    with tifffile.TiffFile(tiff_path) as tif:
        desc = tif.pages[0].tags.get("ImageDescription")
        desc = desc.value if desc is not None else None

        if n_timepoints is None:
            return tif.asarray(), desc

        store = tif.aszarr()
        try:
            z = zarr.open(store, mode="r")
            n = min(int(n_timepoints), z.shape[0])
            sub = np.asarray(z[:n])
        finally:
            store.close()

    return sub, desc


def compute_zinit_offset_curve(zinit_debug):
    """Reconstruct the global match-vs-z-offset curve that
    FindInitZ_stack_global_fixed_spacing maximises, from its return_debug output.

    The search slides the whole rigidly-spaced moving stack through the reference
    by one integer offset z0 and scores each position as the summed ZNCC of every
    moving slice k placed at reference plane round(z0 + k*delta*direction). The
    curve is that summed score as a function of z0 -- a sharp single peak means a
    confident z initialisation; a flat / multi-peaked curve means it is poorly
    constrained (likely driven by noise).

    Returns
    -------
    z0_grid : (M,) int        candidate starting offsets
    total   : (M,) float      summed ZNCC score at each z0 (the curve to inspect)
    per_k_z : (M, K) int      the reference plane each slice k lands on per z0
    """
    scores = np.asarray(zinit_debug["scores"], dtype=np.float64)  # (K, Z)
    K, Z = scores.shape
    delta = float(zinit_debug["delta_ref_idx"])
    direction = int(zinit_debug["direction"])

    offsets = np.arange(K, dtype=np.float64) * delta * direction
    z0_min = int(np.ceil(-offsets.min()))
    z0_max = int(np.floor((Z - 1) - offsets.max()))
    z0_grid = np.arange(z0_min, z0_max + 1, dtype=np.int64)

    per_k_z = np.rint(z0_grid[:, None] + offsets[None, :]).astype(np.int64)  # (M, K)
    per_k_z = np.clip(per_k_z, 0, Z - 1)
    total = scores[np.arange(K)[None, :], per_k_z].sum(axis=1)  # (M,)

    return z0_grid, total, per_k_z


def summarize_zinit_peakiness(z0_grid, total, best_z0):
    """One-number sanity metrics for how peaked the z-init curve is."""
    total = np.asarray(total, dtype=np.float64)
    mu, sd = float(total.mean()), float(total.std())
    best_val = float(total.max())
    best_idx = int(np.argmax(total))

    # Second-best value outside a small neighbourhood of the peak (separation).
    mask = np.abs(z0_grid - z0_grid[best_idx]) > max(1, int(0.02 * len(z0_grid)))
    second_val = float(total[mask].max()) if np.any(mask) else float("nan")

    return {
        "best_z0": int(best_z0),
        "argmax_z0": int(z0_grid[best_idx]),
        "peak_score": best_val,
        "peak_z_over_sigma": (best_val - mu) / sd if sd > 0 else float("inf"),
        "peak_minus_runner_up": best_val - second_val,
        "rel_prominence": (best_val - second_val) / best_val if best_val != 0 else float("nan"),
    }


def update_reference_intensity_mapping_from_target_stack(
    F260517_ref_mem,
    target_stack_zyx,
    z_idx,
    option,
    thresFactor,
    maskRange,
    smoothPenalty_raw,
    percentiles,
):
    """Learn a raw-reference -> target intensity mapping on the matched z planes
    only, then apply it to the full reference volume.

    Warping the reference's intensity histogram to match the moving data stops
    the optimiser from mistaking a brightness difference for motion.

    target_stack_zyx : (K, Y, X)
        Initial update -> mean of raw moving frames 0..4.
        Periodic updates -> mean of recent RAW moving frames (carries the
        brightness drift the recalibration is meant to track).

    Returns the adjusted reference (X, Y, Zref) plus the quantile-mapping info.
    """

    target_stack_zyx = np.asarray(target_stack_zyx, dtype=np.float32)

    if target_stack_zyx.ndim != 3:
        raise ValueError(
            f"target_stack_zyx should be (K,Y,X), got {target_stack_zyx.shape}"
        )

    if target_stack_zyx.shape[0] != len(z_idx):
        raise ValueError(
            f"target_stack_zyx K={target_stack_zyx.shape[0]} does not match "
            f"z_idx length={len(z_idx)}"
        )

    # Learn the mapping only on the reference planes that correspond to moving
    # slices, so unmatched reference depth does not bias the histogram.
    ref_source_zyx = F260517_ref_mem[z_idx].astype(np.float32, copy=False)

    if ref_source_zyx.shape != target_stack_zyx.shape:
        raise ValueError(
            f"ref_source_zyx shape {ref_source_zyx.shape} does not match "
            f"target_stack_zyx shape {target_stack_zyx.shape}"
        )

    src_q, tgt_q, used_percentiles = prep.learn_quantile_mapping(
        source=ref_source_zyx,
        target=target_stack_zyx,
        percentiles=percentiles,
    )

    # Apply the learned mapping to the whole reference volume; transpose to the
    # (X, Y, Zref) order the registration engine expects.
    F260517_ref_mem_adj = (
        prep.apply_quantile_mapping(F260517_ref_mem, src_q, tgt_q)
        .transpose(2, 1, 0)
        .astype(np.float32, copy=False)
    )

    # Refresh the reference mask and smoothness penalty for the new intensities.
    option["mask_ref"] = mask.getMask(F260517_ref_mem_adj, thresFactor)
    option["mask_ref"] = mask.bwareafilt3_wei(option["mask_ref"], maskRange)

    Pnltfactor = prep.getSmPnltNormFctr(F260517_ref_mem_adj, option)
    option["smoothPenalty"] = Pnltfactor * smoothPenalty_raw

    return F260517_ref_mem_adj, src_q, tgt_q, used_percentiles


def robust_average_target_z(target_z_list, method="median", trim_percentiles=(10, 90)):
    """Combine per-frame target_z estimates (each shape (K,)) from the warmup
    frames into one stable fixed_target_z (K,). Median/trimmed-mean guard against
    a single bad warmup frame skewing the fixed projection planes."""
    arr = np.stack([np.asarray(z, dtype=np.float32) for z in target_z_list], axis=0)

    if method == "median":
        fixed_z = np.nanmedian(arr, axis=0)

    elif method == "mean":
        fixed_z = np.nanmean(arr, axis=0)

    elif method == "trimmed_mean":
        lo_p, hi_p = trim_percentiles
        fixed = []

        for k_idx in range(arr.shape[1]):
            vals = arr[:, k_idx]
            vals = vals[np.isfinite(vals)]

            if vals.size == 0:
                fixed.append(np.nan)
                continue

            lo = np.percentile(vals, lo_p)
            hi = np.percentile(vals, hi_p)
            vals_trim = vals[(vals >= lo) & (vals <= hi)]

            if vals_trim.size == 0:
                fixed.append(float(np.median(vals)))
            else:
                fixed.append(float(np.mean(vals_trim)))

        fixed_z = np.asarray(fixed, dtype=np.float32)

    else:
        raise ValueError("method should be 'median', 'mean', or 'trimmed_mean'.")

    return fixed_z.astype(np.float32)


def estimate_projection_z_from_phase_simple(
    phase_new,
    z_init,
    ref_shape,
    ref_volume_order="zyx",
    method="trimmed_mean",
    trim_percentiles=(5, 95),
    frame_idx=None,
):
    """For each moving slice k, estimate the single reference z plane it maps to,
    by summarising phase_new[..., 2] over all valid voxels in that slice.

    phase_new : (X, Y, K, 3) -- xyz coordinates in the reference volume.
    z_init    : (K,)         -- fallback when a slice has no valid voxels.

    Returns (target_z_pred (K,), per-slice diagnostics DataFrame).
    """
    if hasattr(phase_new, "get"):
        phase = phase_new.get()
    else:
        phase = np.asarray(phase_new)

    phase = np.asarray(phase, dtype=np.float32)
    z_init = np.asarray(z_init, dtype=np.float32)

    if phase.ndim != 4 or phase.shape[-1] != 3:
        raise ValueError(f"phase_new should be (X,Y,K,3), got {phase.shape}")

    X, Y, K_local, _ = phase.shape

    if len(z_init) != K_local:
        raise ValueError(f"z_init length {len(z_init)} does not match K={K_local}")

    x_ref = phase[..., 0]
    y_ref = phase[..., 1]
    z_ref = phase[..., 2]

    if ref_volume_order == "zyx":
        Zref, Yref, Xref = ref_shape
    elif ref_volume_order == "xyz":
        Xref, Yref, Zref = ref_shape
    else:
        raise ValueError("ref_volume_order should be 'zyx' or 'xyz'.")

    # Only voxels that map inside the reference bounds contribute.
    valid = (
        np.isfinite(x_ref)
        & np.isfinite(y_ref)
        & np.isfinite(z_ref)
        & (x_ref >= 0) & (x_ref <= Xref - 1)
        & (y_ref >= 0) & (y_ref <= Yref - 1)
        & (z_ref >= 0) & (z_ref <= Zref - 1)
    )

    target_z_pred = np.zeros(K_local, dtype=np.float32)
    rows = []

    for k_idx in range(K_local):
        z_k = z_ref[:, :, k_idx]
        valid_k = valid[:, :, k_idx]
        z_valid = z_k[valid_k]

        if z_valid.size == 0:
            z_est = float(z_init[k_idx])

        else:
            if method == "mean":
                z_est = float(np.mean(z_valid))

            elif method == "median":
                z_est = float(np.median(z_valid))

            elif method == "trimmed_mean":
                p_low, p_high = trim_percentiles
                lo = np.percentile(z_valid, p_low)
                hi = np.percentile(z_valid, p_high)
                z_trim = z_valid[(z_valid >= lo) & (z_valid <= hi)]

                if z_trim.size == 0:
                    z_est = float(np.median(z_valid))
                else:
                    z_est = float(np.mean(z_trim))

            else:
                raise ValueError("method should be 'mean', 'median', or 'trimmed_mean'.")

        target_z_pred[k_idx] = z_est

        if z_valid.size > 0:
            dz_valid = z_valid - float(z_init[k_idx])

            rows.append({
                "frame": frame_idx,
                "k": k_idx,
                "z_init": float(z_init[k_idx]),
                "target_z_pred": float(z_est),
                "target_minus_z_init": float(z_est - z_init[k_idx]),
                "z_mean": float(np.mean(z_valid)),
                "z_median": float(np.median(z_valid)),
                "z_p05": float(np.percentile(z_valid, 5)),
                "z_p95": float(np.percentile(z_valid, 95)),
                "dz_mean": float(np.mean(dz_valid)),
                "dz_median": float(np.median(dz_valid)),
                "dz_p05": float(np.percentile(dz_valid, 5)),
                "dz_p95": float(np.percentile(dz_valid, 95)),
                "valid_ratio": float(z_valid.size / (X * Y)),
            })

        else:
            rows.append({
                "frame": frame_idx,
                "k": k_idx,
                "z_init": float(z_init[k_idx]),
                "target_z_pred": float(z_est),
                "target_minus_z_init": float(z_est - z_init[k_idx]),
                "z_mean": np.nan,
                "z_median": np.nan,
                "z_p05": np.nan,
                "z_p95": np.nan,
                "dz_mean": np.nan,
                "dz_median": np.nan,
                "dz_p05": np.nan,
                "dz_p95": np.nan,
                "valid_ratio": 0.0,
            })

    return target_z_pred, pd.DataFrame(rows)


def _make_xy_sampling_grid(X, Y, upsample_factor):
    """Dense sampling grid over the same [0, X-1] x [0, Y-1] moving XY domain."""
    factor = int(upsample_factor)

    if factor < 1:
        raise ValueError("upsample_factor should be >= 1.")

    Xup = int(X * factor)
    Yup = int(Y * factor)

    x_new = np.linspace(0, X - 1, Xup, dtype=np.float32)
    y_new = np.linspace(0, Y - 1, Yup, dtype=np.float32)

    Xg, Yg = np.meshgrid(x_new, y_new, indexing="ij")

    coords_2d = np.vstack([Xg.reshape(-1), Yg.reshape(-1)])

    return coords_2d, Xup, Yup


def upsample_phase_xy_for_supersurface(phase_new, upsample_factor=2):
    """Bilinearly upsample phase_new (X,Y,K,3) -> (X*f, Y*f, K, 3) in moving XY."""
    phase = np.asarray(phase_new, dtype=np.float32)

    if phase.ndim != 4 or phase.shape[-1] != 3:
        raise ValueError(f"phase_new should be (X,Y,K,3), got {phase.shape}")

    factor = int(upsample_factor)

    if factor == 1:
        return phase

    X, Y, K_local, C = phase.shape
    coords_2d, Xup, Yup = _make_xy_sampling_grid(X, Y, factor)

    phase_up = np.empty((Xup, Yup, K_local, C), dtype=np.float32)

    for k_idx in range(K_local):
        for c_idx in range(C):
            phase_up[:, :, k_idx, c_idx] = map_coordinates(
                phase[:, :, k_idx, c_idx],
                coords_2d,
                order=1,
                mode="nearest",
            ).reshape(Xup, Yup)

    return phase_up


def upsample_values_xy_for_supersurface(values_xyk, upsample_factor=2, order=1):
    """Upsample moving values (X,Y,K) in XY. order=1 for the continuous membrane,
    order=0 (nearest) for the sparse bright-cell channel."""
    values = np.asarray(values_xyk, dtype=np.float32)

    if values.ndim != 3:
        raise ValueError(f"values_xyk should be (X,Y,K), got {values.shape}")

    factor = int(upsample_factor)

    if factor == 1:
        return values

    X, Y, K_local = values.shape
    coords_2d, Xup, Yup = _make_xy_sampling_grid(X, Y, factor)

    values_up = np.empty((Xup, Yup, K_local), dtype=np.float32)

    for k_idx in range(K_local):
        values_up[:, :, k_idx] = map_coordinates(
            values[:, :, k_idx],
            coords_2d,
            order=order,
            mode="nearest",
        ).reshape(Xup, Yup)

    return values_up


def save_single_channel_ome_tiff(volume_zyx, out_dir, frame_idx, label="F260517"):
    """Save one (Z, Y, X) volume as a single-channel OME-TIFF in out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    IO.write_multichannel_volume_as_ome_tiff(
        volume=[np.asarray(volume_zyx, dtype=np.float32)],
        out_dir=out_dir,
        frame_idx=frame_idx,
        label=label,
    )


def summarize_coverage_only(frame_idx, coverage, coverage_threshold=0.0):
    """Summarise the forward-scatter coverage map (Nplanes, Y, X). Pixels at or
    below coverage_threshold received no sample (a hole). Prints a per-frame line
    and returns one global + per-plane row for the diagnostics CSV."""
    coverage = np.asarray(coverage, dtype=np.float32)

    if coverage.ndim != 3:
        raise ValueError(f"coverage should be (Nplanes,Y,X), got {coverage.shape}")

    no_coverage = coverage <= coverage_threshold

    no_cov_per_plane = np.mean(no_coverage, axis=(1, 2))
    cov_mean_per_plane = np.mean(coverage, axis=(1, 2))
    cov_max_per_plane = np.max(coverage, axis=(1, 2))

    global_no_cov = float(np.mean(no_coverage))
    worst_k = int(np.argmax(no_cov_per_plane))

    print(
        f"[Coverage] frame {frame_idx}: "
        f"global_no_cov={global_no_cov:.4f}, "
        f"min_plane_no_cov={float(no_cov_per_plane.min()):.4f}, "
        f"median_plane_no_cov={float(np.median(no_cov_per_plane)):.4f}, "
        f"max_plane_no_cov={float(no_cov_per_plane.max()):.4f}, "
        f"worst_k={worst_k}"
    )

    rows = []

    rows.append({
        "frame": int(frame_idx),
        "plane": -1,
        "is_global": True,
        "no_coverage_ratio": global_no_cov,
        "coverage_ratio": 1.0 - global_no_cov,
        "coverage_mean": float(np.mean(coverage)),
        "coverage_max": float(np.max(coverage)),
        "worst_plane_by_no_coverage": worst_k,
        "max_plane_no_coverage_ratio": float(no_cov_per_plane.max()),
    })

    for k_idx in range(coverage.shape[0]):
        rows.append({
            "frame": int(frame_idx),
            "plane": int(k_idx),
            "is_global": False,
            "no_coverage_ratio": float(no_cov_per_plane[k_idx]),
            "coverage_ratio": float(1.0 - no_cov_per_plane[k_idx]),
            "coverage_mean": float(cov_mean_per_plane[k_idx]),
            "coverage_max": float(cov_max_per_plane[k_idx]),
            "worst_plane_by_no_coverage": worst_k,
            "max_plane_no_coverage_ratio": float(no_cov_per_plane.max()),
        })

    return rows


def fill_holes_nearest_limited(
    proj_zyx,
    fill_value=-200.0,
    max_fill_distance=2.0,
    hole_tol=1e-6,
):
    """Fill only small scatter holes (pixels equal to fill_value) from the nearest
    valid pixel, within max_fill_distance. Does not smooth the rest of the image."""
    proj = np.asarray(proj_zyx, dtype=np.float32)
    out = proj.copy()

    for zi in range(out.shape[0]):
        plane = out[zi]

        hole = np.abs(plane - fill_value) <= hole_tol
        valid = (~hole) & np.isfinite(plane)

        if not np.any(hole) or not np.any(valid):
            continue

        dist, indices = distance_transform_edt(~valid, return_indices=True)

        fill_mask = hole & (dist <= max_fill_distance)

        nearest_y = indices[0][fill_mask]
        nearest_x = indices[1][fill_mask]

        plane[fill_mask] = plane[nearest_y, nearest_x]
        out[zi] = plane

    return out


def project_converge_map_gpu(
    phase_new,
    ref_volume_for_shape,
    fixed_target_z,
    ref_volume_order="xyz",
    z_window=3.0,
    downsample_xy=1,
    xy_extra_radius=0,
):
    """Coverage map (Nplanes, Y, X) for phase_new: splat ones and count where at
    least one sample landed. Used to quantify the forward-scatter holes."""
    values_ones = np.ones(phase_new.shape[:-1], dtype=np.float32)

    coverage = calFlowCrossResolution.project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_new,
        ref_volume=ref_volume_for_shape,
        target_z_planes=fixed_target_z,
        values_xyk=values_ones,
        ref_volume_order=ref_volume_order,
        z_window=z_window,
        downsample_xy=downsample_xy,
        fill_value=0.0,
        return_numpy=True,
        output_order="zyx",
        xy_splat_mode="subpixel_footprint",
        xy_extra_radius=xy_extra_radius,
    )

    return coverage


def project_and_save_single_frame_moving_derived(
    frame_idx,
    phase_new,
    fixed_target_z,
    ref_mem_adj_for_projection,
    F260517_ref_sparseCell,
    raw_mem_zyx,
    raw_sparseCell_zyx,
    out_dirs,
    surface_upsample_factor=2,
    surface_mem_value_order=1,
    surface_sparse_value_order=0,
    enable_coverage_diagnostics=True,
    coverage_threshold=0.0,
    enable_hole_filling=False,
    projection_z_window=3.0,
    projection_downsample_xy=1,
    projection_fill_value=-200.0,
    projection_xy_extra_radius=0,
):
    """Save four volumes per frame (raw moving mem, raw moving sparse-cell,
    projected mem, projected sparse-cell) into four folders.

    Projected values come from the RAW MOVING images, not the reference; the
    reference volumes are only passed as the output canvas/shape. With
    surface_upsample_factor > 1 the phase and values are upsampled in moving XY
    before projection to reduce scatter holes.

    The projection_* arguments are explicit parameters (they used to be module
    globals in the notebook) so this helper is self-contained and importable.
    """
    phase_new = np.asarray(phase_new, dtype=np.float32)

    raw_mem_zyx = np.asarray(raw_mem_zyx, dtype=np.float32)
    raw_sparseCell_zyx = np.asarray(raw_sparseCell_zyx, dtype=np.float32)

    raw_mem_xyk = raw_mem_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)
    raw_sparse_xyk = raw_sparseCell_zyx.transpose(2, 1, 0).astype(np.float32, copy=False)

    # 1. Save the raw moving channels as-is.
    save_single_channel_ome_tiff(
        volume_zyx=raw_mem_zyx,
        out_dir=out_dirs["raw_moving_mem"],
        frame_idx=frame_idx,
        label="F260517_raw_mem",
    )

    save_single_channel_ome_tiff(
        volume_zyx=raw_sparseCell_zyx,
        out_dir=out_dirs["raw_moving_sparseCell"],
        frame_idx=frame_idx,
        label="F260517_raw_sparseCell",
    )

    # 2. Supersurface upsampling of phase + values in the moving XY domain.
    if int(surface_upsample_factor) > 1:
        print(
            f"[Supersurface] frame {frame_idx}: "
            f"upsample_factor={surface_upsample_factor}"
        )

    phase_for_projection = upsample_phase_xy_for_supersurface(
        phase_new,
        upsample_factor=surface_upsample_factor,
    )

    mem_values_for_projection = upsample_values_xy_for_supersurface(
        raw_mem_xyk,
        upsample_factor=surface_upsample_factor,
        order=surface_mem_value_order,
    )

    sparse_values_for_projection = upsample_values_xy_for_supersurface(
        raw_sparse_xyk,
        upsample_factor=surface_upsample_factor,
        order=surface_sparse_value_order,
    )

    if phase_for_projection.shape[:-1] != mem_values_for_projection.shape:
        raise ValueError(
            f"phase_for_projection shape {phase_for_projection.shape[:-1]} "
            f"does not match mem_values_for_projection shape {mem_values_for_projection.shape}"
        )

    if phase_for_projection.shape[:-1] != sparse_values_for_projection.shape:
        raise ValueError(
            f"phase_for_projection shape {phase_for_projection.shape[:-1]} "
            f"does not match sparse_values_for_projection shape {sparse_values_for_projection.shape}"
        )

    # 3. Project the raw moving membrane onto the fixed reference planes.
    mem_proj = calFlowCrossResolution.project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_for_projection,
        ref_volume=ref_mem_adj_for_projection,       # canvas/shape only
        target_z_planes=fixed_target_z,
        values_xyk=mem_values_for_projection,        # values come from moving
        ref_volume_order="xyz",
        z_window=projection_z_window,
        downsample_xy=projection_downsample_xy,
        fill_value=projection_fill_value,
        return_numpy=True,
        output_order="zyx",
        xy_splat_mode="subpixel_footprint",
        xy_extra_radius=projection_xy_extra_radius,
    )

    # 4. Project the raw moving sparse-cell channel onto the same planes.
    sparse_proj = calFlowCrossResolution.project_coords_to_fixed_planes_gpu(
        coords_ref_xyk_xyz=phase_for_projection,
        ref_volume=F260517_ref_sparseCell,            # canvas/shape only
        target_z_planes=fixed_target_z,
        values_xyk=sparse_values_for_projection,      # values come from moving
        ref_volume_order="zyx",
        z_window=projection_z_window,
        downsample_xy=projection_downsample_xy,
        fill_value=projection_fill_value,
        return_numpy=True,
        output_order="zyx",
        xy_splat_mode="subpixel_footprint",
        xy_extra_radius=projection_xy_extra_radius,
    )

    coverage_rows = []

    # 5. Coverage diagnostics (does not alter the saved projections).
    if enable_coverage_diagnostics:
        coverage = project_converge_map_gpu(
            phase_new=phase_for_projection,
            ref_volume_for_shape=ref_mem_adj_for_projection,
            fixed_target_z=fixed_target_z,
            ref_volume_order="xyz",
            z_window=projection_z_window,
            downsample_xy=projection_downsample_xy,
            xy_extra_radius=projection_xy_extra_radius,
        )

        if hasattr(coverage, "get"):
            coverage = coverage.get()

        coverage = np.asarray(coverage, dtype=np.float32)

        if coverage.shape != mem_proj.shape:
            raise ValueError(
                f"coverage shape {coverage.shape} does not match projection shape {mem_proj.shape}. "
                "Please check project_converge_map_gpu output_order / downsample settings."
            )

        coverage_rows = summarize_coverage_only(
            frame_idx=frame_idx,
            coverage=coverage,
            coverage_threshold=coverage_threshold,
        )

    # 6. Optional small-hole filling on the moving-derived projections.
    if enable_hole_filling:
        mem_proj = fill_holes_nearest_limited(
            mem_proj,
            fill_value=projection_fill_value,
            max_fill_distance=2.0,
        )

        sparse_proj = fill_holes_nearest_limited(
            sparse_proj,
            fill_value=projection_fill_value,
            max_fill_distance=1.5,
        )

    # 7. Save the projected channels.
    save_single_channel_ome_tiff(
        volume_zyx=mem_proj,
        out_dir=out_dirs["projected_mem"],
        frame_idx=frame_idx,
        label="F260517_projected_mem_from_moving",
    )

    save_single_channel_ome_tiff(
        volume_zyx=sparse_proj,
        out_dir=out_dirs["projected_sparseCell"],
        frame_idx=frame_idx,
        label="F260517_projected_sparseCell_from_moving",
    )

    del (
        mem_proj,
        sparse_proj,
        phase_for_projection,
        mem_values_for_projection,
        sparse_values_for_projection,
    )

    return coverage_rows
