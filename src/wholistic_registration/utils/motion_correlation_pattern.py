"""
motion_correlation_pattern.py

A cleaned, source/mode-response based motion-pattern analysis pipeline.

Core idea
---------
For each MotionEpisode, remove global background cumulative displacement and decompose
patch-wise cumulative displacement into shared-activation motion modes:

    Y_i(t) ~= sum_k h_k(t) * b_ik

where:
    h_k(t): scalar temporal activation of mode k
    b_ik:   2D response vector of patch i to mode k

Then split each mode's response support into spatially coherent MotionRegions.
MotionPattern is built across episodes from MotionRegions whose spatial support,
activation h_k(t), and response vector b are similar.

This file intentionally avoids the old "MotionSource = real biological source" interpretation.
Here MotionMode is only a data-driven source-response/correlation mode.
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import exposure
from tifffile import imwrite

try:
    import networkx as nx
except Exception:
    nx = None
try:
    import cupy as cp
    from cupyx.scipy import ndi as cupy_ndi
    HAS_CUPY = True
except Exception:
    cp = None
    cupy_ndi = None
    HAS_CUPY = False

# =============================================================================
# Data structures
# =============================================================================


class MotionUnit:
    """A patch-level active temporal interval."""

    def __init__(self, time_range=(-1, -1), spatial_coor=(-1, -1), motion=None):
        self.time_range = list(time_range)
        self.spatial_coor = list(spatial_coor)
        self.motion = motion
        self.T = 0 if motion is None else len(motion)

        if motion is not None:
            if self.time_range[1] - self.time_range[0] + 1 != self.T:
                raise ValueError("Time range and motion length conflict.")


class MotionEpisode:
    """A spatiotemporal motion event built from patch-level MotionUnits."""

    def __init__(
        self,
        time_range=(-1, -1),
        region_mask=None,
        episode_id=-1,
        motion_delta=None,
        motion_abs=None,
        global_motion=None,
        global_motion_mode="median",
    ):
        self.time_range = list(time_range)
        self.episode_id = int(episode_id)
        self.spatial_region = (
            np.zeros((1, 1), dtype=np.uint8)
            if region_mask is None
            else np.asarray(region_mask).astype(np.uint8)
        )

        # delta motion is used for event detection / optional debugging
        self.motion_delta = motion_delta
        # alias for compatibility with previous code
        self.motion = motion_delta

        # cumulative displacement is used for mode decomposition
        self.motion_abs = motion_abs
        self.global_motion = global_motion
        self.global_motion_mode = global_motion_mode

        self.modes: List[MotionMode] = []
        self.regions: List[MotionRegion] = []
        self.mode_model: Dict[str, Any] = {}

    def __repr__(self):
        return f"MotionEpisode(id={self.episode_id}, time={self.time_range}, area={int(np.sum(self.spatial_region > 0))})"

    def decompose_motion_modes(self, **kwargs):
        modes = decompose_episode_motion_modes(self, **kwargs)
        self.modes = modes
        return modes

    def split_motion_regions(self, **kwargs):
        regions = split_episode_modes_to_regions(self, **kwargs)
        self.regions = regions
        return regions


class MotionMode:
    """
    A shared-activation source-response mode inside one episode.

    Important: this is NOT assumed to be a real biophysical force source.
    It is a data-driven correlation/mode object represented by h_k(t) and B_k(x).
    """

    def __init__(
        self,
        episode_id=-1,
        mode_id=-1,
        time_range=None,
        activation=None,             # h_k, shape (T,)
        response_field=None,         # B_k(x), shape (X, Y, 2)
        response_strength=None,      # ||B_k(x)||, shape (X, Y)
        response_direction=None,     # normalized B_k(x), shape (X, Y, 2)
        support_mask=None,           # thresholded response support, shape (X, Y)
        compact_response_field=None, # compact B over episode mask, shape (N, 2)
        valid_coords=None,           # compact coords, shape (N, 2)
        background_motion=None,
        reconstructed_motion=None,
        residual_motion=None,
        explained_energy=0.0,
        residual_energy=0.0,
        mode_mass=0.0,
        seed_position=None,
        confidence=1.0,
        metadata=None,
    ):
        self.episode_id = episode_id
        self.mode_id = mode_id
        self.time_range = time_range

        self.activation = activation
        self.response_field = response_field
        self.response_strength = response_strength
        self.response_direction = response_direction
        self.support_mask = support_mask

        self.compact_response_field = compact_response_field
        self.valid_coords = valid_coords

        self.background_motion = background_motion
        self.reconstructed_motion = reconstructed_motion
        self.residual_motion = residual_motion

        self.explained_energy = float(explained_energy)
        self.residual_energy = float(residual_energy)
        self.mode_mass = float(mode_mass)
        self.seed_position = seed_position
        self.confidence = float(confidence)
        self.metadata = metadata or {}

        self.regions: List[MotionRegion] = []


class MotionRegion:
    """
    Spatially coherent region split from a MotionMode response support.

    This is the recommended basic unit for downstream MotionPattern clustering.
    """

    def __init__(
        self,
        episode_id=-1,
        mode_id=-1,
        region_id=-1,
        time_range=None,
        activation=None,
        response_field=None,      # region-local full map, (X, Y, 2)
        response_strength=None,   # region-local full map, (X, Y)
        region_mask=None,         # full map, bool/uint8, (X, Y)
        induced_motion=None,      # representative h(t)*mean_b, (T, 2)
        mean_response_vector=None,
        center_xy=None,
        spatial_cov=None,
        area_effective=0.0,
        strength=0.0,
        metadata=None,
    ):
        self.episode_id = episode_id
        self.mode_id = mode_id
        self.region_id = region_id
        self.component_id = region_id  # compatibility
        self.time_range = time_range

        self.activation = activation
        self.response_field = response_field
        self.response_strength = response_strength
        self.region_mask = region_mask

        self.induced_motion = induced_motion
        self.mean_response_vector = mean_response_vector

        self.center_xy = center_xy
        self.spatial_cov = spatial_cov
        self.area_effective = float(area_effective)
        self.strength = float(strength)
        self.duration = 0 if time_range is None else int(time_range[1] - time_range[0] + 1)
        self.metadata = metadata or {}

        # Compatibility with old MotionComponent / MotionPattern code.
        self.region_magnitude = response_strength
        self.base_func = induced_motion
        self.base_func_resampled = None
        self.temporal_feature = None
        self.activation_resampled = None
        self.activation_feature = None
        self.response_feature = None
        self.time_length_resampled = None

    def show_activation(self, figsize=(7, 4)):
        h = np.asarray(self.activation, dtype=np.float32)
        plt.figure(figsize=figsize)
        plt.plot(np.arange(len(h)), h, marker="o")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"Region {self.region_id} activation, ep={self.episode_id}, mode={self.mode_id}")
        plt.xlabel("relative frame")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class MotionPattern:
    """A cross-episode pattern built from MotionRegion objects."""

    def __init__(self, pattern_id: int, regions: Sequence[MotionRegion]):
        self.pattern_id = int(pattern_id)
        self.regions = list(regions)
        self.components = self.regions  # compatibility
        self.n_members = len(self.regions)
        self.episode_ids: List[int] = []
        self.region_keys: List[Tuple[int, int, int]] = []

        self.prototype_activation = None
        self.prototype_induced_motion = None
        self.prototype_region = None
        self.prototype_response_vector = None
        self.center_xy = None
        self.spatial_cov = None
        self.total_strength = 0.0

        self._summarize()

    def _summarize(self, eps=1e-12):
        if len(self.regions) == 0:
            return

        # ------------------------------------------------------------
        # Basic identifiers
        # ------------------------------------------------------------
        self.region_keys = [
            (
                getattr(r, "episode_id", None),
                getattr(r, "mode_id", None),
                getattr(r, "region_id", None),
            )
            for r in self.regions
        ]

        self.episode_ids = sorted(
            list(set([getattr(r, "episode_id", None) for r in self.regions]))
        )

        weights = np.asarray(
            [
                max(float(getattr(r, "strength", 1.0)), 0.0)
                for r in self.regions
            ],
            dtype=np.float32,
        )

        if float(np.sum(weights)) < eps:
            weights[:] = 1.0

        w = weights / np.maximum(np.sum(weights), eps)
        self.total_strength = float(np.sum(weights))

        # ------------------------------------------------------------
        # 1. Activation summary: variable-length, medoid prototype
        # ------------------------------------------------------------
        activations = []
        activation_weights = []
        activation_region_indices = []

        for idx, (wi, r) in enumerate(zip(w, self.regions)):
            h = getattr(r, "activation", None)

            # fallback
            if h is None:
                h = getattr(r, "activation_resampled", None)

            if h is None:
                continue

            h = np.asarray(h, dtype=np.float32).reshape(-1)
            if len(h) == 0:
                continue

            activations.append(h)
            activation_weights.append(float(wi))
            activation_region_indices.append(idx)

        self.activation_list = activations

        if len(activations) > 0:
            activation_weights = np.asarray(activation_weights, dtype=np.float32)
            activation_weights = activation_weights / np.maximum(
                np.sum(activation_weights), eps
            )

            local_medoid_idx = _find_activation_medoid_index(
                activations,
                weights=activation_weights,
                eps=eps,
            )

            global_medoid_idx = activation_region_indices[local_medoid_idx]
            medoid_region = self.regions[global_medoid_idx]
            medoid_activation = activations[local_medoid_idx].copy()

            self.prototype_region_object = medoid_region
            self.prototype_activation = medoid_activation
            self.prototype_activation_type = "medoid"
            self.prototype_activation_region_index = int(global_medoid_idx)

            # Optional fixed-length version only for plotting / feature export.
            self.prototype_activation_resampled = _resample_1d(
                medoid_activation,
                16,
            )
        else:
            self.prototype_region_object = None
            self.prototype_activation = None
            self.prototype_activation_type = "none"
            self.prototype_activation_region_index = None
            self.prototype_activation_resampled = None

        # ------------------------------------------------------------
        # 2. Prototype induced motion
        #    Use medoid region; do not average variable-length curves.
        # ------------------------------------------------------------
        if self.prototype_region_object is not None:
            proto_motion = getattr(self.prototype_region_object, "induced_motion", None)
            if proto_motion is not None:
                self.prototype_induced_motion = np.asarray(
                    proto_motion,
                    dtype=np.float32,
                )
                try:
                    self.prototype_induced_motion_resampled = _resample_vector_func(
                        self.prototype_induced_motion,
                        16,
                    )
                except Exception:
                    self.prototype_induced_motion_resampled = None
            else:
                self.prototype_induced_motion = None
                self.prototype_induced_motion_resampled = None
        else:
            self.prototype_induced_motion = None
            self.prototype_induced_motion_resampled = None

        # ------------------------------------------------------------
        # 3. Prototype region map
        #    Spatial maps have the same image shape, so weighted averaging is OK.
        # ------------------------------------------------------------
        region_maps = []
        wr = []

        for wi, r in zip(w, self.regions):
            rg = getattr(r, "response_strength", None)
            if rg is None:
                continue

            rg = np.asarray(rg, dtype=np.float32)
            s = float(np.sum(rg))

            if s > eps:
                rg = rg / s

            region_maps.append(rg)
            wr.append(wi)

        if len(region_maps) > 0:
            wr = np.asarray(wr, dtype=np.float32)
            wr = wr / np.maximum(np.sum(wr), eps)

            proto_rg = np.zeros_like(region_maps[0], dtype=np.float32)
            for wi, rg in zip(wr, region_maps):
                if rg.shape != proto_rg.shape:
                    # Defensive check. This should usually not happen within one dataset.
                    continue
                proto_rg += wi * rg

            # Keep old field name for compatibility.
            self.prototype_region = proto_rg

            # Clearer name for future code.
            self.prototype_region_map = proto_rg
        else:
            self.prototype_region = None
            self.prototype_region_map = None

        # ------------------------------------------------------------
        # 4. Prototype response vector
        #    Use weighted average, sign-aligned to medoid activation.
        # ------------------------------------------------------------
        vecs = []
        wv = []

        ref_h = self.prototype_activation

        for wi, r in zip(w, self.regions):
            v = getattr(r, "mean_response_vector", None)
            if v is None:
                continue

            v = np.asarray(v, dtype=np.float32).reshape(-1)
            if v.shape != (2,):
                continue

            h = getattr(r, "activation", None)
            if h is None:
                h = getattr(r, "activation_resampled", None)

            if ref_h is not None and h is not None:
                sign = _sign_by_resampled_corr(ref_h, h, target_len=16)
                v = sign * v

            vecs.append(v)
            wv.append(wi)

        if len(vecs) > 0:
            wv = np.asarray(wv, dtype=np.float32)
            wv = wv / np.maximum(np.sum(wv), eps)

            proto_v = np.zeros(2, dtype=np.float32)
            for wi, v in zip(wv, vecs):
                proto_v += wi * v

            self.prototype_response_vector = proto_v.astype(np.float32)
        else:
            self.prototype_response_vector = None

        # ------------------------------------------------------------
        # 5. Spatial center / covariance
        # ------------------------------------------------------------
        centers = []
        wc = []

        for wi, r in zip(w, self.regions):
            c = getattr(r, "center_xy", None)
            if c is None:
                continue

            c = np.asarray(c, dtype=np.float32)

            if c.shape == (2,) and np.all(np.isfinite(c)):
                centers.append(c)
                wc.append(wi)

        if len(centers) > 0:
            centers = np.stack(centers, axis=0)
            wc = np.asarray(wc, dtype=np.float32)
            wc = wc / np.maximum(np.sum(wc), eps)

            c = np.sum(centers * wc[:, None], axis=0)

            self.center_xy = c.astype(np.float32)

            dc = centers - c[None, :]
            self.spatial_cov = (dc.T @ (dc * wc[:, None])).astype(np.float32)
        else:
            self.center_xy = None
            self.spatial_cov = None

    def summary_dict(self):
        return {
            "pattern_id": self.pattern_id,
            "n_members": self.n_members,
            "episode_ids": self.episode_ids,
            "region_keys": self.region_keys,
            "center_xy": None if self.center_xy is None else self.center_xy.tolist(),
            "total_strength": self.total_strength,
        }


# =============================================================================
# Generic helpers
# =============================================================================

def _safe_corr(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        L = min(a.size, b.size)
        a = a[:L]
        b = b[:L]
    a = a - np.mean(a)
    b = b - np.mean(b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        # fall back to cosine without demeaning
        a0 = np.asarray(a, dtype=np.float32).reshape(-1)
        b0 = np.asarray(b, dtype=np.float32).reshape(-1)
        return float(np.dot(a0, b0) / (np.linalg.norm(a0) * np.linalg.norm(b0) + eps))
    return float(np.dot(a, b) / (na * nb + eps))


def _resample_1d(x, target_len=16):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = len(x)
    if T == 0:
        return np.zeros(target_len, dtype=np.float32)
    if T == target_len:
        return x.copy()
    if T == 1:
        return np.full(target_len, float(x[0]), dtype=np.float32)
    old = np.linspace(0.0, 1.0, T)
    new = np.linspace(0.0, 1.0, target_len)
    return np.interp(new, old, x).astype(np.float32)


def _resample_vector_func(v, target_len=16):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(f"Expected vector function shape (T,2), got {v.shape}")
    T = v.shape[0]
    if T == target_len:
        return v.copy()
    if T <= 1:
        return np.repeat(v, target_len, axis=0).astype(np.float32)
    old = np.linspace(0.0, 1.0, T)
    new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, 2), dtype=np.float32)
    out[:, 0] = np.interp(new, old, v[:, 0])
    out[:, 1] = np.interp(new, old, v[:, 1])
    return out

def _compute_spatial_stats(region_magnitude, eps=1e-12):
    R = np.asarray(region_magnitude, dtype=np.float32)
    if R.ndim != 2:
        raise ValueError(f"region_magnitude must be 2D, got {R.shape}")
    strength = float(np.sum(np.maximum(R, 0.0)))
    if strength <= eps:
        return (
            np.array([np.nan, np.nan], dtype=np.float32),
            np.full((2, 2), np.nan, dtype=np.float32),
            0.0,
            0.0,
        )
    coords = np.argwhere(R > 0)
    w = R[R > 0].astype(np.float32)
    x = coords[:, 0].astype(np.float32)
    y = coords[:, 1].astype(np.float32)
    ws = float(np.sum(w))
    cx = float(np.sum(w * x) / max(ws, eps))
    cy = float(np.sum(w * y) / max(ws, eps))
    dx = x - cx
    dy = y - cy
    cov = np.array(
        [
            [float(np.sum(w * dx * dx) / max(ws, eps)), float(np.sum(w * dx * dy) / max(ws, eps))],
            [float(np.sum(w * dx * dy) / max(ws, eps)), float(np.sum(w * dy * dy) / max(ws, eps))],
        ],
        dtype=np.float32,
    )
    area_eff = float((ws ** 2) / max(float(np.sum(w ** 2)), eps))
    return np.array([cx, cy], dtype=np.float32), cov, area_eff, strength


# =============================================================================
# Patch motion and episode extraction
# =============================================================================


def motions_obtain(motion, mask, patch_size, return_abs=False):
    """
    Compute patch-level frame-to-frame motion and cumulative displacement.

    Parameters
    ----------
    motion : ndarray, shape (T,H,W,2)
        Cumulative displacement relative to reference.
    mask : ndarray, shape (H,W)
    patch_size : int
    return_abs : bool
        If True, return (motion_delta, motion_abs_aligned, mask_patched).
        motion_delta[t] = patch_abs[t+1] - patch_abs[t]
        motion_abs_aligned[t] = patch_abs[t+1]

    Returns
    -------
    motion_delta : (T-1, Xp, Yp, 2)
    motion_abs_aligned : (T-1, Xp, Yp, 2), optional
    mask_patched : (Xp, Yp), bool
    """
    # motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 4 or motion.shape[-1] != 2:
        raise ValueError(f"Expected motion shape (T,H,W,2), got {motion.shape}")

    T, H, W, _ = motion.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
    H_ = n_rows * patch_size
    W_ = n_cols * patch_size

    motion = motion[:, :H_, :W_, :]
    blocks = motion.reshape(T, n_rows, patch_size, n_cols, patch_size, 2)
    patch_abs = blocks.mean(axis=(2, 4))  # (T, Xp, Yp, 2)
    patch_delta = patch_abs[1:] - patch_abs[:-1]
    patch_abs_aligned = patch_abs[1:]

    mask = np.asarray(mask)
    mask_patched = (
        mask[:H_, :W_].reshape(n_rows, patch_size, n_cols, patch_size).mean(axis=(1, 3)) > 0.3
    )

    if return_abs:
        return patch_delta.astype(np.float32), patch_abs_aligned.astype(np.float32), mask_patched.astype(bool)
    return patch_delta.astype(np.float32), mask_patched.astype(bool)


def estimate_rest_state_motion(
    motionMag_patched,
    window_size_t=21,
    window_size_xy=5,
    scale=5.0,
    use_gpu="auto",
):
    """
    Estimate resting-state motion fluctuation.

    If CuPy is available, use GPU median filters, matching the old implementation.
    """
    use_gpu = (HAS_CUPY if use_gpu == "auto" else bool(use_gpu))

    motionMag_np = np.asarray(motionMag_patched, dtype=np.float32)
    T, X, Y = motionMag_np.shape

    wt = int(max(3, min(window_size_t, T if T % 2 == 1 else max(T - 1, 3))))
    mad_t = max(3, T // 4)

    if use_gpu:
        motion_gpu = cp.asarray(motionMag_np)

        median_local = cupy_ndi.median_filter(
            motion_gpu,
            size=(wt, window_size_xy, window_size_xy),
            mode="reflect",
        )

        abs_dev = cp.abs(motion_gpu - median_local)

        mad_local = cupy_ndi.median_filter(
            abs_dev,
            size=(mad_t, window_size_xy + 2, window_size_xy + 2),
            mode="reflect",
        )

        rest_state_motion = scale * 1.4826 * mad_local
        return cp.asnumpy(rest_state_motion).astype(np.float32)

    # CPU fallback
    median_local = ndi.median_filter(
        motionMag_np,
        size=(wt, window_size_xy, window_size_xy),
        mode="reflect",
    )

    abs_dev = np.abs(motionMag_np - median_local)

    mad_local = ndi.median_filter(
        abs_dev,
        size=(mad_t, window_size_xy + 2, window_size_xy + 2),
        mode="reflect",
    )

    return (scale * 1.4826 * mad_local).astype(np.float32)


def getMotionUnit(
    motion_patched,
    motionMag_patched,
    restMotion,
    extend_radius=1,
    save_motion=False,
    use_gpu="auto",
):
    """
    Extract active intervals for each patch.

    GPU path follows the old implementation:
    - active mask on GPU
    - start/end interval detection on GPU
    - only interval grouping on CPU

    save_motion=False is recommended for the new pipeline.
    """
    use_gpu = (HAS_CUPY if use_gpu == "auto" else bool(use_gpu))

    motion_np = np.asarray(motion_patched, dtype=np.float32)
    motionMag_np = np.asarray(motionMag_patched, dtype=np.float32)
    rest_np = np.asarray(restMotion, dtype=np.float32)

    T, X, Y, _ = motion_np.shape

    if use_gpu:
        motionMag_gpu = cp.asarray(motionMag_np)
        rest_gpu = cp.asarray(rest_np)

        active = motionMag_gpu > rest_gpu  # (T, X, Y)

        prev_active = cp.zeros_like(active)
        prev_active[1:] = active[:-1]

        next_active = cp.zeros_like(active)
        next_active[:-1] = active[1:]

        start_mask = active & (~prev_active)
        end_mask = active & (~next_active)

        start_t, start_x, start_y = [cp.asnumpy(a) for a in cp.where(start_mask)]
        end_t, end_x, end_y = [cp.asnumpy(a) for a in cp.where(end_mask)]

        start_lin = start_x * Y + start_y
        end_lin = end_x * Y + end_y

        start_order = np.argsort(start_lin, kind="stable")
        end_order = np.argsort(end_lin, kind="stable")

        start_lin = start_lin[start_order]
        start_t = start_t[start_order]

        end_lin = end_lin[end_order]
        end_t = end_t[end_order]

        units_map = [[[] for _ in range(Y)] for _ in range(X)]
        active_mask = np.zeros((T, X, Y), dtype=bool)

        i0 = 0
        j0 = 0
        n_start = len(start_lin)
        n_end = len(end_lin)

        while i0 < n_start:
            lin_id = start_lin[i0]

            i1 = i0
            while i1 < n_start and start_lin[i1] == lin_id:
                i1 += 1

            while j0 < n_end and end_lin[j0] < lin_id:
                j0 += 1

            j1 = j0
            while j1 < n_end and end_lin[j1] == lin_id:
                j1 += 1

            starts_patch = start_t[i0:i1]
            ends_patch = end_t[j0:j1]

            m = min(len(starts_patch), len(ends_patch))
            if m == 0:
                i0 = i1
                j0 = j1
                continue

            x = int(lin_id // Y)
            y = int(lin_id % Y)

            intervals = []
            for s, e in zip(starts_patch[:m], ends_patch[:m]):
                s2 = max(0, int(s) - int(extend_radius))
                e2 = min(T - 1, int(e) + int(extend_radius))
                intervals.append([s2, e2])

            intervals.sort()
            merged = []
            for s, e in intervals:
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)

            for s, e in merged:
                active_mask[s:e + 1, x, y] = True
                mseg = motion_np[s:e + 1, x, y, :].copy() if save_motion else None
                units_map[x][y].append(
                    MotionUnit(
                        time_range=[s, e],
                        spatial_coor=[x, y],
                        motion=mseg,
                    )
                )

            i0 = i1
            j0 = j1

        return units_map, active_mask

    # CPU fallback: original simple version
    active = motionMag_np > rest_np

    units_map = [[[] for _ in range(Y)] for _ in range(X)]
    active_mask = np.zeros((T, X, Y), dtype=bool)

    for x in range(X):
        for y in range(Y):
            a = active[:, x, y]
            if not np.any(a):
                continue

            padded = np.concatenate([[False], a, [False]])
            diff = np.diff(padded.astype(np.int8))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0] - 1

            intervals = []
            for s, e in zip(starts, ends):
                s2 = max(0, int(s) - int(extend_radius))
                e2 = min(T - 1, int(e) + int(extend_radius))
                intervals.append([s2, e2])

            intervals.sort()
            merged = []
            for s, e in intervals:
                if not merged or s > merged[-1][1] + 1:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)

            for s, e in merged:
                active_mask[s:e + 1, x, y] = True
                mseg = motion_np[s:e + 1, x, y, :].copy() if save_motion else None
                units_map[x][y].append(
                    MotionUnit(
                        time_range=[s, e],
                        spatial_coor=[x, y],
                        motion=mseg,
                    )
                )

    return units_map, active_mask


def filterMotionUnits(
    units_map,
    active_mask,
    k=3,
    n=7,
    thresh=0.5,
    use_gpu="auto",
):
    """
    Remove isolated MotionUnits using local spatiotemporal active support.
    """
    use_gpu = (HAS_CUPY if use_gpu == "auto" else bool(use_gpu))

    active_np = np.asarray(active_mask, dtype=np.float32)

    if use_gpu:
        active_gpu = cp.asarray(active_np)
        smoothed_gpu = cupy_ndi.uniform_filter(
            active_gpu,
            size=(k, n, n),
            mode="nearest",
        )
        smoothed = cp.asnumpy(smoothed_gpu)
    else:
        smoothed = ndi.uniform_filter(
            active_np,
            size=(k, n, n),
            mode="nearest",
        )

    T, X, Y = active_np.shape
    refined = [[[] for _ in range(Y)] for _ in range(X)]

    for x in range(X):
        for y in range(Y):
            for unit in units_map[x][y]:
                t0, t1 = unit.time_range
                if float(smoothed[t0:t1 + 1, x, y].mean()) >= thresh:
                    refined[x][y].append(unit)

    return refined

def _compute_global_motion_series(motion_full_abs, valid_mask=None, mode="median"):
    arr = np.asarray(motion_full_abs, dtype=np.float32)  # (T,X,Y,2)
    if arr.ndim != 4 or arr.shape[-1] != 2:
        raise ValueError(f"motion_full_abs should have shape (T,X,Y,2), got {arr.shape}")
    T, X, Y, C = arr.shape
    if valid_mask is not None:
        vm = np.asarray(valid_mask).astype(bool)
        if vm.shape != (X, Y):
            raise ValueError(f"valid_mask shape {vm.shape} != motion spatial shape {(X,Y)}")
        vals = arr[:, vm, :]  # (T,N,2)
    else:
        vals = arr.reshape(T, X * Y, C)
    if vals.shape[1] == 0:
        vals = arr.reshape(T, X * Y, C)
    if mode == "median":
        g = np.median(vals, axis=1)
    elif mode == "mean":
        g = np.mean(vals, axis=1)
    elif mode == "zero":
        g = np.zeros((T, 2), dtype=np.float32)
    else:
        raise ValueError(f"Unknown global_motion_mode: {mode}")
    return g.astype(np.float32)


def getMotionEpisode(
    motion_units,
    motion_full,
    motion_full_abs=None,
    global_valid_mask=None,
    global_motion_mode="median",
    tolerant_time=1,
    min_total_area=30,
    repair_mask=True,
    closing_iter=1,
    min_cc_area=4,
    dilation_iter=3,
    overlap_threshold=0.3,
):
    """Build MotionEpisodes by grouping MotionUnits with similar time and overlapping region."""
    if nx is None:
        raise ImportError("networkx is required for getMotionEpisode().")

    motion_full = np.asarray(motion_full, dtype=np.float32)
    if motion_full_abs is None:
        motion_full_abs = motion_full
    motion_full_abs = np.asarray(motion_full_abs, dtype=np.float32)

    Xp, Yp = len(motion_units), len(motion_units[0])
    time_dict = defaultdict(list)

    for x in range(Xp):
        for y in range(Yp):
            for mu in motion_units[x][y]:
                time_dict[tuple(mu.time_range)].append(tuple(mu.spatial_coor))

    preliminary_masks = []
    time_ranges = []
    areas = []
    bboxes = []
    structure = np.ones((3, 3), dtype=bool)

    for time_range, coords in time_dict.items():
        mask = np.zeros((Xp, Yp), dtype=bool)
        for x, y in coords:
            mask[int(x), int(y)] = True
        if repair_mask:
            mask = ndi.binary_closing(mask, structure=structure, iterations=int(closing_iter))
            mask = ndi.binary_fill_holes(mask)
            lab, num = ndi.label(mask, structure=structure)
            if num > 0:
                sizes = np.bincount(lab.ravel())
                keep_ids = np.where(sizes >= int(min_cc_area))[0]
                keep_ids = keep_ids[keep_ids != 0]
                mask = np.isin(lab, keep_ids) if len(keep_ids) > 0 else np.zeros_like(mask)
        area = int(mask.sum())
        if area == 0:
            continue
        xs, ys = np.nonzero(mask)
        preliminary_masks.append(mask)
        time_ranges.append([int(time_range[0]), int(time_range[1])])
        areas.append(area)
        bboxes.append((int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())))

    n = len(preliminary_masks)
    if n == 0:
        return []

    time_ranges = np.asarray(time_ranges, dtype=np.int32)
    areas = np.asarray(areas, dtype=np.float32)
    global_motion_all = _compute_global_motion_series(motion_full_abs, valid_mask=global_valid_mask, mode=global_motion_mode)

    dilated_masks = [ndi.binary_dilation(m, iterations=int(dilation_iter)) for m in preliminary_masks]

    time_bucket = defaultdict(list)
    for idx, (t0, t1) in enumerate(time_ranges):
        time_bucket[(int(t0), int(t1))].append(idx)

    def bbox_overlap(box1, box2, margin):
        x1min, x1max, y1min, y1max = box1
        x2min, x2max, y2min, y2max = box2
        return not (
            x1max + margin < x2min or x2max + margin < x1min or
            y1max + margin < y2min or y2max + margin < y1min
        )

    G = nx.Graph()
    G.add_nodes_from(range(n))
    dt_pairs = [(ds, de) for ds in range(-tolerant_time, tolerant_time + 1)
                for de in range(-tolerant_time, tolerant_time + 1)]

    for i in range(n):
        t0, t1 = time_ranges[i]
        candidates = []
        for ds, de in dt_pairs:
            candidates.extend(time_bucket.get((int(t0 + ds), int(t1 + de)), []))
        for j in candidates:
            if j <= i:
                continue
            if not bbox_overlap(bboxes[i], bboxes[j], margin=int(dilation_iter)):
                continue
            inter = np.count_nonzero(dilated_masks[i] & preliminary_masks[j])
            if inter == 0:
                continue
            ratio = float(inter) / max(float(min(areas[i], areas[j])), 1.0)
            if ratio > overlap_threshold:
                G.add_edge(i, j)

    episodes = []
    for comp in nx.connected_components(G):
        comp = list(comp)
        t_min = int(time_ranges[comp, 0].min())
        t_max = int(time_ranges[comp, 1].max())
        mask = np.zeros_like(preliminary_masks[0], dtype=bool)
        for idx in comp:
            mask |= preliminary_masks[idx]
        if repair_mask:
            mask = ndi.binary_closing(mask, structure=structure, iterations=int(closing_iter))
            mask = ndi.binary_fill_holes(mask)
        if int(mask.sum()) < min_total_area:
            continue
        motion_delta_seg = motion_full[t_min:t_max + 1][:, mask, :]
        motion_abs_seg = motion_full_abs[t_min:t_max + 1][:, mask, :]
        global_motion_seg = global_motion_all[t_min:t_max + 1]
        ep = MotionEpisode(
            time_range=[t_min, t_max],
            region_mask=mask.astype(np.uint8),
            episode_id=len(episodes),
            motion_delta=motion_delta_seg,
            motion_abs=motion_abs_seg,
            global_motion=global_motion_seg,
            global_motion_mode=global_motion_mode,
        )
        episodes.append(ep)
    return episodes


# =============================================================================
# Motion mode decomposition: Y_i(t) ~= sum h_k(t) b_ik
# =============================================================================


def _second_diff_matrix(T):
    if T < 3:
        return np.zeros((0, T), dtype=np.float32)
    D2 = np.zeros((T - 2, T), dtype=np.float32)
    for i in range(T - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


def _initialize_modes_spatial_seed(Y_data, valid_coords, Kmax=4, rng=None, min_seed_dist=3.0, eps=1e-12):
    if rng is None:
        rng = np.random.default_rng(0)
    T, N, _ = Y_data.shape
    K = min(int(Kmax), N)
    energy = np.sum(Y_data ** 2, axis=(0, 2))
    order = np.argsort(energy)[::-1]
    seeds = []
    for idx in order:
        if len(seeds) >= K:
            break
        xy = valid_coords[idx].astype(np.float32)
        if len(seeds) == 0:
            seeds.append(int(idx))
        else:
            prev = valid_coords[np.asarray(seeds)].astype(np.float32)
            d = np.linalg.norm(prev - xy[None, :], axis=1)
            if np.min(d) >= min_seed_dist:
                seeds.append(int(idx))
    while len(seeds) < K:
        cand = int(rng.integers(0, N))
        if cand not in seeds:
            seeds.append(cand)

    H = np.zeros((K, T), dtype=np.float32)
    for k, idx in enumerate(seeds):
        traj = Y_data[:, idx, :]  # (T,2)
        mean_vec = np.mean(traj, axis=0)
        nrm = np.linalg.norm(mean_vec)
        if nrm < eps:
            h = np.linalg.norm(traj, axis=1)
        else:
            h = traj @ (mean_vec / nrm)
        hn = np.linalg.norm(h)
        if hn < eps:
            h = rng.normal(size=T).astype(np.float32)
            hn = np.linalg.norm(h)
        H[k] = h / max(float(hn), eps)

    M = np.concatenate([Y_data[:, :, 0].T, Y_data[:, :, 1].T], axis=0).astype(np.float32)
    HHt = H @ H.T + 1e-6 * np.eye(K, dtype=np.float32)
    B = M @ H.T @ np.linalg.inv(HHt)
    return H.astype(np.float32), B.astype(np.float32), seeds


def _group_soft_threshold_B(B, thresh, eps=1e-12):
    B = np.asarray(B, dtype=np.float32).copy()
    N2, K = B.shape
    N = N2 // 2
    Bx = B[:N, :]
    By = B[N:, :]
    norm = np.sqrt(Bx ** 2 + By ** 2)
    scale = np.maximum(0.0, 1.0 - thresh / np.maximum(norm, eps))
    B[:N, :] = Bx * scale
    B[N:, :] = By * scale
    return B


def _mode_column_soft_threshold_B(B, thresh, eps=1e-12):
    B = np.asarray(B, dtype=np.float32).copy()
    col_norm = np.linalg.norm(B, axis=0)
    scale = np.maximum(0.0, 1.0 - thresh / np.maximum(col_norm, eps))
    return B * scale[None, :]


def _fit_motion_modes_minimal(
    M,
    B,
    H,
    lambda_B=0.05,
    lambda_H=0.01,
    lambda_mode=0.01,
    max_iter=100,
    tol=1e-4,
    verbose=True,
    eps=1e-12,
    scaled_B_penalty=True,
    B_scale=None,
):
    M = np.asarray(M, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32)

    N2, T = M.shape
    N = N2 // 2
    K = H.shape[0]

    if K == 0:
        return B, H, []

    D2 = _second_diff_matrix(T)
    L = D2.T @ D2

    total_energy = float(np.sum(M ** 2)) + eps

    if B_scale is None:
        B_scale = _estimate_motion_scale_from_M(M, eps=eps)

    # ------------------------------------------------------------
    # Convert normalized penalties to equivalent raw proximal lambdas
    # ------------------------------------------------------------
    if scaled_B_penalty:
        # patch loss:
        # lambda_B * mean_{i,k} ||b_ik / B_scale||
        lambda_B_eff = lambda_B / (max(N * K * B_scale, eps))

        # mode loss:
        # lambda_mode * mean_k sqrt(mean_i ||b_ik / B_scale||^2)
        # = lambda_mode / (K * sqrt(N) * B_scale) * sum_k ||B_k||
        lambda_mode_eff = lambda_mode / (max(K * np.sqrt(max(N, 1)) * B_scale, eps))
    else:
        lambda_B_eff = lambda_B
        lambda_mode_eff = lambda_mode

    loss_history = []

    for it in range(max_iter):
        B_old = B.copy()
        H_old = H.copy()

        # -------------------------
        # update B
        # -------------------------
        HHt = H @ H.T
        step_B = 1.0 / (np.linalg.norm(HHt, ord=2) + eps)

        # normalized reconstruction gradient
        grad_B = (2.0 / total_energy) * (B @ H - M) @ H.T

        lip_B = (2.0 / total_energy) * np.linalg.norm(HHt, ord=2)
        step_B = 1.0 / (lip_B + eps)

        B_tmp = B - step_B * grad_B

        tau_patch = step_B * lambda_B_eff
        tau_mode = step_B * lambda_mode_eff

        B_tmp = _group_soft_threshold_B(
            B_tmp,
            tau_patch,
            eps=eps,
        )

        B = _mode_column_soft_threshold_B(
            B_tmp,
            tau_mode,
            eps=eps,
        )

        # -------------------------
        # update H
        # -------------------------
        smooth_norm = max(K * max(T - 2, 1), 1)

        BtB = B.T @ B

        lip_H = (
            (2.0 / total_energy) * np.linalg.norm(BtB, ord=2)
            + (2.0 * lambda_H / smooth_norm) * np.linalg.norm(L, ord=2)
            + eps
        )

        step_H = 1.0 / lip_H

        grad_H = (
            (2.0 / total_energy) * B.T @ (B @ H - M)
            + (2.0 * lambda_H / smooth_norm) * (H @ L)
        )

        H = H - step_H * grad_H
        # normalize H and absorb scale into B
        for k in range(K):
            nrm = np.linalg.norm(H[k])
            if nrm > eps:
                H[k] /= nrm
                B[:, k] *= nrm

        # -------------------------
        # diagnostics / loss logging
        # -------------------------
        R = M - B @ H

        recon_raw = 0.5 * float(np.sum(R ** 2))
        recon_norm = float(np.sum(R ** 2) / total_energy)

        if scaled_B_penalty:
            pen = _compute_B_penalties_scaled(
                B,
                lambda_B=lambda_B,
                lambda_mode=lambda_mode,
                B_scale=B_scale,
                eps=eps,
            )
            patch_sparse = pen["patch_loss"]
            mode_sparse = pen["mode_loss"]
            raw_patch = pen["raw_patch"]
            raw_mode = pen["raw_mode"]
        else:
            bmag = np.sqrt(B[:N, :] ** 2 + B[N:, :] ** 2 + eps)
            patch_sparse = lambda_B * float(np.sum(bmag))
            mode_sparse = lambda_mode * float(np.sum(np.linalg.norm(B, axis=0)))
            raw_patch = float(np.sum(bmag))
            raw_mode = float(np.sum(np.linalg.norm(B, axis=0)))

        # smooth 也用 mean 打印，避免 T/K 不同导致不可比
        if T >= 3:
            smooth_raw = float(np.mean((H @ D2.T) ** 2))
            smooth = lambda_H * smooth_raw
        else:
            smooth_raw = 0.0
            smooth = 0.0

        # 注意：这里的 loss 是 normalized diagnostic loss，不再是 raw loss
        loss = recon_norm + patch_sparse + mode_sparse + smooth
        loss_history.append(loss)

        delta = float(
            np.mean(np.abs(B - B_old))
            + np.mean(np.abs(H - H_old))
        )

        if verbose:
            print(
                f"[mode fit] iter={it:03d}, "
                f"loss={loss:.6f}, "
                f"recon_raw={recon_raw:.6f}, "
                f"recon_norm={recon_norm:.6e}, "
                f"patch={patch_sparse:.6f}, "
                f"mode={mode_sparse:.6f}, "
                f"smooth={smooth:.6f}, "
                f"raw_patch={raw_patch:.6f}, "
                f"raw_mode={raw_mode:.6f}, "
                f"B_scale={B_scale:.6f}, "
                f"delta={delta:.3e}"
            )

        if delta < tol:
            break

    return B.astype(np.float32), H.astype(np.float32), loss_history


def _prune_BH_modes(
    B,
    H,
    M,
    support_rel_thresh=0.10,
    min_mode_mass=1e-3,
    min_incremental_energy=0.005,
    min_support_area=3,
    max_mode_density=1.0,
    eps=1e-12,
):
    B = np.asarray(B, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)
    N2, T = M.shape
    N = N2 // 2
    K = H.shape[0]
    if K == 0:
        return B, H, {}
    strength = np.sqrt(B[:N, :] ** 2 + B[N:, :] ** 2)
    mass = np.sum(strength, axis=0)
    support_area = np.zeros(K, dtype=np.int32)
    density = np.zeros(K, dtype=np.float32)
    for k in range(K):
        s = strength[:, k]
        vmax = float(np.max(s))
        if vmax > eps:
            support = s > support_rel_thresh * vmax
            support_area[k] = int(np.sum(support))
            density[k] = float(support_area[k] / max(N, 1))
    R_full = M - B @ H
    err_full = float(np.sum(R_full ** 2))
    total = float(np.sum(M ** 2)) + eps
    incremental = np.zeros(K, dtype=np.float32)
    for k in range(K):
        B_wo = B.copy()
        B_wo[:, k] = 0.0
        err_wo = float(np.sum((M - B_wo @ H) ** 2))
        incremental[k] = max(0.0, (err_wo - err_full) / total)
    keep = []
    for k in range(K):
        if mass[k] < min_mode_mass:
            continue
        if incremental[k] < min_incremental_energy:
            continue
        if support_area[k] < min_support_area:
            continue
        if density[k] > max_mode_density:
            continue
        keep.append(k)
    if len(keep) == 0:
        keep = [int(np.argmax(incremental))]
    keep = np.asarray(keep, dtype=np.int64)
    info = {
        "mode_mass": mass,
        "support_area": support_area,
        "density": density,
        "incremental": incremental,
        "keep": keep,
        "K_before_prune": K,
        "K_after_prune": len(keep),
    }
    return B[:, keep].astype(np.float32), H[keep, :].astype(np.float32), info


def _build_motion_modes_from_BH(
    B,
    H,
    M,
    Y_data,
    valid_coords,
    mask_shape,
    episode_id,
    time_range,
    global_motion,
    min_mode_mass=1e-3,
    min_explained_energy=0.01,
    support_rel_thresh=0.10,
    eps=1e-12,
):
    N2, T = M.shape
    N = N2 // 2
    K = H.shape[0]
    X, Y = mask_shape
    modes = []
    total_energy = float(np.sum(M ** 2)) + eps
    for k in range(K):
        response_compact = np.stack([B[:N, k], B[N:, k]], axis=1).astype(np.float32)
        strength = np.linalg.norm(response_compact, axis=1)
        mass = float(np.sum(strength))
        if mass < min_mode_mass:
            continue
        Mk = B[:, [k]] @ H[[k], :]
        explained = float(np.sum(Mk ** 2) / total_energy)
        if explained < min_explained_energy:
            continue
        field = np.zeros((X, Y, 2), dtype=np.float32)
        A = np.zeros((X, Y), dtype=np.float32)
        for idx, (r, c) in enumerate(valid_coords):
            field[r, c, :] = response_compact[idx]
            A[r, c] = strength[idx]
        vmax = float(np.max(A))
        support = A > support_rel_thresh * vmax if vmax > eps else np.zeros((X, Y), dtype=bool)
        direction = field / np.maximum(A[:, :, None], eps)
        h = H[k].astype(np.float32)
        recon_k = h[:, None, None] * response_compact[None, :, :]
        resid_k = Y_data - recon_k
        residual_energy = float(np.sum(resid_k ** 2) / total_energy)
        seed_position = tuple(np.unravel_index(np.argmax(A), A.shape)) if np.any(A > 0) else None
        mode = MotionMode(
            episode_id=episode_id,
            mode_id=len(modes),
            time_range=list(time_range),
            activation=h,
            response_field=field,
            response_strength=A,
            response_direction=direction,
            support_mask=support.astype(np.uint8),
            compact_response_field=response_compact,
            valid_coords=valid_coords,
            background_motion=global_motion,
            reconstructed_motion=recon_k,
            residual_motion=resid_k,
            explained_energy=explained,
            residual_energy=residual_energy,
            mode_mass=mass,
            seed_position=seed_position,
            confidence=1.0,
            metadata={"support_rel_thresh": support_rel_thresh, "model": "motion_mode_B"},
        )
        modes.append(mode)
    return modes


def _compute_recon_loss(M, B, H, eps=1e-12):
    """
    M: (D, T)
    B: (D, K)
    H: (K, T)
    """
    R = M - B @ H
    return 0.5 * float(np.sum(R * R))


def _compute_recon_r2(M, B, H, eps=1e-12):
    loss = _compute_recon_loss(M, B, H, eps=eps)
    energy = 0.5 * float(np.sum(M * M)) + eps
    return 1.0 - loss / energy


def _normalize_BH(B, H, eps=1e-8):
    """
    Normalize each row of H to unit norm and absorb scale into B columns.

    B: (D, K)
    H: (K, T)
    """
    B = np.asarray(B, dtype=np.float32).copy()
    H = np.asarray(H, dtype=np.float32).copy()

    K = H.shape[0]
    for k in range(K):
        n = float(np.linalg.norm(H[k]))
        if n > eps:
            H[k] /= n
            B[:, k] *= n

    return B, H


def _activation_merge_groups(H, activation_merge_thresh=0.95, eps=1e-8):
    """
    Build merge groups using sign-invariant cosine similarity between activations.

    H: (K, T)
    """
    H = np.asarray(H, dtype=np.float32)
    K = H.shape[0]

    Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), eps)
    sim = np.abs(Hn @ Hn.T)

    parent = list(range(K))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(K):
        for j in range(i + 1, K):
            if sim[i, j] >= activation_merge_thresh:
                union(i, j)

    groups_dict = {}
    for k in range(K):
        groups_dict.setdefault(find(k), []).append(k)

    groups = list(groups_dict.values())
    return groups, sim


def _rank1_merge_one_group(B_group, H_group, eps=1e-8):
    """
    Merge one group by finding the best rank-1 approximation to:

        M_group = B_group @ H_group

    B_group: (D, Kg)
    H_group: (Kg, T)

    Return:
        b_new: (D,)
        h_new: (T,)
        rel_err: rank-1 reconstruction relative error of this group
    """
    M_group = B_group @ H_group  # (D, T)

    energy = float(np.sum(M_group * M_group)) + eps
    if energy <= eps:
        D, _ = B_group.shape
        T = H_group.shape[1]
        return np.zeros(D, dtype=np.float32), np.zeros(T, dtype=np.float32), 1.0

    # Since T is usually small, SVD of (D,T) is acceptable.
    U, S, Vt = np.linalg.svd(M_group, full_matrices=False)

    b_new = (U[:, 0] * S[0]).astype(np.float32)  # (D,)
    h_new = Vt[0].astype(np.float32)             # (T,)

    M_rank1 = b_new[:, None] @ h_new[None, :]
    err = float(np.sum((M_group - M_rank1) ** 2))
    rel_err = err / energy

    return b_new, h_new, rel_err


def _refit_B_given_H(M, H, ridge=1e-6):
    """
    Given M and H, solve optimal B in least squares sense:

        min_B ||M - B H||_F^2

    B = M H^T (H H^T + ridge I)^(-1)

    M: (D, T)
    H: (K, T)

    Return:
        B: (D, K)
    """
    H = np.asarray(H, dtype=np.float32)
    M = np.asarray(M, dtype=np.float32)

    K = H.shape[0]
    gram = H @ H.T
    gram = gram + ridge * np.eye(K, dtype=np.float32)

    rhs = M @ H.T  # (D, K)

    # Solve gram.T X.T = rhs.T
    B = np.linalg.solve(gram.T, rhs.T).T
    return B.astype(np.float32)

def _select_elbow_k_from_curve(K_values, r2_values, target_r2=0.98):
    """
    Select K by:
    1. If target_r2 is reached, choose the smallest K reaching it.
    2. Otherwise use geometric elbow: max distance to line between endpoints.
    """
    K_values = np.asarray(K_values, dtype=np.float32)
    r2_values = np.asarray(r2_values, dtype=np.float32)

    valid = np.isfinite(r2_values)
    K_values = K_values[valid]
    r2_values = r2_values[valid]

    if len(K_values) == 0:
        raise ValueError("No valid K/R2 values for K selection.")

    # Rule 1: smallest K reaching target R2
    hit = np.where(r2_values >= target_r2)[0]
    if len(hit) > 0:
        idx = int(hit[0])
        return int(K_values[idx]), {
            "method": "target_r2",
            "target_r2": float(target_r2),
            "selected_idx": idx,
        }

    # Rule 2: geometric elbow
    if len(K_values) <= 2:
        idx = int(np.argmax(r2_values))
        return int(K_values[idx]), {
            "method": "best_r2_fallback",
            "selected_idx": idx,
        }

    x = (K_values - K_values.min()) / (K_values.max() - K_values.min() + 1e-12)
    y = (r2_values - r2_values.min()) / (r2_values.max() - r2_values.min() + 1e-12)

    p0 = np.array([x[0], y[0]], dtype=np.float32)
    p1 = np.array([x[-1], y[-1]], dtype=np.float32)

    line = p1 - p0
    line_norm = np.linalg.norm(line) + 1e-12

    dists = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi], dtype=np.float32)
        dist = abs(np.cross(line, p - p0)) / line_norm
        dists.append(dist)

    idx = int(np.argmax(dists))
    return int(K_values[idx]), {
        "method": "geometric_elbow",
        "selected_idx": idx,
        "target_r2": float(target_r2),
    }

def _select_K_by_svd_energy(
    M,
    target_r2=0.85,
    Kmax=None,
    K_min=1,
    eps=1e-12,
):
    """
    Select K by cumulative SVD explained energy.

    M: (2N, T)

    Return
    ------
    selected_K : int
    info : dict
    """
    M = np.asarray(M, dtype=np.float32)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    energy = S ** 2
    total_energy = float(np.sum(energy)) + eps
    cum_r2 = np.cumsum(energy) / total_energy

    rank_max = len(S)

    if Kmax is None:
        Kmax_eff = rank_max
    else:
        Kmax_eff = min(int(Kmax), rank_max)

    K_min = max(1, int(K_min))
    Kmax_eff = max(K_min, Kmax_eff)

    selected_K = Kmax_eff

    for k in range(K_min, Kmax_eff + 1):
        if cum_r2[k - 1] >= target_r2:
            selected_K = k
            break

    records = []
    for k in range(1, rank_max + 1):
        records.append(
            {
                "K": int(k),
                "svd_r2": float(cum_r2[k - 1]),
                "singular_value": float(S[k - 1]),
                "energy_fraction": float(energy[k - 1] / total_energy),
            }
        )

    info = {
        "method": "svd_cumulative_energy",
        "target_r2": float(target_r2),
        "selected_K": int(selected_K),
        "K_min": int(K_min),
        "Kmax_input": None if Kmax is None else int(Kmax),
        "Kmax_eff": int(Kmax_eff),
        "rank_max": int(rank_max),
        "singular_values": S.astype(np.float32),
        "cum_r2": cum_r2.astype(np.float32),
        "selected_svd_r2": float(cum_r2[selected_K - 1]),
        "records": records,
    }

    return int(selected_K), info

def _sweep_K_for_episode_modes(
    M,
    Y_data,
    valid_coords,
    Kmax=20,
    K_min=1,
    K_list=None,
    n_init=3,
    short_iter=10,
    lambda_B=0.001,
    lambda_H=0.05,
    lambda_mode=0.005,
    tol=1e-4,
    target_r2=0.98,
    verbose=True,
    random_state=0,
    eps=1e-12,
):
    """
    Try different K and choose an elbow/target K.

    For each K:
        run n_init initializations
        fit only short_iter
        keep best R2
    """
    rng = np.random.default_rng(random_state)

    if K_list is None:
        K_list = list(range(K_min, Kmax + 1))
    else:
        K_list = [int(k) for k in K_list if int(k) >= K_min and int(k) <= Kmax]

    records = []
    best_by_K = {}

    B_scale = _estimate_motion_scale_from_M(M, eps=eps)

    for K in K_list:
        best = None

        for s in range(n_init):
            seed = int(rng.integers(0, 2**31 - 1))
            local_rng = np.random.default_rng(seed)

            H0, B0, seeds = _initialize_modes_spatial_seed(
                Y_data,
                valid_coords,
                Kmax=K,
                rng=local_rng,
                eps=eps,
            )

            if short_iter > 0:
                B_fit, H_fit, loss_hist = _fit_motion_modes_minimal(
                    M,
                    B0,
                    H0,
                    lambda_B=lambda_B,
                    lambda_H=lambda_H,
                    lambda_mode=lambda_mode,
                    max_iter=short_iter,
                    tol=tol,
                    verbose=False,
                    eps=eps,
                    scaled_B_penalty=True,
                    B_scale=B_scale,
                )
            else:
                B_fit, H_fit = B0, H0
                loss_hist = []

            r2 = _compute_recon_r2(M, B_fit, H_fit, eps=eps)
            recon_norm = _compute_recon_loss_normalized(M, B_fit, H_fit, eps=eps)

            pen = _compute_B_penalties_scaled(
                B_fit,
                lambda_B=lambda_B,
                lambda_mode=lambda_mode,
                B_scale=B_scale,
                eps=eps,
            )

            rec = {
                "K": int(K),
                "seed": seed,
                "r2": float(r2),
                "recon_norm": float(recon_norm),
                "patch_loss_scaled": float(pen["patch_loss"]),
                "mode_loss_scaled": float(pen["mode_loss"]),
                "raw_patch_scaled": float(pen["raw_patch"]),
                "raw_mode_scaled": float(pen["raw_mode"]),
                "B": B_fit,
                "H": H_fit,
                "seeds": seeds,
                "loss_history": loss_hist,
            }

            if best is None or rec["r2"] > best["r2"]:
                best = rec

        best_by_K[K] = best
        records.append({
            k: v for k, v in best.items()
            if k not in ("B", "H", "seeds", "loss_history")
        })

        if verbose:
            print(
                f"[K sweep] K={K:02d}, "
                f"best_R2={best['r2']:.6f}, "
                f"recon_norm={best['recon_norm']:.6e}, "
                f"patch={best['patch_loss_scaled']:.6f}, "
                f"mode={best['mode_loss_scaled']:.6f}"
            )

    K_values = [r["K"] for r in records]
    r2_values = [r["r2"] for r in records]

    selected_K, select_info = _select_elbow_k_from_curve(
        K_values,
        r2_values,
        target_r2=target_r2,
    )

    selected = best_by_K[selected_K]

    if verbose:
        print(
            f"[K selected] K={selected_K}, "
            f"method={select_info['method']}, "
            f"R2={selected['r2']:.6f}"
        )

    return selected_K, selected, {
        "records": records,
        "select_info": select_info,
        "K_values": K_values,
        "r2_values": r2_values,
    }

def _merge_modes_reconstruction_preserving(
    M,
    B,
    H,
    groups,
    ridge=1e-6,
    max_r2_drop=0.03,
    max_group_rank1_error=0.20,
    reject_bad_groups=True,
    verbose=True,
):
    """
    Reconstruction-preserving merge.

    Steps:
        1. For each proposed group, approximate its original contribution
           B_group @ H_group by one rank-1 mode.
        2. Stack merged H.
        3. Given merged H, solve optimal B by least squares.
        4. Reject merge if global R2 drop is too large.

    Parameters
    ----------
    M : (D, T)
    B : (D, K)
    H : (K, T)
    groups : list[list[int]]
        Proposed groups from activation similarity.
    max_r2_drop : float
        Maximum allowed drop in reconstruction R2 after merge initialization.
        If exceeded, fall back to no merge.
    max_group_rank1_error : float
        If one proposed group is not well approximated by rank-1, optionally
        keep its members separate.
    reject_bad_groups : bool
        If True, a proposed group with high rank-1 error will not be merged.
    """
    M = np.asarray(M, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    H = np.asarray(H, dtype=np.float32)

    B, H = _normalize_BH(B, H)

    recon_before = _compute_recon_loss(M, B, H)
    r2_before = _compute_recon_r2(M, B, H)

    final_B_cols = []
    final_H_rows = []
    final_groups = []
    group_errors = []

    for group in groups:
        group = list(group)

        if len(group) == 1:
            k = group[0]
            final_B_cols.append(B[:, k].copy())
            final_H_rows.append(H[k].copy())
            final_groups.append(group)
            group_errors.append(0.0)
            continue

        B_group = B[:, group]       # (D, Kg)
        H_group = H[group, :]       # (Kg, T)

        b_new, h_new, rel_err = _rank1_merge_one_group(B_group, H_group)
        group_errors.append(float(rel_err))

        if reject_bad_groups and rel_err > max_group_rank1_error:
            # This group cannot be represented well by one merged mode.
            # Keep members separate.
            for k in group:
                final_B_cols.append(B[:, k].copy())
                final_H_rows.append(H[k].copy())
                final_groups.append([k])
        else:
            final_B_cols.append(b_new)
            final_H_rows.append(h_new)
            final_groups.append(group)

    B_init = np.stack(final_B_cols, axis=1).astype(np.float32)  # (D,Knew)
    H_init = np.stack(final_H_rows, axis=0).astype(np.float32)  # (Knew,T)

    B_init, H_init = _normalize_BH(B_init, H_init)

    # Important: after choosing merged H, recompute globally optimal B.
    B_ls = _refit_B_given_H(M, H_init, ridge=ridge)
    B_ls, H_init = _normalize_BH(B_ls, H_init)

    recon_after = _compute_recon_loss(M, B_ls, H_init)
    r2_after = _compute_recon_r2(M, B_ls, H_init)
    r2_drop = r2_before - r2_after

    accepted = True
    if r2_drop > max_r2_drop:
        # Merge destroys too much reconstruction. Fall back to original modes.
        accepted = False
        B_out, H_out = B, H
        final_groups = [[k] for k in range(H.shape[0])]
        recon_after = recon_before
        r2_after = r2_before
    else:
        B_out, H_out = B_ls, H_init

    diag = {
        "accepted": bool(accepted),
        "K_before": int(H.shape[0]),
        "K_after": int(H_out.shape[0]),
        "recon_before": float(recon_before),
        "recon_after_init": float(recon_after),
        "r2_before": float(r2_before),
        "r2_after_init": float(r2_after),
        "r2_drop": float(r2_drop),
        "group_rank1_errors": group_errors,
        "groups": final_groups,
    }

    if verbose:
        print(
            "[recon-preserving merge] "
            f"K {diag['K_before']} -> {diag['K_after']}, "
            f"accepted={diag['accepted']}, "
            f"recon {diag['recon_before']:.4f} -> {diag['recon_after_init']:.4f}, "
            f"R2 {diag['r2_before']:.4f} -> {diag['r2_after_init']:.4f}, "
            f"R2_drop={diag['r2_drop']:.4f}"
        )

    return B_out, H_out, final_groups, diag

def _estimate_motion_scale_from_M(M, eps=1e-12):
    """
    Estimate a natural motion scale from M.
    M: (2N, T)
    """
    M = np.asarray(M, dtype=np.float32)
    return float(np.sqrt(np.mean(M ** 2)) + eps)

def _compute_B_penalties_scaled(
    B,
    lambda_B=0.001,
    lambda_mode=0.005,
    B_scale=1.0,
    eps=1e-12,
):
    """
    Compute normalized patch/mode penalties from B.

    B: (2N, K)
    B_scale: typical motion scale. We penalize B / B_scale.

    patch penalty:
        mean_{i,k} ||b_ik / B_scale||

    mode penalty:
        mean_k sqrt(mean_i ||b_ik / B_scale||^2)
    """
    B = np.asarray(B, dtype=np.float32)
    D, K = B.shape
    N = D // 2

    Bx = B[:N, :]
    By = B[N:, :]

    bmag = np.sqrt(Bx ** 2 + By ** 2 + eps) / max(B_scale, eps)  # (N, K)

    raw_patch = float(np.mean(bmag))
    patch_loss = float(lambda_B * raw_patch)

    mode_rms = np.sqrt(np.mean(bmag ** 2, axis=0) + eps)  # (K,)
    raw_mode = float(np.mean(mode_rms))
    mode_loss = float(lambda_mode * raw_mode)

    return {
        "raw_patch": raw_patch,
        "patch_loss": patch_loss,
        "raw_mode": raw_mode,
        "mode_loss": mode_loss,
        "B_scale": float(B_scale),
        "B_abs_mean": float(np.mean(np.sqrt(Bx ** 2 + By ** 2 + eps))),
        "B_abs_max": float(np.max(np.sqrt(Bx ** 2 + By ** 2 + eps))),
    }

def _compute_recon_loss_normalized(M, B, H, eps=1e-12):
    """
    Normalized reconstruction error:
        ||M - BH||^2 / ||M||^2
    """
    R = M - B @ H
    return float(np.sum(R ** 2) / (np.sum(M ** 2) + eps))

def decompose_episode_motion_modes(
    episode: MotionEpisode,
    Kmax=4,
    lambda_B=0.05,
    lambda_H=0.01,
    lambda_mode=0.01,
    max_iter=100,
    tol=1e-4,
    min_mode_mass=1e-3,
    min_explained_energy=0.01,
    min_incremental_energy=0.005,
    min_support_area=3,
    max_mode_density=1.0,
    support_rel_thresh=0.10,

    # K selection
    K_selection_method="svd",   # "svd", "sweep", or "fixed"
    K_min=1,
    K_list=None,
    K_sweep_n_init=3,
    K_sweep_short_iter=10,

    # SVD K selection
    svd_target_r2=0.85,

    # old sweep target, only used when K_selection_method="sweep"
    target_r2=0.90,

    # normalized B penalty
    scaled_B_penalty=True,

    # merge control
    merge_redundant_modes=True,
    activation_merge_thresh=0.98,
    merge_ridge=1e-6,
    max_merge_r2_drop=0.03,
    max_group_rank1_error=0.20,
    reject_bad_merge_groups=True,

    refine_after_merge=True,
    final_refine_after_prune=True,
    verbose=True,
    random_state=0,
):
    """
    Decompose one MotionEpisode into MotionModes.

    Model:
        M ≈ B @ H

    where:
        M: (2N, T)
        B: (2N, K)
        H: (K, T)

    Main change:
        Redundant modes are no longer directly merged by averaging.
        We first propose merge groups from activation similarity,
        then perform reconstruction-preserving merge.
    """

    if episode.motion_abs is None:
        raise ValueError("episode.motion_abs is required for motion mode decomposition.")
    if episode.global_motion is None:
        raise ValueError("episode.global_motion is required for motion mode decomposition.")

    rng = np.random.default_rng(random_state)
    eps = 1e-12

    # ------------------------------------------------------------
    # 1. Prepare episode motion matrix M
    # ------------------------------------------------------------
    mask = np.asarray(episode.spatial_region).astype(bool)
    X, Y = mask.shape
    valid_coords = np.argwhere(mask)

    motion_abs = np.asarray(episode.motion_abs, dtype=np.float32)       # (T, N, 2)
    global_motion = np.asarray(episode.global_motion, dtype=np.float32) # (T, 2)

    T, N, C = motion_abs.shape
    if C != 2 or len(valid_coords) != N:
        raise ValueError(
            f"motion_abs shape {motion_abs.shape} inconsistent with "
            f"episode mask count {len(valid_coords)}"
        )

    if global_motion.shape != (T, 2):
        raise ValueError(
            f"global_motion should have shape {(T, 2)}, got {global_motion.shape}"
        )

    # remove global/background motion
    Y_data = motion_abs - global_motion[:, None, :]  # (T, N, 2)

    # M: (2N, T)
    M = np.concatenate(
        [
            Y_data[:, :, 0].T,
            Y_data[:, :, 1].T,
        ],
        axis=0,
    ).astype(np.float32)

    total_energy = float(np.sum(M ** 2)) + eps

    # ------------------------------------------------------------
    # 2. Select K, then full fit with selected K
    # ------------------------------------------------------------
    B_scale = _estimate_motion_scale_from_M(M, eps=eps)

    K_select_info = None
    selected_K = None

    # ------------------------------------------------------------
    # 2. Select K, then full fit with selected K
    # ------------------------------------------------------------
    B_scale = _estimate_motion_scale_from_M(M, eps=eps)

    K_select_info = None
    selected_K = None

    if K_selection_method == "svd":
        # --------------------------------------------------------
        # SVD only selects K.
        # We still initialize B/H by spatial seeds, not by SVD,
        # so the final decomposition is not forced to be SVD-like.
        # --------------------------------------------------------
        selected_K, K_select_info = _select_K_by_svd_energy(
            M,
            target_r2=svd_target_r2,
            Kmax=Kmax,
            K_min=K_min,
            eps=eps,
        )

        H, B, seeds = _initialize_modes_spatial_seed(
            Y_data,
            valid_coords,
            Kmax=selected_K,
            rng=rng,
            eps=eps,
        )

        K_init = H.shape[0]

        if verbose:
            print(
                f"[mode K selection: SVD] episode={episode.episode_id}, "
                f"Kmax={Kmax}, selected_K={selected_K}, "
                f"svd_target_r2={svd_target_r2}, "
                f"selected_svd_R2={K_select_info['selected_svd_r2']:.6f}, "
                f"rank_max={K_select_info['rank_max']}"
            )

    elif K_selection_method == "sweep":
        selected_K, selected_init, K_select_info = _sweep_K_for_episode_modes(
            M=M,
            Y_data=Y_data,
            valid_coords=valid_coords,
            Kmax=Kmax,
            K_min=K_min,
            K_list=K_list,
            n_init=K_sweep_n_init,
            short_iter=K_sweep_short_iter,
            lambda_B=lambda_B,
            lambda_H=lambda_H,
            lambda_mode=lambda_mode,
            tol=tol,
            target_r2=target_r2,
            verbose=verbose,
            random_state=random_state,
            eps=eps,
        )

        B = selected_init["B"]
        H = selected_init["H"]
        seeds = selected_init["seeds"]
        K_init = H.shape[0]

        if verbose:
            print(
                f"[mode K selection: sweep] episode={episode.episode_id}, "
                f"Kmax={Kmax}, selected_K={selected_K}, "
                f"short_R2={selected_init['r2']:.6f}, "
                f"method={K_select_info['select_info']['method']}"
            )

    elif K_selection_method == "fixed":
        # Directly use Kmax as K.
        selected_K = int(Kmax)

        H, B, seeds = _initialize_modes_spatial_seed(
            Y_data,
            valid_coords,
            Kmax=selected_K,
            rng=rng,
            eps=eps,
        )

        K_init = H.shape[0]

        K_select_info = {
            "method": "fixed",
            "selected_K": int(selected_K),
        }

        if verbose:
            print(
                f"[mode K selection: fixed] episode={episode.episode_id}, "
                f"K={selected_K}"
            )

    else:
        raise ValueError(
            f"Unknown K_selection_method: {K_selection_method}. "
            "Use 'svd', 'sweep', or 'fixed'."
        )


    # ------------------------------------------------------------
    # Full fit from selected K initialization
    # ------------------------------------------------------------
    B, H, loss_history = _fit_motion_modes_minimal(
        M,
        B,
        H,
        lambda_B=lambda_B,
        lambda_H=lambda_H,
        lambda_mode=lambda_mode,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        eps=eps,
        scaled_B_penalty=scaled_B_penalty,
        B_scale=B_scale,
    )

    K_after_fit = H.shape[0]

    recon_after_fit = _compute_recon_loss(M, B, H, eps=eps)
    r2_after_fit = _compute_recon_r2(M, B, H, eps=eps)

    if verbose:
        print(
            f"[mode initial fit] episode={episode.episode_id}, "
            f"K={K_after_fit}, recon={recon_after_fit:.6f}, "
            f"R2={r2_after_fit:.6f}"
        )

    # ------------------------------------------------------------
    # 3. Reconstruction-preserving merge
    # ------------------------------------------------------------
    merge_groups = None
    merge_diag = None
    proposed_merge_groups = None

    if merge_redundant_modes and H.shape[0] > 1:
        # 3.1 propose merge groups by activation similarity
        proposed_merge_groups, sim_H = _activation_merge_groups(
            H,
            activation_merge_thresh=activation_merge_thresh,
            eps=eps,
        )

        if verbose:
            group_sizes = [len(g) for g in proposed_merge_groups]
            print(
                f"[mode merge proposal] episode={episode.episode_id}, "
                f"K fit={K_after_fit}, n_groups={len(proposed_merge_groups)}, "
                f"group_sizes={group_sizes}"
            )

        # 3.2 perform reconstruction-preserving merge
        B_merge, H_merge, merge_groups, merge_diag = _merge_modes_reconstruction_preserving(
            M=M,
            B=B,
            H=H,
            groups=proposed_merge_groups,
            ridge=merge_ridge,
            max_r2_drop=max_merge_r2_drop,
            max_group_rank1_error=max_group_rank1_error,
            reject_bad_groups=reject_bad_merge_groups,
            verbose=verbose,
        )

        B, H = B_merge, H_merge

        if verbose:
            print(
                f"[mode merge] episode={episode.episode_id}, "
                f"K fit={K_after_fit}, K merge={H.shape[0]}, "
                f"accepted={merge_diag['accepted']}, "
                f"recon {merge_diag['recon_before']:.6f}"
                f" -> {merge_diag['recon_after_init']:.6f}, "
                f"R2 {merge_diag['r2_before']:.6f}"
                f" -> {merge_diag['r2_after_init']:.6f}, "
                f"R2_drop={merge_diag['r2_drop']:.6f}"
            )

        # 3.3 refine from merged B,H, not random initialization
        if refine_after_merge and H.shape[0] > 0:
            B, H, loss_ref = _fit_motion_modes_minimal(
                M,
                B,
                H,
                lambda_B=lambda_B,
                lambda_H=lambda_H,
                lambda_mode=lambda_mode,
                max_iter=max(10, max_iter // 2),
                tol=tol,
                verbose=verbose,
                eps=eps,
                scaled_B_penalty=scaled_B_penalty,
                B_scale=B_scale,
            )
            loss_history.extend(loss_ref)

            if verbose:
                recon_after_merge_refine = _compute_recon_loss(M, B, H, eps=eps)
                r2_after_merge_refine = _compute_recon_r2(M, B, H, eps=eps)
                print(
                    f"[mode merge refine] episode={episode.episode_id}, "
                    f"K={H.shape[0]}, "
                    f"recon={recon_after_merge_refine:.6f}, "
                    f"R2={r2_after_merge_refine:.6f}"
                )

    K_after_merge = H.shape[0]

    # ------------------------------------------------------------
    # 4. Prune weak / redundant modes
    # ------------------------------------------------------------
    if H.shape[0] > 0:
        B, H, prune_info = _prune_BH_modes(
            B,
            H,
            M,
            support_rel_thresh=support_rel_thresh,
            min_mode_mass=min_mode_mass,
            min_incremental_energy=min_incremental_energy,
            min_support_area=min_support_area,
            max_mode_density=max_mode_density,
            eps=eps,
        )
    else:
        prune_info = {}

    if verbose:
        print(
            f"[mode prune] episode={episode.episode_id}, "
            f"K merge={K_after_merge}, K prune={H.shape[0]}"
        )

    # ------------------------------------------------------------
    # 5. Final refine after prune
    # ------------------------------------------------------------
    if final_refine_after_prune and H.shape[0] > 0:
        B, H, loss_final = _fit_motion_modes_minimal(
            M,
            B,
            H,
            lambda_B=lambda_B,
            lambda_H=lambda_H,
            lambda_mode=lambda_mode,
            max_iter=max(10, max_iter // 3),
            tol=tol,
            verbose=verbose,
            eps=eps,
            scaled_B_penalty=scaled_B_penalty,
            B_scale=B_scale,
        )
        loss_history.extend(loss_final)

        if verbose:
            recon_after_prune_refine = _compute_recon_loss(M, B, H, eps=eps)
            r2_after_prune_refine = _compute_recon_r2(M, B, H, eps=eps)
            print(
                f"[mode final refine] episode={episode.episode_id}, "
                f"K={H.shape[0]}, "
                f"recon={recon_after_prune_refine:.6f}, "
                f"R2={r2_after_prune_refine:.6f}"
            )

    # ------------------------------------------------------------
    # 6. Build MotionMode objects
    # ------------------------------------------------------------
    if H.shape[0] > 0:
        modes = _build_motion_modes_from_BH(
            B,
            H,
            M,
            Y_data,
            valid_coords,
            (X, Y),
            episode.episode_id,
            episode.time_range,
            global_motion,
            min_mode_mass=min_mode_mass,
            min_explained_energy=min_explained_energy,
            support_rel_thresh=support_rel_thresh,
            eps=eps,
        )
    else:
        modes = []

    # ------------------------------------------------------------
    # 7. Save model information
    # ------------------------------------------------------------
    final_recon = _compute_recon_loss(M, B, H, eps=eps) if H.shape[0] > 0 else None
    final_r2 = _compute_recon_r2(M, B, H, eps=eps) if H.shape[0] > 0 else None

    episode.modes = modes
    episode.mode_model = {
        "B": B,
        "H": H,
        "loss_history": loss_history,

        "lambda_B": lambda_B,
        "lambda_H": lambda_H,
        "lambda_mode": lambda_mode,

        "Kmax": Kmax,
        "K_init": K_init,
        "K_after_fit": K_after_fit,
        "K_after_merge": K_after_merge,
        "K_after_prune": H.shape[0],
        "K_modes": len(modes),

        "seeds": seeds,

        "proposed_merge_groups": proposed_merge_groups,
        "merge_groups": merge_groups,
        "merge_diag": merge_diag,

        "prune_info": prune_info,

        "total_energy": total_energy,
        "recon_after_fit": recon_after_fit,
        "r2_after_fit": r2_after_fit,
        "final_recon": final_recon,
        "final_r2": final_r2,

        "K_selection_method": K_selection_method,
        "K_selected": selected_K,
        "K_select_info": K_select_info,
        "svd_target_r2": svd_target_r2,
        "target_r2": target_r2,

        "B_scale": B_scale,
        "scaled_B_penalty": scaled_B_penalty,

        "merge_params": {
            "activation_merge_thresh": activation_merge_thresh,
            "merge_ridge": merge_ridge,
            "max_merge_r2_drop": max_merge_r2_drop,
            "max_group_rank1_error": max_group_rank1_error,
            "reject_bad_merge_groups": reject_bad_merge_groups,
        },
    }

    if verbose:
        print(
            f"[motion modes] episode={episode.episode_id}, "
            f"Kmax={Kmax}, kept modes={len(modes)}, "
            f"final_R2={final_r2}"
        )

    return modes

def getMotionModes(motion_episodes: Sequence[MotionEpisode], **kwargs):
    all_modes = []
    for ep in motion_episodes:
        modes = decompose_episode_motion_modes(ep, **kwargs)
        all_modes.extend(modes)
    return all_modes


# =============================================================================
# Mode -> MotionRegion split
# =============================================================================


def _weighted_mean_response_vector(B, A, mask, eps=1e-8):
    if mask is None or not np.any(mask):
        return np.zeros(2, dtype=np.float32)

    w = A[mask].astype(np.float32)
    b = B[mask].astype(np.float32)

    if np.sum(w) <= eps:
        return b.mean(axis=0).astype(np.float32)

    return (np.sum(w[:, None] * b, axis=0) / (np.sum(w) + eps)).astype(np.float32)


def _region_min_distance(mask_a, mask_b):
    """
    Minimum pixel/patch distance from mask_a to mask_b.
    """
    if not np.any(mask_a) or not np.any(mask_b):
        return np.inf

    dist_to_b = ndi.distance_transform_edt(~mask_b)
    return float(np.min(dist_to_b[mask_a]))


def _cosine_similarity(v1, v2, eps=1e-8):
    v1 = np.asarray(v1, dtype=np.float32).reshape(-1)
    v2 = np.asarray(v2, dtype=np.float32).reshape(-1)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < eps or n2 < eps:
        return 0.0

    return float(np.dot(v1, v2) / (n1 * n2 + eps))


def _merge_or_discard_small_region_masks(
    region_masks,
    A,
    B,
    min_region_area=10,
    discard_region_area=3,
    merge_small_max_dist=4.0,
    merge_small_vector_cos=0.0,
):
    """
    Merge small regions into nearby larger regions, or discard isolated tiny ones.

    region_masks: list of bool masks
    A: response_strength, (X, Y)
    B: response_field, (X, Y, 2)
    """
    if len(region_masks) == 0:
        return []

    region_masks = [m.astype(bool).copy() for m in region_masks]
    areas = np.array([int(m.sum()) for m in region_masks], dtype=np.int32)

    # If everything is small, keep only the largest one if it is not too tiny.
    large_ids = np.where(areas >= min_region_area)[0].tolist()
    small_ids = np.where(areas < min_region_area)[0].tolist()

    if len(large_ids) == 0:
        best = int(np.argmax(areas))
        if areas[best] >= discard_region_area:
            return [region_masks[best]]
        return []

    kept = [region_masks[i].copy() for i in large_ids]

    for sid in small_ids:
        small_mask = region_masks[sid]
        small_area = int(small_mask.sum())

        if small_area < discard_region_area:
            continue

        small_vec = _weighted_mean_response_vector(B, A, small_mask)

        best_j = None
        best_score = np.inf

        for j, big_mask in enumerate(kept):
            d = _region_min_distance(small_mask, big_mask)
            if d > merge_small_max_dist:
                continue

            big_vec = _weighted_mean_response_vector(B, A, big_mask)
            cos = _cosine_similarity(small_vec, big_vec)

            if cos < merge_small_vector_cos:
                continue

            # Prefer nearest, then stronger directional consistency.
            score = d - 0.25 * cos
            if score < best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            kept[best_j] = np.logical_or(kept[best_j], small_mask)
        # else discard it

    return kept


def split_mode_to_regions(
    mode,
    support_rel_thresh=0.10,

    # main spatial tolerance
    split_mode="gap_tolerant_discard_small",
    gap_dilation_iter=4,
    gap_close_iter=4,

    # small fragment handling
    min_region_area=20,
    discard_region_area=3,
    merge_small_regions=False,
    merge_small_max_dist=4.0,
    merge_small_vector_cos=0.0,

    # output
    region_id_start=0,
):
    """
    Split one MotionMode into spatial MotionRegions.

    Recommended mode:
        split_mode="gap_tolerant_discard_small"

    Logic:
        1. response_strength -> raw_support
        2. use closing/dilation only to connect nearby fragments
        3. connected components are computed on the tolerant grouping mask
        4. final region masks only contain raw_support pixels
        5. small regions are directly discarded
    """
    A = np.asarray(mode.response_strength, dtype=np.float32)
    B = np.asarray(mode.response_field, dtype=np.float32)
    h = np.asarray(mode.activation, dtype=np.float32)

    if A.ndim != 2:
        raise ValueError(f"mode.response_strength should be 2D, got {A.shape}")
    if B.ndim != 3 or B.shape[-1] != 2:
        raise ValueError(f"mode.response_field should be (X,Y,2), got {B.shape}")

    vmax = float(np.max(A))
    if vmax <= 0:
        return []

    raw_support = A > (support_rel_thresh * vmax)

    if not np.any(raw_support):
        return []

    structure = np.ones((3, 3), dtype=bool)

    # ------------------------------------------------------------
    # 1. Build region masks
    # ------------------------------------------------------------
    if split_mode in ["whole", "mode_support", "no_split", "loose"]:
        # One mode produces at most one region.
        # If it is too small, discard it.
        if int(raw_support.sum()) >= min_region_area:
            region_masks = [raw_support]
        else:
            region_masks = []

    elif split_mode == "strict":
        # Strict connected components on raw support.
        label_map, num = ndi.label(raw_support, structure=structure)

        region_masks = []
        for rid in range(1, num + 1):
            m = label_map == rid
            if int(m.sum()) >= min_region_area:
                region_masks.append(m)

    elif split_mode in ["gap_tolerant", "gap_tolerant_discard_small"]:
        # Use closing/dilation only for grouping.
        # Final masks still only contain raw_support pixels.
        group_mask = raw_support.copy()

        # closing fills small holes / bridges tiny gaps
        if gap_close_iter is not None and gap_close_iter > 0:
            group_mask = ndi.binary_closing(
                group_mask,
                structure=structure,
                iterations=int(gap_close_iter),
            )

        # dilation makes the connectivity more tolerant
        if gap_dilation_iter is not None and gap_dilation_iter > 0:
            group_mask = ndi.binary_dilation(
                group_mask,
                structure=structure,
                iterations=int(gap_dilation_iter),
            )

        label_map, num = ndi.label(group_mask, structure=structure)

        region_masks = []
        for rid in range(1, num + 1):
            # Keep only original support pixels.
            # Do not include dilated fake pixels.
            m = raw_support & (label_map == rid)

            area = int(m.sum())
            if area >= min_region_area:
                region_masks.append(m)

        # For this mode, small regions are already discarded above.
        # No additional small-fragment merging is needed.
        if split_mode == "gap_tolerant_discard_small":
            merge_small_regions = False

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    # ------------------------------------------------------------
    # 2. Optional old behavior: merge small regions
    # ------------------------------------------------------------
    # Current recommendation: keep merge_small_regions=False.
    # If you explicitly set merge_small_regions=True, this restores old behavior.
    if merge_small_regions and len(region_masks) > 0:
        region_masks = _merge_or_discard_small_region_masks(
            region_masks=region_masks,
            A=A,
            B=B,
            min_region_area=min_region_area,
            discard_region_area=discard_region_area,
            merge_small_max_dist=merge_small_max_dist,
            merge_small_vector_cos=merge_small_vector_cos,
        )
    else:
        # Defensive filtering.
        region_masks = [
            m for m in region_masks
            if int(m.sum()) >= min_region_area
        ]

    if len(region_masks) == 0:
        return []

    # ------------------------------------------------------------
    # 3. Build MotionRegion objects
    # ------------------------------------------------------------
    regions = []

    for local_id, region_mask in enumerate(region_masks):
        area = int(region_mask.sum())
        if area < min_region_area:
            continue

        response_strength = np.where(region_mask, A, 0.0).astype(np.float32)
        response_field = np.where(region_mask[:, :, None], B, 0.0).astype(np.float32)

        mean_b = _weighted_mean_response_vector(B, A, region_mask)

        # Representative induced motion of this region:
        # u_r(t) = h(t) * mean_b
        induced_motion = h[:, None] * mean_b[None, :]

        center_xy, spatial_cov, area_effective, strength = _compute_spatial_stats(
            response_strength
        )

        region = MotionRegion(
            episode_id=mode.episode_id,
            mode_id=mode.mode_id,
            region_id=region_id_start + local_id,
            time_range=mode.time_range,

            activation=h.copy(),
            response_field=response_field,
            response_strength=response_strength,
            region_mask=region_mask.astype(np.uint8),

            induced_motion=induced_motion.astype(np.float32),
            mean_response_vector=mean_b.astype(np.float32),

            center_xy=center_xy,
            spatial_cov=spatial_cov,
            area_effective=float(area_effective),
            strength=float(strength),
            metadata={
                "support_rel_thresh": support_rel_thresh,
                "split_mode": split_mode,
                "gap_dilation_iter": gap_dilation_iter,
                "gap_close_iter": gap_close_iter,
                "min_region_area": min_region_area,
                "discard_region_area": discard_region_area,
                "merge_small_regions": merge_small_regions,
                "merge_small_max_dist": merge_small_max_dist,
                "merge_small_vector_cos": merge_small_vector_cos,
                "raw_area": area,
                "small_region_strategy": (
                    "discard" if not merge_small_regions else "merge_or_discard"
                ),
                "final_region_mask": "raw_support_only",
            },
        )

        region.duration = int(len(h))
        if getattr(mode, "activation_resampled", None) is not None:
            region.activation_resampled = np.asarray(
                mode.activation_resampled,
                dtype=np.float32,
            )
        else:
            try:
                region.activation_resampled = _resample_1d(h, target_len=12)
            except NameError:
                region.activation_resampled = h.copy().astype(np.float32)

        region.activation_feature = region.activation_resampled.copy()

        regions.append(region)

    return regions


def split_episode_modes_to_regions(
    episode: MotionEpisode,
    support_rel_thresh=0.05,
    split_mode="gap_tolerant_discard_small",
    gap_dilation_iter=4,
    gap_close_iter=4,
    min_region_area=20,
    discard_region_area=4,
    merge_small_regions=False,
    merge_small_max_dist=4.0,
    merge_small_vector_cos=0.0,
    verbose=False,
):
    """
    Split all MotionModes in one episode into MotionRegions.

    Current recommended logic:
        - still split according to spatial connectivity;
        - use gap-tolerant connectivity to avoid over-fragmentation;
        - directly discard small regions;
        - do not merge small fragments into nearby regions.
    """
    regions_all = []
    region_id = 0

    modes = getattr(episode, "modes", None)
    if modes is None:
        modes = []

    for mode in modes:
        # Important: clear old regions to avoid duplicate append when rerunning.
        mode.regions = []

        regs = split_mode_to_regions(
            mode,
            support_rel_thresh=support_rel_thresh,
            split_mode=split_mode,
            gap_dilation_iter=gap_dilation_iter,
            gap_close_iter=gap_close_iter,
            min_region_area=min_region_area,
            discard_region_area=discard_region_area,
            merge_small_regions=merge_small_regions,
            merge_small_max_dist=merge_small_max_dist,
            merge_small_vector_cos=merge_small_vector_cos,
            region_id_start=region_id,
        )

        for r in regs:
            r.region_id = region_id
            r.component_id = region_id
            regions_all.append(r)
            mode.regions.append(r)
            region_id += 1

    episode.regions = regions_all

    if verbose:
        print(
            f"[split_episode_modes_to_regions] "
            f"episode={getattr(episode, 'episode_id', None)}, "
            f"modes={len(modes)}, regions={len(regions_all)}"
        )

    return regions_all


def getMotionRegions(
    motion_episodes,
    support_rel_thresh=0.05,

    split_mode="gap_tolerant_discard_small",
    gap_dilation_iter=4,
    gap_close_iter=4,

    min_region_area=20,
    discard_region_area=4,

    # Current recommendation: do not merge small regions.
    # Small fragments are more likely noise / episode extraction artifacts.
    merge_small_regions=False,
    merge_small_max_dist=4.0,
    merge_small_vector_cos=0.0,

    verbose=True,
):
    """
    Convert MotionModes into MotionRegions for all episodes.

    Recommended behavior:
        1. Use tolerant connectivity to avoid excessive fragmentation.
        2. Discard small regions directly.
        3. Do not merge small fragments into nearby large regions.
        4. Reset previous episode.regions and mode.regions every time.
    """
    all_regions = []

    for ep in motion_episodes:
        ep_regions = split_episode_modes_to_regions(
            ep,
            support_rel_thresh=support_rel_thresh,
            split_mode=split_mode,
            gap_dilation_iter=gap_dilation_iter,
            gap_close_iter=gap_close_iter,
            min_region_area=min_region_area,
            discard_region_area=discard_region_area,
            merge_small_regions=merge_small_regions,
            merge_small_max_dist=merge_small_max_dist,
            merge_small_vector_cos=merge_small_vector_cos,
            verbose=False,
        )

        all_regions.extend(ep_regions)

        if verbose:
            modes = getattr(ep, "modes", []) or []
            print(
                f"[getMotionRegions] episode={getattr(ep, 'episode_id', None)}, "
                f"modes={len(modes)}, regions={len(ep_regions)}"
            )

    if verbose:
        n_eps = len(motion_episodes)
        n_modes = sum(len(getattr(ep, "modes", []) or []) for ep in motion_episodes)
        n_regions = len(all_regions)

        print(
            f"[getMotionRegions] total episodes={n_eps}, "
            f"total modes={n_modes}, total regions={n_regions}, "
            f"regions/mode={n_regions / max(n_modes, 1):.3f}"
        )

    return all_regions
# =============================================================================
# MotionRegion pattern clustering
# =============================================================================


def collect_regions_from_episodes(motion_episodes):
    out = []
    for ep in motion_episodes:
        out.extend(getattr(ep, "regions", []) or [])
    return out


def filter_regions_for_patterns(regions, min_strength=0.0, min_area=0.0, min_duration=1):
    kept = []
    for r in regions:
        if getattr(r, "strength", 0.0) < min_strength:
            continue
        if getattr(r, "area_effective", 0.0) < min_area:
            continue
        if getattr(r, "duration", 0) < min_duration:
            continue
        if getattr(r, "mean_response_vector", None) is None:
            continue
        kept.append(r)
    return kept


def _dtw_distance_1d(x, y, eps=1e-8):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return np.inf

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = abs(xi - y[j - 1])
            dp[i, j] = cost + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    return float(dp[n, m] / (n + m + eps))


def _find_activation_medoid_index(activations, weights=None, eps=1e-8):
    """
    Select the region whose activation is most representative of the pattern.
    This does not require equal-length activations.
    """
    n = len(activations)
    if n == 0:
        return None
    if n == 1:
        return 0

    if weights is None:
        weights = np.ones(n, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)

    weights = weights / (np.sum(weights) + eps)

    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        s = 0.0
        for j in range(n):
            if i == j:
                continue
            d = _dtw_distance_1d(activations[i], activations[j], eps=eps)
            s += weights[j] * d
        scores[i] = s

    return int(np.argmin(scores))


def _sign_by_resampled_corr(ref_h, h, target_len=16):
    """
    Only for sign alignment. Raw activations can still keep original length.
    """
    ref_h = np.asarray(ref_h, dtype=np.float32).reshape(-1)
    h = np.asarray(h, dtype=np.float32).reshape(-1)

    if len(ref_h) == 0 or len(h) == 0:
        return 1.0

    ref_rs = _resample_1d(ref_h, target_len)
    h_rs = _resample_1d(h, target_len)

    return 1.0 if _safe_corr(ref_rs, h_rs) >= 0 else -1.0


def _sign_aware_activation_dtw(h1, h2):
    """
    Compare h1 with h2 and -h2.
    Return smaller DTW distance and sign for aligning h2/B2.
    """
    d_pos = _dtw_distance_1d(h1, h2)
    d_neg = _dtw_distance_1d(h1, -np.asarray(h2))

    if d_neg < d_pos:
        return float(d_neg), -1.0
    else:
        return float(d_pos), 1.0

def _get_region_activation(region):
    if hasattr(region, "activation") and region.activation is not None:
        return np.asarray(region.activation, dtype=np.float32).reshape(-1)
    if hasattr(region, "activation_resampled") and region.activation_resampled is not None:
        return np.asarray(region.activation_resampled, dtype=np.float32).reshape(-1)
    return None

def _get_region_mask_simple(region):
    if hasattr(region, "region_mask") and region.region_mask is not None:
        mask = np.asarray(region.region_mask).astype(bool)
        if mask.ndim == 2 and np.any(mask):
            return mask

    if hasattr(region, "response_strength") and region.response_strength is not None:
        A = np.asarray(region.response_strength, dtype=np.float32)
        if A.ndim == 2 and np.max(A) > 0:
            return A > 0

    return None

def _get_region_response_field(region):
    if hasattr(region, "response_field") and region.response_field is not None:
        B = np.asarray(region.response_field, dtype=np.float32)
        if B.ndim == 3 and B.shape[-1] == 2:
            return B
    return None

def _mask_iou(mask1, mask2, eps=1e-8):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter / (union + eps))

def _response_field_distance_on_overlap(region1, region2, sign2=1.0, eps=1e-8):
    """
    Compare b_ik vector values on the common region only.

    D_b = ||B1 - sign*B2|| / (||B1|| + ||B2||)
    """
    mask1 = _get_region_mask_simple(region1)
    mask2 = _get_region_mask_simple(region2)
    B1 = _get_region_response_field(region1)
    B2 = _get_region_response_field(region2)

    if mask1 is None or mask2 is None or B1 is None or B2 is None:
        return np.inf

    if mask1.shape != mask2.shape or B1.shape != B2.shape:
        return np.inf

    common = np.logical_and(mask1, mask2)
    if not np.any(common):
        return np.inf

    v1 = B1[common]                    # (N, 2)
    v2 = sign2 * B2[common]            # (N, 2)

    numerator = np.sqrt(np.sum((v1 - v2) ** 2))
    denom = np.sqrt(np.sum(v1 ** 2)) + np.sqrt(np.sum(v2 ** 2)) + eps

    return float(numerator / denom)

def compute_region_distance_matrix_simple(
    regions,
    min_iou=0.10,
    omega=1.0,
    mu=1.0,
    incompatible_dist=1e6,
    verbose=True,
):
    """
    Pairwise distance for MotionRegionPattern.

    Hard gate:
        IoU(region_i, region_j) >= min_iou

    Soft distance:
        D = omega * D_h + mu * D_b

    where:
        D_h = sign-aware DTW distance between activations
        D_b = response vector field value difference on overlapping support
    """
    n = len(regions)
    D = np.zeros((n, n), dtype=np.float32)
    pair_info = {}

    for i in range(n):
        for j in range(i + 1, n):
            mask_i = _get_region_mask_simple(regions[i])
            mask_j = _get_region_mask_simple(regions[j])

            if mask_i is None or mask_j is None or mask_i.shape != mask_j.shape:
                dist = incompatible_dist
                info = {
                    "compatible": False,
                    "reason": "invalid_mask",
                    "iou": 0.0,
                    "D_h": np.inf,
                    "D_b": np.inf,
                    "distance": dist,
                    "sign": 1.0,
                }
            else:
                iou = _mask_iou(mask_i, mask_j)

                if iou < min_iou:
                    dist = incompatible_dist
                    info = {
                        "compatible": False,
                        "reason": "low_iou",
                        "iou": float(iou),
                        "D_h": np.inf,
                        "D_b": np.inf,
                        "distance": dist,
                        "sign": 1.0,
                    }
                else:
                    h_i = _get_region_activation(regions[i])
                    h_j = _get_region_activation(regions[j])

                    if h_i is None or h_j is None:
                        dist = incompatible_dist
                        info = {
                            "compatible": False,
                            "reason": "invalid_activation",
                            "iou": float(iou),
                            "D_h": np.inf,
                            "D_b": np.inf,
                            "distance": dist,
                            "sign": 1.0,
                        }
                    else:
                        D_h, sign_j = _sign_aware_activation_dtw(h_i, h_j)
                        D_b = _response_field_distance_on_overlap(
                            regions[i],
                            regions[j],
                            sign2=sign_j,
                        )

                        if not np.isfinite(D_h) or not np.isfinite(D_b):
                            dist = incompatible_dist
                            compatible = False
                            reason = "invalid_distance"
                        else:
                            dist = omega * D_h + mu * D_b
                            compatible = True
                            reason = "ok"

                        info = {
                            "compatible": bool(compatible),
                            "reason": reason,
                            "iou": float(iou),
                            "D_h": float(D_h),
                            "D_b": float(D_b),
                            "distance": float(dist),
                            "sign": float(sign_j),
                        }

            D[i, j] = dist
            D[j, i] = dist
            pair_info[(i, j)] = info

    if verbose:
        finite = D[(D > 0) & (D < incompatible_dist)]
        if finite.size > 0:
            print(
                f"[compute_region_distance_matrix_simple] n={n}, "
                f"finite_pairs={finite.size // 2}, "
                f"dist min/median/max="
                f"{finite.min():.4f}/{np.median(finite):.4f}/{finite.max():.4f}"
            )
        else:
            print(f"[compute_region_distance_matrix_simple] n={n}, no compatible pairs.")

    return D, pair_info

def cluster_regions_hierarchical(
    dist_mat,
    cluster_dist_thresh=0.8,
    linkage_method="complete",
    incompatible_dist=1e6,
    verbose=True,
):
    """
    Hierarchical clustering from precomputed distance matrix.
    Complete linkage is recommended to avoid graph-chain effect.
    """
    dist_mat = np.asarray(dist_mat, dtype=np.float32)
    n = dist_mat.shape[0]

    if n == 0:
        return [], np.array([], dtype=np.int32)

    if n == 1:
        return [[0]], np.array([0], dtype=np.int32)

    D = dist_mat.copy()
    D[~np.isfinite(D)] = incompatible_dist
    np.fill_diagonal(D, 0.0)

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=linkage_method)

    raw_labels = fcluster(
        Z,
        t=cluster_dist_thresh,
        criterion="distance",
    )

    unique_labels = sorted(np.unique(raw_labels))
    label_map = {lab: idx for idx, lab in enumerate(unique_labels)}
    labels = np.array([label_map[x] for x in raw_labels], dtype=np.int32)

    groups = []
    for gid in range(len(unique_labels)):
        groups.append(np.where(labels == gid)[0].tolist())

    if verbose:
        print(
            f"[cluster_regions_hierarchical] "
            f"linkage={linkage_method}, "
            f"cluster_dist_thresh={cluster_dist_thresh}, "
            f"n_groups={len(groups)}"
        )
        print("[cluster_regions_hierarchical] group sizes:", [len(g) for g in groups])

    return groups, labels


def build_motion_patterns_from_groups(regions, groups):
    """
    Build MotionPattern objects from clustered MotionRegion groups.

    Parameters
    ----------
    regions : list[MotionRegion]
        The filtered MotionRegion list.
    groups : list[list[int]]
        Each group contains indices into `regions`.

    Returns
    -------
    patterns : list[MotionPattern]
    """
    patterns = []

    for pid, group in enumerate(groups):
        group_regions = [regions[i] for i in group]
        pattern = MotionPattern(
            pattern_id=pid,
            regions=group_regions,
        )
        patterns.append(pattern)

    return patterns

def getMotionRegionPattern(
    motion_episodes,
    min_strength=0.0,
    min_area=5,
    min_duration=1,

    min_iou=0.10,
    omega=1.0,
    mu=1.0,

    cluster_dist_thresh=0.8,
    linkage_method="complete",
    incompatible_dist=1e6,

    verbose=True,
):
    """
    Build MotionPatterns from MotionRegions.

    Logic:
        1. Collect MotionRegions from all episodes.
        2. Filter weak/small/short regions.
        3. Pairwise hard gate by region IoU.
        4. Pairwise distance:
               D = omega * DTW(h_i, h_j)
                 + mu    * response_field_distance(b_i, b_j)
        5. Complete-linkage hierarchical clustering.
    """
    all_regions = collect_regions_from_episodes(motion_episodes)

    if verbose:
        print(f"[getMotionRegionPattern] collected regions: {len(all_regions)}")

    kept_regions = filter_regions_for_patterns(
        all_regions,
        min_strength=min_strength,
        min_area=min_area,
        min_duration=min_duration,
    )

    if verbose:
        print(f"[getMotionRegionPattern] kept regions: {len(kept_regions)}")

    if len(kept_regions) == 0:
        return [], [], [], np.array([], dtype=np.int32), {}

    dist_mat, pair_info = compute_region_distance_matrix_simple(
        kept_regions,
        min_iou=min_iou,
        omega=omega,
        mu=mu,
        incompatible_dist=incompatible_dist,
        verbose=verbose,
    )

    groups, labels = cluster_regions_hierarchical(
        dist_mat,
        cluster_dist_thresh=cluster_dist_thresh,
        linkage_method=linkage_method,
        incompatible_dist=incompatible_dist,
        verbose=verbose,
    )

    patterns = build_motion_patterns_from_groups(
        kept_regions,
        groups,
    )

    info = {
        "distance_matrix": dist_mat,
        "pair_info": pair_info,
        "labels": labels,
        "params": {
            "min_strength": min_strength,
            "min_area": min_area,
            "min_duration": min_duration,
            "min_iou": min_iou,
            "omega": omega,
            "mu": mu,
            "cluster_dist_thresh": cluster_dist_thresh,
            "linkage_method": linkage_method,
            "incompatible_dist": incompatible_dist,
        },
    }

    return patterns, kept_regions, groups, labels, info

def pattern_to_binary_mask(
    pattern,
    support_rel_thresh=0.20,
    use_region_union=True,
    keep_largest_cc=False,
):
    """
    Convert one MotionPattern to a binary spatial mask.

    Priority:
        1. union of member region masks;
        2. prototype_region_map / prototype_region fallback.
    """
    masks = []

    if use_region_union:
        for r in getattr(pattern, "regions", []) or []:
            m = getattr(r, "region_mask", None)
            if m is not None:
                masks.append(np.asarray(m).astype(bool))

    if len(masks) > 0:
        mask = np.any(np.stack(masks, axis=0), axis=0)
    else:
        region_map = getattr(pattern, "prototype_region_map", None)
        if region_map is None:
            region_map = getattr(pattern, "prototype_region", None)

        if region_map is None:
            return None

        region_map = np.asarray(region_map, dtype=np.float32)
        vmax = float(np.nanmax(region_map))

        if vmax <= 0:
            return None

        mask = region_map > support_rel_thresh * vmax

    if keep_largest_cc:
        labeled, num = ndi.label(mask)
        if num > 0:
            areas = np.array([(labeled == rid).sum() for rid in range(1, num + 1)])
            rid = int(np.argmax(areas)) + 1
            mask = labeled == rid

    return mask.astype(bool)

def find_patterns_overlapping_region(
    patterns,
    query_mask,
    min_iou=0.05,
    min_query_covered=0.10,
    min_pattern_covered=0.01,
    support_rel_thresh=0.20,
    use_region_union=True,
    keep_largest_cc=False,
    sort_by="query_covered",
):
    """
    Find all motion patterns overlapping with a user-specified region.

    Parameters
    ----------
    patterns:
        list of MotionPattern.

    query_mask:
        boolean array, same spatial shape as pattern masks.

    min_iou:
        minimum IoU between pattern mask and query mask.

    min_query_covered:
        require at least this fraction of the query region covered by pattern.

    min_pattern_covered:
        require at least this fraction of the pattern covered by query.

    Returns
    -------
    rows:
        list of dicts, each containing pattern_id and overlap statistics.
    """
    query_mask = np.asarray(query_mask).astype(bool)

    if query_mask.sum() == 0:
        raise ValueError("query_mask is empty.")

    rows = []

    for p in patterns:
        pid = getattr(p, "pattern_id", None)

        pmask = pattern_to_binary_mask(
            p,
            support_rel_thresh=support_rel_thresh,
            use_region_union=use_region_union,
            keep_largest_cc=keep_largest_cc,
        )

        if pmask is None:
            continue

        if pmask.shape != query_mask.shape:
            raise ValueError(
                f"Pattern {pid} mask shape {pmask.shape} != query_mask shape {query_mask.shape}"
            )

        inter = np.logical_and(pmask, query_mask).sum()
        union = np.logical_or(pmask, query_mask).sum()

        pattern_area = int(pmask.sum())
        query_area = int(query_mask.sum())

        if union == 0 or pattern_area == 0 or query_area == 0:
            continue

        iou = inter / union
        query_covered = inter / query_area
        pattern_covered = inter / pattern_area

        if (
            iou >= min_iou
            or query_covered >= min_query_covered
            or pattern_covered >= min_pattern_covered
        ):
            rows.append({
                "pattern_id": pid,
                "n_regions": len(getattr(p, "regions", []) or []),
                "pattern_area": pattern_area,
                "query_area": query_area,
                "intersection": int(inter),
                "iou": float(iou),
                "query_covered": float(query_covered),
                "pattern_covered": float(pattern_covered),
                "pattern": p,
                "pattern_mask": pmask,
            })

    if sort_by is not None:
        rows = sorted(rows, key=lambda x: x[sort_by], reverse=True)

    return rows

# =============================================================================
# Visualization
# =============================================================================


def _auto_contrast(img):
    img = np.asarray(img)
    if img.size == 0:
        return img
    p1, p99 = np.percentile(img, [1, 99])
    if p99 <= p1:
        return img
    return np.clip((img - p1) / (p99 - p1), 0, 1)


def _get_BH_from_episode(ep):
    if not hasattr(ep, "mode_model") or ep.mode_model is None:
        raise ValueError("episode.mode_model is missing.")
    B = np.asarray(ep.mode_model["B"], dtype=np.float32)  # (2N, K)
    H = np.asarray(ep.mode_model["H"], dtype=np.float32)  # (K, T)
    return B, H


def _episode_motion_matrix(ep, use_global_subtracted=True):
    motion_abs = np.asarray(ep.motion_abs, dtype=np.float32)  # (T,N,2)

    if use_global_subtracted:
        global_motion = np.asarray(ep.global_motion, dtype=np.float32)  # (T,2)
        Y = motion_abs - global_motion[:, None, :]
    else:
        Y = motion_abs

    M = np.concatenate(
        [Y[:, :, 0].T, Y[:, :, 1].T],
        axis=0,
    ).astype(np.float32)  # (2N,T)

    return M, Y


def _B_column_to_maps(ep, b_col):
    """
    b_col: (2N,)
    return:
        bx_map, by_map, mag_map, mask
    """
    mask = np.asarray(ep.spatial_region).astype(bool)
    coords = np.argwhere(mask)
    N = len(coords)

    bx = b_col[:N]
    by = b_col[N:]

    bx_map = np.zeros(mask.shape, dtype=np.float32)
    by_map = np.zeros(mask.shape, dtype=np.float32)

    bx_map[coords[:, 0], coords[:, 1]] = bx
    by_map[coords[:, 0], coords[:, 1]] = by

    mag_map = np.sqrt(bx_map ** 2 + by_map ** 2)

    return bx_map, by_map, mag_map, mask


def _frame_vector_to_maps(ep, frame_vec):
    """
    frame_vec: (2N,)
    """
    return _B_column_to_maps(ep, frame_vec)


def _safe_norm(x, eps=1e-12):
    return float(np.sqrt(np.sum(np.asarray(x) ** 2)) + eps)


def diagnose_temporal_basis_like_sources(ep, eps=1e-12):
    B, H = _get_BH_from_episode(ep)

    K, T = H.shape

    rows = []

    for k in range(K):
        h = H[k].astype(np.float32)
        abs_h = np.abs(h)

        l1 = float(np.sum(abs_h)) + eps
        l2_sq = float(np.sum(h ** 2)) + eps

        peak_frame = int(np.argmax(abs_h))
        peak_value = float(abs_h[peak_frame])

        # 越接近 1，越像单帧 one-hot；越接近 1/T，越分散
        peak_fraction = peak_value / l1

        # participation ratio:
        # one-hot -> 1
        # uniform over T frames -> T
        temporal_pr = (l1 ** 2) / l2_sq

        # entropy normalized to [0,1]
        p = abs_h / l1
        entropy = -float(np.sum(p * np.log(p + eps))) / np.log(T + eps)

        b_norm = _safe_norm(B[:, k], eps=eps)

        rows.append({
            "mode": k,
            "T": T,
            "peak_frame": peak_frame,
            "peak_fraction": peak_fraction,
            "temporal_participation_ratio": temporal_pr,
            "temporal_entropy": entropy,
            "B_norm": b_norm,
        })

    return rows


def visualize_episode_sources_overview(
    ep,
    max_modes=20,
    sort_by="B_norm",   # "B_norm", "peak_frame", None
    quiver_step=3,
    figsize=(16, 10),
):
    B, H = _get_BH_from_episode(ep)
    K, T = H.shape

    diag = diagnose_temporal_basis_like_sources(ep)

    if sort_by == "B_norm":
        order = sorted(range(K), key=lambda k: diag[k]["B_norm"], reverse=True)
    elif sort_by == "peak_frame":
        order = sorted(range(K), key=lambda k: diag[k]["peak_frame"])
    else:
        order = list(range(K))

    order = order[:min(max_modes, K)]

    n = len(order)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    # --------- Activation heatmap ---------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    H_show = H[order]
    im = ax.imshow(H_show, aspect="auto", interpolation="nearest")
    ax.set_title("Source activations H[k,t]")
    ax.set_xlabel("relative frame t")
    ax.set_ylabel("source index after sorting")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(k) for k in order])
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    for kk in order:
        h = H[kk]
        ax.plot(np.arange(T), h, marker="o", linewidth=1, label=f"k={kk}")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Activation curves")
    ax.set_xlabel("relative frame t")
    ax.set_ylabel("h_k(t)")
    ax.grid(True, alpha=0.3)
    if len(order) <= 10:
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.show()

    # --------- Spatial response maps ---------
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)

    for ax_i, k in enumerate(order):
        ax = axes[ax_i]
        bx_map, by_map, mag_map, mask = _B_column_to_maps(ep, B[:, k])

        ax.imshow(mask, cmap="gray", alpha=0.2)
        im = ax.imshow(np.ma.masked_where(mag_map <= 0, mag_map), cmap="magma", alpha=0.9)

        yy, xx = np.where(mag_map > 0)
        if len(yy) > 0:
            keep = (yy % quiver_step == 0) & (xx % quiver_step == 0)
            yy2 = yy[keep]
            xx2 = xx[keep]
            ax.quiver(
                xx2,
                yy2,
                bx_map[yy2, xx2],
                by_map[yy2, xx2],
                angles="xy",
                scale_units="xy",
            )

        d = diag[k]
        title = (
            f"k={k}, peak={d['peak_frame']}, "
            f"PR={d['temporal_participation_ratio']:.2f}, "
            f"pf={d['peak_fraction']:.2f}"
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    for j in range(len(order), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Episode {getattr(ep, 'episode_id', None)} source spatial responses",
        fontsize=14,
    )
    plt.tight_layout()
    plt.show()

    return diag


def visualize_frame_source_contributions(ep, figsize=(10, 5)):
    B, H = _get_BH_from_episode(ep)
    K, T = H.shape

    B_norm_sq = np.sum(B ** 2, axis=0)  # (K,)
    E = (H ** 2) * B_norm_sq[:, None]   # (K,T)

    E_sum_t = np.sum(E, axis=0, keepdims=True) + 1e-12
    frac = E / E_sum_t  # 每一帧由每个 source 解释的比例

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    im = ax.imshow(frac, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax.set_title("Frame-wise source contribution fraction")
    ax.set_xlabel("relative frame t")
    ax.set_ylabel("source k")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    dominant_k = np.argmax(frac, axis=0)
    dominance = np.max(frac, axis=0)
    ax.plot(np.arange(T), dominant_k, marker="o", label="dominant source")
    ax2 = ax.twinx()
    ax2.plot(np.arange(T), dominance, marker="s", linestyle="--", label="dominance")
    ax.set_xlabel("relative frame t")
    ax.set_ylabel("dominant source k")
    ax2.set_ylabel("dominance fraction")
    ax.set_title("Dominant source per frame")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("dominant source per frame:", dominant_k)
    print("dominance fraction:", dominance)

    return frac, dominant_k, dominance


def visualize_episode_modes(
    episode: MotionEpisode,
    ref_img=None,
    patch_size=7,
    max_modes=8,
    rel_thresh=0.10,
    arrow_step=2,
    arrow_scale=None,
    vector_order="xy",
    flip_y_vector=False,
    figsize_per_row=3.2,
    save_path=None,
    show=True,
):
    modes = getattr(episode, "modes", [])
    if len(modes) == 0:
        print(f"Episode {episode.episode_id}: no modes.")
        return None
    modes = modes[:max_modes]
    n_rows = 1 + len(modes)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.0, n_rows * figsize_per_row), squeeze=False)
    mask = np.asarray(episode.spatial_region).astype(bool)
    ax = axes[0, 0]
    ax.imshow(mask, cmap="gray")
    ax.set_title(f"Episode {episode.episode_id}\ntime={episode.time_range}, modes={len(getattr(episode, 'modes', []))}")
    ax.axis("off")
    # observed/recon/resid maps
    obs_map, rec_map, res_map = compute_episode_mode_reconstruction_maps(episode)
    for col, (title, M) in enumerate([("Observed RMS", obs_map), ("Reconstructed RMS", rec_map), ("Residual RMS", res_map)], start=1):
        ax = axes[0, col]
        if M is not None:
            im = ax.imshow(M, cmap="magma")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(mask, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    for row, mode in enumerate(modes, start=1):
        A = np.asarray(mode.response_strength, dtype=np.float32)
        B = np.asarray(mode.response_field, dtype=np.float32)
        h = np.asarray(mode.activation, dtype=np.float32)
        vmax = float(np.max(A)) if A.size > 0 else 0.0
        support = A > rel_thresh * vmax if vmax > 0 else np.zeros_like(A, dtype=bool)
        ax = axes[row, 0]
        im = ax.imshow(A, cmap="magma")
        if np.any(support):
            ax.contour(support.astype(float), levels=[0.5], linewidths=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Mode {mode.mode_id}\nE={mode.explained_energy:.3f}, mass={mode.mode_mass:.2f}")
        ax.axis("off")

        ax = axes[row, 1]
        ax.imshow(mask, cmap="gray", alpha=0.35)
        ax.imshow(support.astype(float), cmap="viridis", alpha=0.75)
        ax.set_title(f"support area={int(support.sum())}")
        ax.axis("off")

        ax = axes[row, 2]
        ax.imshow(A, cmap="gray", alpha=0.45)
        rr, cc = np.where(support)
        if len(rr) > 0:
            keep = np.arange(len(rr))
            if arrow_step and arrow_step > 1:
                keep = keep[((rr[keep] % arrow_step) == 0) & ((cc[keep] % arrow_step) == 0)]
            rr2, cc2 = rr[keep], cc[keep]
            if vector_order == "xy":
                U, V = B[rr2, cc2, 0], B[rr2, cc2, 1]
            elif vector_order == "yx":
                U, V = B[rr2, cc2, 1], B[rr2, cc2, 0]
            else:
                raise ValueError("vector_order must be 'xy' or 'yx'.")
            if flip_y_vector:
                V = -V
            if arrow_scale is None:
                ax.quiver(cc2, rr2, U, V, angles="xy", scale_units="xy")
            else:
                ax.quiver(cc2, rr2, U, V, angles="xy", scale_units="xy", scale=arrow_scale)
        ax.set_xlim(-0.5, A.shape[1] - 0.5)
        ax.set_ylim(A.shape[0] - 0.5, -0.5)
        ax.set_title("response vector field")
        ax.axis("off")

        ax = axes[row, 3]
        ax.plot(np.arange(len(h)), h, marker="o")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("activation h(t)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compute_episode_mode_reconstruction_maps(episode: MotionEpisode):
    if episode.motion_abs is None or episode.global_motion is None:
        return None, None, None
    mask = np.asarray(episode.spatial_region).astype(bool)
    coords = np.argwhere(mask)
    motion_abs = np.asarray(episode.motion_abs, dtype=np.float32)
    global_motion = np.asarray(episode.global_motion, dtype=np.float32)
    Y_data = motion_abs - global_motion[:, None, :]
    recon = np.zeros_like(Y_data, dtype=np.float32)
    for mode in getattr(episode, "modes", []):
        h = np.asarray(mode.activation, dtype=np.float32)
        B = np.asarray(mode.response_field, dtype=np.float32)
        Bc = B[coords[:, 0], coords[:, 1], :]
        recon += h[:, None, None] * Bc[None, :, :]
    resid = Y_data - recon
    obs_mag = np.sqrt(np.mean(np.sum(Y_data ** 2, axis=-1), axis=0))
    rec_mag = np.sqrt(np.mean(np.sum(recon ** 2, axis=-1), axis=0))
    res_mag = np.sqrt(np.mean(np.sum(resid ** 2, axis=-1), axis=0))
    return (
        _compact_to_full(obs_mag, coords, mask.shape),
        _compact_to_full(rec_mag, coords, mask.shape),
        _compact_to_full(res_mag, coords, mask.shape),
    )


def _compact_to_full(values, coords, shape, fill=0.0):
    values = np.asarray(values)
    coords = np.asarray(coords)
    if values.ndim == 1:
        out = np.full(shape, fill, dtype=np.float32)
        for i, (r, c) in enumerate(coords):
            out[r, c] = values[i]
        return out
    else:
        out = np.full((*shape, values.shape[1]), fill, dtype=np.float32)
        for i, (r, c) in enumerate(coords):
            out[r, c, :] = values[i]
        return out


def visualize_episode_regions(
    episode: MotionEpisode,
    ref_img=None,
    patch_size=7,
    max_regions=20,
    rel_thresh=0.10,
    figsize=(10, 8),
    save_path=None,
    show=True,
):
    regions = getattr(episode, "regions", [])
    if len(regions) == 0:
        print(f"Episode {episode.episode_id}: no regions.")
        return None
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if ref_img is not None:
        ax.imshow(_auto_contrast(ref_img), cmap="gray", origin="upper")
    else:
        ax.imshow(np.asarray(episode.spatial_region).T, cmap="gray", origin="upper")
    cmap = cm.get_cmap("tab20")
    for i, r in enumerate(regions[:max_regions]):
        A = np.asarray(r.response_strength, dtype=np.float32)
        vmax = float(np.max(A))
        if vmax <= 0:
            continue
        mask = A > rel_thresh * vmax
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
        vals = A[mask]
        vals = vals / (vmax + 1e-12)
        color = np.array(cmap(i % 20))
        X = coords[:, 0]
        Y = coords[:, 1]
        col = Y * patch_size + patch_size // 2
        row = X * patch_size + patch_size // 2
        rgba = np.tile(color, (len(vals), 1))
        rgba[:, 3] = 0.25 + 0.65 * vals
        ax.scatter(col, row, s=15 + 45 * vals, c=rgba, edgecolors="none", label=f"R{i}/M{r.mode_id}")
        v = np.asarray(r.mean_response_vector, dtype=np.float32)
        if np.all(np.isfinite(v)):
            # draw one representative arrow at region center
            cx, cy = r.center_xy
            ax.arrow(cy * patch_size + patch_size // 2, cx * patch_size + patch_size // 2,
                     v[1] * 5.0, v[0] * 5.0, color=color, head_width=3, head_length=4,
                     length_includes_head=True, alpha=0.8)
    ax.set_title(f"Episode {episode.episode_id}: MotionRegions n={len(regions)}")
    ax.set_aspect("equal")
    if ref_img is not None:
        H, W = ref_img.shape[:2]
        ax.set_xlim([0, W])
        ax.set_ylim([H, 0])
    else:
        ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def compare_sources_to_observed_frames(
    ep,
    max_modes=20,
    sort_by_peak=True,
    quiver_step=3,
    figsize_per_row=(12, 3),
):
    B, H = _get_BH_from_episode(ep)
    M, Y = _episode_motion_matrix(ep, use_global_subtracted=True)

    K, T = H.shape
    diag = diagnose_temporal_basis_like_sources(ep)

    if sort_by_peak:
        order = sorted(range(K), key=lambda k: diag[k]["peak_frame"])
    else:
        order = list(range(K))

    order = order[:min(max_modes, K)]

    n = len(order)
    fig, axes = plt.subplots(n, 3, figsize=(figsize_per_row[0], figsize_per_row[1] * n))
    if n == 1:
        axes = axes[None, :]

    for row, k in enumerate(order):
        h = H[k]
        t_peak = int(np.argmax(np.abs(h)))

        # source response B_k
        bx_b, by_b, mag_b, mask = _B_column_to_maps(ep, B[:, k])

        # source contribution at peak frame: h_k(t_peak) * B_k
        contrib_vec = B[:, k] * h[t_peak]
        bx_c, by_c, mag_c, _ = _frame_vector_to_maps(ep, contrib_vec)

        # observed motion at peak frame
        obs_vec = M[:, t_peak]
        bx_o, by_o, mag_o, _ = _frame_vector_to_maps(ep, obs_vec)

        maps = [
            (mag_b, bx_b, by_b, f"B_k response | k={k}, peak={t_peak}"),
            (mag_c, bx_c, by_c, f"h_k(t_peak) B_k"),
            (mag_o, bx_o, by_o, f"Observed M[:, {t_peak}]"),
        ]

        for col, (mag, bx, by, title) in enumerate(maps):
            ax = axes[row, col]
            ax.imshow(mask, cmap="gray", alpha=0.2)
            ax.imshow(np.ma.masked_where(mag <= 0, mag), cmap="magma", alpha=0.9)

            yy, xx = np.where(mag > 0)
            if len(yy) > 0:
                keep = (yy % quiver_step == 0) & (xx % quiver_step == 0)
                yy2 = yy[keep]
                xx2 = xx[keep]
                ax.quiver(
                    xx2,
                    yy2,
                    bx[yy2, xx2],
                    by[yy2, xx2],
                    angles="xy",
                    scale_units="xy",
                )

            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def summarize_temporal_basis_likeness(episodes):
    rows = []

    for ei, ep in enumerate(episodes):
        if not hasattr(ep, "mode_model") or ep.mode_model is None:
            continue

        B, H = _get_BH_from_episode(ep)
        K, T = H.shape

        diag = diagnose_temporal_basis_like_sources(ep)

        peak_fracs = np.array([d["peak_fraction"] for d in diag], dtype=np.float32)
        prs = np.array([d["temporal_participation_ratio"] for d in diag], dtype=np.float32)
        ent = np.array([d["temporal_entropy"] for d in diag], dtype=np.float32)
        peaks = np.array([d["peak_frame"] for d in diag], dtype=np.int32)

        unique_peak_count = len(np.unique(peaks))

        rows.append({
            "episode_index": ei,
            "episode_id": getattr(ep, "episode_id", None),
            "T": T,
            "K": K,
            "mean_peak_fraction": float(np.mean(peak_fracs)),
            "median_peak_fraction": float(np.median(peak_fracs)),
            "mean_temporal_PR": float(np.mean(prs)),
            "median_temporal_PR": float(np.median(prs)),
            "mean_entropy": float(np.mean(ent)),
            "unique_peak_count": unique_peak_count,
            "unique_peak_count_over_T": unique_peak_count / max(T, 1),
        })

    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except Exception:
        return rows


def save_episode_mode_region_gallery(
    motion_episodes,
    out_dir,
    episode_ids=None,
    max_episodes=None,
    patch_size=7,
    max_modes=8,
    max_regions=20,
    rel_thresh=0.10,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if episode_ids is None:
        eps = list(motion_episodes)
    else:
        ids = set(episode_ids)
        eps = [ep for ep in motion_episodes if ep.episode_id in ids]
    if max_episodes is not None:
        eps = eps[:max_episodes]
    saved = []
    for ep in eps:
        if len(getattr(ep, "modes", [])) > 0:
            p = out_dir / f"episode_{ep.episode_id:04d}_modes.png"
            visualize_episode_modes(ep, patch_size=patch_size, max_modes=max_modes, rel_thresh=rel_thresh, save_path=p, show=False)
            saved.append(p)
        if len(getattr(ep, "regions", [])) > 0:
            p = out_dir / f"episode_{ep.episode_id:04d}_regions.png"
            visualize_episode_regions(ep, patch_size=patch_size, max_regions=max_regions, rel_thresh=rel_thresh, save_path=p, show=False)
            saved.append(p)
    print(f"Saved {len(saved)} figures to {out_dir}")
    return saved


def _region_center_from_mask(mask):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    return pts.mean(axis=0)


def _get_region_mask(region):
    if hasattr(region, "region_mask") and region.region_mask is not None:
        return np.asarray(region.region_mask).astype(bool)

    if hasattr(region, "response_strength") and region.response_strength is not None:
        A = np.asarray(region.response_strength, dtype=np.float32)
        if A.size > 0 and np.max(A) > 0:
            return A > 0

    return None


def _get_region_strength(region):
    if hasattr(region, "response_strength") and region.response_strength is not None:
        return np.asarray(region.response_strength, dtype=np.float32)

    mask = _get_region_mask(region)
    if mask is None:
        return None

    return mask.astype(np.float32)


def diagnose_episode_svd_rank(ep, use_global_subtracted=True, ranks=(1, 2, 3, 5, 10, 20)):
    """
    Check the best possible low-rank reconstruction upper bound for one episode.
    """
    if use_global_subtracted:
        Y = np.asarray(ep.motion_abs, dtype=np.float32) - np.asarray(ep.global_motion, dtype=np.float32)[:, None, :]
    else:
        Y = np.asarray(ep.motion_abs, dtype=np.float32)

    # Y: (T, N, 2)
    T, N, _ = Y.shape

    # M: (2N, T)
    M = np.concatenate([Y[:, :, 0].T, Y[:, :, 1].T], axis=0)

    U, S, Vt = np.linalg.svd(M, full_matrices=False)

    total_energy = np.sum(S ** 2) + 1e-12

    result = {}
    for r in ranks:
        r = min(r, len(S))
        explained = np.sum(S[:r] ** 2) / total_energy
        result[r] = float(explained)

    return result


def visualize_episode_regions_overview(
    episode,
    max_regions=80,
    min_area=0,
    color_by="region",   # "region" or "mode"
    show_id=True,
    show_center=True,
    figsize=(8, 8),
    title=None,
):
    """
    Visualize all MotionRegions in one episode.

    color_by:
        "region": each region has a different label/color
        "mode": regions from the same parent mode have the same label/color
    """
    regions = getattr(episode, "regions", [])
    if len(regions) == 0:
        print(f"Episode {getattr(episode, 'episode_id', None)} has no regions.")
        return None

    mask_ep = np.asarray(episode.spatial_region).astype(bool)
    label_map = np.zeros(mask_ep.shape, dtype=np.int32)

    region_records = []

    for idx, r in enumerate(regions):
        mask = _get_region_mask(r)
        if mask is None:
            continue

        area = int(mask.sum())
        if area < min_area:
            continue

        region_records.append((idx, r, mask, area))

    # sort by area, keep largest max_regions
    region_records = sorted(region_records, key=lambda x: x[3], reverse=True)
    region_records = region_records[:max_regions]

    for display_idx, (idx, r, mask, area) in enumerate(region_records, start=1):
        if color_by == "mode":
            val = int(getattr(r, "mode_id", 0)) + 1
        else:
            val = display_idx

        label_map[mask] = val

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(mask_ep, cmap="gray", alpha=0.25)
    im = ax.imshow(np.ma.masked_where(label_map == 0, label_map), cmap="tab20", alpha=0.85)

    if show_id or show_center:
        for display_idx, (idx, r, mask, area) in enumerate(region_records, start=1):
            center = _region_center_from_mask(mask)
            if center is None:
                continue

            y, x = center
            if show_center:
                ax.scatter([x], [y], s=15, c="white", edgecolors="black", linewidths=0.5)

            if show_id:
                mode_id = getattr(r, "mode_id", -1)
                region_id = getattr(r, "region_id", idx)
                txt = f"{mode_id}:{region_id}" if color_by == "mode" else str(display_idx)
                ax.text(
                    x, y, txt,
                    color="white",
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1, edgecolor="none"),
                )

    ep_id = getattr(episode, "episode_id", None)
    tr = getattr(episode, "time_range", None)
    if title is None:
        title = f"Episode {ep_id} regions | time={tr} | n={len(region_records)}"

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return fig
# =============================================================================
# Diagnostics
# =============================================================================


def mode_activation_corr_matrix(ep: MotionEpisode):
    modes = getattr(ep, "modes", [])
    if len(modes) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    H = np.stack([np.asarray(m.activation).reshape(-1) for m in modes], axis=0)
    H = H - H.mean(axis=1, keepdims=True)
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    return (H @ H.T).astype(np.float32)


def mode_response_iou_matrix(ep: MotionEpisode, rel_thresh=0.10):
    modes = getattr(ep, "modes", [])
    n = len(modes)
    out = np.zeros((n, n), dtype=np.float32)
    masks = []
    for m in modes:
        A = np.asarray(m.response_strength)
        if A.size == 0 or np.max(A) <= 0:
            masks.append(np.zeros_like(A, dtype=bool))
        else:
            masks.append(A > rel_thresh * np.max(A))
    for i in range(n):
        for j in range(n):
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            out[i, j] = inter / (union + 1e-12)
    return out


def mode_incremental_contribution(ep: MotionEpisode):
    model = getattr(ep, "mode_model", None)
    if not model or "B" not in model or "H" not in model:
        return np.array([], dtype=np.float32)
    B = model["B"]
    H = model["H"]
    motion_abs = np.asarray(ep.motion_abs, dtype=np.float32)
    global_motion = np.asarray(ep.global_motion, dtype=np.float32)
    Y_data = motion_abs - global_motion[:, None, :]
    M = np.concatenate([Y_data[:, :, 0].T, Y_data[:, :, 1].T], axis=0)
    err_full = float(np.sum((M - B @ H) ** 2))
    total = float(np.sum(M ** 2)) + 1e-12
    vals = []
    for k in range(H.shape[0]):
        B_wo = B.copy()
        B_wo[:, k] = 0.0
        err_wo = float(np.sum((M - B_wo @ H) ** 2))
        vals.append((err_wo - err_full) / total)
    return np.asarray(vals, dtype=np.float32)

