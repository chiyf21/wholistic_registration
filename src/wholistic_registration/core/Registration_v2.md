# Registration_v2: Complete Pipeline Documentation

## Overview

`Registration_v2` is a volumetric image registration pipeline designed for dual-channel microscopy data (typically membrane + calcium channels). It performs motion correction by registering each frame to a dynamically-updated reference image, outputting results as OME-TIFF files.

---

## Key Differences: Registration vs Registration_v2

| Aspect | Registration | Registration_v2 |
|--------|-------------|-----------------|
| **Output Format** | Zarr arrays | OME-TIFF per frame + HDF5 for motion |
| **Anchor Strategy** | Two-phase: anchors first, then fill | Single-phase: process chunks sequentially |
| **downsampleT Usage** | Creates sparse anchor frames, then fills gaps | Defines chunk size for batch processing |
| **Parallel Support** | None | Has `parallel` parameter (stub for multi-GPU) |
| **Reference Update** | Per-frame sliding window | Per-chunk sliding window |

### Parameters Only Used in Registration (not Registration_v2)

The following parameters from `DefineParams` are used in `Registration` but **NOT** in `Registration_v2`:

| Parameter | Purpose in Registration | Status in Registration_v2 |
|-----------|------------------------|---------------------------|
| None critical | Both use similar config | All major params are used |

Both functions use the same config parameters. The key difference is **how** they use `downsampleT`:
- **Registration**: Uses it to create anchor frames (every Nth frame), then fills in between
- **Registration_v2**: Uses it as the chunk size for sequential batch processing

---

## Algorithm Walkthrough

### Phase 1: Initialization and Setup (Lines 536-606)

```
1. Load configuration from TOML file
2. Extract paths: input ND2, output directory
3. Get metadata: dimensions, total frames, downsampling factors
4. Create output directories:
   └── output_root/
       ├── membrane/    (OME-TIFF per frame)
       ├── calcium/     (OME-TIFF per frame)
       ├── reference/   (optional, OME-TIFF per chunk)
       └── motion/      (optional, HDF5 per frame)
```

**Key Point**: Unlike `Registration` which pre-allocates a zarr array for all frames, `Registration_v2` writes individual files incrementally.

---

### Phase 2: Middle Block Processing (Lines 607-664)

**Purpose**: Establish a stable initial reference from the temporal center of the video.

```
Timeline visualization:
|-------|=======MIDDLE=======|-------|
0      mid_start           mid_end  total_frames
        ↑                    ↑
        |←── mid_chunk_size ─→|
```

**Steps**:
1. **Calculate middle block bounds**:
   ```python
   mid_start = (total_frames // 2) - (mid_chunk_size // 2)
   mid_end = mid_start + mid_chunk_size
   ```

2. **Load middle frames** from ND2 file:
   - Read membrane channel (channel=1)
   - Read calcium channel (channel=0)
   - Apply XY and Z downsampling

3. **Compute initial reference** using `compute_reference_from_block()`:
   - Picks the most temporally-stable frame via correlation analysis
   - If dual_channel: combines as `membrane + k * transform(calcium)`

4. **Register middle block** against reference:
   - Calls `wbi_registration_3d()` or `wbi_registration_2d()`
   - Outputs: registered membrane, calcium, motion fields

5. **Save results** as individual OME-TIFF files

---

### Phase 3: Backward Processing (Lines 440-503)

**Purpose**: Process frames from middle toward the beginning, maintaining temporal continuity by always registering against recently-registered frames.

```
Direction: ←←←←←←←←←
|←chunk→|←chunk→|←chunk→|=====MIDDLE=====|
0                      mid_start
```

---

#### The Rolling Reference Window Explained

The **rolling window** is a buffer of the N most recently registered frames (where N = `chunk_size`). This window serves as the source for computing the reference image for the next chunk.

**Why a rolling window?**
- Registration quality degrades if the reference is too different from the moving image
- By using recently-registered frames as reference, we ensure temporal continuity
- The reference "follows" the data through time, adapting to gradual changes

**Initialization** (from the registered middle block):
```python
# Take the FIRST chunk_size frames from the middle block
# These are temporally closest to what we'll process next (going backward)
ref_windows_mem = np.array(mem_mid_reg[0:chunk_size])  # Shape: (chunk_size, Z, Y, X)
ref_windows_ca  = np.array(ca_mid_reg[0:chunk_size])   # Shape: (chunk_size, Z, Y, X)
```

```
Middle block (registered):
Frame indices: [mid_start, mid_start+1, ..., mid_end-1]
                ↑_________________________↑
                |   chunk_size frames     |
                └─────────────────────────┘
                       ↓
                ref_windows_mem/ca
```

---

#### Reference Computation: `compute_reference_from_block()`

This function takes the window of frames and produces a single reference image:

```python
def compute_reference_from_block(mem_block, ca_block, config):
    """
    Input:
      mem_block: (chunk_size, Z, Y, X) - membrane channel frames
      ca_block:  (chunk_size, Z, Y, X) - calcium channel frames
      config:    dict with channel settings
    
    Output:
      reference image: (Z, Y, X) - single volume
    """
```

**Step 1: Find the most stable frame via correlation**
```python
# pick_initial_reference() computes pairwise correlations between all frames
# and finds the frame most similar to all others

# 1. Flatten each frame to a vector
frames_flat = frames.reshape(T, -1)  # (T, Z*Y*X)

# 2. Compute correlation matrix
frames_demeaned = frames_flat - frames_flat.mean(axis=1, keepdims=True)
cc = frames_demeaned @ frames_demeaned.T  # (T, T) correlation matrix
cc = cc / (diag[:, None] * diag[None, :])  # Normalize to [-1, 1]

# 3. For each frame, compute mean correlation with top-N most similar frames
ncorr = min(max_corr_frames, T - 1)
CCsort = -cp.sort(-cc, axis=1)  # Sort correlations descending
bestCC = CCsort[:, 1:ncorr].mean(axis=1)  # Mean of top correlations (exclude self)

# 4. Pick the frame with highest mean correlation
imax = int(cp.argmax(bestCC).item())

# 5. Get indices of top-N frames most correlated with the best frame
indsort = cp.argsort(-cc[imax])
top_indices = indsort[:ncorr]
```

**Step 2: Average the top frames**
```python
# Average the top-N most stable frames at full resolution
mem_ref = frames[top_indices].mean(axis=0)  # (Z, Y, X)
```

**Step 3: Combine channels (if dual_channel=True)**
```python
if dual_channel:
    # Average calcium from same top frames
    Ca_ref = ca_block[top_indices].mean(axis=0)
    
    # Transform calcium: e.g., k * log10(1 + calcium)
    Ca_ref_transform = k * np.log10(1 + Ca_ref)
    
    # Combine: membrane + weighted_transformed_calcium
    reference = mem_ref + Ca_ref_transform
else:
    reference = mem_ref
```

**Visual explanation**:
```
Window frames:     [F0, F1, F2, ..., F_chunk_size]
                    ↓   ↓   ↓
Correlation matrix: ┌─────────────────────┐
                    │ 1.0  0.9  0.7  ...  │
                    │ 0.9  1.0  0.8  ...  │
                    │ 0.7  0.8  1.0  ...  │
                    │ ...                 │
                    └─────────────────────┘
                           ↓
Best frame index:    imax = 1  (highest mean correlation)
                           ↓
Top-N indices:      [1, 0, 2, 5, ...]  (sorted by correlation to frame 1)
                           ↓
Reference image:    mean(F1, F0, F2, F5, ...)  →  Single (Z,Y,X) volume
```

---

#### Loop Structure (Backward)

```python
for idx in range(mid_start, -1, -downsampleT):
    # idx steps: mid_start → mid_start-downsampleT → ... → 0
    
    end_idx = idx
    start_idx = max(0, end_idx - downsampleT)
    
    # Chunk to process: frames [start_idx, end_idx)
    frames_backward = list(range(start_idx, end_idx))
```

**Concrete Example** (mid_start=100, downsampleT=20, chunk_size=10):

| Iteration | idx | start_idx | end_idx | Frames processed | Reference from |
|-----------|-----|-----------|---------|------------------|----------------|
| 1 | 100 | 80 | 100 | [80-99] | Middle block frames [100-109] |
| 2 | 80 | 60 | 80 | [60-79] | Just-registered frames [80-89] |
| 3 | 60 | 40 | 60 | [40-59] | Just-registered frames [60-69] |
| 4 | 40 | 20 | 40 | [20-39] | Just-registered frames [40-49] |
| 5 | 20 | 0 | 20 | [0-19] | Just-registered frames [20-29] |

---

#### Window Update Strategy

After registering a chunk, the window is updated with the newly registered frames:

```python
# Replace entire window with first chunk_size frames from just-registered batch
ref_windows_mem = np.array(mem_backward_reg[0:chunk_size])
ref_windows_ca  = np.array(ca_backward_reg[0:chunk_size])
```

**Why use `[0:chunk_size]` (first frames of chunk)?**

When going **backward** in time, we want the reference to come from frames that are temporally **ahead** (higher frame numbers). After registering chunk [60-79], the first frames (60-69) are closest to what we'll process next (40-59).

```
Time:  0    20    40    60    80    100   (mid_start)
       ←─────────────────────────────────
                            ↑
                    [60-79] just registered
                     ↑_________↑
                     |chunk_size|
                     └──────────┘
                           ↓
                    New ref_window = [60-69]
                           ↓
              Will be used to register [40-59]
```

**Contrast with forward direction** (uses `[-chunk_size:]`):
```python
# For forward: use LAST chunk_size frames (temporally closest to next chunk)
ref_windows_mem = np.array(mem_forward_reg[-chunk_size:])
```

```
Time:  (mid_end)   140   160   180   200
       ─────────────────────────────────→
              ↑
       [140-159] just registered
              ↑_________↑
              |chunk_size|
              └──────────┘
                    ↓
         New ref_window = [150-159]
                    ↓
       Will be used to register [160-179]
```

---

### Phase 4: Forward Processing (Lines 745-805)

**Purpose**: Process frames from middle toward the end.

```
Direction: →→→→→→→→→
|=====MIDDLE=====|→chunk→|→chunk→|→chunk→|
                mid_end              total_frames
```

**Mirror of backward**, but:
- Window initialized from END of middle block: `mem_mid_reg[-chunk_size:]`
- Loop goes UP: `range(mid_end, total_frames, downsampleT)`
- Window updated with LAST `chunk_size` frames from each registered batch

---

## Data Flow Diagram

```
                    ┌─────────────────────────────────────┐
                    │         INPUT: ND2 File             │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │      Read Middle Block Frames       │
                    │   (mid_chunk_size frames @ center)  │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  Compute Initial Reference Image    │
                    │  (correlation-based frame picking)  │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │    Register Middle Block → Save     │
                    └─────────────────┬───────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
    ┌─────────▼─────────┐                         ┌───────────▼───────────┐
    │ BACKWARD LOOP     │                         │   FORWARD LOOP        │
    │ (mid_start → 0)   │                         │   (mid_end → end)     │
    │                   │                         │                       │
    │ For each chunk:   │                         │ For each chunk:       │
    │ 1. Load frames    │                         │ 1. Load frames        │
    │ 2. Ref from window│                         │ 2. Ref from window    │
    │ 3. Register       │                         │ 3. Register           │
    │ 4. Save TIFFs     │                         │ 4. Save TIFFs         │
    │ 5. Update window  │                         │ 5. Update window      │
    └─────────┬─────────┘                         └───────────┬───────────┘
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │           OUTPUT FILES              │
                    │  membrane/*.tif, calcium/*.tif      │
                    │  reference/*.tif, motion/*.h5       │
                    └─────────────────────────────────────┘
```

---

## Parameter Reference

### Downsampling Parameters

| Parameter | Effect |
|-----------|--------|
| `downsampleXY` | Spatial reduction factor. XY dimensions divided by this value. |
| `downsampleZ` | List of Z-slice indices to include (0-indexed). |
| `downsampleT` | **Chunk size** for batch processing in Registration_v2. |

### Reference Parameters

| Parameter | Effect |
|-----------|--------|
| `chunk_size` | Size of rolling window for reference computation. |
| `mid_chunk_size` | Number of frames at video center used for initial reference. |

### Dual-Channel Parameters

| Parameter | Effect |
|-----------|--------|
| `dual_channel` | If True: `combined = membrane + k * function(calcium)` |
| `function` | Transform: `'log10'` → log₁₀(1+x), `'sqrt'`, `'log2'`, `'raw'` |
| `k` | Weight for calcium channel contribution. |

### Pyramid Registration Parameters

| Parameter | Effect |
|-----------|--------|
| `layer` | Multi-scale pyramid depth. More layers = larger displacement capture. |
| `r` | Patch radius. Patch size = (2r+1)×(2r+1). |
| `iter` | Max iterations per pyramid level. |
| `smoothPenalty` | Regularization strength. Higher = smoother motion fields. |

---

## Code Critique and Issues

### 🔴 Bugs / Errors

1. **Off-by-one in backward loop (Line 696)**
   ```python
   for idx in range(mid_start, -1, -downsampleT):
   ```
   This starts at `mid_start` which is ALREADY processed in the middle block. Should be:
   ```python
   for idx in range(mid_start - downsampleT, -1, -downsampleT):
   ```
   
2. **Empty frames list possible (Line 700)**
   ```python
   frames_backward = list(range(start_idx, end_idx))
   ```
   When `idx == mid_start` and `start_idx = max(0, mid_start - downsampleT)`, this can include already-processed frames or produce empty ranges.

3. **Unused variable `archors` (Line 688)**
   ```python
   archors = []
   ```
   This list is initialized but never used. Leftover from `Registration` where it tracked anchor indices.

4. **Inconsistent reference saving filename (Lines 664, 739, 801)**
   ```python
   IO.write_volume_as_ome_tiff(ref_frame, out_ref, 'ref', f'{mid_start}~{mid_end}', configPath)
   ```
   Uses `~` as separator, but `ReferenceComparation` function expects `_` in regex pattern:
   ```python
   re.match(r"vol_chref_(\d+)_(\d+).tif", frame)  # Expects underscores!
   ```

5. **Duplicate code in if/else (Lines 660-663, 735-738, 797-800)**
   ```python
   if Dim == 3:
       ref_frame = ref_img.copy()
   else:
       ref_frame = ref_img.copy()  # Same code in both branches!
   ```

### 🟡 Inefficiencies

1. **Redundant reference computation (Line 693 + 707)**
   ```python
   ref_img = reference.compute_reference_from_block(ref_windows_mem, ref_windows_ca, config)  # Line 693
   # ... then again ...
   ref_img = reference.compute_reference_from_block(ref_windows_mem, ref_windows_ca, config)  # Line 707
   ```
   Computes reference twice with same data. Remove line 693.

2. **Reading entire chunks when partial would suffice**
   The window update uses only `chunk_size` frames but registers entire `downsampleT`-sized chunks. If `downsampleT > chunk_size`, memory is wasted.

3. **No GPU memory management**
   For large volumes, GPU memory isn't explicitly cleared between chunks. Consider adding:
   ```python
   import gc
   gc.collect()
   cp.get_default_memory_pool().free_all_blocks()  # If using CuPy
   ```

4. **Sequential I/O for motion files**
   Each frame's motion field is saved individually to HDF5. Batching these writes would be faster.

5. **np.squeeze() used unsafely (Lines 623-624, 703-704, 766-767)**
   ```python
   mem_mid = np.squeeze(mem_mid)
   ```
   If any dimension is accidentally 1, this could corrupt the expected shape. Use explicit indexing:
   ```python
   mem_mid = mem_mid[:, 0, ...]  # Remove known singleton dimension
   ```

### 🟢 Suggestions for Improvement

1. **Add progress bar**
   ```python
   from tqdm import tqdm
   for idx in tqdm(range(...), desc="Backward registration"):
   ```

2. **Checkpoint/resume capability**
   Check if output files exist before processing to allow resumption after crashes.

3. **Validation of rolling window size**
   Add check: `assert chunk_size <= downsampleT`, otherwise window update logic fails.

4. **Consolidate file naming**
   Create a helper function for consistent filename generation.

---

## Example Usage

```python
from wholistic_registration.core import main_function

# Step 1: Define parameters and generate config
main_function.DefineParams(
    configFile='./configs/config.toml',
    inputFile='/path/to/data.nd2',
    outputFile='./output/',
    downsampleXY=4,
    downsampleT=20,  # Process 20 frames per chunk
    downsampleZ=[4, 5, 6],
    chunk_size=10,   # Rolling window of 10 frames for reference
    mid_chunk_size=40,
    dual_channel=True,
    function='log10',
    k=50,
    layer=1,
    verbose=True
)

# Step 2: Run registration
main_function.Registration_v2('./configs/config.toml')
```

---

## Output Structure

```
output_root/
├── membrane/
│   ├── vol_ch1_000000.tif
│   ├── vol_ch1_000001.tif
│   └── ...
├── calcium/
│   ├── vol_ch0_000000.tif
│   ├── vol_ch0_000001.tif
│   └── ...
├── reference/  (if save_ref=True)
│   ├── vol_chref_100~140.tif  # Reference for middle block
│   ├── vol_chref_80~100.tif   # Reference for first backward chunk
│   └── ...
└── motion/  (if save_motion=True)
    ├── motion_000000.h5
    ├── motion_000001.h5
    └── ...
```

