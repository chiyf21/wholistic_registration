# Wholistic Registration v2

A clean, modular reimplementation of the volumetric image registration pipeline.

## Features

- **Multiple input formats**: ND2, Zarr, TIFF (single or series)
- **OME-TIFF output**: Standard format with embedded metadata
- **Clean configuration**: YAML-based with validation
- **Progress tracking**: Rich progress bars and logging
- **Callbacks**: Hook into pipeline events for monitoring
- **GPU acceleration**: CuPy-based with CPU fallback
- **Comprehensive tests**: Unit tests with synthetic data

## Installation

The v2 module is part of the wholistic_registration package. Additional dependencies:

```bash
pip install pyyaml rich tifffile zarr h5py scipy scikit-image
pip install cupy-cuda11x  # For GPU support (adjust for your CUDA version)
```

## Quick Start

### Basic Usage

```python
from wholistic_registration.v2 import RegistrationPipeline, RegistrationConfig

# Load configuration
config = RegistrationConfig.from_yaml("config.yaml")

# Run pipeline
pipeline = RegistrationPipeline(config)
pipeline.run()
```

### Creating Configuration

```python
from wholistic_registration.v2.config import (
    RegistrationConfig,
    DownsampleConfig,
    ChannelConfig,
    PyramidConfig,
)

config = RegistrationConfig(
    input_path="/path/to/data.nd2",
    output_dir="/path/to/output/",
    
    downsample=DownsampleConfig(
        xy=4,              # Downsample XY by 4x
        z_slices=[4,5,6],  # Use only these Z slices
        t_chunk=20,        # Process 20 frames per chunk
    ),
    
    channels=ChannelConfig(
        dual_channel=True,  # Use membrane + calcium
        transform='log10',  # log10(1+x) transform for calcium
        k=50.0,            # Weight for calcium contribution
    ),
    
    pyramid=PyramidConfig(
        layers=1,          # Pyramid depth
        patch_radius=5,    # Patch size for motion estimation
        iterations=10,     # Iterations per level
    ),
)

# Save to YAML for later use
config.save_yaml("my_config.yaml")
```

### YAML Configuration

```yaml
input_path: /path/to/data.nd2
output_dir: /path/to/output/

downsample:
  xy: 4
  z_slices: [4, 5, 6]
  t_chunk: 20

channels:
  dual_channel: true
  transform: log10
  k: 50.0

reference:
  window_size: 20
  initial_frames: 40

pyramid:
  layers: 1
  patch_radius: 5
  iterations: 10
  smooth_penalty: 0.08

output:
  save_reference: true
  save_motion: false
  compression: zlib

backend:
  device: cuda
  gpu_id: 0
```

### Using Callbacks

```python
def on_frame_complete(event, data):
    print(f"Frame {data['frame_idx']}: error {data['error']['final']:.4f}")

def on_chunk_complete(event, data):
    print(f"Completed {data['chunk']} chunk: frames {data['frames']}")

pipeline = RegistrationPipeline(config)
pipeline.callbacks.register('on_frame_complete', on_frame_complete)
pipeline.callbacks.register('on_chunk_complete', on_chunk_complete)
pipeline.run()
```

## Module Structure

```
v2/
├── config/           # Configuration dataclasses
│   └── settings.py   # All config definitions
├── io/               # Input/output
│   ├── readers.py    # ND2, Zarr, TIFF readers
│   ├── writers.py    # OME-TIFF writer
│   └── metadata.py   # Unified metadata
├── core/             # Registration algorithms
│   ├── transforms.py # Channel transforms
│   ├── reference.py  # Reference computation
│   └── registration.py # Frame registration
├── pipeline/         # Pipeline orchestration
│   └── runner.py     # Main pipeline
├── utils/            # Utilities
│   ├── array_ops.py  # GPU/CPU operations
│   ├── logging.py    # Logging + progress
│   └── validation.py # Input validation
├── tests/            # Unit tests
│   ├── synthetic_data.py  # Test data generator
│   └── test_*.py     # Test modules
└── examples/         # Usage examples
    └── synthetic_example.py
```

## Running Tests

```bash
cd src/wholistic_registration/v2
pytest tests/ -v
```

## Differences from v1

### Output Format
- **v1**: Zarr or OME-TIFF (configurable)
- **v2**: OME-TIFF only (standard format)

### Configuration
- **v1**: TOML file with many sections
- **v2**: YAML with validated dataclasses

### Parameter Names

| v1 Parameter | v2 Parameter | Notes |
|--------------|--------------|-------|
| `downsampleT` | `downsample.t_chunk` | Renamed for clarity |
| `downsampleXY` | `downsample.xy` | Same meaning |
| `downsampleZ` | `downsample.z_slices` | Same meaning |
| `chunk_size` | `reference.window_size` | Rolling window size |
| `mid_chunk_size` | `reference.initial_frames` | Initial block size |
| `layer` | `pyramid.layers` | Same meaning |
| `r` | `pyramid.patch_radius` | Renamed for clarity |
| `iter` | `pyramid.iterations` | Same meaning |
| `smoothPenalty` | `pyramid.smooth_penalty` | Same meaning |

### Algorithm
The core motion estimation algorithm (`calFlow3d_Wei_v1`) is **unchanged**. 
Only the interface and organization have been cleaned up.

## Synthetic Data Testing

Generate test data with known motion:

```python
from wholistic_registration.v2.tests.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
)

config = SyntheticDataConfig(
    n_frames=50,
    shape_zyx=(10, 128, 128),
    motion_type='sine',  # 'translation', 'sine', 'drift', 'none'
    motion_amplitude=5.0,
)

generator = SyntheticDataGenerator(config)
membrane, calcium, motion_gt = generator.generate()

# Save for testing
generator.save_as_zarr("/tmp/test_data.zarr")
```

