#!/usr/bin/env python3
"""
Example: Registration on Synthetic Data

This example demonstrates the full registration pipeline using
synthetic data with known ground truth motion.

Run from the v2 directory:
    python -m examples.synthetic_example
    
Or:
    python examples/synthetic_example.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import tempfile
import shutil

from wholistic_registration.v2.tests.synthetic_data import (
    SyntheticDataGenerator,
    SyntheticDataConfig,
)
from wholistic_registration.v2.config import (
    RegistrationConfig,
    DownsampleConfig,
    ChannelConfig,
    PyramidConfig,
    ReferenceConfig,
)
from wholistic_registration.v2.core.reference import ReferenceComputer
from wholistic_registration.v2.core.transforms import combine_channels


def main():
    """Run synthetic data example."""
    print("=" * 60)
    print("Wholistic Registration v2 - Synthetic Data Example")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\nWorking directory: {temp_dir}")
    
    try:
        # Step 1: Generate synthetic data
        print("\n[1] Generating synthetic data...")
        data_config = SyntheticDataConfig(
            n_frames=30,
            shape_zyx=(5, 64, 64),
            n_spheres=15,
            motion_type='sine',
            motion_amplitude=3.0,
            noise_level=0.05,
            random_seed=42,
        )
        
        generator = SyntheticDataGenerator(data_config)
        membrane, calcium, motion_gt = generator.generate()
        
        print(f"   Generated membrane: {membrane.shape}")
        print(f"   Generated calcium: {calcium.shape}")
        print(f"   Ground truth motion: {motion_gt.shape}")
        print(f"   Motion amplitude: max={np.abs(motion_gt).max():.2f} pixels")
        
        # Save as Zarr for testing
        zarr_path = generator.save_as_zarr(str(temp_dir / "synthetic.zarr"))
        print(f"   Saved to: {zarr_path}")
        
        # Step 2: Test reference computation
        print("\n[2] Testing reference computation...")
        ref_computer = ReferenceComputer(
            max_correlation_frames=20,
            device='cpu',  # Use CPU for example (no GPU required)
        )
        
        channel_config = ChannelConfig(
            dual_channel=True,
            transform='log10',
            k=50.0,
        )
        
        # Use middle frames for reference
        mid_start = 10
        mid_end = 20
        reference = ref_computer.compute(
            membrane[mid_start:mid_end],
            calcium[mid_start:mid_end],
            channel_config,
        )
        
        print(f"   Reference shape: {reference.shape}")
        print(f"   Reference range: [{reference.min():.1f}, {reference.max():.1f}]")
        
        # Step 3: Test channel combination
        print("\n[3] Testing channel transforms...")
        combined = combine_channels(
            membrane[0], calcium[0],
            transform='log10', k=50.0
        )
        print(f"   Membrane mean: {membrane[0].mean():.1f}")
        print(f"   Calcium mean: {calcium[0].mean():.1f}")
        print(f"   Combined mean: {combined.mean():.1f}")
        
        # Step 4: Show motion statistics
        print("\n[4] Ground truth motion statistics:")
        print(f"   X motion: mean={motion_gt[..., 2].mean():.2f}, "
              f"std={motion_gt[..., 2].std():.2f}")
        print(f"   Y motion: mean={motion_gt[..., 1].mean():.2f}, "
              f"std={motion_gt[..., 1].std():.2f}")
        print(f"   Z motion: mean={motion_gt[..., 0].mean():.2f}, "
              f"std={motion_gt[..., 0].std():.2f}")
        
        # Step 5: Create example config
        print("\n[5] Creating example configuration...")
        output_dir = temp_dir / "registered_output"
        
        # Note: This creates the config but doesn't run registration
        # (which requires the legacy modules)
        config_dict = {
            'input_path': str(zarr_path),
            'output_dir': str(output_dir),
            'downsample': {
                'xy': 1,
                't_chunk': 10,
            },
            'channels': {
                'dual_channel': True,
                'transform': 'log10',
                'k': 50.0,
            },
            'reference': {
                'window_size': 10,
                'initial_frames': 20,
            },
            'pyramid': {
                'layers': 1,
                'patch_radius': 5,
                'iterations': 10,
            },
        }
        
        # Save example YAML config
        import yaml
        config_path = temp_dir / "example_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"   Saved example config to: {config_path}")
        print("\n   Config contents:")
        with open(config_path) as f:
            for line in f:
                print(f"   {line.rstrip()}")
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)
        
        print(f"\nTo run full registration (requires GPU and legacy modules):")
        print(f"  from wholistic_registration.v2 import RegistrationPipeline, RegistrationConfig")
        print(f"  config = RegistrationConfig.from_yaml('{config_path}')")
        print(f"  pipeline = RegistrationPipeline(config)")
        print(f"  pipeline.run()")
        
    finally:
        # Cleanup
        print(f"\nCleaning up {temp_dir}...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Done!")


if __name__ == "__main__":
    main()

