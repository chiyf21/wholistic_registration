"""
Synthetic data generation for testing registration.

Generates 4D volumes (T, Z, Y, X) with:
- Static structural features (spheres, tubes)
- Known motion patterns (translation, rotation, deformation)
- Realistic noise characteristics
- Dual channels (membrane + calcium)

This allows testing registration accuracy by comparing
recovered motion to ground truth.
"""

from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation.
    
    Attributes:
        n_frames: Number of time points
        shape_zyx: Spatial dimensions (Z, Y, X)
        n_spheres: Number of spherical structures
        sphere_radius_range: (min, max) radius for spheres
        noise_level: Gaussian noise standard deviation (relative to signal)
        motion_type: Type of motion to apply
        motion_amplitude: Maximum motion in pixels
    """
    n_frames: int = 50
    shape_zyx: Tuple[int, int, int] = (10, 128, 128)
    n_spheres: int = 20
    sphere_radius_range: Tuple[int, int] = (3, 8)
    noise_level: float = 0.1
    motion_type: str = 'translation'  # 'translation', 'sine', 'drift', 'none'
    motion_amplitude: float = 5.0
    calcium_variation: float = 0.3  # Relative variation in calcium signal
    random_seed: Optional[int] = 42


class SyntheticDataGenerator:
    """Generates synthetic microscopy data with known motion.
    
    Example:
        >>> generator = SyntheticDataGenerator(SyntheticDataConfig())
        >>> membrane, calcium, motion_gt = generator.generate()
        >>> print(f"Generated data: {membrane.shape}")
        
        # Save as TIFF for testing
        >>> generator.save_as_tiff("/tmp/synthetic_data/")
    """
    
    def __init__(self, config: SyntheticDataConfig):
        """Initialize generator.
        
        Args:
            config: Generation configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
        # Generated data (populated by generate())
        self.membrane: Optional[np.ndarray] = None
        self.calcium: Optional[np.ndarray] = None
        self.motion_ground_truth: Optional[np.ndarray] = None
        self.base_volume: Optional[np.ndarray] = None
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data.
        
        Returns:
            Tuple of:
                - membrane: (T, Z, Y, X) membrane channel
                - calcium: (T, Z, Y, X) calcium channel
                - motion_gt: (T, Z, Y, X, 3) ground truth motion (z, y, x)
        """
        cfg = self.config
        Z, Y, X = cfg.shape_zyx
        T = cfg.n_frames
        
        # Step 1: Generate base structural volume
        self.base_volume = self._generate_structure()
        
        # Step 2: Generate motion field for each frame
        motion_gt = self._generate_motion()
        
        # Step 3: Apply motion to generate membrane channel
        membrane = np.zeros((T, Z, Y, X), dtype=np.float32)
        for t in range(T):
            membrane[t] = self._warp_volume(self.base_volume, motion_gt[t])
        
        # Step 4: Generate calcium channel with dynamic variation
        calcium = self._generate_calcium(membrane)
        
        # Step 5: Add noise
        membrane = self._add_noise(membrane)
        calcium = self._add_noise(calcium)
        
        self.membrane = membrane
        self.calcium = calcium
        self.motion_ground_truth = motion_gt
        
        return membrane, calcium, motion_gt
    
    def _generate_structure(self) -> np.ndarray:
        """Generate base structural features (spheres).
        
        Creates a volume with randomly placed spherical structures
        that mimic cell bodies or other features.
        """
        cfg = self.config
        Z, Y, X = cfg.shape_zyx
        volume = np.zeros((Z, Y, X), dtype=np.float32)
        
        # Create coordinate grids
        zz, yy, xx = np.mgrid[:Z, :Y, :X]
        
        # Add spheres
        for _ in range(cfg.n_spheres):
            # Random center
            cz = self.rng.integers(0, Z)
            cy = self.rng.integers(0, Y)
            cx = self.rng.integers(0, X)
            
            # Random radius
            r = self.rng.integers(cfg.sphere_radius_range[0], cfg.sphere_radius_range[1])
            
            # Random intensity
            intensity = self.rng.uniform(0.5, 1.0)
            
            # Distance from center
            dist = np.sqrt((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2)
            
            # Smooth sphere (Gaussian-like)
            sphere = intensity * np.exp(-0.5 * (dist / (r/2))**2)
            volume = np.maximum(volume, sphere)
        
        # Add some background texture
        background = self.rng.uniform(0.05, 0.1, size=(Z, Y, X)).astype(np.float32)
        volume = volume + background
        
        # Normalize to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        
        # Scale to realistic intensity range
        volume = volume * 1000 + 100
        
        return volume
    
    def _generate_motion(self) -> np.ndarray:
        """Generate motion field for all frames.
        
        Returns:
            (T, Z, Y, X, 3) motion field where last dimension is (dz, dy, dx)
        """
        cfg = self.config
        Z, Y, X = cfg.shape_zyx
        T = cfg.n_frames
        
        motion = np.zeros((T, Z, Y, X, 3), dtype=np.float32)
        
        if cfg.motion_type == 'none':
            return motion
        
        elif cfg.motion_type == 'translation':
            # Global translation that varies over time
            for t in range(T):
                # Slowly varying translation
                phase = 2 * np.pi * t / T
                dx = cfg.motion_amplitude * np.sin(phase)
                dy = cfg.motion_amplitude * np.cos(phase) * 0.5
                dz = cfg.motion_amplitude * np.sin(phase * 0.5) * 0.2
                
                motion[t, :, :, :, 0] = dz
                motion[t, :, :, :, 1] = dy
                motion[t, :, :, :, 2] = dx
        
        elif cfg.motion_type == 'sine':
            # Sinusoidal deformation pattern
            zz, yy, xx = np.mgrid[:Z, :Y, :X]
            for t in range(T):
                phase = 2 * np.pi * t / T
                # Spatially varying motion
                motion[t, :, :, :, 2] = cfg.motion_amplitude * np.sin(
                    phase + 2 * np.pi * yy / Y
                ) * 0.5
                motion[t, :, :, :, 1] = cfg.motion_amplitude * np.cos(
                    phase + 2 * np.pi * xx / X
                ) * 0.3
        
        elif cfg.motion_type == 'drift':
            # Gradual drift over time
            for t in range(T):
                drift = cfg.motion_amplitude * t / T
                motion[t, :, :, :, 2] = drift  # X drift
                motion[t, :, :, :, 1] = drift * 0.3  # Small Y drift
        
        else:
            raise ValueError(f"Unknown motion type: {cfg.motion_type}")
        
        return motion
    
    def _warp_volume(
        self,
        volume: np.ndarray,
        motion_field: np.ndarray,
    ) -> np.ndarray:
        """Apply motion field to warp a volume.
        
        Args:
            volume: (Z, Y, X) input volume
            motion_field: (Z, Y, X, 3) motion vectors (dz, dy, dx)
            
        Returns:
            Warped volume
        """
        from scipy.ndimage import map_coordinates
        
        Z, Y, X = volume.shape
        
        # Create coordinate grids
        zz, yy, xx = np.mgrid[:Z, :Y, :X].astype(np.float32)
        
        # Apply motion
        new_z = zz - motion_field[:, :, :, 0]
        new_y = yy - motion_field[:, :, :, 1]
        new_x = xx - motion_field[:, :, :, 2]
        
        # Warp using interpolation
        coords = np.array([new_z, new_y, new_x])
        warped = map_coordinates(volume, coords, order=1, mode='nearest')
        
        return warped.astype(np.float32)
    
    def _generate_calcium(self, membrane: np.ndarray) -> np.ndarray:
        """Generate calcium channel with temporal dynamics.
        
        Calcium signal is based on membrane but with:
        - Temporal variation (simulating neural activity)
        - Different spatial characteristics
        """
        cfg = self.config
        T = cfg.n_frames
        
        # Start with membrane as base
        calcium = membrane.copy()
        
        # Add temporal variation (activity patterns)
        for t in range(T):
            # Random multiplicative variation
            variation = 1.0 + cfg.calcium_variation * self.rng.uniform(-1, 1)
            calcium[t] = calcium[t] * variation
            
            # Add some sparse "active" regions
            if self.rng.random() > 0.7:
                # Random activation spot
                Z, Y, X = cfg.shape_zyx
                cy, cx = self.rng.integers(0, Y), self.rng.integers(0, X)
                yy, xx = np.mgrid[:Y, :X]
                spot = np.exp(-((yy - cy)**2 + (xx - cx)**2) / (20**2))
                for z in range(Z):
                    calcium[t, z] += spot * 200 * self.rng.uniform(0.5, 1.5)
        
        return calcium
    
    def _add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to data."""
        cfg = self.config
        noise_std = cfg.noise_level * data.std()
        noise = self.rng.normal(0, noise_std, size=data.shape).astype(np.float32)
        return np.maximum(data + noise, 0)  # Clamp to non-negative
    
    def save_as_tiff(self, output_dir: str) -> Path:
        """Save generated data as TIFF files.
        
        Creates:
            - membrane/frame_XXXXXX.tif
            - calcium/frame_XXXXXX.tif
            - motion_ground_truth.npy
            
        Args:
            output_dir: Directory to save files
            
        Returns:
            Path to output directory
        """
        import tifffile
        
        if self.membrane is None:
            raise RuntimeError("Call generate() first")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirs
        (output_path / "membrane").mkdir(exist_ok=True)
        (output_path / "calcium").mkdir(exist_ok=True)
        
        # Save frames
        T = self.membrane.shape[0]
        for t in range(T):
            mem_file = output_path / "membrane" / f"frame_{t:06d}.tif"
            ca_file = output_path / "calcium" / f"frame_{t:06d}.tif"
            
            tifffile.imwrite(mem_file, self.membrane[t])
            tifffile.imwrite(ca_file, self.calcium[t])
        
        # Save ground truth motion
        np.save(output_path / "motion_ground_truth.npy", self.motion_ground_truth)
        
        # Save config
        import json
        config_dict = {
            'n_frames': self.config.n_frames,
            'shape_zyx': self.config.shape_zyx,
            'motion_type': self.config.motion_type,
            'motion_amplitude': self.config.motion_amplitude,
        }
        with open(output_path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return output_path
    
    def save_as_zarr(self, output_path: str) -> Path:
        """Save generated data as Zarr array.
        
        Args:
            output_path: Path for Zarr store
            
        Returns:
            Path to output
        """
        import zarr
        
        if self.membrane is None:
            raise RuntimeError("Call generate() first")
        
        path = Path(output_path)
        
        root = zarr.open(str(path), mode='w')
        root.create_dataset('membrane', data=self.membrane, chunks=(1, -1, -1, -1))
        root.create_dataset('calcium', data=self.calcium, chunks=(1, -1, -1, -1))
        root.create_dataset('motion_ground_truth', data=self.motion_ground_truth)
        
        root.attrs['config'] = {
            'n_frames': self.config.n_frames,
            'shape_zyx': list(self.config.shape_zyx),
            'motion_type': self.config.motion_type,
        }
        
        return path


def generate_simple_test_data(
    n_frames: int = 20,
    shape: Tuple[int, int, int] = (5, 64, 64),
    motion_amplitude: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quick helper to generate simple test data.
    
    Args:
        n_frames: Number of frames
        shape: (Z, Y, X) dimensions
        motion_amplitude: Motion magnitude in pixels
        
    Returns:
        (membrane, calcium, motion_gt) arrays
        
    Example:
        >>> membrane, calcium, motion = generate_simple_test_data()
        >>> print(f"Generated: {membrane.shape}")
    """
    config = SyntheticDataConfig(
        n_frames=n_frames,
        shape_zyx=shape,
        n_spheres=10,
        motion_amplitude=motion_amplitude,
        motion_type='sine',
    )
    generator = SyntheticDataGenerator(config)
    return generator.generate()

