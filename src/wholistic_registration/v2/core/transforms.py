"""
Channel transformations for dual-channel registration.

When using dual-channel mode, the calcium channel is transformed before
combining with the membrane channel:
    combined = membrane + k * transform(calcium)

Available transforms:
    - log10: log₁₀(1 + x) - compresses dynamic range significantly
    - sqrt: √x - mild compression
    - log2: log₂(1 + x) - similar to log10
    - raw: x - no transformation
"""

from typing import Literal, Union
import numpy as np

from ..utils.array_ops import ArrayLike

# Type alias for transform names
TransformType = Literal['log10', 'sqrt', 'log2', 'raw']


def apply_channel_transform(
    data: ArrayLike,
    transform: TransformType,
    k: float = 1.0,
) -> ArrayLike:
    """Apply transformation to channel data.
    
    Args:
        data: Input array (any shape)
        transform: Transform type ('log10', 'sqrt', 'log2', 'raw')
        k: Scaling factor applied after transformation
        
    Returns:
        Transformed array: k * transform(data)
        
    Note:
        - Works with both NumPy and CuPy arrays
        - Preserves array type (GPU arrays stay on GPU)
        
    Example:
        >>> import numpy as np
        >>> data = np.random.rand(10, 100, 100) * 1000
        >>> transformed = apply_channel_transform(data, 'log10', k=50)
    """
    # Get the array module (numpy or cupy)
    xp = np if isinstance(data, np.ndarray) else data.__class__.__module__
    if xp != np:
        import cupy as cp
        xp = cp
    else:
        xp = np
    
    if transform == 'raw':
        return k * data
    elif transform == 'sqrt':
        return k * xp.sqrt(data)
    elif transform == 'log2':
        return k * xp.log2(1 + data)
    elif transform == 'log10':
        return k * xp.log10(1 + data)
    else:
        raise ValueError(
            f"Unknown transform: '{transform}'. "
            f"Valid options: 'log10', 'sqrt', 'log2', 'raw'"
        )


def combine_channels(
    membrane: ArrayLike,
    calcium: ArrayLike,
    transform: TransformType,
    k: float,
) -> ArrayLike:
    """Combine membrane and transformed calcium channels.
    
    Formula: combined = membrane + k * transform(calcium)
    
    Args:
        membrane: Membrane channel data
        calcium: Calcium channel data
        transform: Transform to apply to calcium
        k: Weight for calcium contribution
        
    Returns:
        Combined channel data
    """
    calcium_transformed = apply_channel_transform(calcium, transform, k)
    return membrane + calcium_transformed

