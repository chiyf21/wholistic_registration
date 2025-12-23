"""
Main registration pipeline runner.

Orchestrates the full registration workflow:
1. Read input data
2. Process middle block for initial reference
3. Process backward from middle to start
4. Process forward from middle to end
5. Save outputs as OME-TIFF

Uses a rolling reference window for temporal continuity.
"""

from typing import Optional, List, Callable, Any
from pathlib import Path
import numpy as np

from ..config import RegistrationConfig
from ..io import create_reader, OMETiffWriter, Metadata
from ..core import ReferenceComputer, FrameRegistrar
from ..utils.logging import (
    get_logger, 
    setup_logging, 
    progress_context, 
    CallbackManager,
)
from ..utils.array_ops import free_gpu_memory


class RegistrationPipeline:
    """Main registration pipeline.
    
    Processes a microscopy video through the registration pipeline:
    1. Establishes initial reference from middle of video
    2. Registers outward bidirectionally with rolling reference updates
    3. Saves registered frames as OME-TIFF
    
    Example:
        >>> config = RegistrationConfig.from_yaml("config.yaml")
        >>> pipeline = RegistrationPipeline(config)
        >>> pipeline.run()
        
    With callbacks:
        >>> def on_frame(event, data):
        ...     print(f"Processed frame {data['frame_idx']}")
        >>> 
        >>> pipeline = RegistrationPipeline(config)
        >>> pipeline.callbacks.register('on_frame_complete', on_frame)
        >>> pipeline.run()
    """
    
    def __init__(
        self,
        config: RegistrationConfig,
        log_file: Optional[str] = None,
    ):
        """Initialize pipeline.
        
        Args:
            config: Registration configuration
            log_file: Optional path to write logs
        """
        self.config = config
        self.callbacks = CallbackManager()
        
        # Setup logging
        self.logger = setup_logging(log_file=log_file)
        self.logger.info("Initializing registration pipeline")
        self.logger.info(f"Input: {config.input_path}")
        self.logger.info(f"Output: {config.output_dir}")
        self.logger.info(f"Backend: {config.backend.device}")
        
        # Create reader
        self.reader = create_reader(config.input_path)
        self.metadata = self.reader.metadata
        
        # Store in config for later use
        config._metadata = self.metadata.to_dict()
        
        # Log data info
        self.logger.info(f"Data: {self.metadata}")
        
        # Create writer
        self.writer = OMETiffWriter(
            output_dir=config.output_dir,
            metadata=self.metadata,
            compression=config.output.compression,
        )
        
        # Create core components
        self.reference_computer = ReferenceComputer(
            max_correlation_frames=config.reference.max_correlation_frames,
            device=config.backend.device,
            gpu_id=config.backend.gpu_id,
        )
        
        self.registrar = FrameRegistrar(
            pyramid=config.pyramid,
            mask=config.mask,
            channels=config.channels,
            device=config.backend.device,
            gpu_id=config.backend.gpu_id,
        )
        
        # Save config
        config.save_yaml(Path(config.output_dir) / "config.yaml")
    
    def run(self) -> None:
        """Run the full registration pipeline."""
        self.callbacks.emit('on_start', {'config': self.config})
        
        try:
            self._run_pipeline()
            self.callbacks.emit('on_complete', {'status': 'success'})
            self.logger.info("Registration completed successfully")
        except Exception as e:
            self.callbacks.emit('on_error', {'error': str(e)})
            self.logger.error(f"Registration failed: {e}")
            raise
        finally:
            self.reader.close()
    
    def _run_pipeline(self) -> None:
        """Internal pipeline implementation."""
        cfg = self.config
        total_frames = self.metadata.n_frames
        t_chunk = cfg.downsample.t_chunk
        window_size = cfg.reference.window_size
        
        # Step 1: Process middle block
        self.logger.info("=" * 60)
        self.logger.info("Step 1: Processing middle block for initial reference")
        self.logger.info("=" * 60)
        
        mid_mem, mid_ca, mid_ref = self._process_middle_block()
        
        # Step 2: Process backward
        self.logger.info("=" * 60)
        self.logger.info("Step 2: Processing backward from middle to start")
        self.logger.info("=" * 60)
        
        mid_start = self._get_middle_start()
        self._process_backward(mid_mem, mid_ca, mid_start)
        
        # Step 3: Process forward
        self.logger.info("=" * 60)
        self.logger.info("Step 3: Processing forward from middle to end")
        self.logger.info("=" * 60)
        
        mid_end = self._get_middle_end()
        self._process_forward(mid_mem, mid_ca, mid_end)
        
        self.logger.info("=" * 60)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 60)
    
    def _get_middle_start(self) -> int:
        """Get starting frame of middle block."""
        total = self.metadata.n_frames
        init_frames = self.config.reference.initial_frames
        return (total // 2) - (init_frames // 2)
    
    def _get_middle_end(self) -> int:
        """Get ending frame of middle block (exclusive)."""
        return self._get_middle_start() + self.config.reference.initial_frames
    
    def _process_middle_block(self):
        """Process the middle block to establish initial reference.
        
        Returns:
            Tuple of (registered_membrane, registered_calcium, reference)
        """
        cfg = self.config
        mid_start = self._get_middle_start()
        mid_end = self._get_middle_end()
        frames = list(range(mid_start, mid_end))
        
        self.logger.info(f"Loading middle block: frames {mid_start} to {mid_end-1}")
        
        # Read middle frames
        membrane = self.reader.read_frames(
            frames,
            channel=cfg.channels.membrane_channel,
            z_slices=cfg.downsample.z_slices,
            xy_downsample=cfg.downsample.xy,
        )
        calcium = self.reader.read_frames(
            frames,
            channel=cfg.channels.calcium_channel,
            z_slices=cfg.downsample.z_slices,
            xy_downsample=cfg.downsample.xy,
        )
        
        self.logger.info(f"Loaded {len(frames)} frames, shape: {membrane.shape}")
        
        # Compute initial reference
        self.logger.info("Computing initial reference from middle block...")
        reference = self.reference_computer.compute(
            membrane, calcium, cfg.channels
        )
        self.logger.info(f"Reference shape: {reference.shape}")
        
        # Register middle block
        self.logger.info("Registering middle block...")
        with progress_context("Registering middle block", len(frames)) as advance:
            result = self.registrar.register_batch(
                membrane, calcium, reference,
                return_motion=cfg.output.save_motion,
                verbose=False,
                start_frame_idx=mid_start,
            )
            advance(len(frames))
        
        # Save middle block
        self.logger.info("Saving middle block results...")
        self.writer.write_batch(
            result.membrane_registered,
            result.calcium_registered,
            frames,
            reference=reference if cfg.output.save_reference else None,
            motion_batch=result.motion_fields,
        )
        
        # Emit callbacks
        for i, frame_idx in enumerate(frames):
            self.callbacks.emit('on_frame_complete', {
                'frame_idx': frame_idx,
                'error': result.errors[i],
            })
        
        self.callbacks.emit('on_chunk_complete', {
            'chunk': 'middle',
            'frames': frames,
        })
        
        return result.membrane_registered, result.calcium_registered, reference
    
    def _process_backward(
        self,
        mid_membrane: np.ndarray,
        mid_calcium: np.ndarray,
        mid_start: int,
    ) -> None:
        """Process frames backward from middle block to start.
        
        Uses a rolling window of recently-registered frames to compute
        the reference for each chunk.
        """
        cfg = self.config
        t_chunk = cfg.downsample.t_chunk
        window_size = cfg.reference.window_size
        
        # Initialize rolling window from start of middle block
        # (temporally closest to what we'll process next going backward)
        ref_window_mem = mid_membrane[:window_size].copy()
        ref_window_ca = mid_calcium[:window_size].copy()
        
        # Process chunks going backward
        # Chunks: [mid_start-t_chunk, mid_start), [mid_start-2*t_chunk, mid_start-t_chunk), ...
        chunks_to_process = []
        idx = mid_start
        while idx > 0:
            end_idx = idx
            start_idx = max(0, idx - t_chunk)
            chunks_to_process.append((start_idx, end_idx))
            idx = start_idx
        
        if not chunks_to_process:
            self.logger.info("No backward chunks to process")
            return
        
        total_backward_frames = sum(e - s for s, e in chunks_to_process)
        self.logger.info(
            f"Processing {len(chunks_to_process)} chunks backward, "
            f"{total_backward_frames} frames total"
        )
        
        with progress_context("Backward registration", total_backward_frames) as advance:
            for start_idx, end_idx in chunks_to_process:
                frames = list(range(start_idx, end_idx))
                
                if not frames:
                    continue
                
                self.logger.debug(f"Processing frames {start_idx} to {end_idx-1}")
                
                # Read frames
                membrane = self.reader.read_frames(
                    frames,
                    channel=cfg.channels.membrane_channel,
                    z_slices=cfg.downsample.z_slices,
                    xy_downsample=cfg.downsample.xy,
                )
                calcium = self.reader.read_frames(
                    frames,
                    channel=cfg.channels.calcium_channel,
                    z_slices=cfg.downsample.z_slices,
                    xy_downsample=cfg.downsample.xy,
                )
                
                # Compute reference from window
                reference = self.reference_computer.compute(
                    ref_window_mem, ref_window_ca, cfg.channels
                )
                
                self.callbacks.emit('on_reference_update', {
                    'chunk_start': start_idx,
                    'chunk_end': end_idx,
                })
                
                # Register
                result = self.registrar.register_batch(
                    membrane, calcium, reference,
                    return_motion=cfg.output.save_motion,
                    start_frame_idx=start_idx,
                )
                
                # Save
                self.writer.write_batch(
                    result.membrane_registered,
                    result.calcium_registered,
                    frames,
                    reference=reference if cfg.output.save_reference else None,
                    motion_batch=result.motion_fields,
                )
                
                # Update window with first window_size frames from this chunk
                # (these are temporally closest to next chunk going backward)
                n_to_take = min(window_size, len(frames))
                ref_window_mem = result.membrane_registered[:n_to_take].copy()
                ref_window_ca = result.calcium_registered[:n_to_take].copy()
                
                # Pad if needed
                if n_to_take < window_size:
                    # Keep some from previous window
                    keep = window_size - n_to_take
                    ref_window_mem = np.concatenate([
                        ref_window_mem,
                        ref_window_mem[-1:].repeat(keep, axis=0)
                    ], axis=0)
                    ref_window_ca = np.concatenate([
                        ref_window_ca,
                        ref_window_ca[-1:].repeat(keep, axis=0)
                    ], axis=0)
                
                advance(len(frames))
                
                # Emit callbacks
                for i, frame_idx in enumerate(frames):
                    self.callbacks.emit('on_frame_complete', {
                        'frame_idx': frame_idx,
                        'error': result.errors[i],
                    })
                
                self.callbacks.emit('on_chunk_complete', {
                    'chunk': 'backward',
                    'frames': frames,
                })
                
                # Free GPU memory
                free_gpu_memory()
    
    def _process_forward(
        self,
        mid_membrane: np.ndarray,
        mid_calcium: np.ndarray,
        mid_end: int,
    ) -> None:
        """Process frames forward from middle block to end.
        
        Uses a rolling window of recently-registered frames to compute
        the reference for each chunk.
        """
        cfg = self.config
        t_chunk = cfg.downsample.t_chunk
        window_size = cfg.reference.window_size
        total_frames = self.metadata.n_frames
        
        # Initialize rolling window from end of middle block
        # (temporally closest to what we'll process next going forward)
        ref_window_mem = mid_membrane[-window_size:].copy()
        ref_window_ca = mid_calcium[-window_size:].copy()
        
        # Process chunks going forward
        chunks_to_process = []
        idx = mid_end
        while idx < total_frames:
            start_idx = idx
            end_idx = min(idx + t_chunk, total_frames)
            chunks_to_process.append((start_idx, end_idx))
            idx = end_idx
        
        if not chunks_to_process:
            self.logger.info("No forward chunks to process")
            return
        
        total_forward_frames = sum(e - s for s, e in chunks_to_process)
        self.logger.info(
            f"Processing {len(chunks_to_process)} chunks forward, "
            f"{total_forward_frames} frames total"
        )
        
        with progress_context("Forward registration", total_forward_frames) as advance:
            for start_idx, end_idx in chunks_to_process:
                frames = list(range(start_idx, end_idx))
                
                if not frames:
                    continue
                
                self.logger.debug(f"Processing frames {start_idx} to {end_idx-1}")
                
                # Read frames
                membrane = self.reader.read_frames(
                    frames,
                    channel=cfg.channels.membrane_channel,
                    z_slices=cfg.downsample.z_slices,
                    xy_downsample=cfg.downsample.xy,
                )
                calcium = self.reader.read_frames(
                    frames,
                    channel=cfg.channels.calcium_channel,
                    z_slices=cfg.downsample.z_slices,
                    xy_downsample=cfg.downsample.xy,
                )
                
                # Compute reference from window
                reference = self.reference_computer.compute(
                    ref_window_mem, ref_window_ca, cfg.channels
                )
                
                self.callbacks.emit('on_reference_update', {
                    'chunk_start': start_idx,
                    'chunk_end': end_idx,
                })
                
                # Register
                result = self.registrar.register_batch(
                    membrane, calcium, reference,
                    return_motion=cfg.output.save_motion,
                    start_frame_idx=start_idx,
                )
                
                # Save
                self.writer.write_batch(
                    result.membrane_registered,
                    result.calcium_registered,
                    frames,
                    reference=reference if cfg.output.save_reference else None,
                    motion_batch=result.motion_fields,
                )
                
                # Update window with last window_size frames from this chunk
                # (these are temporally closest to next chunk going forward)
                n_to_take = min(window_size, len(frames))
                ref_window_mem = result.membrane_registered[-n_to_take:].copy()
                ref_window_ca = result.calcium_registered[-n_to_take:].copy()
                
                # Pad if needed
                if n_to_take < window_size:
                    keep = window_size - n_to_take
                    ref_window_mem = np.concatenate([
                        ref_window_mem[0:1].repeat(keep, axis=0),
                        ref_window_mem,
                    ], axis=0)
                    ref_window_ca = np.concatenate([
                        ref_window_ca[0:1].repeat(keep, axis=0),
                        ref_window_ca,
                    ], axis=0)
                
                advance(len(frames))
                
                # Emit callbacks
                for i, frame_idx in enumerate(frames):
                    self.callbacks.emit('on_frame_complete', {
                        'frame_idx': frame_idx,
                        'error': result.errors[i],
                    })
                
                self.callbacks.emit('on_chunk_complete', {
                    'chunk': 'forward',
                    'frames': frames,
                })
                
                # Free GPU memory
                free_gpu_memory()

