"""
Logging utilities with Rich progress bar support.

Provides a unified logging interface with:
- Standard Python logging
- Rich progress bars for long operations
- Callback hooks for integration with external tools
"""

import logging
import sys
from typing import Optional, Callable, Any, List
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.progress import (
        Progress, 
        SpinnerColumn, 
        TextColumn, 
        BarColumn, 
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
    )
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Module-level logger
_logger: Optional[logging.Logger] = None
_console: Optional["Console"] = None


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """Set up logging for the registration pipeline.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Optional file path to write logs
        use_rich: Use Rich for pretty console output (if available)
        
    Returns:
        Configured logger instance
    """
    global _logger, _console
    
    logger = logging.getLogger("wholistic_registration.v2")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    if use_rich and RICH_AVAILABLE:
        _console = Console()
        console_handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        )
    
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the current logger, creating one if necessary."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


@contextmanager
def progress_context(
    description: str,
    total: int,
    disable: bool = False,
):
    """Context manager for progress tracking.
    
    Uses Rich progress bar if available, otherwise prints simple updates.
    
    Args:
        description: Description shown next to progress bar
        total: Total number of steps
        disable: If True, disables progress output
        
    Yields:
        A callable that advances progress by 1 step
        
    Example:
        >>> with progress_context("Processing frames", total=100) as advance:
        ...     for frame in frames:
        ...         process(frame)
        ...         advance()
    """
    if disable:
        yield lambda: None
        return
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_console or Console(),
        ) as progress:
            task = progress.add_task(description, total=total)
            
            def advance(n: int = 1):
                progress.update(task, advance=n)
            
            yield advance
    else:
        # Simple fallback without Rich
        current = [0]
        
        def advance(n: int = 1):
            current[0] += n
            if current[0] % max(1, total // 20) == 0 or current[0] == total:
                pct = 100 * current[0] / total
                print(f"\r{description}: {current[0]}/{total} ({pct:.1f}%)", end="", flush=True)
                if current[0] == total:
                    print()  # Newline at end
        
        yield advance


@dataclass
class CallbackManager:
    """Manages callbacks for pipeline events.
    
    Callbacks allow external tools to hook into the registration pipeline
    for monitoring, visualization, or custom processing.
    
    Supported events:
        - on_start: Called when pipeline starts
        - on_frame_complete: Called after each frame is registered
        - on_chunk_complete: Called after each chunk is registered
        - on_error: Called when an error occurs
        - on_complete: Called when pipeline finishes
    
    Example:
        >>> def my_callback(event: str, data: dict):
        ...     print(f"Event: {event}, Frame: {data.get('frame_id')}")
        >>> 
        >>> callbacks = CallbackManager()
        >>> callbacks.register('on_frame_complete', my_callback)
    """
    
    _callbacks: dict = field(default_factory=lambda: {
        'on_start': [],
        'on_frame_complete': [],
        'on_chunk_complete': [],
        'on_reference_update': [],
        'on_error': [],
        'on_complete': [],
    })
    
    def register(self, event: str, callback: Callable[[str, dict], Any]) -> None:
        """Register a callback for an event.
        
        Args:
            event: Event name (e.g., 'on_frame_complete')
            callback: Function taking (event_name, data_dict)
        """
        if event not in self._callbacks:
            raise ValueError(f"Unknown event: {event}. Valid: {list(self._callbacks.keys())}")
        self._callbacks[event].append(callback)
    
    def emit(self, event: str, data: dict) -> None:
        """Emit an event to all registered callbacks.
        
        Args:
            event: Event name
            data: Event data dictionary
        """
        logger = get_logger()
        for callback in self._callbacks.get(event, []):
            try:
                callback(event, data)
            except Exception as e:
                logger.warning(f"Callback error for {event}: {e}")
    
    def clear(self, event: Optional[str] = None) -> None:
        """Clear callbacks.
        
        Args:
            event: If specified, clear only callbacks for this event.
                   If None, clear all callbacks.
        """
        if event:
            self._callbacks[event] = []
        else:
            for key in self._callbacks:
                self._callbacks[key] = []

