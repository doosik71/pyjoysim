"""
Exception handling framework for PyJoySim.

Provides custom exception classes and error handling utilities for
consistent error management throughout the application.
"""

import sys
import traceback
from typing import Any, Dict, Optional, Type, Union
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PyJoySimError(Exception):
    """
    Base exception class for all PyJoySim-specific errors.
    
    Provides structured error information including severity, error codes,
    and additional context data.
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        """
        Initialize PyJoySim error.
        
        Args:
            message: Human-readable error message
            error_code: Unique error identifier
            severity: Error severity level
            context: Additional context information
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.context = context or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
        }
    
    def __str__(self) -> str:
        """String representation of the error."""
        parts = [f"{self.error_code}: {self.message}"]
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        
        return " | ".join(parts)


# Configuration errors
class ConfigurationError(PyJoySimError):
    """Raised when there's a configuration-related error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class InvalidConfigValueError(ConfigurationError):
    """Raised when a configuration value is invalid."""
    
    def __init__(self, key: str, value: Any, expected: str, **kwargs):
        message = f"Invalid value '{value}' for config key '{key}'. Expected: {expected}"
        super().__init__(message, config_key=key, **kwargs)


# Input system errors
class InputError(PyJoySimError):
    """Base class for input-related errors."""
    pass


class JoystickError(InputError):
    """Raised when there's a joystick-related error."""
    
    def __init__(self, message: str, joystick_id: Optional[int] = None, **kwargs):
        context = kwargs.get('context', {})
        if joystick_id is not None:
            context['joystick_id'] = joystick_id
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class JoystickNotFoundError(JoystickError):
    """Raised when a joystick is not found or disconnected."""
    
    def __init__(self, joystick_id: int, **kwargs):
        message = f"Joystick {joystick_id} not found or disconnected"
        super().__init__(message, joystick_id=joystick_id, **kwargs)


class InputMappingError(InputError):
    """Raised when there's an error with input mapping configuration."""
    pass


# Physics errors
class PhysicsError(PyJoySimError):
    """Base class for physics-related errors."""
    pass


class PhysicsEngineError(PhysicsError):
    """Raised when there's a physics engine initialization or operation error."""
    pass


class CollisionError(PhysicsError):
    """Raised when there's a collision detection/response error."""
    pass


# Rendering errors
class RenderingError(PyJoySimError):
    """Base class for rendering-related errors."""
    pass


class ShaderError(RenderingError):
    """Raised when there's a shader compilation or linking error."""
    
    def __init__(self, message: str, shader_type: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if shader_type:
            context['shader_type'] = shader_type
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class TextureError(RenderingError):
    """Raised when there's a texture loading or processing error."""
    pass


class ModelError(RenderingError):
    """Raised when there's a model loading or processing error."""
    pass


# Simulation errors
class SimulationError(PyJoySimError):
    """Base class for simulation-related errors."""
    pass


class SimulationNotFoundError(SimulationError):
    """Raised when a requested simulation is not found."""
    
    def __init__(self, simulation_name: str, **kwargs):
        message = f"Simulation '{simulation_name}' not found"
        context = kwargs.get('context', {})
        context['simulation_name'] = simulation_name
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class SimulationStateError(SimulationError):
    """Raised when a simulation is in an invalid state for the requested operation."""
    pass


# Resource errors
class ResourceError(PyJoySimError):
    """Base class for resource-related errors."""
    pass


class FileNotFoundError(ResourceError):
    """Raised when a required file is not found."""
    
    def __init__(self, file_path: str, **kwargs):
        message = f"File not found: {file_path}"
        context = kwargs.get('context', {})
        context['file_path'] = file_path
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class AssetLoadingError(ResourceError):
    """Raised when there's an error loading game assets."""
    pass


# Performance errors
class PerformanceError(PyJoySimError):
    """Raised when performance thresholds are exceeded."""
    
    def __init__(self, message: str, metric: str, value: float, threshold: float, **kwargs):
        context = kwargs.get('context', {})
        context.update({
            'metric': metric,
            'value': value,
            'threshold': threshold
        })
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class ErrorHandler:
    """
    Centralized error handling and reporting system.
    
    Provides utilities for error logging, crash reporting, and recovery.
    """
    
    def __init__(self):
        self._error_callbacks = []
        self._crash_handlers = []
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle an error with appropriate logging and recovery actions.
        
        Args:
            error: The exception to handle
            context: Additional context information
        """
        # Import here to avoid circular imports
        from .logging import get_logger
        
        logger = get_logger("error_handler")
        
        # Create error context
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        
        if context:
            error_context.update(context)
        
        # Add PyJoySim-specific information for our custom errors
        if isinstance(error, PyJoySimError):
            error_context.update({
                "error_code": error.error_code,
                "severity": error.severity.value,
                "pyjoysim_context": error.context,
            })
            
            # Log with appropriate level based on severity
            if error.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.HIGH):
                logger.error("PyJoySim error occurred", extra=error_context)
            else:
                logger.warning("PyJoySim error occurred", extra=error_context)
        else:
            # Generic exception
            logger.error("Unexpected error occurred", extra=error_context)
        
        # Call registered error callbacks
        for callback in self._error_callbacks:
            try:
                callback(error, error_context)
            except Exception as callback_error:
                logger.error(f"Error in error callback: {callback_error}")
    
    def handle_crash(self, error: Exception, emergency_save: bool = True) -> None:
        """
        Handle a critical error that might cause the application to crash.
        
        Args:
            error: The critical exception
            emergency_save: Whether to attempt emergency save operations
        """
        from .logging import get_logger
        
        logger = get_logger("crash_handler")
        logger.critical("Critical error - application may crash", extra={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "emergency_save": emergency_save,
        })
        
        # Run crash handlers
        for handler in self._crash_handlers:
            try:
                handler(error, emergency_save)
            except Exception as handler_error:
                logger.critical(f"Error in crash handler: {handler_error}")
        
        # Attempt emergency save if requested
        if emergency_save:
            self._emergency_save()
    
    def _emergency_save(self) -> None:
        """Attempt to save critical application state before crash."""
        try:
            from ..config import get_settings
            settings = get_settings()
            settings.save_settings()
        except Exception as e:
            print(f"Emergency save failed: {e}", file=sys.stderr)
    
    def register_error_callback(self, callback) -> None:
        """Register a callback to be called when errors occur."""
        self._error_callbacks.append(callback)
    
    def register_crash_handler(self, handler) -> None:
        """Register a handler to be called during critical errors."""
        self._crash_handlers.append(handler)
    
    def setup_global_exception_handler(self) -> None:
        """Set up global exception handler for unhandled exceptions."""
        def exception_handler(exc_type: Type[BaseException], 
                            exc_value: BaseException, 
                            exc_traceback) -> None:
            """Global exception handler."""
            if issubclass(exc_type, KeyboardInterrupt):
                # Allow normal KeyboardInterrupt handling
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            # Handle as crash
            self.handle_crash(exc_value, emergency_save=True)
            
            # Call original exception handler for fallback
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = exception_handler


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Handle an error using the global error handler.
    
    Args:
        error: The exception to handle
        context: Additional context information
    """
    get_error_handler().handle_error(error, context)


def handle_crash(error: Exception, emergency_save: bool = True) -> None:
    """
    Handle a critical error using the global error handler.
    
    Args:
        error: The critical exception
        emergency_save: Whether to attempt emergency save
    """
    get_error_handler().handle_crash(error, emergency_save)


def safe_execute(func, *args, **kwargs) -> Any:
    """
    Execute a function safely with automatic error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or None if error occurred
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, context={
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs),
        })
        return None


def setup_exception_handling() -> None:
    """Set up global exception handling."""
    error_handler = get_error_handler()
    error_handler.setup_global_exception_handler()