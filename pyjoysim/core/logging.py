"""
Structured logging system for PyJoySim.

Provides centralized logging configuration with support for multiple outputs,
structured formatting, and performance monitoring.
"""

import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured log messages.
    
    Supports both JSON and human-readable formats.
    """
    
    def __init__(self, fmt_type: str = "human", include_extra: bool = True):
        """
        Initialize structured formatter.
        
        Args:
            fmt_type: Format type ("human" or "json")
            include_extra: Include extra fields in output
        """
        self.fmt_type = fmt_type
        self.include_extra = include_extra
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured data."""
        # Basic log data
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add thread info if available
        if hasattr(record, 'thread') and record.thread:
            log_data["thread_id"] = record.thread
            log_data["thread_name"] = getattr(record, 'threadName', '')
        
        # Add process info
        log_data["process_id"] = record.process
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'exc_info', 'exc_text',
                    'stack_info', 'getMessage'
                }:
                    try:
                        # Ensure value is JSON serializable
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_data["extra"] = extra_fields
        
        # Format output
        if self.fmt_type == "json":
            return json.dumps(log_data, ensure_ascii=False)
        else:
            return self._format_human_readable(log_data)
    
    def _format_human_readable(self, log_data: Dict[str, Any]) -> str:
        """Format log data as human-readable string."""
        timestamp = log_data["timestamp"][:19]  # Remove microseconds
        level = log_data["level"]
        logger = log_data["logger"]
        message = log_data["message"]
        location = f"{log_data['module']}:{log_data['function']}:{log_data['line']}"
        
        # Color coding for different levels
        level_colors = {
            "DEBUG": "\033[36m",    # Cyan
            "INFO": "\033[32m",     # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",    # Red
            "CRITICAL": "\033[35m", # Magenta
        }
        reset_color = "\033[0m"
        
        # Check if we're outputting to a terminal
        use_colors = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()
        
        if use_colors and level in level_colors:
            level_str = f"{level_colors[level]}{level:<8}{reset_color}"
        else:
            level_str = f"{level:<8}"
        
        # Basic format
        formatted = f"{timestamp} {level_str} {logger:<20} {message}"
        
        # Add location in debug mode
        if level == "DEBUG":
            formatted += f" [{location}]"
        
        # Add exception info if present
        if "exception" in log_data:
            formatted += f"\n{log_data['exception']}"
        
        # Add extra fields if present
        if "extra" in log_data and log_data["extra"]:
            extra_str = ", ".join(f"{k}={v}" for k, v in log_data["extra"].items())
            formatted += f" | {extra_str}"
        
        return formatted


class PerformanceFilter(logging.Filter):
    """Filter that adds performance timing information to log records."""
    
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add timing information to log record."""
        record.elapsed_time = time.time() - self.start_time
        return True


class LoggerManager:
    """
    Centralized logger management for PyJoySim.
    
    Handles configuration of multiple loggers with different outputs and formats.
    """
    
    def __init__(self):
        self._loggers: Dict[str, logging.Logger] = {}
        self._configured = False
        self._log_dir: Optional[Path] = None
    
    def configure(self, 
                  log_level: str = "INFO",
                  log_dir: Optional[Union[str, Path]] = None,
                  console_output: bool = True,
                  file_output: bool = True,
                  json_format: bool = False,
                  max_file_size: int = 10 * 1024 * 1024,  # 10MB
                  backup_count: int = 5) -> None:
        """
        Configure the logging system.
        
        Args:
            log_level: Minimum log level to output
            log_dir: Directory for log files (creates if doesn't exist)
            console_output: Enable console output
            file_output: Enable file output
            json_format: Use JSON format for file output
            max_file_size: Maximum size of each log file
            backup_count: Number of backup files to keep
        """
        if self._configured:
            return
        
        # Set up log directory
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path.home() / ".pyjoysim" / "logs"
        
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add performance filter
        perf_filter = PerformanceFilter()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(StructuredFormatter("human"))
            console_handler.addFilter(perf_filter)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if file_output:
            # Main log file (rotating)
            file_handler = logging.handlers.RotatingFileHandler(
                self._log_dir / "pyjoysim.log",
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # File gets all messages
            
            if json_format:
                file_handler.setFormatter(StructuredFormatter("json"))
            else:
                file_handler.setFormatter(StructuredFormatter("human", include_extra=False))
            
            file_handler.addFilter(perf_filter)
            root_logger.addHandler(file_handler)
            
            # Error log file (errors and critical only)
            error_handler = logging.handlers.RotatingFileHandler(
                self._log_dir / "pyjoysim_errors.log",
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(StructuredFormatter("human"))
            error_handler.addFilter(perf_filter)
            root_logger.addHandler(error_handler)
        
        self._configured = True
        
        # Log configuration completion
        logger = self.get_logger("logging")
        logger.info("Logging system configured", extra={
            "log_level": log_level,
            "log_dir": str(self._log_dir),
            "console_output": console_output,
            "file_output": file_output,
            "json_format": json_format
        })
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Configured logger instance
        """
        if name not in self._loggers:
            logger = logging.getLogger(f"pyjoysim.{name}")
            self._loggers[name] = logger
        
        return self._loggers[name]
    
    def set_level(self, level: str, logger_name: Optional[str] = None) -> None:
        """
        Set log level for a specific logger or all loggers.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            logger_name: Specific logger name, or None for all
        """
        log_level = getattr(logging, level.upper())
        
        if logger_name:
            if logger_name in self._loggers:
                self._loggers[logger_name].setLevel(log_level)
        else:
            # Set level for root logger and all handlers
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(log_level)
    
    def shutdown(self) -> None:
        """Shutdown the logging system gracefully."""
        logger = self.get_logger("logging")
        logger.info("Shutting down logging system")
        logging.shutdown()
    
    @property
    def log_directory(self) -> Optional[Path]:
        """Get the log directory path."""
        return self._log_dir


# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None


def get_logger_manager() -> LoggerManager:
    """Get the global logger manager instance."""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    return _logger_manager


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module/component.
    
    Args:
        name: Logger name (typically module name)
        
    Returns:
        Configured logger instance
    """
    return get_logger_manager().get_logger(name)


def configure_logging(log_level: str = "INFO", **kwargs) -> None:
    """
    Configure the logging system with the given parameters.
    
    Args:
        log_level: Minimum log level
        **kwargs: Additional configuration options
    """
    # Import here to avoid circular imports
    from ..config import get_settings
    
    settings = get_settings()
    
    # Use settings for default values
    final_level = log_level or settings.log_level
    
    manager = get_logger_manager()
    manager.configure(
        log_level=final_level,
        console_output=kwargs.get('console_output', True),
        file_output=kwargs.get('file_output', not settings.is_development_mode()),
        json_format=kwargs.get('json_format', False),
        **kwargs
    )


def shutdown_logging() -> None:
    """Shutdown the logging system."""
    manager = get_logger_manager()
    manager.shutdown()