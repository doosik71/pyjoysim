"""
Main entry point for PyJoySim application.

This module provides the main application entry point and initialization logic
for the joystick-based simulation platform.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from .config import get_config, get_settings
from .core.logging import configure_logging, get_logger, shutdown_logging
from .core.exceptions import setup_exception_handling, handle_error, handle_crash


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="pyjoysim",
        description="PyJoySim - Joystick-based Python simulation platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyjoysim                          # Start with default configuration
  pyjoysim --debug                  # Start in debug mode
  pyjoysim --config myconfig.json   # Use custom configuration
  pyjoysim --simulation car         # Start specific simulation
  pyjoysim --list-joysticks         # List available joysticks
  pyjoysim --version                # Show version information
        """
    )
    
    # Basic options
    parser.add_argument(
        "--version", 
        action="version", 
        version="PyJoySim 0.1.0"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with verbose logging"
    )
    
    parser.add_argument(
        "--config", 
        type=str,
        metavar="FILE",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set logging level (overrides config)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        metavar="FILE",
        help="Log to specific file instead of default location"
    )
    
    # Simulation options
    parser.add_argument(
        "--simulation",
        type=str,
        choices=["car", "robot", "drone", "spaceship", "submarine"],
        help="Start specific simulation directly"
    )
    
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start in fullscreen mode"
    )
    
    parser.add_argument(
        "--window-size",
        type=str,
        metavar="WIDTHxHEIGHT",
        help="Set window size (e.g., 1920x1080)"
    )
    
    # Development options
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio system"
    )
    
    parser.add_argument(
        "--performance-mode",
        action="store_true",
        help="Enable performance mode (lower quality, higher FPS)"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    # Utility options
    parser.add_argument(
        "--list-joysticks",
        action="store_true",
        help="List available joysticks and exit"
    )
    
    parser.add_argument(
        "--test-joystick",
        type=int,
        metavar="ID",
        help="Test specific joystick and exit"
    )
    
    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Reset configuration to defaults"
    )
    
    return parser


def parse_window_size(size_str: str) -> tuple[int, int]:
    """
    Parse window size string into width/height tuple.
    
    Args:
        size_str: Size string in format "WIDTHxHEIGHT"
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        ValueError: If format is invalid
    """
    try:
        width_str, height_str = size_str.lower().split('x')
        width = int(width_str)
        height = int(height_str)
        
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        
        return width, height
    except ValueError as e:
        raise ValueError(f"Invalid window size format '{size_str}'. Use WIDTHxHEIGHT (e.g., 1920x1080)") from e


def apply_command_line_overrides(args: argparse.Namespace) -> None:
    """
    Apply command line argument overrides to configuration.
    
    Args:
        args: Parsed command line arguments
    """
    config = get_config()
    
    # Debug mode
    if args.debug:
        config.set("app.debug", True)
        config.set("app.log_level", "DEBUG")
    
    # Log level override
    if args.log_level:
        config.set("app.log_level", args.log_level)
    
    # Window options
    if args.fullscreen:
        config.set("app.window.fullscreen", True)
    
    if args.window_size:
        try:
            width, height = parse_window_size(args.window_size)
            config.set("app.window.width", width)
            config.set("app.window.height", height)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Audio options
    if args.no_audio:
        config.set("audio.mute", True)
    
    # Performance options
    if args.performance_mode:
        config.set("simulation.performance_mode", True)


def list_joysticks() -> None:
    """List available joysticks and their information."""
    try:
        import pygame
        pygame.init()
        
        joystick_count = pygame.joystick.get_count()
        print(f"Found {joystick_count} joystick(s):")
        
        if joystick_count == 0:
            print("  No joysticks detected. Make sure your controller is connected.")
            return
        
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()
            
            print(f"  {i}: {joystick.get_name()}")
            print(f"      Axes: {joystick.get_numaxes()}")
            print(f"      Buttons: {joystick.get_numbuttons()}")
            print(f"      Hats: {joystick.get_numhats()}")
            
            if hasattr(joystick, 'get_guid'):
                print(f"      GUID: {joystick.get_guid()}")
            
            joystick.quit()
        
        pygame.quit()
    except ImportError:
        print("Error: pygame not available. Install with: pip install pygame")
        sys.exit(1)
    except Exception as e:
        print(f"Error listing joysticks: {e}")
        sys.exit(1)


def test_joystick(joystick_id: int) -> None:
    """
    Test a specific joystick by showing its inputs.
    
    Args:
        joystick_id: ID of joystick to test
    """
    try:
        import pygame
        pygame.init()
        
        joystick_count = pygame.joystick.get_count()
        if joystick_id >= joystick_count:
            print(f"Error: Joystick {joystick_id} not found. Available: 0-{joystick_count-1}")
            sys.exit(1)
        
        joystick = pygame.joystick.Joystick(joystick_id)
        joystick.init()
        
        print(f"Testing joystick {joystick_id}: {joystick.get_name()}")
        print("Press Ctrl+C to exit")
        print("-" * 50)
        
        clock = pygame.time.Clock()
        
        while True:
            pygame.event.pump()
            
            # Clear line and print current state
            print("\r" + " " * 80 + "\r", end="")
            
            # Axes
            axes = []
            for i in range(joystick.get_numaxes()):
                value = joystick.get_axis(i)
                axes.append(f"A{i}:{value:+.2f}")
            
            # Buttons
            buttons = []
            for i in range(joystick.get_numbuttons()):
                if joystick.get_button(i):
                    buttons.append(f"B{i}")
            
            # Hats
            hats = []
            for i in range(joystick.get_numhats()):
                hat = joystick.get_hat(i)
                if hat != (0, 0):
                    hats.append(f"H{i}:{hat}")
            
            status_parts = []
            if axes:
                status_parts.append(" ".join(axes))
            if buttons:
                status_parts.append("Buttons:" + ",".join(buttons))
            if hats:
                status_parts.append(" ".join(hats))
            
            status = " | ".join(status_parts) if status_parts else "No input"
            print(f"\r{status}", end="", flush=True)
            
            clock.tick(30)  # 30 FPS update
            
    except KeyboardInterrupt:
        print("\nJoystick test stopped.")
    except ImportError:
        print("Error: pygame not available. Install with: pip install pygame")
        sys.exit(1)
    except Exception as e:
        print(f"Error testing joystick: {e}")
        sys.exit(1)
    finally:
        if 'pygame' in locals():
            pygame.quit()


def initialize_application(args: argparse.Namespace) -> bool:
    """
    Initialize the PyJoySim application.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        # Load configuration
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"Error: Configuration file '{args.config}' not found")
                return False
            
            # Create new config with specified file
            from .config import Config, set_config
            config = Config(config_path)
            set_config(config)
        
        # Apply command line overrides
        apply_command_line_overrides(args)
        
        # Set up logging
        settings = get_settings()
        log_kwargs = {}
        if args.log_file:
            log_kwargs['log_dir'] = Path(args.log_file).parent
        
        configure_logging(
            log_level=settings.log_level,
            **log_kwargs
        )
        
        # Set up exception handling
        setup_exception_handling()
        
        logger = get_logger("main")
        logger.info("PyJoySim application starting", extra={
            "version": "0.1.0",
            "debug_mode": settings.debug_mode,
            "target_fps": settings.target_fps,
            "window_size": settings.window_size,
        })
        
        return True
        
    except Exception as e:
        print(f"Failed to initialize application: {e}")
        handle_error(e)
        return False


def run_application(args: argparse.Namespace) -> int:
    """
    Run the main application.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logger = get_logger("main")
    
    try:
        logger.info("Starting PyJoySim simulation platform")
        
        # For now, just show a placeholder message
        # In future phases, this will start the actual simulation system
        settings = get_settings()
        
        print(f"🎮 PyJoySim v0.1.0")
        print(f"Target FPS: {settings.target_fps}")
        print(f"Window Size: {settings.window_size}")
        print(f"Debug Mode: {settings.debug_mode}")
        
        if args.simulation:
            print(f"Starting simulation: {args.simulation}")
        else:
            print("No specific simulation requested - showing main menu (placeholder)")
        
        print("\nPress Ctrl+C to exit")
        print("=" * 50)
        print("NOTE: This is Phase 1 - basic initialization only.")
        print("Actual simulation features will be implemented in Phase 2.")
        print("=" * 50)
        
        # Placeholder main loop
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        
        logger.info("Application shutdown requested")
        return 0
        
    except Exception as e:
        logger.critical("Critical error in main application", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })
        handle_crash(e)
        return 1


def main() -> int:
    """
    Main entry point for PyJoySim application.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    try:
        # Handle utility commands first (no full initialization needed)
        if args.reset_config:
            print("Resetting configuration to defaults...")
            config = get_config()
            config.reset_to_defaults()
            config.save()
            print("Configuration reset complete.")
            return 0
        
        if args.list_joysticks:
            list_joysticks()
            return 0
        
        if args.test_joystick is not None:
            test_joystick(args.test_joystick)
            return 0
        
        # Initialize application
        if not initialize_application(args):
            return 1
        
        # Run main application
        return run_application(args)
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    finally:
        # Clean up
        try:
            shutdown_logging()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())