"""
Joystick management system for PyJoySim.

This module provides centralized joystick detection, initialization, and management
with support for multiple controllers and hot-plugging.
"""

import pygame
import time
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from enum import Enum

from ..config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import JoystickError, JoystickNotFoundError


class JoystickState(Enum):
    """Joystick connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class JoystickInfo:
    """Information about a joystick."""
    id: int
    name: str
    guid: str
    num_axes: int
    num_buttons: int
    num_hats: int
    state: JoystickState
    last_seen: float
    pygame_joystick: Optional[pygame.joystick.Joystick] = None


@dataclass
class JoystickInput:
    """Current input state from a joystick."""
    joystick_id: int
    axes: List[float]
    buttons: List[bool]
    hats: List[tuple[int, int]]
    timestamp: float
    
    def __post_init__(self):
        """Validate input data."""
        if not self.axes:
            self.axes = []
        if not self.buttons:
            self.buttons = []
        if not self.hats:
            self.hats = []


class JoystickManager:
    """
    Central manager for joystick detection, initialization, and input handling.
    
    Features:
    - Multi-joystick support (up to configured maximum)
    - Hot-plug detection (connect/disconnect during runtime)
    - Automatic initialization and cleanup
    - Event-based notifications
    - Performance monitoring
    """
    
    def __init__(self):
        """Initialize the joystick manager."""
        self.logger = get_logger("joystick_manager")
        self.settings = get_settings()
        
        # State tracking
        self._joysticks: Dict[int, JoystickInfo] = {}
        self._input_states: Dict[int, JoystickInput] = {}
        self._initialized = False
        self._pygame_initialized = False
        
        # Configuration
        self._max_joysticks = self.settings.max_joysticks
        self._deadzone = self.settings.joystick_deadzone
        self._hotplug_enabled = self.settings.hotplug_enabled
        
        # Event callbacks
        self._connect_callbacks: List[Callable[[int, JoystickInfo], None]] = []
        self._disconnect_callbacks: List[Callable[[int, JoystickInfo], None]] = []
        self._input_callbacks: List[Callable[[int, JoystickInput], None]] = []
        
        # Performance tracking
        self._last_scan_time = 0.0
        self._scan_interval = 1.0  # Scan for new joysticks every second
        self._input_poll_count = 0
        self._last_performance_log = 0.0
        
        self.logger.debug("JoystickManager initialized", extra={
            "max_joysticks": self._max_joysticks,
            "deadzone": self._deadzone,
            "hotplug_enabled": self._hotplug_enabled
        })
    
    def initialize(self) -> bool:
        """
        Initialize the joystick subsystem.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            self.logger.warning("JoystickManager already initialized")
            return True
        
        try:
            # Initialize pygame if not already done
            if not pygame.get_init():
                pygame.init()
                self._pygame_initialized = True
                self.logger.debug("Pygame initialized")
            
            # Initialize joystick subsystem
            if not pygame.joystick.get_init():
                pygame.joystick.init()
                self.logger.debug("Pygame joystick subsystem initialized")
            
            # Initial scan for joysticks
            self._scan_joysticks()
            
            self._initialized = True
            self.logger.info("JoystickManager initialization complete", extra={
                "joysticks_found": len(self._joysticks),
                "pygame_init": self._pygame_initialized
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize JoystickManager", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    def shutdown(self) -> None:
        """Shutdown the joystick subsystem and cleanup resources."""
        if not self._initialized:
            return
        
        self.logger.info("Shutting down JoystickManager")
        
        # Disconnect all joysticks
        for joystick_id in list(self._joysticks.keys()):
            self._disconnect_joystick(joystick_id)
        
        # Shutdown pygame subsystems if we initialized them
        try:
            if pygame.joystick.get_init():
                pygame.joystick.quit()
                self.logger.debug("Pygame joystick subsystem shutdown")
            
            if self._pygame_initialized and pygame.get_init():
                pygame.quit()
                self._pygame_initialized = False
                self.logger.debug("Pygame shutdown")
        except Exception as e:
            self.logger.warning("Error during pygame shutdown", extra={"error": str(e)})
        
        self._initialized = False
        self.logger.info("JoystickManager shutdown complete")
    
    def update(self) -> None:
        """
        Update joystick states and handle hot-plugging.
        
        Should be called regularly from the main loop.
        """
        if not self._initialized:
            return
        
        current_time = time.time()
        
        # Scan for new/disconnected joysticks periodically
        if self._hotplug_enabled and (current_time - self._last_scan_time) >= self._scan_interval:
            self._scan_joysticks()
            self._last_scan_time = current_time
        
        # Update input states for all connected joysticks
        self._update_input_states()
        
        # Performance logging
        self._input_poll_count += 1
        if (current_time - self._last_performance_log) >= 10.0:  # Every 10 seconds
            self._log_performance_stats()
            self._last_performance_log = current_time
    
    def _scan_joysticks(self) -> None:
        """Scan for connected joysticks and update the joystick list."""
        try:
            pygame.event.pump()  # Process pygame events
            joystick_count = pygame.joystick.get_count()
            current_time = time.time()
            
            # Track which joysticks are currently present
            present_joysticks: Set[int] = set()
            
            for i in range(min(joystick_count, self._max_joysticks)):
                present_joysticks.add(i)
                
                if i not in self._joysticks:
                    # New joystick detected
                    self._connect_joystick(i)
                else:
                    # Update last seen time
                    self._joysticks[i].last_seen = current_time
            
            # Check for disconnected joysticks
            disconnected = set(self._joysticks.keys()) - present_joysticks
            for joystick_id in disconnected:
                self._disconnect_joystick(joystick_id)
            
        except Exception as e:
            self.logger.error("Error scanning joysticks", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
    
    def _connect_joystick(self, joystick_id: int) -> None:
        """
        Connect and initialize a joystick.
        
        Args:
            joystick_id: ID of the joystick to connect
        """
        try:
            pygame_joystick = pygame.joystick.Joystick(joystick_id)
            pygame_joystick.init()
            
            # Get joystick information
            name = pygame_joystick.get_name()
            guid = getattr(pygame_joystick, 'get_guid', lambda: f"unknown_{joystick_id}")()
            num_axes = pygame_joystick.get_numaxes()
            num_buttons = pygame_joystick.get_numbuttons()
            num_hats = pygame_joystick.get_numhats()
            
            # Create joystick info
            joystick_info = JoystickInfo(
                id=joystick_id,
                name=name,
                guid=guid,
                num_axes=num_axes,
                num_buttons=num_buttons,
                num_hats=num_hats,
                state=JoystickState.CONNECTED,
                last_seen=time.time(),
                pygame_joystick=pygame_joystick
            )
            
            self._joysticks[joystick_id] = joystick_info
            
            # Initialize input state
            self._input_states[joystick_id] = JoystickInput(
                joystick_id=joystick_id,
                axes=[0.0] * num_axes,
                buttons=[False] * num_buttons,
                hats=[(0, 0)] * num_hats,
                timestamp=time.time()
            )
            
            self.logger.info("Joystick connected", extra={
                "joystick_id": joystick_id,
                "name": name,
                "guid": guid,
                "axes": num_axes,
                "buttons": num_buttons,
                "hats": num_hats
            })
            
            # Notify callbacks
            for callback in self._connect_callbacks:
                try:
                    callback(joystick_id, joystick_info)
                except Exception as e:
                    self.logger.error("Error in connect callback", extra={"error": str(e)})
            
        except Exception as e:
            self.logger.error("Failed to connect joystick", extra={
                "joystick_id": joystick_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Mark as error state
            if joystick_id in self._joysticks:
                self._joysticks[joystick_id].state = JoystickState.ERROR
    
    def _disconnect_joystick(self, joystick_id: int) -> None:
        """
        Disconnect and cleanup a joystick.
        
        Args:
            joystick_id: ID of the joystick to disconnect
        """
        if joystick_id not in self._joysticks:
            return
        
        joystick_info = self._joysticks[joystick_id]
        
        try:
            # Cleanup pygame joystick
            if joystick_info.pygame_joystick:
                joystick_info.pygame_joystick.quit()
            
            self.logger.info("Joystick disconnected", extra={
                "joystick_id": joystick_id,
                "name": joystick_info.name
            })
            
            # Notify callbacks before removing
            for callback in self._disconnect_callbacks:
                try:
                    callback(joystick_id, joystick_info)
                except Exception as e:
                    self.logger.error("Error in disconnect callback", extra={"error": str(e)})
            
        except Exception as e:
            self.logger.error("Error disconnecting joystick", extra={
                "joystick_id": joystick_id,
                "error": str(e)
            })
        finally:
            # Remove from tracking
            del self._joysticks[joystick_id]
            if joystick_id in self._input_states:
                del self._input_states[joystick_id]
    
    def _update_input_states(self) -> None:
        """Update input states for all connected joysticks."""
        current_time = time.time()
        
        for joystick_id, joystick_info in self._joysticks.items():
            if joystick_info.state != JoystickState.CONNECTED:
                continue
            
            try:
                pygame_joystick = joystick_info.pygame_joystick
                if not pygame_joystick:
                    continue
                
                # Read axes with deadzone applied
                axes = []
                for i in range(joystick_info.num_axes):
                    raw_value = pygame_joystick.get_axis(i)
                    # Apply deadzone
                    if abs(raw_value) < self._deadzone:
                        axes.append(0.0)
                    else:
                        # Scale to account for deadzone
                        if raw_value > 0:
                            axes.append((raw_value - self._deadzone) / (1.0 - self._deadzone))
                        else:
                            axes.append((raw_value + self._deadzone) / (1.0 - self._deadzone))
                
                # Read buttons
                buttons = [pygame_joystick.get_button(i) for i in range(joystick_info.num_buttons)]
                
                # Read hats (D-pad)
                hats = [pygame_joystick.get_hat(i) for i in range(joystick_info.num_hats)]
                
                # Create input state
                input_state = JoystickInput(
                    joystick_id=joystick_id,
                    axes=axes,
                    buttons=buttons,
                    hats=hats,
                    timestamp=current_time
                )
                
                # Store state
                self._input_states[joystick_id] = input_state
                
                # Notify input callbacks
                for callback in self._input_callbacks:
                    try:
                        callback(joystick_id, input_state)
                    except Exception as e:
                        self.logger.error("Error in input callback", extra={"error": str(e)})
                
            except Exception as e:
                self.logger.error("Error reading joystick input", extra={
                    "joystick_id": joystick_id,
                    "error": str(e)
                })
                joystick_info.state = JoystickState.ERROR
    
    def _log_performance_stats(self) -> None:
        """Log performance statistics."""
        poll_rate = self._input_poll_count / 10.0  # Polls per second over last 10 seconds
        self.logger.debug("Joystick performance stats", extra={
            "connected_joysticks": len(self._joysticks),
            "input_poll_rate": poll_rate,
            "deadzone": self._deadzone,
            "hotplug_enabled": self._hotplug_enabled
        })
        self._input_poll_count = 0
    
    # Public API methods
    
    def get_joystick_count(self) -> int:
        """Get the number of connected joysticks."""
        return len([j for j in self._joysticks.values() if j.state == JoystickState.CONNECTED])
    
    def get_joystick_info(self, joystick_id: int) -> Optional[JoystickInfo]:
        """
        Get information about a specific joystick.
        
        Args:
            joystick_id: ID of the joystick
            
        Returns:
            JoystickInfo if found, None otherwise
        """
        return self._joysticks.get(joystick_id)
    
    def get_all_joysticks(self) -> Dict[int, JoystickInfo]:
        """Get information about all joysticks."""
        return self._joysticks.copy()
    
    def get_input_state(self, joystick_id: int) -> Optional[JoystickInput]:
        """
        Get current input state for a joystick.
        
        Args:
            joystick_id: ID of the joystick
            
        Returns:
            JoystickInput if found, None otherwise
        """
        return self._input_states.get(joystick_id)
    
    def is_joystick_connected(self, joystick_id: int) -> bool:
        """
        Check if a joystick is connected.
        
        Args:
            joystick_id: ID of the joystick
            
        Returns:
            True if connected, False otherwise
        """
        joystick = self._joysticks.get(joystick_id)
        return joystick is not None and joystick.state == JoystickState.CONNECTED
    
    def set_deadzone(self, deadzone: float) -> None:
        """
        Set the analog stick deadzone.
        
        Args:
            deadzone: Deadzone value (0.0 - 1.0)
        """
        if not 0.0 <= deadzone <= 1.0:
            raise ValueError("Deadzone must be between 0.0 and 1.0")
        
        self._deadzone = deadzone
        self.logger.info("Deadzone updated", extra={"new_deadzone": deadzone})
    
    def enable_hotplug(self, enabled: bool) -> None:
        """
        Enable or disable hot-plug detection.
        
        Args:
            enabled: True to enable, False to disable
        """
        self._hotplug_enabled = enabled
        self.logger.info("Hotplug setting updated", extra={"enabled": enabled})
    
    # Event callback registration
    
    def add_connect_callback(self, callback: Callable[[int, JoystickInfo], None]) -> None:
        """Register a callback for joystick connect events."""
        self._connect_callbacks.append(callback)
    
    def add_disconnect_callback(self, callback: Callable[[int, JoystickInfo], None]) -> None:
        """Register a callback for joystick disconnect events."""
        self._disconnect_callbacks.append(callback)
    
    def add_input_callback(self, callback: Callable[[int, JoystickInput], None]) -> None:
        """Register a callback for joystick input events."""
        self._input_callbacks.append(callback)
    
    def remove_connect_callback(self, callback: Callable[[int, JoystickInfo], None]) -> None:
        """Remove a joystick connect callback."""
        if callback in self._connect_callbacks:
            self._connect_callbacks.remove(callback)
    
    def remove_disconnect_callback(self, callback: Callable[[int, JoystickInfo], None]) -> None:
        """Remove a joystick disconnect callback."""
        if callback in self._disconnect_callbacks:
            self._disconnect_callbacks.remove(callback)
    
    def remove_input_callback(self, callback: Callable[[int, JoystickInput], None]) -> None:
        """Remove a joystick input callback."""
        if callback in self._input_callbacks:
            self._input_callbacks.remove(callback)
    
    # Context manager support
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise JoystickError("Failed to initialize JoystickManager")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Global joystick manager instance
_joystick_manager: Optional[JoystickManager] = None


def get_joystick_manager() -> JoystickManager:
    """Get the global joystick manager instance."""
    global _joystick_manager
    if _joystick_manager is None:
        _joystick_manager = JoystickManager()
    return _joystick_manager


def reset_joystick_manager() -> None:
    """Reset the global joystick manager instance."""
    global _joystick_manager
    if _joystick_manager and _joystick_manager._initialized:
        _joystick_manager.shutdown()
    _joystick_manager = None