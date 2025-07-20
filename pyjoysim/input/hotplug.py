"""
Hot-plug support for joystick detection and management.

This module provides enhanced hot-plug detection capabilities for
runtime joystick connect/disconnect events.
"""

import time
import threading
from typing import Dict, Set, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

import pygame

from .joystick_manager import JoystickManager, JoystickInfo, JoystickState
from ..config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import JoystickError


class HotplugEventType(Enum):
    """Types of hotplug events."""
    DEVICE_ADDED = "device_added"
    DEVICE_REMOVED = "device_removed"
    DEVICE_CHANGED = "device_changed"


@dataclass
class HotplugEvent:
    """Represents a hotplug event."""
    event_type: HotplugEventType
    joystick_id: int
    device_info: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class HotplugDetector:
    """
    Enhanced hotplug detection for joystick devices.
    
    Provides more responsive detection than the basic scanning in JoystickManager
    by monitoring pygame events and system-level device changes.
    """
    
    def __init__(self, joystick_manager: JoystickManager):
        """
        Initialize the hotplug detector.
        
        Args:
            joystick_manager: Reference to the main joystick manager
        """
        self.logger = get_logger("hotplug_detector")
        self.settings = get_settings()
        self.joystick_manager = joystick_manager
        
        # State tracking
        self._running = False
        self._detection_thread: Optional[threading.Thread] = None
        self._known_devices: Set[int] = set()
        self._last_device_count = 0
        
        # Event callbacks
        self._hotplug_callbacks: List[Callable[[HotplugEvent], None]] = []
        
        # Configuration
        self._detection_interval = 0.5  # Check every 500ms for fast response
        self._event_debounce_time = 0.2  # Debounce events to prevent spam
        self._last_event_time: Dict[int, float] = {}
        
        self.logger.debug("HotplugDetector initialized")
    
    def start(self) -> bool:
        """
        Start hotplug detection.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            self.logger.warning("HotplugDetector already running")
            return True
        
        try:
            # Initialize known devices
            self._scan_initial_devices()
            
            # Start detection thread
            self._running = True
            self._detection_thread = threading.Thread(
                target=self._detection_loop,
                name="HotplugDetector",
                daemon=True
            )
            self._detection_thread.start()
            
            self.logger.info("HotplugDetector started", extra={
                "initial_devices": len(self._known_devices),
                "detection_interval": self._detection_interval
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to start HotplugDetector", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            self._running = False
            return False
    
    def stop(self) -> None:
        """Stop hotplug detection."""
        if not self._running:
            return
        
        self.logger.info("Stopping HotplugDetector")
        self._running = False
        
        # Wait for detection thread to finish
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=2.0)
            if self._detection_thread.is_alive():
                self.logger.warning("Detection thread did not stop gracefully")
        
        self.logger.info("HotplugDetector stopped")
    
    def _scan_initial_devices(self) -> None:
        """Scan for initially connected devices."""
        try:
            pygame.event.pump()
            device_count = pygame.joystick.get_count()
            
            for i in range(device_count):
                self._known_devices.add(i)
            
            self._last_device_count = device_count
            
            self.logger.debug("Initial device scan complete", extra={
                "device_count": device_count,
                "known_devices": list(self._known_devices)
            })
            
        except Exception as e:
            self.logger.error("Error scanning initial devices", extra={
                "error": str(e)
            })
    
    def _detection_loop(self) -> None:
        """Main detection loop running in separate thread."""
        self.logger.debug("HotplugDetector detection loop started")
        
        while self._running:
            try:
                self._check_device_changes()
                self._process_pygame_events()
                time.sleep(self._detection_interval)
                
            except Exception as e:
                self.logger.error("Error in detection loop", extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                # Continue running unless explicitly stopped
                time.sleep(1.0)
        
        self.logger.debug("HotplugDetector detection loop ended")
    
    def _check_device_changes(self) -> None:
        """Check for device count changes."""
        try:
            pygame.event.pump()
            current_count = pygame.joystick.get_count()
            
            if current_count != self._last_device_count:
                self.logger.debug("Device count changed", extra={
                    "previous": self._last_device_count,
                    "current": current_count
                })
                
                # Determine what changed
                current_devices = set(range(current_count))
                
                # Check for new devices
                new_devices = current_devices - self._known_devices
                for device_id in new_devices:
                    self._handle_device_added(device_id)
                
                # Check for removed devices
                removed_devices = self._known_devices - current_devices
                for device_id in removed_devices:
                    self._handle_device_removed(device_id)
                
                # Update tracking
                self._known_devices = current_devices
                self._last_device_count = current_count
            
        except Exception as e:
            self.logger.error("Error checking device changes", extra={
                "error": str(e)
            })
    
    def _process_pygame_events(self) -> None:
        """Process pygame events for additional device information."""
        try:
            # Process pygame events that might contain device info
            for event in pygame.event.get([pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED]):
                if event.type == pygame.JOYDEVICEADDED:
                    device_id = getattr(event, 'device_index', None)
                    if device_id is not None:
                        self._handle_device_added(device_id, event_data=event.__dict__)
                
                elif event.type == pygame.JOYDEVICEREMOVED:
                    device_id = getattr(event, 'instance_id', None)
                    if device_id is not None:
                        self._handle_device_removed(device_id, event_data=event.__dict__)
            
        except Exception as e:
            self.logger.error("Error processing pygame events", extra={
                "error": str(e)
            })
    
    def _handle_device_added(self, device_id: int, event_data: Optional[Dict] = None) -> None:
        """
        Handle device addition.
        
        Args:
            device_id: ID of the added device
            event_data: Optional pygame event data
        """
        current_time = time.time()
        
        # Debounce events
        if (device_id in self._last_event_time and 
            current_time - self._last_event_time[device_id] < self._event_debounce_time):
            return
        
        self._last_event_time[device_id] = current_time
        
        try:
            # Get device information
            device_info = {}
            if event_data:
                device_info.update(event_data)
            
            # Try to get additional info from pygame
            try:
                if device_id < pygame.joystick.get_count():
                    temp_joystick = pygame.joystick.Joystick(device_id)
                    device_info.update({
                        "name": temp_joystick.get_name(),
                        "guid": getattr(temp_joystick, 'get_guid', lambda: f"unknown_{device_id}")(),
                        "num_axes": temp_joystick.get_numaxes(),
                        "num_buttons": temp_joystick.get_numbuttons(),
                        "num_hats": temp_joystick.get_numhats()
                    })
            except Exception:
                pass  # Continue without detailed info
            
            # Create hotplug event
            hotplug_event = HotplugEvent(
                event_type=HotplugEventType.DEVICE_ADDED,
                joystick_id=device_id,
                device_info=device_info,
                timestamp=current_time
            )
            
            self.logger.info("Device added", extra={
                "device_id": device_id,
                "device_info": device_info
            })
            
            # Notify callbacks
            self._notify_callbacks(hotplug_event)
            
            # Update known devices
            self._known_devices.add(device_id)
            
        except Exception as e:
            self.logger.error("Error handling device addition", extra={
                "device_id": device_id,
                "error": str(e)
            })
    
    def _handle_device_removed(self, device_id: int, event_data: Optional[Dict] = None) -> None:
        """
        Handle device removal.
        
        Args:
            device_id: ID of the removed device
            event_data: Optional pygame event data
        """
        current_time = time.time()
        
        # Debounce events
        if (device_id in self._last_event_time and 
            current_time - self._last_event_time[device_id] < self._event_debounce_time):
            return
        
        self._last_event_time[device_id] = current_time
        
        try:
            # Get device information if available
            device_info = {}
            if event_data:
                device_info.update(event_data)
            
            # Create hotplug event
            hotplug_event = HotplugEvent(
                event_type=HotplugEventType.DEVICE_REMOVED,
                joystick_id=device_id,
                device_info=device_info,
                timestamp=current_time
            )
            
            self.logger.info("Device removed", extra={
                "device_id": device_id,
                "device_info": device_info
            })
            
            # Notify callbacks
            self._notify_callbacks(hotplug_event)
            
            # Update known devices
            self._known_devices.discard(device_id)
            
            # Clean up event tracking
            if device_id in self._last_event_time:
                del self._last_event_time[device_id]
            
        except Exception as e:
            self.logger.error("Error handling device removal", extra={
                "device_id": device_id,
                "error": str(e)
            })
    
    def _notify_callbacks(self, event: HotplugEvent) -> None:
        """Notify all registered callbacks of a hotplug event."""
        for callback in self._hotplug_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error("Error in hotplug callback", extra={
                    "error": str(e),
                    "event_type": event.event_type.value,
                    "device_id": event.joystick_id
                })
    
    def add_callback(self, callback: Callable[[HotplugEvent], None]) -> None:
        """
        Add a callback for hotplug events.
        
        Args:
            callback: Function to call when hotplug events occur
        """
        self._hotplug_callbacks.append(callback)
        self.logger.debug("Hotplug callback added", extra={
            "total_callbacks": len(self._hotplug_callbacks)
        })
    
    def remove_callback(self, callback: Callable[[HotplugEvent], None]) -> None:
        """
        Remove a hotplug callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self._hotplug_callbacks:
            self._hotplug_callbacks.remove(callback)
            self.logger.debug("Hotplug callback removed", extra={
                "total_callbacks": len(self._hotplug_callbacks)
            })
    
    def force_scan(self) -> None:
        """Force an immediate scan for device changes."""
        if self._running:
            try:
                self._check_device_changes()
                self.logger.debug("Force scan completed")
            except Exception as e:
                self.logger.error("Error in force scan", extra={
                    "error": str(e)
                })
        else:
            self.logger.warning("Cannot force scan - detector not running")
    
    def get_known_devices(self) -> Set[int]:
        """Get the set of currently known device IDs."""
        return self._known_devices.copy()
    
    def is_running(self) -> bool:
        """Check if the detector is currently running."""
        return self._running


class IntegratedHotplugManager:
    """
    Manager that integrates hotplug detection with the main joystick manager.
    
    This class provides a seamless integration between hotplug detection
    and the main joystick management system.
    """
    
    def __init__(self, joystick_manager: JoystickManager):
        """
        Initialize the integrated hotplug manager.
        
        Args:
            joystick_manager: Main joystick manager instance
        """
        self.logger = get_logger("integrated_hotplug")
        self.joystick_manager = joystick_manager
        
        # Create hotplug detector
        self.detector = HotplugDetector(joystick_manager)
        
        # Register for hotplug events
        self.detector.add_callback(self._handle_hotplug_event)
        
        self.logger.debug("IntegratedHotplugManager initialized")
    
    def start(self) -> bool:
        """
        Start integrated hotplug management.
        
        Returns:
            True if started successfully, False otherwise
        """
        success = self.detector.start()
        if success:
            self.logger.info("Integrated hotplug management started")
        return success
    
    def stop(self) -> None:
        """Stop integrated hotplug management."""
        self.detector.stop()
        self.logger.info("Integrated hotplug management stopped")
    
    def _handle_hotplug_event(self, event: HotplugEvent) -> None:
        """
        Handle hotplug events and coordinate with joystick manager.
        
        Args:
            event: Hotplug event to handle
        """
        try:
            if event.event_type == HotplugEventType.DEVICE_ADDED:
                # Trigger joystick manager to scan for new devices
                self.joystick_manager._scan_joysticks()
                
                self.logger.info("Handled device addition", extra={
                    "device_id": event.joystick_id,
                    "timestamp": event.timestamp
                })
                
            elif event.event_type == HotplugEventType.DEVICE_REMOVED:
                # The joystick manager will handle this during its next scan
                # or we can proactively trigger a scan
                self.joystick_manager._scan_joysticks()
                
                self.logger.info("Handled device removal", extra={
                    "device_id": event.joystick_id,
                    "timestamp": event.timestamp
                })
            
        except Exception as e:
            self.logger.error("Error handling hotplug event", extra={
                "event_type": event.event_type.value,
                "device_id": event.joystick_id,
                "error": str(e)
            })
    
    def force_rescan(self) -> None:
        """Force a rescan of all devices."""
        self.detector.force_scan()
        self.joystick_manager._scan_joysticks()
        self.logger.debug("Force rescan completed")
    
    def is_active(self) -> bool:
        """Check if hotplug management is active."""
        return self.detector.is_running()


# Global hotplug manager instance
_hotplug_manager: Optional[IntegratedHotplugManager] = None


def get_hotplug_manager() -> Optional[IntegratedHotplugManager]:
    """Get the global hotplug manager instance."""
    return _hotplug_manager


def initialize_hotplug_manager(joystick_manager: JoystickManager) -> IntegratedHotplugManager:
    """
    Initialize the global hotplug manager.
    
    Args:
        joystick_manager: Main joystick manager instance
        
    Returns:
        IntegratedHotplugManager instance
    """
    global _hotplug_manager
    if _hotplug_manager is None:
        _hotplug_manager = IntegratedHotplugManager(joystick_manager)
    return _hotplug_manager


def shutdown_hotplug_manager() -> None:
    """Shutdown the global hotplug manager."""
    global _hotplug_manager
    if _hotplug_manager:
        _hotplug_manager.stop()
        _hotplug_manager = None