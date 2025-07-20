"""
Input processing system for PyJoySim.

This module provides high-level input processing, event generation, and
input mapping functionality for joystick inputs.
"""

import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from .joystick_manager import JoystickInput, JoystickInfo
from ..config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import InputError, InputMappingError


class InputEventType(Enum):
    """Types of input events."""
    BUTTON_PRESS = "button_press"
    BUTTON_RELEASE = "button_release"
    AXIS_CHANGE = "axis_change"
    HAT_CHANGE = "hat_change"
    STICK_MOVE = "stick_move"
    TRIGGER_CHANGE = "trigger_change"


class AxisType(Enum):
    """Types of axis inputs."""
    LINEAR = "linear"      # Regular axis (-1.0 to 1.0)
    TRIGGER = "trigger"    # Trigger axis (0.0 to 1.0)
    STICK_X = "stick_x"    # Analog stick X axis
    STICK_Y = "stick_y"    # Analog stick Y axis


@dataclass
class InputEvent:
    """Represents an input event."""
    event_type: InputEventType
    joystick_id: int
    timestamp: float
    
    # Event-specific data
    button_id: Optional[int] = None
    axis_id: Optional[int] = None
    axis_value: Optional[float] = None
    hat_id: Optional[int] = None
    hat_value: Optional[tuple[int, int]] = None
    
    # Processed data
    stick_x: Optional[float] = None
    stick_y: Optional[float] = None
    trigger_value: Optional[float] = None
    
    def __str__(self) -> str:
        """String representation of the event."""
        parts = [f"{self.event_type.value}"]
        
        if self.button_id is not None:
            parts.append(f"button={self.button_id}")
        if self.axis_id is not None:
            parts.append(f"axis={self.axis_id}:{self.axis_value:.3f}")
        if self.hat_id is not None:
            parts.append(f"hat={self.hat_id}:{self.hat_value}")
        if self.stick_x is not None or self.stick_y is not None:
            parts.append(f"stick=({self.stick_x:.3f},{self.stick_y:.3f})")
        if self.trigger_value is not None:
            parts.append(f"trigger={self.trigger_value:.3f}")
        
        return f"InputEvent(js={self.joystick_id}, {', '.join(parts)})"


@dataclass
class AxisMapping:
    """Configuration for axis mapping."""
    axis_id: int
    axis_type: AxisType
    invert: bool = False
    scale: float = 1.0
    offset: float = 0.0
    curve: str = "linear"  # linear, quadratic, cubic
    
    def process_value(self, raw_value: float) -> float:
        """
        Process raw axis value through mapping.
        
        Args:
            raw_value: Raw axis value from joystick
            
        Returns:
            Processed value
        """
        # Apply offset and scale
        value = (raw_value + self.offset) * self.scale
        
        # Apply inversion
        if self.invert:
            value = -value
        
        # Apply curve
        if self.curve == "quadratic":
            # Preserve sign, apply quadratic curve
            value = (value ** 2) * (1 if value >= 0 else -1)
        elif self.curve == "cubic":
            value = value ** 3
        
        # Clamp to valid range
        if self.axis_type == AxisType.TRIGGER:
            return max(0.0, min(1.0, value))
        else:
            return max(-1.0, min(1.0, value))


@dataclass
class ButtonMapping:
    """Configuration for button mapping."""
    button_id: int
    repeat_enabled: bool = False
    repeat_delay: float = 0.5  # Initial delay before repeat
    repeat_rate: float = 0.1   # Time between repeats
    
    # Internal state
    _last_press_time: float = field(default=0.0, init=False)
    _last_repeat_time: float = field(default=0.0, init=False)
    _is_pressed: bool = field(default=False, init=False)
    
    def should_repeat(self, current_time: float) -> bool:
        """
        Check if button should generate repeat event.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if repeat event should be generated
        """
        if not self.repeat_enabled or not self._is_pressed:
            return False
        
        # Check if initial delay has passed
        if current_time - self._last_press_time < self.repeat_delay:
            return False
        
        # Check if repeat rate has passed
        return current_time - self._last_repeat_time >= self.repeat_rate
    
    def on_press(self, timestamp: float) -> None:
        """Handle button press."""
        self._is_pressed = True
        self._last_press_time = timestamp
        self._last_repeat_time = timestamp
    
    def on_release(self, timestamp: float) -> None:
        """Handle button release."""
        self._is_pressed = False
    
    def on_repeat(self, timestamp: float) -> None:
        """Handle button repeat."""
        self._last_repeat_time = timestamp


class InputFilter(ABC):
    """Abstract base class for input filters."""
    
    @abstractmethod
    def process_input(self, input_state: JoystickInput) -> JoystickInput:
        """
        Process input through filter.
        
        Args:
            input_state: Input state to process
            
        Returns:
            Filtered input state
        """
        pass


class SmoothingFilter(InputFilter):
    """Smoothing filter for reducing input noise."""
    
    def __init__(self, smoothing_factor: float = 0.2):
        """
        Initialize smoothing filter.
        
        Args:
            smoothing_factor: Smoothing factor (0.0 = no smoothing, 1.0 = max smoothing)
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))
        self._previous_axes: Dict[int, List[float]] = {}
    
    def process_input(self, input_state: JoystickInput) -> JoystickInput:
        """Apply smoothing to axis values."""
        joystick_id = input_state.joystick_id
        
        # Initialize previous values if needed
        if joystick_id not in self._previous_axes:
            self._previous_axes[joystick_id] = input_state.axes.copy()
            return input_state
        
        # Apply smoothing
        smoothed_axes = []
        prev_axes = self._previous_axes[joystick_id]
        
        for i, current_value in enumerate(input_state.axes):
            if i < len(prev_axes):
                # Exponential smoothing
                smoothed_value = (prev_axes[i] * self.smoothing_factor + 
                                current_value * (1.0 - self.smoothing_factor))
                smoothed_axes.append(smoothed_value)
            else:
                smoothed_axes.append(current_value)
        
        # Update previous values
        self._previous_axes[joystick_id] = smoothed_axes.copy()
        
        # Create new input state with smoothed axes
        return JoystickInput(
            joystick_id=input_state.joystick_id,
            axes=smoothed_axes,
            buttons=input_state.buttons,
            hats=input_state.hats,
            timestamp=input_state.timestamp
        )


class DeadzonFilter(InputFilter):
    """Filter for applying custom deadzones to specific axes."""
    
    def __init__(self, axis_deadzones: Dict[int, float]):
        """
        Initialize deadzone filter.
        
        Args:
            axis_deadzones: Mapping of axis ID to deadzone value
        """
        self.axis_deadzones = axis_deadzones
    
    def process_input(self, input_state: JoystickInput) -> JoystickInput:
        """Apply custom deadzones to specified axes."""
        filtered_axes = []
        
        for i, value in enumerate(input_state.axes):
            if i in self.axis_deadzones:
                deadzone = self.axis_deadzones[i]
                if abs(value) < deadzone:
                    filtered_axes.append(0.0)
                else:
                    # Scale to account for deadzone
                    if value > 0:
                        filtered_axes.append((value - deadzone) / (1.0 - deadzone))
                    else:
                        filtered_axes.append((value + deadzone) / (1.0 - deadzone))
            else:
                filtered_axes.append(value)
        
        return JoystickInput(
            joystick_id=input_state.joystick_id,
            axes=filtered_axes,
            buttons=input_state.buttons,
            hats=input_state.hats,
            timestamp=input_state.timestamp
        )


@dataclass
class InputState:
    """
    High-level input state for simulation use.
    
    This class provides a simplified interface for accessing input state
    in simulations, abstracting away the low-level joystick details.
    """
    # Joystick states (list of joystick inputs)
    joystick_inputs: List[JoystickInput] = field(default_factory=list)
    
    # Keyboard state (if implemented)
    keys_pressed: set = field(default_factory=set)
    
    # Combined state helpers
    def joystick_available(self) -> bool:
        """Check if any joystick is available."""
        return len(self.joystick_inputs) > 0
    
    def get_joystick_state(self, joystick_id: int) -> Optional[Dict[str, Any]]:
        """Get state for specific joystick as a dictionary."""
        for input_data in self.joystick_inputs:
            if input_data.joystick_id == joystick_id:
                # Convert to dictionary format for easy access
                buttons = {}
                for i, pressed in enumerate(input_data.buttons):
                    buttons[i] = pressed
                
                axes = {}
                for i, value in enumerate(input_data.axes):
                    axes[f'axis_{i}'] = value
                
                return {
                    'buttons': buttons,
                    **axes
                }
        return None
    
    def get_keys_pressed(self) -> set:
        """Get currently pressed keys."""
        return self.keys_pressed.copy()


class InputProcessor:
    """
    High-level input processor for converting raw joystick input into game events.
    
    Features:
    - Event generation from input changes
    - Button repeat handling
    - Axis mapping and processing
    - Input filtering pipeline
    - Performance monitoring
    """
    
    def __init__(self, joystick_manager=None):
        """Initialize the input processor."""
        self.logger = get_logger("input_processor")
        self.settings = get_settings()
        
        # Store joystick manager reference
        self.joystick_manager = joystick_manager
        
        # State tracking
        self._previous_inputs: Dict[int, JoystickInput] = {}
        self._joystick_mappings: Dict[int, Dict[str, Any]] = {}
        
        # Input filters
        self._filters: List[InputFilter] = []
        
        # Event callbacks
        self._event_callbacks: List[Callable[[InputEvent], None]] = []
        
        # Button mappings and state
        self._button_mappings: Dict[int, Dict[int, ButtonMapping]] = {}
        
        # Axis mappings
        self._axis_mappings: Dict[int, Dict[int, AxisMapping]] = {}
        
        # Performance tracking
        self._events_processed = 0
        self._last_performance_log = 0.0
        
        # Configuration
        self._event_threshold = 0.01  # Minimum change to generate axis event
        
        self.logger.debug("InputProcessor initialized")
    
    def add_filter(self, input_filter: InputFilter) -> None:
        """
        Add an input filter to the processing pipeline.
        
        Args:
            input_filter: Filter to add
        """
        self._filters.append(input_filter)
        self.logger.debug("Input filter added", extra={
            "filter_type": type(input_filter).__name__,
            "total_filters": len(self._filters)
        })
    
    def remove_filter(self, input_filter: InputFilter) -> None:
        """
        Remove an input filter from the processing pipeline.
        
        Args:
            input_filter: Filter to remove
        """
        if input_filter in self._filters:
            self._filters.remove(input_filter)
            self.logger.debug("Input filter removed", extra={
                "filter_type": type(input_filter).__name__,
                "total_filters": len(self._filters)
            })
    
    def set_button_mapping(self, joystick_id: int, button_mappings: Dict[int, ButtonMapping]) -> None:
        """
        Set button mappings for a joystick.
        
        Args:
            joystick_id: ID of the joystick
            button_mappings: Mapping of button ID to ButtonMapping
        """
        self._button_mappings[joystick_id] = button_mappings
        self.logger.debug("Button mappings set", extra={
            "joystick_id": joystick_id,
            "button_count": len(button_mappings)
        })
    
    def set_axis_mapping(self, joystick_id: int, axis_mappings: Dict[int, AxisMapping]) -> None:
        """
        Set axis mappings for a joystick.
        
        Args:
            joystick_id: ID of the joystick
            axis_mappings: Mapping of axis ID to AxisMapping
        """
        self._axis_mappings[joystick_id] = axis_mappings
        self.logger.debug("Axis mappings set", extra={
            "joystick_id": joystick_id,
            "axis_count": len(axis_mappings)
        })
    
    def process_input(self, joystick_id: int, raw_input: JoystickInput) -> List[InputEvent]:
        """
        Process raw joystick input and generate events.
        
        Args:
            joystick_id: ID of the joystick
            raw_input: Raw input from joystick
            
        Returns:
            List of generated input events
        """
        events = []
        
        try:
            # Apply input filters
            filtered_input = raw_input
            for filter_obj in self._filters:
                filtered_input = filter_obj.process_input(filtered_input)
            
            # Get previous input state
            previous_input = self._previous_inputs.get(joystick_id)
            
            # Generate events from input changes
            if previous_input is not None:
                events.extend(self._generate_button_events(filtered_input, previous_input))
                events.extend(self._generate_axis_events(filtered_input, previous_input))
                events.extend(self._generate_hat_events(filtered_input, previous_input))
            
            # Generate button repeat events
            events.extend(self._generate_repeat_events(joystick_id, filtered_input.timestamp))
            
            # Store current input as previous
            self._previous_inputs[joystick_id] = filtered_input
            
            # Notify event callbacks
            for event in events:
                for callback in self._event_callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        self.logger.error("Error in event callback", extra={"error": str(e)})
            
            # Update performance counters
            self._events_processed += len(events)
            self._log_performance_if_needed()
            
        except Exception as e:
            self.logger.error("Error processing input", extra={
                "joystick_id": joystick_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
        
        return events
    
    def _generate_button_events(self, current: JoystickInput, previous: JoystickInput) -> List[InputEvent]:
        """Generate button press/release events."""
        events = []
        joystick_id = current.joystick_id
        
        for i, (curr_pressed, prev_pressed) in enumerate(zip(current.buttons, previous.buttons)):
            if curr_pressed != prev_pressed:
                # Get button mapping if available
                button_mapping = None
                if (joystick_id in self._button_mappings and 
                    i in self._button_mappings[joystick_id]):
                    button_mapping = self._button_mappings[joystick_id][i]
                
                if curr_pressed:
                    # Button press
                    event = InputEvent(
                        event_type=InputEventType.BUTTON_PRESS,
                        joystick_id=joystick_id,
                        timestamp=current.timestamp,
                        button_id=i
                    )
                    events.append(event)
                    
                    if button_mapping:
                        button_mapping.on_press(current.timestamp)
                else:
                    # Button release
                    event = InputEvent(
                        event_type=InputEventType.BUTTON_RELEASE,
                        joystick_id=joystick_id,
                        timestamp=current.timestamp,
                        button_id=i
                    )
                    events.append(event)
                    
                    if button_mapping:
                        button_mapping.on_release(current.timestamp)
        
        return events
    
    def _generate_axis_events(self, current: JoystickInput, previous: JoystickInput) -> List[InputEvent]:
        """Generate axis change events."""
        events = []
        joystick_id = current.joystick_id
        
        for i, (curr_value, prev_value) in enumerate(zip(current.axes, previous.axes)):
            if abs(curr_value - prev_value) > self._event_threshold:
                # Apply axis mapping if available
                processed_value = curr_value
                if (joystick_id in self._axis_mappings and 
                    i in self._axis_mappings[joystick_id]):
                    axis_mapping = self._axis_mappings[joystick_id][i]
                    processed_value = axis_mapping.process_value(curr_value)
                
                event = InputEvent(
                    event_type=InputEventType.AXIS_CHANGE,
                    joystick_id=joystick_id,
                    timestamp=current.timestamp,
                    axis_id=i,
                    axis_value=processed_value
                )
                
                # Add processed data based on axis type
                if (joystick_id in self._axis_mappings and 
                    i in self._axis_mappings[joystick_id]):
                    axis_mapping = self._axis_mappings[joystick_id][i]
                    
                    if axis_mapping.axis_type == AxisType.TRIGGER:
                        event.trigger_value = processed_value
                        event.event_type = InputEventType.TRIGGER_CHANGE
                    elif axis_mapping.axis_type in (AxisType.STICK_X, AxisType.STICK_Y):
                        # For stick events, we might want to combine X and Y
                        if axis_mapping.axis_type == AxisType.STICK_X:
                            event.stick_x = processed_value
                        else:
                            event.stick_y = processed_value
                        event.event_type = InputEventType.STICK_MOVE
                
                events.append(event)
        
        return events
    
    def _generate_hat_events(self, current: JoystickInput, previous: JoystickInput) -> List[InputEvent]:
        """Generate hat (D-pad) change events."""
        events = []
        joystick_id = current.joystick_id
        
        for i, (curr_hat, prev_hat) in enumerate(zip(current.hats, previous.hats)):
            if curr_hat != prev_hat:
                event = InputEvent(
                    event_type=InputEventType.HAT_CHANGE,
                    joystick_id=joystick_id,
                    timestamp=current.timestamp,
                    hat_id=i,
                    hat_value=curr_hat
                )
                events.append(event)
        
        return events
    
    def _generate_repeat_events(self, joystick_id: int, timestamp: float) -> List[InputEvent]:
        """Generate button repeat events."""
        events = []
        
        if joystick_id not in self._button_mappings:
            return events
        
        for button_id, button_mapping in self._button_mappings[joystick_id].items():
            if button_mapping.should_repeat(timestamp):
                event = InputEvent(
                    event_type=InputEventType.BUTTON_PRESS,
                    joystick_id=joystick_id,
                    timestamp=timestamp,
                    button_id=button_id
                )
                events.append(event)
                button_mapping.on_repeat(timestamp)
        
        return events
    
    def _log_performance_if_needed(self) -> None:
        """Log performance statistics periodically."""
        current_time = time.time()
        if (current_time - self._last_performance_log) >= 10.0:  # Every 10 seconds
            event_rate = self._events_processed / 10.0
            self.logger.debug("Input processor performance", extra={
                "events_per_second": event_rate,
                "active_joysticks": len(self._previous_inputs),
                "total_filters": len(self._filters)
            })
            self._events_processed = 0
            self._last_performance_log = current_time
    
    def add_event_callback(self, callback: Callable[[InputEvent], None]) -> None:
        """
        Register a callback for input events.
        
        Args:
            callback: Function to call when events occur
        """
        self._event_callbacks.append(callback)
    
    def remove_event_callback(self, callback: Callable[[InputEvent], None]) -> None:
        """
        Remove an event callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
    
    def clear_joystick_state(self, joystick_id: int) -> None:
        """
        Clear stored state for a joystick (e.g., when disconnected).
        
        Args:
            joystick_id: ID of joystick to clear
        """
        if joystick_id in self._previous_inputs:
            del self._previous_inputs[joystick_id]
        
        # Reset button states
        if joystick_id in self._button_mappings:
            for button_mapping in self._button_mappings[joystick_id].values():
                button_mapping._is_pressed = False
        
        self.logger.debug("Joystick state cleared", extra={"joystick_id": joystick_id})
    
    def process_input(self) -> InputState:
        """
        Process input from all joysticks and return InputState.
        
        Returns:
            InputState containing current input state
        """
        input_state = InputState()
        
        if self.joystick_manager:
            # Get current input from all joysticks
            all_inputs = self.joystick_manager.get_all_inputs()
            input_state.joystick_inputs = all_inputs
        
        # Keyboard support would be added here if implemented
        # For now, return empty keyboard state
        
        return input_state


# Global input processor instance
_input_processor: Optional[InputProcessor] = None


def get_input_processor() -> InputProcessor:
    """Get the global input processor instance."""
    global _input_processor
    if _input_processor is None:
        _input_processor = InputProcessor()
    return _input_processor


def reset_input_processor() -> None:
    """Reset the global input processor instance."""
    global _input_processor
    _input_processor = None