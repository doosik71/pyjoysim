"""
Input management system for joystick and controller handling
"""

from .joystick_manager import (
    JoystickManager, 
    JoystickInfo, 
    JoystickInput, 
    JoystickState,
    get_joystick_manager,
    reset_joystick_manager
)
from .input_processor import (
    InputProcessor, 
    InputEvent, 
    InputEventType,
    AxisMapping,
    ButtonMapping,
    AxisType,
    InputFilter,
    SmoothingFilter,
    DeadzonFilter,
    get_input_processor,
    reset_input_processor
)
from .config_manager import (
    InputConfigManager,
    JoystickProfile,
    InputProfile,
    get_input_config_manager,
    reset_input_config_manager
)
from .hotplug import (
    HotplugDetector,
    IntegratedHotplugManager,
    HotplugEvent,
    HotplugEventType,
    get_hotplug_manager,
    initialize_hotplug_manager,
    shutdown_hotplug_manager
)
from .testing import (
    InputTester,
    TestCase,
    TestReport,
    TestType,
    TestResult
)

__all__ = [
    # Core managers
    "JoystickManager",
    "InputProcessor", 
    "InputConfigManager",
    
    # Data classes
    "JoystickInfo",
    "JoystickInput",
    "InputEvent",
    "JoystickProfile",
    "HotplugEvent",
    "TestCase",
    "TestReport",
    
    # Enums
    "JoystickState",
    "InputEventType",
    "AxisType",
    "InputProfile",
    "HotplugEventType",
    "TestType",
    "TestResult",
    
    # Mappings and filters
    "AxisMapping",
    "ButtonMapping",
    "InputFilter",
    "SmoothingFilter",
    "DeadzonFilter",
    
    # Hotplug support
    "HotplugDetector",
    "IntegratedHotplugManager",
    
    # Testing
    "InputTester",
    
    # Factory functions
    "get_joystick_manager",
    "reset_joystick_manager",
    "get_input_processor",
    "reset_input_processor",
    "get_input_config_manager",
    "reset_input_config_manager",
    "get_hotplug_manager",
    "initialize_hotplug_manager",
    "shutdown_hotplug_manager",
]