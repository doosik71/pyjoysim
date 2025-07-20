"""
User interface system for PyJoySim.

This module provides comprehensive UI components including:
- Main application window and navigation
- Simulation selection and management
- Real-time control panels
- Settings management
- Debug overlays and performance monitoring
"""

from .overlay import (
    OverlayManager,
    OverlayElement,
    OverlayPosition,
    OverlayStyle,
    PerformanceMonitor,
    PhysicsDebugOverlay,
    InputDebugOverlay,
    CustomInfoPanel
)

from .main_window import (
    MainWindow, WindowState, MenuButton, UITheme
)

from .simulation_selector import (
    SimulationSelector, SelectorState, SimulationCard, CategoryFilter
)

from .control_panel import (
    ControlPanel, PanelState, ControlButton, MetricDisplay,
    create_control_panel
)

from .simulation_switcher import (
    SimulationSwitcher, SwitchState, SimulationSnapshot, TransitionConfig,
    create_simulation_switcher
)

from .settings_manager import (
    SettingsManager, SettingType, SettingDefinition,
    GraphicsSettings, AudioSettings, InputSettings, 
    SimulationSettings, UISettings,
    get_settings_manager, save_settings, load_settings
)

__all__ = [
    # Overlay system
    "OverlayManager",
    "OverlayElement",
    "OverlayPosition",
    "OverlayStyle",
    
    # Built-in overlays
    "PerformanceMonitor",
    "PhysicsDebugOverlay", 
    "InputDebugOverlay",
    "CustomInfoPanel",
    
    # Main UI components
    "MainWindow",
    "WindowState",
    "MenuButton",
    "UITheme",
    
    # Simulation selection
    "SimulationSelector",
    "SelectorState",
    "SimulationCard",
    "CategoryFilter",
    
    # Control panel
    "ControlPanel",
    "PanelState",
    "ControlButton",
    "MetricDisplay",
    "create_control_panel",
    
    # Simulation switching
    "SimulationSwitcher",
    "SwitchState",
    "SimulationSnapshot",
    "TransitionConfig",
    "create_simulation_switcher",
    
    # Settings management
    "SettingsManager",
    "SettingType",
    "SettingDefinition",
    "GraphicsSettings",
    "AudioSettings",
    "InputSettings",
    "SimulationSettings",
    "UISettings",
    "get_settings_manager",
    "save_settings",
    "load_settings",
]