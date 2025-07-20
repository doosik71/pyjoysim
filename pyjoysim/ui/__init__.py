"""
User interface system for PyJoySim.

This module provides UI components including debug overlays,
performance monitors, and interactive controls.
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
]