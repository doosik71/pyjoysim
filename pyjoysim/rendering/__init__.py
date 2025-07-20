"""
Rendering system for PyJoySim.

This module provides 2D/3D rendering capabilities with camera controls,
viewport management, and performance optimization.
"""

from .engine import (
    RenderEngine,
    Renderer2D,
    RenderEngineType,
    Color,
    StandardColors,
    BlendMode,
    Viewport,
    RenderStats,
    get_render_engine,
    create_render_engine,
    reset_render_engine
)
from .camera import (
    Camera2D,
    CameraController,
    CameraBounds
)

__all__ = [
    # Core rendering
    "RenderEngine",
    "Renderer2D", 
    "RenderEngineType",
    
    # Color and styling
    "Color",
    "StandardColors",
    "BlendMode",
    
    # Viewport and statistics
    "Viewport",
    "RenderStats",
    
    # Camera system
    "Camera2D",
    "CameraController", 
    "CameraBounds",
    
    # Factory functions
    "get_render_engine",
    "create_render_engine",
    "reset_render_engine",
]