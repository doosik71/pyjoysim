"""
Rendering engine abstract interface and implementations.

This module defines the core rendering architecture with abstract
base classes and concrete implementations for different rendering backends.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

import pygame
import pygame.gfxdraw
from pygame import Surface, Rect

from ..physics import Vector2D, PhysicsObject
from ..config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import RenderError, InitializationError


class RenderEngineType(Enum):
    """Types of rendering engines."""
    RENDERER_2D = "renderer_2d"
    RENDERER_3D = "renderer_3d"


class BlendMode(Enum):
    """Blending modes for rendering."""
    NORMAL = "normal"
    ALPHA = "alpha"
    ADD = "add"
    MULTIPLY = "multiply"


@dataclass
class Color:
    """RGBA color representation."""
    r: int
    g: int
    b: int
    a: int = 255
    
    def __post_init__(self):
        """Validate color values."""
        self.r = max(0, min(255, self.r))
        self.g = max(0, min(255, self.g))
        self.b = max(0, min(255, self.b))
        self.a = max(0, min(255, self.a))
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to RGB tuple."""
        return (self.r, self.g, self.b)
    
    def to_tuple_rgba(self) -> Tuple[int, int, int, int]:
        """Convert to RGBA tuple."""
        return (self.r, self.g, self.b, self.a)
    
    def with_alpha(self, alpha: int) -> 'Color':
        """Create new color with different alpha."""
        return Color(self.r, self.g, self.b, alpha)
    
    @classmethod
    def from_hex(cls, hex_color: str) -> 'Color':
        """Create color from hex string."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            return cls(
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16)
            )
        elif len(hex_color) == 8:
            return cls(
                int(hex_color[0:2], 16),
                int(hex_color[2:4], 16),
                int(hex_color[4:6], 16),
                int(hex_color[6:8], 16)
            )
        else:
            raise ValueError(f"Invalid hex color: {hex_color}")


@dataclass
class RenderStats:
    """Rendering performance statistics."""
    frame_count: int = 0
    total_render_time: float = 0.0
    objects_rendered: int = 0
    draw_calls: int = 0
    vertices_rendered: int = 0
    textures_used: int = 0
    average_frame_time: float = 0.0
    fps: float = 0.0
    
    def update(self, frame_time: float, objects: int = 0, draw_calls: int = 0):
        """Update statistics with new frame data."""
        self.frame_count += 1
        self.total_render_time += frame_time
        self.objects_rendered += objects
        self.draw_calls += draw_calls
        
        self.average_frame_time = self.total_render_time / self.frame_count
        if frame_time > 0:
            self.fps = 1.0 / frame_time


@dataclass
class Viewport:
    """Viewport configuration for rendering."""
    x: int = 0
    y: int = 0
    width: int = 800
    height: int = 600
    
    def get_rect(self) -> Rect:
        """Get pygame Rect for this viewport."""
        return Rect(self.x, self.y, self.width, self.height)
    
    def get_center(self) -> Vector2D:
        """Get center point of viewport."""
        return Vector2D(self.x + self.width // 2, self.y + self.height // 2)
    
    def get_aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 1.0


class RenderEngine(ABC):
    """
    Abstract base class for rendering engines.
    
    Defines the interface that all rendering engine implementations must follow.
    """
    
    def __init__(self):
        """Initialize the rendering engine."""
        self.logger = get_logger("render_engine")
        self.settings = get_settings()
        
        # Engine state
        self._initialized = False
        self._viewport = Viewport()
        self._background_color = Color(50, 50, 50)  # Dark gray
        
        # Rendering state
        self._current_surface: Optional[Surface] = None
        self._blend_mode = BlendMode.NORMAL
        
        # Statistics
        self._stats = RenderStats()
        self._last_frame_time = 0.0
        
        # Callbacks
        self._pre_render_callbacks: List[Callable] = []
        self._post_render_callbacks: List[Callable] = []
    
    @abstractmethod
    def initialize(self, width: int = 800, height: int = 600, title: str = "PyJoySim") -> bool:
        """
        Initialize the rendering engine.
        
        Args:
            width: Window width
            height: Window height
            title: Window title
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the rendering engine and clean up resources."""
        pass
    
    @abstractmethod
    def begin_frame(self) -> None:
        """Begin a new rendering frame."""
        pass
    
    @abstractmethod
    def end_frame(self) -> None:
        """End the current rendering frame and present to screen."""
        pass
    
    @abstractmethod
    def clear(self, color: Optional[Color] = None) -> None:
        """
        Clear the rendering surface.
        
        Args:
            color: Clear color (uses background color if None)
        """
        pass
    
    @abstractmethod
    def draw_circle(self, 
                   center: Vector2D, 
                   radius: float, 
                   color: Color,
                   fill: bool = True,
                   width: int = 1) -> None:
        """
        Draw a circle.
        
        Args:
            center: Circle center point
            radius: Circle radius
            color: Circle color
            fill: Whether to fill the circle
            width: Line width for outline
        """
        pass
    
    @abstractmethod
    def draw_rectangle(self, 
                      position: Vector2D, 
                      width: float, 
                      height: float, 
                      color: Color,
                      fill: bool = True,
                      line_width: int = 1) -> None:
        """
        Draw a rectangle.
        
        Args:
            position: Rectangle position (center)
            width: Rectangle width
            height: Rectangle height
            color: Rectangle color
            fill: Whether to fill the rectangle
            line_width: Line width for outline
        """
        pass
    
    @abstractmethod
    def draw_line(self, 
                 start: Vector2D, 
                 end: Vector2D, 
                 color: Color,
                 width: int = 1) -> None:
        """
        Draw a line.
        
        Args:
            start: Line start point
            end: Line end point
            color: Line color
            width: Line width
        """
        pass
    
    @abstractmethod
    def draw_polygon(self, 
                    points: List[Vector2D], 
                    color: Color,
                    fill: bool = True,
                    width: int = 1) -> None:
        """
        Draw a polygon.
        
        Args:
            points: Polygon vertices
            color: Polygon color
            fill: Whether to fill the polygon
            width: Line width for outline
        """
        pass
    
    @abstractmethod
    def draw_text(self, 
                 text: str, 
                 position: Vector2D, 
                 color: Color,
                 font_size: int = 16,
                 font_name: Optional[str] = None) -> None:
        """
        Draw text.
        
        Args:
            text: Text to draw
            position: Text position
            color: Text color
            font_size: Font size
            font_name: Font name (uses default if None)
        """
        pass
    
    # Common implementation methods
    
    def set_viewport(self, viewport: Viewport) -> None:
        """Set the rendering viewport."""
        self._viewport = viewport
        self.logger.debug("Viewport updated", extra={
            "width": viewport.width,
            "height": viewport.height,
            "x": viewport.x,
            "y": viewport.y
        })
    
    def get_viewport(self) -> Viewport:
        """Get the current viewport."""
        return self._viewport
    
    def set_background_color(self, color: Color) -> None:
        """Set the background clear color."""
        self._background_color = color
    
    def get_background_color(self) -> Color:
        """Get the background clear color."""
        return self._background_color
    
    def set_blend_mode(self, mode: BlendMode) -> None:
        """Set the blending mode."""
        self._blend_mode = mode
    
    def get_blend_mode(self) -> BlendMode:
        """Get the current blending mode."""
        return self._blend_mode
    
    def get_stats(self) -> RenderStats:
        """Get rendering statistics."""
        return self._stats
    
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized
    
    def add_pre_render_callback(self, callback: Callable) -> None:
        """Add a pre-render callback."""
        self._pre_render_callbacks.append(callback)
    
    def add_post_render_callback(self, callback: Callable) -> None:
        """Add a post-render callback."""
        self._post_render_callbacks.append(callback)
    
    def _call_pre_render_callbacks(self) -> None:
        """Call all pre-render callbacks."""
        for callback in self._pre_render_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error("Error in pre-render callback", extra={"error": str(e)})
    
    def _call_post_render_callbacks(self) -> None:
        """Call all post-render callbacks."""
        for callback in self._post_render_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error("Error in post-render callback", extra={"error": str(e)})


class Renderer2D(RenderEngine):
    """
    2D rendering engine implementation using pygame.
    
    Provides hardware-accelerated 2D rendering with sprite support.
    """
    
    def __init__(self):
        """Initialize the 2D renderer."""
        super().__init__()
        
        # Pygame-specific attributes
        self._screen: Optional[Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._fonts: Dict[Tuple[str, int], pygame.font.Font] = {}
        self._surfaces: Dict[str, Surface] = {}
        
        # Rendering optimization
        self._dirty_rects: List[Rect] = []
        self._use_dirty_rects = False
        
        self.logger.debug("Renderer2D created")
    
    def initialize(self, width: int = 800, height: int = 600, title: str = "PyJoySim") -> bool:
        """Initialize the pygame rendering system."""
        if self._initialized:
            self.logger.warning("Renderer2D already initialized")
            return True
        
        try:
            # Initialize pygame
            pygame.init()
            pygame.display.set_caption(title)
            
            # Create display surface
            self._screen = pygame.display.set_mode((width, height))
            self._clock = pygame.time.Clock()
            
            # Set viewport
            self._viewport = Viewport(0, 0, width, height)
            self._current_surface = self._screen
            
            # Initialize font system
            pygame.font.init()
            
            self._initialized = True
            
            self.logger.info("Renderer2D initialized", extra={
                "width": width,
                "height": height,
                "title": title
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize Renderer2D", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    def shutdown(self) -> None:
        """Shutdown the pygame rendering system."""
        if not self._initialized:
            return
        
        self.logger.info("Shutting down Renderer2D")
        
        # Clean up fonts
        self._fonts.clear()
        
        # Clean up surfaces
        self._surfaces.clear()
        
        # Shutdown pygame
        try:
            pygame.font.quit()
            pygame.display.quit()
            pygame.quit()
        except Exception as e:
            self.logger.warning("Error during pygame shutdown", extra={"error": str(e)})
        
        self._screen = None
        self._clock = None
        self._current_surface = None
        self._initialized = False
        
        self.logger.info("Renderer2D shutdown complete")
    
    def begin_frame(self) -> None:
        """Begin a new rendering frame."""
        if not self._initialized:
            return
        
        frame_start = time.time()
        self._last_frame_time = frame_start
        
        # Clear dirty rects from previous frame
        self._dirty_rects.clear()
        
        # Call pre-render callbacks
        self._call_pre_render_callbacks()
    
    def end_frame(self) -> None:
        """End the current rendering frame and present to screen."""
        if not self._initialized:
            return
        
        try:
            # Update display
            if self._use_dirty_rects and self._dirty_rects:
                pygame.display.update(self._dirty_rects)
            else:
                pygame.display.flip()
            
            # Update frame timing
            if self._clock:
                self._clock.tick(60)  # Target 60 FPS
            
            # Update statistics
            frame_time = time.time() - self._last_frame_time
            self._stats.update(frame_time, draw_calls=len(self._dirty_rects))
            
            # Call post-render callbacks
            self._call_post_render_callbacks()
            
        except Exception as e:
            self.logger.error("Error ending frame", extra={"error": str(e)})
    
    def clear(self, color: Optional[Color] = None) -> None:
        """Clear the rendering surface."""
        if not self._initialized or not self._current_surface:
            return
        
        clear_color = color or self._background_color
        self._current_surface.fill(clear_color.to_tuple())
        
        # Add full screen to dirty rects
        if self._use_dirty_rects:
            self._dirty_rects.append(self._current_surface.get_rect())
    
    def draw_circle(self, 
                   center: Vector2D, 
                   radius: float, 
                   color: Color,
                   fill: bool = True,
                   width: int = 1) -> None:
        """Draw a circle using pygame."""
        if not self._initialized or not self._current_surface:
            return
        
        try:
            # Convert world coordinates to screen coordinates
            screen_pos = self._world_to_screen(center)
            screen_radius = max(1, int(radius))
            
            if fill:
                pygame.draw.circle(
                    self._current_surface,
                    color.to_tuple(),
                    (int(screen_pos.x), int(screen_pos.y)),
                    screen_radius
                )
            else:
                pygame.draw.circle(
                    self._current_surface,
                    color.to_tuple(),
                    (int(screen_pos.x), int(screen_pos.y)),
                    screen_radius,
                    width
                )
            
            # Add to dirty rects
            if self._use_dirty_rects:
                rect = Rect(
                    int(screen_pos.x - screen_radius),
                    int(screen_pos.y - screen_radius),
                    screen_radius * 2,
                    screen_radius * 2
                )
                self._dirty_rects.append(rect)
                
        except Exception as e:
            self.logger.error("Error drawing circle", extra={"error": str(e)})
    
    def draw_rectangle(self, 
                      position: Vector2D, 
                      width: float, 
                      height: float, 
                      color: Color,
                      fill: bool = True,
                      line_width: int = 1) -> None:
        """Draw a rectangle using pygame."""
        if not self._initialized or not self._current_surface:
            return
        
        try:
            # Convert world coordinates to screen coordinates
            screen_pos = self._world_to_screen(position)
            screen_width = max(1, int(width))
            screen_height = max(1, int(height))
            
            # Create rectangle (centered on position)
            rect = Rect(
                int(screen_pos.x - screen_width // 2),
                int(screen_pos.y - screen_height // 2),
                screen_width,
                screen_height
            )
            
            if fill:
                pygame.draw.rect(self._current_surface, color.to_tuple(), rect)
            else:
                pygame.draw.rect(self._current_surface, color.to_tuple(), rect, line_width)
            
            # Add to dirty rects
            if self._use_dirty_rects:
                self._dirty_rects.append(rect)
                
        except Exception as e:
            self.logger.error("Error drawing rectangle", extra={"error": str(e)})
    
    def draw_line(self, 
                 start: Vector2D, 
                 end: Vector2D, 
                 color: Color,
                 width: int = 1) -> None:
        """Draw a line using pygame."""
        if not self._initialized or not self._current_surface:
            return
        
        try:
            # Convert world coordinates to screen coordinates
            screen_start = self._world_to_screen(start)
            screen_end = self._world_to_screen(end)
            
            pygame.draw.line(
                self._current_surface,
                color.to_tuple(),
                (int(screen_start.x), int(screen_start.y)),
                (int(screen_end.x), int(screen_end.y)),
                width
            )
            
            # Add to dirty rects
            if self._use_dirty_rects:
                min_x = min(screen_start.x, screen_end.x) - width
                min_y = min(screen_start.y, screen_end.y) - width
                max_x = max(screen_start.x, screen_end.x) + width
                max_y = max(screen_start.y, screen_end.y) + width
                
                rect = Rect(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
                self._dirty_rects.append(rect)
                
        except Exception as e:
            self.logger.error("Error drawing line", extra={"error": str(e)})
    
    def draw_polygon(self, 
                    points: List[Vector2D], 
                    color: Color,
                    fill: bool = True,
                    width: int = 1) -> None:
        """Draw a polygon using pygame."""
        if not self._initialized or not self._current_surface or len(points) < 3:
            return
        
        try:
            # Convert world coordinates to screen coordinates
            screen_points = [
                (int(self._world_to_screen(point).x), int(self._world_to_screen(point).y))
                for point in points
            ]
            
            if fill:
                pygame.draw.polygon(self._current_surface, color.to_tuple(), screen_points)
            else:
                pygame.draw.polygon(self._current_surface, color.to_tuple(), screen_points, width)
            
            # Add to dirty rects
            if self._use_dirty_rects:
                min_x = min(p[0] for p in screen_points) - width
                min_y = min(p[1] for p in screen_points) - width
                max_x = max(p[0] for p in screen_points) + width
                max_y = max(p[1] for p in screen_points) + width
                
                rect = Rect(int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))
                self._dirty_rects.append(rect)
                
        except Exception as e:
            self.logger.error("Error drawing polygon", extra={"error": str(e)})
    
    def draw_text(self, 
                 text: str, 
                 position: Vector2D, 
                 color: Color,
                 font_size: int = 16,
                 font_name: Optional[str] = None) -> None:
        """Draw text using pygame."""
        if not self._initialized or not self._current_surface:
            return
        
        try:
            # Get or create font
            font = self._get_font(font_name, font_size)
            
            # Render text
            text_surface = font.render(text, True, color.to_tuple())
            
            # Convert world coordinates to screen coordinates
            screen_pos = self._world_to_screen(position)
            
            # Blit text to surface
            self._current_surface.blit(text_surface, (int(screen_pos.x), int(screen_pos.y)))
            
            # Add to dirty rects
            if self._use_dirty_rects:
                rect = text_surface.get_rect()
                rect.topleft = (int(screen_pos.x), int(screen_pos.y))
                self._dirty_rects.append(rect)
                
        except Exception as e:
            self.logger.error("Error drawing text", extra={"error": str(e)})
    
    def _get_font(self, font_name: Optional[str], font_size: int) -> pygame.font.Font:
        """Get or create a font."""
        font_key = (font_name or "default", font_size)
        
        if font_key not in self._fonts:
            try:
                if font_name:
                    font = pygame.font.Font(font_name, font_size)
                else:
                    font = pygame.font.Font(None, font_size)
                self._fonts[font_key] = font
            except Exception as e:
                self.logger.warning("Failed to load font, using default", extra={
                    "font_name": font_name,
                    "font_size": font_size,
                    "error": str(e)
                })
                # Fallback to default font
                font = pygame.font.Font(None, font_size)
                self._fonts[font_key] = font
        
        return self._fonts[font_key]
    
    def _world_to_screen(self, world_pos: Vector2D) -> Vector2D:
        """Convert world coordinates to screen coordinates."""
        # For now, use simple 1:1 mapping with screen center as origin
        screen_x = world_pos.x + self._viewport.width // 2
        screen_y = self._viewport.height // 2 - world_pos.y  # Flip Y axis
        
        return Vector2D(screen_x, screen_y)
    
    def _screen_to_world(self, screen_pos: Vector2D) -> Vector2D:
        """Convert screen coordinates to world coordinates."""
        world_x = screen_pos.x - self._viewport.width // 2
        world_y = self._viewport.height // 2 - screen_pos.y  # Flip Y axis
        
        return Vector2D(world_x, world_y)
    
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions."""
        return (self._viewport.width, self._viewport.height)
    
    def enable_dirty_rect_optimization(self, enabled: bool = True) -> None:
        """Enable or disable dirty rectangle optimization."""
        self._use_dirty_rects = enabled
        self.logger.debug("Dirty rect optimization", extra={"enabled": enabled})


# Predefined colors
class StandardColors:
    """Collection of standard color constants."""
    
    BLACK = Color(0, 0, 0)
    WHITE = Color(255, 255, 255)
    RED = Color(255, 0, 0)
    GREEN = Color(0, 255, 0)
    BLUE = Color(0, 0, 255)
    YELLOW = Color(255, 255, 0)
    CYAN = Color(0, 255, 255)
    MAGENTA = Color(255, 0, 255)
    GRAY = Color(128, 128, 128)
    DARK_GRAY = Color(64, 64, 64)
    LIGHT_GRAY = Color(192, 192, 192)
    ORANGE = Color(255, 165, 0)
    PURPLE = Color(128, 0, 128)
    BROWN = Color(165, 42, 42)
    PINK = Color(255, 192, 203)
    LIME = Color(0, 255, 0)
    NAVY = Color(0, 0, 128)
    TRANSPARENT = Color(0, 0, 0, 0)


# Global rendering engine instance
_render_engine: Optional[RenderEngine] = None


def get_render_engine() -> Optional[RenderEngine]:
    """Get the global rendering engine instance."""
    return _render_engine


def create_render_engine(engine_type: RenderEngineType) -> RenderEngine:
    """
    Create and set a global rendering engine instance.
    
    Args:
        engine_type: Type of rendering engine to create
        
    Returns:
        Created rendering engine instance
    """
    global _render_engine
    
    if engine_type == RenderEngineType.RENDERER_2D:
        _render_engine = Renderer2D()
    else:
        raise ValueError(f"Unsupported rendering engine type: {engine_type}")
    
    return _render_engine


def reset_render_engine() -> None:
    """Reset the global rendering engine instance."""
    global _render_engine
    if _render_engine and _render_engine.is_initialized():
        _render_engine.shutdown()
    _render_engine = None