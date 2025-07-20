"""
UI overlay system for PyJoySim.

This module provides debug overlays, performance monitors, and interactive
controls for simulations.
"""

import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pygame

from ..physics import Vector2D, PhysicsWorld
from ..rendering import RenderEngine, Color, StandardColors
from ..input import InputEvent, InputEventType
from ..core.logging import get_logger


class OverlayPosition(Enum):
    """Positions for overlay elements."""
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    CENTER = "center"
    CUSTOM = "custom"


class OverlayStyle(Enum):
    """Styles for overlay elements."""
    MINIMAL = "minimal"
    DETAILED = "detailed"
    COMPACT = "compact"
    LARGE = "large"


@dataclass
class OverlayElement:
    """Base class for overlay elements."""
    name: str
    position: OverlayPosition
    visible: bool = True
    x_offset: int = 0
    y_offset: int = 0
    background_color: Optional[Color] = None
    text_color: Color = StandardColors.WHITE
    font_size: int = 14
    
    def get_position(self, viewport_width: int, viewport_height: int) -> Vector2D:
        """Calculate actual position based on viewport size."""
        if self.position == OverlayPosition.TOP_LEFT:
            return Vector2D(10 + self.x_offset, 10 + self.y_offset)
        elif self.position == OverlayPosition.TOP_RIGHT:
            return Vector2D(viewport_width - 200 + self.x_offset, 10 + self.y_offset)
        elif self.position == OverlayPosition.BOTTOM_LEFT:
            return Vector2D(10 + self.x_offset, viewport_height - 100 + self.y_offset)
        elif self.position == OverlayPosition.BOTTOM_RIGHT:
            return Vector2D(viewport_width - 200 + self.x_offset, viewport_height - 100 + self.y_offset)
        elif self.position == OverlayPosition.CENTER:
            return Vector2D(viewport_width // 2 + self.x_offset, viewport_height // 2 + self.y_offset)
        else:  # CUSTOM
            return Vector2D(self.x_offset, self.y_offset)


@dataclass
class PerformanceMonitor(OverlayElement):
    """Performance monitoring overlay."""
    style: OverlayStyle = OverlayStyle.COMPACT
    update_interval: float = 0.5  # Update every 500ms
    history_length: int = 60  # Keep 60 samples
    
    # Performance data
    fps_history: List[float] = field(default_factory=list)
    frame_time_history: List[float] = field(default_factory=list)
    last_update: float = 0.0
    
    def update(self, fps: float, frame_time: float) -> None:
        """Update performance data."""
        current_time = time.time()
        
        if current_time - self.last_update >= self.update_interval:
            self.fps_history.append(fps)
            self.frame_time_history.append(frame_time * 1000)  # Convert to ms
            
            # Limit history
            if len(self.fps_history) > self.history_length:
                self.fps_history.pop(0)
            if len(self.frame_time_history) > self.history_length:
                self.frame_time_history.pop(0)
            
            self.last_update = current_time
    
    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self.fps_history:
            return {}
        
        return {
            "current_fps": self.fps_history[-1] if self.fps_history else 0.0,
            "avg_fps": sum(self.fps_history) / len(self.fps_history),
            "min_fps": min(self.fps_history),
            "max_fps": max(self.fps_history),
            "current_frame_time": self.frame_time_history[-1] if self.frame_time_history else 0.0,
            "avg_frame_time": sum(self.frame_time_history) / len(self.frame_time_history) if self.frame_time_history else 0.0
        }


@dataclass
class PhysicsDebugOverlay(OverlayElement):
    """Physics debugging overlay."""
    show_object_count: bool = True
    show_collision_count: bool = True
    show_constraint_count: bool = True
    show_world_info: bool = True
    show_performance: bool = True


@dataclass
class InputDebugOverlay(OverlayElement):
    """Input debugging overlay."""
    show_joystick_count: bool = True
    show_recent_events: bool = True
    event_history_length: int = 10
    recent_events: List[str] = field(default_factory=list)
    
    def add_input_event(self, event: InputEvent) -> None:
        """Add an input event to the history."""
        event_str = f"{event.event_type.value}: JS{event.joystick_id}"
        
        if event.button_id is not None:
            event_str += f" BTN{event.button_id}"
        elif event.axis_id is not None:
            event_str += f" AXIS{event.axis_id}={event.axis_value:.2f}"
        
        self.recent_events.append(event_str)
        
        # Limit history
        if len(self.recent_events) > self.event_history_length:
            self.recent_events.pop(0)


@dataclass
class CustomInfoPanel(OverlayElement):
    """Custom information panel."""
    title: str = "Info"
    info_lines: List[str] = field(default_factory=list)
    max_lines: int = 10
    
    def add_info(self, key: str, value: Any) -> None:
        """Add or update an info line."""
        info_str = f"{key}: {value}"
        
        # Remove existing line with same key
        self.info_lines = [line for line in self.info_lines if not line.startswith(f"{key}:")]
        
        # Add new line
        self.info_lines.append(info_str)
        
        # Limit lines
        if len(self.info_lines) > self.max_lines:
            self.info_lines.pop(0)
    
    def set_info(self, info_dict: Dict[str, Any]) -> None:
        """Set all info from dictionary."""
        self.info_lines.clear()
        for key, value in info_dict.items():
            self.add_info(key, value)


class OverlayManager:
    """
    Manager for UI overlays and debug information.
    
    Provides a centralized system for managing debug overlays,
    performance monitors, and interactive UI elements.
    """
    
    def __init__(self, viewport_width: int = 800, viewport_height: int = 600):
        """
        Initialize the overlay manager.
        
        Args:
            viewport_width: Viewport width in pixels
            viewport_height: Viewport height in pixels
        """
        self.logger = get_logger("overlay_manager")
        
        # Viewport dimensions
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        
        # Overlay elements
        self.elements: Dict[str, OverlayElement] = {}
        self.render_order: List[str] = []
        
        # Global settings
        self.global_visible = True
        self.global_alpha = 255
        
        # Built-in overlays
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.physics_debug: Optional[PhysicsDebugOverlay] = None
        self.input_debug: Optional[InputDebugOverlay] = None
        
        self.logger.debug("OverlayManager initialized", extra={
            "viewport": f"{viewport_width}x{viewport_height}"
        })
    
    def set_viewport_size(self, width: int, height: int) -> None:
        """Update viewport size."""
        self.viewport_width = width
        self.viewport_height = height
    
    def add_element(self, element: OverlayElement, z_order: int = 0) -> None:
        """
        Add an overlay element.
        
        Args:
            element: Overlay element to add
            z_order: Rendering order (higher values render on top)
        """
        self.elements[element.name] = element
        
        # Insert in render order based on z_order
        inserted = False
        for i, existing_name in enumerate(self.render_order):
            if existing_name in self.elements:
                # For now, just append (TODO: implement proper z-ordering)
                pass
        
        if not inserted:
            self.render_order.append(element.name)
        
        self.logger.debug("Overlay element added", extra={
            "name": element.name,
            "type": type(element).__name__
        })
    
    def remove_element(self, name: str) -> bool:
        """
        Remove an overlay element.
        
        Args:
            name: Element name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.elements:
            del self.elements[name]
            if name in self.render_order:
                self.render_order.remove(name)
            self.logger.debug("Overlay element removed", extra={"name": name})
            return True
        return False
    
    def get_element(self, name: str) -> Optional[OverlayElement]:
        """Get an overlay element by name."""
        return self.elements.get(name)
    
    def set_element_visible(self, name: str, visible: bool) -> None:
        """Set visibility of an element."""
        if name in self.elements:
            self.elements[name].visible = visible
    
    def toggle_element_visible(self, name: str) -> bool:
        """
        Toggle visibility of an element.
        
        Returns:
            New visibility state
        """
        if name in self.elements:
            self.elements[name].visible = not self.elements[name].visible
            return self.elements[name].visible
        return False
    
    def set_global_visible(self, visible: bool) -> None:
        """Set global visibility for all overlays."""
        self.global_visible = visible
    
    def toggle_global_visible(self) -> bool:
        """Toggle global visibility."""
        self.global_visible = not self.global_visible
        return self.global_visible
    
    def enable_performance_monitor(self, 
                                  position: OverlayPosition = OverlayPosition.TOP_LEFT,
                                  style: OverlayStyle = OverlayStyle.COMPACT) -> None:
        """Enable built-in performance monitor."""
        self.performance_monitor = PerformanceMonitor(
            name="performance_monitor",
            position=position,
            style=style
        )
        self.add_element(self.performance_monitor)
    
    def enable_physics_debug(self, 
                           position: OverlayPosition = OverlayPosition.TOP_RIGHT) -> None:
        """Enable built-in physics debug overlay."""
        self.physics_debug = PhysicsDebugOverlay(
            name="physics_debug",
            position=position
        )
        self.add_element(self.physics_debug)
    
    def enable_input_debug(self, 
                         position: OverlayPosition = OverlayPosition.BOTTOM_LEFT) -> None:
        """Enable built-in input debug overlay."""
        self.input_debug = InputDebugOverlay(
            name="input_debug",
            position=position
        )
        self.add_element(self.input_debug)
    
    def update(self, 
              fps: float = 0.0, 
              frame_time: float = 0.0,
              physics_world: Optional[PhysicsWorld] = None) -> None:
        """
        Update overlay data.
        
        Args:
            fps: Current FPS
            frame_time: Frame time in seconds
            physics_world: Physics world for debug info
        """
        # Update performance monitor
        if self.performance_monitor:
            self.performance_monitor.update(fps, frame_time)
        
        # Update physics debug info (data is pulled during render)
        
        # Other updates can be added here
    
    def handle_input_event(self, event: InputEvent) -> None:
        """Handle input events for debug overlays."""
        if self.input_debug:
            self.input_debug.add_input_event(event)
    
    def render(self, 
              renderer: RenderEngine,
              physics_world: Optional[PhysicsWorld] = None,
              joystick_manager: Optional[Any] = None) -> None:
        """
        Render all overlay elements.
        
        Args:
            renderer: Rendering engine
            physics_world: Physics world for debug info
            joystick_manager: Joystick manager for debug info
        """
        if not self.global_visible:
            return
        
        for element_name in self.render_order:
            element = self.elements.get(element_name)
            if not element or not element.visible:
                continue
            
            try:
                self._render_element(element, renderer, physics_world, joystick_manager)
            except Exception as e:
                self.logger.error("Error rendering overlay element", extra={
                    "element": element_name,
                    "error": str(e)
                })
    
    def _render_element(self, 
                       element: OverlayElement,
                       renderer: RenderEngine,
                       physics_world: Optional[PhysicsWorld],
                       joystick_manager: Optional[Any]) -> None:
        """Render a specific overlay element."""
        position = element.get_position(self.viewport_width, self.viewport_height)
        
        if isinstance(element, PerformanceMonitor):
            self._render_performance_monitor(element, renderer, position)
        elif isinstance(element, PhysicsDebugOverlay):
            self._render_physics_debug(element, renderer, position, physics_world)
        elif isinstance(element, InputDebugOverlay):
            self._render_input_debug(element, renderer, position, joystick_manager)
        elif isinstance(element, CustomInfoPanel):
            self._render_custom_info_panel(element, renderer, position)
    
    def _render_performance_monitor(self, 
                                   monitor: PerformanceMonitor,
                                   renderer: RenderEngine,
                                   position: Vector2D) -> None:
        """Render performance monitor."""
        stats = monitor.get_stats()
        if not stats:
            return
        
        y_offset = 0
        line_height = monitor.font_size + 2
        
        # Title
        if monitor.style != OverlayStyle.MINIMAL:
            renderer.draw_text(
                "Performance",
                Vector2D(position.x, position.y + y_offset),
                monitor.text_color,
                monitor.font_size + 2
            )
            y_offset += line_height + 5
        
        # FPS info
        if monitor.style == OverlayStyle.COMPACT:
            fps_text = f"FPS: {stats['current_fps']:.1f}"
            renderer.draw_text(
                fps_text,
                Vector2D(position.x, position.y + y_offset),
                monitor.text_color,
                monitor.font_size
            )
            y_offset += line_height
            
            frame_text = f"Frame: {stats['current_frame_time']:.1f}ms"
            renderer.draw_text(
                frame_text,
                Vector2D(position.x, position.y + y_offset),
                monitor.text_color,
                monitor.font_size
            )
        
        elif monitor.style == OverlayStyle.DETAILED:
            details = [
                f"FPS: {stats['current_fps']:.1f} (avg: {stats['avg_fps']:.1f})",
                f"Min: {stats['min_fps']:.1f}, Max: {stats['max_fps']:.1f}",
                f"Frame: {stats['current_frame_time']:.1f}ms",
                f"Avg Frame: {stats['avg_frame_time']:.1f}ms"
            ]
            
            for detail in details:
                renderer.draw_text(
                    detail,
                    Vector2D(position.x, position.y + y_offset),
                    monitor.text_color,
                    monitor.font_size
                )
                y_offset += line_height
    
    def _render_physics_debug(self, 
                             debug: PhysicsDebugOverlay,
                             renderer: RenderEngine,
                             position: Vector2D,
                             physics_world: Optional[PhysicsWorld]) -> None:
        """Render physics debug overlay."""
        if not physics_world:
            return
        
        y_offset = 0
        line_height = debug.font_size + 2
        
        # Title
        renderer.draw_text(
            "Physics Debug",
            Vector2D(position.x, position.y + y_offset),
            debug.text_color,
            debug.font_size + 2
        )
        y_offset += line_height + 5
        
        # Object count
        if debug.show_object_count:
            obj_text = f"Objects: {physics_world.get_object_count()}"
            renderer.draw_text(
                obj_text,
                Vector2D(position.x, position.y + y_offset),
                debug.text_color,
                debug.font_size
            )
            y_offset += line_height
        
        # Constraint count
        if debug.show_constraint_count:
            constraint_text = f"Constraints: {physics_world.constraint_manager.get_constraint_count()}"
            renderer.draw_text(
                constraint_text,
                Vector2D(position.x, position.y + y_offset),
                debug.text_color,
                debug.font_size
            )
            y_offset += line_height
        
        # World info
        if debug.show_world_info:
            gravity = physics_world.get_gravity()
            gravity_text = f"Gravity: ({gravity.x:.1f}, {gravity.y:.1f})"
            renderer.draw_text(
                gravity_text,
                Vector2D(position.x, position.y + y_offset),
                debug.text_color,
                debug.font_size
            )
            y_offset += line_height
            
            state_text = f"State: {'Running' if physics_world.is_running() else 'Stopped'}"
            if physics_world.is_paused():
                state_text = "State: Paused"
            
            renderer.draw_text(
                state_text,
                Vector2D(position.x, position.y + y_offset),
                debug.text_color,
                debug.font_size
            )
            y_offset += line_height
        
        # Performance
        if debug.show_performance:
            stats = physics_world.engine.get_stats()
            perf_text = f"Physics FPS: {stats.fps:.1f}"
            renderer.draw_text(
                perf_text,
                Vector2D(position.x, position.y + y_offset),
                debug.text_color,
                debug.font_size
            )
    
    def _render_input_debug(self, 
                           debug: InputDebugOverlay,
                           renderer: RenderEngine,
                           position: Vector2D,
                           joystick_manager: Optional[Any]) -> None:
        """Render input debug overlay."""
        y_offset = 0
        line_height = debug.font_size + 2
        
        # Title
        renderer.draw_text(
            "Input Debug",
            Vector2D(position.x, position.y + y_offset),
            debug.text_color,
            debug.font_size + 2
        )
        y_offset += line_height + 5
        
        # Joystick count
        if debug.show_joystick_count and joystick_manager:
            js_count = joystick_manager.get_joystick_count()
            js_text = f"Joysticks: {js_count}"
            renderer.draw_text(
                js_text,
                Vector2D(position.x, position.y + y_offset),
                debug.text_color,
                debug.font_size
            )
            y_offset += line_height
        
        # Recent events
        if debug.show_recent_events:
            renderer.draw_text(
                "Recent Events:",
                Vector2D(position.x, position.y + y_offset),
                debug.text_color,
                debug.font_size
            )
            y_offset += line_height
            
            for event_str in debug.recent_events[-5:]:  # Show last 5 events
                renderer.draw_text(
                    f"  {event_str}",
                    Vector2D(position.x, position.y + y_offset),
                    debug.text_color.with_alpha(200),  # Slightly transparent
                    debug.font_size - 2
                )
                y_offset += line_height - 2
    
    def _render_custom_info_panel(self, 
                                 panel: CustomInfoPanel,
                                 renderer: RenderEngine,
                                 position: Vector2D) -> None:
        """Render custom info panel."""
        y_offset = 0
        line_height = panel.font_size + 2
        
        # Title
        renderer.draw_text(
            panel.title,
            Vector2D(position.x, position.y + y_offset),
            panel.text_color,
            panel.font_size + 2
        )
        y_offset += line_height + 5
        
        # Info lines
        for info_line in panel.info_lines:
            renderer.draw_text(
                info_line,
                Vector2D(position.x, position.y + y_offset),
                panel.text_color,
                panel.font_size
            )
            y_offset += line_height
    
    def create_custom_panel(self, 
                           name: str,
                           title: str,
                           position: OverlayPosition = OverlayPosition.BOTTOM_RIGHT) -> CustomInfoPanel:
        """
        Create a custom info panel.
        
        Args:
            name: Panel name
            title: Panel title
            position: Panel position
            
        Returns:
            Created custom info panel
        """
        panel = CustomInfoPanel(
            name=name,
            position=position,
            title=title
        )
        self.add_element(panel)
        return panel
    
    def handle_key_event(self, key: int) -> bool:
        """
        Handle keyboard events for overlay controls.
        
        Args:
            key: Pygame key code
            
        Returns:
            True if event was handled, False otherwise
        """
        # F1 - Toggle all overlays
        if key == pygame.K_F1:
            self.toggle_global_visible()
            return True
        
        # F2 - Toggle performance monitor
        elif key == pygame.K_F2:
            if self.performance_monitor:
                self.toggle_element_visible("performance_monitor")
            else:
                self.enable_performance_monitor()
            return True
        
        # F3 - Toggle physics debug
        elif key == pygame.K_F3:
            if self.physics_debug:
                self.toggle_element_visible("physics_debug")
            else:
                self.enable_physics_debug()
            return True
        
        # F4 - Toggle input debug
        elif key == pygame.K_F4:
            if self.input_debug:
                self.toggle_element_visible("input_debug")
            else:
                self.enable_input_debug()
            return True
        
        return False