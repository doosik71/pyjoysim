"""
Camera system for PyJoySim rendering.

This module provides camera functionality for controlling the view
and transforming between world and screen coordinates.
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

from ..physics import Vector2D, PhysicsObject
from ..core.logging import get_logger


@dataclass
class CameraBounds:
    """Bounds for camera movement."""
    min_x: float = float('-inf')
    max_x: float = float('inf')
    min_y: float = float('-inf')
    max_y: float = float('inf')
    
    def clamp_position(self, position: Vector2D) -> Vector2D:
        """Clamp position to bounds."""
        return Vector2D(
            max(self.min_x, min(self.max_x, position.x)),
            max(self.min_y, min(self.max_y, position.y))
        )


class Camera2D:
    """
    2D camera for viewport control and coordinate transformation.
    
    Provides functionality for panning, zooming, and following objects.
    """
    
    def __init__(self, 
                 viewport_width: int = 800, 
                 viewport_height: int = 600):
        """
        Initialize the camera.
        
        Args:
            viewport_width: Width of the viewport in pixels
            viewport_height: Height of the viewport in pixels
        """
        self.logger = get_logger("camera_2d")
        
        # Viewport dimensions
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        
        # Camera transform
        self.position = Vector2D(0, 0)
        self.zoom = 1.0
        self.rotation = 0.0  # In radians
        
        # Camera constraints
        self.min_zoom = 0.1
        self.max_zoom = 10.0
        self.bounds: Optional[CameraBounds] = None
        
        # Following system
        self.follow_target: Optional[PhysicsObject] = None
        self.follow_smoothing = 0.1  # 0 = instant, 1 = no movement
        self.follow_offset = Vector2D(0, 0)
        
        # Shake effect
        self.shake_intensity = 0.0
        self.shake_duration = 0.0
        self.shake_time_remaining = 0.0
        self._shake_offset = Vector2D(0, 0)
        
        self.logger.debug("Camera2D created", extra={
            "viewport": f"{viewport_width}x{viewport_height}"
        })
    
    def set_viewport_size(self, width: int, height: int) -> None:
        """
        Set the viewport size.
        
        Args:
            width: New viewport width
            height: New viewport height
        """
        self.viewport_width = width
        self.viewport_height = height
        
        self.logger.debug("Viewport size updated", extra={
            "width": width,
            "height": height
        })
    
    def set_position(self, position: Vector2D) -> None:
        """
        Set camera position.
        
        Args:
            position: New camera position in world coordinates
        """
        if self.bounds:
            position = self.bounds.clamp_position(position)
        
        self.position = position
    
    def move(self, offset: Vector2D) -> None:
        """
        Move camera by offset.
        
        Args:
            offset: Movement offset in world coordinates
        """
        new_position = self.position + offset
        self.set_position(new_position)
    
    def set_zoom(self, zoom: float) -> None:
        """
        Set camera zoom level.
        
        Args:
            zoom: Zoom level (1.0 = normal, >1.0 = zoomed in, <1.0 = zoomed out)
        """
        self.zoom = max(self.min_zoom, min(self.max_zoom, zoom))
    
    def zoom_by(self, factor: float) -> None:
        """
        Zoom by a factor.
        
        Args:
            factor: Zoom factor to multiply current zoom by
        """
        self.set_zoom(self.zoom * factor)
    
    def set_rotation(self, rotation: float) -> None:
        """
        Set camera rotation.
        
        Args:
            rotation: Rotation angle in radians
        """
        self.rotation = rotation
    
    def rotate_by(self, angle: float) -> None:
        """
        Rotate camera by angle.
        
        Args:
            angle: Rotation angle in radians
        """
        self.rotation += angle
    
    def set_bounds(self, bounds: Optional[CameraBounds]) -> None:
        """
        Set camera movement bounds.
        
        Args:
            bounds: Camera bounds or None to remove bounds
        """
        self.bounds = bounds
        
        # Clamp current position to new bounds
        if self.bounds:
            self.position = self.bounds.clamp_position(self.position)
    
    def follow_object(self, 
                     target: Optional[PhysicsObject], 
                     smoothing: float = 0.1,
                     offset: Vector2D = Vector2D(0, 0)) -> None:
        """
        Set camera to follow a physics object.
        
        Args:
            target: Object to follow (None to stop following)
            smoothing: Follow smoothing factor (0 = instant, 1 = no movement)
            offset: Offset from target position
        """
        self.follow_target = target
        self.follow_smoothing = max(0.0, min(1.0, smoothing))
        self.follow_offset = offset
        
        if target:
            self.logger.debug("Camera following object", extra={
                "target": target.name,
                "smoothing": smoothing,
                "offset": offset.to_tuple()
            })
        else:
            self.logger.debug("Camera stopped following")
    
    def add_shake(self, intensity: float, duration: float) -> None:
        """
        Add camera shake effect.
        
        Args:
            intensity: Shake intensity (pixels)
            duration: Shake duration (seconds)
        """
        self.shake_intensity = max(self.shake_intensity, intensity)
        self.shake_duration = duration
        self.shake_time_remaining = duration
        
        self.logger.debug("Camera shake added", extra={
            "intensity": intensity,
            "duration": duration
        })
    
    def update(self, dt: float) -> None:
        """
        Update camera state.
        
        Args:
            dt: Time delta in seconds
        """
        # Update following
        if self.follow_target and self.follow_target.is_active:
            target_pos = self.follow_target.position + self.follow_offset
            
            if self.follow_smoothing > 0:
                # Smooth following
                diff = target_pos - self.position
                move_amount = diff * (1.0 - self.follow_smoothing) * dt * 10.0  # Scale for 60fps
                self.move(move_amount)
            else:
                # Instant following
                self.set_position(target_pos)
        
        # Update shake effect
        if self.shake_time_remaining > 0:
            self.shake_time_remaining = max(0, self.shake_time_remaining - dt)
            
            if self.shake_time_remaining > 0:
                # Calculate shake offset
                shake_factor = self.shake_time_remaining / self.shake_duration
                current_intensity = self.shake_intensity * shake_factor
                
                # Random shake offset
                import random
                angle = random.uniform(0, 2 * math.pi)
                magnitude = random.uniform(0, current_intensity)
                
                self._shake_offset = Vector2D(
                    math.cos(angle) * magnitude,
                    math.sin(angle) * magnitude
                )
            else:
                self._shake_offset = Vector2D(0, 0)
    
    def world_to_screen(self, world_pos: Vector2D) -> Vector2D:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            world_pos: Position in world coordinates
            
        Returns:
            Position in screen coordinates
        """
        # Apply camera transform
        relative_pos = world_pos - self.position
        
        # Apply rotation if needed
        if self.rotation != 0:
            cos_rot = math.cos(-self.rotation)
            sin_rot = math.sin(-self.rotation)
            
            rotated_x = relative_pos.x * cos_rot - relative_pos.y * sin_rot
            rotated_y = relative_pos.x * sin_rot + relative_pos.y * cos_rot
            
            relative_pos = Vector2D(rotated_x, rotated_y)
        
        # Apply zoom
        screen_pos = relative_pos * self.zoom
        
        # Convert to screen space (center origin, flip Y)
        screen_x = screen_pos.x + self.viewport_width // 2
        screen_y = self.viewport_height // 2 - screen_pos.y
        
        # Apply shake offset
        screen_x += self._shake_offset.x
        screen_y += self._shake_offset.y
        
        return Vector2D(screen_x, screen_y)
    
    def screen_to_world(self, screen_pos: Vector2D) -> Vector2D:
        """
        Convert screen coordinates to world coordinates.
        
        Args:
            screen_pos: Position in screen coordinates
            
        Returns:
            Position in world coordinates
        """
        # Remove shake offset
        adjusted_screen_x = screen_pos.x - self._shake_offset.x
        adjusted_screen_y = screen_pos.y - self._shake_offset.y
        
        # Convert from screen space
        relative_x = adjusted_screen_x - self.viewport_width // 2
        relative_y = self.viewport_height // 2 - adjusted_screen_y
        
        relative_pos = Vector2D(relative_x, relative_y)
        
        # Remove zoom
        relative_pos = relative_pos / self.zoom
        
        # Remove rotation if needed
        if self.rotation != 0:
            cos_rot = math.cos(self.rotation)
            sin_rot = math.sin(self.rotation)
            
            unrotated_x = relative_pos.x * cos_rot - relative_pos.y * sin_rot
            unrotated_y = relative_pos.x * sin_rot + relative_pos.y * cos_rot
            
            relative_pos = Vector2D(unrotated_x, unrotated_y)
        
        # Add camera position
        world_pos = relative_pos + self.position
        
        return world_pos
    
    def get_world_bounds(self) -> Tuple[Vector2D, Vector2D]:
        """
        Get the world-space bounds of the current view.
        
        Returns:
            Tuple of (top_left, bottom_right) world coordinates
        """
        top_left_screen = Vector2D(0, 0)
        bottom_right_screen = Vector2D(self.viewport_width, self.viewport_height)
        
        top_left_world = self.screen_to_world(top_left_screen)
        bottom_right_world = self.screen_to_world(bottom_right_screen)
        
        return (top_left_world, bottom_right_world)
    
    def is_point_visible(self, world_pos: Vector2D, margin: float = 0) -> bool:
        """
        Check if a world point is visible in the current view.
        
        Args:
            world_pos: Point in world coordinates
            margin: Additional margin around viewport (in world units)
            
        Returns:
            True if point is visible, False otherwise
        """
        screen_pos = self.world_to_screen(world_pos)
        
        return (
            -margin <= screen_pos.x <= self.viewport_width + margin and
            -margin <= screen_pos.y <= self.viewport_height + margin
        )
    
    def is_circle_visible(self, center: Vector2D, radius: float) -> bool:
        """
        Check if a circle is visible in the current view.
        
        Args:
            center: Circle center in world coordinates
            radius: Circle radius in world units
            
        Returns:
            True if circle is visible, False otherwise
        """
        return self.is_point_visible(center, radius * self.zoom)
    
    def is_rectangle_visible(self, position: Vector2D, width: float, height: float) -> bool:
        """
        Check if a rectangle is visible in the current view.
        
        Args:
            position: Rectangle center in world coordinates
            width: Rectangle width in world units
            height: Rectangle height in world units
            
        Returns:
            True if rectangle is visible, False otherwise
        """
        half_width = width * 0.5
        half_height = height * 0.5
        
        # Check if any corner is visible
        corners = [
            Vector2D(position.x - half_width, position.y - half_height),
            Vector2D(position.x + half_width, position.y - half_height),
            Vector2D(position.x + half_width, position.y + half_height),
            Vector2D(position.x - half_width, position.y + half_height)
        ]
        
        for corner in corners:
            if self.is_point_visible(corner):
                return True
        
        # Check if viewport is inside rectangle
        top_left, bottom_right = self.get_world_bounds()
        viewport_center = Vector2D(
            (top_left.x + bottom_right.x) * 0.5,
            (top_left.y + bottom_right.y) * 0.5
        )
        
        return (
            abs(viewport_center.x - position.x) <= half_width and
            abs(viewport_center.y - position.y) <= half_height
        )
    
    def zoom_to_fit(self, world_bounds: Tuple[Vector2D, Vector2D], margin: float = 0.1) -> None:
        """
        Zoom camera to fit world bounds in view.
        
        Args:
            world_bounds: Tuple of (min_point, max_point) in world coordinates
            margin: Margin factor (0.1 = 10% margin)
        """
        min_point, max_point = world_bounds
        
        # Calculate bounds size
        bounds_width = max_point.x - min_point.x
        bounds_height = max_point.y - min_point.y
        
        if bounds_width <= 0 or bounds_height <= 0:
            return
        
        # Calculate required zoom to fit bounds
        zoom_x = self.viewport_width / bounds_width
        zoom_y = self.viewport_height / bounds_height
        required_zoom = min(zoom_x, zoom_y) * (1.0 - margin)
        
        # Set zoom and center on bounds
        self.set_zoom(required_zoom)
        center = Vector2D(
            (min_point.x + max_point.x) * 0.5,
            (min_point.y + max_point.y) * 0.5
        )
        self.set_position(center)
        
        self.logger.debug("Camera zoomed to fit", extra={
            "bounds": (min_point.to_tuple(), max_point.to_tuple()),
            "zoom": required_zoom,
            "center": center.to_tuple()
        })
    
    def zoom_to_object(self, obj: PhysicsObject, margin: float = 0.2) -> None:
        """
        Zoom camera to focus on a specific object.
        
        Args:
            obj: Object to focus on
            margin: Margin factor around object
        """
        # For now, use a simple approach with estimated object bounds
        # TODO: Get actual object bounds from colliders
        estimated_size = 5.0  # Default object size
        
        min_point = Vector2D(
            obj.position.x - estimated_size,
            obj.position.y - estimated_size
        )
        max_point = Vector2D(
            obj.position.x + estimated_size,
            obj.position.y + estimated_size
        )
        
        self.zoom_to_fit((min_point, max_point), margin)
    
    def get_view_matrix(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get 2D transformation matrix for the current view.
        
        Returns:
            Tuple of (a, b, c, d, e, f) for 2D transformation matrix
        """
        # This is a simplified 2D transformation matrix
        # Format: [a c e]
        #         [b d f]
        #         [0 0 1]
        
        cos_rot = math.cos(-self.rotation)
        sin_rot = math.sin(-self.rotation)
        
        # Scale and rotation
        a = self.zoom * cos_rot
        b = self.zoom * sin_rot
        c = -self.zoom * sin_rot
        d = self.zoom * cos_rot
        
        # Translation (including shake)
        e = -self.position.x * self.zoom + self.viewport_width // 2 + self._shake_offset.x
        f = self.position.y * self.zoom + self.viewport_height // 2 + self._shake_offset.y
        
        return (a, b, c, d, e, f)


class CameraController:
    """
    High-level camera controller with common behaviors.
    """
    
    def __init__(self, camera: Camera2D):
        """
        Initialize camera controller.
        
        Args:
            camera: Camera to control
        """
        self.camera = camera
        self.logger = get_logger("camera_controller")
        
        # Movement settings
        self.pan_speed = 100.0  # pixels per second
        self.zoom_speed = 2.0   # zoom factor per second
        self.rotation_speed = math.pi  # radians per second
        
        # Input state
        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False
        self.zoom_in = False
        self.zoom_out = False
        self.rotate_left = False
        self.rotate_right = False
    
    def update(self, dt: float) -> None:
        """
        Update camera based on current input state.
        
        Args:
            dt: Time delta in seconds
        """
        # Handle movement
        move_x = 0.0
        move_y = 0.0
        
        if self.move_left:
            move_x -= self.pan_speed * dt
        if self.move_right:
            move_x += self.pan_speed * dt
        if self.move_up:
            move_y += self.pan_speed * dt
        if self.move_down:
            move_y -= self.pan_speed * dt
        
        if move_x != 0 or move_y != 0:
            # Convert screen movement to world movement
            world_movement = Vector2D(move_x, move_y) / self.camera.zoom
            self.camera.move(world_movement)
        
        # Handle zoom
        if self.zoom_in:
            self.camera.zoom_by(1.0 + self.zoom_speed * dt)
        if self.zoom_out:
            self.camera.zoom_by(1.0 - self.zoom_speed * dt)
        
        # Handle rotation
        if self.rotate_left:
            self.camera.rotate_by(-self.rotation_speed * dt)
        if self.rotate_right:
            self.camera.rotate_by(self.rotation_speed * dt)
    
    def reset_to_origin(self) -> None:
        """Reset camera to origin with default settings."""
        self.camera.set_position(Vector2D(0, 0))
        self.camera.set_zoom(1.0)
        self.camera.set_rotation(0.0)
        self.camera.follow_object(None)
        
        self.logger.debug("Camera reset to origin")
    
    def set_movement_speed(self, pan_speed: float, zoom_speed: float, rotation_speed: float) -> None:
        """
        Set movement speeds.
        
        Args:
            pan_speed: Pan speed in pixels per second
            zoom_speed: Zoom speed factor per second
            rotation_speed: Rotation speed in radians per second
        """
        self.pan_speed = max(0, pan_speed)
        self.zoom_speed = max(0, zoom_speed)
        self.rotation_speed = max(0, rotation_speed)