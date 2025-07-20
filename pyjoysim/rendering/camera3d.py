"""
3D Camera system for PyJoySim rendering.

This module provides 3D camera functionality with support for multiple
camera modes including free camera, follow camera, and fixed cameras.
"""

import math
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
try:
    import pyrr
    from pyrr import Matrix44, Vector3, Vector4
except ImportError:
    # Fallback to numpy if pyrr is not available
    import numpy as np
    Matrix44 = np.ndarray
    Vector3 = np.ndarray
    Vector4 = np.ndarray

from ..core.logging import get_logger


class CameraMode(Enum):
    """3D Camera modes."""
    FREE = "free"           # Free camera movement
    FOLLOW = "follow"       # Follow target object
    FIXED = "fixed"         # Fixed position and orientation
    FIRST_PERSON = "first_person"   # First person view from target
    THIRD_PERSON = "third_person"   # Third person view of target
    ORBIT = "orbit"         # Orbit around target


@dataclass
class Camera3DSettings:
    """Settings for 3D camera behavior."""
    # Movement settings
    move_speed: float = 10.0        # Units per second
    rotation_speed: float = 2.0     # Radians per second
    zoom_speed: float = 5.0         # Units per second
    
    # Mouse sensitivity
    mouse_sensitivity: float = 0.005
    
    # Field of view
    fov: float = math.pi / 3        # 60 degrees
    near_plane: float = 0.1
    far_plane: float = 1000.0
    
    # Follow camera settings
    follow_distance: float = 10.0
    follow_height: float = 5.0
    follow_smoothing: float = 0.1   # 0 = instant, 1 = no movement
    
    # Orbit camera settings
    orbit_distance: float = 15.0
    orbit_height: float = 10.0
    orbit_speed: float = 1.0
    
    # Third person settings
    third_person_distance: float = 8.0
    third_person_height: float = 3.0
    third_person_offset: Vector3 = None
    
    def __post_init__(self):
        if self.third_person_offset is None:
            self.third_person_offset = np.array([0.0, 0.0, 0.0])


class Camera3D:
    """
    3D camera with multiple modes and smooth transitions.
    
    Supports free camera, follow camera, first/third person views,
    and orbital camera modes with smooth interpolation.
    """
    
    def __init__(self, 
                 viewport_width: int = 800, 
                 viewport_height: int = 600,
                 settings: Optional[Camera3DSettings] = None):
        """
        Initialize the 3D camera.
        
        Args:
            viewport_width: Width of the viewport in pixels
            viewport_height: Height of the viewport in pixels
            settings: Camera behavior settings
        """
        self.logger = get_logger("camera_3d")
        
        # Viewport dimensions
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.aspect_ratio = viewport_width / viewport_height
        
        # Camera settings
        self.settings = settings or Camera3DSettings()
        
        # Camera transform
        self.position = np.array([0.0, 5.0, 10.0])  # Start behind and above origin
        self.rotation = np.array([0.0, 0.0, 0.0])   # Euler angles (pitch, yaw, roll)
        self.up_vector = np.array([0.0, 1.0, 0.0])
        
        # Derived vectors (calculated from rotation)
        self.forward_vector = np.array([0.0, 0.0, -1.0])
        self.right_vector = np.array([1.0, 0.0, 0.0])
        self._update_vectors()
        
        # Camera mode and target
        self.mode = CameraMode.FREE
        self.target_object = None
        self.target_position = np.array([0.0, 0.0, 0.0])
        
        # Orbit mode state
        self.orbit_angle = 0.0
        self.orbit_elevation = 0.3  # radians above horizontal
        
        # Smooth movement state
        self._target_position = self.position.copy()
        self._target_rotation = self.rotation.copy()
        
        # Input state for free camera
        self.move_forward = False
        self.move_backward = False
        self.move_left = False
        self.move_right = False
        self.move_up = False
        self.move_down = False
        
        self.logger.debug("Camera3D created", extra={
            "viewport": f"{viewport_width}x{viewport_height}",
            "mode": self.mode.value
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
        self.aspect_ratio = width / height
        
        self.logger.debug("Viewport size updated", extra={
            "width": width,
            "height": height,
            "aspect_ratio": self.aspect_ratio
        })
    
    def set_mode(self, mode: CameraMode, target: Optional[object] = None) -> None:
        """
        Set camera mode.
        
        Args:
            mode: New camera mode
            target: Target object for follow/orbit modes (required for those modes)
        """
        if mode in [CameraMode.FOLLOW, CameraMode.ORBIT, CameraMode.FIRST_PERSON, 
                   CameraMode.THIRD_PERSON] and target is None:
            self.logger.warning("Target required for camera mode", extra={"mode": mode.value})
            return
        
        self.mode = mode
        self.target_object = target
        
        # Reset orbit angle when switching to orbit mode
        if mode == CameraMode.ORBIT:
            self.orbit_angle = 0.0
        
        self.logger.debug("Camera mode changed", extra={
            "mode": mode.value,
            "has_target": target is not None
        })
    
    def set_position(self, position: np.ndarray) -> None:
        """
        Set camera position.
        
        Args:
            position: New camera position as [x, y, z]
        """
        self.position = np.array(position)
        self._target_position = self.position.copy()
    
    def set_rotation(self, rotation: np.ndarray) -> None:
        """
        Set camera rotation.
        
        Args:
            rotation: Euler angles [pitch, yaw, roll] in radians
        """
        self.rotation = np.array(rotation)
        self._target_rotation = self.rotation.copy()
        self._update_vectors()
    
    def look_at(self, target: np.ndarray, up: Optional[np.ndarray] = None) -> None:
        """
        Point camera to look at a target position.
        
        Args:
            target: Target position as [x, y, z]
            up: Up vector (defaults to world up)
        """
        if up is None:
            up = np.array([0.0, 1.0, 0.0])
        
        # Calculate direction from camera to target
        direction = target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:
            return  # Target is too close
        
        direction = direction / distance
        
        # Calculate pitch (rotation around X axis)
        pitch = math.asin(-direction[1])
        
        # Calculate yaw (rotation around Y axis)
        yaw = math.atan2(direction[0], -direction[2])
        
        # Keep roll at 0 for now
        roll = 0.0
        
        self.set_rotation(np.array([pitch, yaw, roll]))
    
    def move(self, offset: np.ndarray) -> None:
        """
        Move camera by offset in world coordinates.
        
        Args:
            offset: Movement offset as [x, y, z]
        """
        self.position += offset
        self._target_position = self.position.copy()
    
    def move_local(self, offset: np.ndarray) -> None:
        """
        Move camera by offset in local coordinates.
        
        Args:
            offset: Movement offset in local space [forward, right, up]
        """
        world_offset = (
            self.forward_vector * offset[0] +
            self.right_vector * offset[1] +
            self.up_vector * offset[2]
        )
        self.move(world_offset)
    
    def rotate(self, delta_rotation: np.ndarray) -> None:
        """
        Rotate camera by delta rotation.
        
        Args:
            delta_rotation: Rotation delta [pitch, yaw, roll] in radians
        """
        self.rotation += delta_rotation
        
        # Clamp pitch to prevent camera flipping
        self.rotation[0] = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.rotation[0]))
        
        self._target_rotation = self.rotation.copy()
        self._update_vectors()
    
    def _update_vectors(self) -> None:
        """Update forward, right, and up vectors from rotation."""
        pitch, yaw, roll = self.rotation
        
        # Calculate forward vector
        self.forward_vector = np.array([
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
            -math.cos(yaw) * math.cos(pitch)
        ])
        
        # Calculate right vector
        world_up = np.array([0.0, 1.0, 0.0])
        self.right_vector = np.cross(self.forward_vector, world_up)
        self.right_vector = self.right_vector / np.linalg.norm(self.right_vector)
        
        # Calculate up vector
        self.up_vector = np.cross(self.right_vector, self.forward_vector)
        self.up_vector = self.up_vector / np.linalg.norm(self.up_vector)
    
    def update(self, dt: float) -> None:
        """
        Update camera state based on current mode.
        
        Args:
            dt: Time delta in seconds
        """
        if self.mode == CameraMode.FREE:
            self._update_free_camera(dt)
        elif self.mode == CameraMode.FOLLOW:
            self._update_follow_camera(dt)
        elif self.mode == CameraMode.ORBIT:
            self._update_orbit_camera(dt)
        elif self.mode == CameraMode.FIRST_PERSON:
            self._update_first_person_camera(dt)
        elif self.mode == CameraMode.THIRD_PERSON:
            self._update_third_person_camera(dt)
        
        # Apply smooth movement if enabled
        if self.settings.follow_smoothing > 0 and self.mode != CameraMode.FREE:
            self._apply_smooth_movement(dt)
    
    def _update_free_camera(self, dt: float) -> None:
        """Update free camera based on input state."""
        # Handle movement input
        local_movement = np.array([0.0, 0.0, 0.0])
        
        if self.move_forward:
            local_movement[0] += self.settings.move_speed * dt
        if self.move_backward:
            local_movement[0] -= self.settings.move_speed * dt
        if self.move_right:
            local_movement[1] += self.settings.move_speed * dt
        if self.move_left:
            local_movement[1] -= self.settings.move_speed * dt
        if self.move_up:
            local_movement[2] += self.settings.move_speed * dt
        if self.move_down:
            local_movement[2] -= self.settings.move_speed * dt
        
        if np.any(local_movement != 0):
            self.move_local(local_movement)
    
    def _update_follow_camera(self, dt: float) -> None:
        """Update follow camera to track target."""
        if not self.target_object:
            return
        
        # Get target position (assuming target has position attribute)
        if hasattr(self.target_object, 'position'):
            if hasattr(self.target_object.position, 'x'):
                # Vector2D or similar
                target_pos = np.array([
                    self.target_object.position.x,
                    self.settings.follow_height,
                    self.target_object.position.y
                ])
            else:
                # Already a numpy array or similar
                target_pos = np.array(self.target_object.position)
        else:
            self.logger.warning("Target object has no position attribute")
            return
        
        # Calculate camera position behind target
        if hasattr(self.target_object, 'rotation'):
            # Use target's rotation to position camera behind it
            target_rotation = getattr(self.target_object, 'rotation', 0)
            offset_x = -math.sin(target_rotation) * self.settings.follow_distance
            offset_z = math.cos(target_rotation) * self.settings.follow_distance
        else:
            # Default to being behind in Z direction
            offset_x = 0
            offset_z = self.settings.follow_distance
        
        desired_position = target_pos + np.array([offset_x, self.settings.follow_height, offset_z])
        
        # Set target position for smooth movement
        self._target_position = desired_position
        
        # Look at target
        self.look_at(target_pos)
    
    def _update_orbit_camera(self, dt: float) -> None:
        """Update orbit camera to circle around target."""
        if not self.target_object:
            return
        
        # Update orbit angle
        self.orbit_angle += self.settings.orbit_speed * dt
        
        # Get target position
        if hasattr(self.target_object, 'position'):
            if hasattr(self.target_object.position, 'x'):
                target_pos = np.array([
                    self.target_object.position.x,
                    0.0,
                    self.target_object.position.y
                ])
            else:
                target_pos = np.array(self.target_object.position)
        else:
            target_pos = np.array([0.0, 0.0, 0.0])
        
        # Calculate orbit position
        orbit_x = math.cos(self.orbit_angle) * self.settings.orbit_distance
        orbit_z = math.sin(self.orbit_angle) * self.settings.orbit_distance
        orbit_y = self.settings.orbit_height
        
        desired_position = target_pos + np.array([orbit_x, orbit_y, orbit_z])
        
        # Set target position for smooth movement
        self._target_position = desired_position
        
        # Look at target
        self.look_at(target_pos)
    
    def _update_first_person_camera(self, dt: float) -> None:
        """Update first person camera to match target position."""
        if not self.target_object:
            return
        
        # Position camera at target position
        if hasattr(self.target_object, 'position'):
            if hasattr(self.target_object.position, 'x'):
                self._target_position = np.array([
                    self.target_object.position.x,
                    1.8,  # Eye height
                    self.target_object.position.y
                ])
            else:
                self._target_position = np.array(self.target_object.position)
                self._target_position[1] += 1.8  # Add eye height
        
        # Match target rotation if available
        if hasattr(self.target_object, 'rotation'):
            target_rotation = getattr(self.target_object, 'rotation', 0)
            self._target_rotation = np.array([0, target_rotation, 0])
    
    def _update_third_person_camera(self, dt: float) -> None:
        """Update third person camera to follow behind target."""
        if not self.target_object:
            return
        
        # Get target position and rotation
        if hasattr(self.target_object, 'position'):
            if hasattr(self.target_object.position, 'x'):
                target_pos = np.array([
                    self.target_object.position.x,
                    0.0,
                    self.target_object.position.y
                ])
            else:
                target_pos = np.array(self.target_object.position)
        else:
            target_pos = np.array([0.0, 0.0, 0.0])
        
        # Calculate camera position behind and above target
        if hasattr(self.target_object, 'rotation'):
            target_rotation = getattr(self.target_object, 'rotation', 0)
            offset_x = -math.sin(target_rotation) * self.settings.third_person_distance
            offset_z = math.cos(target_rotation) * self.settings.third_person_distance
        else:
            offset_x = 0
            offset_z = self.settings.third_person_distance
        
        desired_position = target_pos + np.array([
            offset_x + self.settings.third_person_offset[0],
            self.settings.third_person_height + self.settings.third_person_offset[1],
            offset_z + self.settings.third_person_offset[2]
        ])
        
        self._target_position = desired_position
        
        # Look at target
        look_at_target = target_pos + np.array([0, 1.0, 0])  # Look slightly above target
        self.look_at(look_at_target)
    
    def _apply_smooth_movement(self, dt: float) -> None:
        """Apply smooth interpolation to camera position and rotation."""
        if self.settings.follow_smoothing <= 0:
            # Instant movement
            self.position = self._target_position.copy()
            self.rotation = self._target_rotation.copy()
            self._update_vectors()
            return
        
        # Smooth movement
        move_factor = (1.0 - self.settings.follow_smoothing) * dt * 10.0
        
        # Interpolate position
        pos_diff = self._target_position - self.position
        if np.linalg.norm(pos_diff) > 0.01:  # Only move if significant difference
            self.position += pos_diff * move_factor
        
        # Interpolate rotation
        rot_diff = self._target_rotation - self.rotation
        
        # Handle angle wrapping for yaw
        if rot_diff[1] > math.pi:
            rot_diff[1] -= 2 * math.pi
        elif rot_diff[1] < -math.pi:
            rot_diff[1] += 2 * math.pi
        
        if np.linalg.norm(rot_diff) > 0.01:  # Only rotate if significant difference
            self.rotation += rot_diff * move_factor
            self._update_vectors()
    
    def handle_mouse_movement(self, delta_x: float, delta_y: float) -> None:
        """
        Handle mouse movement for camera rotation.
        
        Args:
            delta_x: Mouse movement in X direction
            delta_y: Mouse movement in Y direction
        """
        if self.mode == CameraMode.FREE:
            # Apply mouse sensitivity
            yaw_delta = -delta_x * self.settings.mouse_sensitivity
            pitch_delta = -delta_y * self.settings.mouse_sensitivity
            
            self.rotate(np.array([pitch_delta, yaw_delta, 0]))
    
    def handle_scroll(self, delta: float) -> None:
        """
        Handle mouse scroll for zoom/movement.
        
        Args:
            delta: Scroll delta
        """
        if self.mode == CameraMode.FREE:
            # Move forward/backward
            movement = self.forward_vector * delta * self.settings.zoom_speed
            self.move(movement)
        elif self.mode == CameraMode.ORBIT:
            # Adjust orbit distance
            self.settings.orbit_distance = max(1.0, 
                self.settings.orbit_distance - delta * self.settings.zoom_speed)
        elif self.mode == CameraMode.THIRD_PERSON:
            # Adjust third person distance
            self.settings.third_person_distance = max(1.0,
                self.settings.third_person_distance - delta * self.settings.zoom_speed)
    
    def get_view_matrix(self) -> np.ndarray:
        """
        Get the view matrix for rendering.
        
        Returns:
            4x4 view matrix
        """
        try:
            # Use pyrr if available for better performance
            eye = Vector3(self.position)
            target = Vector3(self.position + self.forward_vector)
            up = Vector3(self.up_vector)
            return pyrr.matrix44.create_look_at(eye, target, up)
        except (NameError, AttributeError):
            # Fallback to manual calculation
            return self._create_look_at_matrix(
                self.position,
                self.position + self.forward_vector,
                self.up_vector
            )
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the projection matrix for rendering.
        
        Returns:
            4x4 projection matrix
        """
        try:
            # Use pyrr if available
            return pyrr.matrix44.create_perspective_projection_matrix(
                self.settings.fov,
                self.aspect_ratio,
                self.settings.near_plane,
                self.settings.far_plane
            )
        except (NameError, AttributeError):
            # Fallback to manual calculation
            return self._create_perspective_matrix(
                self.settings.fov,
                self.aspect_ratio,
                self.settings.near_plane,
                self.settings.far_plane
            )
    
    def _create_look_at_matrix(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Create look-at matrix manually."""
        f = target - eye
        f = f / np.linalg.norm(f)
        
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        
        u = np.cross(s, f)
        
        result = np.eye(4)
        result[0, 0] = s[0]
        result[1, 0] = s[1]
        result[2, 0] = s[2]
        result[0, 1] = u[0]
        result[1, 1] = u[1]
        result[2, 1] = u[2]
        result[0, 2] = -f[0]
        result[1, 2] = -f[1]
        result[2, 2] = -f[2]
        result[3, 0] = -np.dot(s, eye)
        result[3, 1] = -np.dot(u, eye)
        result[3, 2] = np.dot(f, eye)
        
        return result
    
    def _create_perspective_matrix(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix manually."""
        f = 1.0 / math.tan(fov * 0.5)
        
        result = np.zeros((4, 4))
        result[0, 0] = f / aspect
        result[1, 1] = f
        result[2, 2] = (far + near) / (near - far)
        result[2, 3] = -1.0
        result[3, 2] = (2.0 * far * near) / (near - far)
        
        return result
    
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            world_pos: Position in world coordinates [x, y, z]
            
        Returns:
            Tuple of (screen_x, screen_y, depth)
        """
        # Convert to homogeneous coordinates
        world_pos_h = np.append(world_pos, 1.0)
        
        # Apply view and projection transforms
        view_matrix = self.get_view_matrix()
        proj_matrix = self.get_projection_matrix()
        
        view_pos = view_matrix @ world_pos_h
        clip_pos = proj_matrix @ view_pos
        
        # Perspective divide
        if clip_pos[3] != 0:
            ndc_pos = clip_pos[:3] / clip_pos[3]
        else:
            ndc_pos = clip_pos[:3]
        
        # Convert to screen coordinates
        screen_x = (ndc_pos[0] + 1.0) * 0.5 * self.viewport_width
        screen_y = (1.0 - ndc_pos[1]) * 0.5 * self.viewport_height
        depth = ndc_pos[2]
        
        return (screen_x, screen_y, depth)
    
    def screen_to_world(self, screen_x: float, screen_y: float, depth: float = 0.0) -> np.ndarray:
        """
        Convert screen coordinates to world coordinates.
        
        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            depth: Depth value (0 = near plane, 1 = far plane)
            
        Returns:
            World position [x, y, z]
        """
        # Convert to normalized device coordinates
        ndc_x = (screen_x / self.viewport_width) * 2.0 - 1.0
        ndc_y = 1.0 - (screen_y / self.viewport_height) * 2.0
        ndc_z = depth * 2.0 - 1.0
        
        # Create clip coordinates
        clip_pos = np.array([ndc_x, ndc_y, ndc_z, 1.0])
        
        # Apply inverse projection and view transforms
        try:
            proj_matrix = self.get_projection_matrix()
            view_matrix = self.get_view_matrix()
            
            inv_proj = np.linalg.inv(proj_matrix)
            inv_view = np.linalg.inv(view_matrix)
            
            view_pos = inv_proj @ clip_pos
            view_pos = view_pos / view_pos[3]  # Perspective divide
            
            world_pos = inv_view @ view_pos
            
            return world_pos[:3]
        except np.linalg.LinAlgError:
            # Fallback to camera position if matrix inversion fails
            return self.position.copy()
    
    def is_point_visible(self, world_pos: np.ndarray, margin: float = 0.0) -> bool:
        """
        Check if a world point is visible in the current view.
        
        Args:
            world_pos: Point in world coordinates [x, y, z]
            margin: Additional margin around viewport
            
        Returns:
            True if point is visible, False otherwise
        """
        screen_x, screen_y, depth = self.world_to_screen(world_pos)
        
        return (
            -margin <= screen_x <= self.viewport_width + margin and
            -margin <= screen_y <= self.viewport_height + margin and
            0.0 <= depth <= 1.0  # Within depth range
        )
    
    def is_sphere_visible(self, center: np.ndarray, radius: float) -> bool:
        """
        Check if a sphere is visible in the current view.
        
        Args:
            center: Sphere center in world coordinates [x, y, z]
            radius: Sphere radius
            
        Returns:
            True if sphere is visible, False otherwise
        """
        # Simple check: test if center is visible with radius margin
        # TODO: Implement proper frustum culling
        return self.is_point_visible(center, radius * 50)  # Rough screen space conversion
    
    def reset_to_default(self) -> None:
        """Reset camera to default position and settings."""
        self.set_position(np.array([0.0, 5.0, 10.0]))
        self.set_rotation(np.array([0.0, 0.0, 0.0]))
        self.set_mode(CameraMode.FREE)
        
        self.logger.debug("Camera reset to default")