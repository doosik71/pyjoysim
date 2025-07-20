"""
Physics engine abstract interface and implementations.

This module defines the core physics engine architecture with abstract
base classes and concrete implementations for different physics backends.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import pymunk
import pymunk.pygame_util

from ..config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import PhysicsError, InitializationError


class PhysicsEngineType(Enum):
    """Types of physics engines."""
    PHYSICS_2D = "physics_2d"
    PHYSICS_3D = "physics_3d"


class BodyType(Enum):
    """Types of physics bodies."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    KINEMATIC = "kinematic"


@dataclass
class Vector2D:
    """2D vector for physics calculations."""
    x: float
    y: float
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def normalized(self) -> 'Vector2D':
        """Get normalized vector."""
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return self / mag
    
    def dot(self, other: 'Vector2D') -> float:
        """Calculate dot product."""
        return self.x * other.x + self.y * other.y
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)


@dataclass
class PhysicsStats:
    """Physics engine performance statistics."""
    simulation_time: float = 0.0
    step_count: int = 0
    object_count: int = 0
    collision_count: int = 0
    constraint_count: int = 0
    average_step_time: float = 0.0
    fps: float = 0.0
    
    def update(self, step_time: float):
        """Update statistics with new step."""
        self.simulation_time += step_time
        self.step_count += 1
        self.average_step_time = self.simulation_time / self.step_count
        if step_time > 0:
            self.fps = 1.0 / step_time


class PhysicsEngine(ABC):
    """
    Abstract base class for physics engines.
    
    Defines the interface that all physics engine implementations must follow.
    """
    
    def __init__(self):
        """Initialize the physics engine."""
        self.logger = get_logger("physics_engine")
        self.settings = get_settings()
        
        # Engine state
        self._initialized = False
        self._running = False
        self._paused = False
        
        # Physics settings
        self._gravity = Vector2D(0, 0)
        self._time_step = 1.0 / 60.0  # 60 FPS default
        self._iterations = 10
        
        # Statistics
        self._stats = PhysicsStats()
        self._last_step_time = 0.0
        
        # Objects tracking
        self._objects: Dict[int, Any] = {}
        self._next_object_id = 1
        
        # Event callbacks
        self._collision_callbacks: List[Callable] = []
        self._pre_step_callbacks: List[Callable] = []
        self._post_step_callbacks: List[Callable] = []
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the physics engine.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the physics engine and clean up resources."""
        pass
    
    @abstractmethod
    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance the physics simulation by one time step.
        
        Args:
            dt: Time step in seconds (uses default if None)
        """
        pass
    
    @abstractmethod
    def create_rigid_body(self, 
                         body_type: BodyType,
                         position: Vector2D,
                         angle: float = 0.0,
                         mass: float = 1.0) -> int:
        """
        Create a rigid body in the physics world.
        
        Args:
            body_type: Type of body (dynamic, static, kinematic)
            position: Initial position
            angle: Initial rotation angle in radians
            mass: Body mass (ignored for static bodies)
            
        Returns:
            Object ID for the created body
        """
        pass
    
    @abstractmethod
    def add_circle_collider(self, 
                           body_id: int, 
                           radius: float,
                           offset: Vector2D = Vector2D(0, 0)) -> int:
        """
        Add a circle collider to a body.
        
        Args:
            body_id: ID of the body to add collider to
            radius: Circle radius
            offset: Local offset from body center
            
        Returns:
            Collider ID
        """
        pass
    
    @abstractmethod
    def add_box_collider(self, 
                        body_id: int,
                        width: float,
                        height: float,
                        offset: Vector2D = Vector2D(0, 0)) -> int:
        """
        Add a box collider to a body.
        
        Args:
            body_id: ID of the body to add collider to
            width: Box width
            height: Box height
            offset: Local offset from body center
            
        Returns:
            Collider ID
        """
        pass
    
    @abstractmethod
    def remove_object(self, object_id: int) -> bool:
        """
        Remove an object from the physics world.
        
        Args:
            object_id: ID of object to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_object_position(self, object_id: int) -> Optional[Vector2D]:
        """
        Get the position of a physics object.
        
        Args:
            object_id: ID of the object
            
        Returns:
            Object position or None if not found
        """
        pass
    
    @abstractmethod
    def get_object_rotation(self, object_id: int) -> Optional[float]:
        """
        Get the rotation of a physics object.
        
        Args:
            object_id: ID of the object
            
        Returns:
            Object rotation in radians or None if not found
        """
        pass
    
    @abstractmethod
    def set_object_position(self, object_id: int, position: Vector2D) -> bool:
        """
        Set the position of a physics object.
        
        Args:
            object_id: ID of the object
            position: New position
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def set_object_rotation(self, object_id: int, angle: float) -> bool:
        """
        Set the rotation of a physics object.
        
        Args:
            object_id: ID of the object
            angle: New rotation in radians
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def apply_force(self, object_id: int, force: Vector2D, point: Optional[Vector2D] = None) -> bool:
        """
        Apply a force to a physics object.
        
        Args:
            object_id: ID of the object
            force: Force vector to apply
            point: Point of application (body center if None)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def apply_impulse(self, object_id: int, impulse: Vector2D, point: Optional[Vector2D] = None) -> bool:
        """
        Apply an impulse to a physics object.
        
        Args:
            object_id: ID of the object
            impulse: Impulse vector to apply
            point: Point of application (body center if None)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # Common implementation methods
    
    def set_gravity(self, gravity: Vector2D) -> None:
        """Set world gravity."""
        self._gravity = gravity
        self.logger.debug("Gravity updated", extra={
            "gravity_x": gravity.x,
            "gravity_y": gravity.y
        })
    
    def get_gravity(self) -> Vector2D:
        """Get world gravity."""
        return self._gravity
    
    def set_time_step(self, time_step: float) -> None:
        """Set physics time step."""
        if time_step <= 0:
            raise ValueError("Time step must be positive")
        self._time_step = time_step
        self.logger.debug("Time step updated", extra={"time_step": time_step})
    
    def get_time_step(self) -> float:
        """Get physics time step."""
        return self._time_step
    
    def set_iterations(self, iterations: int) -> None:
        """Set solver iterations."""
        if iterations <= 0:
            raise ValueError("Iterations must be positive")
        self._iterations = iterations
        self.logger.debug("Iterations updated", extra={"iterations": iterations})
    
    def get_iterations(self) -> int:
        """Get solver iterations."""
        return self._iterations
    
    def pause(self) -> None:
        """Pause the physics simulation."""
        self._paused = True
        self.logger.debug("Physics simulation paused")
    
    def resume(self) -> None:
        """Resume the physics simulation."""
        self._paused = False
        self.logger.debug("Physics simulation resumed")
    
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self._paused
    
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running
    
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized
    
    def get_stats(self) -> PhysicsStats:
        """Get physics engine statistics."""
        return self._stats
    
    def get_object_count(self) -> int:
        """Get number of physics objects."""
        return len(self._objects)
    
    def add_collision_callback(self, callback: Callable) -> None:
        """Add a collision event callback."""
        self._collision_callbacks.append(callback)
    
    def remove_collision_callback(self, callback: Callable) -> None:
        """Remove a collision event callback."""
        if callback in self._collision_callbacks:
            self._collision_callbacks.remove(callback)
    
    def add_pre_step_callback(self, callback: Callable) -> None:
        """Add a pre-step callback."""
        self._pre_step_callbacks.append(callback)
    
    def add_post_step_callback(self, callback: Callable) -> None:
        """Add a post-step callback."""
        self._post_step_callbacks.append(callback)
    
    def _get_next_object_id(self) -> int:
        """Get next available object ID."""
        object_id = self._next_object_id
        self._next_object_id += 1
        return object_id
    
    def _call_pre_step_callbacks(self, dt: float) -> None:
        """Call all pre-step callbacks."""
        for callback in self._pre_step_callbacks:
            try:
                callback(dt)
            except Exception as e:
                self.logger.error("Error in pre-step callback", extra={"error": str(e)})
    
    def _call_post_step_callbacks(self, dt: float) -> None:
        """Call all post-step callbacks."""
        for callback in self._post_step_callbacks:
            try:
                callback(dt)
            except Exception as e:
                self.logger.error("Error in post-step callback", extra={"error": str(e)})


class Physics2D(PhysicsEngine):
    """
    2D physics engine implementation using pymunk.
    
    Provides real-time 2D rigid body dynamics with collision detection.
    """
    
    def __init__(self):
        """Initialize the 2D physics engine."""
        super().__init__()
        
        # Pymunk-specific attributes
        self._space: Optional[pymunk.Space] = None
        self._bodies: Dict[int, pymunk.Body] = {}
        self._shapes: Dict[int, pymunk.Shape] = {}
        self._constraints: Dict[int, pymunk.Constraint] = {}
        
        # Collision tracking
        self._collision_handler = None
        self._active_collisions: set = set()
        
        self.logger.debug("Physics2D engine created")
    
    def initialize(self) -> bool:
        """Initialize the pymunk physics space."""
        if self._initialized:
            self.logger.warning("Physics2D already initialized")
            return True
        
        try:
            # Create pymunk space
            self._space = pymunk.Space()
            
            # Set default gravity
            self._space.gravity = self._gravity.to_tuple()
            
            # Configure space properties
            self._space.iterations = self._iterations
            self._space.sleep_time_threshold = 0.5
            self._space.collision_slop = 0.1
            
            # Set up collision handler
            self._setup_collision_handler()
            
            self._initialized = True
            self._running = True
            
            self.logger.info("Physics2D engine initialized", extra={
                "gravity": self._gravity.to_tuple(),
                "iterations": self._iterations,
                "time_step": self._time_step
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize Physics2D", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    def shutdown(self) -> None:
        """Shutdown the physics engine."""
        if not self._initialized:
            return
        
        self.logger.info("Shutting down Physics2D engine")
        
        # Clear all objects
        self._objects.clear()
        self._bodies.clear()
        self._shapes.clear()
        self._constraints.clear()
        self._active_collisions.clear()
        
        # Clean up space
        if self._space:
            # Remove all bodies and shapes from space
            for body in list(self._space.bodies):
                self._space.remove(body)
            for shape in list(self._space.shapes):
                self._space.remove(shape)
            for constraint in list(self._space.constraints):
                self._space.remove(constraint)
            
            self._space = None
        
        self._initialized = False
        self._running = False
        
        self.logger.info("Physics2D engine shutdown complete")
    
    def step(self, dt: Optional[float] = None) -> None:
        """Advance the physics simulation."""
        if not self._initialized or self._paused:
            return
        
        if dt is None:
            dt = self._time_step
        
        step_start = time.time()
        
        try:
            # Call pre-step callbacks
            self._call_pre_step_callbacks(dt)
            
            # Update space properties if changed
            if self._space.gravity != self._gravity.to_tuple():
                self._space.gravity = self._gravity.to_tuple()
            
            if self._space.iterations != self._iterations:
                self._space.iterations = self._iterations
            
            # Step the physics simulation
            self._space.step(dt)
            
            # Update statistics
            step_time = time.time() - step_start
            self._stats.update(step_time)
            self._stats.object_count = len(self._objects)
            self._stats.collision_count = len(self._active_collisions)
            self._stats.constraint_count = len(self._constraints)
            
            # Call post-step callbacks
            self._call_post_step_callbacks(dt)
            
        except Exception as e:
            self.logger.error("Error during physics step", extra={
                "error": str(e),
                "dt": dt,
                "step_count": self._stats.step_count
            })
    
    def create_rigid_body(self, 
                         body_type: BodyType,
                         position: Vector2D,
                         angle: float = 0.0,
                         mass: float = 1.0) -> int:
        """Create a rigid body in the physics world."""
        if not self._initialized:
            raise PhysicsError("Physics engine not initialized")
        
        try:
            object_id = self._get_next_object_id()
            
            # Create pymunk body based on type
            if body_type == BodyType.STATIC:
                body = pymunk.Body(body_type=pymunk.Body.STATIC)
            elif body_type == BodyType.KINEMATIC:
                body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            else:  # DYNAMIC
                moment = pymunk.moment_for_circle(mass, 0, 1)  # Default moment
                body = pymunk.Body(mass, moment)
            
            # Set initial position and angle
            body.position = position.to_tuple()
            body.angle = angle
            
            # Add to space and tracking
            self._space.add(body)
            self._bodies[object_id] = body
            self._objects[object_id] = {
                "type": "body",
                "body_type": body_type,
                "pymunk_body": body
            }
            
            self.logger.debug("Rigid body created", extra={
                "object_id": object_id,
                "body_type": body_type.value,
                "position": position.to_tuple(),
                "angle": angle,
                "mass": mass
            })
            
            return object_id
            
        except Exception as e:
            self.logger.error("Failed to create rigid body", extra={
                "error": str(e),
                "body_type": body_type.value,
                "position": position.to_tuple()
            })
            raise PhysicsError(f"Failed to create rigid body: {e}")
    
    def add_circle_collider(self, 
                           body_id: int, 
                           radius: float,
                           offset: Vector2D = Vector2D(0, 0)) -> int:
        """Add a circle collider to a body."""
        if body_id not in self._bodies:
            raise PhysicsError(f"Body {body_id} not found")
        
        try:
            body = self._bodies[body_id]
            shape = pymunk.Circle(body, radius, offset.to_tuple())
            
            # Set default properties
            shape.friction = 0.7
            shape.elasticity = 0.5
            
            # Add to space and tracking
            self._space.add(shape)
            
            collider_id = self._get_next_object_id()
            self._shapes[collider_id] = shape
            self._objects[collider_id] = {
                "type": "collider",
                "collider_type": "circle",
                "body_id": body_id,
                "pymunk_shape": shape
            }
            
            self.logger.debug("Circle collider added", extra={
                "collider_id": collider_id,
                "body_id": body_id,
                "radius": radius,
                "offset": offset.to_tuple()
            })
            
            return collider_id
            
        except Exception as e:
            self.logger.error("Failed to add circle collider", extra={
                "error": str(e),
                "body_id": body_id,
                "radius": radius
            })
            raise PhysicsError(f"Failed to add circle collider: {e}")
    
    def add_box_collider(self, 
                        body_id: int,
                        width: float,
                        height: float,
                        offset: Vector2D = Vector2D(0, 0)) -> int:
        """Add a box collider to a body."""
        if body_id not in self._bodies:
            raise PhysicsError(f"Body {body_id} not found")
        
        try:
            body = self._bodies[body_id]
            
            # Create box vertices
            w, h = width / 2, height / 2
            vertices = [(-w, -h), (w, -h), (w, h), (-w, h)]
            
            # Apply offset
            if offset.x != 0 or offset.y != 0:
                vertices = [(x + offset.x, y + offset.y) for x, y in vertices]
            
            shape = pymunk.Poly(body, vertices)
            
            # Set default properties
            shape.friction = 0.7
            shape.elasticity = 0.5
            
            # Add to space and tracking
            self._space.add(shape)
            
            collider_id = self._get_next_object_id()
            self._shapes[collider_id] = shape
            self._objects[collider_id] = {
                "type": "collider",
                "collider_type": "box",
                "body_id": body_id,
                "pymunk_shape": shape
            }
            
            self.logger.debug("Box collider added", extra={
                "collider_id": collider_id,
                "body_id": body_id,
                "width": width,
                "height": height,
                "offset": offset.to_tuple()
            })
            
            return collider_id
            
        except Exception as e:
            self.logger.error("Failed to add box collider", extra={
                "error": str(e),
                "body_id": body_id,
                "width": width,
                "height": height
            })
            raise PhysicsError(f"Failed to add box collider: {e}")
    
    def remove_object(self, object_id: int) -> bool:
        """Remove an object from the physics world."""
        if object_id not in self._objects:
            return False
        
        try:
            obj_data = self._objects[object_id]
            
            if obj_data["type"] == "body":
                body = self._bodies[object_id]
                
                # Remove all shapes attached to this body
                shapes_to_remove = []
                for shape_id, shape in self._shapes.items():
                    if shape.body == body:
                        shapes_to_remove.append(shape_id)
                
                for shape_id in shapes_to_remove:
                    self._space.remove(self._shapes[shape_id])
                    del self._shapes[shape_id]
                    del self._objects[shape_id]
                
                # Remove body
                self._space.remove(body)
                del self._bodies[object_id]
                
            elif obj_data["type"] == "collider":
                shape = self._shapes[object_id]
                self._space.remove(shape)
                del self._shapes[object_id]
            
            del self._objects[object_id]
            
            self.logger.debug("Object removed", extra={"object_id": object_id})
            return True
            
        except Exception as e:
            self.logger.error("Failed to remove object", extra={
                "error": str(e),
                "object_id": object_id
            })
            return False
    
    def get_object_position(self, object_id: int) -> Optional[Vector2D]:
        """Get the position of a physics object."""
        if object_id not in self._bodies:
            return None
        
        body = self._bodies[object_id]
        return Vector2D(body.position.x, body.position.y)
    
    def get_object_rotation(self, object_id: int) -> Optional[float]:
        """Get the rotation of a physics object."""
        if object_id not in self._bodies:
            return None
        
        body = self._bodies[object_id]
        return body.angle
    
    def set_object_position(self, object_id: int, position: Vector2D) -> bool:
        """Set the position of a physics object."""
        if object_id not in self._bodies:
            return False
        
        try:
            body = self._bodies[object_id]
            body.position = position.to_tuple()
            return True
        except Exception as e:
            self.logger.error("Failed to set object position", extra={
                "error": str(e),
                "object_id": object_id
            })
            return False
    
    def set_object_rotation(self, object_id: int, angle: float) -> bool:
        """Set the rotation of a physics object."""
        if object_id not in self._bodies:
            return False
        
        try:
            body = self._bodies[object_id]
            body.angle = angle
            return True
        except Exception as e:
            self.logger.error("Failed to set object rotation", extra={
                "error": str(e),
                "object_id": object_id
            })
            return False
    
    def apply_force(self, object_id: int, force: Vector2D, point: Optional[Vector2D] = None) -> bool:
        """Apply a force to a physics object."""
        if object_id not in self._bodies:
            return False
        
        try:
            body = self._bodies[object_id]
            force_tuple = force.to_tuple()
            
            if point is None:
                body.apply_force_at_world_point(force_tuple, body.position)
            else:
                body.apply_force_at_world_point(force_tuple, point.to_tuple())
            
            return True
        except Exception as e:
            self.logger.error("Failed to apply force", extra={
                "error": str(e),
                "object_id": object_id
            })
            return False
    
    def apply_impulse(self, object_id: int, impulse: Vector2D, point: Optional[Vector2D] = None) -> bool:
        """Apply an impulse to a physics object."""
        if object_id not in self._bodies:
            return False
        
        try:
            body = self._bodies[object_id]
            impulse_tuple = impulse.to_tuple()
            
            if point is None:
                body.apply_impulse_at_world_point(impulse_tuple, body.position)
            else:
                body.apply_impulse_at_world_point(impulse_tuple, point.to_tuple())
            
            return True
        except Exception as e:
            self.logger.error("Failed to apply impulse", extra={
                "error": str(e),
                "object_id": object_id
            })
            return False
    
    def _setup_collision_handler(self) -> None:
        """Set up collision detection handlers."""
        def collision_begin(arbiter, space, data):
            """Handle collision begin."""
            shape_a, shape_b = arbiter.shapes
            
            # Find object IDs
            obj_a_id = None
            obj_b_id = None
            
            for obj_id, shape in self._shapes.items():
                if shape == shape_a:
                    obj_a_id = obj_id
                elif shape == shape_b:
                    obj_b_id = obj_id
            
            if obj_a_id is not None and obj_b_id is not None:
                collision_pair = (min(obj_a_id, obj_b_id), max(obj_a_id, obj_b_id))
                self._active_collisions.add(collision_pair)
                
                # Call collision callbacks
                for callback in self._collision_callbacks:
                    try:
                        callback("begin", obj_a_id, obj_b_id, arbiter)
                    except Exception as e:
                        self.logger.error("Error in collision callback", extra={"error": str(e)})
            
            return True
        
        def collision_separate(arbiter, space, data):
            """Handle collision separate."""
            shape_a, shape_b = arbiter.shapes
            
            # Find object IDs
            obj_a_id = None
            obj_b_id = None
            
            for obj_id, shape in self._shapes.items():
                if shape == shape_a:
                    obj_a_id = obj_id
                elif shape == shape_b:
                    obj_b_id = obj_id
            
            if obj_a_id is not None and obj_b_id is not None:
                collision_pair = (min(obj_a_id, obj_b_id), max(obj_a_id, obj_b_id))
                self._active_collisions.discard(collision_pair)
                
                # Call collision callbacks
                for callback in self._collision_callbacks:
                    try:
                        callback("separate", obj_a_id, obj_b_id, arbiter)
                    except Exception as e:
                        self.logger.error("Error in collision callback", extra={"error": str(e)})
        
        # Set up default collision handler
        self._collision_handler = self._space.add_default_collision_handler()
        self._collision_handler.begin = collision_begin
        self._collision_handler.separate = collision_separate
    
    def get_pymunk_space(self) -> Optional[pymunk.Space]:
        """Get the underlying pymunk space for advanced operations."""
        return self._space


# Global physics engine instance
_physics_engine: Optional[PhysicsEngine] = None


def get_physics_engine() -> Optional[PhysicsEngine]:
    """Get the global physics engine instance."""
    return _physics_engine


def create_physics_engine(engine_type: PhysicsEngineType) -> PhysicsEngine:
    """
    Create and set a global physics engine instance.
    
    Args:
        engine_type: Type of physics engine to create
        
    Returns:
        Created physics engine instance
    """
    global _physics_engine
    
    if engine_type == PhysicsEngineType.PHYSICS_2D:
        _physics_engine = Physics2D()
    else:
        raise ValueError(f"Unsupported physics engine type: {engine_type}")
    
    return _physics_engine


def reset_physics_engine() -> None:
    """Reset the global physics engine instance."""
    global _physics_engine
    if _physics_engine and _physics_engine.is_initialized():
        _physics_engine.shutdown()
    _physics_engine = None