"""
Physics world management for PyJoySim.

This module provides high-level physics world management, including
collision handling, object lifecycle, and simulation control.
"""

import time
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .engine import PhysicsEngine, Vector2D, Physics2D, PhysicsEngineType, create_physics_engine
from .objects import PhysicsObject, RigidBody, StaticBody, Material
from .constraints import Constraint, ConstraintManager
from ..config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import PhysicsError


class CollisionEventType(Enum):
    """Types of collision events."""
    BEGIN = "begin"
    PERSIST = "persist"
    END = "end"


@dataclass
class ContactPoint:
    """Information about a collision contact point."""
    position: Vector2D
    normal: Vector2D
    depth: float
    impulse: float


@dataclass
class CollisionEvent:
    """Information about a collision event."""
    event_type: CollisionEventType
    object_a_id: int
    object_b_id: int
    contact_points: List[ContactPoint]
    timestamp: float = field(default_factory=time.time)


class CollisionHandler:
    """
    Handler for collision events between physics objects.
    """
    
    def __init__(self, 
                 object_filter: Optional[Callable[[int, int], bool]] = None,
                 collision_callback: Optional[Callable[[CollisionEvent], None]] = None):
        """
        Initialize collision handler.
        
        Args:
            object_filter: Optional filter function to determine if collision should be processed
            collision_callback: Optional callback for collision events
        """
        self.object_filter = object_filter
        self.collision_callback = collision_callback
        self.logger = get_logger("collision_handler")
        
        # State tracking
        self._active_collisions: Set[Tuple[int, int]] = set()
        self._collision_count = 0
        
    def handle_collision(self, event_type: str, obj_a_id: int, obj_b_id: int, arbiter: Any) -> bool:
        """
        Handle a collision event from the physics engine.
        
        Args:
            event_type: Type of collision event ("begin", "persist", "separate")
            obj_a_id: ID of first object
            obj_b_id: ID of second object
            arbiter: Physics engine arbiter object
            
        Returns:
            True to allow collision, False to ignore
        """
        # Apply object filter
        if self.object_filter and not self.object_filter(obj_a_id, obj_b_id):
            return False
        
        collision_pair = (min(obj_a_id, obj_b_id), max(obj_a_id, obj_b_id))
        
        try:
            if event_type == "begin":
                self._active_collisions.add(collision_pair)
                self._collision_count += 1
                
                event = CollisionEvent(
                    event_type=CollisionEventType.BEGIN,
                    object_a_id=obj_a_id,
                    object_b_id=obj_b_id,
                    contact_points=self._extract_contact_points(arbiter)
                )
                
            elif event_type == "separate":
                self._active_collisions.discard(collision_pair)
                
                event = CollisionEvent(
                    event_type=CollisionEventType.END,
                    object_a_id=obj_a_id,
                    object_b_id=obj_b_id,
                    contact_points=[]
                )
            
            else:  # persist or other
                if collision_pair not in self._active_collisions:
                    return True  # Skip if we haven't seen begin event
                
                event = CollisionEvent(
                    event_type=CollisionEventType.PERSIST,
                    object_a_id=obj_a_id,
                    object_b_id=obj_b_id,
                    contact_points=self._extract_contact_points(arbiter)
                )
            
            # Call collision callback
            if self.collision_callback:
                self.collision_callback(event)
            
            return True
            
        except Exception as e:
            self.logger.error("Error handling collision", extra={
                "error": str(e),
                "event_type": event_type,
                "obj_a_id": obj_a_id,
                "obj_b_id": obj_b_id
            })
            return True
    
    def _extract_contact_points(self, arbiter: Any) -> List[ContactPoint]:
        """Extract contact points from physics engine arbiter."""
        contact_points = []
        
        try:
            # TODO: Implement contact point extraction based on physics engine
            # This is a placeholder implementation
            if hasattr(arbiter, 'contact_point_set'):
                for contact in arbiter.contact_point_set.points:
                    contact_points.append(ContactPoint(
                        position=Vector2D(contact.point_a.x, contact.point_a.y),
                        normal=Vector2D(contact.normal.x, contact.normal.y),
                        depth=contact.distance,
                        impulse=0.0  # TODO: Calculate impulse
                    ))
        except Exception as e:
            self.logger.debug("Could not extract contact points", extra={"error": str(e)})
        
        return contact_points
    
    def get_active_collision_count(self) -> int:
        """Get number of active collisions."""
        return len(self._active_collisions)
    
    def get_total_collision_count(self) -> int:
        """Get total number of collisions processed."""
        return self._collision_count
    
    def is_collision_active(self, obj_a_id: int, obj_b_id: int) -> bool:
        """Check if two objects are currently colliding."""
        collision_pair = (min(obj_a_id, obj_b_id), max(obj_a_id, obj_b_id))
        return collision_pair in self._active_collisions


class PhysicsWorld:
    """
    High-level physics world manager.
    
    Provides a complete physics simulation environment with object management,
    collision handling, and simulation control.
    """
    
    def __init__(self, 
                 engine_type: PhysicsEngineType = PhysicsEngineType.PHYSICS_2D,
                 gravity: Vector2D = Vector2D(0, -9.81)):
        """
        Initialize the physics world.
        
        Args:
            engine_type: Type of physics engine to use
            gravity: World gravity vector
        """
        self.logger = get_logger("physics_world")
        self.settings = get_settings()
        
        # Create physics engine
        self.engine = create_physics_engine(engine_type)
        if not self.engine.initialize():
            raise PhysicsError("Failed to initialize physics engine")
        
        self.engine.set_gravity(gravity)
        
        # Object management
        self._objects: Dict[int, PhysicsObject] = {}
        self._object_groups: Dict[str, List[int]] = {}
        self._next_object_id = 1
        
        # Constraint management
        self.constraint_manager = ConstraintManager(self.engine)
        
        # Collision handling
        self._collision_handlers: Dict[str, CollisionHandler] = {}
        self._default_collision_handler: Optional[CollisionHandler] = None
        
        # Simulation state
        self._is_running = False
        self._time_accumulator = 0.0
        self._fixed_time_step = 1.0 / 60.0  # 60 FPS
        self._max_sub_steps = 3
        
        # Performance tracking
        self._step_count = 0
        self._total_simulation_time = 0.0
        self._last_performance_log = 0.0
        
        # Setup default collision handling
        self._setup_collision_handling()
        
        self.logger.info("PhysicsWorld initialized", extra={
            "engine_type": engine_type.value,
            "gravity": gravity.to_tuple()
        })
    
    def _setup_collision_handling(self) -> None:
        """Setup collision event handling."""
        def collision_callback(event_type: str, obj_a_id: int, obj_b_id: int, arbiter: Any):
            # Find appropriate collision handler
            handler = self._find_collision_handler(obj_a_id, obj_b_id)
            if handler:
                return handler.handle_collision(event_type, obj_a_id, obj_b_id, arbiter)
            return True
        
        self.engine.add_collision_callback(collision_callback)
    
    def _find_collision_handler(self, obj_a_id: int, obj_b_id: int) -> Optional[CollisionHandler]:
        """Find the appropriate collision handler for two objects."""
        # TODO: Implement collision layer/group based handler selection
        return self._default_collision_handler
    
    def add_object(self, physics_object: PhysicsObject, group: str = "default") -> int:
        """
        Add a physics object to the world.
        
        Args:
            physics_object: Object to add
            group: Group name for organization
            
        Returns:
            Assigned object ID
        """
        object_id = self._next_object_id
        self._next_object_id += 1
        
        self._objects[object_id] = physics_object
        
        if group not in self._object_groups:
            self._object_groups[group] = []
        self._object_groups[group].append(object_id)
        
        self.logger.debug("Object added to world", extra={
            "object_id": object_id,
            "object_name": physics_object.name,
            "group": group
        })
        
        return object_id
    
    def remove_object(self, object_id: int) -> bool:
        """
        Remove an object from the world.
        
        Args:
            object_id: ID of object to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if object_id not in self._objects:
            return False
        
        physics_object = self._objects[object_id]
        
        # Remove from groups
        for group_objects in self._object_groups.values():
            if object_id in group_objects:
                group_objects.remove(object_id)
        
        # Destroy physics object
        physics_object.destroy()
        del self._objects[object_id]
        
        self.logger.debug("Object removed from world", extra={
            "object_id": object_id,
            "object_name": physics_object.name
        })
        
        return True
    
    def get_object(self, object_id: int) -> Optional[PhysicsObject]:
        """
        Get an object by ID.
        
        Args:
            object_id: Object ID
            
        Returns:
            Physics object or None if not found
        """
        return self._objects.get(object_id)
    
    def get_objects_in_group(self, group: str) -> List[PhysicsObject]:
        """
        Get all objects in a group.
        
        Args:
            group: Group name
            
        Returns:
            List of objects in the group
        """
        if group not in self._object_groups:
            return []
        
        objects = []
        for object_id in self._object_groups[group]:
            if object_id in self._objects:
                objects.append(self._objects[object_id])
        
        return objects
    
    def remove_group(self, group: str) -> int:
        """
        Remove all objects in a group.
        
        Args:
            group: Group name
            
        Returns:
            Number of objects removed
        """
        if group not in self._object_groups:
            return 0
        
        object_ids = self._object_groups[group].copy()
        count = 0
        
        for object_id in object_ids:
            if self.remove_object(object_id):
                count += 1
        
        del self._object_groups[group]
        
        self.logger.debug("Object group removed", extra={
            "group": group,
            "objects_removed": count
        })
        
        return count
    
    def set_collision_handler(self, 
                            handler: CollisionHandler, 
                            name: str = "default") -> None:
        """
        Set a collision handler.
        
        Args:
            handler: Collision handler
            name: Handler name
        """
        self._collision_handlers[name] = handler
        
        if name == "default":
            self._default_collision_handler = handler
        
        self.logger.debug("Collision handler set", extra={"handler_name": name})
    
    def start_simulation(self) -> None:
        """Start the physics simulation."""
        if self._is_running:
            self.logger.warning("Simulation already running")
            return
        
        self._is_running = True
        self._time_accumulator = 0.0
        
        self.logger.info("Physics simulation started")
    
    def stop_simulation(self) -> None:
        """Stop the physics simulation."""
        if not self._is_running:
            return
        
        self._is_running = False
        self.engine.pause()
        
        self.logger.info("Physics simulation stopped")
    
    def step(self, dt: float) -> None:
        """
        Step the physics simulation.
        
        Uses fixed timestep with accumulation for stable simulation.
        
        Args:
            dt: Time delta since last step
        """
        if not self._is_running:
            return
        
        # Accumulate time
        self._time_accumulator += dt
        
        # Perform fixed timestep updates
        sub_steps = 0
        while self._time_accumulator >= self._fixed_time_step and sub_steps < self._max_sub_steps:
            start_time = time.time()
            
            # Step physics engine
            self.engine.step(self._fixed_time_step)
            
            # Update statistics
            step_time = time.time() - start_time
            self._total_simulation_time += step_time
            self._step_count += 1
            
            # Reduce accumulator
            self._time_accumulator -= self._fixed_time_step
            sub_steps += 1
        
        # Log performance periodically
        current_time = time.time()
        if (current_time - self._last_performance_log) >= 10.0:
            self._log_performance_stats()
            self._last_performance_log = current_time
    
    def _log_performance_stats(self) -> None:
        """Log performance statistics."""
        if self._step_count == 0:
            return
        
        avg_step_time = self._total_simulation_time / self._step_count
        simulation_fps = 1.0 / max(avg_step_time, 0.001)
        
        stats = self.engine.get_stats()
        
        self.logger.debug("Physics world performance", extra={
            "total_objects": len(self._objects),
            "total_constraints": self.constraint_manager.get_constraint_count(),
            "simulation_fps": simulation_fps,
            "average_step_time_ms": avg_step_time * 1000,
            "physics_stats": {
                "step_count": stats.step_count,
                "object_count": stats.object_count,
                "collision_count": stats.collision_count
            }
        })
    
    def set_gravity(self, gravity: Vector2D) -> None:
        """Set world gravity."""
        self.engine.set_gravity(gravity)
        self.logger.debug("World gravity updated", extra={"gravity": gravity.to_tuple()})
    
    def get_gravity(self) -> Vector2D:
        """Get world gravity."""
        return self.engine.get_gravity()
    
    def set_time_step(self, time_step: float) -> None:
        """Set fixed time step."""
        if time_step <= 0:
            raise ValueError("Time step must be positive")
        
        self._fixed_time_step = time_step
        self.logger.debug("Fixed time step updated", extra={"time_step": time_step})
    
    def get_time_step(self) -> float:
        """Get fixed time step."""
        return self._fixed_time_step
    
    def pause(self) -> None:
        """Pause the simulation."""
        self.engine.pause()
        self.logger.debug("Simulation paused")
    
    def resume(self) -> None:
        """Resume the simulation."""
        self.engine.resume()
        self.logger.debug("Simulation resumed")
    
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self.engine.is_paused()
    
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._is_running
    
    def get_object_count(self) -> int:
        """Get total number of objects."""
        return len(self._objects)
    
    def get_group_names(self) -> List[str]:
        """Get list of all group names."""
        return list(self._object_groups.keys())
    
    def query_point(self, point: Vector2D) -> List[int]:
        """
        Query objects at a specific point.
        
        Args:
            point: Point to query
            
        Returns:
            List of object IDs at the point
        """
        # TODO: Implement point query in physics engine
        return []
    
    def query_aabb(self, min_point: Vector2D, max_point: Vector2D) -> List[int]:
        """
        Query objects within an axis-aligned bounding box.
        
        Args:
            min_point: Minimum corner of AABB
            max_point: Maximum corner of AABB
            
        Returns:
            List of object IDs within the AABB
        """
        # TODO: Implement AABB query in physics engine
        return []
    
    def raycast(self, 
               start: Vector2D, 
               end: Vector2D, 
               layer_mask: int = 0xFFFFFFFF) -> Optional[Tuple[int, Vector2D, Vector2D]]:
        """
        Perform a raycast query.
        
        Args:
            start: Ray start point
            end: Ray end point
            layer_mask: Collision layer mask
            
        Returns:
            Tuple of (object_id, hit_point, hit_normal) or None if no hit
        """
        # TODO: Implement raycast in physics engine
        return None
    
    def shutdown(self) -> None:
        """Shutdown the physics world."""
        self.logger.info("Shutting down physics world")
        
        # Stop simulation
        self.stop_simulation()
        
        # Remove all objects
        for object_id in list(self._objects.keys()):
            self.remove_object(object_id)
        
        # Clear constraint manager
        for group in self.constraint_manager.get_group_names():
            self.constraint_manager.remove_group(group)
        
        # Shutdown physics engine
        self.engine.shutdown()
        
        self.logger.info("Physics world shutdown complete")


# Global physics world instance
_physics_world: Optional[PhysicsWorld] = None


def get_physics_world() -> Optional[PhysicsWorld]:
    """Get the global physics world instance."""
    return _physics_world


def create_physics_world(engine_type: PhysicsEngineType = PhysicsEngineType.PHYSICS_2D,
                        gravity: Vector2D = Vector2D(0, -9.81)) -> PhysicsWorld:
    """
    Create and set a global physics world instance.
    
    Args:
        engine_type: Type of physics engine to use
        gravity: World gravity vector
        
    Returns:
        Created physics world instance
    """
    global _physics_world
    _physics_world = PhysicsWorld(engine_type, gravity)
    return _physics_world


def reset_physics_world() -> None:
    """Reset the global physics world instance."""
    global _physics_world
    if _physics_world:
        _physics_world.shutdown()
    _physics_world = None