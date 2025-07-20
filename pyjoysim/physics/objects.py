"""
Physics objects and components for PyJoySim.

This module provides high-level physics object abstractions built on top
of the physics engine interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .engine import PhysicsEngine, Vector2D, BodyType
from ..core.logging import get_logger
from ..core.exceptions import PhysicsError


class ColliderType(Enum):
    """Types of colliders."""
    CIRCLE = "circle"
    BOX = "box"
    POLYGON = "polygon"
    EDGE = "edge"


@dataclass
class Material:
    """Material properties for physics objects."""
    friction: float = 0.7
    elasticity: float = 0.5  # Bounciness
    density: float = 1.0
    name: str = "default"
    
    def __post_init__(self):
        """Validate material properties."""
        self.friction = max(0.0, self.friction)
        self.elasticity = max(0.0, min(1.0, self.elasticity))
        self.density = max(0.01, self.density)


class PhysicsObject(ABC):
    """
    Abstract base class for all physics objects.
    
    Provides common functionality for objects that exist in the physics world.
    """
    
    def __init__(self, 
                 physics_engine: PhysicsEngine,
                 name: str = "PhysicsObject"):
        """
        Initialize the physics object.
        
        Args:
            physics_engine: Physics engine instance
            name: Name for this object
        """
        self.physics_engine = physics_engine
        self.name = name
        self.logger = get_logger(f"physics_object.{name}")
        
        # Object tracking
        self._object_id: Optional[int] = None
        self._is_active = True
        self._user_data: Dict[str, Any] = {}
        
        # Transform
        self._position = Vector2D(0, 0)
        self._rotation = 0.0
        
        # Physics properties
        self._mass = 1.0
        self._material = Material()
        
        self.logger.debug("PhysicsObject created", extra={"name": name})
    
    @property
    def object_id(self) -> Optional[int]:
        """Get the physics engine object ID."""
        return self._object_id
    
    @property
    def position(self) -> Vector2D:
        """Get current position."""
        if self._object_id is not None:
            pos = self.physics_engine.get_object_position(self._object_id)
            if pos is not None:
                self._position = pos
        return self._position
    
    @position.setter
    def position(self, value: Vector2D) -> None:
        """Set position."""
        self._position = value
        if self._object_id is not None:
            self.physics_engine.set_object_position(self._object_id, value)
    
    @property
    def rotation(self) -> float:
        """Get current rotation in radians."""
        if self._object_id is not None:
            rot = self.physics_engine.get_object_rotation(self._object_id)
            if rot is not None:
                self._rotation = rot
        return self._rotation
    
    @rotation.setter
    def rotation(self, value: float) -> None:
        """Set rotation in radians."""
        self._rotation = value
        if self._object_id is not None:
            self.physics_engine.set_object_rotation(self._object_id, value)
    
    @property
    def mass(self) -> float:
        """Get mass."""
        return self._mass
    
    @mass.setter
    def mass(self, value: float) -> None:
        """Set mass."""
        if value <= 0:
            raise ValueError("Mass must be positive")
        self._mass = value
        # Note: Mass changes require recreating the physics body
    
    @property
    def material(self) -> Material:
        """Get material properties."""
        return self._material
    
    @material.setter
    def material(self, value: Material) -> None:
        """Set material properties."""
        self._material = value
        self._update_material_properties()
    
    @property
    def is_active(self) -> bool:
        """Check if object is active."""
        return self._is_active
    
    def set_user_data(self, key: str, value: Any) -> None:
        """Set custom user data."""
        self._user_data[key] = value
    
    def get_user_data(self, key: str, default: Any = None) -> Any:
        """Get custom user data."""
        return self._user_data.get(key, default)
    
    def apply_force(self, force: Vector2D, point: Optional[Vector2D] = None) -> bool:
        """
        Apply a force to the object.
        
        Args:
            force: Force vector to apply
            point: Point of application (center of mass if None)
            
        Returns:
            True if successful, False otherwise
        """
        if self._object_id is None:
            return False
        
        return self.physics_engine.apply_force(self._object_id, force, point)
    
    def apply_impulse(self, impulse: Vector2D, point: Optional[Vector2D] = None) -> bool:
        """
        Apply an impulse to the object.
        
        Args:
            impulse: Impulse vector to apply
            point: Point of application (center of mass if None)
            
        Returns:
            True if successful, False otherwise
        """
        if self._object_id is None:
            return False
        
        return self.physics_engine.apply_impulse(self._object_id, impulse, point)
    
    def destroy(self) -> bool:
        """
        Remove the object from the physics world.
        
        Returns:
            True if successful, False otherwise
        """
        if self._object_id is None:
            return False
        
        success = self.physics_engine.remove_object(self._object_id)
        if success:
            self._object_id = None
            self._is_active = False
            self.logger.debug("PhysicsObject destroyed", extra={"name": self.name})
        
        return success
    
    @abstractmethod
    def _create_physics_body(self) -> bool:
        """
        Create the underlying physics body.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def _update_material_properties(self) -> None:
        """Update material properties in the physics engine."""
        # This would need to be implemented based on specific physics backend
        # For now, we just log the change
        self.logger.debug("Material properties updated", extra={
            "name": self.name,
            "friction": self._material.friction,
            "elasticity": self._material.elasticity,
            "density": self._material.density
        })


class RigidBody(PhysicsObject):
    """
    Dynamic rigid body that responds to forces and collisions.
    """
    
    def __init__(self, 
                 physics_engine: PhysicsEngine,
                 position: Vector2D = Vector2D(0, 0),
                 rotation: float = 0.0,
                 mass: float = 1.0,
                 name: str = "RigidBody"):
        """
        Initialize a dynamic rigid body.
        
        Args:
            physics_engine: Physics engine instance
            position: Initial position
            rotation: Initial rotation in radians
            mass: Body mass
            name: Object name
        """
        super().__init__(physics_engine, name)
        
        self._position = position
        self._rotation = rotation
        self._mass = mass
        
        # Velocity tracking
        self._linear_velocity = Vector2D(0, 0)
        self._angular_velocity = 0.0
        
        # Create physics body
        if not self._create_physics_body():
            raise PhysicsError(f"Failed to create rigid body: {name}")
    
    def _create_physics_body(self) -> bool:
        """Create the physics body in the engine."""
        try:
            self._object_id = self.physics_engine.create_rigid_body(
                BodyType.DYNAMIC,
                self._position,
                self._rotation,
                self._mass
            )
            
            self.logger.debug("RigidBody physics body created", extra={
                "name": self.name,
                "object_id": self._object_id,
                "position": self._position.to_tuple(),
                "rotation": self._rotation,
                "mass": self._mass
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create RigidBody physics body", extra={
                "name": self.name,
                "error": str(e)
            })
            return False
    
    @property
    def linear_velocity(self) -> Vector2D:
        """Get linear velocity."""
        # TODO: Implement velocity retrieval from physics engine
        return self._linear_velocity
    
    @linear_velocity.setter
    def linear_velocity(self, value: Vector2D) -> None:
        """Set linear velocity."""
        self._linear_velocity = value
        # TODO: Implement velocity setting in physics engine
    
    @property
    def angular_velocity(self) -> float:
        """Get angular velocity."""
        # TODO: Implement angular velocity retrieval from physics engine
        return self._angular_velocity
    
    @angular_velocity.setter
    def angular_velocity(self, value: float) -> None:
        """Set angular velocity."""
        self._angular_velocity = value
        # TODO: Implement angular velocity setting in physics engine


class StaticBody(PhysicsObject):
    """
    Static body that doesn't move but can collide with dynamic bodies.
    """
    
    def __init__(self, 
                 physics_engine: PhysicsEngine,
                 position: Vector2D = Vector2D(0, 0),
                 rotation: float = 0.0,
                 name: str = "StaticBody"):
        """
        Initialize a static body.
        
        Args:
            physics_engine: Physics engine instance
            position: Position
            rotation: Rotation in radians
            name: Object name
        """
        super().__init__(physics_engine, name)
        
        self._position = position
        self._rotation = rotation
        self._mass = 0.0  # Static bodies have infinite mass
        
        # Create physics body
        if not self._create_physics_body():
            raise PhysicsError(f"Failed to create static body: {name}")
    
    def _create_physics_body(self) -> bool:
        """Create the physics body in the engine."""
        try:
            self._object_id = self.physics_engine.create_rigid_body(
                BodyType.STATIC,
                self._position,
                self._rotation,
                0.0  # Mass ignored for static bodies
            )
            
            self.logger.debug("StaticBody physics body created", extra={
                "name": self.name,
                "object_id": self._object_id,
                "position": self._position.to_tuple(),
                "rotation": self._rotation
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create StaticBody physics body", extra={
                "name": self.name,
                "error": str(e)
            })
            return False


class KinematicBody(PhysicsObject):
    """
    Kinematic body that moves via velocity but isn't affected by forces.
    """
    
    def __init__(self, 
                 physics_engine: PhysicsEngine,
                 position: Vector2D = Vector2D(0, 0),
                 rotation: float = 0.0,
                 name: str = "KinematicBody"):
        """
        Initialize a kinematic body.
        
        Args:
            physics_engine: Physics engine instance
            position: Initial position
            rotation: Initial rotation in radians
            name: Object name
        """
        super().__init__(physics_engine, name)
        
        self._position = position
        self._rotation = rotation
        self._mass = 0.0  # Kinematic bodies have infinite mass
        
        # Velocity control
        self._linear_velocity = Vector2D(0, 0)
        self._angular_velocity = 0.0
        
        # Create physics body
        if not self._create_physics_body():
            raise PhysicsError(f"Failed to create kinematic body: {name}")
    
    def _create_physics_body(self) -> bool:
        """Create the physics body in the engine."""
        try:
            self._object_id = self.physics_engine.create_rigid_body(
                BodyType.KINEMATIC,
                self._position,
                self._rotation,
                0.0  # Mass ignored for kinematic bodies
            )
            
            self.logger.debug("KinematicBody physics body created", extra={
                "name": self.name,
                "object_id": self._object_id,
                "position": self._position.to_tuple(),
                "rotation": self._rotation
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create KinematicBody physics body", extra={
                "name": self.name,
                "error": str(e)
            })
            return False


class Collider(ABC):
    """
    Abstract base class for collision shapes.
    """
    
    def __init__(self, 
                 physics_object: PhysicsObject,
                 collider_type: ColliderType,
                 offset: Vector2D = Vector2D(0, 0),
                 material: Optional[Material] = None):
        """
        Initialize the collider.
        
        Args:
            physics_object: Parent physics object
            collider_type: Type of collider
            offset: Local offset from object center
            material: Material properties (uses object's material if None)
        """
        self.physics_object = physics_object
        self.collider_type = collider_type
        self.offset = offset
        self.material = material or physics_object.material
        self.logger = get_logger(f"collider.{physics_object.name}")
        
        # Collider tracking
        self._collider_id: Optional[int] = None
        self._is_trigger = False
        self._collision_layers = 0x01  # Default layer
        
        # Create collider in physics engine
        if not self._create_collider():
            raise PhysicsError(f"Failed to create collider for {physics_object.name}")
    
    @property
    def collider_id(self) -> Optional[int]:
        """Get the collider ID."""
        return self._collider_id
    
    @property
    def is_trigger(self) -> bool:
        """Check if this is a trigger collider."""
        return self._is_trigger
    
    @is_trigger.setter
    def is_trigger(self, value: bool) -> None:
        """Set trigger mode."""
        self._is_trigger = value
        # TODO: Update physics engine properties
    
    @property
    def collision_layers(self) -> int:
        """Get collision layer mask."""
        return self._collision_layers
    
    @collision_layers.setter
    def collision_layers(self, value: int) -> None:
        """Set collision layer mask."""
        self._collision_layers = value
        # TODO: Update physics engine properties
    
    @abstractmethod
    def _create_collider(self) -> bool:
        """
        Create the collider in the physics engine.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def destroy(self) -> bool:
        """
        Remove the collider from the physics world.
        
        Returns:
            True if successful, False otherwise
        """
        if self._collider_id is None:
            return False
        
        success = self.physics_object.physics_engine.remove_object(self._collider_id)
        if success:
            self._collider_id = None
            self.logger.debug("Collider destroyed", extra={
                "object_name": self.physics_object.name
            })
        
        return success


class CircleCollider(Collider):
    """Circle-shaped collider."""
    
    def __init__(self,
                 physics_object: PhysicsObject,
                 radius: float,
                 offset: Vector2D = Vector2D(0, 0),
                 material: Optional[Material] = None):
        """
        Initialize a circle collider.
        
        Args:
            physics_object: Parent physics object
            radius: Circle radius
            offset: Local offset from object center
            material: Material properties
        """
        self.radius = radius
        
        if radius <= 0:
            raise ValueError("Circle radius must be positive")
        
        super().__init__(physics_object, ColliderType.CIRCLE, offset, material)
    
    def _create_collider(self) -> bool:
        """Create the circle collider in the physics engine."""
        if self.physics_object.object_id is None:
            return False
        
        try:
            self._collider_id = self.physics_object.physics_engine.add_circle_collider(
                self.physics_object.object_id,
                self.radius,
                self.offset
            )
            
            self.logger.debug("CircleCollider created", extra={
                "object_name": self.physics_object.name,
                "collider_id": self._collider_id,
                "radius": self.radius,
                "offset": self.offset.to_tuple()
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create CircleCollider", extra={
                "object_name": self.physics_object.name,
                "error": str(e)
            })
            return False


class BoxCollider(Collider):
    """Box-shaped collider."""
    
    def __init__(self,
                 physics_object: PhysicsObject,
                 width: float,
                 height: float,
                 offset: Vector2D = Vector2D(0, 0),
                 material: Optional[Material] = None):
        """
        Initialize a box collider.
        
        Args:
            physics_object: Parent physics object
            width: Box width
            height: Box height
            offset: Local offset from object center
            material: Material properties
        """
        self.width = width
        self.height = height
        
        if width <= 0 or height <= 0:
            raise ValueError("Box dimensions must be positive")
        
        super().__init__(physics_object, ColliderType.BOX, offset, material)
    
    def _create_collider(self) -> bool:
        """Create the box collider in the physics engine."""
        if self.physics_object.object_id is None:
            return False
        
        try:
            self._collider_id = self.physics_object.physics_engine.add_box_collider(
                self.physics_object.object_id,
                self.width,
                self.height,
                self.offset
            )
            
            self.logger.debug("BoxCollider created", extra={
                "object_name": self.physics_object.name,
                "collider_id": self._collider_id,
                "width": self.width,
                "height": self.height,
                "offset": self.offset.to_tuple()
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create BoxCollider", extra={
                "object_name": self.physics_object.name,
                "error": str(e)
            })
            return False


class PolygonCollider(Collider):
    """Polygon-shaped collider."""
    
    def __init__(self,
                 physics_object: PhysicsObject,
                 vertices: List[Vector2D],
                 offset: Vector2D = Vector2D(0, 0),
                 material: Optional[Material] = None):
        """
        Initialize a polygon collider.
        
        Args:
            physics_object: Parent physics object
            vertices: List of vertices in local coordinates
            offset: Local offset from object center
            material: Material properties
        """
        self.vertices = vertices
        
        if len(vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        
        super().__init__(physics_object, ColliderType.POLYGON, offset, material)
    
    def _create_collider(self) -> bool:
        """Create the polygon collider in the physics engine."""
        # TODO: Implement polygon collider creation in physics engine
        self.logger.warning("PolygonCollider not yet implemented")
        return False


# Predefined materials
class StandardMaterials:
    """Collection of standard material presets."""
    
    METAL = Material(friction=0.4, elasticity=0.2, density=7.8, name="metal")
    WOOD = Material(friction=0.6, elasticity=0.3, density=0.6, name="wood")
    RUBBER = Material(friction=0.9, elasticity=0.8, density=1.2, name="rubber")
    ICE = Material(friction=0.1, elasticity=0.1, density=0.9, name="ice")
    CONCRETE = Material(friction=0.8, elasticity=0.1, density=2.4, name="concrete")
    BOUNCY = Material(friction=0.3, elasticity=0.95, density=0.5, name="bouncy")
    STICKY = Material(friction=1.5, elasticity=0.0, density=1.0, name="sticky")


# Convenience factory functions
def create_ball(physics_engine: PhysicsEngine,
                position: Vector2D,
                radius: float = 1.0,
                mass: float = 1.0,
                material: Material = StandardMaterials.RUBBER,
                name: str = "Ball") -> Tuple[RigidBody, CircleCollider]:
    """
    Create a ball (dynamic body with circle collider).
    
    Args:
        physics_engine: Physics engine instance
        position: Initial position
        radius: Ball radius
        mass: Ball mass
        material: Material properties
        name: Object name
        
    Returns:
        Tuple of (rigid_body, circle_collider)
    """
    body = RigidBody(physics_engine, position, 0.0, mass, name)
    body.material = material
    collider = CircleCollider(body, radius, Vector2D(0, 0), material)
    
    return body, collider


def create_box(physics_engine: PhysicsEngine,
               position: Vector2D,
               width: float = 2.0,
               height: float = 2.0,
               mass: float = 1.0,
               material: Material = StandardMaterials.WOOD,
               name: str = "Box") -> Tuple[RigidBody, BoxCollider]:
    """
    Create a box (dynamic body with box collider).
    
    Args:
        physics_engine: Physics engine instance
        position: Initial position
        width: Box width
        height: Box height
        mass: Box mass
        material: Material properties
        name: Object name
        
    Returns:
        Tuple of (rigid_body, box_collider)
    """
    body = RigidBody(physics_engine, position, 0.0, mass, name)
    body.material = material
    collider = BoxCollider(body, width, height, Vector2D(0, 0), material)
    
    return body, collider


def create_static_wall(physics_engine: PhysicsEngine,
                       position: Vector2D,
                       width: float = 10.0,
                       height: float = 1.0,
                       rotation: float = 0.0,
                       material: Material = StandardMaterials.CONCRETE,
                       name: str = "Wall") -> Tuple[StaticBody, BoxCollider]:
    """
    Create a static wall.
    
    Args:
        physics_engine: Physics engine instance
        position: Wall position
        width: Wall width
        height: Wall height
        rotation: Wall rotation in radians
        material: Material properties
        name: Object name
        
    Returns:
        Tuple of (static_body, box_collider)
    """
    body = StaticBody(physics_engine, position, rotation, name)
    body.material = material
    collider = BoxCollider(body, width, height, Vector2D(0, 0), material)
    
    return body, collider