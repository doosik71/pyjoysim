"""
Physics simulation system for PyJoySim.

This module provides physics simulation capabilities including:
- 2D rigid body dynamics
- Collision detection and response  
- Constraints and joints
- Physics object management
"""

from .engine import (
    PhysicsEngine,
    Physics2D,
    Vector2D,
    BodyType,
    PhysicsEngineType,
    PhysicsStats,
    get_physics_engine,
    create_physics_engine,
    reset_physics_engine
)
from .objects import (
    PhysicsObject,
    RigidBody,
    StaticBody,
    KinematicBody,
    Collider,
    CircleCollider,
    BoxCollider,
    PolygonCollider,
    Material,
    StandardMaterials,
    create_ball,
    create_box,
    create_static_wall
)
from .constraints import (
    Constraint,
    Joint,
    PinJoint,
    SlideJoint,
    PivotJoint,
    MotorJoint,
    SpringJoint,
    ConstraintType,
    ConstraintManager,
    create_pin_joint,
    create_fixed_joint,
    create_hinge_joint
)
from .world import (
    PhysicsWorld,
    CollisionHandler,
    CollisionEvent,
    CollisionEventType,
    ContactPoint,
    get_physics_world,
    create_physics_world,
    reset_physics_world
)

__all__ = [
    # Core engine
    "PhysicsEngine",
    "Physics2D", 
    "PhysicsEngineType",
    "Vector2D",
    "BodyType",
    "PhysicsStats",
    
    # Physics objects
    "PhysicsObject",
    "RigidBody",
    "StaticBody", 
    "KinematicBody",
    
    # Colliders
    "Collider",
    "CircleCollider",
    "BoxCollider",
    "PolygonCollider",
    
    # Materials
    "Material",
    "StandardMaterials",
    
    # Constraints and joints
    "Constraint",
    "Joint",
    "PinJoint",
    "SlideJoint", 
    "PivotJoint",
    "MotorJoint",
    "SpringJoint",
    "ConstraintType",
    "ConstraintManager",
    
    # World and collision
    "PhysicsWorld",
    "CollisionHandler",
    "CollisionEvent",
    "CollisionEventType",
    "ContactPoint",
    
    # Factory functions
    "get_physics_engine",
    "create_physics_engine",
    "reset_physics_engine",
    "get_physics_world",
    "create_physics_world",
    "reset_physics_world",
    
    # Object creation helpers
    "create_ball",
    "create_box",
    "create_static_wall",
    "create_pin_joint",
    "create_fixed_joint",
    "create_hinge_joint",
]