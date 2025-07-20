"""
Physics engine abstraction layer
"""

from .physics_engine import PhysicsEngine
from .physics_2d import Physics2D
from .physics_3d import Physics3D

__all__ = [
    "PhysicsEngine",
    "Physics2D",
    "Physics3D",
]