"""
3D Physics engine using PyBullet for PyJoySim.

This module provides 3D physics simulation capabilities including:
- 3D rigid body dynamics
- 3D collision detection and response
- 3D constraints and joints
- Advanced physics effects (gravity, air resistance, etc.)
"""

import math
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

from ..core.logging import get_logger


class PhysicsMode(Enum):
    """3D Physics simulation modes."""
    DIRECT = "direct"           # Direct GUI mode with visualization
    GUI = "gui"                 # GUI mode with PyBullet visualizer
    HEADLESS = "headless"       # No visualization


@dataclass 
class Vector3D:
    """3D vector for physics calculations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def magnitude(self) -> float:
        """Calculate vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalized(self) -> 'Vector3D':
        """Return normalized vector."""
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector3D(0, 0, 0)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vector3D':
        """Create from numpy array."""
        return cls(arr[0], arr[1], arr[2])


@dataclass
class Quaternion:
    """Quaternion for 3D rotations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Convert to tuple (x, y, z, w)."""
        return (self.x, self.y, self.z, self.w)
    
    def to_euler(self) -> Vector3D:
        """Convert to Euler angles (roll, pitch, yaw)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return Vector3D(roll, pitch, yaw)
    
    @classmethod
    def from_euler(cls, euler: Vector3D) -> 'Quaternion':
        """Create from Euler angles (roll, pitch, yaw)."""
        cy = math.cos(euler.z * 0.5)
        sy = math.sin(euler.z * 0.5)
        cp = math.cos(euler.y * 0.5)
        sp = math.sin(euler.y * 0.5)
        cr = math.cos(euler.x * 0.5)
        sr = math.sin(euler.x * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return cls(x, y, z, w)


@dataclass
class PhysicsMaterial3D:
    """Material properties for 3D physics objects."""
    friction: float = 0.5           # Surface friction
    restitution: float = 0.2        # Bounciness (0-1)
    density: float = 1.0            # Mass density
    linear_damping: float = 0.1     # Linear velocity damping
    angular_damping: float = 0.1    # Angular velocity damping
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for PyBullet."""
        return {
            'lateralFriction': self.friction,
            'restitution': self.restitution,
            'linearDamping': self.linear_damping,
            'angularDamping': self.angular_damping
        }


class Shape3DType(Enum):
    """3D shape types for collision detection."""
    BOX = "box"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CAPSULE = "capsule"
    CONE = "cone"
    PLANE = "plane"
    MESH = "mesh"


@dataclass
class Shape3D:
    """3D collision shape definition."""
    shape_type: Shape3DType
    dimensions: Vector3D = None  # Box: width, height, depth; Sphere: radius, 0, 0
    mesh_path: Optional[str] = None            # For mesh shapes
    scale: Vector3D = None        # Scaling factor
    
    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = Vector3D(1, 1, 1)
        if self.scale is None:
            self.scale = Vector3D(1, 1, 1)
    
    def create_collision_shape(self) -> int:
        """Create PyBullet collision shape."""
        if not PYBULLET_AVAILABLE:
            raise RuntimeError("PyBullet not available")
        
        if self.shape_type == Shape3DType.BOX:
            return p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.dimensions.x/2, self.dimensions.y/2, self.dimensions.z/2]
            )
        elif self.shape_type == Shape3DType.SPHERE:
            return p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=self.dimensions.x
            )
        elif self.shape_type == Shape3DType.CYLINDER:
            return p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=self.dimensions.x,
                height=self.dimensions.y
            )
        elif self.shape_type == Shape3DType.CAPSULE:
            return p.createCollisionShape(
                p.GEOM_CAPSULE,
                radius=self.dimensions.x,
                height=self.dimensions.y
            )
        elif self.shape_type == Shape3DType.CONE:
            return p.createCollisionShape(
                p.GEOM_CONE,
                radius=self.dimensions.x,
                height=self.dimensions.y
            )
        elif self.shape_type == Shape3DType.PLANE:
            return p.createCollisionShape(
                p.GEOM_PLANE,
                planeNormal=[0, 0, 1],
                planeConstant=0
            )
        elif self.shape_type == Shape3DType.MESH and self.mesh_path:
            return p.createCollisionShape(
                p.GEOM_MESH,
                fileName=self.mesh_path,
                meshScale=self.scale.to_tuple()
            )
        else:
            raise ValueError(f"Unsupported shape type: {self.shape_type}")


class Body3DType(Enum):
    """3D body types."""
    STATIC = "static"         # Zero mass, never moves
    KINEMATIC = "kinematic"   # Zero mass, user controlled
    DYNAMIC = "dynamic"       # Has mass, affected by forces


class PhysicsObject3D:
    """
    3D physics object with collision and dynamics.
    """
    
    def __init__(self,
                 name: str,
                 shape: Shape3D,
                 body_type: Body3DType = Body3DType.DYNAMIC,
                 mass: float = 1.0,
                 position: Optional[Vector3D] = None,
                 rotation: Optional[Quaternion] = None,
                 material: Optional[PhysicsMaterial3D] = None):
        """
        Initialize 3D physics object.
        
        Args:
            name: Object name for identification
            shape: Collision shape
            body_type: Body type (static, kinematic, dynamic)
            mass: Object mass (ignored for static/kinematic)
            position: Initial position
            rotation: Initial rotation
            material: Physics material properties
        """
        self.name = name
        self.shape = shape
        self.body_type = body_type
        self.mass = mass if body_type == Body3DType.DYNAMIC else 0.0
        self.position = position or Vector3D()
        self.rotation = rotation or Quaternion()
        self.material = material or PhysicsMaterial3D()
        
        # PyBullet object ID (set when added to world)
        self.body_id: Optional[int] = None
        self.collision_shape_id: Optional[int] = None
        
        # Physics properties
        self.linear_velocity = Vector3D()
        self.angular_velocity = Vector3D()
        self.is_active = True
        
        # User data
        self.user_data: Dict[str, Any] = {}
        
        self.logger = get_logger(f"physics_object_3d.{name}")
    
    def create_body(self) -> int:
        """
        Create PyBullet rigid body.
        
        Returns:
            PyBullet body ID
        """
        if not PYBULLET_AVAILABLE:
            raise RuntimeError("PyBullet not available")
        
        # Create collision shape
        self.collision_shape_id = self.shape.create_collision_shape()
        
        # Create visual shape (same as collision for now)
        visual_shape_id = -1  # Use collision shape as visual
        
        # Create multibody
        self.body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=self.collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.position.to_tuple(),
            baseOrientation=self.rotation.to_tuple()
        )
        
        # Apply material properties
        material_dict = self.material.to_dict()
        p.changeDynamics(
            self.body_id, -1,  # -1 for base link
            **material_dict
        )
        
        # Set initial velocities
        if self.body_type == Body3DType.DYNAMIC:
            p.resetBaseVelocity(
                self.body_id,
                self.linear_velocity.to_tuple(),
                self.angular_velocity.to_tuple()
            )
        
        self.logger.debug("PhysicsObject3D created", extra={
            "body_id": self.body_id,
            "mass": self.mass,
            "body_type": self.body_type.value
        })
        
        return self.body_id
    
    def update_from_physics(self) -> None:
        """Update object state from physics simulation."""
        if self.body_id is None:
            return
        
        # Get position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        self.position = Vector3D.from_array(np.array(pos))
        self.rotation = Quaternion(orn[0], orn[1], orn[2], orn[3])
        
        # Get velocities
        lin_vel, ang_vel = p.getBaseVelocity(self.body_id)
        self.linear_velocity = Vector3D.from_array(np.array(lin_vel))
        self.angular_velocity = Vector3D.from_array(np.array(ang_vel))
    
    def set_position(self, position: Vector3D) -> None:
        """Set object position."""
        self.position = position
        if self.body_id is not None:
            p.resetBasePositionAndOrientation(
                self.body_id,
                position.to_tuple(),
                self.rotation.to_tuple()
            )
    
    def set_rotation(self, rotation: Quaternion) -> None:
        """Set object rotation."""
        self.rotation = rotation
        if self.body_id is not None:
            p.resetBasePositionAndOrientation(
                self.body_id,
                self.position.to_tuple(),
                rotation.to_tuple()
            )
    
    def set_velocity(self, linear: Vector3D, angular: Vector3D = Vector3D()) -> None:
        """Set object velocity."""
        self.linear_velocity = linear
        self.angular_velocity = angular
        if self.body_id is not None and self.body_type == Body3DType.DYNAMIC:
            p.resetBaseVelocity(
                self.body_id,
                linear.to_tuple(),
                angular.to_tuple()
            )
    
    def apply_force(self, force: Vector3D, position: Optional[Vector3D] = None) -> None:
        """
        Apply force to object.
        
        Args:
            force: Force vector in world coordinates
            position: Application point (world coords), defaults to center of mass
        """
        if self.body_id is None or self.body_type != Body3DType.DYNAMIC:
            return
        
        if position is None:
            # Apply force at center of mass
            p.applyExternalForce(
                self.body_id, -1,
                force.to_tuple(),
                self.position.to_tuple(),
                p.WORLD_FRAME
            )
        else:
            # Apply force at specific position
            p.applyExternalForce(
                self.body_id, -1,
                force.to_tuple(),
                position.to_tuple(),
                p.WORLD_FRAME
            )
    
    def apply_torque(self, torque: Vector3D) -> None:
        """Apply torque to object."""
        if self.body_id is None or self.body_type != Body3DType.DYNAMIC:
            return
        
        p.applyExternalTorque(
            self.body_id, -1,
            torque.to_tuple(),
            p.WORLD_FRAME
        )
    
    def remove_from_world(self) -> None:
        """Remove object from physics world."""
        if self.body_id is not None:
            p.removeBody(self.body_id)
            self.body_id = None
        
        if self.collision_shape_id is not None:
            # Note: PyBullet automatically manages collision shapes
            self.collision_shape_id = None


class Physics3D:
    """
    3D Physics engine using PyBullet.
    
    Provides 3D rigid body simulation with collision detection,
    constraints, and advanced physics effects.
    """
    
    def __init__(self, mode: PhysicsMode = PhysicsMode.HEADLESS):
        """
        Initialize 3D physics engine.
        
        Args:
            mode: Physics simulation mode
        """
        if not PYBULLET_AVAILABLE:
            raise RuntimeError(
                "PyBullet is required for 3D physics. "
                "Install with: pip install pybullet"
            )
        
        self.logger = get_logger("physics_3d")
        self.mode = mode
        
        # Initialize PyBullet
        if mode == PhysicsMode.GUI:
            self.physics_client = p.connect(p.GUI)
        elif mode == PhysicsMode.DIRECT:
            self.physics_client = p.connect(p.DIRECT)
        else:  # HEADLESS
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up basic environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Physics settings
        self.gravity = Vector3D(0, -9.81, 0)  # Earth gravity
        self.time_step = 1.0 / 60.0          # 60 FPS
        self.sub_steps = 1                    # Physics sub-steps
        
        # Object management
        self.objects: Dict[str, PhysicsObject3D] = {}
        self.ground_plane_id: Optional[int] = None
        
        # Initialize physics world
        self._setup_world()
        
        self.logger.info("Physics3D engine initialized", extra={
            "mode": mode.value,
            "gravity": self.gravity.to_tuple()
        })
    
    def _setup_world(self) -> None:
        """Set up the physics world with default settings."""
        # Set gravity
        p.setGravity(*self.gravity.to_tuple())
        
        # Set time step
        p.setTimeStep(self.time_step)
        
        # Set physics engine parameters
        p.setPhysicsEngineParameter(
            numSubSteps=self.sub_steps,
            numSolverIterations=50,
            enableConeFriction=1
        )
        
        # Create ground plane
        self.create_ground_plane()
    
    def create_ground_plane(self, size: float = 100.0, position: Vector3D = Vector3D(0, 0, 0)) -> None:
        """
        Create a ground plane.
        
        Args:
            size: Size of the ground plane
            position: Position of the ground plane
        """
        if self.ground_plane_id is not None:
            p.removeBody(self.ground_plane_id)
        
        # Create plane shape
        plane_shape = p.createCollisionShape(p.GEOM_PLANE)
        
        # Create plane body
        self.ground_plane_id = p.createMultiBody(
            baseMass=0,  # Static body
            baseCollisionShapeIndex=plane_shape,
            basePosition=position.to_tuple()
        )
        
        # Set plane material
        p.changeDynamics(
            self.ground_plane_id, -1,
            lateralFriction=0.8,
            restitution=0.1
        )
        
        self.logger.debug("Ground plane created", extra={
            "size": size,
            "position": position.to_tuple()
        })
    
    def add_object(self, obj: PhysicsObject3D) -> None:
        """
        Add physics object to the world.
        
        Args:
            obj: Physics object to add
        """
        if obj.name in self.objects:
            self.logger.warning("Object already exists", extra={"name": obj.name})
            return
        
        # Create physics body
        obj.create_body()
        
        # Store object
        self.objects[obj.name] = obj
        
        self.logger.debug("Object added to physics world", extra={
            "name": obj.name,
            "body_id": obj.body_id
        })
    
    def remove_object(self, name: str) -> None:
        """
        Remove physics object from the world.
        
        Args:
            name: Object name to remove
        """
        if name not in self.objects:
            self.logger.warning("Object not found", extra={"name": name})
            return
        
        obj = self.objects[name]
        obj.remove_from_world()
        del self.objects[name]
        
        self.logger.debug("Object removed from physics world", extra={"name": name})
    
    def get_object(self, name: str) -> Optional[PhysicsObject3D]:
        """
        Get physics object by name.
        
        Args:
            name: Object name
            
        Returns:
            Physics object or None if not found
        """
        return self.objects.get(name)
    
    def step(self, dt: Optional[float] = None) -> None:
        """
        Step the physics simulation.
        
        Args:
            dt: Time step (uses default if None)
        """
        if dt is not None:
            p.setTimeStep(dt)
        
        # Step simulation
        p.stepSimulation()
        
        # Update all objects
        for obj in self.objects.values():
            if obj.is_active:
                obj.update_from_physics()
    
    def set_gravity(self, gravity: Vector3D) -> None:
        """
        Set world gravity.
        
        Args:
            gravity: Gravity vector
        """
        self.gravity = gravity
        p.setGravity(*gravity.to_tuple())
        
        self.logger.debug("Gravity changed", extra={"gravity": gravity.to_tuple()})
    
    def set_time_step(self, time_step: float) -> None:
        """
        Set physics time step.
        
        Args:
            time_step: Time step in seconds
        """
        self.time_step = time_step
        p.setTimeStep(time_step)
    
    def raycast(self, 
                start: Vector3D, 
                end: Vector3D) -> Optional[Tuple[PhysicsObject3D, Vector3D, Vector3D]]:
        """
        Perform raycast in the physics world.
        
        Args:
            start: Ray start position
            end: Ray end position
            
        Returns:
            Tuple of (hit_object, hit_position, hit_normal) or None if no hit
        """
        result = p.rayTest(start.to_tuple(), end.to_tuple())
        
        if result and result[0][0] != -1:  # Hit something
            hit_body_id = result[0][0]
            hit_position = Vector3D.from_array(np.array(result[0][3]))
            hit_normal = Vector3D.from_array(np.array(result[0][4]))
            
            # Find object by body ID
            for obj in self.objects.values():
                if obj.body_id == hit_body_id:
                    return (obj, hit_position, hit_normal)
        
        return None
    
    def get_contacts(self, obj1_name: str, obj2_name: Optional[str] = None) -> List[Dict]:
        """
        Get contact points between objects.
        
        Args:
            obj1_name: First object name
            obj2_name: Second object name (all contacts if None)
            
        Returns:
            List of contact point dictionaries
        """
        obj1 = self.get_object(obj1_name)
        if not obj1 or obj1.body_id is None:
            return []
        
        contacts = []
        
        if obj2_name is None:
            # Get all contacts for obj1
            contact_points = p.getContactPoints(bodyA=obj1.body_id)
        else:
            obj2 = self.get_object(obj2_name)
            if not obj2 or obj2.body_id is None:
                return []
            
            # Get contacts between obj1 and obj2
            contact_points = p.getContactPoints(bodyA=obj1.body_id, bodyB=obj2.body_id)
        
        for contact in contact_points:
            contacts.append({
                'position': Vector3D.from_array(np.array(contact[5])),
                'normal': Vector3D.from_array(np.array(contact[7])),
                'distance': contact[8],
                'force': contact[9]
            })
        
        return contacts
    
    def reset(self) -> None:
        """Reset the physics simulation."""
        # Remove all objects
        for obj_name in list(self.objects.keys()):
            self.remove_object(obj_name)
        
        # Reset simulation
        p.resetSimulation()
        
        # Re-setup world
        self._setup_world()
        
        self.logger.debug("Physics simulation reset")
    
    def cleanup(self) -> None:
        """Clean up physics engine resources."""
        self.reset()
        p.disconnect(self.physics_client)
        
        self.logger.info("Physics3D engine cleaned up")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get physics engine statistics.
        
        Returns:
            Dictionary with physics stats
        """
        return {
            "num_objects": len(self.objects),
            "gravity": self.gravity.to_tuple(),
            "time_step": self.time_step,
            "sub_steps": self.sub_steps,
            "mode": self.mode.value
        }


# Convenience functions for creating common objects
def create_box_3d(name: str, 
                  size: Vector3D = Vector3D(1, 1, 1),
                  position: Vector3D = Vector3D(),
                  mass: float = 1.0,
                  material: Optional[PhysicsMaterial3D] = None) -> PhysicsObject3D:
    """Create a 3D box physics object."""
    shape = Shape3D(Shape3DType.BOX, size)
    return PhysicsObject3D(name, shape, Body3DType.DYNAMIC, mass, position, Quaternion(), material)


def create_sphere_3d(name: str,
                     radius: float = 1.0,
                     position: Vector3D = Vector3D(),
                     mass: float = 1.0,
                     material: Optional[PhysicsMaterial3D] = None) -> PhysicsObject3D:
    """Create a 3D sphere physics object."""
    shape = Shape3D(Shape3DType.SPHERE, Vector3D(radius, 0, 0))
    return PhysicsObject3D(name, shape, Body3DType.DYNAMIC, mass, position, Quaternion(), material)


def create_static_ground_3d(name: str = "ground",
                           position: Vector3D = Vector3D()) -> PhysicsObject3D:
    """Create a static ground plane."""
    shape = Shape3D(Shape3DType.PLANE)
    return PhysicsObject3D(name, shape, Body3DType.STATIC, 0.0, position)