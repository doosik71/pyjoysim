"""
Physics constraints and joints for PyJoySim.

This module provides various constraint types for connecting and limiting
the movement of physics objects.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import pymunk

from .engine import PhysicsEngine, Vector2D
from .objects import PhysicsObject
from ..core.logging import get_logger
from ..core.exceptions import PhysicsError


class ConstraintType(Enum):
    """Types of physics constraints."""
    PIN_JOINT = "pin_joint"
    SLIDE_JOINT = "slide_joint"
    PIVOT_JOINT = "pivot_joint"
    MOTOR_JOINT = "motor_joint"
    SPRING_JOINT = "spring_joint"
    GEAR_JOINT = "gear_joint"
    DISTANCE_JOINT = "distance_joint"


@dataclass
class ConstraintBreakInfo:
    """Information about a broken constraint."""
    constraint_id: int
    force: float
    timestamp: float
    reason: str


class Constraint(ABC):
    """
    Abstract base class for physics constraints.
    
    Constraints limit the relative motion between two physics objects.
    """
    
    def __init__(self,
                 physics_engine: PhysicsEngine,
                 object_a: PhysicsObject,
                 object_b: PhysicsObject,
                 constraint_type: ConstraintType,
                 name: str = "Constraint"):
        """
        Initialize the constraint.
        
        Args:
            physics_engine: Physics engine instance
            object_a: First physics object
            object_b: Second physics object
            constraint_type: Type of constraint
            name: Constraint name
        """
        self.physics_engine = physics_engine
        self.object_a = object_a
        self.object_b = object_b
        self.constraint_type = constraint_type
        self.name = name
        self.logger = get_logger(f"constraint.{name}")
        
        # Constraint tracking
        self._constraint_id: Optional[int] = None
        self._is_active = True
        self._break_force = float('inf')  # Force required to break constraint
        self._collide_connected = False  # Allow connected objects to collide
        
        # Validation
        if object_a.object_id is None or object_b.object_id is None:
            raise PhysicsError("Both objects must have valid physics bodies")
        
        self.logger.debug("Constraint created", extra={
            "name": name,
            "type": constraint_type.value,
            "object_a": object_a.name,
            "object_b": object_b.name
        })
    
    @property
    def constraint_id(self) -> Optional[int]:
        """Get the constraint ID."""
        return self._constraint_id
    
    @property
    def is_active(self) -> bool:
        """Check if constraint is active."""
        return self._is_active
    
    @property
    def break_force(self) -> float:
        """Get break force threshold."""
        return self._break_force
    
    @break_force.setter
    def break_force(self, value: float) -> None:
        """Set break force threshold."""
        self._break_force = max(0.0, value)
        self._update_break_force()
    
    @property
    def collide_connected(self) -> bool:
        """Check if connected objects can collide."""
        return self._collide_connected
    
    @collide_connected.setter
    def collide_connected(self, value: bool) -> None:
        """Set whether connected objects can collide."""
        self._collide_connected = value
        self._update_collision_settings()
    
    @abstractmethod
    def _create_constraint(self) -> bool:
        """
        Create the constraint in the physics engine.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def _update_break_force(self) -> None:
        """Update break force in physics engine."""
        # TODO: Implement break force update in physics engine
        pass
    
    def _update_collision_settings(self) -> None:
        """Update collision settings in physics engine."""
        # TODO: Implement collision settings update in physics engine
        pass
    
    def destroy(self) -> bool:
        """
        Remove the constraint from the physics world.
        
        Returns:
            True if successful, False otherwise
        """
        if self._constraint_id is None:
            return False
        
        # TODO: Implement constraint removal in physics engine
        self._constraint_id = None
        self._is_active = False
        
        self.logger.debug("Constraint destroyed", extra={"name": self.name})
        return True
    
    def get_reaction_force(self) -> Optional[Vector2D]:
        """
        Get the current reaction force at the constraint.
        
        Returns:
            Reaction force vector or None if not available
        """
        # TODO: Implement reaction force calculation
        return None
    
    def get_reaction_torque(self) -> Optional[float]:
        """
        Get the current reaction torque at the constraint.
        
        Returns:
            Reaction torque or None if not available
        """
        # TODO: Implement reaction torque calculation
        return None


class Joint(Constraint):
    """Base class for joint constraints that connect two objects at specific points."""
    
    def __init__(self,
                 physics_engine: PhysicsEngine,
                 object_a: PhysicsObject,
                 object_b: PhysicsObject,
                 constraint_type: ConstraintType,
                 anchor_a: Vector2D,
                 anchor_b: Vector2D,
                 name: str = "Joint"):
        """
        Initialize the joint.
        
        Args:
            physics_engine: Physics engine instance
            object_a: First physics object
            object_b: Second physics object
            constraint_type: Type of joint
            anchor_a: Anchor point on object A (local coordinates)
            anchor_b: Anchor point on object B (local coordinates)
            name: Joint name
        """
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b
        
        super().__init__(physics_engine, object_a, object_b, constraint_type, name)


class PinJoint(Joint):
    """
    Pin joint that connects two objects at specific points but allows rotation.
    """
    
    def __init__(self,
                 physics_engine: PhysicsEngine,
                 object_a: PhysicsObject,
                 object_b: PhysicsObject,
                 anchor_a: Vector2D = Vector2D(0, 0),
                 anchor_b: Vector2D = Vector2D(0, 0),
                 name: str = "PinJoint"):
        """
        Initialize a pin joint.
        
        Args:
            physics_engine: Physics engine instance
            object_a: First physics object
            object_b: Second physics object
            anchor_a: Anchor point on object A (local coordinates)
            anchor_b: Anchor point on object B (local coordinates)
            name: Joint name
        """
        super().__init__(physics_engine, object_a, object_b, ConstraintType.PIN_JOINT,
                        anchor_a, anchor_b, name)
        
        if not self._create_constraint():
            raise PhysicsError(f"Failed to create pin joint: {name}")
    
    def _create_constraint(self) -> bool:
        """Create the pin joint in the physics engine."""
        try:
            # TODO: Implement pin joint creation in physics engine
            # For now, we'll simulate success
            self._constraint_id = 1  # Placeholder
            
            self.logger.debug("PinJoint created", extra={
                "name": self.name,
                "constraint_id": self._constraint_id,
                "anchor_a": self.anchor_a.to_tuple(),
                "anchor_b": self.anchor_b.to_tuple()
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create PinJoint", extra={
                "name": self.name,
                "error": str(e)
            })
            return False


class SlideJoint(Joint):
    """
    Slide joint that constrains movement along a line but allows sliding and rotation.
    """
    
    def __init__(self,
                 physics_engine: PhysicsEngine,
                 object_a: PhysicsObject,
                 object_b: PhysicsObject,
                 anchor_a: Vector2D,
                 anchor_b: Vector2D,
                 min_distance: float = 0.0,
                 max_distance: float = float('inf'),
                 name: str = "SlideJoint"):
        """
        Initialize a slide joint.
        
        Args:
            physics_engine: Physics engine instance
            object_a: First physics object
            object_b: Second physics object
            anchor_a: Anchor point on object A (local coordinates)
            anchor_b: Anchor point on object B (local coordinates)
            min_distance: Minimum allowed distance
            max_distance: Maximum allowed distance
            name: Joint name
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        if min_distance < 0 or max_distance < min_distance:
            raise ValueError("Invalid distance constraints")
        
        super().__init__(physics_engine, object_a, object_b, ConstraintType.SLIDE_JOINT,
                        anchor_a, anchor_b, name)
        
        if not self._create_constraint():
            raise PhysicsError(f"Failed to create slide joint: {name}")
    
    def _create_constraint(self) -> bool:
        """Create the slide joint in the physics engine."""
        try:
            # TODO: Implement slide joint creation in physics engine
            self._constraint_id = 2  # Placeholder
            
            self.logger.debug("SlideJoint created", extra={
                "name": self.name,
                "constraint_id": self._constraint_id,
                "min_distance": self.min_distance,
                "max_distance": self.max_distance
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create SlideJoint", extra={
                "name": self.name,
                "error": str(e)
            })
            return False


class PivotJoint(Joint):
    """
    Pivot joint that constrains two objects to rotate around a common point.
    """
    
    def __init__(self,
                 physics_engine: PhysicsEngine,
                 object_a: PhysicsObject,
                 object_b: PhysicsObject,
                 pivot_point: Vector2D,
                 name: str = "PivotJoint"):
        """
        Initialize a pivot joint.
        
        Args:
            physics_engine: Physics engine instance
            object_a: First physics object
            object_b: Second physics object
            pivot_point: Pivot point in world coordinates
            name: Joint name
        """
        # Convert world pivot to local anchors
        # TODO: Implement proper world-to-local coordinate conversion
        anchor_a = pivot_point  # Simplified for now
        anchor_b = pivot_point  # Simplified for now
        
        super().__init__(physics_engine, object_a, object_b, ConstraintType.PIVOT_JOINT,
                        anchor_a, anchor_b, name)
        
        self.pivot_point = pivot_point
        
        if not self._create_constraint():
            raise PhysicsError(f"Failed to create pivot joint: {name}")
    
    def _create_constraint(self) -> bool:
        """Create the pivot joint in the physics engine."""
        try:
            # TODO: Implement pivot joint creation in physics engine
            self._constraint_id = 3  # Placeholder
            
            self.logger.debug("PivotJoint created", extra={
                "name": self.name,
                "constraint_id": self._constraint_id,
                "pivot_point": self.pivot_point.to_tuple()
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create PivotJoint", extra={
                "name": self.name,
                "error": str(e)
            })
            return False


class MotorJoint(Constraint):
    """
    Motor joint that applies rotational or linear motion between two objects.
    """
    
    def __init__(self,
                 physics_engine: PhysicsEngine,
                 object_a: PhysicsObject,
                 object_b: PhysicsObject,
                 motor_speed: float = 0.0,
                 max_force: float = float('inf'),
                 name: str = "MotorJoint"):
        """
        Initialize a motor joint.
        
        Args:
            physics_engine: Physics engine instance
            object_a: First physics object
            object_b: Second physics object
            motor_speed: Target motor speed
            max_force: Maximum force the motor can apply
            name: Joint name
        """
        self.motor_speed = motor_speed
        self.max_force = max_force
        
        super().__init__(physics_engine, object_a, object_b, ConstraintType.MOTOR_JOINT, name)
        
        if not self._create_constraint():
            raise PhysicsError(f"Failed to create motor joint: {name}")
    
    def _create_constraint(self) -> bool:
        """Create the motor joint in the physics engine."""
        try:
            # TODO: Implement motor joint creation in physics engine
            self._constraint_id = 4  # Placeholder
            
            self.logger.debug("MotorJoint created", extra={
                "name": self.name,
                "constraint_id": self._constraint_id,
                "motor_speed": self.motor_speed,
                "max_force": self.max_force
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create MotorJoint", extra={
                "name": self.name,
                "error": str(e)
            })
            return False
    
    def set_motor_speed(self, speed: float) -> bool:
        """
        Set the motor speed.
        
        Args:
            speed: New motor speed
            
        Returns:
            True if successful, False otherwise
        """
        self.motor_speed = speed
        # TODO: Update motor speed in physics engine
        return True
    
    def set_max_force(self, force: float) -> bool:
        """
        Set the maximum motor force.
        
        Args:
            force: New maximum force
            
        Returns:
            True if successful, False otherwise
        """
        self.max_force = max(0.0, force)
        # TODO: Update max force in physics engine
        return True


class SpringJoint(Joint):
    """
    Spring joint that applies spring forces between two objects.
    """
    
    def __init__(self,
                 physics_engine: PhysicsEngine,
                 object_a: PhysicsObject,
                 object_b: PhysicsObject,
                 anchor_a: Vector2D,
                 anchor_b: Vector2D,
                 rest_length: float,
                 spring_constant: float = 1000.0,
                 damping: float = 50.0,
                 name: str = "SpringJoint"):
        """
        Initialize a spring joint.
        
        Args:
            physics_engine: Physics engine instance
            object_a: First physics object
            object_b: Second physics object
            anchor_a: Anchor point on object A (local coordinates)
            anchor_b: Anchor point on object B (local coordinates)
            rest_length: Natural length of the spring
            spring_constant: Spring stiffness
            damping: Spring damping coefficient
            name: Joint name
        """
        self.rest_length = rest_length
        self.spring_constant = spring_constant
        self.damping = damping
        
        if rest_length < 0 or spring_constant < 0 or damping < 0:
            raise ValueError("Spring parameters must be non-negative")
        
        super().__init__(physics_engine, object_a, object_b, ConstraintType.SPRING_JOINT,
                        anchor_a, anchor_b, name)
        
        if not self._create_constraint():
            raise PhysicsError(f"Failed to create spring joint: {name}")
    
    def _create_constraint(self) -> bool:
        """Create the spring joint in the physics engine."""
        try:
            # TODO: Implement spring joint creation in physics engine
            self._constraint_id = 5  # Placeholder
            
            self.logger.debug("SpringJoint created", extra={
                "name": self.name,
                "constraint_id": self._constraint_id,
                "rest_length": self.rest_length,
                "spring_constant": self.spring_constant,
                "damping": self.damping
            })
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to create SpringJoint", extra={
                "name": self.name,
                "error": str(e)
            })
            return False
    
    def set_spring_properties(self, 
                            spring_constant: Optional[float] = None,
                            damping: Optional[float] = None,
                            rest_length: Optional[float] = None) -> bool:
        """
        Update spring properties.
        
        Args:
            spring_constant: New spring constant (optional)
            damping: New damping coefficient (optional)
            rest_length: New rest length (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if spring_constant is not None:
            self.spring_constant = max(0.0, spring_constant)
        
        if damping is not None:
            self.damping = max(0.0, damping)
        
        if rest_length is not None:
            self.rest_length = max(0.0, rest_length)
        
        # TODO: Update spring properties in physics engine
        return True
    
    def get_current_length(self) -> Optional[float]:
        """
        Get the current length of the spring.
        
        Returns:
            Current spring length or None if not available
        """
        # TODO: Calculate current spring length from object positions
        return None
    
    def get_spring_force(self) -> Optional[float]:
        """
        Get the current spring force magnitude.
        
        Returns:
            Spring force magnitude or None if not available
        """
        current_length = self.get_current_length()
        if current_length is None:
            return None
        
        extension = current_length - self.rest_length
        return self.spring_constant * extension


class ConstraintManager:
    """
    Manager for physics constraints and joints.
    
    Provides high-level interface for creating and managing constraints.
    """
    
    def __init__(self, physics_engine: PhysicsEngine):
        """
        Initialize the constraint manager.
        
        Args:
            physics_engine: Physics engine instance
        """
        self.physics_engine = physics_engine
        self.logger = get_logger("constraint_manager")
        
        # Constraint tracking
        self._constraints: Dict[int, Constraint] = {}
        self._constraint_groups: Dict[str, List[int]] = {}
        self._next_constraint_id = 1
        
        # Break event tracking
        self._break_callbacks: List[callable] = []
        
        self.logger.debug("ConstraintManager initialized")
    
    def add_constraint(self, constraint: Constraint, group: str = "default") -> int:
        """
        Add a constraint to management.
        
        Args:
            constraint: Constraint to add
            group: Group name for organization
            
        Returns:
            Assigned constraint ID
        """
        constraint_id = self._next_constraint_id
        self._next_constraint_id += 1
        
        self._constraints[constraint_id] = constraint
        
        if group not in self._constraint_groups:
            self._constraint_groups[group] = []
        self._constraint_groups[group].append(constraint_id)
        
        self.logger.debug("Constraint added to manager", extra={
            "constraint_id": constraint_id,
            "constraint_name": constraint.name,
            "group": group
        })
        
        return constraint_id
    
    def remove_constraint(self, constraint_id: int) -> bool:
        """
        Remove a constraint from management.
        
        Args:
            constraint_id: ID of constraint to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        if constraint_id not in self._constraints:
            return False
        
        constraint = self._constraints[constraint_id]
        
        # Remove from groups
        for group_constraints in self._constraint_groups.values():
            if constraint_id in group_constraints:
                group_constraints.remove(constraint_id)
        
        # Destroy constraint
        constraint.destroy()
        del self._constraints[constraint_id]
        
        self.logger.debug("Constraint removed from manager", extra={
            "constraint_id": constraint_id,
            "constraint_name": constraint.name
        })
        
        return True
    
    def get_constraint(self, constraint_id: int) -> Optional[Constraint]:
        """
        Get a constraint by ID.
        
        Args:
            constraint_id: Constraint ID
            
        Returns:
            Constraint object or None if not found
        """
        return self._constraints.get(constraint_id)
    
    def get_constraints_in_group(self, group: str) -> List[Constraint]:
        """
        Get all constraints in a group.
        
        Args:
            group: Group name
            
        Returns:
            List of constraints in the group
        """
        if group not in self._constraint_groups:
            return []
        
        constraints = []
        for constraint_id in self._constraint_groups[group]:
            if constraint_id in self._constraints:
                constraints.append(self._constraints[constraint_id])
        
        return constraints
    
    def remove_group(self, group: str) -> int:
        """
        Remove all constraints in a group.
        
        Args:
            group: Group name
            
        Returns:
            Number of constraints removed
        """
        if group not in self._constraint_groups:
            return 0
        
        constraint_ids = self._constraint_groups[group].copy()
        count = 0
        
        for constraint_id in constraint_ids:
            if self.remove_constraint(constraint_id):
                count += 1
        
        del self._constraint_groups[group]
        
        self.logger.debug("Constraint group removed", extra={
            "group": group,
            "constraints_removed": count
        })
        
        return count
    
    def get_constraint_count(self) -> int:
        """Get total number of managed constraints."""
        return len(self._constraints)
    
    def get_group_names(self) -> List[str]:
        """Get list of all group names."""
        return list(self._constraint_groups.keys())
    
    def add_break_callback(self, callback: callable) -> None:
        """Add a callback for constraint break events."""
        self._break_callbacks.append(callback)
    
    def remove_break_callback(self, callback: callable) -> None:
        """Remove a constraint break callback."""
        if callback in self._break_callbacks:
            self._break_callbacks.remove(callback)


# Convenience factory functions
def create_pin_joint(physics_engine: PhysicsEngine,
                    object_a: PhysicsObject,
                    object_b: PhysicsObject,
                    world_point: Vector2D) -> PinJoint:
    """
    Create a pin joint at a world point.
    
    Args:
        physics_engine: Physics engine instance
        object_a: First object
        object_b: Second object
        world_point: Joint location in world coordinates
        
    Returns:
        Created pin joint
    """
    # TODO: Convert world point to local anchors properly
    anchor_a = world_point - object_a.position
    anchor_b = world_point - object_b.position
    
    return PinJoint(physics_engine, object_a, object_b, anchor_a, anchor_b)


def create_fixed_joint(physics_engine: PhysicsEngine,
                      object_a: PhysicsObject,
                      object_b: PhysicsObject) -> List[Constraint]:
    """
    Create a fixed joint (pin + rotation constraint).
    
    Args:
        physics_engine: Physics engine instance
        object_a: First object
        object_b: Second object
        
    Returns:
        List of constraints that make up the fixed joint
    """
    # Create pin joint at object A's position
    pin_joint = PinJoint(physics_engine, object_a, object_b, 
                        Vector2D(0, 0), object_b.position - object_a.position)
    
    # TODO: Add rotation constraint when implemented
    
    return [pin_joint]


def create_hinge_joint(physics_engine: PhysicsEngine,
                      object_a: PhysicsObject,
                      object_b: PhysicsObject,
                      world_point: Vector2D,
                      motor_speed: float = 0.0) -> List[Constraint]:
    """
    Create a hinge joint (pivot + optional motor).
    
    Args:
        physics_engine: Physics engine instance
        object_a: First object
        object_b: Second object
        world_point: Hinge location in world coordinates
        motor_speed: Optional motor speed
        
    Returns:
        List of constraints that make up the hinge joint
    """
    constraints = []
    
    # Create pivot joint
    pivot = PivotJoint(physics_engine, object_a, object_b, world_point)
    constraints.append(pivot)
    
    # Add motor if requested
    if motor_speed != 0.0:
        motor = MotorJoint(physics_engine, object_a, object_b, motor_speed)
        constraints.append(motor)
    
    return constraints