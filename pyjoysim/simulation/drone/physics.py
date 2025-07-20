"""
Quadrotor drone physics simulation.

This module implements realistic quadrotor physics including:
- 4-rotor thrust and torque calculations
- Aerodynamic effects
- Battery consumption model
- Wind effects
- Crash detection
"""

import math
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...physics.physics3d import Vector3D, Quaternion, PhysicsObject3D, PhysicsMaterial3D
from ...core.logging import get_logger


class RotorDirection(Enum):
    """Rotor rotation direction."""
    CLOCKWISE = 1
    COUNTERCLOCKWISE = -1


@dataclass
class RotorConfiguration:
    """Configuration for a single rotor."""
    position: Vector3D  # Position relative to drone center
    direction: RotorDirection  # Rotation direction
    thrust_coefficient: float = 1.0  # Thrust scaling factor
    torque_coefficient: float = 0.1  # Torque scaling factor
    max_rpm: float = 8000.0  # Maximum RPM


@dataclass
class DronePhysicsParameters:
    """Physical parameters for drone simulation."""
    # Basic properties
    mass: float = 1.5  # kg
    arm_length: float = 0.25  # m (distance from center to rotor)
    
    # Moment of inertia (kg⋅m²)
    inertia_xx: float = 0.029
    inertia_yy: float = 0.029  
    inertia_zz: float = 0.055
    
    # Rotor properties
    rotor_count: int = 4
    max_thrust_per_rotor: float = 6.0  # N
    rotor_time_constant: float = 0.1  # s (rotor spin-up time)
    
    # Aerodynamic coefficients
    drag_coefficient: float = 0.1
    air_density: float = 1.225  # kg/m³ at sea level
    reference_area: float = 0.1  # m²
    
    # Battery model
    battery_capacity: float = 5000  # mAh
    battery_voltage: float = 11.1  # V
    hover_current: float = 15.0  # A
    
    # Crash parameters
    max_tilt_angle: float = math.pi / 3  # 60 degrees
    max_velocity: float = 20.0  # m/s
    min_battery_voltage: float = 9.0  # V


class QuadrotorPhysics:
    """
    Quadrotor physics simulation with realistic flight dynamics.
    
    Implements:
    - 4-rotor thrust and torque generation
    - Aerodynamic drag and effects
    - Battery consumption
    - Wind effects
    - Crash detection
    """
    
    def __init__(self, params: Optional[DronePhysicsParameters] = None):
        """
        Initialize quadrotor physics.
        
        Args:
            params: Physical parameters for the drone
        """
        self.logger = get_logger("quadrotor_physics")
        
        # Use default parameters if none provided
        self.params = params or DronePhysicsParameters()
        
        # Rotor configuration for standard X-configuration quadrotor
        self._setup_rotor_configuration()
        
        # Current rotor states
        self.rotor_speeds = np.zeros(4)  # Current RPM for each rotor
        self.rotor_commands = np.zeros(4)  # Commanded RPM for each rotor
        
        # Battery state
        self.battery_charge = self.params.battery_capacity  # mAh remaining
        self.battery_voltage = self.params.battery_voltage  # Current voltage
        
        # Wind effects
        self.wind_velocity = Vector3D()  # World-space wind velocity
        self.turbulence_strength = 0.0  # Turbulence intensity
        
        # Crash state
        self.is_crashed = False
        self.crash_reason = ""
        
        # Physics integration
        self.last_update_time = 0.0
        
        self.logger.info("QuadrotorPhysics initialized", extra={
            "mass": self.params.mass,
            "rotor_count": self.params.rotor_count,
            "max_thrust": self.params.max_thrust_per_rotor * self.params.rotor_count
        })
    
    def _setup_rotor_configuration(self):
        """Setup standard X-configuration quadrotor rotor layout."""
        arm_length = self.params.arm_length
        
        # Standard X-configuration:
        # Rotor 0 (front-right): CW
        # Rotor 1 (back-left): CW  
        # Rotor 2 (front-left): CCW
        # Rotor 3 (back-right): CCW
        
        self.rotors = [
            RotorConfiguration(
                position=Vector3D(arm_length, 0, arm_length),  # Front-right
                direction=RotorDirection.CLOCKWISE
            ),
            RotorConfiguration(
                position=Vector3D(-arm_length, 0, -arm_length),  # Back-left
                direction=RotorDirection.CLOCKWISE
            ),
            RotorConfiguration(
                position=Vector3D(-arm_length, 0, arm_length),  # Front-left
                direction=RotorDirection.COUNTERCLOCKWISE
            ),
            RotorConfiguration(
                position=Vector3D(arm_length, 0, -arm_length),  # Back-right
                direction=RotorDirection.COUNTERCLOCKWISE
            )
        ]
    
    def set_rotor_commands(self, commands: List[float]):
        """
        Set rotor speed commands (0.0 to 1.0).
        
        Args:
            commands: List of 4 rotor commands (0.0 = stopped, 1.0 = max speed)
        """
        if len(commands) != 4:
            raise ValueError("Must provide exactly 4 rotor commands")
        
        # Convert to RPM
        max_rpm = self.rotors[0].max_rpm
        self.rotor_commands = np.array([cmd * max_rpm for cmd in commands])
        
        # Clamp to valid range
        self.rotor_commands = np.clip(self.rotor_commands, 0, max_rpm)
    
    def update(self, dt: float, physics_object: PhysicsObject3D) -> Tuple[Vector3D, Vector3D]:
        """
        Update quadrotor physics and return forces/torques to apply.
        
        Args:
            dt: Time step in seconds
            physics_object: The drone's physics object
            
        Returns:
            Tuple of (force, torque) to apply to the physics object
        """
        if self.is_crashed:
            return Vector3D(), Vector3D()
        
        # Update rotor speeds (first-order lag)
        self._update_rotor_speeds(dt)
        
        # Calculate forces and torques
        thrust_force, thrust_torque = self._calculate_thrust_forces()
        drag_force = self._calculate_drag_force(physics_object)
        wind_force = self._calculate_wind_effects(physics_object)
        
        # Total force and torque
        total_force = thrust_force + drag_force + wind_force
        total_torque = thrust_torque
        
        # Update battery
        self._update_battery(dt)
        
        # Check for crash conditions
        self._check_crash_conditions(physics_object)
        
        return total_force, total_torque
    
    def _update_rotor_speeds(self, dt: float):
        """Update rotor speeds with first-order lag."""
        time_constant = self.params.rotor_time_constant
        
        for i in range(4):
            # First-order lag: new_speed = old_speed + (command - old_speed) * dt / time_constant
            speed_diff = self.rotor_commands[i] - self.rotor_speeds[i]
            self.rotor_speeds[i] += speed_diff * dt / time_constant
    
    def _calculate_thrust_forces(self) -> Tuple[Vector3D, Vector3D]:
        """Calculate thrust forces and torques from rotors."""
        total_force = Vector3D()
        total_torque = Vector3D()
        
        for i, rotor in enumerate(self.rotors):
            # Thrust force (always upward in rotor frame)
            rpm = self.rotor_speeds[i]
            thrust_magnitude = self._rpm_to_thrust(rpm)
            thrust_force_local = Vector3D(0, thrust_magnitude, 0)
            
            # Torque from thrust (around Z-axis)
            torque_magnitude = self._rpm_to_torque(rpm)
            thrust_torque_local = Vector3D(0, 0, torque_magnitude * rotor.direction.value)
            
            # Apply rotor position offset for moment calculation
            moment_arm = rotor.position
            induced_torque = Vector3D(
                thrust_magnitude * moment_arm.z,  # Roll torque
                0,  # No pitch torque for symmetric rotor
                -thrust_magnitude * moment_arm.x  # Yaw torque
            )
            
            # Accumulate forces and torques
            total_force += thrust_force_local
            total_torque += thrust_torque_local + induced_torque
        
        return total_force, total_torque
    
    def _rpm_to_thrust(self, rpm: float) -> float:
        """Convert RPM to thrust force."""
        # Simplified thrust model: T = k * rpm²
        k_thrust = self.params.max_thrust_per_rotor / (self.rotors[0].max_rpm ** 2)
        return k_thrust * rpm * rpm
    
    def _rpm_to_torque(self, rpm: float) -> float:
        """Convert RPM to rotor torque."""
        # Simplified torque model: Q = k * rpm²
        k_torque = 0.01  # Empirical constant
        return k_torque * rpm * rpm
    
    def _calculate_drag_force(self, physics_object: PhysicsObject3D) -> Vector3D:
        """Calculate aerodynamic drag force."""
        velocity = physics_object.linear_velocity
        speed = velocity.magnitude()
        
        if speed < 0.1:  # Avoid division by zero
            return Vector3D()
        
        # Drag force: F_drag = -0.5 * ρ * Cd * A * v² * (v / |v|)
        drag_magnitude = (0.5 * self.params.air_density * 
                         self.params.drag_coefficient * 
                         self.params.reference_area * 
                         speed * speed)
        
        # Direction opposite to velocity
        drag_direction = velocity.normalized() * -1
        
        return drag_direction * drag_magnitude
    
    def _calculate_wind_effects(self, physics_object: PhysicsObject3D) -> Vector3D:
        """Calculate wind and turbulence effects."""
        if self.wind_velocity.magnitude() < 0.1 and self.turbulence_strength < 0.1:
            return Vector3D()
        
        # Relative wind velocity (drone velocity relative to air)
        relative_velocity = physics_object.linear_velocity - self.wind_velocity
        
        # Add turbulence (random variations)
        if self.turbulence_strength > 0:
            turbulence = Vector3D(
                np.random.normal(0, self.turbulence_strength),
                np.random.normal(0, self.turbulence_strength),
                np.random.normal(0, self.turbulence_strength)
            )
            relative_velocity += turbulence
        
        # Wind force is similar to drag but uses relative velocity
        speed = relative_velocity.magnitude()
        if speed < 0.1:
            return Vector3D()
        
        wind_force_magnitude = (0.5 * self.params.air_density * 
                               self.params.drag_coefficient * 
                               self.params.reference_area * 
                               speed * speed)
        
        wind_direction = relative_velocity.normalized() * -1
        return wind_direction * wind_force_magnitude
    
    def _update_battery(self, dt: float):
        """Update battery charge and voltage."""
        if self.battery_charge <= 0:
            return
        
        # Calculate current consumption based on rotor speeds
        total_power = 0
        for rpm in self.rotor_speeds:
            # Power consumption increases with RPM²
            normalized_rpm = rpm / self.rotors[0].max_rpm
            power = self.params.hover_current * (normalized_rpm ** 2) * self.params.battery_voltage
            total_power += power
        
        # Convert power to current (P = V * I)
        current = total_power / self.params.battery_voltage
        
        # Update charge (mAh)
        charge_used = current * (dt / 3600)  # Convert seconds to hours
        self.battery_charge = max(0, self.battery_charge - charge_used)
        
        # Update voltage (simplified linear drop)
        voltage_drop = (1 - self.battery_charge / self.params.battery_capacity) * 2.0
        self.battery_voltage = max(self.params.min_battery_voltage, 
                                  self.params.battery_voltage - voltage_drop)
    
    def _check_crash_conditions(self, physics_object: PhysicsObject3D):
        """Check for crash conditions."""
        if self.is_crashed:
            return
        
        # Get drone orientation (convert quaternion to euler angles)
        euler = physics_object.rotation.to_euler()
        tilt_angle = max(abs(euler.x), abs(euler.z))  # Roll or pitch
        
        # Check tilt angle
        if tilt_angle > self.params.max_tilt_angle:
            self._crash("Excessive tilt angle")
            return
        
        # Check velocity
        speed = physics_object.linear_velocity.magnitude()
        if speed > self.params.max_velocity:
            self._crash("Excessive velocity")
            return
        
        # Check battery
        if self.battery_voltage < self.params.min_battery_voltage:
            self._crash("Battery depleted")
            return
        
        # Check ground collision (simplified)
        if physics_object.position.y < 0.1:  # Very close to ground
            if speed > 2.0:  # Hard landing
                self._crash("Hard landing")
                return
    
    def _crash(self, reason: str):
        """Set drone to crashed state."""
        self.is_crashed = True
        self.crash_reason = reason
        self.rotor_commands.fill(0)
        self.rotor_speeds.fill(0)
        
        self.logger.warning("Drone crashed", extra={"reason": reason})
    
    def reset(self):
        """Reset drone to initial state."""
        self.rotor_speeds.fill(0)
        self.rotor_commands.fill(0)
        self.battery_charge = self.params.battery_capacity
        self.battery_voltage = self.params.battery_voltage
        self.is_crashed = False
        self.crash_reason = ""
        self.wind_velocity = Vector3D()
        self.turbulence_strength = 0.0
        
        self.logger.debug("Drone physics reset")
    
    def set_wind(self, velocity: Vector3D, turbulence: float = 0.0):
        """
        Set wind conditions.
        
        Args:
            velocity: Wind velocity vector in world coordinates
            turbulence: Turbulence intensity (0.0 = none, 1.0 = strong)
        """
        self.wind_velocity = velocity
        self.turbulence_strength = max(0.0, min(1.0, turbulence))
    
    def get_status(self) -> dict:
        """Get current drone status."""
        return {
            "rotor_speeds": self.rotor_speeds.tolist(),
            "rotor_commands": self.rotor_commands.tolist(),
            "battery_charge": self.battery_charge,
            "battery_voltage": self.battery_voltage,
            "battery_percentage": (self.battery_charge / self.params.battery_capacity) * 100,
            "is_crashed": self.is_crashed,
            "crash_reason": self.crash_reason,
            "wind_speed": self.wind_velocity.magnitude(),
            "turbulence": self.turbulence_strength
        }
    
    def get_rotor_thrust_percentage(self) -> List[float]:
        """Get current thrust percentage for each rotor (0.0 to 1.0)."""
        max_thrust = self.params.max_thrust_per_rotor
        thrusts = []
        
        for rpm in self.rotor_speeds:
            thrust = self._rpm_to_thrust(rpm)
            percentage = thrust / max_thrust
            thrusts.append(min(1.0, percentage))
        
        return thrusts