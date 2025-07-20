"""
Spaceship physics system with orbital mechanics.

This module implements realistic space physics including:
- Zero gravity environment
- Orbital mechanics and Kepler's laws
- Gravitational influences from celestial bodies
- Newtonian physics in space
- Attitude control and momentum conservation
"""

import math
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...physics.physics3d import Vector3D, Quaternion, PhysicsObject3D
from ...core.logging import get_logger


class CelestialBodyType(Enum):
    """Types of celestial bodies."""
    STAR = "star"
    PLANET = "planet"
    MOON = "moon"
    ASTEROID = "asteroid"
    SPACE_STATION = "space_station"


@dataclass
class CelestialBody:
    """Represents a celestial body with gravitational influence."""
    name: str
    body_type: CelestialBodyType
    position: Vector3D
    mass: float  # kg
    radius: float  # meters
    
    # Orbital parameters
    orbital_velocity: Vector3D = None
    parent_body: Optional['CelestialBody'] = None
    
    def __post_init__(self):
        if self.orbital_velocity is None:
            self.orbital_velocity = Vector3D()


@dataclass
class SpaceshipPhysicsParameters:
    """Parameters for spaceship physics simulation."""
    # Basic properties
    dry_mass: float = 500.0  # kg (without fuel)
    fuel_capacity: float = 1000.0  # kg
    
    # Propulsion
    main_engine_thrust: float = 1000.0  # N
    rcs_thrust: float = 100.0  # N per thruster
    specific_impulse: float = 300.0  # seconds (engine efficiency)
    
    # Physical dimensions
    length: float = 10.0  # meters
    width: float = 3.0  # meters
    height: float = 3.0  # meters
    
    # Moments of inertia (kg⋅m²)
    moment_of_inertia_x: float = 1000.0
    moment_of_inertia_y: float = 5000.0
    moment_of_inertia_z: float = 5000.0
    
    # Atmospheric properties (for atmospheric flight)
    drag_coefficient: float = 0.5
    cross_sectional_area: float = 9.0  # m²


class SpaceEnvironment:
    """
    Manages the space environment including celestial bodies and physics.
    """
    
    def __init__(self):
        """Initialize space environment."""
        self.logger = get_logger("space_environment")
        
        # Gravitational constant
        self.G = 6.67430e-11  # m³/kg⋅s²
        
        # Celestial bodies
        self.celestial_bodies: List[CelestialBody] = []
        
        # Environment parameters
        self.vacuum_drag = 0.0  # No air resistance in space
        self.radiation_level = 1.0  # Arbitrary units
        self.temperature = -270.0  # Celsius (near absolute zero)
        
        # Create basic solar system
        self._create_basic_solar_system()
        
        self.logger.info("Space environment initialized")
    
    def _create_basic_solar_system(self):
        """Create a simplified solar system for simulation."""
        # Sun (at origin)
        sun = CelestialBody(
            name="Sun",
            body_type=CelestialBodyType.STAR,
            position=Vector3D(0, 0, 0),
            mass=1.989e30,  # kg
            radius=696340000  # meters
        )
        self.celestial_bodies.append(sun)
        
        # Earth (simplified orbit)
        earth = CelestialBody(
            name="Earth",
            body_type=CelestialBodyType.PLANET,
            position=Vector3D(149597870700, 0, 0),  # 1 AU
            mass=5.972e24,  # kg
            radius=6371000,  # meters
            orbital_velocity=Vector3D(0, 0, 29780),  # m/s
            parent_body=sun
        )
        self.celestial_bodies.append(earth)
        
        # Moon (simplified)
        moon = CelestialBody(
            name="Moon",
            body_type=CelestialBodyType.MOON,
            position=Vector3D(149597870700 + 384400000, 0, 0),  # Earth + Moon distance
            mass=7.342e22,  # kg
            radius=1737400,  # meters
            orbital_velocity=Vector3D(0, 0, 29780 + 1022),  # Earth velocity + Moon orbital velocity
            parent_body=earth
        )
        self.celestial_bodies.append(moon)
        
        self.logger.debug("Basic solar system created with {} bodies".format(len(self.celestial_bodies)))
    
    def calculate_gravitational_force(self, position: Vector3D, mass: float) -> Vector3D:
        """
        Calculate total gravitational force at given position.
        
        Args:
            position: Position in space
            mass: Mass of object
            
        Returns:
            Total gravitational force vector
        """
        total_force = Vector3D()
        
        for body in self.celestial_bodies:
            # Calculate distance vector
            r_vector = body.position - position
            distance = r_vector.magnitude()
            
            # Skip if too close (inside the body)
            if distance < body.radius:
                continue
            
            # Calculate gravitational force magnitude
            force_magnitude = (self.G * body.mass * mass) / (distance ** 2)
            
            # Calculate force direction
            force_direction = r_vector.normalized()
            force = force_direction * force_magnitude
            
            total_force += force
        
        return total_force
    
    def get_nearest_body(self, position: Vector3D) -> Optional[CelestialBody]:
        """Get the nearest celestial body to given position."""
        if not self.celestial_bodies:
            return None
        
        nearest_body = None
        min_distance = float('inf')
        
        for body in self.celestial_bodies:
            distance = (body.position - position).magnitude()
            if distance < min_distance:
                min_distance = distance
                nearest_body = body
        
        return nearest_body
    
    def update(self, dt: float):
        """Update celestial body positions (simplified orbital mechanics)."""
        for body in self.celestial_bodies:
            if body.parent_body is not None:
                # Simple circular orbit update
                body.position += body.orbital_velocity * dt


class SpaceshipPhysics:
    """
    Advanced spaceship physics with orbital mechanics.
    
    Features:
    - Newtonian physics in zero gravity
    - Orbital mechanics calculations
    - Fuel consumption modeling
    - Attitude control via RCS
    - Gravitational influences
    """
    
    def __init__(self, params: SpaceshipPhysicsParameters, environment: SpaceEnvironment):
        """
        Initialize spaceship physics.
        
        Args:
            params: Spaceship physics parameters
            environment: Space environment
        """
        self.logger = get_logger("spaceship_physics")
        self.params = params
        self.environment = environment
        
        # Current state
        self.current_fuel_mass = params.fuel_capacity
        self.main_engine_throttle = 0.0  # 0.0 to 1.0
        self.rcs_thrust_vector = Vector3D()  # Net RCS thrust
        
        # Engine status
        self.main_engine_enabled = True
        self.rcs_enabled = True
        
        # Performance tracking
        self.total_fuel_consumed = 0.0
        self.total_thrust_time = 0.0
        self.delta_v_used = 0.0  # Change in velocity (important in space)
        
        # Orbital parameters (calculated)
        self.orbital_velocity = 0.0
        self.orbital_altitude = 0.0
        self.apogee = 0.0
        self.perigee = 0.0
        
        self.logger.info("Spaceship physics initialized")
    
    def get_total_mass(self) -> float:
        """Get current total mass including fuel."""
        return self.params.dry_mass + self.current_fuel_mass
    
    def calculate_thrust_force(self) -> Vector3D:
        """Calculate total thrust force from all propulsion systems."""
        total_thrust = Vector3D()
        
        # Main engine thrust (typically along +X axis)
        if self.main_engine_enabled and self.current_fuel_mass > 0:
            main_thrust_magnitude = self.params.main_engine_thrust * self.main_engine_throttle
            main_thrust = Vector3D(main_thrust_magnitude, 0, 0)
            total_thrust += main_thrust
        
        # RCS thrust (any direction)
        if self.rcs_enabled and self.current_fuel_mass > 0:
            total_thrust += self.rcs_thrust_vector
        
        return total_thrust
    
    def calculate_fuel_consumption(self, thrust_magnitude: float, dt: float) -> float:
        """
        Calculate fuel consumption based on thrust and time.
        
        Args:
            thrust_magnitude: Total thrust magnitude in Newtons
            dt: Time step in seconds
            
        Returns:
            Fuel consumed in kg
        """
        if thrust_magnitude <= 0:
            return 0.0
        
        # Fuel consumption based on specific impulse
        # fuel_flow = thrust / (specific_impulse * g0)
        g0 = 9.80665  # Standard gravity
        fuel_flow_rate = thrust_magnitude / (self.params.specific_impulse * g0)
        fuel_consumed = fuel_flow_rate * dt
        
        return min(fuel_consumed, self.current_fuel_mass)
    
    def calculate_orbital_parameters(self, position: Vector3D, velocity: Vector3D):
        """Calculate orbital parameters for educational display."""
        # Find nearest major body (usually Earth)
        nearest_body = self.environment.get_nearest_body(position)
        if not nearest_body or nearest_body.body_type == CelestialBodyType.ASTEROID:
            return
        
        # Calculate distance from center
        r_vector = position - nearest_body.position
        distance = r_vector.magnitude()
        
        # Calculate orbital velocity
        self.orbital_velocity = velocity.magnitude()
        self.orbital_altitude = distance - nearest_body.radius
        
        # Simplified orbital parameters (circular orbit approximation)
        if distance > nearest_body.radius:
            mu = self.environment.G * nearest_body.mass  # Gravitational parameter
            
            # Calculate orbital energy
            kinetic_energy = 0.5 * self.orbital_velocity ** 2
            potential_energy = -mu / distance
            total_energy = kinetic_energy + potential_energy
            
            # Calculate semi-major axis
            if total_energy < 0:  # Elliptical orbit
                semi_major_axis = -mu / (2 * total_energy)
                self.apogee = semi_major_axis * 2 - nearest_body.radius
                self.perigee = distance - nearest_body.radius  # Approximation
            else:  # Hyperbolic trajectory
                self.apogee = float('inf')
                self.perigee = distance - nearest_body.radius
    
    def update(self, dt: float, spaceship_body: PhysicsObject3D) -> Tuple[Vector3D, Vector3D]:
        """
        Update spaceship physics and return forces/torques.
        
        Args:
            dt: Time step in seconds
            spaceship_body: Physics body representing the spaceship
            
        Returns:
            Tuple of (force, torque) to apply to physics body
        """
        # Calculate thrust forces
        thrust_force = self.calculate_thrust_force()
        thrust_magnitude = thrust_force.magnitude()
        
        # Calculate fuel consumption
        fuel_consumed = self.calculate_fuel_consumption(thrust_magnitude, dt)
        self.current_fuel_mass -= fuel_consumed
        self.current_fuel_mass = max(0.0, self.current_fuel_mass)
        
        # Update tracking variables
        self.total_fuel_consumed += fuel_consumed
        if thrust_magnitude > 0:
            self.total_thrust_time += dt
            # Calculate delta-v (change in velocity)
            acceleration = thrust_magnitude / self.get_total_mass()
            self.delta_v_used += acceleration * dt
        
        # Calculate gravitational forces
        gravitational_force = self.environment.calculate_gravitational_force(
            spaceship_body.position, self.get_total_mass()
        )
        
        # Total force on spaceship
        total_force = thrust_force + gravitational_force
        
        # Calculate torques (simplified - RCS can provide attitude control)
        torque = Vector3D()  # Attitude control would be implemented here
        
        # Update orbital parameters for display
        self.calculate_orbital_parameters(spaceship_body.position, spaceship_body.linear_velocity)
        
        # Update mass in physics body
        spaceship_body.mass = self.get_total_mass()
        
        return total_force, torque
    
    def set_main_engine_throttle(self, throttle: float):
        """Set main engine throttle (0.0 to 1.0)."""
        self.main_engine_throttle = max(0.0, min(1.0, throttle))
    
    def set_rcs_thrust(self, thrust_vector: Vector3D):
        """Set RCS thrust vector."""
        # Limit RCS thrust magnitude
        max_rcs_magnitude = self.params.rcs_thrust * 8  # Assume 8 RCS thrusters
        if thrust_vector.magnitude() > max_rcs_magnitude:
            self.rcs_thrust_vector = thrust_vector.normalized() * max_rcs_magnitude
        else:
            self.rcs_thrust_vector = thrust_vector
    
    def get_status(self) -> Dict[str, any]:
        """Get current spaceship status for UI display."""
        fuel_percentage = (self.current_fuel_mass / self.params.fuel_capacity) * 100
        
        return {
            "total_mass": self.get_total_mass(),
            "fuel_mass": self.current_fuel_mass,
            "fuel_percentage": fuel_percentage,
            "main_engine_throttle": self.main_engine_throttle * 100,
            "delta_v_remaining": self._calculate_delta_v_remaining(),
            "orbital_velocity": self.orbital_velocity,
            "orbital_altitude": self.orbital_altitude,
            "apogee": self.apogee,
            "perigee": self.perigee,
            "total_fuel_consumed": self.total_fuel_consumed,
            "delta_v_used": self.delta_v_used
        }
    
    def _calculate_delta_v_remaining(self) -> float:
        """Calculate remaining delta-v capability."""
        if self.current_fuel_mass <= 0:
            return 0.0
        
        # Tsiolkovsky rocket equation: Δv = v_e * ln(m0/m1)
        # where v_e = specific_impulse * g0
        g0 = 9.80665
        exhaust_velocity = self.params.specific_impulse * g0
        mass_ratio = self.get_total_mass() / self.params.dry_mass
        
        return exhaust_velocity * math.log(mass_ratio)
    
    def reset(self):
        """Reset spaceship to initial state."""
        self.current_fuel_mass = self.params.fuel_capacity
        self.main_engine_throttle = 0.0
        self.rcs_thrust_vector = Vector3D()
        self.total_fuel_consumed = 0.0
        self.total_thrust_time = 0.0
        self.delta_v_used = 0.0
        
        self.logger.debug("Spaceship physics reset")