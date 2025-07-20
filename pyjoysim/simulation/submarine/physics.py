"""
Submarine physics system with underwater mechanics.

This module implements realistic underwater physics including:
- Buoyancy and hydrostatic pressure
- Water resistance and drag
- Ocean currents and thermoclines
- Submarine dynamics and control surfaces
- Pressure hull stress modeling
"""

import math
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...physics.physics3d import Vector3D, Quaternion, PhysicsObject3D
from ...core.logging import get_logger


class WaterType(Enum):
    """Types of water environments."""
    FRESH_WATER = "fresh_water"
    SALT_WATER = "salt_water"
    ARCTIC_WATER = "arctic_water"
    TROPICAL_WATER = "tropical_water"


@dataclass
class UnderwaterLayer:
    """Represents a layer of water with different properties."""
    depth_min: float  # meters
    depth_max: float  # meters
    temperature: float  # Celsius
    salinity: float   # parts per thousand
    density: float    # kg/m³
    current_velocity: Vector3D  # m/s
    turbidity: float  # 0.0 to 1.0 (visibility factor)


@dataclass
class SubmarinePhysicsParameters:
    """Parameters for submarine physics simulation."""
    # Basic properties
    length: float = 80.0          # meters
    beam: float = 8.0             # meters (width)
    height: float = 10.0          # meters
    displacement: float = 3000.0  # tonnes when surfaced
    
    # Hull properties
    pressure_hull_thickness: float = 0.05  # meters
    max_operating_depth: float = 300.0     # meters
    crush_depth: float = 600.0             # meters
    
    # Propulsion
    max_power: float = 5000.0     # kW
    propeller_diameter: float = 3.0  # meters
    max_speed_surface: float = 20.0  # knots
    max_speed_submerged: float = 15.0  # knots
    
    # Control surfaces
    rudder_area: float = 15.0     # m²
    stern_plane_area: float = 10.0  # m²
    sail_plane_area: float = 8.0   # m²
    
    # Ballast and trim
    main_ballast_capacity: float = 500.0  # m³
    trim_tank_capacity: float = 50.0      # m³
    
    # Drag coefficients
    drag_coefficient_surface: float = 0.05
    drag_coefficient_submerged: float = 0.08


class UnderwaterEnvironment:
    """
    Manages the underwater environment including ocean layers and conditions.
    """
    
    def __init__(self, water_type: WaterType = WaterType.SALT_WATER):
        """Initialize underwater environment."""
        self.logger = get_logger("underwater_environment")
        self.water_type = water_type
        
        # Water layers (thermoclines, etc.)
        self.water_layers: List[UnderwaterLayer] = []
        
        # Environmental parameters
        self.sea_level_pressure = 101325.0  # Pa (1 atmosphere)
        self.gravity = 9.81  # m/s²
        
        # Ocean floor
        self.ocean_floor_depth = 4000.0  # meters
        self.seafloor_topology = {}  # Could store height map
        
        # Marine life and obstacles
        self.marine_obstacles = []
        
        # Weather effects
        self.surface_wave_height = 2.0  # meters
        self.surface_current = Vector3D(0.5, 0, 0.2)  # m/s
        
        # Create realistic water column
        self._create_water_column()
        
        self.logger.info(f"Underwater environment initialized ({water_type.value})")
    
    def _create_water_column(self):
        """Create realistic water column with thermoclines."""
        if self.water_type == WaterType.SALT_WATER:
            # Typical ocean layers
            self.water_layers = [
                # Surface mixed layer
                UnderwaterLayer(
                    depth_min=0.0, depth_max=50.0,
                    temperature=20.0, salinity=35.0, density=1025.0,
                    current_velocity=Vector3D(0.3, 0, 0.1),
                    turbidity=0.1
                ),
                # Thermocline
                UnderwaterLayer(
                    depth_min=50.0, depth_max=200.0,
                    temperature=8.0, salinity=35.0, density=1026.0,
                    current_velocity=Vector3D(0.1, 0, 0.05),
                    turbidity=0.05
                ),
                # Deep water
                UnderwaterLayer(
                    depth_min=200.0, depth_max=2000.0,
                    temperature=4.0, salinity=34.8, density=1028.0,
                    current_velocity=Vector3D(0.05, 0, 0.02),
                    turbidity=0.02
                ),
                # Abyssal
                UnderwaterLayer(
                    depth_min=2000.0, depth_max=6000.0,
                    temperature=2.0, salinity=34.7, density=1029.0,
                    current_velocity=Vector3D(0.02, 0, 0.01),
                    turbidity=0.01
                )
            ]
        else:
            # Fresh water (simplified)
            self.water_layers = [
                UnderwaterLayer(
                    depth_min=0.0, depth_max=1000.0,
                    temperature=15.0, salinity=0.0, density=1000.0,
                    current_velocity=Vector3D(0.1, 0, 0.05),
                    turbidity=0.1
                )
            ]
    
    def get_water_properties(self, depth: float) -> UnderwaterLayer:
        """Get water properties at specified depth."""
        depth = abs(depth)  # Depth is positive downward
        
        for layer in self.water_layers:
            if layer.depth_min <= depth <= layer.depth_max:
                return layer
        
        # Return deepest layer if beyond known depths
        return self.water_layers[-1] if self.water_layers else UnderwaterLayer(
            depth_min=0, depth_max=1000, temperature=4.0, salinity=35.0, 
            density=1025.0, current_velocity=Vector3D(), turbidity=0.1
        )
    
    def calculate_pressure(self, depth: float) -> float:
        """
        Calculate water pressure at given depth.
        
        Args:
            depth: Depth in meters (positive downward)
            
        Returns:
            Pressure in Pascals
        """
        depth = abs(depth)
        
        # Get water density at this depth
        layer = self.get_water_properties(depth)
        
        # Hydrostatic pressure: P = P0 + ρgh
        pressure = self.sea_level_pressure + (layer.density * self.gravity * depth)
        
        return pressure
    
    def calculate_buoyancy_force(self, submerged_volume: float, depth: float) -> float:
        """
        Calculate buoyancy force.
        
        Args:
            submerged_volume: Volume of submarine submerged in m³
            depth: Current depth in meters
            
        Returns:
            Buoyancy force in Newtons (upward positive)
        """
        if submerged_volume <= 0:
            return 0.0
        
        layer = self.get_water_properties(depth)
        buoyancy = layer.density * self.gravity * submerged_volume
        
        return buoyancy
    
    def calculate_drag_force(self, velocity: Vector3D, cross_sectional_area: float, 
                           drag_coefficient: float, depth: float) -> Vector3D:
        """
        Calculate drag force in water.
        
        Args:
            velocity: Velocity vector
            cross_sectional_area: Cross-sectional area facing flow
            drag_coefficient: Drag coefficient
            depth: Current depth
            
        Returns:
            Drag force vector (opposing motion)
        """
        if velocity.magnitude() == 0:
            return Vector3D()
        
        layer = self.get_water_properties(depth)
        
        # Add current to relative velocity
        relative_velocity = velocity - layer.current_velocity
        speed = relative_velocity.magnitude()
        
        if speed == 0:
            return Vector3D()
        
        # Drag force: F = 0.5 * ρ * v² * Cd * A
        drag_magnitude = 0.5 * layer.density * speed * speed * drag_coefficient * cross_sectional_area
        
        # Drag opposes motion
        drag_direction = -relative_velocity.normalized()
        
        return drag_direction * drag_magnitude
    
    def get_visibility_range(self, depth: float) -> float:
        """
        Get visibility range at given depth.
        
        Args:
            depth: Current depth in meters
            
        Returns:
            Visibility range in meters
        """
        layer = self.get_water_properties(depth)
        
        # Base visibility affected by turbidity and depth
        base_visibility = 100.0  # meters in clear water
        turbidity_factor = 1.0 - layer.turbidity
        depth_factor = max(0.1, 1.0 - (depth / 1000.0))  # Visibility decreases with depth
        
        visibility = base_visibility * turbidity_factor * depth_factor
        
        return max(1.0, visibility)  # Minimum 1 meter visibility


class SubmarinePhysics:
    """
    Advanced submarine physics with underwater dynamics.
    
    Features:
    - Buoyancy and hydrostatic pressure calculations
    - Water resistance and drag modeling
    - Ballast tank simulation
    - Control surface effectiveness
    - Depth and trim control
    """
    
    def __init__(self, params: SubmarinePhysicsParameters, environment: UnderwaterEnvironment):
        """
        Initialize submarine physics.
        
        Args:
            params: Submarine physics parameters
            environment: Underwater environment
        """
        self.logger = get_logger("submarine_physics")
        self.params = params
        self.environment = environment
        
        # Current submarine state
        self.current_depth = 0.0  # meters (positive downward)
        self.is_surfaced = True
        self.submerged_volume = 0.0  # m³
        
        # Control inputs
        self.engine_power = 0.0      # 0.0 to 1.0
        self.rudder_angle = 0.0      # degrees
        self.stern_plane_angle = 0.0 # degrees
        self.sail_plane_angle = 0.0  # degrees
        
        # Ballast and trim
        self.main_ballast_filled = 0.0   # 0.0 to 1.0
        self.trim_tank_balance = 0.0     # -1.0 to 1.0 (forward/aft)
        
        # Performance tracking
        self.total_distance = 0.0
        self.max_depth_reached = 0.0
        self.pressure_hull_stress = 0.0  # Percentage of maximum
        
        # Emergency systems
        self.emergency_blow_active = False
        self.damage_level = 0.0  # 0.0 to 1.0
        
        self.logger.info("Submarine physics initialized")
    
    def calculate_submerged_volume(self, depth: float) -> float:
        """
        Calculate volume of submarine that is submerged.
        
        Args:
            depth: Current depth (positive downward)
            
        Returns:
            Submerged volume in m³
        """
        if depth <= 0:
            return 0.0
        
        # Simplified: assume submarine is fully submerged if keel depth > height/2
        hull_draft = self.params.height / 2
        
        if depth >= hull_draft:
            # Fully submerged
            total_volume = (self.params.length * self.params.beam * self.params.height * 0.7)  # 70% hull volume
            return total_volume
        else:
            # Partially submerged
            submersion_ratio = depth / hull_draft
            total_volume = (self.params.length * self.params.beam * self.params.height * 0.7)
            return total_volume * submersion_ratio
    
    def calculate_propulsion_force(self) -> Vector3D:
        """Calculate propulsion force from engine and propeller."""
        if self.engine_power <= 0:
            return Vector3D()
        
        # Power-to-thrust conversion (simplified)
        max_thrust = self.params.max_power * 1000  # Convert kW to watts, then to Newtons (simplified)
        current_thrust = max_thrust * self.engine_power
        
        # Efficiency depends on depth and speed
        depth_efficiency = 1.0 if self.current_depth > 0 else 0.8  # Better efficiency submerged
        thrust_force = current_thrust * depth_efficiency
        
        # Thrust along submarine's longitudinal axis (X-axis)
        return Vector3D(thrust_force, 0, 0)
    
    def calculate_control_surface_forces(self, velocity: Vector3D) -> Tuple[Vector3D, Vector3D]:
        """
        Calculate forces and torques from control surfaces.
        
        Args:
            velocity: Current velocity vector
            
        Returns:
            Tuple of (force, torque) vectors
        """
        speed = velocity.magnitude()
        if speed < 0.1:  # No control authority at very low speed
            return Vector3D(), Vector3D()
        
        # Dynamic pressure
        layer = self.environment.get_water_properties(self.current_depth)
        dynamic_pressure = 0.5 * layer.density * speed * speed
        
        # Rudder force (yaw control)
        rudder_force_magnitude = dynamic_pressure * self.params.rudder_area * math.sin(math.radians(self.rudder_angle))
        rudder_force = Vector3D(0, 0, rudder_force_magnitude)
        
        # Stern planes (pitch control)
        stern_force_magnitude = dynamic_pressure * self.params.stern_plane_area * math.sin(math.radians(self.stern_plane_angle))
        stern_force = Vector3D(0, stern_force_magnitude, 0)
        
        # Sail planes (pitch control, forward location)
        sail_force_magnitude = dynamic_pressure * self.params.sail_plane_area * math.sin(math.radians(self.sail_plane_angle))
        sail_force = Vector3D(0, sail_force_magnitude, 0)
        
        # Total force
        total_force = rudder_force + stern_force + sail_force
        
        # Torques (simplified lever arms)
        rudder_torque = Vector3D(0, rudder_force_magnitude * (self.params.length * 0.4), 0)  # Yaw
        stern_torque = Vector3D(0, 0, -stern_force_magnitude * (self.params.length * 0.4))   # Pitch
        sail_torque = Vector3D(0, 0, sail_force_magnitude * (self.params.length * 0.3))     # Pitch
        
        total_torque = rudder_torque + stern_torque + sail_torque
        
        return total_force, total_torque
    
    def calculate_ballast_effects(self) -> float:
        """
        Calculate weight change due to ballast tanks.
        
        Returns:
            Additional weight in Newtons
        """
        # Main ballast tanks
        ballast_water_volume = self.main_ballast_filled * self.params.main_ballast_capacity
        
        # Get water density at current depth
        layer = self.environment.get_water_properties(self.current_depth)
        ballast_weight = ballast_water_volume * layer.density * self.environment.gravity
        
        return ballast_weight
    
    def check_pressure_hull_integrity(self) -> bool:
        """
        Check if pressure hull can withstand current depth.
        
        Returns:
            True if hull is safe, False if compromised
        """
        if self.current_depth <= 0:
            self.pressure_hull_stress = 0.0
            return True
        
        # Calculate hull stress as percentage of maximum operating depth
        self.pressure_hull_stress = (self.current_depth / self.params.max_operating_depth) * 100
        
        # Check for hull failure
        if self.current_depth > self.params.crush_depth:
            self.damage_level = min(1.0, self.damage_level + 0.1)  # Catastrophic damage
            return False
        elif self.current_depth > self.params.max_operating_depth:
            self.damage_level = min(1.0, self.damage_level + 0.01)  # Gradual damage
            return False
        
        return True
    
    def update(self, dt: float, submarine_body: PhysicsObject3D) -> Tuple[Vector3D, Vector3D]:
        """
        Update submarine physics and return forces/torques.
        
        Args:
            dt: Time step in seconds
            submarine_body: Physics body representing the submarine
            
        Returns:
            Tuple of (force, torque) to apply to physics body
        """
        # Update current depth
        self.current_depth = max(0.0, -submarine_body.position.y)  # Y is up, depth is positive down
        self.is_surfaced = (self.current_depth < 1.0)
        
        # Calculate submerged volume
        self.submerged_volume = self.calculate_submerged_volume(self.current_depth)
        
        # Calculate buoyancy force
        buoyancy_force = self.environment.calculate_buoyancy_force(self.submerged_volume, self.current_depth)
        buoyancy = Vector3D(0, buoyancy_force, 0)  # Upward
        
        # Calculate weight from ballast
        ballast_weight = self.calculate_ballast_effects()
        weight_force = Vector3D(0, -ballast_weight, 0)  # Downward
        
        # Calculate propulsion
        propulsion_force = self.calculate_propulsion_force()
        
        # Calculate drag
        velocity = submarine_body.linear_velocity
        drag_coefficient = (self.params.drag_coefficient_submerged if not self.is_surfaced 
                          else self.params.drag_coefficient_surface)
        cross_sectional_area = self.params.beam * self.params.height
        drag_force = self.environment.calculate_drag_force(velocity, cross_sectional_area, 
                                                          drag_coefficient, self.current_depth)
        
        # Calculate control surface forces
        control_force, control_torque = self.calculate_control_surface_forces(velocity)
        
        # Check hull integrity
        hull_ok = self.check_pressure_hull_integrity()
        if not hull_ok:
            self.logger.warning(f"Hull stress at {self.pressure_hull_stress:.1f}% - depth {self.current_depth:.1f}m")
        
        # Total forces
        total_force = buoyancy + weight_force + propulsion_force + drag_force + control_force
        total_torque = control_torque
        
        # Update tracking
        distance_delta = velocity.magnitude() * dt
        self.total_distance += distance_delta
        self.max_depth_reached = max(self.max_depth_reached, self.current_depth)
        
        return total_force, total_torque
    
    def set_engine_power(self, power: float):
        """Set engine power (0.0 to 1.0)."""
        self.engine_power = max(0.0, min(1.0, power))
    
    def set_rudder_angle(self, angle: float):
        """Set rudder angle in degrees."""
        self.rudder_angle = max(-30.0, min(30.0, angle))
    
    def set_stern_planes(self, angle: float):
        """Set stern plane angle in degrees."""
        self.stern_plane_angle = max(-20.0, min(20.0, angle))
    
    def set_sail_planes(self, angle: float):
        """Set sail plane angle in degrees."""
        self.sail_plane_angle = max(-15.0, min(15.0, angle))
    
    def set_ballast_level(self, level: float):
        """Set main ballast tank fill level (0.0 to 1.0)."""
        self.main_ballast_filled = max(0.0, min(1.0, level))
    
    def emergency_blow_ballast(self):
        """Emergency blow main ballast tanks."""
        self.emergency_blow_active = True
        self.main_ballast_filled = 0.0
        self.logger.warning("Emergency ballast blow activated")
    
    def get_status(self) -> Dict[str, any]:
        """Get submarine physics status for UI display."""
        layer = self.environment.get_water_properties(self.current_depth)
        pressure = self.environment.calculate_pressure(self.current_depth)
        
        return {
            "depth": self.current_depth,
            "is_surfaced": self.is_surfaced,
            "submerged_volume": self.submerged_volume,
            "pressure": pressure / 1000,  # Convert to kPa
            "pressure_hull_stress": self.pressure_hull_stress,
            "engine_power": self.engine_power * 100,
            "rudder_angle": self.rudder_angle,
            "stern_plane_angle": self.stern_plane_angle,
            "ballast_level": self.main_ballast_filled * 100,
            "emergency_blow_active": self.emergency_blow_active,
            "damage_level": self.damage_level * 100,
            "water_temperature": layer.temperature,
            "water_density": layer.density,
            "visibility_range": self.environment.get_visibility_range(self.current_depth),
            "total_distance": self.total_distance,
            "max_depth_reached": self.max_depth_reached
        }
    
    def reset(self):
        """Reset submarine physics to initial state."""
        self.current_depth = 0.0
        self.is_surfaced = True
        self.submerged_volume = 0.0
        self.engine_power = 0.0
        self.rudder_angle = 0.0
        self.stern_plane_angle = 0.0
        self.sail_plane_angle = 0.0
        self.main_ballast_filled = 0.0
        self.trim_tank_balance = 0.0
        self.total_distance = 0.0
        self.max_depth_reached = 0.0
        self.pressure_hull_stress = 0.0
        self.emergency_blow_active = False
        self.damage_level = 0.0
        
        self.logger.debug("Submarine physics reset")