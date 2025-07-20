"""
Spaceship propulsion systems.

This module implements various propulsion systems for spaceship simulation:
- Main engine for primary propulsion
- RCS (Reaction Control System) for attitude control
- Fuel management and efficiency calculations
- Engine performance modeling
"""

import math
import time
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...physics.physics3d import Vector3D, Quaternion
from ...core.logging import get_logger


class EngineType(Enum):
    """Types of spacecraft engines."""
    CHEMICAL_ROCKET = "chemical_rocket"
    ION_DRIVE = "ion_drive"
    NUCLEAR_THERMAL = "nuclear_thermal"
    COLD_GAS = "cold_gas"


class ThrusterLocation(Enum):
    """RCS thruster locations on spacecraft."""
    FORWARD_PORT = "forward_port"
    FORWARD_STARBOARD = "forward_starboard"
    FORWARD_TOP = "forward_top"
    FORWARD_BOTTOM = "forward_bottom"
    AFT_PORT = "aft_port"
    AFT_STARBOARD = "aft_starboard"
    AFT_TOP = "aft_top"
    AFT_BOTTOM = "aft_bottom"


@dataclass
class EngineParameters:
    """Parameters for a spacecraft engine."""
    name: str
    engine_type: EngineType
    max_thrust: float  # Newtons
    specific_impulse: float  # seconds
    fuel_flow_rate: float  # kg/s at max thrust
    throttleable: bool = True
    min_throttle: float = 0.0  # Minimum throttle setting
    restart_capable: bool = True
    ignition_delay: float = 0.0  # seconds


@dataclass
class RCSThruster:
    """Individual RCS thruster."""
    location: ThrusterLocation
    position: Vector3D  # Position relative to center of mass
    direction: Vector3D  # Thrust direction (unit vector)
    max_thrust: float  # Newtons
    current_thrust: float = 0.0  # Current thrust setting (0.0 to 1.0)


class PropulsionSystem:
    """
    Main propulsion system for spacecraft.
    
    Manages primary engines for translation and orbital maneuvers.
    """
    
    def __init__(self, engine_params: EngineParameters):
        """
        Initialize propulsion system.
        
        Args:
            engine_params: Engine parameters
        """
        self.logger = get_logger("propulsion_system")
        self.params = engine_params
        
        # Engine state
        self.throttle_setting = 0.0  # 0.0 to 1.0
        self.engine_enabled = True
        self.engine_ignited = False
        self.ignition_timer = 0.0
        
        # Performance tracking
        self.total_burn_time = 0.0
        self.total_fuel_consumed = 0.0
        self.ignition_count = 0
        
        # Temperature and wear (simplified)
        self.engine_temperature = 273.15  # Kelvin (0°C)
        self.wear_factor = 0.0  # 0.0 to 1.0 (1.0 = completely worn out)
        
        self.logger.info(f"Propulsion system initialized: {engine_params.name}")
    
    def set_throttle(self, throttle: float):
        """
        Set engine throttle.
        
        Args:
            throttle: Throttle setting (0.0 to 1.0)
        """
        if not self.params.throttleable and throttle > 0:
            throttle = 1.0  # Non-throttleable engines are either off or full thrust
        
        # Apply minimum throttle constraint
        if throttle > 0 and throttle < self.params.min_throttle:
            throttle = self.params.min_throttle
        
        self.throttle_setting = max(0.0, min(1.0, throttle))
    
    def ignite(self) -> bool:
        """
        Attempt to ignite the engine.
        
        Returns:
            True if ignition successful or already ignited
        """
        if not self.engine_enabled:
            return False
        
        if self.engine_ignited:
            return True
        
        if not self.params.restart_capable and self.ignition_count > 0:
            self.logger.warning("Engine is not restart capable")
            return False
        
        # Start ignition sequence
        self.ignition_timer = self.params.ignition_delay
        if self.ignition_timer <= 0:
            self.engine_ignited = True
            self.ignition_count += 1
            self.logger.info("Engine ignited")
        
        return True
    
    def shutdown(self):
        """Shutdown the engine."""
        self.engine_ignited = False
        self.throttle_setting = 0.0
        self.ignition_timer = 0.0
        self.logger.info("Engine shutdown")
    
    def update(self, dt: float) -> Tuple[float, float]:
        """
        Update propulsion system.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Tuple of (thrust_force, fuel_consumed)
        """
        # Handle ignition delay
        if self.ignition_timer > 0:
            self.ignition_timer -= dt
            if self.ignition_timer <= 0:
                self.engine_ignited = True
                self.ignition_count += 1
                self.logger.info("Engine ignition complete")
        
        # Calculate thrust and fuel consumption
        thrust_force = 0.0
        fuel_consumed = 0.0
        
        if self.engine_ignited and self.throttle_setting > 0:
            # Calculate effective thrust considering wear
            wear_efficiency = 1.0 - (self.wear_factor * 0.3)  # Max 30% performance loss
            thrust_force = self.params.max_thrust * self.throttle_setting * wear_efficiency
            
            # Calculate fuel consumption
            fuel_flow = self.params.fuel_flow_rate * self.throttle_setting
            fuel_consumed = fuel_flow * dt
            
            # Update tracking
            self.total_burn_time += dt
            self.total_fuel_consumed += fuel_consumed
            
            # Update engine temperature (simplified)
            target_temp = 273.15 + (self.throttle_setting * 2000)  # Up to 2000K
            temp_change_rate = 500  # K/s
            if self.engine_temperature < target_temp:
                self.engine_temperature = min(target_temp, 
                                            self.engine_temperature + temp_change_rate * dt)
            
            # Update wear factor (very slowly)
            self.wear_factor += dt * 1e-6  # Minimal wear per second
            self.wear_factor = min(1.0, self.wear_factor)
        else:
            # Engine cooling down
            cooling_rate = 200  # K/s
            self.engine_temperature = max(273.15, 
                                        self.engine_temperature - cooling_rate * dt)
        
        return thrust_force, fuel_consumed
    
    def get_status(self) -> Dict[str, any]:
        """Get propulsion system status."""
        return {
            "engine_name": self.params.name,
            "engine_type": self.params.engine_type.value,
            "throttle_setting": self.throttle_setting * 100,  # Percentage
            "engine_enabled": self.engine_enabled,
            "engine_ignited": self.engine_ignited,
            "ignition_timer": self.ignition_timer,
            "total_burn_time": self.total_burn_time,
            "total_fuel_consumed": self.total_fuel_consumed,
            "ignition_count": self.ignition_count,
            "engine_temperature": self.engine_temperature - 273.15,  # Celsius
            "wear_factor": self.wear_factor * 100,  # Percentage
            "thrust_output": self.params.max_thrust * self.throttle_setting
        }


class RCSSystem:
    """
    Reaction Control System for spacecraft attitude control.
    
    Manages small thrusters for precise attitude and translation control.
    """
    
    def __init__(self):
        """Initialize RCS system."""
        self.logger = get_logger("rcs_system")
        
        # RCS thrusters
        self.thrusters: List[RCSThruster] = []
        self.rcs_enabled = True
        
        # Control inputs
        self.translation_command = Vector3D()  # X, Y, Z translation
        self.rotation_command = Vector3D()     # Roll, pitch, yaw
        
        # Performance tracking
        self.total_rcs_fuel_consumed = 0.0
        self.total_rcs_burn_time = 0.0
        
        # Create standard RCS thruster layout
        self._create_standard_layout()
        
        self.logger.info("RCS system initialized with {} thrusters".format(len(self.thrusters)))
    
    def _create_standard_layout(self):
        """Create standard RCS thruster layout."""
        # Typical spacecraft has 8 RCS thrusters
        thruster_force = 100.0  # Newtons per thruster
        
        # Forward thrusters (for backward translation and attitude)
        self.thrusters.append(RCSThruster(
            location=ThrusterLocation.FORWARD_PORT,
            position=Vector3D(-5.0, 0.0, -1.5),  # Left side, forward
            direction=Vector3D(0, 0, 1),  # Thrust in +Z direction
            max_thrust=thruster_force
        ))
        
        self.thrusters.append(RCSThruster(
            location=ThrusterLocation.FORWARD_STARBOARD,
            position=Vector3D(-5.0, 0.0, 1.5),  # Right side, forward
            direction=Vector3D(0, 0, -1),  # Thrust in -Z direction
            max_thrust=thruster_force
        ))
        
        self.thrusters.append(RCSThruster(
            location=ThrusterLocation.FORWARD_TOP,
            position=Vector3D(-5.0, 1.5, 0.0),  # Top, forward
            direction=Vector3D(0, -1, 0),  # Thrust in -Y direction
            max_thrust=thruster_force
        ))
        
        self.thrusters.append(RCSThruster(
            location=ThrusterLocation.FORWARD_BOTTOM,
            position=Vector3D(-5.0, -1.5, 0.0),  # Bottom, forward
            direction=Vector3D(0, 1, 0),  # Thrust in +Y direction
            max_thrust=thruster_force
        ))
        
        # Aft thrusters (for forward translation and attitude)
        self.thrusters.append(RCSThruster(
            location=ThrusterLocation.AFT_PORT,
            position=Vector3D(5.0, 0.0, -1.5),  # Left side, aft
            direction=Vector3D(0, 0, 1),  # Thrust in +Z direction
            max_thrust=thruster_force
        ))
        
        self.thrusters.append(RCSThruster(
            location=ThrusterLocation.AFT_STARBOARD,
            position=Vector3D(5.0, 0.0, 1.5),  # Right side, aft
            direction=Vector3D(0, 0, -1),  # Thrust in -Z direction
            max_thrust=thruster_force
        ))
        
        self.thrusters.append(RCSThruster(
            location=ThrusterLocation.AFT_TOP,
            position=Vector3D(5.0, 1.5, 0.0),  # Top, aft
            direction=Vector3D(0, -1, 0),  # Thrust in -Y direction
            max_thrust=thruster_force
        ))
        
        self.thrusters.append(RCSThruster(
            location=ThrusterLocation.AFT_BOTTOM,
            position=Vector3D(5.0, -1.5, 0.0),  # Bottom, aft
            direction=Vector3D(0, 1, 0),  # Thrust in +Y direction
            max_thrust=thruster_force
        ))
    
    def set_translation_command(self, translation: Vector3D):
        """
        Set translation command.
        
        Args:
            translation: Translation vector (-1.0 to 1.0 in each axis)
        """
        self.translation_command = Vector3D(
            max(-1.0, min(1.0, translation.x)),
            max(-1.0, min(1.0, translation.y)),
            max(-1.0, min(1.0, translation.z))
        )
    
    def set_rotation_command(self, rotation: Vector3D):
        """
        Set rotation command.
        
        Args:
            rotation: Rotation vector (-1.0 to 1.0 in each axis)
        """
        self.rotation_command = Vector3D(
            max(-1.0, min(1.0, rotation.x)),
            max(-1.0, min(1.0, rotation.y)),
            max(-1.0, min(1.0, rotation.z))
        )
    
    def calculate_thruster_commands(self) -> Vector3D:
        """
        Calculate net force from RCS thrusters based on commands.
        
        Returns:
            Net force vector from all active thrusters
        """
        if not self.rcs_enabled:
            return Vector3D()
        
        # Reset all thruster commands
        for thruster in self.thrusters:
            thruster.current_thrust = 0.0
        
        # Calculate thruster firing for translation
        # This is a simplified mapping - real spacecraft have complex control algorithms
        
        # Y translation (up/down)
        if self.translation_command.y > 0:  # Up translation
            for thruster in self.thrusters:
                if thruster.direction.y > 0:  # Thrusters firing upward
                    thruster.current_thrust = self.translation_command.y
        elif self.translation_command.y < 0:  # Down translation
            for thruster in self.thrusters:
                if thruster.direction.y < 0:  # Thrusters firing downward
                    thruster.current_thrust = -self.translation_command.y
        
        # Z translation (left/right)
        if self.translation_command.z > 0:  # Right translation
            for thruster in self.thrusters:
                if thruster.direction.z > 0:  # Thrusters firing rightward
                    thruster.current_thrust = self.translation_command.z
        elif self.translation_command.z < 0:  # Left translation
            for thruster in self.thrusters:
                if thruster.direction.z < 0:  # Thrusters firing leftward
                    thruster.current_thrust = -self.translation_command.z
        
        # Calculate net force
        net_force = Vector3D()
        for thruster in self.thrusters:
            if thruster.current_thrust > 0:
                thrust_vector = thruster.direction * (thruster.max_thrust * thruster.current_thrust)
                net_force += thrust_vector
        
        return net_force
    
    def update(self, dt: float) -> Tuple[Vector3D, float]:
        """
        Update RCS system.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Tuple of (net_force, fuel_consumed)
        """
        net_force = self.calculate_thruster_commands()
        
        # Calculate fuel consumption
        fuel_consumed = 0.0
        active_thrusters = 0
        
        for thruster in self.thrusters:
            if thruster.current_thrust > 0:
                # Simplified fuel consumption
                fuel_flow_rate = 0.1  # kg/s per thruster at full thrust
                thruster_fuel = fuel_flow_rate * thruster.current_thrust * dt
                fuel_consumed += thruster_fuel
                active_thrusters += 1
        
        # Update tracking
        self.total_rcs_fuel_consumed += fuel_consumed
        if active_thrusters > 0:
            self.total_rcs_burn_time += dt
        
        return net_force, fuel_consumed
    
    def get_status(self) -> Dict[str, any]:
        """Get RCS system status."""
        active_thrusters = sum(1 for t in self.thrusters if t.current_thrust > 0)
        
        return {
            "rcs_enabled": self.rcs_enabled,
            "total_thrusters": len(self.thrusters),
            "active_thrusters": active_thrusters,
            "translation_command": {
                "x": self.translation_command.x,
                "y": self.translation_command.y,
                "z": self.translation_command.z
            },
            "rotation_command": {
                "roll": self.rotation_command.x,
                "pitch": self.rotation_command.y,
                "yaw": self.rotation_command.z
            },
            "total_fuel_consumed": self.total_rcs_fuel_consumed,
            "total_burn_time": self.total_rcs_burn_time
        }
    
    def reset(self):
        """Reset RCS system."""
        self.translation_command = Vector3D()
        self.rotation_command = Vector3D()
        self.total_rcs_fuel_consumed = 0.0
        self.total_rcs_burn_time = 0.0
        
        for thruster in self.thrusters:
            thruster.current_thrust = 0.0
        
        self.logger.debug("RCS system reset")