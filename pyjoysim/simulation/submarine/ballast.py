"""
Ballast tank system for submarine depth and trim control.

This module implements submarine ballast and trim systems:
- Main ballast tanks for diving/surfacing
- Trim tanks for fine depth and attitude control
- Emergency blow systems
- Water/air management
"""

import math
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from ...physics.physics3d import Vector3D
from ...core.logging import get_logger


class BallastTankType(Enum):
    """Types of ballast tanks."""
    MAIN_BALLAST = "main_ballast"
    TRIM_TANK = "trim_tank"
    VARIABLE_BALLAST = "variable_ballast"
    SAFETY_TANK = "safety_tank"


class ValveState(Enum):
    """Valve positions."""
    CLOSED = "closed"
    OPEN = "open"
    PARTIALLY_OPEN = "partially_open"
    EMERGENCY_OPEN = "emergency_open"


@dataclass
class BallastTank:
    """Individual ballast tank."""
    name: str
    tank_type: BallastTankType
    capacity: float  # m³
    position: Vector3D  # Position relative to submarine center
    
    # Current state
    water_level: float = 0.0  # 0.0 to 1.0
    air_pressure: float = 101325.0  # Pa
    
    # Valve states
    flood_valve_state: ValveState = ValveState.CLOSED
    blow_valve_state: ValveState = ValveState.CLOSED
    
    # Physical properties
    flood_rate: float = 0.1   # Fraction per second when flooding
    blow_rate: float = 0.15   # Fraction per second when blowing
    
    def get_water_volume(self) -> float:
        """Get current water volume in tank."""
        return self.capacity * self.water_level
    
    def get_air_volume(self) -> float:
        """Get current air volume in tank."""
        return self.capacity * (1.0 - self.water_level)


class BallastSystem:
    """
    Comprehensive ballast system for submarine depth and trim control.
    
    Manages main ballast tanks, trim tanks, and emergency systems.
    """
    
    def __init__(self):
        """Initialize ballast system."""
        self.logger = get_logger("ballast_system")
        
        # Ballast tanks
        self.tanks: List[BallastTank] = []
        
        # High pressure air system
        self.hp_air_pressure = 25000000.0  # Pa (250 bar)
        self.hp_air_capacity = 10.0  # m³ at standard pressure
        self.current_hp_air = 10.0   # m³ available
        
        # Control state
        self.master_blow_active = False
        self.emergency_blow_active = False
        self.auto_trim_enabled = True
        
        # Target ballast states
        self.target_depth = 0.0
        self.target_trim_angle = 0.0  # degrees (bow up positive)
        
        # Performance tracking
        self.total_air_used = 0.0
        self.ballast_operations = 0
        
        # Create standard ballast tank layout
        self._create_standard_layout()
        
        self.logger.info(f"Ballast system initialized with {len(self.tanks)} tanks")
    
    def _create_standard_layout(self):
        """Create standard submarine ballast tank layout."""
        # Main ballast tanks (port and starboard)
        self.tanks.extend([
            # Forward main ballast tanks
            BallastTank(
                name="Forward Port MBT",
                tank_type=BallastTankType.MAIN_BALLAST,
                capacity=50.0,  # m³
                position=Vector3D(25.0, 2.0, -3.0),  # Forward, high, port
                flood_rate=0.08,
                blow_rate=0.12
            ),
            BallastTank(
                name="Forward Starboard MBT",
                tank_type=BallastTankType.MAIN_BALLAST,
                capacity=50.0,
                position=Vector3D(25.0, 2.0, 3.0),   # Forward, high, starboard
                flood_rate=0.08,
                blow_rate=0.12
            ),
            
            # Aft main ballast tanks
            BallastTank(
                name="Aft Port MBT",
                tank_type=BallastTankType.MAIN_BALLAST,
                capacity=60.0,
                position=Vector3D(-25.0, 2.0, -3.0), # Aft, high, port
                flood_rate=0.08,
                blow_rate=0.12
            ),
            BallastTank(
                name="Aft Starboard MBT",
                tank_type=BallastTankType.MAIN_BALLAST,
                capacity=60.0,
                position=Vector3D(-25.0, 2.0, 3.0),  # Aft, high, starboard
                flood_rate=0.08,
                blow_rate=0.12
            ),
        ])
        
        # Trim tanks
        self.tanks.extend([
            # Forward trim tank
            BallastTank(
                name="Forward Trim",
                tank_type=BallastTankType.TRIM_TANK,
                capacity=15.0,
                position=Vector3D(30.0, -2.0, 0.0),  # Forward, low, centerline
                flood_rate=0.05,
                blow_rate=0.08
            ),
            
            # Aft trim tank
            BallastTank(
                name="Aft Trim",
                tank_type=BallastTankType.TRIM_TANK,
                capacity=15.0,
                position=Vector3D(-30.0, -2.0, 0.0), # Aft, low, centerline
                flood_rate=0.05,
                blow_rate=0.08
            ),
            
            # Variable ballast (amidships)
            BallastTank(
                name="Variable Ballast",
                tank_type=BallastTankType.VARIABLE_BALLAST,
                capacity=20.0,
                position=Vector3D(0.0, -3.0, 0.0),   # Center, low
                flood_rate=0.03,
                blow_rate=0.05
            ),
        ])
        
        # Safety tank
        self.tanks.append(
            BallastTank(
                name="Safety Tank",
                tank_type=BallastTankType.SAFETY_TANK,
                capacity=8.0,
                position=Vector3D(0.0, 1.0, 0.0),    # Center, high
                flood_rate=0.1,
                blow_rate=0.2
            )
        )
    
    def get_tank_by_name(self, name: str) -> Optional[BallastTank]:
        """Get tank by name."""
        for tank in self.tanks:
            if tank.name == name:
                return tank
        return None
    
    def get_tanks_by_type(self, tank_type: BallastTankType) -> List[BallastTank]:
        """Get all tanks of specified type."""
        return [tank for tank in self.tanks if tank.tank_type == tank_type]
    
    def flood_tank(self, tank_name: str, target_level: float = 1.0):
        """
        Flood a specific tank.
        
        Args:
            tank_name: Name of tank to flood
            target_level: Target water level (0.0 to 1.0)
        """
        tank = self.get_tank_by_name(tank_name)
        if tank:
            tank.flood_valve_state = ValveState.OPEN
            self.ballast_operations += 1
            self.logger.debug(f"Flooding tank: {tank_name}")
    
    def blow_tank(self, tank_name: str, target_level: float = 0.0):
        """
        Blow a specific tank.
        
        Args:
            tank_name: Name of tank to blow
            target_level: Target water level (0.0 to 1.0)
        """
        tank = self.get_tank_by_name(tank_name)
        if tank and self.current_hp_air > 0:
            tank.blow_valve_state = ValveState.OPEN
            self.ballast_operations += 1
            self.logger.debug(f"Blowing tank: {tank_name}")
    
    def flood_main_ballast_tanks(self):
        """Flood all main ballast tanks for diving."""
        main_tanks = self.get_tanks_by_type(BallastTankType.MAIN_BALLAST)
        for tank in main_tanks:
            tank.flood_valve_state = ValveState.OPEN
            tank.blow_valve_state = ValveState.CLOSED
        
        self.ballast_operations += 1
        self.logger.info("Flooding main ballast tanks - diving")
    
    def blow_main_ballast_tanks(self):
        """Blow all main ballast tanks for surfacing."""
        if self.current_hp_air <= 0:
            self.logger.warning("Insufficient high pressure air for blowing tanks")
            return
        
        main_tanks = self.get_tanks_by_type(BallastTankType.MAIN_BALLAST)
        for tank in main_tanks:
            tank.blow_valve_state = ValveState.OPEN
            tank.flood_valve_state = ValveState.CLOSED
        
        self.ballast_operations += 1
        self.logger.info("Blowing main ballast tanks - surfacing")
    
    def emergency_blow_all(self):
        """Emergency blow all ballast tanks."""
        self.emergency_blow_active = True
        self.master_blow_active = True
        
        for tank in self.tanks:
            tank.blow_valve_state = ValveState.EMERGENCY_OPEN
            tank.flood_valve_state = ValveState.CLOSED
        
        self.logger.warning("EMERGENCY BLOW ALL TANKS ACTIVATED")
    
    def set_trim(self, trim_angle_degrees: float):
        """
        Set submarine trim angle.
        
        Args:
            trim_angle_degrees: Desired trim angle (bow up positive)
        """
        self.target_trim_angle = max(-10.0, min(10.0, trim_angle_degrees))
        
        forward_trim = self.get_tank_by_name("Forward Trim")
        aft_trim = self.get_tank_by_name("Aft Trim")
        
        if forward_trim and aft_trim:
            if trim_angle_degrees > 1.0:  # Bow up
                # Flood forward trim, blow aft trim
                forward_trim.flood_valve_state = ValveState.OPEN
                aft_trim.blow_valve_state = ValveState.OPEN
            elif trim_angle_degrees < -1.0:  # Bow down
                # Blow forward trim, flood aft trim
                forward_trim.blow_valve_state = ValveState.OPEN
                aft_trim.flood_valve_state = ValveState.OPEN
            else:  # Neutral trim
                # Balance both tanks
                pass
    
    def auto_trim_control(self, current_trim_angle: float, dt: float):
        """
        Automatic trim control to maintain level attitude.
        
        Args:
            current_trim_angle: Current submarine trim angle
            dt: Time step in seconds
        """
        if not self.auto_trim_enabled:
            return
        
        trim_error = self.target_trim_angle - current_trim_angle
        
        # Only make adjustments for significant trim errors
        if abs(trim_error) > 0.5:  # 0.5 degree deadband
            self.set_trim(self.target_trim_angle)
    
    def update(self, dt: float, current_depth: float, water_pressure: float):
        """
        Update ballast system state.
        
        Args:
            dt: Time step in seconds
            current_depth: Current submarine depth
            water_pressure: Current water pressure
        """
        for tank in self.tanks:
            self._update_tank(tank, dt, water_pressure)
        
        # Update air consumption
        self._update_air_consumption(dt)
        
        # Check for automatic operations
        if self.emergency_blow_active:
            # Emergency blow continues until tanks are empty
            all_empty = all(tank.water_level < 0.1 for tank in self.tanks)
            if all_empty:
                self.emergency_blow_active = False
                self.master_blow_active = False
                self.logger.info("Emergency blow completed")
    
    def _update_tank(self, tank: BallastTank, dt: float, water_pressure: float):
        """Update individual tank state."""
        # Update tank air pressure based on depth (simplified)
        if tank.water_level > 0:
            tank.air_pressure = max(101325.0, water_pressure)
        
        # Handle flooding
        if tank.flood_valve_state in [ValveState.OPEN, ValveState.PARTIALLY_OPEN]:
            flood_rate = tank.flood_rate
            if tank.flood_valve_state == ValveState.PARTIALLY_OPEN:
                flood_rate *= 0.5
            
            tank.water_level = min(1.0, tank.water_level + flood_rate * dt)
            
            # Auto-close valve when tank is full
            if tank.water_level >= 0.99:
                tank.flood_valve_state = ValveState.CLOSED
        
        # Handle blowing
        if tank.blow_valve_state in [ValveState.OPEN, ValveState.EMERGENCY_OPEN, ValveState.PARTIALLY_OPEN]:
            blow_rate = tank.blow_rate
            
            if tank.blow_valve_state == ValveState.EMERGENCY_OPEN:
                blow_rate *= 2.0  # Faster emergency blow
            elif tank.blow_valve_state == ValveState.PARTIALLY_OPEN:
                blow_rate *= 0.5
            
            # Can only blow if we have high pressure air
            if self.current_hp_air > 0:
                tank.water_level = max(0.0, tank.water_level - blow_rate * dt)
                
                # Auto-close valve when tank is empty
                if tank.water_level <= 0.01:
                    tank.blow_valve_state = ValveState.CLOSED
    
    def _update_air_consumption(self, dt: float):
        """Update high pressure air consumption."""
        air_consumption_rate = 0.0
        
        for tank in self.tanks:
            if tank.blow_valve_state in [ValveState.OPEN, ValveState.EMERGENCY_OPEN]:
                # Air consumption proportional to tank capacity and blow rate
                consumption = tank.capacity * 0.01 * dt  # Simplified consumption model
                if tank.blow_valve_state == ValveState.EMERGENCY_OPEN:
                    consumption *= 2.0
                air_consumption_rate += consumption
        
        # Update available air
        self.current_hp_air -= air_consumption_rate
        self.current_hp_air = max(0.0, self.current_hp_air)
        self.total_air_used += air_consumption_rate
        
        # Warning for low air
        if self.current_hp_air < 2.0 and air_consumption_rate > 0:
            self.logger.warning("Low high pressure air remaining")
    
    def calculate_ballast_weight(self) -> float:
        """
        Calculate total weight of water in ballast tanks.
        
        Returns:
            Total ballast weight in kg
        """
        total_weight = 0.0
        water_density = 1025.0  # kg/m³ (seawater)
        
        for tank in self.tanks:
            water_volume = tank.get_water_volume()
            weight = water_volume * water_density
            total_weight += weight
        
        return total_weight
    
    def calculate_ballast_moment(self) -> Vector3D:
        """
        Calculate ballast moment for trim and list effects.
        
        Returns:
            Moment vector about submarine center of mass
        """
        total_moment = Vector3D()
        water_density = 1025.0  # kg/m³
        
        for tank in self.tanks:
            water_volume = tank.get_water_volume()
            if water_volume > 0:
                weight = water_volume * water_density
                # Moment = weight * position
                moment = tank.position * weight
                total_moment += moment
        
        return total_moment
    
    def get_status(self) -> Dict[str, any]:
        """Get ballast system status."""
        main_tanks = self.get_tanks_by_type(BallastTankType.MAIN_BALLAST)
        trim_tanks = self.get_tanks_by_type(BallastTankType.TRIM_TANK)
        
        # Calculate average levels
        main_ballast_level = sum(tank.water_level for tank in main_tanks) / len(main_tanks) if main_tanks else 0.0
        trim_level = sum(tank.water_level for tank in trim_tanks) / len(trim_tanks) if trim_tanks else 0.0
        
        # Individual tank status
        tank_status = {}
        for tank in self.tanks:
            tank_status[tank.name] = {
                "water_level": tank.water_level * 100,  # Percentage
                "flood_valve": tank.flood_valve_state.value,
                "blow_valve": tank.blow_valve_state.value,
                "air_pressure": tank.air_pressure / 1000  # kPa
            }
        
        return {
            "main_ballast_level": main_ballast_level * 100,
            "trim_level": trim_level * 100,
            "hp_air_remaining": self.current_hp_air,
            "hp_air_percentage": (self.current_hp_air / self.hp_air_capacity) * 100,
            "emergency_blow_active": self.emergency_blow_active,
            "master_blow_active": self.master_blow_active,
            "auto_trim_enabled": self.auto_trim_enabled,
            "target_trim_angle": self.target_trim_angle,
            "total_ballast_weight": self.calculate_ballast_weight(),
            "total_air_used": self.total_air_used,
            "ballast_operations": self.ballast_operations,
            "tanks": tank_status
        }
    
    def reset(self):
        """Reset ballast system to initial state."""
        # Reset all tanks
        for tank in self.tanks:
            tank.water_level = 0.0
            tank.air_pressure = 101325.0
            tank.flood_valve_state = ValveState.CLOSED
            tank.blow_valve_state = ValveState.CLOSED
        
        # Reset air system
        self.current_hp_air = self.hp_air_capacity
        self.total_air_used = 0.0
        
        # Reset control state
        self.master_blow_active = False
        self.emergency_blow_active = False
        self.auto_trim_enabled = True
        self.target_depth = 0.0
        self.target_trim_angle = 0.0
        self.ballast_operations = 0
        
        self.logger.debug("Ballast system reset")