"""
Life support system for spaceship simulation.

This module implements critical life support systems including:
- Oxygen generation and consumption
- Carbon dioxide scrubbing
- Power management
- Thermal regulation
- Radiation protection
"""

import math
import time
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from ...physics.physics3d import Vector3D
from ...core.logging import get_logger


class SystemStatus(Enum):
    """Status of life support systems."""
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class AlertLevel(Enum):
    """Alert levels for life support systems."""
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


@dataclass
class LifeSupportParameters:
    """Parameters for life support systems."""
    # Crew parameters
    crew_count: int = 3
    oxygen_consumption_rate: float = 0.84  # kg/day per person
    co2_production_rate: float = 1.04      # kg/day per person
    water_consumption_rate: float = 3.52   # kg/day per person
    
    # Storage capacities
    oxygen_capacity: float = 100.0      # kg
    water_capacity: float = 1000.0      # kg
    food_capacity: float = 500.0        # kg
    power_capacity: float = 100.0       # kWh
    
    # System parameters
    co2_scrubber_efficiency: float = 0.95
    oxygen_generator_efficiency: float = 0.85
    power_consumption_base: float = 5.0  # kW
    thermal_regulation_power: float = 2.0  # kW
    
    # Environmental parameters
    habitable_temp_min: float = 18.0    # Celsius
    habitable_temp_max: float = 24.0    # Celsius
    safe_pressure_min: float = 80.0     # kPa
    safe_pressure_max: float = 110.0    # kPa
    max_co2_ppm: float = 5000.0         # Parts per million


class LifeSupportSystem:
    """
    Comprehensive life support system for spacecraft.
    
    Manages all critical systems required for crew survival in space.
    """
    
    def __init__(self, params: LifeSupportParameters):
        """
        Initialize life support system.
        
        Args:
            params: Life support parameters
        """
        self.logger = get_logger("life_support")
        self.params = params
        
        # Current resource levels
        self.oxygen_level = params.oxygen_capacity  # kg
        self.water_level = params.water_capacity    # kg
        self.food_level = params.food_capacity      # kg
        self.power_level = params.power_capacity    # kWh
        
        # Environmental conditions
        self.cabin_temperature = 21.0  # Celsius
        self.cabin_pressure = 101.325   # kPa (1 atmosphere)
        self.co2_level = 400.0          # ppm (parts per million)
        self.humidity = 45.0            # Percentage
        
        # System status
        self.oxygen_generator_status = SystemStatus.NOMINAL
        self.co2_scrubber_status = SystemStatus.NOMINAL
        self.thermal_system_status = SystemStatus.NOMINAL
        self.power_system_status = SystemStatus.NOMINAL
        
        # Power consumption tracking
        self.total_power_consumed = 0.0
        self.power_generation_rate = 0.0  # Solar panels, fuel cells, etc.
        
        # Alerts and warnings
        self.active_alerts: List[str] = []
        self.alert_level = AlertLevel.GREEN
        
        # Mission parameters
        self.mission_duration = 0.0  # seconds
        self.crew_active = True
        
        self.logger.info("Life support system initialized for {} crew members".format(params.crew_count))
    
    def set_power_generation(self, generation_rate: float):
        """
        Set power generation rate from solar panels, fuel cells, etc.
        
        Args:
            generation_rate: Power generation in kW
        """
        self.power_generation_rate = max(0.0, generation_rate)
    
    def set_crew_activity_level(self, activity_multiplier: float):
        """
        Set crew activity level affecting consumption rates.
        
        Args:
            activity_multiplier: Activity multiplier (1.0 = normal, 1.5 = high activity)
        """
        self.activity_multiplier = max(0.5, min(2.0, activity_multiplier))
    
    def update(self, dt: float, space_environment_temperature: float = -270.0):
        """
        Update life support systems.
        
        Args:
            dt: Time step in seconds
            space_environment_temperature: External temperature in Celsius
        """
        dt_hours = dt / 3600.0  # Convert to hours
        dt_days = dt / 86400.0  # Convert to days
        
        self.mission_duration += dt
        
        # Update resource consumption
        self._update_resource_consumption(dt_days)
        
        # Update environmental systems
        self._update_environmental_systems(dt, space_environment_temperature)
        
        # Update power systems
        self._update_power_systems(dt_hours)
        
        # Check system health
        self._check_system_health()
        
        # Update alerts
        self._update_alerts()
    
    def _update_resource_consumption(self, dt_days: float):
        """Update consumption of oxygen, water, food, and CO2 production."""
        if not self.crew_active:
            return
        
        # Oxygen consumption
        oxygen_consumed = (self.params.oxygen_consumption_rate * 
                          self.params.crew_count * dt_days)
        self.oxygen_level = max(0.0, self.oxygen_level - oxygen_consumed)
        
        # Water consumption
        water_consumed = (self.params.water_consumption_rate * 
                         self.params.crew_count * dt_days)
        self.water_level = max(0.0, self.water_level - water_consumed)
        
        # Food consumption (simplified)
        food_consumed = (2.0 * self.params.crew_count * dt_days)  # 2 kg/day per person
        self.food_level = max(0.0, self.food_level - food_consumed)
        
        # CO2 production
        if self.co2_scrubber_status != SystemStatus.OFFLINE:
            co2_produced = (self.params.co2_production_rate * 
                           self.params.crew_count * dt_days)
            # CO2 scrubber removes CO2
            co2_removed = co2_produced * self.params.co2_scrubber_efficiency
            net_co2 = co2_produced - co2_removed
            
            # Update CO2 level (simplified atmospheric model)
            self.co2_level += net_co2 * 1000  # Convert to ppm
            self.co2_level = max(400.0, self.co2_level)  # Minimum background level
        else:
            # No CO2 scrubbing - CO2 accumulates
            co2_produced = (self.params.co2_production_rate * 
                           self.params.crew_count * dt_days)
            self.co2_level += co2_produced * 1000
    
    def _update_environmental_systems(self, dt: float, external_temp: float):
        """Update cabin environment (temperature, pressure, humidity)."""
        # Thermal regulation
        if self.thermal_system_status != SystemStatus.OFFLINE:
            # Calculate heat loss to space
            temp_difference = self.cabin_temperature - external_temp
            heat_loss_rate = 0.1 * temp_difference  # Simplified heat transfer
            
            # Thermal system tries to maintain temperature
            target_temp = 21.0  # Target temperature
            temp_error = target_temp - self.cabin_temperature
            
            # Thermal regulation (simplified PID-like control)
            thermal_power = abs(temp_error) * 0.5  # Power needed for regulation
            if self.power_level > thermal_power * dt / 3600.0:
                # Can maintain temperature
                self.cabin_temperature += temp_error * 0.1 * dt
            else:
                # Insufficient power - temperature drifts
                self.cabin_temperature -= heat_loss_rate * dt * 0.01
        else:
            # No thermal regulation - temperature drifts toward external
            temp_difference = self.cabin_temperature - external_temp
            cooling_rate = temp_difference * 0.001  # Very slow without regulation
            self.cabin_temperature -= cooling_rate * dt
        
        # Pressure regulation (simplified)
        if self.oxygen_level > 0 and self.thermal_system_status != SystemStatus.OFFLINE:
            target_pressure = 101.325  # kPa
            pressure_error = target_pressure - self.cabin_pressure
            self.cabin_pressure += pressure_error * 0.05 * dt
        else:
            # Pressure loss in emergency
            self.cabin_pressure = max(0.0, self.cabin_pressure - 0.1 * dt)
        
        # Humidity control (simplified)
        if self.water_level > 0:
            target_humidity = 45.0
            humidity_error = target_humidity - self.humidity
            self.humidity += humidity_error * 0.02 * dt
    
    def _update_power_systems(self, dt_hours: float):
        """Update power generation and consumption."""
        # Calculate power consumption
        base_consumption = self.params.power_consumption_base
        thermal_consumption = 0.0
        
        if self.thermal_system_status != SystemStatus.OFFLINE:
            thermal_consumption = self.params.thermal_regulation_power
        
        life_support_consumption = 1.0  # Additional systems
        total_consumption = base_consumption + thermal_consumption + life_support_consumption
        
        # Net power change
        net_power = (self.power_generation_rate - total_consumption) * dt_hours
        self.power_level += net_power
        self.power_level = max(0.0, min(self.params.power_capacity, self.power_level))
        
        self.total_power_consumed += total_consumption * dt_hours
    
    def _check_system_health(self):
        """Check health of all life support systems."""
        # Oxygen generator status
        if self.oxygen_level < self.params.oxygen_capacity * 0.1:
            self.oxygen_generator_status = SystemStatus.CRITICAL
        elif self.oxygen_level < self.params.oxygen_capacity * 0.3:
            self.oxygen_generator_status = SystemStatus.DEGRADED
        else:
            self.oxygen_generator_status = SystemStatus.NOMINAL
        
        # CO2 scrubber status
        if self.co2_level > self.params.max_co2_ppm:
            self.co2_scrubber_status = SystemStatus.CRITICAL
        elif self.co2_level > self.params.max_co2_ppm * 0.8:
            self.co2_scrubber_status = SystemStatus.DEGRADED
        else:
            self.co2_scrubber_status = SystemStatus.NOMINAL
        
        # Thermal system status
        temp_in_range = (self.params.habitable_temp_min <= self.cabin_temperature <= 
                        self.params.habitable_temp_max)
        if not temp_in_range:
            if abs(self.cabin_temperature - 21.0) > 10:
                self.thermal_system_status = SystemStatus.CRITICAL
            else:
                self.thermal_system_status = SystemStatus.DEGRADED
        else:
            self.thermal_system_status = SystemStatus.NOMINAL
        
        # Power system status
        if self.power_level < self.params.power_capacity * 0.1:
            self.power_system_status = SystemStatus.CRITICAL
        elif self.power_level < self.params.power_capacity * 0.3:
            self.power_system_status = SystemStatus.DEGRADED
        else:
            self.power_system_status = SystemStatus.NOMINAL
    
    def _update_alerts(self):
        """Update alerts and warning levels."""
        self.active_alerts.clear()
        
        # Check for critical conditions
        critical_conditions = []
        warning_conditions = []
        
        # Oxygen alerts
        if self.oxygen_level < self.params.oxygen_capacity * 0.1:
            critical_conditions.append("CRITICAL: Oxygen level critically low")
        elif self.oxygen_level < self.params.oxygen_capacity * 0.3:
            warning_conditions.append("WARNING: Low oxygen level")
        
        # CO2 alerts
        if self.co2_level > self.params.max_co2_ppm:
            critical_conditions.append("CRITICAL: CO2 level dangerous")
        elif self.co2_level > self.params.max_co2_ppm * 0.8:
            warning_conditions.append("WARNING: Elevated CO2 level")
        
        # Temperature alerts
        if (self.cabin_temperature < self.params.habitable_temp_min - 5 or 
            self.cabin_temperature > self.params.habitable_temp_max + 5):
            critical_conditions.append("CRITICAL: Cabin temperature out of safe range")
        elif (self.cabin_temperature < self.params.habitable_temp_min or 
              self.cabin_temperature > self.params.habitable_temp_max):
            warning_conditions.append("WARNING: Cabin temperature suboptimal")
        
        # Power alerts
        if self.power_level < self.params.power_capacity * 0.1:
            critical_conditions.append("CRITICAL: Power level critically low")
        elif self.power_level < self.params.power_capacity * 0.3:
            warning_conditions.append("WARNING: Low power level")
        
        # Water alerts
        if self.water_level < self.params.water_capacity * 0.1:
            critical_conditions.append("CRITICAL: Water supply critically low")
        elif self.water_level < self.params.water_capacity * 0.3:
            warning_conditions.append("WARNING: Low water supply")
        
        # Food alerts
        if self.food_level < self.params.food_capacity * 0.1:
            warning_conditions.append("WARNING: Food supply low")
        
        # Set alert level and messages
        if critical_conditions:
            self.alert_level = AlertLevel.RED
            self.active_alerts.extend(critical_conditions)
        elif warning_conditions:
            if len(warning_conditions) > 2:
                self.alert_level = AlertLevel.ORANGE
            else:
                self.alert_level = AlertLevel.YELLOW
            self.active_alerts.extend(warning_conditions)
        else:
            self.alert_level = AlertLevel.GREEN
    
    def get_status(self) -> Dict[str, any]:
        """Get comprehensive life support status."""
        # Calculate remaining time for critical resources
        oxygen_days_remaining = (self.oxygen_level / 
                               (self.params.oxygen_consumption_rate * self.params.crew_count))
        water_days_remaining = (self.water_level / 
                              (self.params.water_consumption_rate * self.params.crew_count))
        food_days_remaining = (self.food_level / (2.0 * self.params.crew_count))
        
        return {
            # Resource levels
            "oxygen_level": self.oxygen_level,
            "oxygen_percentage": (self.oxygen_level / self.params.oxygen_capacity) * 100,
            "oxygen_days_remaining": oxygen_days_remaining,
            
            "water_level": self.water_level,
            "water_percentage": (self.water_level / self.params.water_capacity) * 100,
            "water_days_remaining": water_days_remaining,
            
            "food_level": self.food_level,
            "food_percentage": (self.food_level / self.params.food_capacity) * 100,
            "food_days_remaining": food_days_remaining,
            
            "power_level": self.power_level,
            "power_percentage": (self.power_level / self.params.power_capacity) * 100,
            
            # Environmental conditions
            "cabin_temperature": self.cabin_temperature,
            "cabin_pressure": self.cabin_pressure,
            "co2_level": self.co2_level,
            "humidity": self.humidity,
            
            # System status
            "oxygen_generator_status": self.oxygen_generator_status.value,
            "co2_scrubber_status": self.co2_scrubber_status.value,
            "thermal_system_status": self.thermal_system_status.value,
            "power_system_status": self.power_system_status.value,
            
            # Mission data
            "mission_duration_hours": self.mission_duration / 3600.0,
            "crew_count": self.params.crew_count,
            "crew_active": self.crew_active,
            
            # Alerts
            "alert_level": self.alert_level.value,
            "active_alerts": self.active_alerts.copy(),
            
            # Power data
            "power_generation_rate": self.power_generation_rate,
            "total_power_consumed": self.total_power_consumed
        }
    
    def emergency_oxygen_release(self) -> bool:
        """
        Emergency oxygen release from reserves.
        
        Returns:
            True if emergency oxygen was available
        """
        emergency_oxygen = 10.0  # kg
        if emergency_oxygen > 0:
            self.oxygen_level += emergency_oxygen
            self.oxygen_level = min(self.params.oxygen_capacity, self.oxygen_level)
            self.logger.warning("Emergency oxygen released")
            return True
        return False
    
    def activate_emergency_power(self) -> bool:
        """
        Activate emergency power systems.
        
        Returns:
            True if emergency power was activated
        """
        emergency_power = 20.0  # kWh
        if emergency_power > 0:
            self.power_level += emergency_power
            self.power_level = min(self.params.power_capacity, self.power_level)
            self.logger.warning("Emergency power activated")
            return True
        return False
    
    def reset(self):
        """Reset life support system to initial state."""
        self.oxygen_level = self.params.oxygen_capacity
        self.water_level = self.params.water_capacity
        self.food_level = self.params.food_capacity
        self.power_level = self.params.power_capacity
        
        self.cabin_temperature = 21.0
        self.cabin_pressure = 101.325
        self.co2_level = 400.0
        self.humidity = 45.0
        
        self.oxygen_generator_status = SystemStatus.NOMINAL
        self.co2_scrubber_status = SystemStatus.NOMINAL
        self.thermal_system_status = SystemStatus.NOMINAL
        self.power_system_status = SystemStatus.NOMINAL
        
        self.total_power_consumed = 0.0
        self.mission_duration = 0.0
        self.active_alerts.clear()
        self.alert_level = AlertLevel.GREEN
        
        self.logger.debug("Life support system reset")