"""
Sonar system for submarine navigation and detection.

This module implements basic sonar systems:
- Active sonar for navigation and obstacle detection
- Passive sonar for stealth operations
- Sonar signature management
- Underwater acoustics simulation
"""

import math
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...physics.physics3d import Vector3D
from ...core.logging import get_logger


class SonarMode(Enum):
    """Sonar operating modes."""
    ACTIVE = "active"
    PASSIVE = "passive"
    SILENT = "silent"


class ContactType(Enum):
    """Types of sonar contacts."""
    SUBMARINE = "submarine"
    SURFACE_VESSEL = "surface_vessel"
    MARINE_LIFE = "marine_life"
    OBSTACLE = "obstacle"
    SEAFLOOR = "seafloor"
    UNKNOWN = "unknown"


@dataclass
class SonarContact:
    """Represents a sonar contact."""
    contact_id: str
    contact_type: ContactType
    position: Vector3D
    range_meters: float
    bearing_degrees: float
    elevation_degrees: float
    signal_strength: float  # 0.0 to 1.0
    confidence: float      # 0.0 to 1.0
    last_update_time: float


class SonarSystem:
    """
    Basic sonar system for submarine navigation and detection.
    
    Provides active and passive sonar capabilities with simplified
    underwater acoustics.
    """
    
    def __init__(self):
        """Initialize sonar system."""
        self.logger = get_logger("sonar_system")
        
        # Sonar configuration
        self.mode = SonarMode.PASSIVE
        self.active_sonar_enabled = True
        self.passive_sonar_enabled = True
        
        # Active sonar parameters
        self.active_range = 5000.0      # meters
        self.active_power = 0.5         # 0.0 to 1.0
        self.ping_interval = 2.0        # seconds
        self.last_ping_time = 0.0
        
        # Passive sonar parameters
        self.passive_range = 10000.0    # meters
        self.noise_threshold = 0.1      # Minimum signal to detect
        
        # Detection arrays
        self.hydrophones = 12           # Number of hydrophone elements
        self.sonar_frequency = 3500.0   # Hz
        
        # Current contacts
        self.contacts: List[SonarContact] = []
        self.contact_id_counter = 0
        self.max_contacts = 20
        
        # Environmental factors
        self.background_noise = 0.2     # Ocean background noise
        self.thermocline_effects = True
        
        # Performance tracking
        self.total_pings = 0
        self.contacts_detected = 0
        self.false_alarms = 0
        
        self.logger.info("Sonar system initialized")
    
    def set_mode(self, mode: SonarMode):
        """Set sonar operating mode."""
        old_mode = self.mode
        self.mode = mode
        
        if mode == SonarMode.SILENT:
            self.active_sonar_enabled = False
        elif mode == SonarMode.ACTIVE:
            self.active_sonar_enabled = True
            self.passive_sonar_enabled = True
        elif mode == SonarMode.PASSIVE:
            self.active_sonar_enabled = False
            self.passive_sonar_enabled = True
        
        self.logger.info(f"Sonar mode changed from {old_mode.value} to {mode.value}")
    
    def set_active_power(self, power: float):
        """Set active sonar transmit power (0.0 to 1.0)."""
        self.active_power = max(0.0, min(1.0, power))
        self.active_range = 2000.0 + (self.active_power * 8000.0)  # 2-10km range
    
    def ping(self, submarine_position: Vector3D, submarine_depth: float) -> bool:
        """
        Transmit active sonar ping.
        
        Args:
            submarine_position: Current submarine position
            submarine_depth: Current depth
            
        Returns:
            True if ping was transmitted
        """
        if not self.active_sonar_enabled or self.mode == SonarMode.SILENT:
            return False
        
        current_time = time.time()
        if current_time - self.last_ping_time < self.ping_interval:
            return False
        
        self.last_ping_time = current_time
        self.total_pings += 1
        
        # Simulate active sonar detection
        self._simulate_active_detection(submarine_position, submarine_depth)
        
        self.logger.debug(f"Active sonar ping transmitted (power: {self.active_power:.2f})")
        return True
    
    def passive_listen(self, submarine_position: Vector3D, submarine_depth: float, dt: float):
        """
        Passive sonar listening.
        
        Args:
            submarine_position: Current submarine position
            submarine_depth: Current depth
            dt: Time step for continuous listening
        """
        if not self.passive_sonar_enabled:
            return
        
        # Simulate passive sonar detection
        self._simulate_passive_detection(submarine_position, submarine_depth, dt)
    
    def _simulate_active_detection(self, submarine_position: Vector3D, submarine_depth: float):
        """Simulate active sonar returns (simplified)."""
        # Simulate some contacts for demonstration
        # In a real simulation, this would query the environment for objects
        
        # Seafloor detection
        seafloor_depth = 4000.0  # Assume 4km deep ocean
        if submarine_depth < seafloor_depth:
            seafloor_range = seafloor_depth - submarine_depth
            if seafloor_range <= self.active_range:
                contact = SonarContact(
                    contact_id=f"SF_{self.contact_id_counter}",
                    contact_type=ContactType.SEAFLOOR,
                    position=Vector3D(submarine_position.x, -seafloor_depth, submarine_position.z),
                    range_meters=seafloor_range,
                    bearing_degrees=0.0,
                    elevation_degrees=-90.0,
                    signal_strength=0.8,
                    confidence=0.9,
                    last_update_time=time.time()
                )
                self._add_or_update_contact(contact)
        
        # Simulate some marine life contacts
        if np.random.random() < 0.3:  # 30% chance of marine life detection
            range_m = np.random.uniform(100, self.active_range)
            bearing = np.random.uniform(0, 360)
            
            contact = SonarContact(
                contact_id=f"ML_{self.contact_id_counter}",
                contact_type=ContactType.MARINE_LIFE,
                position=self._calculate_contact_position(submarine_position, range_m, bearing, 0),
                range_meters=range_m,
                bearing_degrees=bearing,
                elevation_degrees=0.0,
                signal_strength=np.random.uniform(0.2, 0.5),
                confidence=0.6,
                last_update_time=time.time()
            )
            self._add_or_update_contact(contact)
    
    def _simulate_passive_detection(self, submarine_position: Vector3D, submarine_depth: float, dt: float):
        """Simulate passive sonar detection (simplified)."""
        # Simulate distant surface vessels
        if np.random.random() < 0.1 * dt:  # Low probability per second
            range_m = np.random.uniform(2000, self.passive_range)
            bearing = np.random.uniform(0, 360)
            
            contact = SonarContact(
                contact_id=f"SV_{self.contact_id_counter}",
                contact_type=ContactType.SURFACE_VESSEL,
                position=self._calculate_contact_position(submarine_position, range_m, bearing, submarine_depth),
                range_meters=range_m,
                bearing_degrees=bearing,
                elevation_degrees=0.0,
                signal_strength=np.random.uniform(0.1, 0.4),
                confidence=0.5,
                last_update_time=time.time()
            )
            self._add_or_update_contact(contact)
    
    def _calculate_contact_position(self, submarine_pos: Vector3D, range_m: float, 
                                  bearing_deg: float, elevation_deg: float) -> Vector3D:
        """Calculate contact position from range and bearing."""
        bearing_rad = math.radians(bearing_deg)
        elevation_rad = math.radians(elevation_deg)
        
        x_offset = range_m * math.cos(elevation_rad) * math.cos(bearing_rad)
        y_offset = range_m * math.sin(elevation_rad)
        z_offset = range_m * math.cos(elevation_rad) * math.sin(bearing_rad)
        
        return submarine_pos + Vector3D(x_offset, y_offset, z_offset)
    
    def _add_or_update_contact(self, new_contact: SonarContact):
        """Add new contact or update existing one."""
        # Check if this is an update to existing contact
        for i, existing in enumerate(self.contacts):
            if (existing.contact_type == new_contact.contact_type and
                (new_contact.position - existing.position).magnitude() < 500.0):  # Within 500m
                # Update existing contact
                self.contacts[i] = new_contact
                return
        
        # Add new contact
        if len(self.contacts) >= self.max_contacts:
            # Remove oldest contact
            self.contacts.pop(0)
        
        new_contact.contact_id = f"{new_contact.contact_type.value}_{self.contact_id_counter}"
        self.contact_id_counter += 1
        self.contacts.append(new_contact)
        self.contacts_detected += 1
    
    def get_contacts_by_type(self, contact_type: ContactType) -> List[SonarContact]:
        """Get all contacts of specified type."""
        return [contact for contact in self.contacts if contact.contact_type == contact_type]
    
    def get_nearest_contact(self, contact_type: Optional[ContactType] = None) -> Optional[SonarContact]:
        """Get nearest contact, optionally filtered by type."""
        candidates = self.contacts
        if contact_type:
            candidates = self.get_contacts_by_type(contact_type)
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda c: c.range_meters)
    
    def clear_old_contacts(self, max_age_seconds: float = 60.0):
        """Remove contacts older than specified age."""
        current_time = time.time()
        self.contacts = [
            contact for contact in self.contacts 
            if current_time - contact.last_update_time < max_age_seconds
        ]
    
    def update(self, dt: float, submarine_position: Vector3D, submarine_depth: float):
        """
        Update sonar system.
        
        Args:
            dt: Time step in seconds
            submarine_position: Current submarine position
            submarine_depth: Current depth
        """
        # Passive listening (always active unless in silent mode)
        if self.mode != SonarMode.SILENT:
            self.passive_listen(submarine_position, submarine_depth, dt)
        
        # Active sonar pinging (if enabled and not in silent mode)
        if self.mode == SonarMode.ACTIVE:
            self.ping(submarine_position, submarine_depth)
        
        # Clean up old contacts
        self.clear_old_contacts()
        
        # Update environmental effects
        self._update_environmental_effects(submarine_depth)
    
    def _update_environmental_effects(self, depth: float):
        """Update environmental effects on sonar performance."""
        # Depth affects sonar performance
        if depth > 200.0:  # Below thermocline
            self.background_noise = 0.15  # Less surface noise
            self.passive_range *= 1.2     # Better propagation
        else:
            self.background_noise = 0.25  # More surface noise
            self.passive_range *= 0.9     # Worse propagation
    
    def get_tactical_display(self) -> Dict[str, any]:
        """Get tactical sonar display information."""
        # Organize contacts by bearing sectors
        bearing_sectors = {}
        for i in range(0, 360, 30):  # 30-degree sectors
            sector_contacts = [
                contact for contact in self.contacts
                if i <= contact.bearing_degrees < i + 30
            ]
            if sector_contacts:
                bearing_sectors[f"{i:03d}-{i+30:03d}"] = len(sector_contacts)
        
        return {
            "mode": self.mode.value,
            "active_range": self.active_range,
            "passive_range": self.passive_range,
            "total_contacts": len(self.contacts),
            "bearing_sectors": bearing_sectors,
            "nearest_contact": {
                "type": self.get_nearest_contact().contact_type.value if self.get_nearest_contact() else None,
                "range": self.get_nearest_contact().range_meters if self.get_nearest_contact() else None,
                "bearing": self.get_nearest_contact().bearing_degrees if self.get_nearest_contact() else None
            },
            "background_noise": self.background_noise,
            "last_ping": time.time() - self.last_ping_time if self.last_ping_time > 0 else None
        }
    
    def get_status(self) -> Dict[str, any]:
        """Get sonar system status."""
        contacts_by_type = {}
        for contact_type in ContactType:
            count = len(self.get_contacts_by_type(contact_type))
            if count > 0:
                contacts_by_type[contact_type.value] = count
        
        return {
            "mode": self.mode.value,
            "active_sonar_enabled": self.active_sonar_enabled,
            "passive_sonar_enabled": self.passive_sonar_enabled,
            "active_power": self.active_power * 100,  # Percentage
            "active_range": self.active_range,
            "passive_range": self.passive_range,
            "ping_interval": self.ping_interval,
            "total_contacts": len(self.contacts),
            "contacts_by_type": contacts_by_type,
            "total_pings": self.total_pings,
            "contacts_detected": self.contacts_detected,
            "background_noise": self.background_noise,
            "hydrophones": self.hydrophones,
            "sonar_frequency": self.sonar_frequency
        }
    
    def reset(self):
        """Reset sonar system to initial state."""
        self.mode = SonarMode.PASSIVE
        self.active_sonar_enabled = True
        self.passive_sonar_enabled = True
        self.active_power = 0.5
        self.contacts.clear()
        self.contact_id_counter = 0
        self.total_pings = 0
        self.contacts_detected = 0
        self.false_alarms = 0
        self.last_ping_time = 0.0
        self.background_noise = 0.2
        
        self.logger.debug("Sonar system reset")