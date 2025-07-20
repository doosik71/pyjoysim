"""
Vehicle simulation modules for PyJoySim.

This module provides various vehicle simulations including cars, trucks,
motorcycles, and other vehicle types with realistic physics.
"""

from .car_simulation import (
    CarSimulation, Car, CarType, CarConfiguration, CarState, CarPhysics,
    CAR_SIMULATION_METADATA
)
from .track_system import (
    TrackLoader, TrackInstance, TrackData, TrackWall, Checkpoint
)

__all__ = [
    # Car simulation
    "CarSimulation",
    "Car", 
    "CarType",
    "CarConfiguration",
    "CarState",
    "CarPhysics",
    "CAR_SIMULATION_METADATA",
    
    # Track system
    "TrackLoader",
    "TrackInstance", 
    "TrackData",
    "TrackWall",
    "Checkpoint",
]