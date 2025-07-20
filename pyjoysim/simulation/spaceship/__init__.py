"""
Spaceship simulation module for PyJoySim.

This module provides realistic spaceship simulation with:
- Zero gravity physics and orbital mechanics
- Propulsion systems (main engine and RCS)
- Life support and fuel management
- Space environment simulation
- Educational features for space physics
"""

from .spaceship_simulation import SpaceshipSimulation
from .physics import SpaceshipPhysics, SpaceEnvironment
from .propulsion import PropulsionSystem, RCSSystem
from .life_support import LifeSupportSystem
from .metadata import SPACESHIP_SIMULATION_METADATA

__all__ = [
    'SpaceshipSimulation',
    'SpaceshipPhysics',
    'SpaceEnvironment',
    'PropulsionSystem',
    'RCSSystem', 
    'LifeSupportSystem',
    'SPACESHIP_SIMULATION_METADATA'
]