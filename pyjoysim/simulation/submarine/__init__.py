"""
Submarine simulation module for PyJoySim.

This module provides realistic submarine simulation with:
- Underwater physics with buoyancy and pressure
- Ballast tank systems for depth control  
- Sonar and navigation systems
- Underwater environment simulation
- Educational features for marine physics
"""

from .submarine_simulation import SubmarineSimulation
from .physics import SubmarinePhysics, UnderwaterEnvironment
from .ballast import BallastSystem
from .sonar import SonarSystem
from .metadata import SUBMARINE_SIMULATION_METADATA

__all__ = [
    'SubmarineSimulation',
    'SubmarinePhysics',
    'UnderwaterEnvironment',
    'BallastSystem',
    'SonarSystem',
    'SUBMARINE_SIMULATION_METADATA'
]