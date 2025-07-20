"""
Core system components
"""

# Import from simulation module for compatibility
from ..simulation.manager import SimulationManager
from ..simulation.base import BaseSimulation

__all__ = [
    "SimulationManager", 
    "BaseSimulation",
]