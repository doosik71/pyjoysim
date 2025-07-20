"""
Simulation framework for PyJoySim.

This module provides the foundation for all simulation types with
standardized lifecycle management and domain-specific implementations.
"""

from .base import (
    BaseSimulation,
    SimulationConfig,
    SimulationState,
    SimulationStats
)
from .manager import (
    SimulationManager,
    SimulationRegistry,
    SimulationMetadata,
    SimulationEntry,
    SimulationCategory,
    get_simulation_manager,
    reset_simulation_manager,
    register_simulation
)

__all__ = [
    # Core simulation framework
    "BaseSimulation",
    "SimulationConfig",
    "SimulationState", 
    "SimulationStats",
    
    # Simulation management
    "SimulationManager",
    "SimulationRegistry",
    "SimulationMetadata",
    "SimulationEntry",
    "SimulationCategory",
    
    # Factory functions
    "get_simulation_manager",
    "reset_simulation_manager",
    
    # Decorators
    "register_simulation",
]