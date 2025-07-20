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

# Import simulation modules to trigger registration
try:
    from .vehicle import CarSimulation, CAR_SIMULATION_METADATA
    from .robot import RobotArmSimulation, ROBOT_ARM_SIMULATION_METADATA
    
    # Register vehicle simulations
    register_simulation(CarSimulation, CAR_SIMULATION_METADATA)
    
    # Register robot simulations 
    register_simulation(RobotArmSimulation, ROBOT_ARM_SIMULATION_METADATA)
    
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings
    warnings.warn(f"Some simulations could not be imported: {e}")
    pass

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