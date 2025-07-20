"""
Simulation manager for PyJoySim.

This module provides high-level management for multiple simulations,
including selection, execution, and resource management.
"""

import importlib
import time
from typing import Dict, List, Optional, Type, Any, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .base import BaseSimulation, SimulationConfig, SimulationState
from ..config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import SimulationError


class SimulationCategory(Enum):
    """Categories of simulations."""
    PHYSICS = "physics"
    VEHICLE = "vehicle"
    ROBOT = "robot"
    GAME = "game"
    EDUCATION = "education"
    RESEARCH = "research"
    DEMO = "demo"


@dataclass
class SimulationMetadata:
    """Metadata for a simulation."""
    name: str
    display_name: str
    description: str
    category: SimulationCategory
    author: str = "Unknown"
    version: str = "1.0"
    difficulty: str = "Beginner"  # Beginner, Intermediate, Advanced
    tags: List[str] = None
    requirements: List[str] = None
    thumbnail_path: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.requirements is None:
            self.requirements = []


@dataclass
class SimulationEntry:
    """Entry in the simulation registry."""
    metadata: SimulationMetadata
    simulation_class: Type[BaseSimulation]
    module_path: str
    config: Optional[SimulationConfig] = None
    
    def create_instance(self, custom_config: Optional[SimulationConfig] = None) -> BaseSimulation:
        """
        Create an instance of the simulation.
        
        Args:
            custom_config: Optional custom configuration
            
        Returns:
            Simulation instance
        """
        config = custom_config or self.config or SimulationConfig()
        return self.simulation_class(self.metadata.name, config)


class SimulationRegistry:
    """Registry for available simulations."""
    
    def __init__(self):
        """Initialize the simulation registry."""
        self.logger = get_logger("simulation_registry")
        self._simulations: Dict[str, SimulationEntry] = {}
        self._categories: Dict[SimulationCategory, List[str]] = {}
        
        # Initialize categories
        for category in SimulationCategory:
            self._categories[category] = []
    
    def register_simulation(self, 
                          simulation_class: Type[BaseSimulation],
                          metadata: SimulationMetadata,
                          module_path: str,
                          config: Optional[SimulationConfig] = None) -> None:
        """
        Register a simulation.
        
        Args:
            simulation_class: Simulation class
            metadata: Simulation metadata
            module_path: Module path for the simulation
            config: Default configuration
        """
        entry = SimulationEntry(metadata, simulation_class, module_path, config)
        self._simulations[metadata.name] = entry
        
        # Add to category
        if metadata.category not in self._categories:
            self._categories[metadata.category] = []
        self._categories[metadata.category].append(metadata.name)
        
        self.logger.debug("Simulation registered", extra={
            "name": metadata.name,
            "category": metadata.category.value,
            "module": module_path
        })
    
    def unregister_simulation(self, name: str) -> bool:
        """
        Unregister a simulation.
        
        Args:
            name: Simulation name
            
        Returns:
            True if unregistered, False if not found
        """
        if name not in self._simulations:
            return False
        
        entry = self._simulations[name]
        del self._simulations[name]
        
        # Remove from category
        if name in self._categories[entry.metadata.category]:
            self._categories[entry.metadata.category].remove(name)
        
        self.logger.debug("Simulation unregistered", extra={"name": name})
        return True
    
    def get_simulation(self, name: str) -> Optional[SimulationEntry]:
        """Get simulation entry by name."""
        return self._simulations.get(name)
    
    def get_all_simulations(self) -> Dict[str, SimulationEntry]:
        """Get all registered simulations."""
        return self._simulations.copy()
    
    def get_simulations_by_category(self, category: SimulationCategory) -> List[SimulationEntry]:
        """Get all simulations in a category."""
        simulation_names = self._categories.get(category, [])
        return [self._simulations[name] for name in simulation_names if name in self._simulations]
    
    def search_simulations(self, 
                          query: str = "",
                          category: Optional[SimulationCategory] = None,
                          tags: Optional[List[str]] = None) -> List[SimulationEntry]:
        """
        Search for simulations.
        
        Args:
            query: Search query (matches name, display_name, description)
            category: Filter by category
            tags: Filter by tags (must have all tags)
            
        Returns:
            List of matching simulation entries
        """
        results = []
        query_lower = query.lower()
        
        for entry in self._simulations.values():
            metadata = entry.metadata
            
            # Category filter
            if category and metadata.category != category:
                continue
            
            # Tags filter
            if tags:
                if not all(tag in metadata.tags for tag in tags):
                    continue
            
            # Query filter
            if query:
                searchable_text = f"{metadata.name} {metadata.display_name} {metadata.description}".lower()
                if query_lower not in searchable_text:
                    continue
            
            results.append(entry)
        
        return results
    
    def get_categories(self) -> List[SimulationCategory]:
        """Get all categories with registered simulations."""
        return [cat for cat, sims in self._categories.items() if sims]


class SimulationManager:
    """
    High-level manager for simulation execution and resource management.
    
    Provides functionality for discovering, selecting, and running simulations.
    """
    
    def __init__(self):
        """Initialize the simulation manager."""
        self.logger = get_logger("simulation_manager")
        self.settings = get_settings()
        
        # Registry and current state
        self.registry = SimulationRegistry()
        self.current_simulation: Optional[BaseSimulation] = None
        self.current_entry: Optional[SimulationEntry] = None
        
        # Resource management
        self._simulation_history: List[str] = []
        self._max_history = 10
        
        # Event callbacks
        self.simulation_started_callbacks: List[Callable[[str], None]] = []
        self.simulation_stopped_callbacks: List[Callable[[str], None]] = []
        self.simulation_error_callbacks: List[Callable[[str, Exception], None]] = []
        
        self.logger.debug("SimulationManager initialized")
    
    def discover_simulations(self, search_paths: Optional[List[str]] = None) -> int:
        """
        Discover and register simulations from search paths.
        
        Args:
            search_paths: Paths to search for simulations
            
        Returns:
            Number of simulations discovered
        """
        if search_paths is None:
            search_paths = [
                "pyjoysim.simulation.demos",
                "pyjoysim.simulation.vehicle",
                "pyjoysim.simulation.robot",
                "pyjoysim.simulation.game"
            ]
        
        discovered_count = 0
        
        for search_path in search_paths:
            try:
                count = self._discover_in_module(search_path)
                discovered_count += count
                self.logger.debug("Discovered simulations in module", extra={
                    "module": search_path,
                    "count": count
                })
            except Exception as e:
                self.logger.warning("Failed to discover simulations in module", extra={
                    "module": search_path,
                    "error": str(e)
                })
        
        self.logger.info("Simulation discovery complete", extra={
            "total_discovered": discovered_count,
            "total_registered": len(self.registry.get_all_simulations())
        })
        
        return discovered_count
    
    def _discover_in_module(self, module_path: str) -> int:
        """Discover simulations in a specific module."""
        try:
            module = importlib.import_module(module_path)
            count = 0
            
            # Look for simulation classes and metadata
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                
                # Check if it's a simulation class
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseSimulation) and 
                    attr != BaseSimulation):
                    
                    # Look for associated metadata
                    metadata_name = f"{attr_name.upper()}_METADATA"
                    if hasattr(module, metadata_name):
                        metadata = getattr(module, metadata_name)
                        if isinstance(metadata, SimulationMetadata):
                            self.registry.register_simulation(
                                attr, metadata, module_path
                            )
                            count += 1
            
            return count
            
        except ImportError as e:
            self.logger.debug("Module not found", extra={
                "module": module_path,
                "error": str(e)
            })
            return 0
    
    def list_simulations(self, 
                        category: Optional[SimulationCategory] = None) -> List[SimulationEntry]:
        """
        List available simulations.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of simulation entries
        """
        if category:
            return self.registry.get_simulations_by_category(category)
        else:
            return list(self.registry.get_all_simulations().values())
    
    def get_simulation_info(self, name: str) -> Optional[SimulationMetadata]:
        """
        Get detailed information about a simulation.
        
        Args:
            name: Simulation name
            
        Returns:
            Simulation metadata or None if not found
        """
        entry = self.registry.get_simulation(name)
        return entry.metadata if entry else None
    
    def run_simulation(self, 
                      name: str, 
                      config: Optional[SimulationConfig] = None) -> bool:
        """
        Run a simulation by name.
        
        Args:
            name: Simulation name
            config: Optional custom configuration
            
        Returns:
            True if simulation started successfully, False otherwise
        """
        # Stop current simulation if running
        if self.current_simulation:
            self.stop_current_simulation()
        
        # Get simulation entry
        entry = self.registry.get_simulation(name)
        if not entry:
            self.logger.error("Simulation not found", extra={"name": name})
            return False
        
        try:
            # Create simulation instance
            self.current_simulation = entry.create_instance(config)
            self.current_entry = entry
            
            # Add to history
            self._add_to_history(name)
            
            # Notify callbacks
            for callback in self.simulation_started_callbacks:
                try:
                    callback(name)
                except Exception as e:
                    self.logger.error("Error in simulation started callback", extra={"error": str(e)})
            
            self.logger.info("Starting simulation", extra={
                "name": name,
                "category": entry.metadata.category.value
            })
            
            # Run simulation (this will block until simulation ends)
            self.current_simulation.run()
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to run simulation", extra={
                "name": name,
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Notify error callbacks
            for callback in self.simulation_error_callbacks:
                try:
                    callback(name, e)
                except Exception as cb_error:
                    self.logger.error("Error in simulation error callback", extra={"error": str(cb_error)})
            
            return False
        
        finally:
            # Clean up
            self._cleanup_current_simulation()
    
    def stop_current_simulation(self) -> bool:
        """
        Stop the currently running simulation.
        
        Returns:
            True if stopped, False if no simulation running
        """
        if not self.current_simulation:
            return False
        
        simulation_name = self.current_simulation.name
        
        try:
            self.current_simulation.stop()
            self.logger.info("Simulation stopped", extra={"name": simulation_name})
            
            # Notify callbacks
            for callback in self.simulation_stopped_callbacks:
                try:
                    callback(simulation_name)
                except Exception as e:
                    self.logger.error("Error in simulation stopped callback", extra={"error": str(e)})
            
            return True
            
        except Exception as e:
            self.logger.error("Error stopping simulation", extra={
                "name": simulation_name,
                "error": str(e)
            })
            return False
    
    def pause_current_simulation(self) -> bool:
        """Pause the current simulation."""
        if not self.current_simulation:
            return False
        
        try:
            self.current_simulation.pause()
            return True
        except Exception as e:
            self.logger.error("Error pausing simulation", extra={"error": str(e)})
            return False
    
    def resume_current_simulation(self) -> bool:
        """Resume the current simulation."""
        if not self.current_simulation:
            return False
        
        try:
            self.current_simulation.resume()
            return True
        except Exception as e:
            self.logger.error("Error resuming simulation", extra={"error": str(e)})
            return False
    
    def get_current_simulation_status(self) -> Optional[Dict[str, Any]]:
        """Get status of current simulation."""
        if not self.current_simulation:
            return None
        
        return {
            "name": self.current_simulation.name,
            "state": self.current_simulation.state.value,
            "runtime": time.time() - self.current_simulation.start_time if self.current_simulation.start_time > 0 else 0,
            "stats": self.current_simulation.stats.__dict__,
            "config": self.current_simulation.config.__dict__
        }
    
    def get_simulation_history(self) -> List[str]:
        """Get list of recently run simulations."""
        return self._simulation_history.copy()
    
    def search_simulations(self, **kwargs) -> List[SimulationEntry]:
        """Search for simulations with given criteria."""
        return self.registry.search_simulations(**kwargs)
    
    def register_simulation(self, 
                          simulation_class: Type[BaseSimulation],
                          metadata: SimulationMetadata,
                          module_path: str = "custom",
                          config: Optional[SimulationConfig] = None) -> None:
        """Register a custom simulation."""
        self.registry.register_simulation(simulation_class, metadata, module_path, config)
    
    def _add_to_history(self, name: str) -> None:
        """Add simulation to history."""
        if name in self._simulation_history:
            self._simulation_history.remove(name)
        
        self._simulation_history.insert(0, name)
        
        # Limit history size
        if len(self._simulation_history) > self._max_history:
            self._simulation_history = self._simulation_history[:self._max_history]
    
    def _cleanup_current_simulation(self) -> None:
        """Clean up current simulation state."""
        self.current_simulation = None
        self.current_entry = None
    
    # Event callback management
    
    def add_simulation_started_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for simulation started events."""
        self.simulation_started_callbacks.append(callback)
    
    def add_simulation_stopped_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for simulation stopped events."""
        self.simulation_stopped_callbacks.append(callback)
    
    def add_simulation_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Add callback for simulation error events."""
        self.simulation_error_callbacks.append(callback)


# Global simulation manager instance
_simulation_manager: Optional[SimulationManager] = None


def get_simulation_manager() -> SimulationManager:
    """Get the global simulation manager instance."""
    global _simulation_manager
    if _simulation_manager is None:
        _simulation_manager = SimulationManager()
    return _simulation_manager


def reset_simulation_manager() -> None:
    """Reset the global simulation manager instance."""
    global _simulation_manager
    if _simulation_manager and _simulation_manager.current_simulation:
        _simulation_manager.stop_current_simulation()
    _simulation_manager = None


# Decorator for easy simulation registration
def register_simulation(metadata: SimulationMetadata, 
                       config: Optional[SimulationConfig] = None):
    """
    Decorator for registering simulations.
    
    Args:
        metadata: Simulation metadata
        config: Default configuration
    """
    def decorator(simulation_class: Type[BaseSimulation]):
        manager = get_simulation_manager()
        module_path = simulation_class.__module__
        manager.register_simulation(simulation_class, metadata, module_path, config)
        return simulation_class
    
    return decorator