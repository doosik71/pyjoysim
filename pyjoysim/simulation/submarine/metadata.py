"""
Submarine simulation metadata for registration.
"""

from ..manager import SimulationMetadata, SimulationCategory

# Submarine simulation metadata
SUBMARINE_SIMULATION_METADATA = SimulationMetadata(
    name="submarine",
    display_name="Submarine Exploration",
    description="Realistic submarine simulation with underwater physics, ballast systems, and sonar navigation",
    category=SimulationCategory.VEHICLE,  # Could add MARINE category later
    author="PyJoySim Team",
    version="1.0.0",
    difficulty="intermediate",
    tags=["submarine", "underwater", "marine", "3d", "physics", "exploration", "education"],
    requirements=["pybullet", "moderngl", "numpy"]
)