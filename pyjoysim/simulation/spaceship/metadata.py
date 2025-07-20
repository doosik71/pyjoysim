"""
Spaceship simulation metadata for registration.
"""

from ..manager import SimulationMetadata, SimulationCategory

# Spaceship simulation metadata
SPACESHIP_SIMULATION_METADATA = SimulationMetadata(
    name="spaceship",
    display_name="Space Exploration",
    description="Realistic spaceship simulation with orbital mechanics, life support, and deep space exploration",
    category=SimulationCategory.AERIAL,  # Could add SPACE category later
    author="PyJoySim Team",
    version="1.0.0",
    difficulty="advanced",
    tags=["spaceship", "orbital", "space", "3d", "physics", "exploration", "education"],
    requirements=["pybullet", "moderngl", "numpy"]
)