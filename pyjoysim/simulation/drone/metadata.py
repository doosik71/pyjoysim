"""
Drone simulation metadata for registration.
"""

from ..manager import SimulationMetadata, SimulationCategory

# Drone simulation metadata  
DRONE_SIMULATION_METADATA = SimulationMetadata(
    name="drone",
    display_name="Quadrotor Drone",
    description="Realistic quadrotor drone simulation with multiple flight modes and sensor simulation",
    category=SimulationCategory.AERIAL,
    author="PyJoySim Team",
    version="1.0.0",
    difficulty="intermediate",
    tags=["drone", "quadrotor", "flight", "3d", "sensors", "autonomous"],
    requirements=["pybullet", "moderngl"]
)