"""
Robot simulation modules for PyJoySim.

This module provides various robot simulations including:
- 3-DOF robot arm with kinematics and control
- Mobile robot simulations (planned for Phase 3)
"""

from .robot_arm_simulation import (
    RobotArmSimulation, RobotArmConfiguration, RobotState, 
    Kinematics, RobotArmPhysics, RobotArmController, ControlMode,
    Link, JointConstraints, ROBOT_ARM_SIMULATION_METADATA
)
from .mobile_robot_simulation import (
    MobileRobotSimulation, MOBILE_ROBOT_SIMULATION_METADATA
)

__all__ = [
    # Robot arm simulation
    "RobotArmSimulation",
    "RobotArmConfiguration", 
    "RobotState",
    "Kinematics",
    "RobotArmPhysics",
    "RobotArmController",
    "ControlMode",
    "Link",
    "JointConstraints",
    "ROBOT_ARM_SIMULATION_METADATA",
    
    # Mobile robot simulation
    "MobileRobotSimulation",
    "MOBILE_ROBOT_SIMULATION_METADATA",
]