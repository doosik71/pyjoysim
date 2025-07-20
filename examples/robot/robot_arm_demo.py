#!/usr/bin/env python3
"""
Robot Arm Simulation Demo

This script demonstrates the 3-DOF robot arm simulation with:
- Forward and inverse kinematics
- Interactive target positioning
- Real-time control with joystick
- Educational visualization features

Usage:
    python examples/robot/robot_arm_demo.py [--mode MODE] [--show-workspace]
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyjoysim.simulation import SimulationConfig
from pyjoysim.simulation.robot import RobotArmSimulation


def main():
    """Main entry point for robot arm simulation demo."""
    parser = argparse.ArgumentParser(description="PyJoySim Robot Arm Simulation Demo")
    parser.add_argument("--mode", type=str, default="end_effector",
                       choices=["end_effector", "joint_direct"],
                       help="Initial control mode")
    parser.add_argument("--show-workspace", action="store_true",
                       help="Show robot workspace boundary")
    parser.add_argument("--window-width", type=int, default=1024,
                       help="Window width")
    parser.add_argument("--window-height", type=int, default=768,
                       help="Window height")
    
    args = parser.parse_args()
    
    # Create simulation config
    config = SimulationConfig(
        window_title="PyJoySim - Robot Arm Simulation Demo",
        window_width=args.window_width,
        window_height=args.window_height,
        enable_debug_draw=True,
        enable_performance_monitoring=True
    )
    
    # Create robot arm simulation
    robot_sim = RobotArmSimulation("robot_arm_demo", config)
    robot_sim.show_workspace = args.show_workspace
    
    try:
        print("\\nPyJoySim Robot Arm Simulation Demo")
        print("=" * 50)
        print(f"Control Mode: {args.mode}")
        print(f"Show Workspace: {args.show_workspace}")
        print()
        print("Educational Features:")
        print("  • Forward/Inverse Kinematics")
        print("  • 3-DOF Robot Arm with Joint Constraints")
        print("  • Real-time Target Positioning")
        print("  • Workspace Visualization")
        print("  • Trajectory Recording")
        print()
        print("Controls:")
        print("  Keyboard:")
        print("    WASD        - Move target position")
        print("    Spacebar    - Set random target")
        print("    Tab         - Toggle control mode")
        print("    W           - Toggle workspace view")
        print("    R           - Reset robot pose")
        print("    ESC         - Exit")
        print()
        print("  Joystick:")
        print("    Left Stick  - Move target X-Y")
        print("    Right Stick - Fine target adjustment")
        print("    A Button    - Set random target")
        print("    B Button    - Toggle control mode")
        print("    X Button    - Toggle workspace view")
        print("    Y Button    - Reset robot pose")
        print()
        print("  Educational Info:")
        print("    • Joint angles displayed in real-time")
        print("    • End effector position coordinates")
        print("    • Distance to target measurement")
        print("    • Visual workspace boundary")
        print("    • Control mode indicator")
        print("=" * 50)
        print()
        print("Starting robot arm simulation...")
        
        # Run the simulation
        robot_sim.run()
        
        print("Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())