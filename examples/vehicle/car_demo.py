#!/usr/bin/env python3
"""
Car Simulation Demo

This script demonstrates the 2D car simulation with:
- Realistic car physics with tire models
- Multiple tracks with checkpoints and lap timing
- Joystick and keyboard controls
- Dashboard with speedometer and controls
- Track switching and car reset functionality

Usage:
    python examples/vehicle/car_demo.py [--track TRACK_NAME] [--car CAR_TYPE]
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyjoysim.simulation import SimulationConfig
from pyjoysim.simulation.vehicle import CarSimulation, CarType


def main():
    """Main entry point for car simulation demo."""
    parser = argparse.ArgumentParser(description="PyJoySim Car Simulation Demo")
    parser.add_argument("--track", type=str, default="oval",
                       help="Track to load (oval, figure8, city_circuit)")
    parser.add_argument("--car", type=str, default="sports_car",
                       choices=["sports_car", "suv", "truck"],
                       help="Car type to use")
    parser.add_argument("--window-width", type=int, default=1024,
                       help="Window width")
    parser.add_argument("--window-height", type=int, default=768,
                       help="Window height")
    
    args = parser.parse_args()
    
    # Create simulation config
    config = SimulationConfig(
        window_title="PyJoySim - Car Simulation Demo",
        window_width=args.window_width,
        window_height=args.window_height,
        enable_debug_draw=True,
        enable_performance_monitoring=True
    )
    
    # Create car simulation
    car_sim = CarSimulation("car_demo", config)
    car_sim.selected_track_name = args.track
    
    try:
        print("\nPyJoySim Car Simulation Demo")
        print("=" * 50)
        print(f"Track: {args.track}")
        print(f"Car Type: {args.car}")
        print()
        print("Controls:")
        print("  Keyboard:")
        print("    Arrow Keys  - Steer/Throttle/Brake")
        print("    Spacebar    - Handbrake")
        print("    C           - Toggle camera follow")
        print("    WASD        - Move camera (when not following)")
        print("    Q/E         - Zoom in/out")
        print("    R           - Reset camera")
        print("    ESC         - Exit")
        print()
        print("  Joystick:")
        print("    Left Stick  - Steering")
        print("    Right Trigger - Throttle")
        print("    Left Trigger  - Brake")
        print("    A Button    - Handbrake")
        print("    B Button    - Toggle camera follow")
        print("    X Button    - Next track")
        print("    Y Button    - Reset car position")
        print()
        print("  Dashboard:")
        print("    Speed, RPM, Gear, Throttle/Brake bars")
        print("    Lap timing and checkpoint info")
        print("=" * 50)
        print()
        print("Starting simulation...")
        
        # Run the simulation
        car_sim.run()
        
        print("Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())