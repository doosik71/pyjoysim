#!/usr/bin/env python3
"""
Drone Simulation Integration Test

This script tests the complete drone simulation system to ensure all components
work together correctly.
"""

import sys
import time
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all drone simulation modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core modules
        from pyjoysim.physics.physics3d import Physics3D, Vector3D, PhysicsObject3D, Shape3D, Shape3DType, Body3DType
        from pyjoysim.simulation.drone import DroneSimulation, FlightMode
        from pyjoysim.simulation.drone.physics import QuadrotorPhysics, DronePhysicsParameters
        from pyjoysim.simulation.drone.sensors import DroneSensors, IMU, GPS, Barometer
        from pyjoysim.simulation.drone.flight_controller import FlightController, FlightControlParameters, ControlInput
        from pyjoysim.simulation.drone.metadata import DRONE_SIMULATION_METADATA
        from pyjoysim.input.input_processor import InputState
        
        print("[PASS] All imports successful")
        return True
        
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False

def test_physics_engine():
    """Test physics engine initialization."""
    print("Testing physics engine...")
    
    try:
        from pyjoysim.physics.physics3d import Physics3D, Vector3D
        
        physics = Physics3D()
        if physics.initialize():
            print("[PASS] Physics engine initialized")
            physics.cleanup()
            return True
        else:
            print("[FAIL] Physics engine initialization failed")
            return False
            
    except Exception as e:
        print(f"[FAIL] Physics engine test failed: {e}")
        traceback.print_exc()
        return False

def test_drone_physics():
    """Test drone physics components."""
    print("Testing drone physics...")
    
    try:
        from pyjoysim.physics.physics3d import Physics3D, Vector3D, PhysicsObject3D, Shape3D, Shape3DType, Body3DType
        from pyjoysim.simulation.drone.physics import QuadrotorPhysics, DronePhysicsParameters
        
        # Create physics parameters
        params = DronePhysicsParameters()
        print(f"  - Physics parameters created (mass: {params.mass}kg, arm_length: {params.arm_length}m)")
        
        # Create quadrotor physics
        drone_physics = QuadrotorPhysics(params)
        print("  - Quadrotor physics created")
        
        # Test rotor commands
        commands = [0.5, 0.5, 0.5, 0.5]
        drone_physics.set_rotor_commands(commands)
        print(f"  - Rotor commands set: {commands}")
        
        # Create mock physics body
        shape = Shape3D(Shape3DType.BOX, Vector3D(0.3, 0.1, 0.3))
        body = PhysicsObject3D(
            name="test_drone",
            shape=shape,
            body_type=Body3DType.DYNAMIC,
            mass=params.mass
        )
        
        # Update physics
        force, torque = drone_physics.update(0.016, body)  # 60 FPS
        print(f"  - Physics update successful (force: {force.magnitude():.2f}N)")
        
        print("[PASS] Drone physics test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Drone physics test failed: {e}")
        traceback.print_exc()
        return False

def test_sensors():
    """Test sensor simulation."""
    print("Testing sensors...")
    
    try:
        from pyjoysim.physics.physics3d import Vector3D, PhysicsObject3D, Shape3D, Shape3DType, Body3DType
        from pyjoysim.simulation.drone.sensors import DroneSensors, IMU, GPS, Barometer
        
        # Create sensors
        sensors = DroneSensors(37.7749, -122.4194)  # San Francisco
        print("  - Drone sensors created")
        
        # Create mock physics body
        shape = Shape3D(Shape3DType.BOX, Vector3D(0.3, 0.1, 0.3))
        body = PhysicsObject3D(
            name="test_drone",
            shape=shape,
            body_type=Body3DType.DYNAMIC,
            mass=1.5,
            position=Vector3D(0, 5, 0)
        )
        
        # Update sensors
        sensors.update(body, 0.016)
        print("  - Sensors updated")
        
        # Check sensor data
        sensor_data = sensors.get_sensor_data()
        print(f"  - IMU data: accel_z = {sensor_data['imu']['accel_z']:.2f}")
        print(f"  - GPS data: lat = {sensor_data['gps']['latitude']:.6f}")
        print(f"  - Barometer: altitude = {sensor_data['barometer']['altitude']:.2f}m")
        
        print("[PASS] Sensors test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Sensors test failed: {e}")
        traceback.print_exc()
        return False

def test_flight_controller():
    """Test flight controller."""
    print("Testing flight controller...")
    
    try:
        from pyjoysim.physics.physics3d import Vector3D, PhysicsObject3D, Shape3D, Shape3DType, Body3DType
        from pyjoysim.simulation.drone.flight_controller import FlightController, FlightControlParameters, ControlInput, FlightMode
        from pyjoysim.simulation.drone.sensors import DroneSensors
        
        # Create flight controller
        controller = FlightController()
        print("  - Flight controller created")
        
        # Create sensors
        sensors = DroneSensors(37.7749, -122.4194)
        
        # Create mock physics body for sensors
        shape = Shape3D(Shape3DType.BOX, Vector3D(0.3, 0.1, 0.3))
        body = PhysicsObject3D(
            name="test_drone",
            shape=shape,
            body_type=Body3DType.DYNAMIC,
            mass=1.5,
            position=Vector3D(0, 5, 0)
        )
        sensors.update(body, 0.016)
        
        # Test different flight modes
        for mode in [FlightMode.MANUAL, FlightMode.STABILIZED, FlightMode.ALTITUDE_HOLD]:
            control_input = ControlInput()
            control_input.mode_switch = mode
            control_input.throttle = 0.5
            control_input.arm_switch = True
            
            # Update controller
            motor_commands = controller.update(control_input, sensors, 0.016)
            print(f"  - Mode {mode.value}: motor commands = {[f'{cmd:.2f}' for cmd in motor_commands]}")
        
        print("[PASS] Flight controller test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Flight controller test failed: {e}")
        traceback.print_exc()
        return False

def test_drone_simulation():
    """Test complete drone simulation."""
    print("Testing complete drone simulation...")
    
    try:
        from pyjoysim.physics.physics3d import Physics3D
        from pyjoysim.simulation.drone import DroneSimulation
        from pyjoysim.input.input_processor import InputState
        
        # Create physics engine
        physics = Physics3D()
        if not physics.initialize():
            print("[FAIL] Physics engine initialization failed")
            return False
        
        # Create drone simulation
        drone_sim = DroneSimulation(physics)
        print("  - Drone simulation created")
        
        # Initialize with configuration
        config = {
            'physics': {
                'mass': 1.5,
                'arm_length': 0.25,
                'max_thrust_per_rotor': 6.0
            },
            'initial_position': [0, 5, 0]
        }
        
        if not drone_sim.initialize(config=config):
            print("[FAIL] Drone simulation initialization failed")
            return False
        
        print("  - Drone simulation initialized")
        
        # Create mock input state
        input_state = InputState()
        
        # Test simulation updates
        for i in range(10):
            drone_sim.update(0.016, input_state)
            
        print("  - Simulation updates successful")
        
        # Test simulation data
        data = drone_sim.get_simulation_data()
        print(f"  - Position: {data['position']}")
        print(f"  - Flight time: {data['flight_time']:.3f}s")
        
        # Test flight modes
        from pyjoysim.simulation.drone import FlightMode
        drone_sim.set_flight_mode(FlightMode.STABILIZED)
        drone_sim.arm_drone(True)
        print("  - Flight mode and arming test successful")
        
        # Cleanup
        drone_sim.cleanup()
        physics.cleanup()
        
        print("[PASS] Complete drone simulation test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Drone simulation test failed: {e}")
        traceback.print_exc()
        return False

def test_simulation_registration():
    """Test simulation registration system."""
    print("Testing simulation registration...")
    
    try:
        from pyjoysim.simulation import get_simulation_manager
        from pyjoysim.simulation.drone import DRONE_SIMULATION_METADATA
        
        # Get simulation manager
        manager = get_simulation_manager()
        print("  - Simulation manager obtained")
        
        # Check if drone simulation is registered
        available_sims = manager.get_available_simulations()
        drone_found = any(sim.name == "drone" for sim in available_sims)
        
        if drone_found:
            print("  - Drone simulation found in registry")
            
            # Get drone simulation info
            for sim in available_sims:
                if sim.name == "drone":
                    print(f"    - Display name: {sim.display_name}")
                    print(f"    - Category: {sim.category.value}")
                    print(f"    - Features: {len(sim.features)} features")
                    break
        else:
            print("[FAIL] Drone simulation not found in registry")
            return False
        
        print("[PASS] Simulation registration test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Simulation registration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Drone Simulation Integration Test ===")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Physics Engine", test_physics_engine),
        ("Drone Physics", test_drone_physics),
        ("Sensors", test_sensors),
        ("Flight Controller", test_flight_controller),
        ("Complete Simulation", test_drone_simulation),
        ("Registration System", test_simulation_registration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                print(f"FAILED: {test_name}")
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            traceback.print_exc()
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Drone simulation system is ready.")
        return 0
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())