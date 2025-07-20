#!/usr/bin/env python3
"""
Basic Drone Simulation Test

This script tests the basic drone simulation components without requiring PyBullet.
"""

import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports that don't require PyBullet."""
    print("Testing basic imports...")
    
    try:
        # Test physics3d basic structures
        from pyjoysim.physics.physics3d import Vector3D, Quaternion
        print("  - Vector3D and Quaternion imported")
        
        # Test drone modules
        from pyjoysim.simulation.drone.physics import DronePhysicsParameters
        from pyjoysim.simulation.drone.sensors import IMU, GPS, Barometer, DroneSensors
        from pyjoysim.simulation.drone.flight_controller import (
            FlightController, FlightControlParameters, ControlInput, FlightMode, PIDController
        )
        from pyjoysim.simulation.drone.metadata import DRONE_SIMULATION_METADATA
        from pyjoysim.input.input_processor import InputState
        
        print("  - All drone modules imported successfully")
        
        print("[PASS] Basic imports successful")
        return True
        
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False

def test_drone_physics_parameters():
    """Test drone physics parameters."""
    print("Testing drone physics parameters...")
    
    try:
        from pyjoysim.simulation.drone.physics import DronePhysicsParameters
        
        # Test default parameters
        params = DronePhysicsParameters()
        print(f"  - Default mass: {params.mass}kg")
        print(f"  - Default arm length: {params.arm_length}m")
        print(f"  - Default max thrust per rotor: {params.max_thrust_per_rotor}N")
        
        # Test custom parameters
        custom_params = DronePhysicsParameters(
            mass=2.0,
            arm_length=0.3,
            max_thrust_per_rotor=8.0
        )
        print(f"  - Custom mass: {custom_params.mass}kg")
        
        print("[PASS] Drone physics parameters test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Drone physics parameters test failed: {e}")
        traceback.print_exc()
        return False

def test_sensors_basic():
    """Test basic sensor functionality without physics."""
    print("Testing basic sensor functionality...")
    
    try:
        from pyjoysim.simulation.drone.sensors import IMU, GPS, Barometer, DroneSensors, SensorNoise
        
        # Test sensor noise
        noise = SensorNoise(bias=0.1, white_noise=0.02, drift=0.001)
        print(f"  - SensorNoise created with bias: {noise.bias}")
        
        # Test IMU
        imu = IMU()
        print("  - IMU created")
        
        # Test GPS
        gps = GPS(37.7749, -122.4194)  # San Francisco
        print("  - GPS created with home coordinates")
        
        # Test Barometer
        barometer = Barometer()
        print("  - Barometer created")
        
        # Test DroneSensors
        sensors = DroneSensors(37.7749, -122.4194)
        print("  - DroneSensors created")
        
        # Test sensor data structure
        sensor_data = sensors.get_sensor_data()
        print(f"  - Sensor data keys: {list(sensor_data.keys())}")
        
        print("[PASS] Basic sensor test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic sensor test failed: {e}")
        traceback.print_exc()
        return False

def test_flight_controller_basic():
    """Test basic flight controller functionality."""
    print("Testing basic flight controller...")
    
    try:
        from pyjoysim.simulation.drone.flight_controller import (
            FlightController, FlightControlParameters, ControlInput, FlightMode, PIDController
        )
        
        # Test PID controller
        pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
        print(f"  - PID controller created with Kp={pid.kp}")
        
        # Test PID update (basic test)
        output = pid.update(setpoint=1.0, measurement=0.5, dt=0.016)
        print(f"  - PID output: {output:.3f}")
        
        # Test flight control parameters
        params = FlightControlParameters()
        print("  - Flight control parameters created")
        
        # Test control input
        control_input = ControlInput(
            roll=0.1,
            pitch=0.2,
            yaw=0.0,
            throttle=0.5,
            mode_switch=FlightMode.STABILIZED
        )
        print(f"  - Control input created, mode: {control_input.mode_switch.value}")
        
        # Test flight controller
        controller = FlightController(params)
        print("  - Flight controller created")
        
        # Test mode setting
        controller.set_mode(FlightMode.ALTITUDE_HOLD)
        status = controller.get_status()
        print(f"  - Flight mode set to: {status['mode']}")
        
        print("[PASS] Basic flight controller test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic flight controller test failed: {e}")
        traceback.print_exc()
        return False

def test_input_state():
    """Test input state functionality."""
    print("Testing input state...")
    
    try:
        from pyjoysim.input.input_processor import InputState
        from pyjoysim.input.joystick_manager import JoystickInput
        
        # Test basic input state
        input_state = InputState()
        print("  - InputState created")
        
        # Test joystick availability
        available = input_state.joystick_available()
        print(f"  - Joystick available: {available}")
        
        # Test with mock joystick input
        mock_joystick = JoystickInput(
            joystick_id=0,
            axes=[0.1, 0.2, 0.0, 0.5],
            buttons=[False, True, False, False],
            hats=[(0, 0)],
            timestamp=0.0
        )
        
        input_state.joystick_inputs = [mock_joystick]
        joystick_data = input_state.get_joystick_state(0)
        print(f"  - Mock joystick data keys: {list(joystick_data.keys()) if joystick_data else None}")
        
        print("[PASS] Input state test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Input state test failed: {e}")
        traceback.print_exc()
        return False

def test_drone_metadata():
    """Test drone simulation metadata."""
    print("Testing drone metadata...")
    
    try:
        from pyjoysim.simulation.drone.metadata import DRONE_SIMULATION_METADATA
        from pyjoysim.simulation.manager import SimulationCategory
        
        metadata = DRONE_SIMULATION_METADATA
        print(f"  - Display name: {metadata.display_name}")
        print(f"  - Category: {metadata.category.value}")
        print(f"  - Version: {metadata.version}")
        print(f"  - Difficulty: {metadata.difficulty}")
        print(f"  - Tags count: {len(metadata.tags)}")
        print(f"  - Requirements: {metadata.requirements}")
        
        # Check if it's the correct category
        assert metadata.category == SimulationCategory.AERIAL
        print("  - Category verified as AERIAL")
        
        print("[PASS] Drone metadata test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Drone metadata test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all basic tests."""
    print("=== Basic Drone Simulation Test ===")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Physics Parameters", test_drone_physics_parameters),
        ("Basic Sensors", test_sensors_basic),
        ("Basic Flight Controller", test_flight_controller_basic),
        ("Input State", test_input_state),
        ("Drone Metadata", test_drone_metadata),
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
        print("[SUCCESS] All basic tests passed! Core drone system is functional.")
        return 0
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())