#!/usr/bin/env python3
"""
Advanced Simulations Test (Phase 3 Week 5-6)

This script tests the spaceship and submarine simulation systems to ensure all components
work together correctly.
"""

import sys
import time
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test that all advanced simulation modules can be imported."""
    print("Testing advanced simulation imports...")
    
    try:
        # Core 3D modules
        from pyjoysim.physics.physics3d import Vector3D, Quaternion
        print("  - Vector3D and Quaternion imported")
        
        # Spaceship modules
        from pyjoysim.simulation.spaceship.physics import SpaceshipPhysics, SpaceEnvironment, SpaceshipPhysicsParameters
        from pyjoysim.simulation.spaceship.propulsion import PropulsionSystem, RCSSystem, EngineParameters, EngineType
        from pyjoysim.simulation.spaceship.life_support import LifeSupportSystem, LifeSupportParameters
        from pyjoysim.simulation.spaceship.spaceship_simulation import SpaceshipSimulation
        from pyjoysim.simulation.spaceship.metadata import SPACESHIP_SIMULATION_METADATA
        print("  - Spaceship modules imported successfully")
        
        # Submarine modules
        from pyjoysim.simulation.submarine.physics import SubmarinePhysics, UnderwaterEnvironment, SubmarinePhysicsParameters
        from pyjoysim.simulation.submarine.ballast import BallastSystem, BallastTank, BallastTankType
        from pyjoysim.simulation.submarine.sonar import SonarSystem, SonarMode, SonarContact
        from pyjoysim.simulation.submarine.submarine_simulation import SubmarineSimulation
        from pyjoysim.simulation.submarine.metadata import SUBMARINE_SIMULATION_METADATA
        print("  - Submarine modules imported successfully")
        
        # Input system
        from pyjoysim.input.input_processor import InputState
        print("  - Input system imported")
        
        print("[PASS] All advanced simulation imports successful")
        return True
        
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False

def test_spaceship_physics():
    """Test spaceship physics components."""
    print("Testing spaceship physics...")
    
    try:
        from pyjoysim.simulation.spaceship.physics import SpaceshipPhysics, SpaceEnvironment, SpaceshipPhysicsParameters
        from pyjoysim.physics.physics3d import Vector3D, PhysicsObject3D, Shape3D, Shape3DType, Body3DType
        
        # Create space environment
        environment = SpaceEnvironment()
        print(f"  - Space environment created with {len(environment.celestial_bodies)} celestial bodies")
        
        # Create spaceship physics
        params = SpaceshipPhysicsParameters()
        spaceship_physics = SpaceshipPhysics(params, environment)
        print(f"  - Spaceship physics created (mass: {params.dry_mass}kg, fuel: {params.fuel_capacity}kg)")
        
        # Test gravitational calculations
        earth_position = Vector3D(149597870700, 0, 0)  # 1 AU from Sun
        test_mass = 1000.0  # kg
        grav_force = environment.calculate_gravitational_force(earth_position, test_mass)
        print(f"  - Gravitational force at Earth orbit: {grav_force.magnitude():.2e}N")
        
        # Test propulsion
        spaceship_physics.set_main_engine_throttle(0.5)
        thrust_force = spaceship_physics.calculate_thrust_force()
        print(f"  - Thrust force at 50% throttle: {thrust_force.magnitude():.0f}N")
        
        # Test fuel consumption
        fuel_consumed = spaceship_physics.calculate_fuel_consumption(1000.0, 1.0)  # 1000N for 1 second
        print(f"  - Fuel consumption (1000N, 1s): {fuel_consumed:.3f}kg")
        
        print("[PASS] Spaceship physics test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Spaceship physics test failed: {e}")
        traceback.print_exc()
        return False

def test_spaceship_propulsion():
    """Test spaceship propulsion systems."""
    print("Testing spaceship propulsion...")
    
    try:
        from pyjoysim.simulation.spaceship.propulsion import PropulsionSystem, RCSSystem, EngineParameters, EngineType
        from pyjoysim.physics.physics3d import Vector3D
        
        # Create main engine
        engine_params = EngineParameters(
            name="Test Engine",
            engine_type=EngineType.CHEMICAL_ROCKET,
            max_thrust=1000.0,
            specific_impulse=300.0,
            fuel_flow_rate=0.34
        )
        
        propulsion = PropulsionSystem(engine_params)
        print(f"  - Main engine created: {engine_params.name}")
        
        # Test engine ignition and throttle
        propulsion.ignite()
        propulsion.set_throttle(0.75)
        thrust, fuel = propulsion.update(1.0)  # 1 second
        print(f"  - Engine test: {thrust:.0f}N thrust, {fuel:.3f}kg fuel consumed")
        
        # Create RCS system
        rcs = RCSSystem()
        print(f"  - RCS created with {len(rcs.thrusters)} thrusters")
        
        # Test RCS translation
        rcs.set_translation_command(Vector3D(0.5, 0.8, 0.0))
        rcs_force, rcs_fuel = rcs.update(1.0)
        print(f"  - RCS test: {rcs_force.magnitude():.1f}N force, {rcs_fuel:.3f}kg fuel")
        
        print("[PASS] Spaceship propulsion test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Spaceship propulsion test failed: {e}")
        traceback.print_exc()
        return False

def test_spaceship_life_support():
    """Test spaceship life support systems."""
    print("Testing spaceship life support...")
    
    try:
        from pyjoysim.simulation.spaceship.life_support import LifeSupportSystem, LifeSupportParameters
        
        # Create life support system
        params = LifeSupportParameters(crew_count=3)
        life_support = LifeSupportSystem(params)
        print(f"  - Life support created for {params.crew_count} crew members")
        
        # Test resource levels
        status = life_support.get_status()
        print(f"  - Initial oxygen: {status['oxygen_percentage']:.1f}%")
        print(f"  - Initial power: {status['power_percentage']:.1f}%")
        print(f"  - Cabin temperature: {status['cabin_temperature']:.1f}°C")
        
        # Simulate time passage
        life_support.set_power_generation(3.0)  # 3kW solar panels
        life_support.update(3600.0, -270.0)     # 1 hour in space
        
        status_after = life_support.get_status()
        print(f"  - After 1 hour - Oxygen: {status_after['oxygen_percentage']:.1f}%")
        print(f"  - After 1 hour - Power: {status_after['power_percentage']:.1f}%")
        print(f"  - Alert level: {status_after['alert_level']}")
        
        print("[PASS] Spaceship life support test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Spaceship life support test failed: {e}")
        traceback.print_exc()
        return False

def test_submarine_physics():
    """Test submarine physics components."""
    print("Testing submarine physics...")
    
    try:
        from pyjoysim.simulation.submarine.physics import (
            SubmarinePhysics, UnderwaterEnvironment, SubmarinePhysicsParameters, WaterType
        )
        from pyjoysim.physics.physics3d import Vector3D, PhysicsObject3D, Shape3D, Shape3DType, Body3DType
        
        # Create underwater environment
        environment = UnderwaterEnvironment(WaterType.SALT_WATER)
        print(f"  - Underwater environment created with {len(environment.water_layers)} layers")
        
        # Test pressure calculation
        pressure_100m = environment.calculate_pressure(100.0)
        print(f"  - Pressure at 100m: {pressure_100m/1000:.0f} kPa")
        
        # Test water properties
        layer = environment.get_water_properties(200.0)
        print(f"  - Water at 200m: {layer.temperature}°C, density {layer.density} kg/m³")
        
        # Create submarine physics
        params = SubmarinePhysicsParameters()
        submarine_physics = SubmarinePhysics(params, environment)
        print(f"  - Submarine physics created (length: {params.length}m, displacement: {params.displacement}t)")
        
        # Test buoyancy
        submerged_volume = submarine_physics.calculate_submerged_volume(50.0)
        buoyancy = environment.calculate_buoyancy_force(submerged_volume, 50.0)
        print(f"  - Buoyancy at 50m depth: {buoyancy/1000:.0f} kN")
        
        # Test drag
        velocity = Vector3D(5.0, 0, 0)  # 5 m/s forward
        drag = environment.calculate_drag_force(velocity, 80.0, 0.08, 50.0)  # 80 m² area, Cd=0.08
        print(f"  - Drag force at 5 m/s: {drag.magnitude()/1000:.1f} kN")
        
        print("[PASS] Submarine physics test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Submarine physics test failed: {e}")
        traceback.print_exc()
        return False

def test_submarine_ballast():
    """Test submarine ballast system."""
    print("Testing submarine ballast...")
    
    try:
        from pyjoysim.simulation.submarine.ballast import BallastSystem, BallastTankType
        
        # Create ballast system
        ballast = BallastSystem()
        print(f"  - Ballast system created with {len(ballast.tanks)} tanks")
        
        # Test tank operations
        ballast.flood_main_ballast_tanks()
        
        # Simulate ballast operations
        for i in range(10):
            ballast.update(1.0, 10.0 * i, 101325 + 10000 * i)  # 1 second steps, increasing depth
        
        status = ballast.get_status()
        print(f"  - Main ballast level: {status['main_ballast_level']:.1f}%")
        print(f"  - HP air remaining: {status['hp_air_percentage']:.1f}%")
        print(f"  - Total ballast weight: {status['total_ballast_weight']/1000:.1f} tonnes")
        
        # Test emergency blow
        ballast.emergency_blow_all()
        print("  - Emergency blow activated")
        
        main_tanks = ballast.get_tanks_by_type(BallastTankType.MAIN_BALLAST)
        print(f"  - Main ballast tanks: {len(main_tanks)} tanks")
        
        print("[PASS] Submarine ballast test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Submarine ballast test failed: {e}")
        traceback.print_exc()
        return False

def test_submarine_sonar():
    """Test submarine sonar system."""
    print("Testing submarine sonar...")
    
    try:
        from pyjoysim.simulation.submarine.sonar import SonarSystem, SonarMode, ContactType
        from pyjoysim.physics.physics3d import Vector3D
        
        # Create sonar system
        sonar = SonarSystem()
        print(f"  - Sonar system created with {sonar.hydrophones} hydrophones")
        
        # Test mode switching
        sonar.set_mode(SonarMode.ACTIVE)
        print(f"  - Sonar mode set to: {sonar.mode.value}")
        
        # Test active ping
        submarine_pos = Vector3D(0, -100, 0)  # 100m depth
        ping_sent = sonar.ping(submarine_pos, 100.0)
        print(f"  - Active ping sent: {ping_sent}")
        
        # Simulate passive listening
        for i in range(5):
            sonar.passive_listen(submarine_pos, 100.0, 1.0)
        
        # Check contacts
        status = sonar.get_status()
        print(f"  - Total contacts detected: {status['total_contacts']}")
        print(f"  - Total pings sent: {status['total_pings']}")
        
        tactical = sonar.get_tactical_display()
        print(f"  - Active range: {tactical['active_range']:.0f}m")
        print(f"  - Passive range: {tactical['passive_range']:.0f}m")
        
        print("[PASS] Submarine sonar test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Submarine sonar test failed: {e}")
        traceback.print_exc()
        return False

def test_simulation_metadata():
    """Test simulation metadata and registration."""
    print("Testing simulation metadata...")
    
    try:
        from pyjoysim.simulation.spaceship.metadata import SPACESHIP_SIMULATION_METADATA
        from pyjoysim.simulation.submarine.metadata import SUBMARINE_SIMULATION_METADATA
        from pyjoysim.simulation.manager import SimulationCategory
        
        # Test spaceship metadata
        spaceship_meta = SPACESHIP_SIMULATION_METADATA
        print(f"  - Spaceship: {spaceship_meta.display_name}")
        print(f"    Category: {spaceship_meta.category.value}")
        print(f"    Difficulty: {spaceship_meta.difficulty}")
        print(f"    Tags: {len(spaceship_meta.tags)} tags")
        
        # Test submarine metadata
        submarine_meta = SUBMARINE_SIMULATION_METADATA
        print(f"  - Submarine: {submarine_meta.display_name}")
        print(f"    Category: {submarine_meta.category.value}")
        print(f"    Difficulty: {submarine_meta.difficulty}")
        print(f"    Tags: {len(submarine_meta.tags)} tags")
        
        # Verify categories exist
        assert spaceship_meta.category in [SimulationCategory.AERIAL, SimulationCategory.VEHICLE]
        assert submarine_meta.category == SimulationCategory.VEHICLE
        print("  - Categories verified")
        
        print("[PASS] Simulation metadata test passed")
        return True
        
    except Exception as e:
        print(f"[FAIL] Simulation metadata test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all advanced simulation tests."""
    print("=== Advanced Simulations Test (Phase 3 Week 5-6) ===")
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Spaceship Physics", test_spaceship_physics),
        ("Spaceship Propulsion", test_spaceship_propulsion),
        ("Spaceship Life Support", test_spaceship_life_support),
        ("Submarine Physics", test_submarine_physics),
        ("Submarine Ballast", test_submarine_ballast),
        ("Submarine Sonar", test_submarine_sonar),
        ("Simulation Metadata", test_simulation_metadata),
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
        print("[SUCCESS] All advanced simulation tests passed! Spaceship and submarine systems are ready.")
        return 0
    else:
        print("[ERROR] Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())