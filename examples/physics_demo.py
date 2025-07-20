#!/usr/bin/env python3
"""
PyJoySim Physics System Demo

This script demonstrates the physics engine capabilities including:
- Rigid body dynamics
- Collision detection
- Constraints and joints
- Material properties

Usage:
    python examples/physics_demo.py [--headless] [--duration SECONDS]
"""

import sys
import time
import argparse
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyjoysim.physics import (
    create_physics_world,
    PhysicsEngineType,
    Vector2D,
    create_ball,
    create_box,
    create_static_wall,
    StandardMaterials,
    PinJoint,
    SpringJoint,
    CollisionHandler,
    CollisionEvent,
    CollisionEventType
)
from pyjoysim.core.logging import get_logger


class PhysicsDemo:
    """Interactive demo of the PyJoySim physics system."""
    
    def __init__(self, headless: bool = False):
        """Initialize the demo."""
        self.logger = get_logger("physics_demo")
        self.headless = headless
        
        # Create physics world
        self.world = create_physics_world(
            PhysicsEngineType.PHYSICS_2D,
            Vector2D(0, -9.81)  # Earth gravity
        )
        
        # Demo objects
        self.objects = {}
        self.constraints = {}
        
        # Collision tracking
        self.collision_count = 0
        
        self.logger.info("Physics demo initialized", extra={"headless": headless})
    
    def setup_demo_scene(self):
        """Set up the demo scene with various physics objects."""
        self.logger.info("Setting up demo scene")
        
        # Create ground
        ground_body, ground_collider = create_static_wall(
            self.world.engine,
            Vector2D(0, -5),
            width=20.0,
            height=1.0,
            material=StandardMaterials.CONCRETE,
            name="Ground"
        )
        ground_id = self.world.add_object(ground_body, "static")
        self.objects["ground"] = (ground_body, ground_collider)
        
        # Create left wall
        left_wall_body, left_wall_collider = create_static_wall(
            self.world.engine,
            Vector2D(-10, 0),
            width=1.0,
            height=20.0,
            material=StandardMaterials.CONCRETE,
            name="LeftWall"
        )
        self.world.add_object(left_wall_body, "static")
        self.objects["left_wall"] = (left_wall_body, left_wall_collider)
        
        # Create right wall
        right_wall_body, right_wall_collider = create_static_wall(
            self.world.engine,
            Vector2D(10, 0),
            width=1.0,
            height=20.0,
            material=StandardMaterials.CONCRETE,
            name="RightWall"
        )
        self.world.add_object(right_wall_body, "static")
        self.objects["right_wall"] = (right_wall_body, right_wall_collider)
        
        # Create bouncing balls
        for i in range(3):
            ball_body, ball_collider = create_ball(
                self.world.engine,
                Vector2D(-6 + i * 3, 8),
                radius=0.5,
                mass=1.0,
                material=StandardMaterials.RUBBER,
                name=f"Ball{i+1}"
            )
            self.world.add_object(ball_body, "dynamic")
            self.objects[f"ball_{i}"] = (ball_body, ball_collider)
            
            # Give balls initial velocity
            ball_body.apply_impulse(Vector2D(i * 2 - 2, 0))
        
        # Create boxes
        for i in range(2):
            box_body, box_collider = create_box(
                self.world.engine,
                Vector2D(-2 + i * 4, 6),
                width=1.0,
                height=1.0,
                mass=2.0,
                material=StandardMaterials.WOOD,
                name=f"Box{i+1}"
            )
            self.world.add_object(box_body, "dynamic")
            self.objects[f"box_{i}"] = (box_body, box_collider)
        
        # Create pendulum
        pendulum_anchor, _ = create_ball(
            self.world.engine,
            Vector2D(5, 8),
            radius=0.1,
            mass=100.0,  # Heavy anchor
            material=StandardMaterials.METAL,
            name="PendulumAnchor"
        )
        self.world.add_object(pendulum_anchor, "kinematic")
        
        pendulum_bob, _ = create_ball(
            self.world.engine,
            Vector2D(5, 5),
            radius=0.3,
            mass=1.0,
            material=StandardMaterials.METAL,
            name="PendulumBob"
        )
        self.world.add_object(pendulum_bob, "dynamic")
        
        # Connect with pin joint
        pendulum_joint = PinJoint(
            self.world.engine,
            pendulum_anchor,
            pendulum_bob,
            Vector2D(0, 0),  # Anchor at center
            Vector2D(0, 0),  # Bob at center
            name="PendulumJoint"
        )
        self.world.constraint_manager.add_constraint(pendulum_joint, "pendulum")
        
        self.objects["pendulum_anchor"] = (pendulum_anchor, None)
        self.objects["pendulum_bob"] = (pendulum_bob, None)
        self.constraints["pendulum_joint"] = pendulum_joint
        
        # Give pendulum initial swing
        pendulum_bob.apply_impulse(Vector2D(3, 0))
        
        # Create spring system
        spring_anchor, _ = create_box(
            self.world.engine,
            Vector2D(-8, 4),
            width=0.5,
            height=0.5,
            mass=100.0,  # Heavy anchor
            material=StandardMaterials.METAL,
            name="SpringAnchor"
        )
        self.world.add_object(spring_anchor, "kinematic")
        
        spring_mass, _ = create_ball(
            self.world.engine,
            Vector2D(-8, 1),
            radius=0.4,
            mass=1.0,
            material=StandardMaterials.BOUNCY,
            name="SpringMass"
        )
        self.world.add_object(spring_mass, "dynamic")
        
        # Connect with spring
        spring_joint = SpringJoint(
            self.world.engine,
            spring_anchor,
            spring_mass,
            Vector2D(0, 0),
            Vector2D(0, 0),
            rest_length=2.0,
            spring_constant=500.0,
            damping=10.0,
            name="SpringJoint"
        )
        self.world.constraint_manager.add_constraint(spring_joint, "spring")
        
        self.objects["spring_anchor"] = (spring_anchor, None)
        self.objects["spring_mass"] = (spring_mass, None)
        self.constraints["spring_joint"] = spring_joint
        
        # Setup collision handler
        collision_handler = CollisionHandler(
            collision_callback=self._on_collision
        )
        self.world.set_collision_handler(collision_handler)
        
        self.logger.info("Demo scene setup complete", extra={
            "total_objects": self.world.get_object_count(),
            "total_constraints": self.world.constraint_manager.get_constraint_count()
        })
    
    def _on_collision(self, event: CollisionEvent):
        """Handle collision events."""
        if event.event_type == CollisionEventType.BEGIN:
            self.collision_count += 1
            
            if not self.headless:
                print(f"Collision #{self.collision_count}: Objects {event.object_a_id} and {event.object_b_id}")
    
    def run_simulation(self, duration: float = 30.0):
        """Run the physics simulation."""
        self.logger.info("Starting physics simulation", extra={"duration": duration})
        
        if not self.headless:
            print(f"\nPhysics Simulation Running ({duration}s)")
            print("=" * 50)
            print("Watch the console for collision events...")
            print("Objects: bouncing balls, boxes, pendulum, spring system")
            print("=" * 50)
        
        # Start simulation
        self.world.start_simulation()
        
        start_time = time.time()
        last_status_time = start_time
        frame_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                current_time = time.time()
                dt = min(1.0 / 60.0, 0.033)  # Cap at ~30 FPS
                
                # Step physics
                self.world.step(dt)
                frame_count += 1
                
                # Print status every 5 seconds
                if not self.headless and (current_time - last_status_time) >= 5.0:
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed
                    
                    print(f"Time: {elapsed:.1f}s | FPS: {fps:.1f} | Collisions: {self.collision_count}")
                    
                    # Print some object positions
                    if "ball_0" in self.objects:
                        ball = self.objects["ball_0"][0]
                        pos = ball.position
                        print(f"  Ball 1 position: ({pos.x:.2f}, {pos.y:.2f})")
                    
                    if "pendulum_bob" in self.objects:
                        bob = self.objects["pendulum_bob"][0]
                        pos = bob.position
                        print(f"  Pendulum bob: ({pos.x:.2f}, {pos.y:.2f})")
                    
                    last_status_time = current_time
                
                # Small sleep to prevent excessive CPU usage
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        
        # Stop simulation
        self.world.stop_simulation()
        
        # Final statistics
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        physics_stats = self.world.engine.get_stats()
        
        results = {
            "duration": elapsed,
            "frame_count": frame_count,
            "average_fps": fps,
            "total_collisions": self.collision_count,
            "total_objects": self.world.get_object_count(),
            "total_constraints": self.world.constraint_manager.get_constraint_count(),
            "physics_steps": physics_stats.step_count,
            "average_step_time": physics_stats.average_step_time * 1000,  # ms
        }
        
        if not self.headless:
            print(f"\nSimulation Complete!")
            print("=" * 50)
            print(f"Duration: {results['duration']:.2f}s")
            print(f"Frames: {results['frame_count']}")
            print(f"Average FPS: {results['average_fps']:.1f}")
            print(f"Total Collisions: {results['total_collisions']}")
            print(f"Physics Steps: {results['physics_steps']}")
            print(f"Avg Step Time: {results['average_step_time']:.2f}ms")
            print("=" * 50)
        
        self.logger.info("Simulation completed", extra=results)
        
        return results
    
    def test_physics_features(self):
        """Test specific physics features."""
        self.logger.info("Testing physics features")
        
        if not self.headless:
            print("\nTesting Physics Features:")
            print("-" * 30)
        
        tests_passed = 0
        total_tests = 0
        
        # Test 1: Object creation and positioning
        total_tests += 1
        try:
            test_ball, _ = create_ball(
                self.world.engine,
                Vector2D(100, 100),
                radius=1.0,
                name="TestBall"
            )
            
            pos = test_ball.position
            if abs(pos.x - 100) < 0.1 and abs(pos.y - 100) < 0.1:
                tests_passed += 1
                if not self.headless:
                    print("✓ Object positioning test passed")
            else:
                if not self.headless:
                    print("✗ Object positioning test failed")
                    
            test_ball.destroy()
            
        except Exception as e:
            self.logger.error("Object positioning test failed", extra={"error": str(e)})
            if not self.headless:
                print("✗ Object positioning test failed")
        
        # Test 2: Force application
        total_tests += 1
        try:
            test_body, _ = create_ball(
                self.world.engine,
                Vector2D(0, 10),
                radius=0.5,
                name="ForceTestBall"
            )
            self.world.add_object(test_body, "test")
            
            # Apply upward force
            test_body.apply_force(Vector2D(0, 100))
            
            # Step simulation briefly
            for _ in range(10):
                self.world.step(1.0 / 60.0)
            
            # Check if object moved (should move up against gravity)
            pos = test_body.position
            if pos.y > 10:
                tests_passed += 1
                if not self.headless:
                    print("✓ Force application test passed")
            else:
                if not self.headless:
                    print("✗ Force application test failed")
            
            self.world.remove_group("test")
            
        except Exception as e:
            self.logger.error("Force application test failed", extra={"error": str(e)})
            if not self.headless:
                print("✗ Force application test failed")
        
        # Test 3: Constraint creation
        total_tests += 1
        try:
            body_a, _ = create_ball(
                self.world.engine,
                Vector2D(0, 0),
                radius=0.5,
                name="ConstraintTestA"
            )
            body_b, _ = create_ball(
                self.world.engine,
                Vector2D(2, 0),
                radius=0.5,
                name="ConstraintTestB"
            )
            
            joint = PinJoint(
                self.world.engine,
                body_a,
                body_b,
                Vector2D(0, 0),
                Vector2D(0, 0),
                name="TestJoint"
            )
            
            if joint.constraint_id is not None:
                tests_passed += 1
                if not self.headless:
                    print("✓ Constraint creation test passed")
            else:
                if not self.headless:
                    print("✗ Constraint creation test failed")
            
            # Cleanup
            joint.destroy()
            body_a.destroy()
            body_b.destroy()
            
        except Exception as e:
            self.logger.error("Constraint creation test failed", extra={"error": str(e)})
            if not self.headless:
                print("✗ Constraint creation test failed")
        
        # Test results
        success_rate = (tests_passed / total_tests) * 100 if total_tests > 0 else 0
        
        if not self.headless:
            print(f"\nTest Results: {tests_passed}/{total_tests} passed ({success_rate:.1f}%)")
        
        self.logger.info("Physics feature tests completed", extra={
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "success_rate": success_rate
        })
        
        return tests_passed == total_tests
    
    def cleanup(self):
        """Clean up demo resources."""
        self.logger.info("Cleaning up physics demo")
        
        # Clear all objects and constraints
        for group in self.world.get_group_names():
            self.world.remove_group(group)
        
        for group in self.world.constraint_manager.get_group_names():
            self.world.constraint_manager.remove_group(group)
        
        # Shutdown world
        self.world.shutdown()
        
        self.logger.info("Physics demo cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PyJoySim Physics System Demo")
    parser.add_argument("--headless", action="store_true", 
                       help="Run without console output (for testing)")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Simulation duration in seconds")
    parser.add_argument("--test-only", action="store_true",
                       help="Run only feature tests, skip full simulation")
    
    args = parser.parse_args()
    
    # Create and initialize demo
    demo = PhysicsDemo(headless=args.headless)
    
    success = True
    
    try:
        if args.test_only:
            # Run only feature tests
            success = demo.test_physics_features()
        else:
            # Setup scene and run full simulation
            demo.setup_demo_scene()
            
            # Run feature tests first
            test_success = demo.test_physics_features()
            
            # Run simulation
            results = demo.run_simulation(args.duration)
            
            # Check if simulation ran successfully
            simulation_success = (
                results["frame_count"] > 0 and
                results["average_fps"] > 10.0 and
                results["physics_steps"] > 0
            )
            
            success = test_success and simulation_success
            
            if not args.headless:
                status = "SUCCESS" if success else "FAILED"
                print(f"\nOverall Demo Status: {status}")
        
        return 0 if success else 1
    
    except Exception as e:
        print(f"Demo error: {e}")
        return 1
        
    finally:
        demo.cleanup()


if __name__ == "__main__":
    sys.exit(main())