"""
Unit tests for car simulation components.

Tests the car physics, track system, and simulation logic.
"""

import unittest
import math
from unittest.mock import Mock, patch

from pyjoysim.simulation.vehicle import (
    Car, CarType, CarConfiguration, CarState, CarPhysics,
    CarSimulation, TrackLoader, TrackData, TrackWall, Checkpoint
)
from pyjoysim.physics import Vector2D


class TestCarConfiguration(unittest.TestCase):
    """Test car configuration system."""
    
    def test_car_config_creation(self):
        """Test car configuration creation."""
        config = CarConfiguration.get_config(CarType.SPORTS_CAR)
        
        self.assertIsInstance(config, CarConfiguration)
        self.assertEqual(config.mass, 1200)
        self.assertEqual(config.wheelbase, 2.5)
        self.assertGreater(config.max_engine_force, 0)
        self.assertGreater(config.max_brake_force, 0)
    
    def test_different_car_types(self):
        """Test that different car types have different configurations."""
        sports_config = CarConfiguration.get_config(CarType.SPORTS_CAR)
        suv_config = CarConfiguration.get_config(CarType.SUV)
        truck_config = CarConfiguration.get_config(CarType.TRUCK)
        
        # Mass should increase: sports car < SUV < truck
        self.assertLess(sports_config.mass, suv_config.mass)
        self.assertLess(suv_config.mass, truck_config.mass)
        
        # Sports car should have better performance characteristics
        self.assertGreater(sports_config.tire_grip, truck_config.tire_grip)
        self.assertLess(sports_config.drag_coefficient, truck_config.drag_coefficient)


class TestCarState(unittest.TestCase):
    """Test car state management."""
    
    def test_car_state_creation(self):
        """Test car state initialization."""
        state = CarState(
            position=Vector2D(10, 20),
            velocity=Vector2D(5, 0),
            angular_velocity=0.1,
            heading=math.pi/4,
            steering_angle=0.2,
            throttle=0.5,
            brake=0.0,
            handbrake=False,
            gear=2,
            rpm=3000,
            speed_kmh=0
        )
        
        self.assertEqual(state.position.x, 10)
        self.assertEqual(state.position.y, 20)
        self.assertEqual(state.gear, 2)
        self.assertEqual(state.rpm, 3000)
        
        # Speed should be calculated from velocity
        expected_speed = 5 * 3.6  # m/s to km/h
        self.assertAlmostEqual(state.speed_kmh, expected_speed, places=1)


class TestCarPhysics(unittest.TestCase):
    """Test car physics calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CarConfiguration.get_config(CarType.SPORTS_CAR)
        self.physics = CarPhysics(self.config)
    
    def test_physics_initialization(self):
        """Test physics system initialization."""
        self.assertEqual(self.physics.config, self.config)
        self.assertGreater(self.physics.air_density, 0)
        self.assertGreater(self.physics.tire_lateral_stiffness, 0)
    
    def test_force_calculation_stationary(self):
        """Test force calculation for stationary car."""
        state = CarState(
            position=Vector2D(0, 0),
            velocity=Vector2D(0, 0),
            angular_velocity=0,
            heading=0,
            steering_angle=0,
            throttle=0,
            brake=0,
            handbrake=False,
            gear=1,
            rpm=800,
            speed_kmh=0
        )
        
        force, torque = self.physics.calculate_forces(state, 0.016)
        
        # For stationary car with no input, forces should be minimal
        self.assertLess(abs(force.x), 100)  # Small forces due to numerical precision
        self.assertLess(abs(force.y), 100)
        self.assertLess(abs(torque), 10)
    
    def test_throttle_produces_forward_force(self):
        """Test that throttle input produces forward force."""
        state = CarState(
            position=Vector2D(0, 0),
            velocity=Vector2D(1, 0),  # Moving forward slowly
            angular_velocity=0,
            heading=0,  # Facing right (positive X)
            steering_angle=0,
            throttle=1.0,  # Full throttle
            brake=0,
            handbrake=False,
            gear=1,
            rpm=3000,
            speed_kmh=3.6
        )
        
        force, torque = self.physics.calculate_forces(state, 0.016)
        
        # Should have positive X force (forward)
        self.assertGreater(force.x, 0)
    
    def test_brake_opposes_motion(self):
        """Test that braking opposes motion."""
        state = CarState(
            position=Vector2D(0, 0),
            velocity=Vector2D(10, 0),  # Moving forward
            angular_velocity=0,
            heading=0,
            steering_angle=0,
            throttle=0,
            brake=1.0,  # Full brake
            handbrake=False,
            gear=1,
            rpm=2000,
            speed_kmh=36
        )
        
        force, torque = self.physics.calculate_forces(state, 0.016)
        
        # Brake force should oppose motion (negative X)
        self.assertLess(force.x, 0)


class TestTrackSystem(unittest.TestCase):
    """Test track loading and management."""
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation and crossing detection."""
        checkpoint = Checkpoint(
            position=Vector2D(10, 0),
            width=5,
            angle=math.pi/2,  # Vertical line
            checkpoint_id=1,
            is_finish_line=False
        )
        
        # Test crossing from left to right
        prev_pos = Vector2D(5, 0)
        current_pos = Vector2D(15, 0)
        self.assertTrue(checkpoint.is_crossed(prev_pos, current_pos))
        
        # Test not crossing (parallel movement)
        prev_pos = Vector2D(5, 5)
        current_pos = Vector2D(15, 5)
        self.assertFalse(checkpoint.is_crossed(prev_pos, current_pos))
    
    def test_track_data_serialization(self):
        """Test track data serialization to/from dict."""
        track = TrackData(
            name="Test Track",
            description="A test track",
            spawn_position=Vector2D(0, 0),
            spawn_heading=0.0
        )
        
        # Add a wall
        wall = TrackWall(
            start=Vector2D(-10, -5),
            end=Vector2D(10, -5),
            thickness=2.0
        )
        track.walls.append(wall)
        
        # Add a checkpoint
        checkpoint = Checkpoint(
            position=Vector2D(0, 0),
            width=10,
            angle=0,
            checkpoint_id=0,
            is_finish_line=True
        )
        track.checkpoints.append(checkpoint)
        
        # Serialize to dict
        data = track.to_dict()
        self.assertEqual(data["name"], "Test Track")
        self.assertEqual(len(data["walls"]), 1)
        self.assertEqual(len(data["checkpoints"]), 1)
        
        # Deserialize from dict
        track2 = TrackData.from_dict(data)
        self.assertEqual(track2.name, track.name)
        self.assertEqual(len(track2.walls), 1)
        self.assertEqual(len(track2.checkpoints), 1)
        self.assertEqual(track2.walls[0].start.x, wall.start.x)
        self.assertEqual(track2.checkpoints[0].checkpoint_id, checkpoint.checkpoint_id)
    
    def test_track_loader_basic_tracks(self):
        """Test track loader creating basic tracks."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch('pathlib.Path.mkdir'):
                with patch('builtins.open', create=True):
                    loader = TrackLoader("test_tracks")
                    loader.create_basic_tracks()
                    
                    # Should create tracks without errors
                    # (Actual file I/O is mocked)


class TestCarSimulation(unittest.TestCase):
    """Test car simulation integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock pygame to avoid actual window creation
        self.pygame_patcher = patch('pyjoysim.simulation.vehicle.car_simulation.pygame')
        self.mock_pygame = self.pygame_patcher.start()
        
        # Mock physics world
        self.physics_world_patcher = patch('pyjoysim.simulation.base.create_physics_world')
        self.mock_physics_world = self.physics_world_patcher.start()
        mock_world = Mock()
        mock_world.engine = Mock()
        self.mock_physics_world.return_value = mock_world
        
        # Mock rendering
        self.render_patcher = patch('pyjoysim.simulation.base.create_render_engine')
        self.mock_render = self.render_patcher.start()
        self.mock_render.return_value = Mock()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.pygame_patcher.stop()
        self.physics_world_patcher.stop() 
        self.render_patcher.stop()
    
    def test_car_simulation_creation(self):
        """Test car simulation creation."""
        sim = CarSimulation()
        
        self.assertEqual(sim.name, "car_simulation")
        self.assertEqual(sim.selected_track_name, "oval")
        self.assertIsNotNone(sim.track_loader)
    
    def test_control_input_processing(self):
        """Test that control inputs are processed correctly."""
        sim = CarSimulation()
        
        # Set control inputs
        sim.steering_input = 0.5
        sim.throttle_input = 0.8
        sim.brake_input = 0.2
        sim.handbrake_active = True
        
        self.assertEqual(sim.steering_input, 0.5)
        self.assertEqual(sim.throttle_input, 0.8)
        self.assertEqual(sim.brake_input, 0.2)
        self.assertTrue(sim.handbrake_active)


class TestCarIntegration(unittest.TestCase):
    """Test car and physics integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock physics engine
        self.mock_physics_engine = Mock()
        
        # Mock physics body creation
        mock_body = Mock()
        mock_body.position = Vector2D(0, 0)
        mock_body.velocity = Vector2D(0, 0)
        mock_body.angular_velocity = 0.0
        mock_body.angle = 0.0
        mock_body.is_active = True
        
        # Mock create_box to return our mock body
        self.create_box_patcher = patch('pyjoysim.simulation.vehicle.car_simulation.create_box')
        self.mock_create_box = self.create_box_patcher.start()
        self.mock_create_box.return_value = (mock_body, Mock())
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.create_box_patcher.stop()
    
    def test_car_creation_and_controls(self):
        """Test car creation and control application."""
        car = Car(
            self.mock_physics_engine,
            Vector2D(10, 20),
            CarType.SPORTS_CAR,
            "TestCar"
        )
        
        # Test initial state
        self.assertEqual(car.name, "TestCar")
        self.assertEqual(car.car_type, CarType.SPORTS_CAR)
        self.assertEqual(car.state.gear, 1)
        self.assertEqual(car.state.rpm, 800.0)
        
        # Test control application
        car.set_controls(
            throttle=0.7,
            brake=0.3,
            steering=0.5,
            handbrake=True
        )
        
        self.assertEqual(car.state.throttle, 0.7)
        self.assertEqual(car.state.brake, 0.3)
        self.assertTrue(car.state.handbrake)
        
        # Steering should be converted to radians
        max_angle = math.radians(car.config.max_steering_angle)
        expected_steering = 0.5 * max_angle
        self.assertAlmostEqual(car.state.steering_angle, expected_steering, places=5)
    
    def test_car_update_cycle(self):
        """Test car update cycle."""
        car = Car(
            self.mock_physics_engine,
            Vector2D(0, 0),
            CarType.SPORTS_CAR,
            "TestCar"
        )
        
        # Set some inputs
        car.set_controls(0.5, 0.0, 0.2, False)
        
        # Update should not raise errors
        car.update(0.016)  # 60 FPS
        
        # RPM should be updated
        self.assertGreaterEqual(car.state.rpm, 800.0)
    
    def test_dashboard_data(self):
        """Test dashboard data extraction."""
        car = Car(
            self.mock_physics_engine,
            Vector2D(0, 0),
            CarType.SPORTS_CAR,
            "TestCar"
        )
        
        dashboard = car.get_dashboard_data()
        
        # Should contain expected keys
        expected_keys = ["speed_kmh", "rpm", "gear", "throttle", "brake", "steering_angle"]
        for key in expected_keys:
            self.assertIn(key, dashboard)
        
        # Values should be reasonable
        self.assertGreaterEqual(dashboard["speed_kmh"], 0)
        self.assertGreaterEqual(dashboard["rpm"], 0)
        self.assertGreaterEqual(dashboard["gear"], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)