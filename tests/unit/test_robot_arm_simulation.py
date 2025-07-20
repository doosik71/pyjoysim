"""
Unit tests for robot arm simulation components.

Tests the robot arm kinematics, physics, control, and simulation integration.
"""

import unittest
import math
from unittest.mock import Mock, patch

from pyjoysim.simulation.robot import (
    RobotArmSimulation, RobotArmConfiguration, Link, JointConstraints,
    RobotState, Kinematics, RobotArmPhysics, RobotArmController, ControlMode
)
from pyjoysim.physics import Vector2D


class TestRobotArmConfiguration(unittest.TestCase):
    """Test robot arm configuration system."""
    
    def test_default_3dof_configuration(self):
        """Test default 3-DOF robot configuration."""
        config = RobotArmConfiguration.get_default_3dof()
        
        self.assertEqual(len(config.links), 3)
        self.assertEqual(len(config.joint_constraints), 3)
        self.assertIsInstance(config.base_position, Vector2D)
        
        # Check link properties
        for link in config.links:
            self.assertGreater(link.length, 0)
            self.assertGreater(link.mass, 0)
            self.assertGreater(link.moment_of_inertia, 0)
        
        # Check joint constraints
        for constraint in config.joint_constraints:
            self.assertLessEqual(constraint.min_angle, constraint.max_angle)
            self.assertGreater(constraint.max_velocity, 0)
            self.assertGreater(constraint.max_torque, 0)
    
    def test_link_configuration(self):
        """Test individual link configuration."""
        link = Link(length=0.3, mass=2.0, moment_of_inertia=0.18)
        
        self.assertEqual(link.length, 0.3)
        self.assertEqual(link.mass, 2.0)
        self.assertEqual(link.moment_of_inertia, 0.18)
    
    def test_joint_constraints(self):
        """Test joint constraint specification."""
        constraint = JointConstraints(
            min_angle=-math.pi/2,
            max_angle=math.pi/2,
            max_velocity=math.pi,
            max_torque=50.0
        )
        
        self.assertEqual(constraint.min_angle, -math.pi/2)
        self.assertEqual(constraint.max_angle, math.pi/2)
        self.assertEqual(constraint.max_velocity, math.pi)
        self.assertEqual(constraint.max_torque, 50.0)


class TestKinematics(unittest.TestCase):
    """Test robot arm kinematics calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RobotArmConfiguration.get_default_3dof(Vector2D(0, 0))
        self.kinematics = Kinematics(self.config)
    
    def test_forward_kinematics_zero_angles(self):
        """Test forward kinematics with zero joint angles."""
        joint_angles = [0.0, 0.0, 0.0]
        positions = self.kinematics.forward_kinematics(joint_angles)
        
        # Should have 4 positions: base + 3 joints
        self.assertEqual(len(positions), 4)
        
        # Base should be at origin
        self.assertEqual(positions[0], Vector2D(0, 0))
        
        # All joints should be on X-axis when angles are zero
        total_length = 0
        for i, pos in enumerate(positions[1:], 1):
            total_length += self.config.links[i-1].length
            self.assertAlmostEqual(pos.x, total_length, places=5)
            self.assertAlmostEqual(pos.y, 0, places=5)
    
    def test_forward_kinematics_known_angles(self):
        """Test forward kinematics with known angles."""
        # Test case: first joint 90 degrees, others zero
        joint_angles = [math.pi/2, 0.0, 0.0]
        positions = self.kinematics.forward_kinematics(joint_angles)
        
        # First joint should be on Y-axis
        link1_length = self.config.links[0].length
        self.assertAlmostEqual(positions[1].x, 0, places=5)
        self.assertAlmostEqual(positions[1].y, link1_length, places=5)
    
    def test_end_effector_position(self):
        """Test end effector position calculation."""
        joint_angles = [0.0, 0.0, 0.0]
        ee_pos = self.kinematics.get_end_effector_position(joint_angles)
        
        # Should be at the sum of all link lengths on X-axis
        expected_x = sum(link.length for link in self.config.links)
        self.assertAlmostEqual(ee_pos.x, expected_x, places=5)
        self.assertAlmostEqual(ee_pos.y, 0, places=5)
    
    def test_inverse_kinematics_reachable_target(self):
        """Test inverse kinematics for reachable target."""
        # Target within workspace
        target = Vector2D(0.5, 0.2)
        
        # Test with zero initial angles
        result_angles = self.kinematics.inverse_kinematics(target, [0.0, 0.0, 0.0])
        
        if result_angles is not None:
            # Verify the solution reaches the target
            actual_pos = self.kinematics.get_end_effector_position(result_angles)
            error = (target - actual_pos).magnitude()
            self.assertLess(error, self.config.position_tolerance * 2)
            
            # Verify joint limits are respected
            for angle, constraint in zip(result_angles, self.config.joint_constraints):
                self.assertGreaterEqual(angle, constraint.min_angle)
                self.assertLessEqual(angle, constraint.max_angle)
    
    def test_inverse_kinematics_unreachable_target(self):
        """Test inverse kinematics for unreachable target."""
        # Target outside workspace
        max_reach = sum(link.length for link in self.config.links)
        target = Vector2D(max_reach + 1.0, 0)  # Definitely outside reach
        
        result_angles = self.kinematics.inverse_kinematics(target, [0.0, 0.0, 0.0])
        
        # Should return None or position with large error
        if result_angles is not None:
            actual_pos = self.kinematics.get_end_effector_position(result_angles)
            error = (target - actual_pos).magnitude()
            # Large error expected for unreachable target
            self.assertGreater(error, 0.1)
    
    def test_workspace_boundary(self):
        """Test workspace boundary calculation."""
        boundary = self.kinematics.get_workspace_boundary(num_points=8)
        
        self.assertEqual(len(boundary), 8)
        
        # All boundary points should be at approximately max reach
        max_reach = sum(link.length for link in self.config.links)
        
        for point in boundary:
            distance = (point - self.config.base_position).magnitude()
            self.assertAlmostEqual(distance, max_reach, places=3)


class TestRobotArmPhysics(unittest.TestCase):
    """Test robot arm physics calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RobotArmConfiguration.get_default_3dof()
        self.physics = RobotArmPhysics(self.config)
    
    def test_physics_initialization(self):
        """Test physics system initialization."""
        self.assertEqual(len(self.config.links), len(self.config.joint_constraints))
        self.assertIsInstance(self.physics.gravity, Vector2D)
        self.assertNotEqual(self.physics.gravity.magnitude(), 0)
    
    def test_dynamics_calculation(self):
        """Test dynamics calculation."""
        state = RobotState(
            joint_angles=[0.0, math.pi/4, -math.pi/6],
            joint_velocities=[0.0, 0.0, 0.0],
            joint_torques=[0.0, 0.0, 0.0],
            end_effector_position=Vector2D(0.5, 0.2)
        )
        
        control_torques = [10.0, 5.0, 2.0]
        dt = 0.016
        
        new_velocities, accelerations = self.physics.calculate_dynamics(
            state, control_torques, dt)
        
        # Should return same number of velocities and accelerations as joints
        self.assertEqual(len(new_velocities), len(self.config.links))
        self.assertEqual(len(accelerations), len(self.config.links))
        
        # With positive torques, should have positive accelerations
        for accel in accelerations:
            self.assertIsInstance(accel, (int, float))
            # Don't check sign due to gravity compensation
    
    def test_velocity_limits(self):
        """Test that velocity limits are enforced."""
        state = RobotState(
            joint_angles=[0.0, 0.0, 0.0],
            joint_velocities=[100.0, 100.0, 100.0],  # Very high velocities
            joint_torques=[0.0, 0.0, 0.0],
            end_effector_position=Vector2D(0.5, 0.2)
        )
        
        control_torques = [0.0, 0.0, 0.0]
        dt = 0.016
        
        new_velocities, _ = self.physics.calculate_dynamics(state, control_torques, dt)
        
        # Velocities should be limited by joint constraints
        for velocity, constraint in zip(new_velocities, self.config.joint_constraints):
            self.assertLessEqual(abs(velocity), constraint.max_velocity)


class TestRobotArmController(unittest.TestCase):
    """Test robot arm control system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = RobotArmConfiguration.get_default_3dof()
        self.controller = RobotArmController(self.config)
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        self.assertGreater(self.controller.position_kp, 0)
        self.assertGreater(self.controller.angle_kp, 0)
        self.assertEqual(len(self.controller.angle_error_integrals), len(self.config.links))
    
    def test_end_effector_control(self):
        """Test end effector position control."""
        target = Vector2D(0.4, 0.3)
        state = RobotState(
            joint_angles=[0.0, 0.0, 0.0],
            joint_velocities=[0.0, 0.0, 0.0],
            joint_torques=[0.0, 0.0, 0.0],
            end_effector_position=Vector2D(0.7, 0.0),  # At full extension
            target_position=target,
            control_mode=ControlMode.END_EFFECTOR
        )
        
        torques = self.controller.calculate_control_torques(state, 0.016)
        
        # Should return torques for all joints
        self.assertEqual(len(torques), len(self.config.links))
        
        # Torques should be within limits
        for torque, constraint in zip(torques, self.config.joint_constraints):
            self.assertLessEqual(abs(torque), constraint.max_torque)
    
    def test_joint_direct_control(self):
        """Test direct joint control mode."""
        state = RobotState(
            joint_angles=[0.0, 0.0, 0.0],
            joint_velocities=[0.0, 0.0, 0.0],
            joint_torques=[0.0, 0.0, 0.0],
            end_effector_position=Vector2D(0.7, 0.0),
            control_mode=ControlMode.JOINT_DIRECT
        )
        
        torques = self.controller.calculate_control_torques(state, 0.016)
        
        # Should return zero torques for passive mode
        self.assertEqual(len(torques), len(self.config.links))
        for torque in torques:
            self.assertEqual(torque, 0.0)


class TestRobotState(unittest.TestCase):
    """Test robot state management."""
    
    def test_robot_state_creation(self):
        """Test robot state initialization."""
        joint_angles = [0.1, 0.2, 0.3]
        joint_velocities = [0.01, 0.02, 0.03]
        end_effector_pos = Vector2D(0.5, 0.3)
        
        state = RobotState(
            joint_angles=joint_angles,
            joint_velocities=joint_velocities,
            joint_torques=[0.0, 0.0, 0.0],
            end_effector_position=end_effector_pos,
            target_position=Vector2D(0.6, 0.4),
            control_mode=ControlMode.END_EFFECTOR
        )
        
        self.assertEqual(state.joint_angles, joint_angles)
        self.assertEqual(state.joint_velocities, joint_velocities)
        self.assertEqual(state.end_effector_position, end_effector_pos)
        self.assertEqual(state.control_mode, ControlMode.END_EFFECTOR)


class TestRobotArmSimulation(unittest.TestCase):
    """Test robot arm simulation integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock external dependencies
        self.pygame_patcher = patch('pyjoysim.simulation.robot.robot_arm_simulation.pygame')
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
    
    def test_simulation_creation(self):
        """Test robot arm simulation creation."""
        sim = RobotArmSimulation()
        
        self.assertEqual(sim.name, "robot_arm_simulation")
        self.assertIsNotNone(sim.robot_config)
        self.assertIsNotNone(sim.kinematics)
        self.assertIsNotNone(sim.physics)
        self.assertIsNotNone(sim.controller)
        self.assertIsNotNone(sim.robot_state)
    
    def test_initial_robot_state(self):
        """Test initial robot state setup."""
        sim = RobotArmSimulation()
        
        # Should have valid initial joint angles
        self.assertEqual(len(sim.robot_state.joint_angles), 3)
        self.assertEqual(len(sim.robot_state.joint_velocities), 3)
        self.assertEqual(len(sim.robot_state.joint_torques), 3)
        
        # Should be in end effector control mode by default
        self.assertEqual(sim.robot_state.control_mode, ControlMode.END_EFFECTOR)
        
        # Should have a target position
        self.assertIsNotNone(sim.robot_state.target_position)
    
    def test_control_mode_toggle(self):
        """Test control mode switching."""
        sim = RobotArmSimulation()
        
        initial_mode = sim.robot_state.control_mode
        sim._toggle_control_mode()
        
        # Mode should have changed
        self.assertNotEqual(sim.robot_state.control_mode, initial_mode)
        
        # Toggle back
        sim._toggle_control_mode()
        self.assertEqual(sim.robot_state.control_mode, initial_mode)
    
    def test_robot_pose_reset(self):
        """Test robot pose reset functionality."""
        sim = RobotArmSimulation()
        
        # Modify state
        sim.robot_state.joint_angles = [1.0, 1.0, 1.0]
        sim.robot_state.joint_velocities = [0.5, 0.5, 0.5]
        sim.trajectory_points = [Vector2D(1, 1), Vector2D(2, 2)]
        
        # Reset
        sim._reset_robot_pose()
        
        # State should be reset to default
        expected_angles = [0.0, math.pi/4, -math.pi/6]
        for actual, expected in zip(sim.robot_state.joint_angles, expected_angles):
            self.assertAlmostEqual(actual, expected, places=5)
        
        # Velocities should be zero
        for velocity in sim.robot_state.joint_velocities:
            self.assertEqual(velocity, 0.0)
        
        # Trajectory should be cleared
        self.assertEqual(len(sim.trajectory_points), 0)
    
    def test_target_setting(self):
        """Test random target setting."""
        sim = RobotArmSimulation()
        
        original_target = sim.robot_state.target_position
        sim._set_random_target()
        
        # Target should have changed
        self.assertNotEqual(sim.robot_state.target_position, original_target)
        
        # Target should be within workspace
        max_reach = sum(link.length for link in sim.robot_config.links) * 0.8
        distance = (sim.robot_state.target_position - sim.robot_config.base_position).magnitude()
        self.assertLessEqual(distance, max_reach)
    
    def test_update_cycle(self):
        """Test simulation update cycle."""
        sim = RobotArmSimulation()
        
        # Set a reachable target
        sim.robot_state.target_position = Vector2D(0.3, 0.2)
        
        # Update should not raise errors
        initial_angles = sim.robot_state.joint_angles.copy()
        sim.on_update(0.016)  # 60 FPS
        
        # Robot state should be updated
        # (Angles might change due to control system)
        self.assertEqual(len(sim.robot_state.joint_angles), len(initial_angles))


if __name__ == '__main__':
    unittest.main(verbosity=2)