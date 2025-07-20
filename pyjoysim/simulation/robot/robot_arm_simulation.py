"""
Robot Arm Simulation Implementation

This module provides a comprehensive 3-DOF robot arm simulation with:
- Realistic joint physics and constraints
- Forward and inverse kinematics
- Interactive target positioning
- Educational visualization features
"""

import math
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pygame

from pyjoysim.simulation.base import BaseSimulation
from pyjoysim.simulation import SimulationConfig
from pyjoysim.physics import Vector2D, create_circle, create_box
from pyjoysim.input import InputEvent, InputEventType
from pyjoysim.rendering import Color


class ControlMode(Enum):
    """Robot arm control modes."""
    JOINT_DIRECT = "joint_direct"  # Direct joint angle control
    END_EFFECTOR = "end_effector"  # End effector position control
    SEQUENCE = "sequence"  # Pre-programmed sequence


@dataclass
class JointConstraints:
    """Joint angle constraints and limits."""
    min_angle: float  # Minimum angle in radians
    max_angle: float  # Maximum angle in radians
    max_velocity: float  # Maximum angular velocity in rad/s
    max_torque: float  # Maximum torque in N*m


@dataclass
class Link:
    """Robot arm link properties."""
    length: float  # Link length in meters
    mass: float  # Link mass in kg
    moment_of_inertia: float  # Moment of inertia in kg*m^2
    
    
@dataclass
class RobotArmConfiguration:
    """Robot arm configuration and parameters."""
    
    # Physical properties
    links: List[Link]
    joint_constraints: List[JointConstraints]
    base_position: Vector2D
    
    # Control parameters
    position_tolerance: float = 0.01  # m
    angle_tolerance: float = 0.017  # ~1 degree in radians
    max_step_size: float = 0.1  # Maximum step for path planning
    
    # Rendering parameters
    link_width: float = 20.0  # Pixels
    joint_radius: float = 15.0  # Pixels
    base_radius: float = 25.0  # Pixels
    
    @classmethod
    def get_default_3dof(cls, base_position: Vector2D = None) -> 'RobotArmConfiguration':
        """Create default 3-DOF robot arm configuration."""
        if base_position is None:
            base_position = Vector2D(0, 0)
            
        # Link specifications (length, mass, moment of inertia)
        links = [
            Link(length=0.3, mass=2.0, moment_of_inertia=0.18),  # Shoulder link
            Link(length=0.25, mass=1.5, moment_of_inertia=0.078),  # Upper arm
            Link(length=0.15, mass=1.0, moment_of_inertia=0.019)   # Forearm
        ]
        
        # Joint constraints (min, max angles, max velocity, max torque)
        joint_constraints = [
            JointConstraints(-math.pi, math.pi, math.pi, 50.0),      # Base rotation
            JointConstraints(-math.pi/2, math.pi/2, math.pi, 30.0),  # Shoulder joint
            JointConstraints(-math.pi, math.pi/2, math.pi, 20.0)     # Elbow joint
        ]
        
        return cls(
            links=links,
            joint_constraints=joint_constraints,
            base_position=base_position
        )


@dataclass
class RobotState:
    """Current state of the robot arm."""
    joint_angles: List[float]  # Current joint angles in radians
    joint_velocities: List[float]  # Current joint velocities in rad/s
    joint_torques: List[float]  # Applied joint torques in N*m
    end_effector_position: Vector2D  # Current end effector position
    target_position: Optional[Vector2D] = None  # Target position for IK
    control_mode: ControlMode = ControlMode.END_EFFECTOR


class Kinematics:
    """Forward and inverse kinematics calculations."""
    
    def __init__(self, config: RobotArmConfiguration):
        self.config = config
        
    def forward_kinematics(self, joint_angles: List[float]) -> List[Vector2D]:
        """
        Calculate forward kinematics to get all joint positions.
        
        Returns:
            List of positions for base, joint1, joint2, end_effector
        """
        positions = [self.config.base_position]
        
        current_pos = self.config.base_position
        current_angle = 0.0
        
        for i, (angle, link) in enumerate(zip(joint_angles, self.config.links)):
            current_angle += angle
            
            # Calculate next joint position
            dx = link.length * math.cos(current_angle)
            dy = link.length * math.sin(current_angle)
            
            current_pos = Vector2D(current_pos.x + dx, current_pos.y + dy)
            positions.append(current_pos)
            
        return positions
    
    def get_end_effector_position(self, joint_angles: List[float]) -> Vector2D:
        """Get the end effector position from joint angles."""
        positions = self.forward_kinematics(joint_angles)
        return positions[-1]
    
    def inverse_kinematics(self, target_position: Vector2D, 
                          current_angles: Optional[List[float]] = None) -> Optional[List[float]]:
        """
        Calculate inverse kinematics using numerical method (Jacobian-based).
        
        Args:
            target_position: Desired end effector position
            current_angles: Current joint angles as starting point
            
        Returns:
            Joint angles to reach target, or None if unreachable
        """
        if current_angles is None:
            current_angles = [0.0] * len(self.config.links)
        
        angles = current_angles.copy()
        max_iterations = 100
        step_size = 0.1
        
        for iteration in range(max_iterations):
            current_pos = self.get_end_effector_position(angles)
            
            # Check if we're close enough
            error = target_position - current_pos
            if error.magnitude() < self.config.position_tolerance:
                return angles
            
            # Calculate Jacobian matrix
            jacobian = self._calculate_jacobian(angles)
            
            # Calculate angle adjustments using pseudo-inverse
            delta_angles = self._solve_jacobian(jacobian, error, step_size)
            
            # Apply constraints and update angles
            for i, (delta, constraint) in enumerate(zip(delta_angles, self.config.joint_constraints)):
                angles[i] += delta
                angles[i] = max(constraint.min_angle, min(constraint.max_angle, angles[i]))
        
        # Check if final position is acceptable
        final_pos = self.get_end_effector_position(angles)
        final_error = target_position - final_pos
        
        if final_error.magnitude() < self.config.position_tolerance * 3:  # Relaxed tolerance
            return angles
        
        return None  # Failed to converge
    
    def _calculate_jacobian(self, joint_angles: List[float]) -> List[List[float]]:
        """Calculate the Jacobian matrix for the current configuration."""
        epsilon = 0.001  # Small angle for numerical differentiation
        jacobian = []
        
        current_pos = self.get_end_effector_position(joint_angles)
        
        for i in range(len(joint_angles)):
            # Perturb joint i slightly
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += epsilon
            
            perturbed_pos = self.get_end_effector_position(perturbed_angles)
            
            # Calculate partial derivatives
            dx_dtheta = (perturbed_pos.x - current_pos.x) / epsilon
            dy_dtheta = (perturbed_pos.y - current_pos.y) / epsilon
            
            jacobian.append([dx_dtheta, dy_dtheta])
        
        return jacobian
    
    def _solve_jacobian(self, jacobian: List[List[float]], error: Vector2D, 
                       step_size: float) -> List[float]:
        """Solve the Jacobian equation using simple transpose method."""
        # For small 2D problems, use Jacobian transpose method
        delta_angles = []
        
        for j_row in jacobian:
            # Dot product of Jacobian row with error vector
            delta = step_size * (j_row[0] * error.x + j_row[1] * error.y)
            delta_angles.append(delta)
        
        return delta_angles
    
    def get_workspace_boundary(self, num_points: int = 100) -> List[Vector2D]:
        """Calculate the workspace boundary of the robot arm."""
        boundary_points = []
        
        # Maximum reach
        max_reach = sum(link.length for link in self.config.links)
        
        # Minimum reach (when all joints are folded)
        min_reach = abs(self.config.links[0].length - 
                       sum(link.length for link in self.config.links[1:]))
        
        # Create boundary points
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # Outer boundary
            outer_point = Vector2D(
                self.config.base_position.x + max_reach * math.cos(angle),
                self.config.base_position.y + max_reach * math.sin(angle)
            )
            boundary_points.append(outer_point)
        
        return boundary_points


class RobotArmPhysics:
    """Physics simulation for robot arm dynamics."""
    
    def __init__(self, config: RobotArmConfiguration):
        self.config = config
        self.gravity = Vector2D(0, -9.81)  # Gravity acceleration
        
    def calculate_dynamics(self, state: RobotState, control_torques: List[float], 
                          dt: float) -> Tuple[List[float], List[float]]:
        """
        Calculate joint accelerations using simplified dynamics.
        
        Returns:
            Tuple of (new_velocities, new_accelerations)
        """
        # Simplified dynamics without full Newton-Euler
        # For educational purposes, we use basic physics
        
        accelerations = []
        new_velocities = []
        
        for i, (link, constraint, velocity, torque) in enumerate(
            zip(self.config.links, self.config.joint_constraints, 
                state.joint_velocities, control_torques)):
            
            # Simple torque-based acceleration
            # τ = I * α (ignoring coupling terms for simplicity)
            acceleration = torque / link.moment_of_inertia
            
            # Add gravity compensation (simplified)
            gravity_torque = self._calculate_gravity_torque(state.joint_angles, i)
            acceleration += gravity_torque / link.moment_of_inertia
            
            # Apply velocity damping
            damping = 0.1  # Damping coefficient
            acceleration -= damping * velocity
            
            # Limit acceleration
            max_accel = 10.0  # rad/s^2
            acceleration = max(-max_accel, min(max_accel, acceleration))
            
            accelerations.append(acceleration)
            
            # Update velocity
            new_velocity = velocity + acceleration * dt
            
            # Apply velocity limits
            new_velocity = max(-constraint.max_velocity, 
                             min(constraint.max_velocity, new_velocity))
            
            new_velocities.append(new_velocity)
        
        return new_velocities, accelerations
    
    def _calculate_gravity_torque(self, joint_angles: List[float], joint_index: int) -> float:
        """Calculate gravity compensation torque for a joint."""
        # Simplified gravity compensation
        # In reality, this would consider the full dynamics
        
        if joint_index == 0:  # Base joint (rotation only)
            return 0.0
        
        # Estimate center of mass effect
        link = self.config.links[joint_index]
        angle_sum = sum(joint_angles[:joint_index + 1])
        
        # Simple approximation of gravity effect
        gravity_torque = link.mass * self.gravity.magnitude() * (link.length / 2) * math.cos(angle_sum)
        
        return gravity_torque


class RobotArmController:
    """Robot arm control system."""
    
    def __init__(self, config: RobotArmConfiguration):
        self.config = config
        self.kinematics = Kinematics(config)
        
        # PID controller parameters
        self.position_kp = 50.0
        self.position_ki = 0.1
        self.position_kd = 5.0
        
        self.angle_kp = 20.0
        self.angle_ki = 0.05
        self.angle_kd = 2.0
        
        # Error integration for PID
        self.position_error_integral = Vector2D(0, 0)
        self.angle_error_integrals = [0.0] * len(config.links)
        self.previous_position_error = Vector2D(0, 0)
        self.previous_angle_errors = [0.0] * len(config.links)
    
    def calculate_control_torques(self, state: RobotState, dt: float) -> List[float]:
        """Calculate control torques based on current state and control mode."""
        
        if state.control_mode == ControlMode.END_EFFECTOR:
            return self._end_effector_control(state, dt)
        elif state.control_mode == ControlMode.JOINT_DIRECT:
            return self._joint_direct_control(state, dt)
        else:
            return [0.0] * len(self.config.links)
    
    def _end_effector_control(self, state: RobotState, dt: float) -> List[float]:
        """Position control of end effector using inverse kinematics."""
        if state.target_position is None:
            return [0.0] * len(self.config.links)
        
        # Calculate desired joint angles using IK
        desired_angles = self.kinematics.inverse_kinematics(
            state.target_position, state.joint_angles)
        
        if desired_angles is None:
            # Target unreachable, apply minimal torques
            return [0.0] * len(self.config.links)
        
        # Use joint-level PID control to reach desired angles
        torques = []
        for i, (current_angle, desired_angle, constraint) in enumerate(
            zip(state.joint_angles, desired_angles, self.config.joint_constraints)):
            
            # Calculate angle error
            angle_error = desired_angle - current_angle
            
            # Wrap angle error to [-π, π]
            while angle_error > math.pi:
                angle_error -= 2 * math.pi
            while angle_error < -math.pi:
                angle_error += 2 * math.pi
            
            # PID calculation
            self.angle_error_integrals[i] += angle_error * dt
            derivative = (angle_error - self.previous_angle_errors[i]) / dt
            
            torque = (self.angle_kp * angle_error + 
                     self.angle_ki * self.angle_error_integrals[i] +
                     self.angle_kd * derivative)
            
            # Apply torque limits
            torque = max(-constraint.max_torque, min(constraint.max_torque, torque))
            torques.append(torque)
            
            self.previous_angle_errors[i] = angle_error
        
        return torques
    
    def _joint_direct_control(self, state: RobotState, dt: float) -> List[float]:
        """Direct joint angle control (for manual operation)."""
        # This would be used when user directly controls joint angles
        # For now, return zero torques (passive mode)
        return [0.0] * len(self.config.links)


class RobotArmSimulation(BaseSimulation):
    """
    3-DOF Robot Arm Simulation
    
    Features:
    - Realistic joint physics and constraints
    - Forward and inverse kinematics
    - Multiple control modes
    - Interactive target positioning
    - Educational visualization
    """
    
    def __init__(self, name: str = "robot_arm_simulation", config: Optional[SimulationConfig] = None):
        super().__init__(name, config)
        
        # Robot configuration
        base_pos = Vector2D(0, -100)  # Position robot base below center
        self.robot_config = RobotArmConfiguration.get_default_3dof(base_pos)
        
        # Robot components
        self.kinematics = Kinematics(self.robot_config)
        self.physics = RobotArmPhysics(self.robot_config)
        self.controller = RobotArmController(self.robot_config)
        
        # Robot state
        initial_angles = [0.0, math.pi/4, -math.pi/6]  # Start in interesting pose
        self.robot_state = RobotState(
            joint_angles=initial_angles,
            joint_velocities=[0.0] * len(self.robot_config.links),
            joint_torques=[0.0] * len(self.robot_config.links),
            end_effector_position=self.kinematics.get_end_effector_position(initial_angles),
            target_position=Vector2D(0.4, 0.2),  # Initial target
            control_mode=ControlMode.END_EFFECTOR
        )
        
        # Input handling
        self.target_input = Vector2D(0, 0)  # Joystick input for target positioning
        self.joint_inputs = [0.0] * len(self.robot_config.links)  # Direct joint inputs
        self.mode_switch_pressed = False
        self.target_set_pressed = False
        
        # Visualization
        self.show_workspace = True
        self.show_trajectory = True
        self.trajectory_points = []  # Store end effector trajectory
        self.workspace_boundary = self.kinematics.get_workspace_boundary()
        
        # Colors
        self.colors = {
            'base': Color(80, 80, 80),
            'link1': Color(200, 100, 100),
            'link2': Color(100, 200, 100),
            'link3': Color(100, 100, 200),
            'joint': Color(60, 60, 60),
            'target': Color(255, 255, 0),
            'end_effector': Color(255, 0, 255),
            'workspace': Color(150, 150, 150, 100),
            'trajectory': Color(0, 255, 255, 150)
        }
        
    def on_initialize(self) -> None:
        """Initialize robot arm physics bodies."""
        # Create physics bodies for robot links
        self.link_bodies = []
        self.joint_bodies = []
        
        # For visualization purposes, we'll create simple physics bodies
        # In a full implementation, these would be properly connected with joints
        
        positions = self.kinematics.forward_kinematics(self.robot_state.joint_angles)
        
        # Create base
        base_body, _ = create_circle(
            self.physics_world.engine,
            positions[0],
            self.robot_config.base_radius / 100.0,  # Convert pixels to meters
            density=10.0,
            is_static=True
        )
        self.joint_bodies.append(base_body)
        
        # Create links as bodies (for visualization)
        for i, (link, pos) in enumerate(zip(self.robot_config.links, positions[1:])):
            link_body, _ = create_box(
                self.physics_world.engine,
                pos,
                link.length,
                0.05,  # Link thickness
                density=link.mass / (link.length * 0.05)
            )
            self.link_bodies.append(link_body)
    
    def on_update(self, dt: float) -> None:
        """Update robot arm simulation."""
        # Process input for target positioning
        self._update_target_from_input(dt)
        
        # Calculate control torques
        control_torques = self.controller.calculate_control_torques(self.robot_state, dt)
        
        # Update physics
        new_velocities, accelerations = self.physics.calculate_dynamics(
            self.robot_state, control_torques, dt)
        
        # Update robot state
        self.robot_state.joint_velocities = new_velocities
        self.robot_state.joint_torques = control_torques
        
        # Update joint angles
        for i in range(len(self.robot_state.joint_angles)):
            self.robot_state.joint_angles[i] += self.robot_state.joint_velocities[i] * dt
            
            # Apply joint limits
            constraint = self.robot_config.joint_constraints[i]
            self.robot_state.joint_angles[i] = max(
                constraint.min_angle, 
                min(constraint.max_angle, self.robot_state.joint_angles[i])
            )
        
        # Update end effector position
        self.robot_state.end_effector_position = self.kinematics.get_end_effector_position(
            self.robot_state.joint_angles)
        
        # Update trajectory
        if len(self.trajectory_points) == 0 or \
           (self.robot_state.end_effector_position - self.trajectory_points[-1]).magnitude() > 0.01:
            self.trajectory_points.append(self.robot_state.end_effector_position)
            
        # Limit trajectory length
        if len(self.trajectory_points) > 200:
            self.trajectory_points.pop(0)
    
    def on_render(self) -> None:
        """Render robot arm visualization."""
        if not self.render_engine:
            return
        
        # Get current joint positions
        positions = self.kinematics.forward_kinematics(self.robot_state.joint_angles)
        
        # Convert to screen coordinates
        def to_screen(world_pos: Vector2D) -> Vector2D:
            return Vector2D(
                world_pos.x * 200 + self.config.window_width // 2,
                -world_pos.y * 200 + self.config.window_height // 2
            )
        
        # Draw workspace boundary
        if self.show_workspace:
            workspace_screen = [to_screen(p) for p in self.workspace_boundary]
            if len(workspace_screen) > 2:
                pygame.draw.polygon(
                    self.render_engine.screen,
                    self.colors['workspace'].to_tuple(),
                    [(p.x, p.y) for p in workspace_screen]
                )
        
        # Draw trajectory
        if self.show_trajectory and len(self.trajectory_points) > 1:
            trajectory_screen = [to_screen(p) for p in self.trajectory_points]
            if len(trajectory_screen) > 1:
                pygame.draw.lines(
                    self.render_engine.screen,
                    self.colors['trajectory'].to_tuple(),
                    False,
                    [(p.x, p.y) for p in trajectory_screen],
                    3
                )
        
        # Draw robot links
        link_colors = [self.colors['link1'], self.colors['link2'], self.colors['link3']]
        
        for i in range(len(positions) - 1):
            start_screen = to_screen(positions[i])
            end_screen = to_screen(positions[i + 1])
            
            # Draw link
            pygame.draw.line(
                self.render_engine.screen,
                link_colors[i].to_tuple(),
                (start_screen.x, start_screen.y),
                (end_screen.x, end_screen.y),
                int(self.robot_config.link_width)
            )
        
        # Draw joints
        for i, pos in enumerate(positions):
            screen_pos = to_screen(pos)
            radius = self.robot_config.joint_radius if i > 0 else self.robot_config.base_radius
            color = self.colors['joint'] if i > 0 else self.colors['base']
            
            pygame.draw.circle(
                self.render_engine.screen,
                color.to_tuple(),
                (int(screen_pos.x), int(screen_pos.y)),
                int(radius)
            )
        
        # Draw target position
        if self.robot_state.target_position:
            target_screen = to_screen(self.robot_state.target_position)
            pygame.draw.circle(
                self.render_engine.screen,
                self.colors['target'].to_tuple(),
                (int(target_screen.x), int(target_screen.y)),
                10
            )
            # Draw crosshair
            pygame.draw.line(
                self.render_engine.screen,
                self.colors['target'].to_tuple(),
                (target_screen.x - 15, target_screen.y),
                (target_screen.x + 15, target_screen.y),
                2
            )
            pygame.draw.line(
                self.render_engine.screen,
                self.colors['target'].to_tuple(),
                (target_screen.x, target_screen.y - 15),
                (target_screen.x, target_screen.y + 15),
                2
            )
        
        # Draw end effector
        ee_screen = to_screen(self.robot_state.end_effector_position)
        pygame.draw.circle(
            self.render_engine.screen,
            self.colors['end_effector'].to_tuple(),
            (int(ee_screen.x), int(ee_screen.y)),
            8
        )
        
        # Draw UI information
        self._draw_ui_info()
    
    def _draw_ui_info(self) -> None:
        """Draw robot arm status information."""
        if not self.render_engine:
            return
        
        font = pygame.font.Font(None, 24)
        info_y = 10
        
        # Control mode
        mode_text = f"Mode: {self.robot_state.control_mode.value}"
        mode_surface = font.render(mode_text, True, (255, 255, 255))
        self.render_engine.screen.blit(mode_surface, (10, info_y))
        info_y += 30
        
        # Joint angles
        for i, angle in enumerate(self.robot_state.joint_angles):
            angle_deg = math.degrees(angle)
            angle_text = f"Joint {i+1}: {angle_deg:.1f}°"
            angle_surface = font.render(angle_text, True, (255, 255, 255))
            self.render_engine.screen.blit(angle_surface, (10, info_y))
            info_y += 25
        
        # End effector position
        ee_pos = self.robot_state.end_effector_position
        pos_text = f"End Effector: ({ee_pos.x:.3f}, {ee_pos.y:.3f}) m"
        pos_surface = font.render(pos_text, True, (255, 255, 255))
        self.render_engine.screen.blit(pos_surface, (10, info_y))
        info_y += 25
        
        # Target position
        if self.robot_state.target_position:
            target_pos = self.robot_state.target_position
            target_text = f"Target: ({target_pos.x:.3f}, {target_pos.y:.3f}) m"
            target_surface = font.render(target_text, True, (255, 255, 255))
            self.render_engine.screen.blit(target_surface, (10, info_y))
            info_y += 25
            
            # Distance to target
            distance = (self.robot_state.end_effector_position - target_pos).magnitude()
            dist_text = f"Distance: {distance:.3f} m"
            dist_surface = font.render(dist_text, True, (255, 255, 255))
            self.render_engine.screen.blit(dist_surface, (10, info_y))
            info_y += 25
        
        # Controls help
        help_y = self.config.window_height - 150
        help_texts = [
            "Controls:",
            "Left Stick: Move target X-Y",
            "Right Stick: Fine adjust target",
            "A Button: Set new target",
            "B Button: Toggle control mode",
            "X Button: Toggle workspace view",
            "Y Button: Reset robot pose"
        ]
        
        for text in help_texts:
            help_surface = font.render(text, True, (200, 200, 200))
            self.render_engine.screen.blit(help_surface, (10, help_y))
            help_y += 20
    
    def on_input_event(self, event: InputEvent) -> None:
        """Handle input events for robot arm control."""
        if event.event_type == InputEventType.AXIS_CHANGE:
            if event.axis_id == 0:  # Left stick X - Target X movement
                self.target_input.x = event.axis_value
            elif event.axis_id == 1:  # Left stick Y - Target Y movement
                self.target_input.y = -event.axis_value  # Invert Y for natural movement
            elif event.axis_id == 3:  # Right stick X - Fine X adjustment
                self.target_input.x += event.axis_value * 0.1
            elif event.axis_id == 4:  # Right stick Y - Fine Y adjustment
                self.target_input.y += -event.axis_value * 0.1  # Invert Y
                
        elif event.event_type == InputEventType.BUTTON_PRESS:
            if event.button_id == 0:  # A button - Set new target
                self.target_set_pressed = True
            elif event.button_id == 1:  # B button - Toggle control mode
                self._toggle_control_mode()
            elif event.button_id == 2:  # X button - Toggle workspace view
                self.show_workspace = not self.show_workspace
            elif event.button_id == 3:  # Y button - Reset robot pose
                self._reset_robot_pose()
        
        elif event.event_type == InputEventType.KEY_PRESS:
            if event.key == pygame.K_SPACE:
                self.target_set_pressed = True
            elif event.key == pygame.K_TAB:
                self._toggle_control_mode()
            elif event.key == pygame.K_w:
                self.show_workspace = not self.show_workspace
            elif event.key == pygame.K_r:
                self._reset_robot_pose()
    
    def _update_target_from_input(self, dt: float) -> None:
        """Update target position based on input."""
        if self.robot_state.control_mode != ControlMode.END_EFFECTOR:
            return
        
        # Apply target input
        if abs(self.target_input.x) > 0.1 or abs(self.target_input.y) > 0.1:
            if self.robot_state.target_position is None:
                self.robot_state.target_position = Vector2D(0.3, 0.2)
            
            # Move target based on input
            move_speed = 0.5  # m/s
            delta_x = self.target_input.x * move_speed * dt
            delta_y = self.target_input.y * move_speed * dt
            
            new_target = Vector2D(
                self.robot_state.target_position.x + delta_x,
                self.robot_state.target_position.y + delta_y
            )
            
            # Keep target within reasonable bounds
            max_reach = sum(link.length for link in self.robot_config.links) * 0.9
            distance_from_base = (new_target - self.robot_config.base_position).magnitude()
            
            if distance_from_base <= max_reach:
                self.robot_state.target_position = new_target
        
        # Set new random target if requested
        if self.target_set_pressed:
            self._set_random_target()
            self.target_set_pressed = False
    
    def _toggle_control_mode(self) -> None:
        """Toggle between control modes."""
        if self.robot_state.control_mode == ControlMode.END_EFFECTOR:
            self.robot_state.control_mode = ControlMode.JOINT_DIRECT
        else:
            self.robot_state.control_mode = ControlMode.END_EFFECTOR
            # Set a reasonable target when switching to end effector mode
            if self.robot_state.target_position is None:
                self._set_random_target()
    
    def _set_random_target(self) -> None:
        """Set a random reachable target position."""
        import random
        
        max_reach = sum(link.length for link in self.robot_config.links) * 0.8
        min_reach = 0.1
        
        # Generate random position within workspace
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(min_reach, max_reach)
        
        self.robot_state.target_position = Vector2D(
            self.robot_config.base_position.x + distance * math.cos(angle),
            self.robot_config.base_position.y + distance * math.sin(angle)
        )
    
    def _reset_robot_pose(self) -> None:
        """Reset robot to default pose."""
        self.robot_state.joint_angles = [0.0, math.pi/4, -math.pi/6]
        self.robot_state.joint_velocities = [0.0] * len(self.robot_config.links)
        self.robot_state.joint_torques = [0.0] * len(self.robot_config.links)
        self.robot_state.end_effector_position = self.kinematics.get_end_effector_position(
            self.robot_state.joint_angles)
        
        # Clear trajectory
        self.trajectory_points = []
        
        # Reset PID controllers
        self.controller.position_error_integral = Vector2D(0, 0)
        self.controller.angle_error_integrals = [0.0] * len(self.robot_config.links)
        self.controller.previous_position_error = Vector2D(0, 0)
        self.controller.previous_angle_errors = [0.0] * len(self.robot_config.links)


# Simulation metadata for registration
ROBOT_ARM_SIMULATION_METADATA = {
    'name': 'robot_arm_simulation',
    'display_name': '로봇 팔 시뮬레이션',
    'description': '3-DOF 로봇 팔 제어 및 교육용 시뮬레이션',
    'category': 'robot',
    'difficulty': 'intermediate',
    'requires_joystick': True,
    'educational_topics': [
        '로봇 공학',
        '순운동학/역운동학',
        '제어 시스템',
        '궤적 계획'
    ]
}