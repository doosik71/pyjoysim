"""
Main drone simulation class.

This module integrates all drone components into a complete simulation:
- Quadrotor physics
- Flight controller
- Sensor simulation
- Visual effects
- Educational features
"""

import math
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from ..base import BaseSimulation
from ...physics.physics3d import (
    Physics3D, Vector3D, Quaternion, PhysicsObject3D, 
    Shape3D, Shape3DType, Body3DType, PhysicsMaterial3D
)
from ...rendering.renderer3d import RenderObject3D, Material3D, create_render_object_from_physics
from ...rendering.camera3d import Camera3D, CameraMode
from ...input.input_processor import InputState
from ...core.logging import get_logger

from .physics import QuadrotorPhysics, DronePhysicsParameters
from .sensors import DroneSensors
from .flight_controller import FlightController, FlightControlParameters, ControlInput, FlightMode


class DroneSimulation(BaseSimulation):
    """
    Complete drone simulation with realistic flight dynamics.
    
    Features:
    - Realistic quadrotor physics with 4-rotor model
    - Multiple flight modes (manual, stabilized, altitude hold, etc.)
    - Sensor simulation (IMU, GPS, barometer)
    - Visual effects and educational features
    - Joystick/keyboard control
    """
    
    def __init__(self, physics_engine: Physics3D):
        """
        Initialize drone simulation.
        
        Args:
            physics_engine: 3D physics engine instance
        """
        super().__init__(physics_engine)
        
        self.logger = get_logger("drone_simulation")
        
        # Simulation components
        self.drone_physics: Optional[QuadrotorPhysics] = None
        self.flight_controller: Optional[FlightController] = None
        self.sensors: Optional[DroneSensors] = None
        
        # Physics object
        self.drone_body: Optional[PhysicsObject3D] = None
        
        # Rendering
        self.drone_render_object: Optional[RenderObject3D] = None
        self.rotor_render_objects: List[RenderObject3D] = []
        
        # Control state
        self.control_input = ControlInput()
        self.last_update_time = 0.0
        
        # Educational features
        self.show_forces = False
        self.show_trajectory = False
        self.trajectory_points: List[Vector3D] = []
        self.max_trajectory_points = 1000
        
        # Wind simulation
        self.wind_enabled = False
        self.wind_strength = 0.0
        self.wind_direction = 0.0  # degrees
        
        # Performance metrics
        self.flight_time = 0.0
        self.total_distance = 0.0
        self.last_position = Vector3D()
        
        self.logger.info("DroneSimulation initialized")
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the drone simulation.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            True if initialization successful
        """
        try:
            # Get configuration
            config = kwargs.get('config', {})
            
            # Initialize drone physics
            physics_params = DronePhysicsParameters()
            if 'physics' in config:
                physics_config = config['physics']
                physics_params.mass = physics_config.get('mass', physics_params.mass)
                physics_params.arm_length = physics_config.get('arm_length', physics_params.arm_length)
                physics_params.max_thrust_per_rotor = physics_config.get(
                    'max_thrust_per_rotor', physics_params.max_thrust_per_rotor)
            
            self.drone_physics = QuadrotorPhysics(physics_params)
            
            # Initialize flight controller
            control_params = FlightControlParameters()
            if 'control' in config:
                # Could customize control parameters here
                pass
            
            self.flight_controller = FlightController(control_params)
            
            # Initialize sensors
            home_lat = config.get('home_latitude', 37.7749)
            home_lon = config.get('home_longitude', -122.4194)
            self.sensors = DroneSensors(home_lat, home_lon)
            
            # Create drone physics body
            self._create_drone_body()
            
            # Set initial position
            initial_pos = config.get('initial_position', [0, 5, 0])
            self.drone_body.set_position(Vector3D(*initial_pos))
            
            # Initialize rendering
            if kwargs.get('renderer'):
                self._create_drone_visuals(kwargs['renderer'])
            
            self.last_update_time = time.time()
            
            self.logger.info("Drone simulation initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize drone simulation", extra={"error": str(e)})
            return False
    
    def _create_drone_body(self):
        """Create the main drone physics body."""
        # Create box shape for drone body
        body_size = Vector3D(0.3, 0.1, 0.3)  # 30cm x 10cm x 30cm
        shape = Shape3D(Shape3DType.BOX, body_size)
        
        # Create physics material
        material = PhysicsMaterial3D(
            friction=0.5,
            restitution=0.2,
            density=1.0,
            linear_damping=0.1,
            angular_damping=0.1
        )
        
        # Create physics object
        self.drone_body = PhysicsObject3D(
            name="drone_body",
            shape=shape,
            body_type=Body3DType.DYNAMIC,
            mass=self.drone_physics.params.mass,
            material=material
        )
        
        # Add to physics engine
        self.physics_engine.add_object(self.drone_body)
        
        self.logger.debug("Drone physics body created")
    
    def _create_drone_visuals(self, renderer):
        """Create visual representation of the drone."""
        # Main body
        self.drone_render_object = create_render_object_from_physics(
            self.drone_body, renderer, "default")
        
        # Set drone body color (dark gray)
        self.drone_render_object.material.diffuse = (0.3, 0.3, 0.3)
        renderer.add_render_object(self.drone_render_object)
        
        # Create rotor visuals (simplified as small cylinders)
        rotor_positions = [
            Vector3D(0.25, 0.05, 0.25),   # Front-right
            Vector3D(-0.25, 0.05, -0.25), # Back-left
            Vector3D(-0.25, 0.05, 0.25),  # Front-left
            Vector3D(0.25, 0.05, -0.25)   # Back-right
        ]
        
        for i, pos in enumerate(rotor_positions):
            # Create rotor shape
            rotor_shape = Shape3D(Shape3DType.CYLINDER, Vector3D(0.05, 0.02, 0.05))
            
            # Create rotor physics object (just for rendering)
            rotor_obj = PhysicsObject3D(
                name=f"rotor_{i}",
                shape=rotor_shape,
                body_type=Body3DType.KINEMATIC,
                mass=0.0,
                position=pos
            )
            
            # Create render object
            rotor_render = create_render_object_from_physics(rotor_obj, renderer, "red")
            rotor_render.material.diffuse = (0.8, 0.2, 0.2)  # Red rotors
            
            renderer.add_render_object(rotor_render)
            self.rotor_render_objects.append(rotor_render)
        
        self.logger.debug("Drone visual objects created")
    
    def update(self, dt: float, input_state: InputState) -> None:
        """
        Update the drone simulation.
        
        Args:
            dt: Time step in seconds
            input_state: Current input state
        """
        if not self.drone_body or not self.drone_physics or not self.flight_controller:
            return
        
        # Update control inputs from joystick/keyboard
        self._update_control_inputs(input_state)
        
        # Update sensors
        self.sensors.update(self.drone_body, dt)
        
        # Update flight controller
        motor_commands = self.flight_controller.update(self.control_input, self.sensors, dt)
        
        # Update drone physics
        self.drone_physics.set_rotor_commands(motor_commands)
        
        # Apply wind effects
        if self.wind_enabled:
            wind_velocity = Vector3D(
                math.cos(math.radians(self.wind_direction)) * self.wind_strength,
                0,
                math.sin(math.radians(self.wind_direction)) * self.wind_strength
            )
            self.drone_physics.set_wind(wind_velocity, self.wind_strength * 0.1)
        
        # Get forces and torques from physics
        force, torque = self.drone_physics.update(dt, self.drone_body)
        
        # Apply forces to physics body
        self.drone_body.apply_force(force)
        self.drone_body.apply_torque(torque)
        
        # Update rendering
        self._update_visuals()
        
        # Update educational features
        self._update_educational_features(dt)
        
        # Update performance metrics
        self._update_metrics(dt)
    
    def _update_control_inputs(self, input_state: InputState):
        """Update control inputs from input state."""
        # Map joystick axes to control inputs
        if input_state.joystick_available():
            joystick = input_state.get_joystick_state(0)
            if joystick:
                # Standard mapping for drone control
                self.control_input.roll = joystick.get('axis_0', 0.0)      # Right stick X
                self.control_input.pitch = -joystick.get('axis_1', 0.0)   # Right stick Y (inverted)
                self.control_input.yaw = joystick.get('axis_2', 0.0)      # Left stick X
                self.control_input.throttle = -joystick.get('axis_3', 0.0) # Left stick Y (inverted)
                
                # Button mappings
                buttons = joystick.get('buttons', {})
                self.control_input.arm_switch = buttons.get(0, False)     # Button A
                self.control_input.emergency_stop = buttons.get(3, False) # Button Y
                
                # Mode switching with D-pad or buttons
                if buttons.get(4, False):  # Left shoulder
                    self.control_input.mode_switch = FlightMode.MANUAL
                elif buttons.get(5, False):  # Right shoulder
                    self.control_input.mode_switch = FlightMode.STABILIZED
                elif buttons.get(1, False):  # Button B
                    self.control_input.mode_switch = FlightMode.ALTITUDE_HOLD
                elif buttons.get(2, False):  # Button X
                    self.control_input.mode_switch = FlightMode.POSITION_HOLD
        
        # Keyboard fallback
        keys = input_state.get_keys_pressed()
        if keys:
            # WASD for movement
            if 'w' in keys:
                self.control_input.pitch = min(1.0, self.control_input.pitch + 0.02)
            if 's' in keys:
                self.control_input.pitch = max(-1.0, self.control_input.pitch - 0.02)
            if 'a' in keys:
                self.control_input.roll = max(-1.0, self.control_input.roll - 0.02)
            if 'd' in keys:
                self.control_input.roll = min(1.0, self.control_input.roll + 0.02)
            
            # QE for yaw
            if 'q' in keys:
                self.control_input.yaw = max(-1.0, self.control_input.yaw - 0.02)
            if 'e' in keys:
                self.control_input.yaw = min(1.0, self.control_input.yaw + 0.02)
            
            # Space/Shift for throttle
            if 'space' in keys:
                self.control_input.throttle = min(1.0, self.control_input.throttle + 0.02)
            if 'shift' in keys:
                self.control_input.throttle = max(0.0, self.control_input.throttle - 0.02)
            
            # Mode keys
            if '1' in keys:
                self.control_input.mode_switch = FlightMode.MANUAL
            elif '2' in keys:
                self.control_input.mode_switch = FlightMode.STABILIZED
            elif '3' in keys:
                self.control_input.mode_switch = FlightMode.ALTITUDE_HOLD
            elif '4' in keys:
                self.control_input.mode_switch = FlightMode.POSITION_HOLD
            
            # Arm/disarm
            if 'enter' in keys:
                self.control_input.arm_switch = True
            if 'backspace' in keys:
                self.control_input.emergency_stop = True
    
    def _update_visuals(self):
        """Update visual representation."""
        if not self.drone_render_object:
            return
        
        # Update main body position and rotation
        self.drone_render_object.position = self.drone_body.position
        self.drone_render_object.rotation = self.drone_body.rotation
        
        # Update rotor positions and animations
        for i, rotor_render in enumerate(self.rotor_render_objects):
            # Position relative to drone body
            rotor_positions = [
                Vector3D(0.25, 0.05, 0.25),   # Front-right
                Vector3D(-0.25, 0.05, -0.25), # Back-left
                Vector3D(-0.25, 0.05, 0.25),  # Front-left
                Vector3D(0.25, 0.05, -0.25)   # Back-right
            ]
            
            # Transform rotor position to world coordinates
            # (simplified - would use proper transformation)
            world_pos = self.drone_body.position + rotor_positions[i]
            rotor_render.position = world_pos
            rotor_render.rotation = self.drone_body.rotation
            
            # Animate rotor spin (visual effect)
            if self.drone_physics:
                thrust_percentages = self.drone_physics.get_rotor_thrust_percentage()
                if i < len(thrust_percentages):
                    # Color intensity based on thrust
                    intensity = thrust_percentages[i]
                    rotor_render.material.diffuse = (
                        0.8, 
                        0.2 + intensity * 0.6,  # Green component increases with thrust
                        0.2
                    )
    
    def _update_educational_features(self, dt: float):
        """Update educational visualization features."""
        # Update trajectory
        if self.show_trajectory:
            current_pos = self.drone_body.position
            
            # Add point if moved enough
            if (not self.trajectory_points or 
                (current_pos - self.trajectory_points[-1]).magnitude() > 0.5):
                self.trajectory_points.append(current_pos.copy())
                
                # Limit trajectory length
                if len(self.trajectory_points) > self.max_trajectory_points:
                    self.trajectory_points.pop(0)
    
    def _update_metrics(self, dt: float):
        """Update performance metrics."""
        self.flight_time += dt
        
        # Calculate distance traveled
        current_pos = self.drone_body.position
        if self.last_position:
            distance_delta = (current_pos - self.last_position).magnitude()
            self.total_distance += distance_delta
        
        self.last_position = current_pos.copy()
    
    def handle_camera_update(self, camera: Camera3D, dt: float) -> None:
        """
        Update camera for drone simulation.
        
        Args:
            camera: Camera object to update
            dt: Time step in seconds
        """
        if not self.drone_body:
            return
        
        # Set drone as camera target for follow modes
        if camera.mode in [CameraMode.FOLLOW, CameraMode.ORBIT, 
                          CameraMode.FIRST_PERSON, CameraMode.THIRD_PERSON]:
            camera.target_object = self.drone_body
    
    def get_simulation_data(self) -> Dict[str, Any]:
        """Get current simulation data for UI display."""
        data = {
            "simulation_type": "drone",
            "flight_time": self.flight_time,
            "total_distance": self.total_distance
        }
        
        if self.drone_body:
            data.update({
                "position": self.drone_body.position.to_tuple(),
                "velocity": self.drone_body.linear_velocity.to_tuple(),
                "altitude": self.drone_body.position.y
            })
        
        if self.drone_physics:
            data.update({
                "drone_status": self.drone_physics.get_status()
            })
        
        if self.flight_controller:
            data.update({
                "flight_controller": self.flight_controller.get_status()
            })
        
        if self.sensors:
            data.update({
                "sensors": self.sensors.get_sensor_data()
            })
        
        return data
    
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        super().reset()
        
        if self.drone_body:
            # Reset position and velocity
            self.drone_body.set_position(Vector3D(0, 5, 0))
            self.drone_body.set_velocity(Vector3D(), Vector3D())
        
        if self.drone_physics:
            self.drone_physics.reset()
        
        if self.flight_controller:
            self.flight_controller.reset()
        
        if self.sensors:
            self.sensors.reset()
        
        # Reset control input
        self.control_input = ControlInput()
        
        # Reset metrics
        self.flight_time = 0.0
        self.total_distance = 0.0
        self.trajectory_points.clear()
        
        self.logger.debug("Drone simulation reset")
    
    def cleanup(self) -> None:
        """Clean up simulation resources."""
        super().cleanup()
        
        if self.drone_body:
            self.physics_engine.remove_object(self.drone_body.name)
            self.drone_body = None
        
        self.logger.debug("Drone simulation cleaned up")
    
    # Additional control methods
    
    def set_wind(self, strength: float, direction: float = 0.0, enabled: bool = True):
        """
        Set wind conditions.
        
        Args:
            strength: Wind strength in m/s
            direction: Wind direction in degrees
            enabled: Enable/disable wind
        """
        self.wind_enabled = enabled
        self.wind_strength = max(0.0, strength)
        self.wind_direction = direction % 360.0
        
        self.logger.info("Wind conditions set", extra={
            "strength": strength,
            "direction": direction,
            "enabled": enabled
        })
    
    def set_flight_mode(self, mode: FlightMode):
        """Set flight mode programmatically."""
        if self.flight_controller:
            self.flight_controller.set_mode(mode)
            self.logger.info("Flight mode set", extra={"mode": mode.value})
    
    def arm_drone(self, armed: bool = True):
        """Arm or disarm the drone."""
        if self.flight_controller:
            self.flight_controller.arm(armed)
    
    def emergency_stop(self):
        """Activate emergency stop."""
        if self.flight_controller:
            self.flight_controller.emergency_stop()
    
    def toggle_educational_features(self, show_forces: bool = None, show_trajectory: bool = None):
        """Toggle educational visualization features."""
        if show_forces is not None:
            self.show_forces = show_forces
        if show_trajectory is not None:
            self.show_trajectory = show_trajectory
            if not show_trajectory:
                self.trajectory_points.clear()
        
        self.logger.debug("Educational features toggled", extra={
            "show_forces": self.show_forces,
            "show_trajectory": self.show_trajectory
        })
    
    def get_help_text(self) -> List[str]:
        """Get help text for drone controls."""
        return [
            "=== Drone Simulation Controls ===",
            "",
            "Gamepad Controls:",
            "  Right Stick: Roll/Pitch",
            "  Left Stick: Yaw/Throttle",
            "  A Button: Arm/Disarm",
            "  Y Button: Emergency Stop",
            "  Shoulders: Switch Modes",
            "",
            "Keyboard Controls:",
            "  WASD: Roll/Pitch",
            "  QE: Yaw",
            "  Space/Shift: Throttle",
            "  1-4: Flight Modes",
            "  Enter: Arm",
            "  Backspace: Emergency Stop",
            "",
            "Flight Modes:",
            "  1: Manual",
            "  2: Stabilized",
            "  3: Altitude Hold",
            "  4: Position Hold",
            "",
            "Camera: F1-F3 to switch modes"
        ]