"""
Drone flight controller system.

This module implements various flight control modes for quadrotor drones:
- Manual mode (direct rotor control)
- Stabilized mode (auto-leveling)
- Altitude hold mode
- Position hold mode
- Return to home mode
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...physics.physics3d import Vector3D, Quaternion
from .sensors import DroneSensors, IMUData, GPSData, BarometerData
from ...core.logging import get_logger


class FlightMode(Enum):
    """Flight control modes."""
    MANUAL = "manual"                    # Direct rotor control
    STABILIZED = "stabilized"            # Auto-leveling 
    ALTITUDE_HOLD = "altitude_hold"      # Maintain altitude
    POSITION_HOLD = "position_hold"      # Maintain position (GPS)
    RETURN_TO_HOME = "return_to_home"    # Automatic return
    LAND = "land"                        # Controlled landing
    FAILSAFE = "failsafe"               # Emergency mode


@dataclass
class ControlInput:
    """Control inputs from pilot."""
    # Stick inputs (-1.0 to 1.0)
    roll: float = 0.0       # Right stick left/right
    pitch: float = 0.0      # Right stick up/down  
    yaw: float = 0.0        # Left stick left/right
    throttle: float = 0.0   # Left stick up/down
    
    # Mode switches
    mode_switch: FlightMode = FlightMode.STABILIZED
    arm_switch: bool = False
    emergency_stop: bool = False


@dataclass
class PIDController:
    """PID controller for flight control."""
    kp: float = 1.0    # Proportional gain
    ki: float = 0.0    # Integral gain
    kd: float = 0.0    # Derivative gain
    
    # Internal state
    integral: float = 0.0
    last_error: float = 0.0
    last_time: float = 0.0
    
    # Limits
    output_min: float = -1.0
    output_max: float = 1.0
    integral_max: float = 1.0
    
    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Update PID controller and return control output."""
        if dt <= 0:
            return 0.0
        
        # Calculate error
        error = setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        # Anti-windup
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative term
        if self.last_time > 0:
            d_term = self.kd * (error - self.last_error) / dt
        else:
            d_term = 0.0
        
        # Total output
        output = p_term + i_term + d_term
        
        # Apply limits
        output = max(self.output_min, min(self.output_max, output))
        
        # Store for next iteration
        self.last_error = error
        self.last_time += dt
        
        return output
    
    def reset(self):
        """Reset PID controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = 0.0


@dataclass
class FlightControlParameters:
    """Parameters for flight control system."""
    # Stabilization PID gains
    roll_rate_pid: PIDController = None
    pitch_rate_pid: PIDController = None
    yaw_rate_pid: PIDController = None
    
    roll_angle_pid: PIDController = None
    pitch_angle_pid: PIDController = None
    
    # Altitude control
    altitude_pid: PIDController = None
    climb_rate_pid: PIDController = None
    
    # Position control
    position_x_pid: PIDController = None
    position_y_pid: PIDController = None
    velocity_x_pid: PIDController = None
    velocity_y_pid: PIDController = None
    
    # Control limits
    max_roll_angle: float = math.radians(30)   # 30 degrees
    max_pitch_angle: float = math.radians(30)  # 30 degrees
    max_yaw_rate: float = math.radians(90)     # 90 deg/s
    max_climb_rate: float = 5.0                # 5 m/s
    max_horizontal_velocity: float = 10.0      # 10 m/s
    
    # Hover throttle (0.0 to 1.0)
    hover_throttle: float = 0.5
    
    def __post_init__(self):
        """Initialize PID controllers with default values if not provided."""
        if self.roll_rate_pid is None:
            self.roll_rate_pid = PIDController(kp=0.8, ki=0.1, kd=0.05)
        if self.pitch_rate_pid is None:
            self.pitch_rate_pid = PIDController(kp=0.8, ki=0.1, kd=0.05)
        if self.yaw_rate_pid is None:
            self.yaw_rate_pid = PIDController(kp=1.0, ki=0.2, kd=0.0)
            
        if self.roll_angle_pid is None:
            self.roll_angle_pid = PIDController(kp=6.0, ki=0.0, kd=0.0)
        if self.pitch_angle_pid is None:
            self.pitch_angle_pid = PIDController(kp=6.0, ki=0.0, kd=0.0)
            
        if self.altitude_pid is None:
            self.altitude_pid = PIDController(kp=2.0, ki=0.5, kd=0.0)
        if self.climb_rate_pid is None:
            self.climb_rate_pid = PIDController(kp=0.8, ki=0.2, kd=0.05)
            
        if self.position_x_pid is None:
            self.position_x_pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        if self.position_y_pid is None:
            self.position_y_pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        if self.velocity_x_pid is None:
            self.velocity_x_pid = PIDController(kp=2.0, ki=0.5, kd=0.1)
        if self.velocity_y_pid is None:
            self.velocity_y_pid = PIDController(kp=2.0, ki=0.5, kd=0.1)


class FlightController:
    """
    Drone flight controller with multiple flight modes.
    
    Implements cascaded control loops for stable flight:
    1. Rate control (innermost loop)
    2. Attitude control 
    3. Altitude/Position control (outermost loop)
    """
    
    def __init__(self, params: Optional[FlightControlParameters] = None):
        """
        Initialize flight controller.
        
        Args:
            params: Control parameters (uses defaults if None)
        """
        self.logger = get_logger("flight_controller")
        
        # Control parameters
        self.params = params or FlightControlParameters()
        
        # Current state
        self.current_mode = FlightMode.MANUAL
        self.armed = False
        self.failsafe_active = False
        
        # Setpoints
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw_rate = 0.0
        self.target_altitude = 0.0
        self.target_position = Vector3D()
        
        # Home position (for return-to-home)
        self.home_position = Vector3D()
        self.home_set = False
        
        # State estimation
        self.estimated_roll = 0.0
        self.estimated_pitch = 0.0
        self.estimated_yaw = 0.0
        self.estimated_altitude = 0.0
        self.estimated_position = Vector3D()
        self.estimated_velocity = Vector3D()
        
        # Control outputs
        self.control_outputs = [0.0, 0.0, 0.0, 0.0]  # Motor commands
        
        # Safety limits
        self.min_altitude = 0.5  # meters
        self.max_altitude = 100.0  # meters
        self.max_distance_from_home = 500.0  # meters
        
        self.logger.info("Flight controller initialized", extra={
            "mode": self.current_mode.value
        })
    
    def update(self, 
               control_input: ControlInput,
               sensors: DroneSensors,
               dt: float) -> List[float]:
        """
        Update flight controller and return motor commands.
        
        Args:
            control_input: Pilot inputs
            sensors: Sensor data
            dt: Time step in seconds
            
        Returns:
            List of 4 motor commands (0.0 to 1.0)
        """
        # Update state estimation
        self._update_state_estimation(sensors)
        
        # Handle mode changes and arming
        self._handle_mode_changes(control_input)
        
        # Check safety conditions
        self._check_safety_conditions(control_input)
        
        # Generate control commands based on mode
        if not self.armed or self.failsafe_active:
            self.control_outputs = [0.0, 0.0, 0.0, 0.0]
        else:
            self.control_outputs = self._generate_control_commands(control_input, dt)
        
        return self.control_outputs.copy()
    
    def _update_state_estimation(self, sensors: DroneSensors):
        """Update state estimation from sensor data."""
        # Get sensor data
        imu_data = sensors.imu_data
        gps_data = sensors.gps_data
        baro_data = sensors.barometer_data
        
        # Estimate attitude from IMU (simplified)
        if imu_data:
            # Simple attitude estimation from accelerometer
            # In practice, would use complementary filter or Kalman filter
            accel_roll = math.atan2(imu_data.accel_y, imu_data.accel_z)
            accel_pitch = math.atan2(-imu_data.accel_x, 
                                   math.sqrt(imu_data.accel_y**2 + imu_data.accel_z**2))
            
            # Simple complementary filter (high-pass gyro, low-pass accel)
            alpha = 0.02  # Trust accelerometer 2%
            self.estimated_roll = (1 - alpha) * (self.estimated_roll + imu_data.gyro_x * 0.01) + alpha * accel_roll
            self.estimated_pitch = (1 - alpha) * (self.estimated_pitch + imu_data.gyro_y * 0.01) + alpha * accel_pitch
            
            # Yaw integration (simplified)
            self.estimated_yaw += imu_data.gyro_z * 0.01
        
        # Estimate altitude from barometer
        if baro_data:
            self.estimated_altitude = baro_data.altitude
        
        # Estimate position from GPS
        if gps_data and gps_data.fix_type.value > 0:
            # Convert GPS to local coordinates (simplified)
            # In practice would use proper coordinate transformation
            self.estimated_position = Vector3D(
                gps_data.longitude * 111320.0,  # Rough conversion
                self.estimated_altitude,
                gps_data.latitude * 111320.0
            )
    
    def _handle_mode_changes(self, control_input: ControlInput):
        """Handle flight mode changes and arming."""
        # Handle arming
        if control_input.arm_switch and not self.armed:
            if self._can_arm():
                self.armed = True
                self.logger.info("Drone armed")
        elif not control_input.arm_switch and self.armed:
            self.armed = False
            self.logger.info("Drone disarmed")
        
        # Handle mode changes
        if control_input.mode_switch != self.current_mode:
            old_mode = self.current_mode
            self.current_mode = control_input.mode_switch
            self._on_mode_changed(old_mode, self.current_mode)
        
        # Set home position when first armed
        if self.armed and not self.home_set:
            self.home_position = self.estimated_position.copy()
            self.home_set = True
            self.logger.info("Home position set", extra={
                "position": self.home_position.to_tuple()
            })
    
    def _can_arm(self) -> bool:
        """Check if drone can be armed."""
        # Check if throttle is low
        # Check sensor health
        # Check battery level
        # etc.
        return True  # Simplified
    
    def _on_mode_changed(self, old_mode: FlightMode, new_mode: FlightMode):
        """Handle mode change logic."""
        # Reset PID controllers when changing modes
        if new_mode in [FlightMode.ALTITUDE_HOLD, FlightMode.POSITION_HOLD]:
            self.params.altitude_pid.reset()
            self.params.climb_rate_pid.reset()
        
        if new_mode == FlightMode.POSITION_HOLD:
            self.params.position_x_pid.reset()
            self.params.position_y_pid.reset()
            self.params.velocity_x_pid.reset()
            self.params.velocity_y_pid.reset()
            
            # Set target position to current position
            self.target_position = self.estimated_position.copy()
        
        if new_mode == FlightMode.ALTITUDE_HOLD:
            # Set target altitude to current altitude
            self.target_altitude = self.estimated_altitude
        
        self.logger.info("Flight mode changed", extra={
            "old_mode": old_mode.value,
            "new_mode": new_mode.value
        })
    
    def _check_safety_conditions(self, control_input: ControlInput):
        """Check safety conditions and activate failsafe if needed."""
        self.failsafe_active = False
        
        # Emergency stop
        if control_input.emergency_stop:
            self.failsafe_active = True
            self.logger.warning("Emergency stop activated")
            return
        
        # Altitude limits
        if self.estimated_altitude > self.max_altitude:
            self.failsafe_active = True
            self.logger.warning("Maximum altitude exceeded")
            return
        
        # Distance from home limit
        if self.home_set:
            distance_from_home = (self.estimated_position - self.home_position).magnitude()
            if distance_from_home > self.max_distance_from_home:
                self.failsafe_active = True
                self.logger.warning("Maximum distance from home exceeded")
                return
    
    def _generate_control_commands(self, control_input: ControlInput, dt: float) -> List[float]:
        """Generate motor commands based on flight mode."""
        if self.current_mode == FlightMode.MANUAL:
            return self._manual_control(control_input)
        elif self.current_mode == FlightMode.STABILIZED:
            return self._stabilized_control(control_input, dt)
        elif self.current_mode == FlightMode.ALTITUDE_HOLD:
            return self._altitude_hold_control(control_input, dt)
        elif self.current_mode == FlightMode.POSITION_HOLD:
            return self._position_hold_control(control_input, dt)
        elif self.current_mode == FlightMode.RETURN_TO_HOME:
            return self._return_to_home_control(dt)
        else:
            return [0.0, 0.0, 0.0, 0.0]
    
    def _manual_control(self, control_input: ControlInput) -> List[float]:
        """Direct manual control of rotors."""
        base_throttle = control_input.throttle
        
        # Apply control inputs directly to motor mixing
        roll_command = control_input.roll * 0.5
        pitch_command = control_input.pitch * 0.5
        yaw_command = control_input.yaw * 0.5
        
        # Motor mixing for X-configuration quadrotor
        # Motor 0: front-right, Motor 1: back-left, Motor 2: front-left, Motor 3: back-right
        motor_commands = [
            base_throttle + roll_command + pitch_command - yaw_command,  # Front-right
            base_throttle - roll_command - pitch_command - yaw_command,  # Back-left
            base_throttle - roll_command + pitch_command + yaw_command,  # Front-left
            base_throttle + roll_command - pitch_command + yaw_command   # Back-right
        ]
        
        # Clamp to valid range
        return [max(0.0, min(1.0, cmd)) for cmd in motor_commands]
    
    def _stabilized_control(self, control_input: ControlInput, dt: float) -> List[float]:
        """Stabilized control with auto-leveling."""
        # Convert stick inputs to target angles
        self.target_roll = control_input.roll * self.params.max_roll_angle
        self.target_pitch = control_input.pitch * self.params.max_pitch_angle
        self.target_yaw_rate = control_input.yaw * self.params.max_yaw_rate
        
        # Angle control loops
        roll_rate_cmd = self.params.roll_angle_pid.update(
            self.target_roll, self.estimated_roll, dt)
        pitch_rate_cmd = self.params.pitch_angle_pid.update(
            self.target_pitch, self.estimated_pitch, dt)
        
        # Rate control loops
        roll_output = self.params.roll_rate_pid.update(
            roll_rate_cmd, 0.0, dt)  # Would use actual rate from gyro
        pitch_output = self.params.pitch_rate_pid.update(
            pitch_rate_cmd, 0.0, dt)
        yaw_output = self.params.yaw_rate_pid.update(
            self.target_yaw_rate, 0.0, dt)
        
        # Base throttle
        base_throttle = control_input.throttle
        
        # Motor mixing
        motor_commands = [
            base_throttle + roll_output + pitch_output - yaw_output,  # Front-right
            base_throttle - roll_output - pitch_output - yaw_output,  # Back-left
            base_throttle - roll_output + pitch_output + yaw_output,  # Front-left
            base_throttle + roll_output - pitch_output + yaw_output   # Back-right
        ]
        
        return [max(0.0, min(1.0, cmd)) for cmd in motor_commands]
    
    def _altitude_hold_control(self, control_input: ControlInput, dt: float) -> List[float]:
        """Altitude hold control mode."""
        # Update target altitude from throttle stick
        if abs(control_input.throttle) > 0.1:  # Deadband
            climb_rate = control_input.throttle * self.params.max_climb_rate
            self.target_altitude += climb_rate * dt
            
            # Apply altitude limits
            self.target_altitude = max(self.min_altitude, 
                                     min(self.max_altitude, self.target_altitude))
        
        # Altitude control
        climb_rate_cmd = self.params.altitude_pid.update(
            self.target_altitude, self.estimated_altitude, dt)
        
        # Climb rate control
        throttle_cmd = self.params.climb_rate_pid.update(
            climb_rate_cmd, 0.0, dt)  # Would use actual climb rate
        
        # Add hover throttle
        total_throttle = self.params.hover_throttle + throttle_cmd
        
        # Use stabilized control for attitude
        control_copy = ControlInput(
            roll=control_input.roll,
            pitch=control_input.pitch,
            yaw=control_input.yaw,
            throttle=total_throttle
        )
        
        return self._stabilized_control(control_copy, dt)
    
    def _position_hold_control(self, control_input: ControlInput, dt: float) -> List[float]:
        """Position hold control mode."""
        # Update target position from stick inputs
        if abs(control_input.roll) > 0.1 or abs(control_input.pitch) > 0.1:
            velocity_cmd_x = control_input.roll * self.params.max_horizontal_velocity
            velocity_cmd_y = control_input.pitch * self.params.max_horizontal_velocity
            
            self.target_position.x += velocity_cmd_x * dt
            self.target_position.z += velocity_cmd_y * dt
        
        # Position control
        vel_cmd_x = self.params.position_x_pid.update(
            self.target_position.x, self.estimated_position.x, dt)
        vel_cmd_y = self.params.position_y_pid.update(
            self.target_position.z, self.estimated_position.z, dt)
        
        # Velocity control (converts to attitude commands)
        roll_cmd = self.params.velocity_x_pid.update(
            vel_cmd_x, self.estimated_velocity.x, dt)
        pitch_cmd = self.params.velocity_y_pid.update(
            vel_cmd_y, self.estimated_velocity.z, dt)
        
        # Use altitude hold for vertical control
        control_copy = ControlInput(
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=control_input.yaw,
            throttle=control_input.throttle
        )
        
        return self._altitude_hold_control(control_copy, dt)
    
    def _return_to_home_control(self, dt: float) -> List[float]:
        """Return to home control mode."""
        if not self.home_set:
            # If no home position, just land
            return self._land_control(dt)
        
        # Calculate vector to home
        to_home = self.home_position - self.estimated_position
        distance_to_home = to_home.magnitude()
        
        # If close to home, start landing
        if distance_to_home < 2.0:
            return self._land_control(dt)
        
        # Move towards home
        home_velocity = to_home.normalized() * min(5.0, distance_to_home)
        
        # Convert velocity to attitude commands (simplified)
        roll_cmd = home_velocity.x * 0.2
        pitch_cmd = home_velocity.z * 0.2
        
        # Maintain altitude
        control_input = ControlInput(
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=0.0,
            throttle=0.0  # Altitude hold
        )
        
        return self._altitude_hold_control(control_input, dt)
    
    def _land_control(self, dt: float) -> List[float]:
        """Controlled landing."""
        # Gentle descent
        self.target_altitude = max(0.0, self.target_altitude - 1.0 * dt)
        
        control_input = ControlInput(throttle=0.0)  # Use altitude hold
        return self._altitude_hold_control(control_input, dt)
    
    def get_status(self) -> dict:
        """Get flight controller status."""
        return {
            "mode": self.current_mode.value,
            "armed": self.armed,
            "failsafe": self.failsafe_active,
            "estimated_roll": math.degrees(self.estimated_roll),
            "estimated_pitch": math.degrees(self.estimated_pitch),
            "estimated_yaw": math.degrees(self.estimated_yaw),
            "estimated_altitude": self.estimated_altitude,
            "target_altitude": self.target_altitude,
            "home_set": self.home_set,
            "motor_outputs": self.control_outputs
        }
    
    def set_mode(self, mode: FlightMode):
        """Set flight mode programmatically."""
        if mode != self.current_mode:
            old_mode = self.current_mode
            self.current_mode = mode
            self._on_mode_changed(old_mode, mode)
    
    def arm(self, armed: bool = True):
        """Arm or disarm the drone."""
        if armed and self._can_arm():
            self.armed = True
            self.logger.info("Drone armed programmatically")
        else:
            self.armed = False
            self.logger.info("Drone disarmed programmatically")
    
    def emergency_stop(self):
        """Activate emergency stop."""
        self.failsafe_active = True
        self.armed = False
        self.logger.warning("Emergency stop activated programmatically")
    
    def reset(self):
        """Reset flight controller to initial state."""
        self.current_mode = FlightMode.MANUAL
        self.armed = False
        self.failsafe_active = False
        self.home_set = False
        self.control_outputs = [0.0, 0.0, 0.0, 0.0]
        
        # Reset all PID controllers
        self.params.roll_rate_pid.reset()
        self.params.pitch_rate_pid.reset()
        self.params.yaw_rate_pid.reset()
        self.params.roll_angle_pid.reset()
        self.params.pitch_angle_pid.reset()
        self.params.altitude_pid.reset()
        self.params.climb_rate_pid.reset()
        self.params.position_x_pid.reset()
        self.params.position_y_pid.reset()
        self.params.velocity_x_pid.reset()
        self.params.velocity_y_pid.reset()
        
        self.logger.debug("Flight controller reset")