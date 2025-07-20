"""
Drone sensor simulation.

This module provides realistic sensor simulation for drones including:
- IMU (Inertial Measurement Unit) with accelerometer and gyroscope
- GPS (Global Positioning System) with realistic accuracy and delay
- Barometer for altitude measurement
- Magnetometer for heading
- Camera (basic implementation)
"""

import math
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ...physics.physics3d import Vector3D, Quaternion, PhysicsObject3D
from ...core.logging import get_logger


@dataclass
class SensorNoise:
    """Noise parameters for sensor simulation."""
    bias: float = 0.0       # Constant bias
    white_noise: float = 0.0  # White noise standard deviation
    drift: float = 0.0      # Drift rate
    
    def apply_noise(self, value: float, dt: float) -> float:
        """Apply noise to a sensor value."""
        # Add bias
        noisy_value = value + self.bias
        
        # Add white noise
        if self.white_noise > 0:
            noisy_value += np.random.normal(0, self.white_noise)
        
        # Add drift (accumulating over time)
        if self.drift > 0:
            drift_amount = np.random.normal(0, self.drift * dt)
            self.bias += drift_amount
        
        return noisy_value


class GPSFixType(Enum):
    """GPS fix quality types."""
    NO_FIX = 0
    GPS_FIX = 1
    DGPS_FIX = 2
    RTK_FIX = 3


@dataclass
class GPSData:
    """GPS sensor data."""
    latitude: float = 0.0    # degrees
    longitude: float = 0.0   # degrees
    altitude: float = 0.0    # meters above sea level
    fix_type: GPSFixType = GPSFixType.NO_FIX
    satellites: int = 0
    hdop: float = 99.9       # Horizontal dilution of precision
    speed: float = 0.0       # m/s
    course: float = 0.0      # degrees (heading)
    timestamp: float = 0.0   # Unix timestamp


@dataclass
class IMUData:
    """IMU sensor data."""
    # Accelerometer (m/s²)
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    
    # Gyroscope (rad/s)
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    
    # Magnetometer (µT)
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    
    # Temperature (°C)
    temperature: float = 20.0
    
    timestamp: float = 0.0


@dataclass
class BarometerData:
    """Barometer sensor data."""
    pressure: float = 101325.0  # Pa (sea level)
    altitude: float = 0.0       # meters
    temperature: float = 15.0   # °C
    timestamp: float = 0.0


class IMU:
    """
    Inertial Measurement Unit simulation.
    
    Simulates accelerometer, gyroscope, and magnetometer with realistic
    noise, bias, and drift characteristics.
    """
    
    def __init__(self):
        """Initialize IMU sensor."""
        self.logger = get_logger("imu_sensor")
        
        # Sensor noise parameters (realistic values for consumer-grade IMU)
        self.accel_noise = SensorNoise(bias=0.1, white_noise=0.02, drift=0.001)
        self.gyro_noise = SensorNoise(bias=0.01, white_noise=0.005, drift=0.0001)
        self.mag_noise = SensorNoise(bias=1.0, white_noise=0.5, drift=0.01)
        
        # Sensor scaling and alignment errors
        self.accel_scale_error = 0.01  # 1% scale error
        self.gyro_scale_error = 0.02   # 2% scale error
        
        # Earth's magnetic field (typical values)
        self.magnetic_declination = math.radians(10.0)  # degrees
        self.magnetic_inclination = math.radians(60.0)  # degrees
        self.magnetic_intensity = 50.0  # µT
        
        # Gravity constant
        self.gravity = 9.81  # m/s²
        
        # Last update time for integration
        self.last_update_time = time.time()
        
        self.logger.debug("IMU sensor initialized")
    
    def update(self, physics_object: PhysicsObject3D) -> IMUData:
        """
        Update IMU measurements.
        
        Args:
            physics_object: The drone's physics object
            
        Returns:
            IMUData with current measurements
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Get true acceleration (including gravity)
        true_accel = self._calculate_true_acceleration(physics_object)
        
        # Get true angular velocity
        true_gyro = physics_object.angular_velocity
        
        # Get true magnetic field
        true_mag = self._calculate_magnetic_field(physics_object)
        
        # Apply sensor errors and noise
        imu_data = IMUData()
        
        # Accelerometer
        imu_data.accel_x = self.accel_noise.apply_noise(
            true_accel.x * (1 + self.accel_scale_error), dt)
        imu_data.accel_y = self.accel_noise.apply_noise(
            true_accel.y * (1 + self.accel_scale_error), dt)
        imu_data.accel_z = self.accel_noise.apply_noise(
            true_accel.z * (1 + self.accel_scale_error), dt)
        
        # Gyroscope
        imu_data.gyro_x = self.gyro_noise.apply_noise(
            true_gyro.x * (1 + self.gyro_scale_error), dt)
        imu_data.gyro_y = self.gyro_noise.apply_noise(
            true_gyro.y * (1 + self.gyro_scale_error), dt)
        imu_data.gyro_z = self.gyro_noise.apply_noise(
            true_gyro.z * (1 + self.gyro_scale_error), dt)
        
        # Magnetometer
        imu_data.mag_x = self.mag_noise.apply_noise(true_mag.x, dt)
        imu_data.mag_y = self.mag_noise.apply_noise(true_mag.y, dt)
        imu_data.mag_z = self.mag_noise.apply_noise(true_mag.z, dt)
        
        # Temperature (simplified model)
        imu_data.temperature = 20.0 + np.random.normal(0, 1.0)
        
        imu_data.timestamp = current_time
        
        return imu_data
    
    def _calculate_true_acceleration(self, physics_object: PhysicsObject3D) -> Vector3D:
        """Calculate true acceleration including gravity in body frame."""
        # Get acceleration in world frame (from physics simulation)
        world_accel = Vector3D(0, 0, 0)  # Would get from physics if available
        
        # Add gravity in world frame
        gravity_world = Vector3D(0, -self.gravity, 0)
        total_world_accel = world_accel + gravity_world
        
        # Transform to body frame (IMU measures in body frame)
        # For now, simplified - assume body frame aligned with world frame
        # In full implementation, would use drone's orientation quaternion
        
        return total_world_accel
    
    def _calculate_magnetic_field(self, physics_object: PhysicsObject3D) -> Vector3D:
        """Calculate magnetic field in body frame."""
        # Earth's magnetic field in NED (North-East-Down) frame
        mag_north = self.magnetic_intensity * math.cos(self.magnetic_inclination)
        mag_east = 0.0
        mag_down = self.magnetic_intensity * math.sin(self.magnetic_inclination)
        
        # Apply declination
        mag_x = mag_north * math.cos(self.magnetic_declination)
        mag_y = mag_north * math.sin(self.magnetic_declination)
        mag_z = mag_down
        
        # Transform to body frame (simplified)
        return Vector3D(mag_x, mag_y, mag_z)


class GPS:
    """
    GPS sensor simulation.
    
    Simulates realistic GPS behavior including:
    - Position accuracy degradation
    - Fix acquisition time
    - Satellite visibility
    - Multipath and atmospheric effects
    """
    
    def __init__(self, home_lat: float = 37.7749, home_lon: float = -122.4194):
        """
        Initialize GPS sensor.
        
        Args:
            home_lat: Home latitude in degrees
            home_lon: Home longitude in degrees
        """
        self.logger = get_logger("gps_sensor")
        
        # Home position (reference point)
        self.home_latitude = home_lat
        self.home_longitude = home_lon
        self.home_altitude = 100.0  # meters above sea level
        
        # GPS accuracy parameters
        self.horizontal_accuracy = 3.0  # meters (1 sigma)
        self.vertical_accuracy = 5.0    # meters (1 sigma)
        
        # Fix acquisition
        self.fix_type = GPSFixType.NO_FIX
        self.fix_acquisition_time = 30.0  # seconds to acquire fix
        self.time_since_startup = 0.0
        
        # Satellite simulation
        self.visible_satellites = 0
        self.max_satellites = 12
        
        # Update rate
        self.update_rate = 10.0  # Hz
        self.last_update_time = 0.0
        
        # Position noise
        self.position_noise = SensorNoise(white_noise=1.0)
        
        self.logger.debug("GPS sensor initialized", extra={
            "home_position": (home_lat, home_lon),
            "horizontal_accuracy": self.horizontal_accuracy
        })
    
    def update(self, physics_object: PhysicsObject3D, dt: float) -> Optional[GPSData]:
        """
        Update GPS measurements.
        
        Args:
            physics_object: The drone's physics object
            dt: Time step in seconds
            
        Returns:
            GPSData if update is due, None otherwise
        """
        current_time = time.time()
        
        # Check update rate
        if current_time - self.last_update_time < 1.0 / self.update_rate:
            return None
        
        self.last_update_time = current_time
        self.time_since_startup += dt
        
        # Simulate fix acquisition
        self._update_fix_status()
        
        if self.fix_type == GPSFixType.NO_FIX:
            return GPSData(timestamp=current_time)
        
        # Convert position to GPS coordinates
        position = physics_object.position
        lat, lon, alt = self._position_to_gps(position)
        
        # Apply noise
        lat_noise = self.position_noise.apply_noise(0, dt) / 111320.0  # Convert meters to degrees
        lon_noise = self.position_noise.apply_noise(0, dt) / (111320.0 * math.cos(math.radians(lat)))
        alt_noise = self.position_noise.apply_noise(0, dt)
        
        noisy_lat = lat + lat_noise
        noisy_lon = lon + lon_noise
        noisy_alt = alt + alt_noise
        
        # Calculate speed and course
        velocity = physics_object.linear_velocity
        speed = math.sqrt(velocity.x**2 + velocity.z**2)  # Horizontal speed
        course = math.degrees(math.atan2(velocity.x, velocity.z)) % 360
        
        # Calculate HDOP (simplified)
        hdop = max(1.0, 10.0 / max(1, self.visible_satellites))
        
        return GPSData(
            latitude=noisy_lat,
            longitude=noisy_lon,
            altitude=noisy_alt,
            fix_type=self.fix_type,
            satellites=self.visible_satellites,
            hdop=hdop,
            speed=speed,
            course=course,
            timestamp=current_time
        )
    
    def _update_fix_status(self):
        """Update GPS fix status based on time and conditions."""
        if self.time_since_startup < self.fix_acquisition_time:
            self.fix_type = GPSFixType.NO_FIX
            self.visible_satellites = 0
        elif self.time_since_startup < self.fix_acquisition_time + 10:
            self.fix_type = GPSFixType.GPS_FIX
            self.visible_satellites = min(8, max(4, 
                int(4 + (self.time_since_startup - self.fix_acquisition_time))))
        else:
            self.fix_type = GPSFixType.GPS_FIX
            self.visible_satellites = min(self.max_satellites, 
                max(6, int(8 + np.random.normal(0, 2))))
    
    def _position_to_gps(self, position: Vector3D) -> Tuple[float, float, float]:
        """Convert local position to GPS coordinates."""
        # Simple conversion assuming flat earth (good for small areas)
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        
        lat_offset = position.z / 111320.0
        lon_offset = position.x / (111320.0 * math.cos(math.radians(self.home_latitude)))
        
        latitude = self.home_latitude + lat_offset
        longitude = self.home_longitude + lon_offset
        altitude = self.home_altitude + position.y
        
        return latitude, longitude, altitude


class Barometer:
    """
    Barometric pressure sensor simulation.
    
    Simulates barometric altitude measurement with realistic accuracy
    and environmental effects.
    """
    
    def __init__(self):
        """Initialize barometer sensor."""
        self.logger = get_logger("barometer_sensor")
        
        # Standard atmosphere parameters
        self.sea_level_pressure = 101325.0  # Pa
        self.temperature_lapse_rate = -0.0065  # K/m
        self.sea_level_temperature = 288.15  # K (15°C)
        
        # Sensor characteristics
        self.pressure_noise = SensorNoise(white_noise=50.0)  # Pa
        self.altitude_noise = SensorNoise(white_noise=0.5)   # meters
        
        # Calibration
        self.pressure_offset = 0.0  # Pa
        self.altitude_offset = 0.0  # meters
        
        self.logger.debug("Barometer sensor initialized")
    
    def update(self, physics_object: PhysicsObject3D, dt: float) -> BarometerData:
        """
        Update barometer measurements.
        
        Args:
            physics_object: The drone's physics object
            dt: Time step in seconds
            
        Returns:
            BarometerData with current measurements
        """
        # Get true altitude
        true_altitude = physics_object.position.y
        
        # Calculate pressure using standard atmosphere model
        true_pressure = self._altitude_to_pressure(true_altitude)
        
        # Apply noise and calibration errors
        noisy_pressure = self.pressure_noise.apply_noise(
            true_pressure + self.pressure_offset, dt)
        
        # Calculate altitude from noisy pressure
        calculated_altitude = self._pressure_to_altitude(noisy_pressure)
        noisy_altitude = self.altitude_noise.apply_noise(
            calculated_altitude + self.altitude_offset, dt)
        
        # Temperature at altitude
        temperature = self.sea_level_temperature + self.temperature_lapse_rate * true_altitude
        temperature_celsius = temperature - 273.15
        
        return BarometerData(
            pressure=noisy_pressure,
            altitude=noisy_altitude,
            temperature=temperature_celsius,
            timestamp=time.time()
        )
    
    def _altitude_to_pressure(self, altitude: float) -> float:
        """Convert altitude to atmospheric pressure."""
        # Standard atmosphere model
        temperature = self.sea_level_temperature + self.temperature_lapse_rate * altitude
        pressure = self.sea_level_pressure * (temperature / self.sea_level_temperature) ** 5.25588
        return pressure
    
    def _pressure_to_altitude(self, pressure: float) -> float:
        """Convert atmospheric pressure to altitude."""
        # Inverse of standard atmosphere model
        temperature_ratio = (pressure / self.sea_level_pressure) ** (1/5.25588)
        temperature = temperature_ratio * self.sea_level_temperature
        altitude = (temperature - self.sea_level_temperature) / self.temperature_lapse_rate
        return altitude
    
    def calibrate(self, known_altitude: float, measured_pressure: float):
        """
        Calibrate barometer at a known altitude.
        
        Args:
            known_altitude: True altitude in meters
            measured_pressure: Measured pressure in Pa
        """
        expected_pressure = self._altitude_to_pressure(known_altitude)
        self.pressure_offset = expected_pressure - measured_pressure
        
        self.logger.info("Barometer calibrated", extra={
            "known_altitude": known_altitude,
            "pressure_offset": self.pressure_offset
        })


class DroneSensors:
    """
    Complete drone sensor suite.
    
    Manages all drone sensors and provides unified interface
    for sensor data access.
    """
    
    def __init__(self, home_lat: float = 37.7749, home_lon: float = -122.4194):
        """
        Initialize drone sensors.
        
        Args:
            home_lat: Home latitude in degrees
            home_lon: Home longitude in degrees
        """
        self.logger = get_logger("drone_sensors")
        
        # Initialize sensors
        self.imu = IMU()
        self.gps = GPS(home_lat, home_lon)
        self.barometer = Barometer()
        
        # Sensor data storage
        self.imu_data: Optional[IMUData] = None
        self.gps_data: Optional[GPSData] = None
        self.barometer_data: Optional[BarometerData] = None
        
        # Sensor status
        self.sensors_enabled = True
        
        self.logger.info("Drone sensors initialized")
    
    def update(self, physics_object: PhysicsObject3D, dt: float):
        """
        Update all sensors.
        
        Args:
            physics_object: The drone's physics object
            dt: Time step in seconds
        """
        if not self.sensors_enabled:
            return
        
        # Update IMU (high rate)
        self.imu_data = self.imu.update(physics_object)
        
        # Update GPS (lower rate)
        gps_update = self.gps.update(physics_object, dt)
        if gps_update is not None:
            self.gps_data = gps_update
        
        # Update barometer
        self.barometer_data = self.barometer.update(physics_object, dt)
    
    def get_sensor_data(self) -> dict:
        """Get all current sensor data."""
        return {
            "imu": self.imu_data.__dict__ if self.imu_data else None,
            "gps": self.gps_data.__dict__ if self.gps_data else None,
            "barometer": self.barometer_data.__dict__ if self.barometer_data else None
        }
    
    def enable_sensors(self, enabled: bool = True):
        """Enable or disable all sensors."""
        self.sensors_enabled = enabled
        self.logger.debug(f"Sensors {'enabled' if enabled else 'disabled'}")
    
    def reset(self):
        """Reset all sensors to initial state."""
        self.imu_data = None
        self.gps_data = None
        self.barometer_data = None
        
        # Reset sensor internal states if needed
        self.gps.time_since_startup = 0.0
        self.gps.fix_type = GPSFixType.NO_FIX
        
        self.logger.debug("Sensors reset")