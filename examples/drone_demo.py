#!/usr/bin/env python3
"""
Drone Simulation Demo

This demo showcases the comprehensive drone simulation system with:
- Realistic quadrotor physics
- Multiple flight modes (Manual, Stabilized, Altitude Hold, Position Hold)
- Sensor simulation (IMU, GPS, Barometer)
- Flight controller with PID loops
- Visual effects and educational features
- Joystick/keyboard control

Controls:
- Gamepad: Right stick (Roll/Pitch), Left stick (Yaw/Throttle), A (Arm), Y (Emergency)
- Keyboard: WASD (Roll/Pitch), QE (Yaw), Space/Shift (Throttle), 1-4 (Flight modes)
- Camera: F1-F3 to switch camera modes
"""

import sys
import time
import math
from pathlib import Path

# Add pyjoysim to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame
from pyjoysim.physics.physics3d import Physics3D, Vector3D, PhysicsObject3D, Shape3D, Shape3DType, Body3DType
from pyjoysim.rendering.renderer3d import Renderer3D, Light3D, LightType
from pyjoysim.rendering.camera3d import Camera3D, CameraMode
from pyjoysim.input.joystick_manager import JoystickManager
from pyjoysim.input.input_processor import InputProcessor, InputState
from pyjoysim.simulation.drone import DroneSimulation, FlightMode
from pyjoysim.core.logging import setup_logging, get_logger


class DroneDemo:
    """
    Comprehensive drone simulation demonstration.
    """
    
    def __init__(self):
        """Initialize the drone demo."""
        # Setup logging
        setup_logging(level="INFO")
        self.logger = get_logger("drone_demo")
        
        # Initialize pygame
        pygame.init()
        
        # Window settings
        self.window_width = 1200
        self.window_height = 800
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("PyJoySim - Drone Simulation Demo")
        
        # Initialize systems
        self.physics_engine = Physics3D()
        self.renderer = Renderer3D(self.window_width, self.window_height)
        self.joystick_manager = JoystickManager()
        self.input_processor = InputProcessor(self.joystick_manager)
        
        # Initialize simulation
        self.drone_simulation: DroneSimulation = None
        
        # Camera system
        self.camera = Camera3D()
        self.camera.position = Vector3D(-10, 10, -10)
        self.camera.target = Vector3D(0, 5, 0)
        self.camera.mode = CameraMode.ORBIT
        
        # Demo state
        self.running = False
        self.paused = False
        self.clock = pygame.time.Clock()
        self.fps_target = 60
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # UI state
        self.show_help = True
        self.show_info = True
        self.show_sensors = False
        
        self.logger.info("Drone demo initialized")
    
    def initialize(self) -> bool:
        """
        Initialize all systems.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize physics engine
            if not self.physics_engine.initialize():
                self.logger.error("Failed to initialize physics engine")
                return False
            
            # Initialize renderer
            if not self.renderer.initialize():
                self.logger.error("Failed to initialize renderer")
                return False
            
            # Setup lighting
            self._setup_lighting()
            
            # Create environment
            self._create_environment()
            
            # Initialize drone simulation
            self.drone_simulation = DroneSimulation(self.physics_engine)
            
            # Configuration for drone
            drone_config = {
                'physics': {
                    'mass': 1.5,
                    'arm_length': 0.25,
                    'max_thrust_per_rotor': 6.0
                },
                'control': {
                    'hover_throttle': 0.5,
                    'max_roll_angle': 30,
                    'max_pitch_angle': 30
                },
                'initial_position': [0, 5, 0],
                'home_latitude': 37.7749,
                'home_longitude': -122.4194
            }
            
            if not self.drone_simulation.initialize(config=drone_config, renderer=self.renderer):
                self.logger.error("Failed to initialize drone simulation")
                return False
            
            # Initialize joystick system
            self.joystick_manager.initialize()
            
            self.logger.info("All systems initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize demo: {e}")
            return False
    
    def _setup_lighting(self):
        """Setup scene lighting."""
        # Main directional light (sun)
        sun_light = Light3D(
            type=LightType.DIRECTIONAL,
            position=Vector3D(100, 100, 50),
            direction=Vector3D(-1, -1, -0.5).normalized(),
            ambient=(0.3, 0.3, 0.4),
            diffuse=(0.8, 0.8, 0.7),
            specular=(1.0, 1.0, 0.9)
        )
        self.renderer.add_light(sun_light)
        
        # Fill light
        fill_light = Light3D(
            type=LightType.DIRECTIONAL,
            position=Vector3D(-50, 50, 100),
            direction=Vector3D(0.5, -0.5, -1).normalized(),
            ambient=(0.1, 0.1, 0.2),
            diffuse=(0.3, 0.3, 0.4),
            specular=(0.2, 0.2, 0.3)
        )
        self.renderer.add_light(fill_light)
    
    def _create_environment(self):
        """Create the simulation environment."""
        # Ground plane
        ground_shape = Shape3D(Shape3DType.BOX, Vector3D(50, 0.1, 50))
        ground = PhysicsObject3D(
            name="ground",
            shape=ground_shape,
            body_type=Body3DType.STATIC,
            position=Vector3D(0, 0, 0)
        )
        ground.set_color((0.2, 0.8, 0.2))  # Green ground
        self.physics_engine.add_object(ground)
        self.renderer.add_physics_object(ground)
        
        # Landing pad
        pad_shape = Shape3D(Shape3DType.CYLINDER, Vector3D(2, 0.05, 2))
        landing_pad = PhysicsObject3D(
            name="landing_pad",
            shape=pad_shape,
            body_type=Body3DType.STATIC,
            position=Vector3D(0, 0.1, 0)
        )
        landing_pad.set_color((0.9, 0.1, 0.1))  # Red landing pad
        self.physics_engine.add_object(landing_pad)
        self.renderer.add_physics_object(landing_pad)
        
        # Some obstacles/landmarks
        for i, (x, z) in enumerate([(10, 10), (-10, 10), (10, -10), (-10, -10)]):
            pole_shape = Shape3D(Shape3DType.CYLINDER, Vector3D(0.2, 3, 0.2))
            pole = PhysicsObject3D(
                name=f"pole_{i}",
                shape=pole_shape,
                body_type=Body3DType.STATIC,
                position=Vector3D(x, 3, z)
            )
            pole.set_color((0.6, 0.4, 0.2))  # Brown poles
            self.physics_engine.add_object(pole)
            self.renderer.add_physics_object(pole)
    
    def run(self):
        """Run the demo main loop."""
        self.running = True
        last_time = time.time()
        
        self.logger.info("Starting drone demo")
        
        # Auto-arm the drone after a short delay
        pygame.time.set_timer(pygame.USEREVENT + 1, 3000)  # Auto-arm in 3 seconds
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            self._handle_events()
            
            if not self.paused:
                # Update input
                input_state = self._update_input()
                
                # Update simulation
                self.drone_simulation.update(dt, input_state)
                
                # Update camera
                self.drone_simulation.handle_camera_update(self.camera, dt)
                
                # Update physics
                self.physics_engine.step(dt)
                
                # Update camera
                self.camera.update(dt)
            
            # Render frame
            self._render()
            
            # Update performance metrics
            self._update_fps()
            
            # Control frame rate
            self.clock.tick(self.fps_target)
        
        self.cleanup()
    
    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                self._handle_key_press(event.key)
            
            elif event.type == pygame.USEREVENT + 1:
                # Auto-arm the drone
                if self.drone_simulation:
                    self.drone_simulation.arm_drone(True)
                    self.logger.info("Drone auto-armed for demo")
    
    def _handle_key_press(self, key):
        """Handle key press events."""
        if key == pygame.K_ESCAPE:
            self.running = False
        
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
            self.logger.info(f"Demo {'paused' if self.paused else 'resumed'}")
        
        elif key == pygame.K_r:
            # Reset simulation
            self.drone_simulation.reset()
            self.logger.info("Simulation reset")
        
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        
        elif key == pygame.K_i:
            self.show_info = not self.show_info
        
        elif key == pygame.K_s:
            self.show_sensors = not self.show_sensors
        
        elif key == pygame.K_t:
            # Toggle trajectory display
            self.drone_simulation.toggle_educational_features(show_trajectory=True)
        
        elif key == pygame.K_f:
            # Toggle force display
            self.drone_simulation.toggle_educational_features(show_forces=True)
        
        elif key == pygame.K_w:
            # Enable wind
            self.drone_simulation.set_wind(5.0, 45.0, True)
            self.logger.info("Wind enabled (5 m/s, 45°)")
        
        elif key == pygame.K_c:
            # Cycle camera modes
            modes = [CameraMode.FREE, CameraMode.ORBIT, CameraMode.FOLLOW, CameraMode.FIRST_PERSON]
            current_idx = modes.index(self.camera.mode)
            next_idx = (current_idx + 1) % len(modes)
            self.camera.mode = modes[next_idx]
            self.logger.info(f"Camera mode: {self.camera.mode.value}")
        
        # Flight mode keys
        elif key == pygame.K_1:
            self.drone_simulation.set_flight_mode(FlightMode.MANUAL)
        elif key == pygame.K_2:
            self.drone_simulation.set_flight_mode(FlightMode.STABILIZED)
        elif key == pygame.K_3:
            self.drone_simulation.set_flight_mode(FlightMode.ALTITUDE_HOLD)
        elif key == pygame.K_4:
            self.drone_simulation.set_flight_mode(FlightMode.POSITION_HOLD)
        elif key == pygame.K_5:
            self.drone_simulation.set_flight_mode(FlightMode.RETURN_TO_HOME)
    
    def _update_input(self) -> InputState:
        """Update input state."""
        self.joystick_manager.update()
        return self.input_processor.process_input()
    
    def _render(self):
        """Render the current frame."""
        # Clear screen
        self.window.fill((0, 0, 0))
        
        # Render 3D scene
        self.renderer.render(self.camera)
        
        # Render UI
        self._render_ui()
        
        # Update display
        pygame.display.flip()
    
    def _render_ui(self):
        """Render user interface elements."""
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # Performance info
        fps_text = font.render(f"FPS: {self.current_fps:.1f}", True, (255, 255, 255))
        self.window.blit(fps_text, (10, 10))
        
        if self.show_info and self.drone_simulation:
            # Simulation data
            data = self.drone_simulation.get_simulation_data()
            y_offset = 40
            
            info_lines = [
                f"Mode: {data.get('flight_controller', {}).get('mode', 'Unknown')}",
                f"Armed: {data.get('flight_controller', {}).get('armed', False)}",
                f"Altitude: {data.get('altitude', 0):.2f}m",
                f"Position: ({data.get('position', [0,0,0])[0]:.1f}, {data.get('position', [0,0,0])[2]:.1f})",
                f"Flight Time: {data.get('flight_time', 0):.1f}s",
                f"Distance: {data.get('total_distance', 0):.1f}m"
            ]
            
            for line in info_lines:
                text = small_font.render(line, True, (255, 255, 255))
                self.window.blit(text, (10, y_offset))
                y_offset += 20
        
        if self.show_sensors and self.drone_simulation:
            # Sensor data
            data = self.drone_simulation.get_simulation_data()
            sensors = data.get('sensors', {})
            
            y_offset = 200
            sensor_lines = [
                "=== Sensors ===",
                f"IMU Accel: ({sensors.get('imu', {}).get('accel_x', 0):.2f}, {sensors.get('imu', {}).get('accel_y', 0):.2f}, {sensors.get('imu', {}).get('accel_z', 0):.2f})",
                f"IMU Gyro: ({sensors.get('imu', {}).get('gyro_x', 0):.2f}, {sensors.get('imu', {}).get('gyro_y', 0):.2f}, {sensors.get('imu', {}).get('gyro_z', 0):.2f})",
                f"GPS: {sensors.get('gps', {}).get('latitude', 0):.6f}, {sensors.get('gps', {}).get('longitude', 0):.6f}",
                f"Barometer: {sensors.get('barometer', {}).get('altitude', 0):.2f}m"
            ]
            
            for line in sensor_lines:
                text = small_font.render(line, True, (200, 200, 255))
                self.window.blit(text, (10, y_offset))
                y_offset += 18
        
        if self.show_help:
            # Help text
            help_lines = [
                "=== Controls ===",
                "WASD: Roll/Pitch",
                "QE: Yaw", 
                "Space/Shift: Throttle",
                "1-5: Flight Modes",
                "Enter: Arm/Disarm",
                "",
                "=== Keys ===",
                "C: Cycle Camera",
                "R: Reset",
                "Space: Pause",
                "T: Show Trajectory", 
                "F: Show Forces",
                "W: Enable Wind",
                "H: Toggle Help",
                "I: Toggle Info",
                "S: Toggle Sensors",
                "ESC: Exit"
            ]
            
            x_offset = self.window_width - 200
            y_offset = 50
            
            for line in help_lines:
                if line.startswith("==="):
                    color = (255, 255, 0)  # Yellow for headers
                else:
                    color = (200, 200, 200)  # Light gray for text
                
                text = small_font.render(line, True, color)
                self.window.blit(text, (x_offset, y_offset))
                y_offset += 18
        
        # Status messages
        if self.paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(self.window_width // 2, 50))
            self.window.blit(pause_text, text_rect)
    
    def _update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up drone demo")
        
        if self.drone_simulation:
            self.drone_simulation.cleanup()
        
        if self.renderer:
            self.renderer.cleanup()
        
        if self.physics_engine:
            self.physics_engine.cleanup()
        
        pygame.quit()


def main():
    """Main function."""
    print("=== PyJoySim Drone Simulation Demo ===")
    print()
    print("This demo showcases a comprehensive quadrotor drone simulation with:")
    print("- Realistic physics and flight dynamics")
    print("- Multiple flight modes (Manual, Stabilized, Altitude Hold, Position Hold)")
    print("- Sensor simulation (IMU, GPS, Barometer)")
    print("- Flight controller with PID loops")
    print("- Visual effects and educational features")
    print()
    print("Controls:")
    print("- Keyboard: WASD (Roll/Pitch), QE (Yaw), Space/Shift (Throttle)")
    print("- Flight Modes: 1 (Manual), 2 (Stabilized), 3 (Altitude Hold), 4 (Position Hold)")
    print("- Camera: C to cycle modes")
    print("- Features: T (Trajectory), F (Forces), W (Wind)")
    print("- ESC to exit")
    print()
    
    demo = DroneDemo()
    
    if demo.initialize():
        print("Demo initialized successfully. Starting simulation...")
        demo.run()
    else:
        print("Failed to initialize demo")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())