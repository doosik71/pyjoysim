"""
Car simulation module for PyJoySim.

This module implements a realistic 2D car simulation with proper vehicle
dynamics, steering, acceleration, braking, and tire physics.
"""

import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from ...simulation import BaseSimulation, SimulationConfig, SimulationCategory, SimulationMetadata
from ...physics import Vector2D, PhysicsObject, create_box, StandardMaterials
from ...rendering import StandardColors, Color
from ...input import InputEvent, InputEventType
from ...core.logging import get_logger
from .track_system import TrackLoader, TrackInstance


class CarType(Enum):
    """Types of cars available."""
    SPORTS_CAR = "sports_car"
    SUV = "suv"
    TRUCK = "truck"


@dataclass
class CarConfiguration:
    """Configuration for different car types."""
    mass: float                    # kg
    wheelbase: float              # m
    track_width: float            # m
    length: float                 # m
    width: float                  # m
    max_steering_angle: float     # degrees
    max_engine_force: float       # N
    max_brake_force: float        # N
    drag_coefficient: float       # dimensionless
    tire_grip: float              # friction coefficient
    center_of_mass_offset: float  # m from geometric center
    
    @classmethod
    def get_config(cls, car_type: CarType) -> 'CarConfiguration':
        """Get configuration for a specific car type."""
        configs = {
            CarType.SPORTS_CAR: cls(
                mass=1200,
                wheelbase=2.5,
                track_width=1.8,
                length=4.2,
                width=1.8,
                max_steering_angle=35,
                max_engine_force=4000,
                max_brake_force=6000,
                drag_coefficient=0.25,
                tire_grip=1.2,
                center_of_mass_offset=0.1
            ),
            CarType.SUV: cls(
                mass=2000,
                wheelbase=2.8,
                track_width=1.9,
                length=4.8,
                width=1.9,
                max_steering_angle=30,
                max_engine_force=3500,
                max_brake_force=7000,
                drag_coefficient=0.35,
                tire_grip=0.9,
                center_of_mass_offset=0.2
            ),
            CarType.TRUCK: cls(
                mass=3500,
                wheelbase=3.2,
                track_width=2.0,
                length=6.0,
                width=2.2,
                max_steering_angle=25,
                max_engine_force=5000,
                max_brake_force=8000,
                drag_coefficient=0.45,
                tire_grip=0.8,
                center_of_mass_offset=0.3
            )
        }
        return configs[car_type]


@dataclass
class CarState:
    """Current state of the car."""
    position: Vector2D
    velocity: Vector2D
    angular_velocity: float       # rad/s
    heading: float               # radians
    steering_angle: float        # radians
    throttle: float             # 0.0 to 1.0
    brake: float                # 0.0 to 1.0
    handbrake: bool
    gear: int                   # -1=reverse, 0=neutral, 1+=forward
    rpm: float
    speed_kmh: float
    
    def __post_init__(self):
        """Calculate derived values."""
        self.speed_kmh = self.velocity.magnitude() * 3.6  # m/s to km/h


class CarPhysics:
    """Handles car physics calculations."""
    
    def __init__(self, config: CarConfiguration):
        """Initialize car physics with configuration."""
        self.config = config
        self.logger = get_logger("car_physics")
        
        # Physics constants
        self.air_density = 1.225  # kg/m³
        self.frontal_area = config.width * 1.5  # Estimated frontal area
        
        # Tire model parameters
        self.tire_lateral_stiffness = 80000  # N/rad
        self.tire_longitudinal_stiffness = 10000  # N/slip_ratio
        
    def calculate_forces(self, state: CarState, dt: float) -> Tuple[Vector2D, float]:
        """
        Calculate forces and torques acting on the car.
        
        Args:
            state: Current car state
            dt: Time step
            
        Returns:
            Tuple of (force_vector, torque)
        """
        # Calculate tire forces
        front_force, rear_force = self._calculate_tire_forces(state)
        
        # Engine force
        engine_force = self._calculate_engine_force(state)
        
        # Brake force
        brake_force = self._calculate_brake_force(state)
        
        # Aerodynamic drag
        drag_force = self._calculate_drag_force(state)
        
        # Total longitudinal force (rear wheel drive)
        total_longitudinal = engine_force - brake_force
        
        # Transform forces to world coordinates
        cos_heading = math.cos(state.heading)
        sin_heading = math.sin(state.heading)
        
        # Apply forces at wheel positions
        total_force = Vector2D(
            (total_longitudinal + front_force.x) * cos_heading - front_force.y * sin_heading - drag_force.x,
            (total_longitudinal + front_force.x) * sin_heading + front_force.y * cos_heading - drag_force.y
        )
        
        # Calculate torque from lateral tire forces
        front_moment = front_force.y * self.config.wheelbase * 0.6  # Front axle position
        rear_moment = rear_force.y * self.config.wheelbase * (-0.4)  # Rear axle position
        steering_moment = self._calculate_steering_moment(state, front_force)
        
        total_torque = front_moment + rear_moment + steering_moment
        
        return total_force, total_torque
    
    def _calculate_tire_forces(self, state: CarState) -> Tuple[Vector2D, Vector2D]:
        """Calculate tire forces for front and rear axles."""
        # Simplified tire model using slip angles
        
        # Calculate slip angles
        front_slip_angle = self._calculate_front_slip_angle(state)
        rear_slip_angle = self._calculate_rear_slip_angle(state)
        
        # Calculate lateral forces using linear tire model
        front_lateral = -self.tire_lateral_stiffness * front_slip_angle
        rear_lateral = -self.tire_lateral_stiffness * rear_slip_angle
        
        # Limit forces by tire grip
        max_front_force = self.config.tire_grip * self.config.mass * 9.81 * 0.6  # 60% weight on front
        max_rear_force = self.config.tire_grip * self.config.mass * 9.81 * 0.4   # 40% weight on rear
        
        front_lateral = max(-max_front_force, min(max_front_force, front_lateral))
        rear_lateral = max(-max_rear_force, min(max_rear_force, rear_lateral))
        
        front_force = Vector2D(0, front_lateral)
        rear_force = Vector2D(0, rear_lateral)
        
        return front_force, rear_force
    
    def _calculate_front_slip_angle(self, state: CarState) -> float:
        """Calculate front tire slip angle."""
        if abs(state.velocity.magnitude()) < 0.1:
            return 0.0
        
        # Velocity at front axle
        front_velocity = Vector2D(
            state.velocity.x + state.angular_velocity * self.config.wheelbase * 0.6 * math.sin(state.heading),
            state.velocity.y - state.angular_velocity * self.config.wheelbase * 0.6 * math.cos(state.heading)
        )
        
        # Transform to car coordinates
        cos_heading = math.cos(state.heading)
        sin_heading = math.sin(state.heading)
        
        local_vx = front_velocity.x * cos_heading + front_velocity.y * sin_heading
        local_vy = -front_velocity.x * sin_heading + front_velocity.y * cos_heading
        
        if abs(local_vx) < 0.1:
            return 0.0
        
        slip_angle = math.atan2(local_vy, local_vx) - state.steering_angle
        return slip_angle
    
    def _calculate_rear_slip_angle(self, state: CarState) -> float:
        """Calculate rear tire slip angle."""
        if abs(state.velocity.magnitude()) < 0.1:
            return 0.0
        
        # Velocity at rear axle
        rear_velocity = Vector2D(
            state.velocity.x - state.angular_velocity * self.config.wheelbase * 0.4 * math.sin(state.heading),
            state.velocity.y + state.angular_velocity * self.config.wheelbase * 0.4 * math.cos(state.heading)
        )
        
        # Transform to car coordinates
        cos_heading = math.cos(state.heading)
        sin_heading = math.sin(state.heading)
        
        local_vx = rear_velocity.x * cos_heading + rear_velocity.y * sin_heading
        local_vy = -rear_velocity.x * sin_heading + rear_velocity.y * cos_heading
        
        if abs(local_vx) < 0.1:
            return 0.0
        
        slip_angle = math.atan2(local_vy, local_vx)
        return slip_angle
    
    def _calculate_engine_force(self, state: CarState) -> float:
        """Calculate engine force based on throttle and RPM."""
        if state.gear <= 0:
            return 0.0
        
        # Simple engine model
        base_force = state.throttle * self.config.max_engine_force
        
        # RPM effect (simplified)
        rpm_factor = min(1.0, state.rpm / 6000.0)  # Optimal at 6000 RPM
        if state.rpm > 6000:
            rpm_factor = max(0.3, 1.0 - (state.rpm - 6000) / 2000.0)
        
        return base_force * rpm_factor
    
    def _calculate_brake_force(self, state: CarState) -> float:
        """Calculate braking force."""
        brake_factor = state.brake
        if state.handbrake:
            brake_factor = max(brake_factor, 0.8)
        
        return brake_factor * self.config.max_brake_force
    
    def _calculate_drag_force(self, state: CarState) -> Vector2D:
        """Calculate aerodynamic drag force."""
        if state.velocity.magnitude() < 0.1:
            return Vector2D(0, 0)
        
        drag_magnitude = (0.5 * self.air_density * self.frontal_area * 
                         self.config.drag_coefficient * state.velocity.magnitude_squared())
        
        # Drag opposes velocity
        drag_direction = state.velocity.normalized() * -1
        return drag_direction * drag_magnitude
    
    def _calculate_steering_moment(self, state: CarState, front_force: Vector2D) -> float:
        """Calculate moment from steering."""
        # Self-aligning torque (simplified)
        return -front_force.y * 0.05  # Small trail distance


class Car:
    """Represents a car in the simulation."""
    
    def __init__(self, 
                 physics_engine,
                 position: Vector2D, 
                 car_type: CarType = CarType.SPORTS_CAR,
                 name: str = "Car"):
        """Initialize a car."""
        self.name = name
        self.car_type = car_type
        self.config = CarConfiguration.get_config(car_type)
        self.physics = CarPhysics(self.config)
        self.logger = get_logger(f"car.{name}")
        
        # Initialize state
        self.state = CarState(
            position=position,
            velocity=Vector2D(0, 0),
            angular_velocity=0.0,
            heading=0.0,
            steering_angle=0.0,
            throttle=0.0,
            brake=0.0,
            handbrake=False,
            gear=1,
            rpm=800.0,  # Idle RPM
            speed_kmh=0.0
        )
        
        # Create physics body
        self.physics_body, self.collider = create_box(
            physics_engine,
            position,
            width=self.config.length,
            height=self.config.width,
            mass=self.config.mass,
            material=StandardMaterials.METAL,
            name=name
        )
        
        # Visual properties
        self.color = self._get_car_color()
        
        self.logger.info("Car created", extra={
            "type": car_type.value,
            "position": position.to_tuple(),
            "mass": self.config.mass
        })
    
    def update(self, dt: float) -> None:
        """Update car physics."""
        # Update RPM based on speed and gear
        self._update_rpm()
        
        # Calculate forces
        force, torque = self.physics.calculate_forces(self.state, dt)
        
        # Apply forces to physics body
        self.physics_body.apply_force(force)
        self.physics_body.apply_torque(torque)
        
        # Update state from physics body
        self._sync_state_from_physics()
        
        # Update derived values
        self.state.speed_kmh = self.state.velocity.magnitude() * 3.6
    
    def set_controls(self, 
                    throttle: float, 
                    brake: float, 
                    steering: float,
                    handbrake: bool = False) -> None:
        """Set car controls."""
        self.state.throttle = max(0.0, min(1.0, throttle))
        self.state.brake = max(0.0, min(1.0, brake))
        self.state.handbrake = handbrake
        
        # Convert steering input to steering angle
        max_angle_rad = math.radians(self.config.max_steering_angle)
        self.state.steering_angle = steering * max_angle_rad
    
    def _update_rpm(self) -> None:
        """Update engine RPM based on speed."""
        if self.state.gear <= 0:
            self.state.rpm = 800.0  # Idle
            return
        
        # Simple gear ratio calculation
        gear_ratios = [3.5, 2.5, 1.8, 1.3, 1.0, 0.8]  # 6-speed transmission
        gear_index = min(self.state.gear - 1, len(gear_ratios) - 1)
        gear_ratio = gear_ratios[gear_index]
        
        # Wheel speed to engine RPM
        wheel_speed = self.state.velocity.magnitude()  # m/s
        wheel_rpm = (wheel_speed * 60) / (2 * math.pi * 0.3)  # Assuming 0.3m wheel radius
        engine_rpm = wheel_rpm * gear_ratio * 4.0  # Final drive ratio
        
        # Minimum idle RPM
        self.state.rpm = max(800.0, engine_rpm)
        
        # Auto shift (simplified)
        if self.state.rpm > 6500 and self.state.gear < 6:
            self.state.gear += 1
        elif self.state.rpm < 1500 and self.state.gear > 1:
            self.state.gear -= 1
    
    def _sync_state_from_physics(self) -> None:
        """Sync car state from physics body."""
        self.state.position = self.physics_body.position
        self.state.velocity = self.physics_body.velocity
        self.state.angular_velocity = self.physics_body.angular_velocity
        self.state.heading = self.physics_body.angle
    
    def _get_car_color(self) -> Color:
        """Get color based on car type."""
        colors = {
            CarType.SPORTS_CAR: StandardColors.RED,
            CarType.SUV: StandardColors.BLUE,
            CarType.TRUCK: StandardColors.GREEN
        }
        return colors.get(self.car_type, StandardColors.WHITE)
    
    def get_dashboard_data(self) -> Dict[str, float]:
        """Get data for dashboard display."""
        return {
            "speed_kmh": self.state.speed_kmh,
            "rpm": self.state.rpm,
            "gear": self.state.gear,
            "throttle": self.state.throttle,
            "brake": self.state.brake,
            "steering_angle": math.degrees(self.state.steering_angle)
        }


# Simulation metadata
CAR_SIMULATION_METADATA = SimulationMetadata(
    name="car_simulation",
    display_name="2D Car Simulation",
    description="Realistic car physics simulation with joystick control",
    category=SimulationCategory.VEHICLE,
    author="PyJoySim Team",
    version="1.0",
    difficulty="Intermediate",
    tags=["car", "physics", "driving", "vehicle"],
    requirements=["pygame", "pymunk"]
)


class CarSimulation(BaseSimulation):
    """
    2D car simulation with realistic vehicle dynamics.
    
    Features:
    - Realistic car physics with tire models
    - Multiple car types (sports car, SUV, truck)
    - Joystick steering, acceleration, and braking
    - Dashboard with speedometer, tachometer, gear indicator
    - Track boundaries and collision detection
    """
    
    def __init__(self, name: str = "car_simulation", config: Optional[SimulationConfig] = None):
        """Initialize the car simulation."""
        if config is None:
            config = SimulationConfig(
                window_title="PyJoySim - Car Simulation",
                window_width=1024,
                window_height=768,
                enable_debug_draw=True,
                enable_performance_monitoring=True
            )
        
        super().__init__(name, config)
        
        # Car and track
        self.car: Optional[Car] = None
        self.track_loader = TrackLoader()
        self.current_track: Optional[TrackInstance] = None
        self.selected_track_name = "oval"  # Default track
        
        # Controls
        self.steering_input = 0.0
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.handbrake_active = False
        
        # Camera following
        self.camera_follow_enabled = True
        
        # Lap timing and checkpoints
        self.last_car_position: Optional[Vector2D] = None
        self.current_checkpoint = 0
        self.lap_start_time = 0.0
        
        self.logger.info("Car simulation created")
    
    def on_initialize(self) -> bool:
        """Initialize car simulation."""
        try:
            self.logger.info("Initializing car simulation")
            
            # Create basic tracks if they don't exist
            available_tracks = self.track_loader.list_tracks()
            if not available_tracks:
                self.logger.info("Creating default tracks")
                self.track_loader.create_basic_tracks()
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize car simulation", extra={
                "error": str(e)
            })
            return False
    
    def on_start(self) -> None:
        """Set up the car simulation scene."""
        self.logger.info("Setting up car simulation scene")
        
        # Load track
        self._load_track(self.selected_track_name)
        
        # Create car at spawn position
        self._create_car()
        
        # Set up camera
        self._setup_camera()
        
        self.logger.info("Car simulation scene setup complete")
    
    def _load_track(self, track_name: str) -> None:
        """Load a track by name."""
        track_data = self.track_loader.load_track(track_name)
        if track_data:
            self.current_track = TrackInstance(track_data, self.physics_world.engine)
            # Add track walls to physics world
            for wall_obj in self.current_track.wall_objects:
                self.physics_world.add_object(wall_obj, "track")
        else:
            self.logger.warning("Failed to load track, using fallback", extra={"track": track_name})
            self._create_fallback_track()
    
    def _create_fallback_track(self) -> None:
        """Create a simple fallback track if loading fails."""
        # Simple rectangular track
        track_width = 50
        track_height = 30
        wall_thickness = 2
        
        walls = []
        # Outer boundary
        walls.extend([
            create_box(self.physics_world.engine, Vector2D(0, track_height//2 + wall_thickness//2), 
                      track_width + wall_thickness * 2, wall_thickness, 0, StandardMaterials.CONCRETE, "TopWall")[0],
            create_box(self.physics_world.engine, Vector2D(0, -track_height//2 - wall_thickness//2), 
                      track_width + wall_thickness * 2, wall_thickness, 0, StandardMaterials.CONCRETE, "BottomWall")[0],
            create_box(self.physics_world.engine, Vector2D(-track_width//2 - wall_thickness//2, 0), 
                      wall_thickness, track_height, 0, StandardMaterials.CONCRETE, "LeftWall")[0],
            create_box(self.physics_world.engine, Vector2D(track_width//2 + wall_thickness//2, 0), 
                      wall_thickness, track_height, 0, StandardMaterials.CONCRETE, "RightWall")[0]
        ])
        
        for wall in walls:
            self.physics_world.add_object(wall, "track")
    
    def _create_car(self) -> None:
        """Create the player car."""
        # Use track spawn position if available
        if self.current_track:
            start_position = self.current_track.get_spawn_position()
            start_heading = self.current_track.get_spawn_heading()
        else:
            start_position = Vector2D(0, 0)
            start_heading = 0.0
        
        self.car = Car(
            self.physics_world.engine,
            start_position,
            CarType.SPORTS_CAR,
            "PlayerCar"
        )
        
        # Set initial heading
        self.car.state.heading = start_heading
        self.car.physics_body.set_angle(start_heading)
        
        # Add to physics world
        self.physics_world.add_object(self.car.physics_body, "car")
        
        # Initialize position tracking
        self.last_car_position = start_position
    
    def _setup_camera(self) -> None:
        """Set up camera to follow the car."""
        if self.camera and self.car:
            # Follow the car with some smoothing
            self.camera.follow_object(
                self.car.physics_body,
                smoothing=0.2,
                offset=Vector2D(0, 0)
            )
            # Set appropriate zoom for driving
            self.camera.set_zoom(15.0)
    
    def on_update(self, dt: float) -> None:
        """Update car simulation."""
        if self.car:
            # Store previous position for checkpoint detection
            if self.last_car_position is None:
                self.last_car_position = self.car.state.position
            
            # Apply controls to car
            self.car.set_controls(
                self.throttle_input,
                self.brake_input,
                self.steering_input,
                self.handbrake_active
            )
            
            # Update car physics
            self.car.update(dt)
            
            # Check for checkpoint crossings
            if self.current_track:
                checkpoint_id = self.current_track.check_checkpoint_crossing(
                    self.last_car_position,
                    self.car.state.position,
                    time.time()
                )
                
                if checkpoint_id is not None:
                    self.logger.debug("Checkpoint crossed", extra={"checkpoint": checkpoint_id})
            
            # Update position tracking
            self.last_car_position = self.car.state.position
    
    def on_render(self, renderer) -> None:
        """Render car simulation."""
        # Render track
        self._render_track(renderer)
        
        # Render car
        self._render_car(renderer)
        
        # Render dashboard
        self._render_dashboard(renderer)
        
        # Render lap timing info
        self._render_lap_info(renderer)
    
    def _render_track(self, renderer) -> None:
        """Render track."""
        if self.current_track:
            self.current_track.render(renderer, self.camera)
    
    def _render_car(self, renderer) -> None:
        """Render the car."""
        if not self.car or not self.car.physics_body.is_active:
            return
        
        screen_pos = self.world_to_screen(self.car.state.position)
        scale = self.camera.zoom if self.camera else 20
        
        # Car body
        renderer.draw_rectangle(
            screen_pos,
            self.car.config.length * scale,
            self.car.config.width * scale,
            self.car.color,
            fill=True
        )
        
        # Car outline
        renderer.draw_rectangle(
            screen_pos,
            self.car.config.length * scale,
            self.car.config.width * scale,
            StandardColors.BLACK,
            fill=False,
            line_width=2
        )
        
        # Direction indicator (front of car)
        front_offset = Vector2D(
            math.cos(self.car.state.heading) * self.car.config.length * 0.4,
            math.sin(self.car.state.heading) * self.car.config.length * 0.4
        )
        front_pos = self.world_to_screen(self.car.state.position + front_offset)
        
        renderer.draw_circle(
            front_pos,
            5,
            StandardColors.WHITE,
            fill=True
        )
    
    def _render_dashboard(self, renderer) -> None:
        """Render car dashboard."""
        if not self.car:
            return
        
        dashboard_data = self.car.get_dashboard_data()
        
        # Dashboard background
        dashboard_y = self.config.window_height - 100
        renderer.draw_rectangle(
            Vector2D(self.config.window_width // 2, dashboard_y),
            self.config.window_width,
            100,
            Color(0, 0, 0, 150),  # Semi-transparent black
            fill=True
        )
        
        # Speed
        speed_text = f"Speed: {dashboard_data['speed_kmh']:.1f} km/h"
        renderer.draw_text(
            speed_text,
            Vector2D(20, dashboard_y - 40),
            StandardColors.WHITE,
            font_size=18
        )
        
        # RPM
        rpm_text = f"RPM: {dashboard_data['rpm']:.0f}"
        renderer.draw_text(
            rpm_text,
            Vector2D(20, dashboard_y - 20),
            StandardColors.WHITE,
            font_size=16
        )
        
        # Gear
        gear_text = f"Gear: {dashboard_data['gear']}"
        renderer.draw_text(
            gear_text,
            Vector2D(200, dashboard_y - 40),
            StandardColors.WHITE,
            font_size=16
        )
        
        # Throttle and brake bars
        self._render_control_bars(renderer, dashboard_data, dashboard_y)
    
    def _render_control_bars(self, renderer, dashboard_data: Dict[str, float], y: float) -> None:
        """Render throttle and brake indicator bars."""
        bar_width = 20
        bar_height = 60
        
        # Throttle bar
        throttle_x = self.config.window_width - 100
        throttle_fill_height = bar_height * dashboard_data['throttle']
        
        # Throttle background
        renderer.draw_rectangle(
            Vector2D(throttle_x, y - bar_height//2),
            bar_width,
            bar_height,
            StandardColors.DARK_GRAY,
            fill=True
        )
        
        # Throttle fill
        if throttle_fill_height > 0:
            renderer.draw_rectangle(
                Vector2D(throttle_x, y - bar_height//2 + (bar_height - throttle_fill_height)//2),
                bar_width,
                throttle_fill_height,
                StandardColors.GREEN,
                fill=True
            )
        
        # Throttle label
        renderer.draw_text(
            "T",
            Vector2D(throttle_x - 10, y + bar_height//2 + 5),
            StandardColors.WHITE,
            font_size=12
        )
        
        # Brake bar
        brake_x = throttle_x + 30
        brake_fill_height = bar_height * dashboard_data['brake']
        
        # Brake background
        renderer.draw_rectangle(
            Vector2D(brake_x, y - bar_height//2),
            bar_width,
            bar_height,
            StandardColors.DARK_GRAY,
            fill=True
        )
        
        # Brake fill
        if brake_fill_height > 0:
            renderer.draw_rectangle(
                Vector2D(brake_x, y - bar_height//2 + (bar_height - brake_fill_height)//2),
                bar_width,
                brake_fill_height,
                StandardColors.RED,
                fill=True
            )
        
        # Brake label
        renderer.draw_text(
            "B",
            Vector2D(brake_x - 10, y + bar_height//2 + 5),
            StandardColors.WHITE,
            font_size=12
        )
    
    def _render_lap_info(self, renderer) -> None:
        """Render lap timing information."""
        if not self.current_track:
            return
        
        # Lap info position (top right)
        info_x = self.config.window_width - 200
        info_y = 20
        
        # Current lap
        lap_text = f"Lap: {self.current_track.lap_count}"
        renderer.draw_text(
            lap_text,
            Vector2D(info_x, info_y),
            StandardColors.WHITE,
            font_size=16
        )
        
        # Best lap time
        if self.current_track.best_lap_time < float('inf'):
            best_time_text = f"Best: {self.current_track.best_lap_time:.2f}s"
            renderer.draw_text(
                best_time_text,
                Vector2D(info_x, info_y + 20),
                StandardColors.YELLOW,
                font_size=14
            )
        
        # Current checkpoint
        checkpoint_text = f"Checkpoint: {self.current_track.last_checkpoint_id + 1}/{len(self.current_track.track_data.checkpoints)}"
        renderer.draw_text(
            checkpoint_text,
            Vector2D(info_x, info_y + 40),
            StandardColors.CYAN,
            font_size=12
        )
        
        # Track name
        track_name_text = f"Track: {self.current_track.track_data.name}"
        renderer.draw_text(
            track_name_text,
            Vector2D(20, 20),
            StandardColors.WHITE,
            font_size=14
        )
    
    def on_input_event(self, event: InputEvent) -> None:
        """Handle joystick input for car control."""
        if event.event_type == InputEventType.AXIS_CHANGE:
            if event.axis_id == 0:  # Left stick X - Steering
                self.steering_input = event.axis_value
            
            elif event.axis_id == 5:  # Right trigger - Throttle
                # Convert from -1,1 to 0,1
                self.throttle_input = max(0.0, event.axis_value)
            
            elif event.axis_id == 4:  # Left trigger - Brake
                # Convert from -1,1 to 0,1
                self.brake_input = max(0.0, event.axis_value)
        
        elif event.event_type == InputEventType.BUTTON_PRESS:
            if event.button_id == 0:  # A button - Handbrake
                self.handbrake_active = True
            elif event.button_id == 1:  # B button - Toggle camera follow
                self.camera_follow_enabled = not self.camera_follow_enabled
                if self.camera_follow_enabled and self.car:
                    self.camera.follow_object(self.car.physics_body, smoothing=0.2)
                else:
                    self.camera.follow_object(None)
            elif event.button_id == 2:  # X button - Next track
                self._next_track()
            elif event.button_id == 3:  # Y button - Reset car position
                self._reset_car_position()
        
        elif event.event_type == InputEventType.BUTTON_RELEASE:
            if event.button_id == 0:  # A button - Handbrake
                self.handbrake_active = False
    
    def on_event(self, event) -> bool:
        """Handle keyboard events."""
        if hasattr(event, 'type') and hasattr(event, 'key'):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:  # Throttle
                    self.throttle_input = 1.0
                elif event.key == pygame.K_DOWN:  # Brake
                    self.brake_input = 1.0
                elif event.key == pygame.K_LEFT:  # Steer left
                    self.steering_input = -1.0
                elif event.key == pygame.K_RIGHT:  # Steer right
                    self.steering_input = 1.0
                elif event.key == pygame.K_SPACE:  # Handbrake
                    self.handbrake_active = True
                elif event.key == pygame.K_c:  # Toggle camera follow
                    self.camera_follow_enabled = not self.camera_follow_enabled
                    if self.camera_follow_enabled and self.car:
                        self.camera.follow_object(self.car.physics_body, smoothing=0.2)
                    else:
                        self.camera.follow_object(None)
                return True
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:  # Throttle
                    self.throttle_input = 0.0
                elif event.key == pygame.K_DOWN:  # Brake
                    self.brake_input = 0.0
                elif event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:  # Steering
                    self.steering_input = 0.0
                elif event.key == pygame.K_SPACE:  # Handbrake
                    self.handbrake_active = False
                return True
        
        return True
    
    def _next_track(self) -> None:
        """Switch to the next available track."""
        available_tracks = self.track_loader.list_tracks()
        if not available_tracks:
            return
        
        try:
            current_index = available_tracks.index(self.selected_track_name)
            next_index = (current_index + 1) % len(available_tracks)
            self.selected_track_name = available_tracks[next_index]
        except ValueError:
            self.selected_track_name = available_tracks[0]
        
        self.logger.info("Switching to next track", extra={"track": self.selected_track_name})
        
        # Clean up current track
        if self.current_track:
            self.current_track.cleanup()
            self.current_track = None
        
        # Remove current track objects from physics world
        self.physics_world.remove_group("track")
        
        # Load new track
        self._load_track(self.selected_track_name)
        
        # Reset car position
        self._reset_car_position()
    
    def _reset_car_position(self) -> None:
        """Reset car to spawn position."""
        if not self.car:
            return
        
        # Get spawn position
        if self.current_track:
            spawn_pos = self.current_track.get_spawn_position()
            spawn_heading = self.current_track.get_spawn_heading()
        else:
            spawn_pos = Vector2D(0, 0)
            spawn_heading = 0.0
        
        # Reset car state
        self.car.state.position = spawn_pos
        self.car.state.velocity = Vector2D(0, 0)
        self.car.state.angular_velocity = 0.0
        self.car.state.heading = spawn_heading
        self.car.state.throttle = 0.0
        self.car.state.brake = 0.0
        self.car.state.steering_angle = 0.0
        self.car.state.rpm = 800.0
        self.car.state.gear = 1
        
        # Update physics body
        self.car.physics_body.set_position(spawn_pos)
        self.car.physics_body.set_velocity(Vector2D(0, 0))
        self.car.physics_body.set_angular_velocity(0.0)
        self.car.physics_body.set_angle(spawn_heading)
        
        # Reset timing
        if self.current_track:
            self.current_track.lap_count = 0
            self.current_track.last_checkpoint_id = -1
            self.current_track.current_lap_start_time = 0.0
            self.current_track.last_checkpoint_times.clear()
        
        # Reset input states
        self.steering_input = 0.0
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.handbrake_active = False
        
        self.logger.info("Car position reset")
    
    def on_shutdown(self) -> None:
        """Clean up car simulation."""
        self.logger.info("Cleaning up car simulation")
        
        # Clean up car
        if self.car and self.car.physics_body.is_active:
            self.car.physics_body.destroy()
        
        # Clean up track
        if self.current_track:
            self.current_track.cleanup()