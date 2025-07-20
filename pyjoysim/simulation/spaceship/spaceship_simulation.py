"""
Main spaceship simulation class.

This module integrates all spaceship components into a complete simulation:
- Realistic space physics and orbital mechanics
- Propulsion systems (main engine and RCS)
- Life support systems
- Space environment
- Educational features for space physics
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

from .physics import SpaceshipPhysics, SpaceshipPhysicsParameters, SpaceEnvironment
from .propulsion import PropulsionSystem, RCSSystem, EngineParameters, EngineType
from .life_support import LifeSupportSystem, LifeSupportParameters


class SpaceshipSimulation(BaseSimulation):
    """
    Complete spaceship simulation with realistic space physics.
    
    Features:
    - Realistic space physics with orbital mechanics
    - Advanced propulsion systems (main engine + RCS)
    - Comprehensive life support systems
    - Space environment with celestial bodies
    - Educational features for space physics
    - Joystick/keyboard control
    """
    
    def __init__(self, physics_engine: Physics3D):
        """
        Initialize spaceship simulation.
        
        Args:
            physics_engine: 3D physics engine instance
        """
        super().__init__(physics_engine)
        
        self.logger = get_logger("spaceship_simulation")
        
        # Simulation components
        self.spaceship_physics: Optional[SpaceshipPhysics] = None
        self.space_environment: Optional[SpaceEnvironment] = None
        self.propulsion_system: Optional[PropulsionSystem] = None
        self.rcs_system: Optional[RCSSystem] = None
        self.life_support: Optional[LifeSupportSystem] = None
        
        # Physics object
        self.spaceship_body: Optional[PhysicsObject3D] = None
        
        # Rendering
        self.spaceship_render_object: Optional[RenderObject3D] = None
        self.thruster_render_objects: List[RenderObject3D] = []
        
        # Control state
        self.main_engine_command = 0.0      # 0.0 to 1.0
        self.rcs_translation_command = Vector3D()
        self.rcs_rotation_command = Vector3D()
        
        # Mission state
        self.mission_time = 0.0
        self.autopilot_enabled = False
        self.target_position: Optional[Vector3D] = None
        
        # Educational features
        self.show_orbital_path = False
        self.show_gravitational_field = False
        self.show_velocity_vector = False
        self.orbital_path_points: List[Vector3D] = []
        self.max_orbital_points = 2000
        
        # Performance metrics
        self.total_delta_v_used = 0.0
        self.mission_distance = 0.0
        self.last_position = Vector3D()
        
        self.logger.info("SpaceshipSimulation initialized")
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the spaceship simulation.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            True if initialization successful
        """
        try:
            # Get configuration
            config = kwargs.get('config', {})
            
            # Initialize space environment
            self.space_environment = SpaceEnvironment()
            
            # Initialize spaceship physics
            physics_params = SpaceshipPhysicsParameters()
            if 'physics' in config:
                physics_config = config['physics']
                physics_params.dry_mass = physics_config.get('dry_mass', physics_params.dry_mass)
                physics_params.fuel_capacity = physics_config.get('fuel_capacity', physics_params.fuel_capacity)
                physics_params.main_engine_thrust = physics_config.get('main_engine_thrust', physics_params.main_engine_thrust)
            
            self.spaceship_physics = SpaceshipPhysics(physics_params, self.space_environment)
            
            # Initialize propulsion systems
            main_engine_params = EngineParameters(
                name="Main Engine",
                engine_type=EngineType.CHEMICAL_ROCKET,
                max_thrust=physics_params.main_engine_thrust,
                specific_impulse=physics_params.specific_impulse,
                fuel_flow_rate=physics_params.main_engine_thrust / (physics_params.specific_impulse * 9.80665)
            )
            self.propulsion_system = PropulsionSystem(main_engine_params)
            self.rcs_system = RCSSystem()
            
            # Initialize life support
            life_support_params = LifeSupportParameters()
            if 'life_support' in config:
                ls_config = config['life_support']
                life_support_params.crew_count = ls_config.get('crew_count', life_support_params.crew_count)
            
            self.life_support = LifeSupportSystem(life_support_params)
            
            # Create spaceship physics body
            self._create_spaceship_body()
            
            # Set initial position (Earth orbit by default)
            initial_pos = config.get('initial_position', [149597870700 + 6371000 + 400000, 0, 0])  # 400km Earth orbit
            self.spaceship_body.set_position(Vector3D(*initial_pos))
            
            # Set initial orbital velocity for Earth orbit
            orbital_velocity = config.get('initial_velocity', [0, 0, 7670])  # ~7.67 km/s for 400km orbit
            self.spaceship_body.set_velocity(Vector3D(*orbital_velocity), Vector3D())
            
            # Initialize rendering
            if kwargs.get('renderer'):
                self._create_spaceship_visuals(kwargs['renderer'])
            
            self.mission_time = 0.0
            self.last_position = self.spaceship_body.position.copy()
            
            self.logger.info("Spaceship simulation initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize spaceship simulation", extra={"error": str(e)})
            return False
    
    def _create_spaceship_body(self):
        """Create the main spaceship physics body."""
        # Create elongated box shape for spaceship
        body_size = Vector3D(
            self.spaceship_physics.params.length,
            self.spaceship_physics.params.height,
            self.spaceship_physics.params.width
        )
        shape = Shape3D(Shape3DType.BOX, body_size)
        
        # Create physics material (low friction in space)
        material = PhysicsMaterial3D(
            friction=0.0,  # No friction in space
            restitution=0.5,
            density=1.0,
            linear_damping=0.0,   # No air resistance in space
            angular_damping=0.01  # Minimal rotational damping
        )
        
        # Create physics object
        self.spaceship_body = PhysicsObject3D(
            name="spaceship_body",
            shape=shape,
            body_type=Body3DType.DYNAMIC,
            mass=self.spaceship_physics.get_total_mass(),
            material=material
        )
        
        # Add to physics engine
        self.physics_engine.add_object(self.spaceship_body)
        
        self.logger.debug("Spaceship physics body created")
    
    def _create_spaceship_visuals(self, renderer):
        """Create visual representation of the spaceship."""
        # Main body
        self.spaceship_render_object = create_render_object_from_physics(
            self.spaceship_body, renderer, "default")
        
        # Set spaceship body color (metallic gray)
        self.spaceship_render_object.material.diffuse = (0.7, 0.7, 0.8)
        renderer.add_render_object(self.spaceship_render_object)
        
        # Create thruster visual effects (simplified)
        thruster_positions = [
            Vector3D(-5.0, 0.0, 0.0),   # Main engine at rear
            Vector3D(5.0, 1.0, 0.0),    # RCS thrusters
            Vector3D(5.0, -1.0, 0.0),
            Vector3D(5.0, 0.0, 1.0),
            Vector3D(5.0, 0.0, -1.0)
        ]
        
        for i, pos in enumerate(thruster_positions):
            # Create small thruster nozzle
            thruster_shape = Shape3D(Shape3DType.CYLINDER, Vector3D(0.1, 0.3, 0.1))
            
            thruster_obj = PhysicsObject3D(
                name=f"thruster_{i}",
                shape=thruster_shape,
                body_type=Body3DType.KINEMATIC,
                mass=0.0,
                position=pos
            )
            
            # Create render object
            thruster_render = create_render_object_from_physics(thruster_obj, renderer, "orange")
            thruster_render.material.diffuse = (0.9, 0.5, 0.1)  # Orange thrusters
            
            renderer.add_render_object(thruster_render)
            self.thruster_render_objects.append(thruster_render)
        
        self.logger.debug("Spaceship visual objects created")
    
    def update(self, dt: float, input_state: InputState) -> None:
        """
        Update the spaceship simulation.
        
        Args:
            dt: Time step in seconds
            input_state: Current input state
        """
        if not self.spaceship_body or not self.spaceship_physics:
            return
        
        self.mission_time += dt
        
        # Update control inputs from joystick/keyboard
        self._update_control_inputs(input_state)
        
        # Update space environment
        self.space_environment.update(dt)
        
        # Update propulsion systems
        self._update_propulsion_systems(dt)
        
        # Update spaceship physics
        force, torque = self.spaceship_physics.update(dt, self.spaceship_body)
        
        # Apply forces to physics body
        self.spaceship_body.apply_force(force)
        self.spaceship_body.apply_torque(torque)
        
        # Update life support systems
        space_temp = self.space_environment.temperature
        self.life_support.update(dt, space_temp)
        
        # Update rendering
        self._update_visuals()
        
        # Update educational features
        self._update_educational_features(dt)
        
        # Update performance metrics
        self._update_metrics(dt)
    
    def _update_control_inputs(self, input_state: InputState):
        """Update control inputs from input state."""
        # Map joystick axes to spaceship control
        if input_state.joystick_available():
            joystick = input_state.get_joystick_state(0)
            if joystick:
                # Main engine control (right trigger)
                self.main_engine_command = max(0.0, joystick.get('axis_5', 0.0))  # Right trigger
                
                # RCS translation control (left stick)
                self.rcs_translation_command = Vector3D(
                    joystick.get('axis_0', 0.0),    # X translation (left stick X)
                    joystick.get('axis_3', 0.0),    # Y translation (left stick Y)
                    joystick.get('axis_1', 0.0)     # Z translation (right stick Y)
                )
                
                # RCS rotation control (right stick)
                self.rcs_rotation_command = Vector3D(
                    joystick.get('axis_4', 0.0),    # Roll (right stick X)
                    joystick.get('axis_1', 0.0),    # Pitch (right stick Y)
                    joystick.get('axis_2', 0.0)     # Yaw (left trigger - right trigger)
                )
                
                # Button mappings
                buttons = joystick.get('buttons', {})
                if buttons.get(0, False):  # A button - toggle autopilot
                    self.autopilot_enabled = not self.autopilot_enabled
        
        # Keyboard fallback
        keys = input_state.get_keys_pressed()
        if keys:
            # Main engine control
            if 'space' in keys:
                self.main_engine_command = 1.0
            else:
                self.main_engine_command = 0.0
            
            # RCS translation control
            rcs_x = 0.0
            rcs_y = 0.0
            rcs_z = 0.0
            
            if 'd' in keys:
                rcs_x = 1.0
            elif 'a' in keys:
                rcs_x = -1.0
            
            if 'w' in keys:
                rcs_y = 1.0
            elif 's' in keys:
                rcs_y = -1.0
            
            if 'q' in keys:
                rcs_z = 1.0
            elif 'e' in keys:
                rcs_z = -1.0
            
            self.rcs_translation_command = Vector3D(rcs_x, rcs_y, rcs_z)
            
            # Visual feature toggles
            if 'o' in keys:
                self.show_orbital_path = not self.show_orbital_path
            if 'g' in keys:
                self.show_gravitational_field = not self.show_gravitational_field
            if 'v' in keys:
                self.show_velocity_vector = not self.show_velocity_vector
    
    def _update_propulsion_systems(self, dt: float):
        """Update all propulsion systems."""
        # Update main engine
        self.propulsion_system.set_throttle(self.main_engine_command)
        main_thrust, main_fuel = self.propulsion_system.update(dt)
        
        # Update RCS system
        self.rcs_system.set_translation_command(self.rcs_translation_command)
        self.rcs_system.set_rotation_command(self.rcs_rotation_command)
        rcs_force, rcs_fuel = self.rcs_system.update(dt)
        
        # Apply thrust to spaceship physics
        if main_thrust > 0:
            # Main engine typically points along +X axis
            main_force = Vector3D(main_thrust, 0, 0)
            self.spaceship_physics.set_main_engine_throttle(self.main_engine_command)
        
        # Apply RCS forces
        self.spaceship_physics.set_rcs_thrust(rcs_force)
        
        # Update fuel consumption in spaceship physics
        total_fuel_consumed = main_fuel + rcs_fuel
        if total_fuel_consumed > 0:
            self.spaceship_physics.current_fuel_mass -= total_fuel_consumed
            self.spaceship_physics.current_fuel_mass = max(0.0, self.spaceship_physics.current_fuel_mass)
    
    def _update_visuals(self):
        """Update visual representation."""
        if not self.spaceship_render_object:
            return
        
        # Update main body position and rotation
        self.spaceship_render_object.position = self.spaceship_body.position
        self.spaceship_render_object.rotation = self.spaceship_body.rotation
        
        # Update thruster positions relative to spaceship
        thruster_positions = [
            Vector3D(-5.0, 0.0, 0.0),   # Main engine
            Vector3D(5.0, 1.0, 0.0),    # RCS thrusters
            Vector3D(5.0, -1.0, 0.0),
            Vector3D(5.0, 0.0, 1.0),
            Vector3D(5.0, 0.0, -1.0)
        ]
        
        for i, thruster_render in enumerate(self.thruster_render_objects):
            if i < len(thruster_positions):
                # Transform thruster position to world coordinates
                world_pos = self.spaceship_body.position + thruster_positions[i]
                thruster_render.position = world_pos
                thruster_render.rotation = self.spaceship_body.rotation
                
                # Visual effects for active thrusters
                if i == 0:  # Main engine
                    thrust_intensity = self.main_engine_command
                else:  # RCS thrusters
                    thrust_intensity = min(1.0, self.rcs_translation_command.magnitude())
                
                # Color intensity based on thrust
                thruster_render.material.diffuse = (
                    0.9,
                    0.5 + thrust_intensity * 0.4,
                    0.1 + thrust_intensity * 0.3
                )
    
    def _update_educational_features(self, dt: float):
        """Update educational visualization features."""
        # Update orbital path
        if self.show_orbital_path:
            current_pos = self.spaceship_body.position
            
            # Add point if moved enough (scale appropriate for space)
            if (not self.orbital_path_points or 
                (current_pos - self.orbital_path_points[-1]).magnitude() > 10000):  # 10km spacing
                self.orbital_path_points.append(current_pos.copy())
                
                # Limit path length
                if len(self.orbital_path_points) > self.max_orbital_points:
                    self.orbital_path_points.pop(0)
    
    def _update_metrics(self, dt: float):
        """Update performance metrics."""
        # Calculate distance traveled
        current_pos = self.spaceship_body.position
        if self.last_position:
            distance_delta = (current_pos - self.last_position).magnitude()
            self.mission_distance += distance_delta
        
        self.last_position = current_pos.copy()
        
        # Track delta-v usage
        self.total_delta_v_used = self.spaceship_physics.delta_v_used
    
    def handle_camera_update(self, camera: Camera3D, dt: float) -> None:
        """
        Update camera for spaceship simulation.
        
        Args:
            camera: Camera object to update
            dt: Time step in seconds
        """
        if not self.spaceship_body:
            return
        
        # Set spaceship as camera target for follow modes
        if camera.mode in [CameraMode.FOLLOW, CameraMode.ORBIT, 
                          CameraMode.FIRST_PERSON, CameraMode.THIRD_PERSON]:
            camera.target_object = self.spaceship_body
            
            # Adjust camera distance for space scale
            if camera.mode == CameraMode.ORBIT:
                camera.distance = 50.0  # Closer view for spaceship
    
    def get_simulation_data(self) -> Dict[str, Any]:
        """Get current simulation data for UI display."""
        data = {
            "simulation_type": "spaceship",
            "mission_time": self.mission_time,
            "mission_distance": self.mission_distance
        }
        
        if self.spaceship_body:
            data.update({
                "position": self.spaceship_body.position.to_tuple(),
                "velocity": self.spaceship_body.linear_velocity.to_tuple(),
                "speed": self.spaceship_body.linear_velocity.magnitude()
            })
        
        if self.spaceship_physics:
            data.update({
                "spaceship_status": self.spaceship_physics.get_status()
            })
        
        if self.propulsion_system:
            data.update({
                "main_engine": self.propulsion_system.get_status()
            })
        
        if self.rcs_system:
            data.update({
                "rcs_system": self.rcs_system.get_status()
            })
        
        if self.life_support:
            data.update({
                "life_support": self.life_support.get_status()
            })
        
        return data
    
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        super().reset()
        
        if self.spaceship_body:
            # Reset to Earth orbit
            self.spaceship_body.set_position(Vector3D(149597870700 + 6371000 + 400000, 0, 0))
            self.spaceship_body.set_velocity(Vector3D(0, 0, 7670), Vector3D())
        
        if self.spaceship_physics:
            self.spaceship_physics.reset()
        
        if self.propulsion_system:
            self.propulsion_system.shutdown()
        
        if self.rcs_system:
            self.rcs_system.reset()
        
        if self.life_support:
            self.life_support.reset()
        
        # Reset mission state
        self.mission_time = 0.0
        self.main_engine_command = 0.0
        self.rcs_translation_command = Vector3D()
        self.rcs_rotation_command = Vector3D()
        self.autopilot_enabled = False
        
        # Reset metrics
        self.total_delta_v_used = 0.0
        self.mission_distance = 0.0
        self.orbital_path_points.clear()
        
        self.logger.debug("Spaceship simulation reset")
    
    def cleanup(self) -> None:
        """Clean up simulation resources."""
        super().cleanup()
        
        if self.spaceship_body:
            self.physics_engine.remove_object(self.spaceship_body.name)
            self.spaceship_body = None
        
        self.logger.debug("Spaceship simulation cleaned up")
    
    def get_help_text(self) -> List[str]:
        """Get help text for spaceship controls."""
        return [
            "=== Spaceship Simulation Controls ===",
            "",
            "Gamepad Controls:",
            "  Right Trigger: Main Engine",
            "  Left Stick: RCS Translation",
            "  Right Stick: RCS Rotation",
            "  A Button: Toggle Autopilot",
            "",
            "Keyboard Controls:",
            "  Space: Main Engine",
            "  WASD: RCS Translation",
            "  QE: RCS Up/Down",
            "  Arrow Keys: RCS Rotation",
            "",
            "View Controls:",
            "  O: Toggle Orbital Path",
            "  G: Toggle Gravity Field",
            "  V: Toggle Velocity Vector",
            "",
            "Educational Features:",
            "  - Orbital mechanics visualization",
            "  - Delta-v and fuel efficiency",
            "  - Life support monitoring",
            "  - Realistic space physics",
            "",
            "Camera: F1-F3 to switch modes"
        ]