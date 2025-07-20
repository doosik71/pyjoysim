"""
Main submarine simulation class.

This module integrates all submarine components into a complete simulation:
- Realistic underwater physics with buoyancy
- Ballast tank systems for depth control
- Sonar systems for navigation
- Underwater environment
- Educational features for marine physics
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

from .physics import SubmarinePhysics, SubmarinePhysicsParameters, UnderwaterEnvironment, WaterType
from .ballast import BallastSystem
from .sonar import SonarSystem, SonarMode


class SubmarineSimulation(BaseSimulation):
    """
    Complete submarine simulation with realistic underwater physics.
    
    Features:
    - Realistic underwater physics with buoyancy and pressure
    - Advanced ballast tank system for diving/surfacing
    - Sonar system for navigation and detection
    - Underwater environment with thermoclines
    - Educational features for marine physics
    - Joystick/keyboard control
    """
    
    def __init__(self, physics_engine: Physics3D):
        """
        Initialize submarine simulation.
        
        Args:
            physics_engine: 3D physics engine instance
        """
        super().__init__(physics_engine)
        
        self.logger = get_logger("submarine_simulation")
        
        # Simulation components
        self.submarine_physics: Optional[SubmarinePhysics] = None
        self.underwater_environment: Optional[UnderwaterEnvironment] = None
        self.ballast_system: Optional[BallastSystem] = None
        self.sonar_system: Optional[SonarSystem] = None
        
        # Physics object
        self.submarine_body: Optional[PhysicsObject3D] = None
        
        # Rendering
        self.submarine_render_object: Optional[RenderObject3D] = None
        self.prop_render_object: Optional[RenderObject3D] = None
        self.conning_tower_render_object: Optional[RenderObject3D] = None
        
        # Control state
        self.engine_command = 0.0           # -1.0 to 1.0 (reverse to forward)
        self.rudder_command = 0.0           # -1.0 to 1.0 (port to starboard)
        self.stern_plane_command = 0.0      # -1.0 to 1.0 (dive to surface)
        self.ballast_command = 0.0          # -1.0 to 1.0 (blow to flood)
        
        # Mission state
        self.dive_time = 0.0
        self.target_depth = 0.0
        self.autopilot_enabled = False
        self.emergency_surface = False
        
        # Educational features
        self.show_pressure_lines = False
        self.show_current_vectors = False
        self.show_sonar_contacts = False
        self.depth_history: List[float] = []
        self.max_depth_history = 500
        
        # Performance metrics
        self.total_dive_time = 0.0
        self.max_depth_achieved = 0.0
        self.total_distance = 0.0
        self.last_position = Vector3D()
        
        self.logger.info("SubmarineSimulation initialized")
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the submarine simulation.
        
        Args:
            **kwargs: Configuration parameters
            
        Returns:
            True if initialization successful
        """
        try:
            # Get configuration
            config = kwargs.get('config', {})
            
            # Initialize underwater environment
            water_type = WaterType(config.get('water_type', 'salt_water'))
            self.underwater_environment = UnderwaterEnvironment(water_type)
            
            # Initialize submarine physics
            physics_params = SubmarinePhysicsParameters()
            if 'physics' in config:
                physics_config = config['physics']
                physics_params.length = physics_config.get('length', physics_params.length)
                physics_params.displacement = physics_config.get('displacement', physics_params.displacement)
                physics_params.max_operating_depth = physics_config.get('max_depth', physics_params.max_operating_depth)
            
            self.submarine_physics = SubmarinePhysics(physics_params, self.underwater_environment)
            
            # Initialize ballast system
            self.ballast_system = BallastSystem()
            
            # Initialize sonar system
            self.sonar_system = SonarSystem()
            initial_sonar_mode = SonarMode(config.get('sonar_mode', 'passive'))
            self.sonar_system.set_mode(initial_sonar_mode)
            
            # Create submarine physics body
            self._create_submarine_body()
            
            # Set initial position (at surface by default)
            initial_pos = config.get('initial_position', [0, 0, 0])
            self.submarine_body.set_position(Vector3D(*initial_pos))
            
            # Initialize rendering
            if kwargs.get('renderer'):
                self._create_submarine_visuals(kwargs['renderer'])
            
            self.dive_time = 0.0
            self.last_position = self.submarine_body.position.copy()
            
            self.logger.info("Submarine simulation initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize submarine simulation", extra={"error": str(e)})
            return False
    
    def _create_submarine_body(self):
        """Create the main submarine physics body."""
        # Create elongated hull shape
        body_size = Vector3D(
            self.submarine_physics.params.length,
            self.submarine_physics.params.height,
            self.submarine_physics.params.beam
        )
        shape = Shape3D(Shape3DType.BOX, body_size)
        
        # Create physics material for underwater operation
        material = PhysicsMaterial3D(
            friction=0.3,      # Moderate friction with water
            restitution=0.2,   # Low bounce
            density=1.0,
            linear_damping=0.5,   # High water resistance
            angular_damping=0.8   # High rotational damping in water
        )
        
        # Create physics object
        # Calculate mass from displacement (1 tonne displacement = 1000 kg mass in water)
        mass = self.submarine_physics.params.displacement * 1000
        
        self.submarine_body = PhysicsObject3D(
            name="submarine_body",
            shape=shape,
            body_type=Body3DType.DYNAMIC,
            mass=mass,
            material=material
        )
        
        # Add to physics engine
        self.physics_engine.add_object(self.submarine_body)
        
        self.logger.debug("Submarine physics body created")
    
    def _create_submarine_visuals(self, renderer):
        """Create visual representation of the submarine."""
        # Main hull
        self.submarine_render_object = create_render_object_from_physics(
            self.submarine_body, renderer, "default")
        
        # Set submarine hull color (dark gray/black)
        self.submarine_render_object.material.diffuse = (0.2, 0.2, 0.3)
        renderer.add_render_object(self.submarine_render_object)
        
        # Conning tower/sail
        tower_shape = Shape3D(Shape3DType.BOX, Vector3D(8.0, 8.0, 4.0))
        tower_obj = PhysicsObject3D(
            name="conning_tower",
            shape=tower_shape,
            body_type=Body3DType.KINEMATIC,
            mass=0.0,
            position=Vector3D(0, 6, 0)  # Above main hull
        )
        
        self.conning_tower_render_object = create_render_object_from_physics(tower_obj, renderer, "dark_gray")
        self.conning_tower_render_object.material.diffuse = (0.15, 0.15, 0.25)
        renderer.add_render_object(self.conning_tower_render_object)
        
        # Propeller (simplified)
        prop_shape = Shape3D(Shape3DType.CYLINDER, Vector3D(1.5, 0.2, 1.5))
        prop_obj = PhysicsObject3D(
            name="propeller",
            shape=prop_shape,
            body_type=Body3DType.KINEMATIC,
            mass=0.0,
            position=Vector3D(-self.submarine_physics.params.length * 0.45, 0, 0)  # Rear of submarine
        )
        
        self.prop_render_object = create_render_object_from_physics(prop_obj, renderer, "bronze")
        self.prop_render_object.material.diffuse = (0.8, 0.5, 0.2)  # Bronze color
        renderer.add_render_object(self.prop_render_object)
        
        self.logger.debug("Submarine visual objects created")
    
    def update(self, dt: float, input_state: InputState) -> None:
        """
        Update the submarine simulation.
        
        Args:
            dt: Time step in seconds
            input_state: Current input state
        """
        if not self.submarine_body or not self.submarine_physics:
            return
        
        self.dive_time += dt
        
        # Update control inputs from joystick/keyboard
        self._update_control_inputs(input_state)
        
        # Update submarine systems
        self._update_submarine_systems(dt)
        
        # Update submarine physics
        force, torque = self.submarine_physics.update(dt, self.submarine_body)
        
        # Apply forces to physics body
        self.submarine_body.apply_force(force)
        self.submarine_body.apply_torque(torque)
        
        # Update rendering
        self._update_visuals()
        
        # Update educational features
        self._update_educational_features(dt)
        
        # Update performance metrics
        self._update_metrics(dt)
    
    def _update_control_inputs(self, input_state: InputState):
        """Update control inputs from input state."""
        # Map joystick axes to submarine control
        if input_state.joystick_available():
            joystick = input_state.get_joystick_state(0)
            if joystick:
                # Engine control (left stick Y-axis)
                self.engine_command = -joystick.get('axis_1', 0.0)  # Forward/reverse
                
                # Rudder control (left stick X-axis)
                self.rudder_command = joystick.get('axis_0', 0.0)   # Port/starboard
                
                # Stern planes (right stick Y-axis)
                self.stern_plane_command = -joystick.get('axis_4', 0.0)  # Dive/surface
                
                # Ballast control (triggers)
                left_trigger = joystick.get('axis_2', 0.0)   # Flood ballast
                right_trigger = joystick.get('axis_5', 0.0)  # Blow ballast
                self.ballast_command = right_trigger - left_trigger
                
                # Button mappings
                buttons = joystick.get('buttons', {})
                if buttons.get(0, False):  # A button - emergency surface
                    self.emergency_surface = True
                if buttons.get(1, False):  # B button - dive
                    self.ballast_system.flood_main_ballast_tanks()
                if buttons.get(2, False):  # X button - surface
                    self.ballast_system.blow_main_ballast_tanks()
                if buttons.get(3, False):  # Y button - toggle sonar mode
                    current_mode = self.sonar_system.mode
                    if current_mode == SonarMode.PASSIVE:
                        self.sonar_system.set_mode(SonarMode.ACTIVE)
                    else:
                        self.sonar_system.set_mode(SonarMode.PASSIVE)
        
        # Keyboard fallback
        keys = input_state.get_keys_pressed()
        if keys:
            # Engine control
            if 'w' in keys:
                self.engine_command = 1.0
            elif 's' in keys:
                self.engine_command = -0.5
            else:
                self.engine_command = 0.0
            
            # Rudder control
            if 'd' in keys:
                self.rudder_command = 1.0
            elif 'a' in keys:
                self.rudder_command = -1.0
            else:
                self.rudder_command = 0.0
            
            # Stern planes
            if 'space' in keys:
                self.stern_plane_command = -1.0  # Surface
            elif 'shift' in keys:
                self.stern_plane_command = 1.0   # Dive
            else:
                self.stern_plane_command = 0.0
            
            # Ballast control
            if 'q' in keys:
                self.ballast_system.blow_main_ballast_tanks()
            elif 'e' in keys:
                self.ballast_system.flood_main_ballast_tanks()
            
            # Emergency controls
            if 'x' in keys:
                self.ballast_system.emergency_blow_all()
            
            # Sonar controls
            if 'p' in keys:
                self.sonar_system.ping(self.submarine_body.position, self.submarine_physics.current_depth)
            
            # Visual feature toggles
            if 'v' in keys:
                self.show_pressure_lines = not self.show_pressure_lines
            if 'c' in keys:
                self.show_current_vectors = not self.show_current_vectors
            if 'r' in keys:
                self.show_sonar_contacts = not self.show_sonar_contacts
    
    def _update_submarine_systems(self, dt: float):
        """Update all submarine systems."""
        current_depth = self.submarine_physics.current_depth
        water_pressure = self.underwater_environment.calculate_pressure(current_depth)
        
        # Apply control inputs to submarine physics
        self.submarine_physics.set_engine_power(abs(self.engine_command))
        self.submarine_physics.set_rudder_angle(self.rudder_command * 30.0)  # Max 30 degrees
        self.submarine_physics.set_stern_planes(self.stern_plane_command * 20.0)  # Max 20 degrees
        
        # Update ballast system
        if self.ballast_command > 0.1:  # Blow ballast
            self.ballast_system.blow_main_ballast_tanks()
        elif self.ballast_command < -0.1:  # Flood ballast
            self.ballast_system.flood_main_ballast_tanks()
        
        self.ballast_system.update(dt, current_depth, water_pressure)
        
        # Update sonar system
        self.sonar_system.update(dt, self.submarine_body.position, current_depth)
        
        # Handle emergency surface
        if self.emergency_surface:
            self.ballast_system.emergency_blow_all()
            self.emergency_surface = False
        
        # Auto trim control
        current_pitch = 0.0  # Would calculate from submarine attitude
        self.ballast_system.auto_trim_control(current_pitch, dt)
    
    def _update_visuals(self):
        """Update visual representation."""
        if not self.submarine_render_object:
            return
        
        # Update main body position and rotation
        self.submarine_render_object.position = self.submarine_body.position
        self.submarine_render_object.rotation = self.submarine_body.rotation
        
        # Update conning tower position relative to submarine
        if self.conning_tower_render_object:
            tower_offset = Vector3D(0, 6, 0)
            self.conning_tower_render_object.position = self.submarine_body.position + tower_offset
            self.conning_tower_render_object.rotation = self.submarine_body.rotation
        
        # Update propeller position and rotation
        if self.prop_render_object:
            prop_offset = Vector3D(-self.submarine_physics.params.length * 0.45, 0, 0)
            self.prop_render_object.position = self.submarine_body.position + prop_offset
            self.prop_render_object.rotation = self.submarine_body.rotation
            
            # Animate propeller based on engine power
            if self.engine_command != 0:
                # Color intensity based on engine power
                intensity = abs(self.engine_command)
                self.prop_render_object.material.diffuse = (
                    0.8,
                    0.5 + intensity * 0.3,
                    0.2 + intensity * 0.3
                )
        
        # Update submarine color based on depth
        depth_factor = min(1.0, self.submarine_physics.current_depth / 100.0)
        base_color = (0.2, 0.2, 0.3)
        self.submarine_render_object.material.diffuse = (
            base_color[0] * (1.0 - depth_factor * 0.5),
            base_color[1] * (1.0 - depth_factor * 0.5),
            base_color[2] + depth_factor * 0.2
        )
    
    def _update_educational_features(self, dt: float):
        """Update educational visualization features."""
        # Update depth history
        current_depth = self.submarine_physics.current_depth
        self.depth_history.append(current_depth)
        
        # Limit history length
        if len(self.depth_history) > self.max_depth_history:
            self.depth_history.pop(0)
    
    def _update_metrics(self, dt: float):
        """Update performance metrics."""
        # Track dive time
        if self.submarine_physics.current_depth > 1.0:
            self.total_dive_time += dt
        
        # Track maximum depth
        self.max_depth_achieved = max(self.max_depth_achieved, self.submarine_physics.current_depth)
        
        # Calculate distance traveled
        current_pos = self.submarine_body.position
        if self.last_position:
            distance_delta = (current_pos - self.last_position).magnitude()
            self.total_distance += distance_delta
        
        self.last_position = current_pos.copy()
    
    def handle_camera_update(self, camera: Camera3D, dt: float) -> None:
        """
        Update camera for submarine simulation.
        
        Args:
            camera: Camera object to update
            dt: Time step in seconds
        """
        if not self.submarine_body:
            return
        
        # Set submarine as camera target for follow modes
        if camera.mode in [CameraMode.FOLLOW, CameraMode.ORBIT, 
                          CameraMode.FIRST_PERSON, CameraMode.THIRD_PERSON]:
            camera.target_object = self.submarine_body
            
            # Adjust camera for underwater viewing
            if camera.mode == CameraMode.THIRD_PERSON:
                camera.distance = 100.0  # Closer view underwater
    
    def get_simulation_data(self) -> Dict[str, Any]:
        """Get current simulation data for UI display."""
        data = {
            "simulation_type": "submarine",
            "dive_time": self.dive_time,
            "total_distance": self.total_distance
        }
        
        if self.submarine_body:
            data.update({
                "position": self.submarine_body.position.to_tuple(),
                "velocity": self.submarine_body.linear_velocity.to_tuple(),
                "speed": self.submarine_body.linear_velocity.magnitude()
            })
        
        if self.submarine_physics:
            data.update({
                "submarine_status": self.submarine_physics.get_status()
            })
        
        if self.ballast_system:
            data.update({
                "ballast_system": self.ballast_system.get_status()
            })
        
        if self.sonar_system:
            data.update({
                "sonar_system": self.sonar_system.get_status(),
                "sonar_tactical": self.sonar_system.get_tactical_display()
            })
        
        return data
    
    def reset(self) -> None:
        """Reset the simulation to initial state."""
        super().reset()
        
        if self.submarine_body:
            # Reset to surface
            self.submarine_body.set_position(Vector3D(0, 0, 0))
            self.submarine_body.set_velocity(Vector3D(), Vector3D())
        
        if self.submarine_physics:
            self.submarine_physics.reset()
        
        if self.ballast_system:
            self.ballast_system.reset()
        
        if self.sonar_system:
            self.sonar_system.reset()
        
        # Reset mission state
        self.dive_time = 0.0
        self.target_depth = 0.0
        self.autopilot_enabled = False
        self.emergency_surface = False
        
        # Reset control inputs
        self.engine_command = 0.0
        self.rudder_command = 0.0
        self.stern_plane_command = 0.0
        self.ballast_command = 0.0
        
        # Reset metrics
        self.total_dive_time = 0.0
        self.max_depth_achieved = 0.0
        self.total_distance = 0.0
        self.depth_history.clear()
        
        self.logger.debug("Submarine simulation reset")
    
    def cleanup(self) -> None:
        """Clean up simulation resources."""
        super().cleanup()
        
        if self.submarine_body:
            self.physics_engine.remove_object(self.submarine_body.name)
            self.submarine_body = None
        
        self.logger.debug("Submarine simulation cleaned up")
    
    def get_help_text(self) -> List[str]:
        """Get help text for submarine controls."""
        return [
            "=== Submarine Simulation Controls ===",
            "",
            "Gamepad Controls:",
            "  Left Stick: Engine/Rudder",
            "  Right Stick Y: Stern Planes",
            "  Triggers: Ballast Control",
            "  A Button: Emergency Surface",
            "  B Button: Dive",
            "  X Button: Surface",
            "  Y Button: Toggle Sonar",
            "",
            "Keyboard Controls:",
            "  WS: Engine Forward/Reverse",
            "  AD: Rudder Left/Right",
            "  Space/Shift: Surface/Dive",
            "  QE: Blow/Flood Ballast",
            "  X: Emergency Blow",
            "  P: Sonar Ping",
            "",
            "View Controls:",
            "  V: Toggle Pressure Lines",
            "  C: Toggle Current Vectors",
            "  R: Toggle Sonar Contacts",
            "",
            "Educational Features:",
            "  - Underwater physics with buoyancy",
            "  - Ballast tank management",
            "  - Sonar navigation",
            "  - Pressure and depth effects",
            "",
            "Camera: F1-F3 to switch modes"
        ]