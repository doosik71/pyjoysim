"""
Base simulation framework for PyJoySim.

This module provides the foundation for all simulation types with
standardized lifecycle management and input-physics-rendering pipeline.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import pygame

from ..physics import (
    PhysicsWorld, PhysicsEngineType, Vector2D, 
    get_physics_world, create_physics_world
)
from ..rendering import (
    RenderEngine, RenderEngineType, Camera2D, CameraController,
    get_render_engine, create_render_engine, StandardColors, Color
)
from ..input import (
    get_joystick_manager, get_input_processor, InputEvent
)
from ..config import get_settings
from ..core.logging import get_logger
from ..core.exceptions import SimulationError


class SimulationState(Enum):
    """Simulation execution states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """Configuration for simulation execution."""
    target_fps: int = 60
    physics_fps: int = 60
    enable_vsync: bool = True
    enable_debug_draw: bool = False
    enable_performance_monitoring: bool = True
    window_width: int = 800
    window_height: int = 600
    window_title: str = "PyJoySim Simulation"


@dataclass
class SimulationStats:
    """Runtime statistics for simulation."""
    total_runtime: float = 0.0
    frame_count: int = 0
    physics_steps: int = 0
    average_fps: float = 0.0
    physics_fps: float = 0.0
    frame_time_ms: float = 0.0
    physics_time_ms: float = 0.0
    render_time_ms: float = 0.0
    input_events_processed: int = 0
    
    def update_frame_stats(self, frame_time: float, physics_time: float, render_time: float):
        """Update frame timing statistics."""
        self.frame_count += 1
        self.total_runtime += frame_time
        
        self.frame_time_ms = frame_time * 1000
        self.physics_time_ms = physics_time * 1000
        self.render_time_ms = render_time * 1000
        
        if frame_time > 0:
            self.average_fps = self.frame_count / self.total_runtime
        
        if physics_time > 0:
            self.physics_fps = 1.0 / physics_time


class BaseSimulation(ABC):
    """
    Abstract base class for all simulations.
    
    Provides common functionality for simulation lifecycle, input handling,
    physics stepping, and rendering coordination.
    """
    
    def __init__(self, 
                 name: str = "Simulation",
                 config: Optional[SimulationConfig] = None):
        """
        Initialize the base simulation.
        
        Args:
            name: Simulation name
            config: Simulation configuration
        """
        self.name = name
        self.config = config or SimulationConfig()
        self.logger = get_logger(f"simulation.{name.lower()}")
        
        # Simulation state
        self.state = SimulationState.UNINITIALIZED
        self.start_time = 0.0
        self.last_frame_time = 0.0
        
        # Statistics
        self.stats = SimulationStats()
        
        # Core systems
        self.physics_world: Optional[PhysicsWorld] = None
        self.render_engine: Optional[RenderEngine] = None
        self.camera: Optional[Camera2D] = None
        self.camera_controller: Optional[CameraController] = None
        
        # Input system
        self.joystick_manager = get_joystick_manager()
        self.input_processor = get_input_processor()
        
        # Event handling
        self.pygame_events: List[pygame.event.Event] = []
        self.input_events: List[InputEvent] = []
        
        # Lifecycle callbacks
        self.pre_update_callbacks: List[Callable[[float], None]] = []
        self.post_update_callbacks: List[Callable[[float], None]] = []
        self.pre_render_callbacks: List[Callable[[RenderEngine], None]] = []
        self.post_render_callbacks: List[Callable[[RenderEngine], None]] = []
        
        # Performance monitoring
        self._last_performance_log = 0.0
        self._performance_log_interval = 5.0  # Log every 5 seconds
        
        self.logger.debug("BaseSimulation created", extra={
            "name": name,
            "config": self.config.__dict__
        })
    
    def initialize(self) -> bool:
        """
        Initialize the simulation.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.state != SimulationState.UNINITIALIZED:
            self.logger.warning("Simulation already initialized")
            return True
        
        self.state = SimulationState.INITIALIZING
        
        try:
            # Initialize rendering system
            if not self._initialize_rendering():
                raise SimulationError("Failed to initialize rendering system")
            
            # Initialize physics system
            if not self._initialize_physics():
                raise SimulationError("Failed to initialize physics system")
            
            # Initialize camera system
            if not self._initialize_camera():
                raise SimulationError("Failed to initialize camera system")
            
            # Initialize input system
            if not self._initialize_input():
                raise SimulationError("Failed to initialize input system")
            
            # Call simulation-specific initialization
            if not self.on_initialize():
                raise SimulationError("Simulation-specific initialization failed")
            
            self.state = SimulationState.READY
            
            self.logger.info("Simulation initialized successfully", extra={
                "name": self.name
            })
            
            return True
            
        except Exception as e:
            self.state = SimulationState.ERROR
            self.logger.error("Failed to initialize simulation", extra={
                "name": self.name,
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    def run(self) -> None:
        """Run the simulation main loop."""
        if self.state != SimulationState.READY:
            if not self.initialize():
                raise SimulationError("Cannot run simulation: initialization failed")
        
        self.logger.info("Starting simulation", extra={"name": self.name})
        
        self.state = SimulationState.RUNNING
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        # Call simulation start callback
        self.on_start()
        
        try:
            while self.state == SimulationState.RUNNING:
                current_time = time.time()
                dt = current_time - self.last_frame_time
                self.last_frame_time = current_time
                
                # Update simulation
                self._update(dt)
                
                # Render frame
                self._render()
                
                # Handle events
                if not self._handle_events():
                    break
                
                # Performance logging
                if self.config.enable_performance_monitoring:
                    self._log_performance_if_needed()
            
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        except Exception as e:
            self.state = SimulationState.ERROR
            self.logger.error("Simulation error", extra={
                "name": self.name,
                "error": str(e),
                "error_type": type(e).__name__
            })
        finally:
            self._shutdown()
    
    def stop(self) -> None:
        """Stop the simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.STOPPING
            self.logger.info("Simulation stop requested", extra={"name": self.name})
    
    def pause(self) -> None:
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            if self.physics_world:
                self.physics_world.pause()
            self.logger.info("Simulation paused", extra={"name": self.name})
    
    def resume(self) -> None:
        """Resume the simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            if self.physics_world:
                self.physics_world.resume()
            self.logger.info("Simulation resumed", extra={"name": self.name})
    
    def _initialize_rendering(self) -> bool:
        """Initialize the rendering system."""
        try:
            self.render_engine = create_render_engine(RenderEngineType.RENDERER_2D)
            
            success = self.render_engine.initialize(
                self.config.window_width,
                self.config.window_height,
                self.config.window_title
            )
            
            if success:
                self.render_engine.set_background_color(StandardColors.DARK_GRAY)
                self.logger.debug("Rendering system initialized")
            
            return success
            
        except Exception as e:
            self.logger.error("Failed to initialize rendering", extra={"error": str(e)})
            return False
    
    def _initialize_physics(self) -> bool:
        """Initialize the physics system."""
        try:
            self.physics_world = create_physics_world(
                PhysicsEngineType.PHYSICS_2D,
                Vector2D(0, -9.81)  # Earth gravity
            )
            
            self.physics_world.set_time_step(1.0 / self.config.physics_fps)
            self.physics_world.start_simulation()
            
            self.logger.debug("Physics system initialized")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize physics", extra={"error": str(e)})
            return False
    
    def _initialize_camera(self) -> bool:
        """Initialize the camera system."""
        try:
            self.camera = Camera2D(
                self.config.window_width,
                self.config.window_height
            )
            
            self.camera_controller = CameraController(self.camera)
            
            self.logger.debug("Camera system initialized")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize camera", extra={"error": str(e)})
            return False
    
    def _initialize_input(self) -> bool:
        """Initialize the input system."""
        try:
            # Initialize joystick manager if not already done
            if not self.joystick_manager.is_initialized():
                if not self.joystick_manager.initialize():
                    self.logger.warning("Failed to initialize joystick manager")
            
            # Register input event callback
            self.input_processor.add_event_callback(self._on_input_event)
            
            self.logger.debug("Input system initialized")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize input", extra={"error": str(e)})
            return False
    
    def _update(self, dt: float) -> None:
        """Update simulation systems."""
        if self.state != SimulationState.RUNNING:
            return
        
        update_start = time.time()
        
        # Call pre-update callbacks
        for callback in self.pre_update_callbacks:
            try:
                callback(dt)
            except Exception as e:
                self.logger.error("Error in pre-update callback", extra={"error": str(e)})
        
        # Update input system
        self.joystick_manager.update()
        
        # Process joystick input
        for joystick_id in self.joystick_manager.get_all_joysticks().keys():
            input_state = self.joystick_manager.get_input_state(joystick_id)
            if input_state:
                events = self.input_processor.process_input(joystick_id, input_state)
                # Events are handled via callbacks
        
        # Update camera
        if self.camera:
            self.camera.update(dt)
        
        if self.camera_controller:
            self.camera_controller.update(dt)
        
        # Update physics
        physics_start = time.time()
        if self.physics_world:
            self.physics_world.step(dt)
        physics_time = time.time() - physics_start
        
        # Call simulation-specific update
        self.on_update(dt)
        
        # Call post-update callbacks
        for callback in self.post_update_callbacks:
            try:
                callback(dt)
            except Exception as e:
                self.logger.error("Error in post-update callback", extra={"error": str(e)})
        
        # Update statistics
        update_time = time.time() - update_start
        self.stats.update_frame_stats(dt, physics_time, 0.0)  # Render time updated in _render
        self.stats.physics_steps += 1
    
    def _render(self) -> None:
        """Render the simulation."""
        if not self.render_engine or self.state != SimulationState.RUNNING:
            return
        
        render_start = time.time()
        
        # Begin frame
        self.render_engine.begin_frame()
        
        # Clear screen
        self.render_engine.clear()
        
        # Call pre-render callbacks
        for callback in self.pre_render_callbacks:
            try:
                callback(self.render_engine)
            except Exception as e:
                self.logger.error("Error in pre-render callback", extra={"error": str(e)})
        
        # Call simulation-specific rendering
        self.on_render(self.render_engine)
        
        # Render debug information if enabled
        if self.config.enable_debug_draw:
            self._render_debug_info()
        
        # Call post-render callbacks
        for callback in self.post_render_callbacks:
            try:
                callback(self.render_engine)
            except Exception as e:
                self.logger.error("Error in post-render callback", extra={"error": str(e)})
        
        # End frame
        self.render_engine.end_frame()
        
        # Update render time
        render_time = time.time() - render_start
        self.stats.render_time_ms = render_time * 1000
    
    def _handle_events(self) -> bool:
        """
        Handle system events.
        
        Returns:
            False if simulation should exit, True otherwise
        """
        # Get pygame events
        self.pygame_events = pygame.event.get()
        
        for event in self.pygame_events:
            if event.type == pygame.QUIT:
                self.stop()
                return False
            
            elif event.type == pygame.KEYDOWN:
                if not self._handle_key_down(event.key):
                    return False
            
            elif event.type == pygame.KEYUP:
                self._handle_key_up(event.key)
            
            # Call simulation-specific event handling
            if not self.on_event(event):
                return False
        
        return True
    
    def _handle_key_down(self, key: int) -> bool:
        """
        Handle key down events.
        
        Returns:
            False if simulation should exit, True otherwise
        """
        # Default key bindings
        if key == pygame.K_ESCAPE:
            self.stop()
            return False
        
        elif key == pygame.K_SPACE:
            if self.state == SimulationState.RUNNING:
                self.pause()
            elif self.state == SimulationState.PAUSED:
                self.resume()
        
        # Camera controls
        elif key == pygame.K_w:
            if self.camera_controller:
                self.camera_controller.move_up = True
        elif key == pygame.K_s:
            if self.camera_controller:
                self.camera_controller.move_down = True
        elif key == pygame.K_a:
            if self.camera_controller:
                self.camera_controller.move_left = True
        elif key == pygame.K_d:
            if self.camera_controller:
                self.camera_controller.move_right = True
        elif key == pygame.K_q:
            if self.camera_controller:
                self.camera_controller.zoom_out = True
        elif key == pygame.K_e:
            if self.camera_controller:
                self.camera_controller.zoom_in = True
        elif key == pygame.K_r:
            if self.camera_controller:
                self.camera_controller.reset_to_origin()
        
        return True
    
    def _handle_key_up(self, key: int) -> None:
        """Handle key up events."""
        # Camera controls
        if key == pygame.K_w:
            if self.camera_controller:
                self.camera_controller.move_up = False
        elif key == pygame.K_s:
            if self.camera_controller:
                self.camera_controller.move_down = False
        elif key == pygame.K_a:
            if self.camera_controller:
                self.camera_controller.move_left = False
        elif key == pygame.K_d:
            if self.camera_controller:
                self.camera_controller.move_right = False
        elif key == pygame.K_q:
            if self.camera_controller:
                self.camera_controller.zoom_out = False
        elif key == pygame.K_e:
            if self.camera_controller:
                self.camera_controller.zoom_in = False
    
    def _on_input_event(self, event: InputEvent) -> None:
        """Handle input events from joysticks."""
        self.input_events.append(event)
        self.stats.input_events_processed += 1
        
        # Call simulation-specific input handling
        self.on_input_event(event)
    
    def _render_debug_info(self) -> None:
        """Render debug information overlay."""
        if not self.render_engine:
            return
        
        # Render statistics
        y_offset = 10
        line_height = 20
        
        debug_info = [
            f"FPS: {self.stats.average_fps:.1f}",
            f"Frame: {self.stats.frame_time_ms:.1f}ms",
            f"Physics: {self.stats.physics_time_ms:.1f}ms", 
            f"Render: {self.stats.render_time_ms:.1f}ms",
            f"Objects: {self.physics_world.get_object_count() if self.physics_world else 0}",
            f"Input Events: {self.stats.input_events_processed}",
        ]
        
        if self.camera:
            debug_info.extend([
                f"Camera: ({self.camera.position.x:.1f}, {self.camera.position.y:.1f})",
                f"Zoom: {self.camera.zoom:.2f}",
                f"State: {self.state.value}",
            ])
        
        for i, text in enumerate(debug_info):
            self.render_engine.draw_text(
                text,
                Vector2D(10, y_offset + i * line_height),
                StandardColors.WHITE,
                font_size=14
            )
    
    def _log_performance_if_needed(self) -> None:
        """Log performance statistics periodically."""
        current_time = time.time()
        if (current_time - self._last_performance_log) >= self._performance_log_interval:
            self.logger.debug("Simulation performance", extra={
                "name": self.name,
                "fps": self.stats.average_fps,
                "frame_time_ms": self.stats.frame_time_ms,
                "physics_time_ms": self.stats.physics_time_ms,
                "render_time_ms": self.stats.render_time_ms,
                "objects": self.physics_world.get_object_count() if self.physics_world else 0,
                "input_events": self.stats.input_events_processed
            })
            self._last_performance_log = current_time
    
    def _shutdown(self) -> None:
        """Shutdown simulation systems."""
        self.logger.info("Shutting down simulation", extra={"name": self.name})
        
        self.state = SimulationState.STOPPING
        
        # Call simulation-specific shutdown
        self.on_shutdown()
        
        # Shutdown systems
        if self.physics_world:
            self.physics_world.shutdown()
        
        if self.render_engine:
            self.render_engine.shutdown()
        
        self.state = SimulationState.STOPPED
        
        # Log final statistics
        total_time = time.time() - self.start_time
        self.logger.info("Simulation completed", extra={
            "name": self.name,
            "total_runtime": total_time,
            "total_frames": self.stats.frame_count,
            "average_fps": self.stats.average_fps,
            "input_events_processed": self.stats.input_events_processed
        })
    
    # Abstract methods for simulation-specific implementation
    
    @abstractmethod
    def on_initialize(self) -> bool:
        """
        Simulation-specific initialization.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def on_start(self) -> None:
        """Called when simulation starts running."""
        pass
    
    @abstractmethod
    def on_update(self, dt: float) -> None:
        """
        Simulation-specific update logic.
        
        Args:
            dt: Time delta in seconds
        """
        pass
    
    @abstractmethod
    def on_render(self, renderer: RenderEngine) -> None:
        """
        Simulation-specific rendering.
        
        Args:
            renderer: Rendering engine
        """
        pass
    
    @abstractmethod
    def on_shutdown(self) -> None:
        """Simulation-specific cleanup."""
        pass
    
    def on_event(self, event: pygame.event.Event) -> bool:
        """
        Handle pygame events.
        
        Args:
            event: Pygame event
            
        Returns:
            False if simulation should exit, True otherwise
        """
        return True
    
    def on_input_event(self, event: InputEvent) -> None:
        """
        Handle input events.
        
        Args:
            event: Input event from joystick
        """
        pass
    
    # Utility methods for simulations
    
    def add_pre_update_callback(self, callback: Callable[[float], None]) -> None:
        """Add a pre-update callback."""
        self.pre_update_callbacks.append(callback)
    
    def add_post_update_callback(self, callback: Callable[[float], None]) -> None:
        """Add a post-update callback."""
        self.post_update_callbacks.append(callback)
    
    def add_pre_render_callback(self, callback: Callable[[RenderEngine], None]) -> None:
        """Add a pre-render callback."""
        self.pre_render_callbacks.append(callback)
    
    def add_post_render_callback(self, callback: Callable[[RenderEngine], None]) -> None:
        """Add a post-render callback."""
        self.post_render_callbacks.append(callback)
    
    def world_to_screen(self, world_pos: Vector2D) -> Vector2D:
        """Convert world coordinates to screen coordinates."""
        if self.camera:
            return self.camera.world_to_screen(world_pos)
        else:
            # Fallback to simple conversion
            return Vector2D(
                world_pos.x + self.config.window_width // 2,
                self.config.window_height // 2 - world_pos.y
            )
    
    def screen_to_world(self, screen_pos: Vector2D) -> Vector2D:
        """Convert screen coordinates to world coordinates."""
        if self.camera:
            return self.camera.screen_to_world(screen_pos)
        else:
            # Fallback to simple conversion
            return Vector2D(
                screen_pos.x - self.config.window_width // 2,
                self.config.window_height // 2 - screen_pos.y
            )