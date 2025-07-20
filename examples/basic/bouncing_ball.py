#!/usr/bin/env python3
"""
Simple Bouncing Ball Demo

A minimal demonstration of the PyJoySim framework featuring:
- A single bouncing ball in a bounded world
- Basic physics simulation with gravity and collisions
- Simple 2D rendering
- Joystick input to control the ball

This demo showcases the core functionality in a simple, easy-to-understand way.

Usage:
    python examples/basic/bouncing_ball.py
"""

import sys
import math
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pyjoysim.simulation import (
    BaseSimulation, SimulationConfig, SimulationCategory, 
    SimulationMetadata, register_simulation
)
from pyjoysim.physics import Vector2D, create_ball, create_static_wall, StandardMaterials
from pyjoysim.rendering import StandardColors
from pyjoysim.input import InputEvent, InputEventType
from pyjoysim.core.logging import get_logger


# Simple bouncing ball metadata
BOUNCING_BALL_METADATA = SimulationMetadata(
    name="bouncing_ball",
    display_name="Simple Bouncing Ball",
    description="A simple physics demonstration with a bouncing ball",
    category=SimulationCategory.DEMO,
    author="PyJoySim Team",
    version="1.0",
    difficulty="Beginner",
    tags=["demo", "physics", "simple", "ball"],
    requirements=["pygame", "pymunk"]
)


@register_simulation(BOUNCING_BALL_METADATA)
class BouncingBallSimulation(BaseSimulation):
    """
    Simple bouncing ball simulation.
    
    Features:
    - Single ball with physics
    - Bounded world with walls
    - Joystick control for ball movement
    - Basic collision detection
    """
    
    def __init__(self, name: str = "bouncing_ball", config: SimulationConfig = None):
        """Initialize the bouncing ball simulation."""
        if config is None:
            config = SimulationConfig(
                window_title="PyJoySim - Bouncing Ball Demo",
                window_width=800,
                window_height=600,
                enable_debug_draw=True,
                enable_performance_monitoring=False
            )
        
        super().__init__(name, config)
        
        # Simulation objects
        self.ball = None
        self.walls = {}
        
        # Control settings
        self.force_multiplier = 300.0
        
        self.logger.info("Bouncing ball simulation created")
    
    def on_initialize(self) -> bool:
        """Initialize the bouncing ball demo."""
        try:
            self.logger.info("Initializing bouncing ball demo")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize bouncing ball demo", extra={
                "error": str(e)
            })
            return False
    
    def on_start(self) -> None:
        """Set up the bouncing ball scene."""
        self.logger.info("Setting up bouncing ball scene")
        
        # Create world boundaries
        self._create_boundaries()
        
        # Create the bouncing ball
        self._create_ball()
        
        # Set up camera
        self._setup_camera()
        
        self.logger.info("Bouncing ball scene setup complete")
    
    def _create_boundaries(self) -> None:
        """Create world boundaries."""
        # Ground
        ground, _ = create_static_wall(
            self.physics_world.engine,
            Vector2D(0, -4),
            width=16.0,
            height=1.0,
            material=StandardMaterials.CONCRETE,
            name="Ground"
        )
        self.physics_world.add_object(ground, "boundaries")
        self.walls["ground"] = ground
        
        # Left wall
        left_wall, _ = create_static_wall(
            self.physics_world.engine,
            Vector2D(-8, 0),
            width=1.0,
            height=10.0,
            material=StandardMaterials.CONCRETE,
            name="LeftWall"
        )
        self.physics_world.add_object(left_wall, "boundaries")
        self.walls["left"] = left_wall
        
        # Right wall
        right_wall, _ = create_static_wall(
            self.physics_world.engine,
            Vector2D(8, 0),
            width=1.0,
            height=10.0,
            material=StandardMaterials.CONCRETE,
            name="RightWall"
        )
        self.physics_world.add_object(right_wall, "boundaries")
        self.walls["right"] = right_wall
        
        # Ceiling
        ceiling, _ = create_static_wall(
            self.physics_world.engine,
            Vector2D(0, 6),
            width=16.0,
            height=1.0,
            material=StandardMaterials.CONCRETE,
            name="Ceiling"
        )
        self.physics_world.add_object(ceiling, "boundaries")
        self.walls["ceiling"] = ceiling
    
    def _create_ball(self) -> None:
        """Create the main bouncing ball."""
        self.ball, _ = create_ball(
            self.physics_world.engine,
            Vector2D(0, 2),  # Start in center, above ground
            radius=0.5,
            mass=1.0,
            material=StandardMaterials.BOUNCY,
            name="BouncingBall"
        )
        self.physics_world.add_object(self.ball, "ball")
        
        # Give initial velocity for some action
        self.ball.apply_impulse(Vector2D(50, 100))
    
    def _setup_camera(self) -> None:
        """Set up camera to show the entire scene."""
        if self.camera:
            # Position camera to show the bounded area
            self.camera.set_position(Vector2D(0, 1))
            self.camera.set_zoom(30.0)  # Zoom to show good detail
    
    def on_update(self, dt: float) -> None:
        """Update bouncing ball logic."""
        # Keep ball in bounds (safety check)
        if self.ball and self.ball.is_active:
            pos = self.ball.position
            
            # If ball somehow gets outside bounds, reset it
            if pos.y < -10 or pos.y > 15 or abs(pos.x) > 15:
                self.ball.set_position(Vector2D(0, 2))
                self.ball.set_velocity(Vector2D(0, 0))
                self.logger.debug("Ball reset to center")
    
    def on_render(self, renderer) -> None:
        """Render the bouncing ball scene."""
        # Render walls
        self._render_walls(renderer)
        
        # Render ball
        self._render_ball(renderer)
        
        # Render simple instructions
        self._render_instructions(renderer)
    
    def _render_walls(self, renderer) -> None:
        """Render the boundary walls."""
        for wall_name, wall in self.walls.items():
            if not wall.is_active:
                continue
            
            screen_pos = self.world_to_screen(wall.position)
            
            # Determine wall dimensions
            if wall_name == "ground" or wall_name == "ceiling":
                width = 16.0
                height = 1.0
            else:  # left/right walls
                width = 1.0
                height = 10.0
            
            scale = self.camera.zoom if self.camera else 20
            
            renderer.draw_rectangle(
                screen_pos,
                width * scale,
                height * scale,
                StandardColors.GRAY,
                fill=True
            )
            
            # Add outline
            renderer.draw_rectangle(
                screen_pos,
                width * scale,
                height * scale,
                StandardColors.BLACK,
                fill=False,
                line_width=2
            )
    
    def _render_ball(self, renderer) -> None:
        """Render the bouncing ball."""
        if not self.ball or not self.ball.is_active:
            return
        
        screen_pos = self.world_to_screen(self.ball.position)
        radius = 0.5
        scale = self.camera.zoom if self.camera else 20
        
        # Main ball
        renderer.draw_circle(
            screen_pos,
            radius * scale,
            StandardColors.RED,
            fill=True
        )
        
        # Ball outline
        renderer.draw_circle(
            screen_pos,
            radius * scale,
            StandardColors.BLACK,
            fill=False,
            width=2
        )
        
        # Highlight to show it's interactive
        renderer.draw_circle(
            screen_pos,
            radius * scale * 1.3,
            StandardColors.YELLOW.with_alpha(100),
            fill=False,
            width=3
        )
    
    def _render_instructions(self, renderer) -> None:
        """Render simple control instructions."""
        instructions = [
            "Bouncing Ball Demo",
            "",
            "Controls:",
            "Joystick: Move ball",
            "A Button: Jump",
            "WASD: Move camera",
            "Q/E: Zoom",
            "Space: Pause",
            "ESC: Exit"
        ]
        
        y_start = 20
        line_height = 20
        
        for i, text in enumerate(instructions):
            if text:  # Skip empty lines for spacing
                renderer.draw_text(
                    text,
                    Vector2D(20, y_start + i * line_height),
                    StandardColors.WHITE if text != "Bouncing Ball Demo" else StandardColors.CYAN,
                    font_size=16 if text == "Bouncing Ball Demo" else 12
                )
    
    def on_input_event(self, event: InputEvent) -> None:
        """Handle joystick input events."""
        if not self.ball or not self.ball.is_active:
            return
        
        if event.event_type == InputEventType.AXIS_CHANGE:
            # Use left stick for movement
            if event.axis_id == 0:  # X axis
                force = Vector2D(event.axis_value * self.force_multiplier, 0)
                self.ball.apply_force(force)
            
            elif event.axis_id == 1:  # Y axis
                # Invert Y axis (joystick up = positive force up)
                force = Vector2D(0, -event.axis_value * self.force_multiplier)
                self.ball.apply_force(force)
        
        elif event.event_type == InputEventType.BUTTON_PRESS:
            if event.button_id == 0:  # A button - jump/boost up
                self.ball.apply_impulse(Vector2D(0, 200))
            
            elif event.button_id == 1:  # B button - reset ball
                self.ball.set_position(Vector2D(0, 2))
                self.ball.set_velocity(Vector2D(0, 0))
                self.logger.info("Ball reset to center")
    
    def on_event(self, event) -> bool:
        """Handle keyboard events."""
        if hasattr(event, 'type') and hasattr(event, 'key'):
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset ball
                    if self.ball:
                        self.ball.set_position(Vector2D(0, 2))
                        self.ball.set_velocity(Vector2D(0, 0))
                        self.logger.info("Ball reset to center")
                    return True
                
                elif event.key == pygame.K_j:  # Jump with keyboard
                    if self.ball:
                        self.ball.apply_impulse(Vector2D(0, 200))
                    return True
        
        return True
    
    def on_shutdown(self) -> None:
        """Clean up bouncing ball resources."""
        self.logger.info("Cleaning up bouncing ball demo")
        
        # Clean up objects
        if self.ball and self.ball.is_active:
            self.ball.destroy()
        
        for wall in self.walls.values():
            if wall.is_active:
                wall.destroy()
        
        self.walls.clear()


def main():
    """Main entry point for the bouncing ball demo."""
    # Create simulation config
    config = SimulationConfig(
        window_title="PyJoySim - Simple Bouncing Ball",
        window_width=800,
        window_height=600,
        enable_debug_draw=True,
        enable_performance_monitoring=False
    )
    
    # Create and run simulation
    demo = BouncingBallSimulation(config=config)
    
    try:
        print("\nPyJoySim Simple Bouncing Ball Demo")
        print("=" * 40)
        print("A minimal physics simulation demonstrating:")
        print("- Ball physics with gravity and bouncing")
        print("- Bounded world with collision detection")
        print("- Joystick and keyboard controls")
        print("- Basic 2D rendering")
        print()
        print("Controls:")
        print("  Joystick    - Move ball around")
        print("  A Button    - Jump/boost up")
        print("  B Button    - Reset ball position")
        print("  R           - Reset ball (keyboard)")
        print("  J           - Jump (keyboard)")
        print("  WASD        - Move camera")
        print("  Q/E         - Zoom in/out")
        print("  Space       - Pause/resume")
        print("  ESC         - Exit")
        print("=" * 40)
        print()
        
        # Run the simulation
        demo.run()
        
        print("Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Demo error: {e}")
        return 1


if __name__ == "__main__":
    import pygame  # Need this for event handling
    sys.exit(main())