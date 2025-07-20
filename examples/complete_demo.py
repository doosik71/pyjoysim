#!/usr/bin/env python3
"""
PyJoySim Complete System Demo

This script demonstrates the complete PyJoySim framework including:
- Input processing with joystick support
- Physics simulation with collision detection
- 2D rendering with camera controls
- UI overlays and debug information
- Simulation lifecycle management

Usage:
    python examples/complete_demo.py [--duration SECONDS] [--disable-ui]
"""

import sys
import argparse
import math
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyjoysim.simulation import (
    BaseSimulation, SimulationConfig, SimulationCategory, 
    SimulationMetadata, register_simulation
)
from pyjoysim.physics import (
    Vector2D, create_ball, create_box, create_static_wall,
    StandardMaterials, PinJoint, SpringJoint
)
from pyjoysim.rendering import StandardColors
from pyjoysim.input import InputEvent, InputEventType
from pyjoysim.ui import OverlayManager, OverlayPosition
from pyjoysim.core.logging import get_logger


# Demo simulation metadata
COMPLETE_DEMO_METADATA = SimulationMetadata(
    name="complete_demo",
    display_name="Complete System Demo",
    description="Comprehensive demonstration of all PyJoySim systems",
    category=SimulationCategory.DEMO,
    author="PyJoySim Team",
    version="1.0",
    difficulty="Beginner",
    tags=["demo", "physics", "input", "rendering", "ui"],
    requirements=["pygame", "pymunk"]
)


@register_simulation(COMPLETE_DEMO_METADATA)
class CompleteDemoSimulation(BaseSimulation):
    """
    Complete demonstration simulation showcasing all PyJoySim features.
    
    Features demonstrated:
    - Physics objects (balls, boxes, walls, joints)
    - Joystick input handling with camera control
    - Real-time rendering with camera system
    - Debug overlays and performance monitoring
    - Interactive controls and visual feedback
    """
    
    def __init__(self, name: str = "complete_demo", config: SimulationConfig = None):
        """Initialize the complete demo simulation."""
        if config is None:
            config = SimulationConfig(
                window_title="PyJoySim - Complete System Demo",
                enable_debug_draw=True,
                enable_performance_monitoring=True
            )
        
        super().__init__(name, config)
        
        # Demo objects
        self.demo_objects = {}
        self.demo_constraints = {}
        
        # UI overlay manager
        self.overlay_manager: OverlayManager = None
        
        # Demo state
        self.ball_spawn_timer = 0.0
        self.ball_spawn_interval = 3.0
        self.ball_count = 0
        self.max_balls = 10
        
        # Input tracking
        self.joystick_controlled_object = None
        self.force_multiplier = 500.0
        
        self.logger.info("Complete demo simulation created")
    
    def on_initialize(self) -> bool:
        """Initialize demo-specific systems."""
        try:
            # Initialize overlay manager
            self.overlay_manager = OverlayManager(
                self.config.window_width,
                self.config.window_height
            )
            
            # Enable all debug overlays
            self.overlay_manager.enable_performance_monitor(OverlayPosition.TOP_LEFT)
            self.overlay_manager.enable_physics_debug(OverlayPosition.TOP_RIGHT)
            self.overlay_manager.enable_input_debug(OverlayPosition.BOTTOM_LEFT)
            
            # Create custom info panel
            info_panel = self.overlay_manager.create_custom_panel(
                "demo_info",
                "Demo Controls",
                OverlayPosition.BOTTOM_RIGHT
            )
            
            # Add control instructions
            info_panel.set_info({
                "WASD": "Move Camera",
                "Q/E": "Zoom In/Out",
                "R": "Reset Camera",
                "Space": "Pause/Resume",
                "F1-F4": "Toggle Overlays",
                "JS Left": "Control Ball",
                "JS Buttons": "Special Actions"
            })
            
            # Register overlay input handling
            self.add_pre_render_callback(self._update_overlays)
            self.add_post_render_callback(self._render_overlays)
            
            self.logger.info("Demo-specific systems initialized")
            return True
            
        except Exception as e:
            self.logger.error("Failed to initialize demo systems", extra={
                "error": str(e)
            })
            return False
    
    def on_start(self) -> None:
        """Set up the demo scene."""
        self.logger.info("Setting up demo scene")
        
        # Create boundaries
        self._create_boundaries()
        
        # Create interactive objects
        self._create_interactive_objects()
        
        # Create mechanical systems
        self._create_mechanical_systems()
        
        # Set up camera
        self._setup_camera()
        
        self.logger.info("Demo scene setup complete")
    
    def _create_boundaries(self) -> None:
        """Create world boundaries."""
        # Ground
        ground, _ = create_static_wall(
            self.physics_world.engine,
            Vector2D(0, -8),
            width=30.0,
            height=2.0,
            material=StandardMaterials.CONCRETE,
            name="Ground"
        )
        self.physics_world.add_object(ground, "boundaries")
        self.demo_objects["ground"] = ground
        
        # Left wall
        left_wall, _ = create_static_wall(
            self.physics_world.engine,
            Vector2D(-15, 0),
            width=2.0,
            height=20.0,
            material=StandardMaterials.CONCRETE,
            name="LeftWall"
        )
        self.physics_world.add_object(left_wall, "boundaries")
        self.demo_objects["left_wall"] = left_wall
        
        # Right wall
        right_wall, _ = create_static_wall(
            self.physics_world.engine,
            Vector2D(15, 0),
            width=2.0,
            height=20.0,
            material=StandardMaterials.CONCRETE,
            name="RightWall"
        )
        self.physics_world.add_object(right_wall, "boundaries")
        self.demo_objects["right_wall"] = right_wall
        
        # Ceiling (partial)
        ceiling, _ = create_static_wall(
            self.physics_world.engine,
            Vector2D(0, 12),
            width=10.0,
            height=1.0,
            material=StandardMaterials.CONCRETE,
            name="Ceiling"
        )
        self.physics_world.add_object(ceiling, "boundaries")
        self.demo_objects["ceiling"] = ceiling
    
    def _create_interactive_objects(self) -> None:
        """Create interactive physics objects."""
        # Player-controlled ball
        player_ball, _ = create_ball(
            self.physics_world.engine,
            Vector2D(-8, 5),
            radius=0.8,
            mass=2.0,
            material=StandardMaterials.RUBBER,
            name="PlayerBall"
        )
        self.physics_world.add_object(player_ball, "interactive")
        self.demo_objects["player_ball"] = player_ball
        self.joystick_controlled_object = player_ball
        
        # Bouncy balls
        for i in range(3):
            ball, _ = create_ball(
                self.physics_world.engine,
                Vector2D(-5 + i * 2.5, 8),
                radius=0.6,
                mass=1.0,
                material=StandardMaterials.BOUNCY,
                name=f"BouncyBall{i}"
            )
            self.physics_world.add_object(ball, "interactive")
            self.demo_objects[f"bouncy_ball_{i}"] = ball
            
            # Give balls initial velocity
            ball.apply_impulse(Vector2D((i - 1) * 100, 50))
        
        # Heavy boxes
        for i in range(2):
            box, _ = create_box(
                self.physics_world.engine,
                Vector2D(5 + i * 3, 2),
                width=1.5,
                height=1.5,
                mass=5.0,
                material=StandardMaterials.WOOD,
                name=f"HeavyBox{i}"
            )
            self.physics_world.add_object(box, "interactive")
            self.demo_objects[f"heavy_box_{i}"] = box
    
    def _create_mechanical_systems(self) -> None:
        """Create mechanical systems (pendulums, springs)."""
        # Pendulum system
        pendulum_anchor, _ = create_ball(
            self.physics_world.engine,
            Vector2D(8, 10),
            radius=0.2,
            mass=1000.0,  # Very heavy anchor
            material=StandardMaterials.METAL,
            name="PendulumAnchor"
        )
        self.physics_world.add_object(pendulum_anchor, "mechanical")
        
        pendulum_bob, _ = create_ball(
            self.physics_world.engine,
            Vector2D(8, 6),
            radius=0.5,
            mass=2.0,
            material=StandardMaterials.METAL,
            name="PendulumBob"
        )
        self.physics_world.add_object(pendulum_bob, "mechanical")
        
        # Connect with pin joint
        pendulum_joint = PinJoint(
            self.physics_world.engine,
            pendulum_anchor,
            pendulum_bob,
            Vector2D(0, 0),
            Vector2D(0, 0),
            name="PendulumJoint"
        )
        self.physics_world.constraint_manager.add_constraint(pendulum_joint, "mechanical")
        
        self.demo_objects["pendulum_anchor"] = pendulum_anchor
        self.demo_objects["pendulum_bob"] = pendulum_bob
        self.demo_constraints["pendulum_joint"] = pendulum_joint
        
        # Give pendulum initial swing
        pendulum_bob.apply_impulse(Vector2D(300, 0))
        
        # Spring-mass system
        spring_anchor, _ = create_box(
            self.physics_world.engine,
            Vector2D(-10, 8),
            width=0.8,
            height=0.8,
            mass=1000.0,  # Heavy anchor
            material=StandardMaterials.METAL,
            name="SpringAnchor"
        )
        self.physics_world.add_object(spring_anchor, "mechanical")
        
        spring_mass, _ = create_ball(
            self.physics_world.engine,
            Vector2D(-10, 4),
            radius=0.6,
            mass=1.5,
            material=StandardMaterials.BOUNCY,
            name="SpringMass"
        )
        self.physics_world.add_object(spring_mass, "mechanical")
        
        # Connect with spring
        spring_joint = SpringJoint(
            self.physics_world.engine,
            spring_anchor,
            spring_mass,
            Vector2D(0, 0),
            Vector2D(0, 0),
            rest_length=3.0,
            spring_constant=800.0,
            damping=15.0,
            name="SpringJoint"
        )
        self.physics_world.constraint_manager.add_constraint(spring_joint, "mechanical")
        
        self.demo_objects["spring_anchor"] = spring_anchor
        self.demo_objects["spring_mass"] = spring_mass
        self.demo_constraints["spring_joint"] = spring_joint
        
        # Disturb spring system
        spring_mass.apply_impulse(Vector2D(100, 200))
    
    def _setup_camera(self) -> None:
        """Set up camera to follow the action."""
        if self.camera and self.joystick_controlled_object:
            # Follow the player ball with some smoothing
            self.camera.follow_object(
                self.joystick_controlled_object,
                smoothing=0.3,
                offset=Vector2D(0, 2)  # Offset camera slightly above
            )
            
            # Set reasonable zoom
            self.camera.set_zoom(25.0)  # Zoom in to see detail
    
    def on_update(self, dt: float) -> None:
        """Update demo-specific logic."""
        # Update ball spawning timer
        self.ball_spawn_timer += dt
        
        # Spawn new balls periodically
        if (self.ball_spawn_timer >= self.ball_spawn_interval and 
            self.ball_count < self.max_balls):
            self._spawn_random_ball()
            self.ball_spawn_timer = 0.0
        
        # Update overlay data
        if self.overlay_manager:
            self.overlay_manager.update(
                self.stats.average_fps,
                self.stats.frame_time_ms / 1000.0,
                self.physics_world
            )
    
    def _spawn_random_ball(self) -> None:
        """Spawn a random ball from above."""
        import random
        
        # Random spawn position
        x = random.uniform(-12, 12)
        y = random.uniform(10, 15)
        
        # Random properties
        radius = random.uniform(0.3, 0.8)
        mass = random.uniform(0.5, 2.0)
        
        # Random material
        materials = [
            StandardMaterials.RUBBER,
            StandardMaterials.BOUNCY,
            StandardMaterials.METAL,
            StandardMaterials.WOOD
        ]
        material = random.choice(materials)
        
        # Create ball
        ball, _ = create_ball(
            self.physics_world.engine,
            Vector2D(x, y),
            radius=radius,
            mass=mass,
            material=material,
            name=f"RandomBall{self.ball_count}"
        )
        
        self.physics_world.add_object(ball, "random")
        self.demo_objects[f"random_ball_{self.ball_count}"] = ball
        
        # Random initial velocity
        vel_x = random.uniform(-50, 50)
        vel_y = random.uniform(-20, 20)
        ball.apply_impulse(Vector2D(vel_x, vel_y))
        
        self.ball_count += 1
        self.logger.debug("Spawned random ball", extra={
            "position": (x, y),
            "radius": radius,
            "mass": mass,
            "material": material.name
        })
    
    def on_render(self, renderer) -> None:
        """Render demo-specific graphics."""
        # Render physics objects with custom colors
        self._render_physics_objects(renderer)
        
        # Render connection lines for joints
        self._render_joint_connections(renderer)
        
        # Render force indicators
        self._render_force_indicators(renderer)
    
    def _render_physics_objects(self, renderer) -> None:
        """Render physics objects with visual styling."""
        for obj_name, obj in self.demo_objects.items():
            if not obj.is_active:
                continue
            
            # Get screen position
            screen_pos = self.world_to_screen(obj.position)
            
            # Determine color based on object type
            if "player" in obj_name:
                color = StandardColors.CYAN
            elif "bouncy" in obj_name:
                color = StandardColors.GREEN
            elif "random" in obj_name:
                color = StandardColors.YELLOW
            elif "box" in obj_name:
                color = StandardColors.BROWN
            elif "anchor" in obj_name:
                color = StandardColors.GRAY
            elif "mass" in obj_name or "bob" in obj_name:
                color = StandardColors.ORANGE
            elif "wall" in obj_name or "ground" in obj_name or "ceiling" in obj_name:
                color = StandardColors.DARK_GRAY
            else:
                color = StandardColors.WHITE
            
            # Render based on object type (simplified - would need proper collider info)
            if "ball" in obj_name or "mass" in obj_name or "bob" in obj_name or "anchor" in obj_name:
                # Assume radius based on name
                radius = 0.8 if "player" in obj_name else 0.6
                if "anchor" in obj_name:
                    radius = 0.2
                elif "random" in obj_name:
                    radius = 0.5
                
                renderer.draw_circle(
                    screen_pos,
                    radius * self.camera.zoom if self.camera else radius * 20,
                    color,
                    fill=True
                )
                
                # Add outline
                renderer.draw_circle(
                    screen_pos,
                    radius * self.camera.zoom if self.camera else radius * 20,
                    StandardColors.BLACK,
                    fill=False,
                    width=2
                )
            
            else:  # Boxes and walls
                # Assume size based on name
                if "wall" in obj_name:
                    width = 2.0 if "left" in obj_name or "right" in obj_name else 30.0
                    height = 20.0 if "left" in obj_name or "right" in obj_name else 2.0
                elif "ceiling" in obj_name:
                    width = 10.0
                    height = 1.0
                elif "box" in obj_name:
                    width = height = 1.5
                else:
                    width = height = 1.0
                
                scale = self.camera.zoom if self.camera else 20
                renderer.draw_rectangle(
                    screen_pos,
                    width * scale,
                    height * scale,
                    color,
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
    
    def _render_joint_connections(self, renderer) -> None:
        """Render visual connections for joints."""
        # Pendulum rod
        if ("pendulum_anchor" in self.demo_objects and 
            "pendulum_bob" in self.demo_objects):
            anchor_pos = self.world_to_screen(self.demo_objects["pendulum_anchor"].position)
            bob_pos = self.world_to_screen(self.demo_objects["pendulum_bob"].position)
            
            renderer.draw_line(
                anchor_pos,
                bob_pos,
                StandardColors.GRAY,
                width=3
            )
        
        # Spring visualization
        if ("spring_anchor" in self.demo_objects and 
            "spring_mass" in self.demo_objects):
            anchor_pos = self.world_to_screen(self.demo_objects["spring_anchor"].position)
            mass_pos = self.world_to_screen(self.demo_objects["spring_mass"].position)
            
            # Draw spring as zigzag line
            self._draw_spring_line(renderer, anchor_pos, mass_pos)
    
    def _draw_spring_line(self, renderer, start_pos, end_pos, coils=8) -> None:
        """Draw a spring visualization between two points."""
        # Calculate spring parameters
        dx = end_pos.x - start_pos.x
        dy = end_pos.y - start_pos.y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 1:
            return
        
        # Unit vector
        ux = dx / length
        uy = dy / length
        
        # Perpendicular vector
        px = -uy
        py = ux
        
        # Draw spring coils
        amplitude = 10  # Spring width
        points = [start_pos]
        
        for i in range(1, coils + 1):
            t = i / (coils + 1)
            
            # Alternate left and right
            side = 1 if i % 2 == 1 else -1
            
            # Position along spring
            base_x = start_pos.x + t * dx
            base_y = start_pos.y + t * dy
            
            # Offset perpendicular
            spring_x = base_x + side * amplitude * px
            spring_y = base_y + side * amplitude * py
            
            points.append(Vector2D(spring_x, spring_y))
        
        points.append(end_pos)
        
        # Draw lines between points
        for i in range(len(points) - 1):
            renderer.draw_line(
                points[i],
                points[i + 1],
                StandardColors.RED,
                width=2
            )
    
    def _render_force_indicators(self, renderer) -> None:
        """Render force/velocity indicators."""
        # Show velocity of player ball
        if self.joystick_controlled_object and self.joystick_controlled_object.is_active:
            pos = self.world_to_screen(self.joystick_controlled_object.position)
            
            # Velocity visualization (simplified - would need actual velocity from physics)
            # For now, just show that this is the controlled object
            renderer.draw_circle(
                pos,
                30,  # Outer indicator
                StandardColors.CYAN.with_alpha(100),
                fill=False,
                width=3
            )
    
    def _update_overlays(self, renderer) -> None:
        """Update overlay data before rendering."""
        if self.overlay_manager:
            # Update overlay with simulation data
            info_panel = self.overlay_manager.get_element("demo_info")
            if info_panel:
                info_panel.add_info("Total Objects", self.physics_world.get_object_count())
                info_panel.add_info("Random Balls", self.ball_count)
                info_panel.add_info("Camera Zoom", f"{self.camera.zoom:.2f}" if self.camera else "N/A")
                info_panel.add_info("Camera Pos", 
                                  f"({self.camera.position.x:.1f}, {self.camera.position.y:.1f})" 
                                  if self.camera else "N/A")
    
    def _render_overlays(self, renderer) -> None:
        """Render overlay elements."""
        if self.overlay_manager:
            self.overlay_manager.render(
                renderer,
                self.physics_world,
                self.joystick_manager
            )
    
    def on_input_event(self, event: InputEvent) -> None:
        """Handle joystick input events."""
        # Pass to overlay manager
        if self.overlay_manager:
            self.overlay_manager.handle_input_event(event)
        
        # Control player ball with joystick
        if (self.joystick_controlled_object and 
            self.joystick_controlled_object.is_active):
            
            if event.event_type == InputEventType.AXIS_CHANGE:
                # Use left stick for movement
                if event.axis_id == 0:  # X axis
                    force = Vector2D(event.axis_value * self.force_multiplier, 0)
                    self.joystick_controlled_object.apply_force(force)
                
                elif event.axis_id == 1:  # Y axis
                    force = Vector2D(0, -event.axis_value * self.force_multiplier)  # Invert Y
                    self.joystick_controlled_object.apply_force(force)
            
            elif event.event_type == InputEventType.BUTTON_PRESS:
                if event.button_id == 0:  # A button - jump
                    self.joystick_controlled_object.apply_impulse(Vector2D(0, 500))
                
                elif event.button_id == 1:  # B button - shake camera
                    if self.camera:
                        self.camera.add_shake(10.0, 0.5)
                
                elif event.button_id == 2:  # X button - spawn ball
                    if self.ball_count < self.max_balls:
                        self._spawn_random_ball()
    
    def on_event(self, event) -> bool:
        """Handle pygame events."""
        if event.type == pygame.KEYDOWN:
            # Handle overlay key events
            if self.overlay_manager and self.overlay_manager.handle_key_event(event.key):
                return True
            
            # Demo-specific keys
            if event.key == pygame.K_b:  # Spawn ball
                if self.ball_count < self.max_balls:
                    self._spawn_random_ball()
                return True
            
            elif event.key == pygame.K_c:  # Clear random balls
                self._clear_random_balls()
                return True
            
            elif event.key == pygame.K_g:  # Toggle gravity
                self._toggle_gravity()
                return True
        
        return True
    
    def _clear_random_balls(self) -> None:
        """Clear all random balls."""
        to_remove = []
        for name, obj in self.demo_objects.items():
            if "random_ball" in name:
                obj.destroy()
                to_remove.append(name)
        
        for name in to_remove:
            del self.demo_objects[name]
        
        self.physics_world.remove_group("random")
        self.ball_count = 0
        
        self.logger.info("Cleared all random balls")
    
    def _toggle_gravity(self) -> None:
        """Toggle gravity on/off."""
        current_gravity = self.physics_world.get_gravity()
        
        if abs(current_gravity.y) > 0.1:
            # Turn off gravity
            self.physics_world.set_gravity(Vector2D(0, 0))
            self.logger.info("Gravity disabled")
        else:
            # Turn on gravity
            self.physics_world.set_gravity(Vector2D(0, -9.81))
            self.logger.info("Gravity enabled")
    
    def on_shutdown(self) -> None:
        """Clean up demo resources."""
        self.logger.info("Cleaning up complete demo")
        
        # Clear all demo objects
        for obj in self.demo_objects.values():
            if obj.is_active:
                obj.destroy()
        
        self.demo_objects.clear()
        self.demo_constraints.clear()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PyJoySim Complete System Demo")
    parser.add_argument("--duration", type=float, default=0,
                       help="Demo duration in seconds (0 = run until exit)")
    parser.add_argument("--disable-ui", action="store_true",
                       help="Disable UI overlays")
    
    args = parser.parse_args()
    
    # Create simulation config
    config = SimulationConfig(
        window_title="PyJoySim - Complete System Demo",
        enable_debug_draw=not args.disable_ui,
        enable_performance_monitoring=True,
        window_width=1024,
        window_height=768
    )
    
    # Create and run simulation
    demo = CompleteDemoSimulation(config=config)
    
    try:
        print("\nPyJoySim Complete System Demo")
        print("=" * 50)
        print("Controls:")
        print("  WASD      - Move camera")
        print("  Q/E       - Zoom in/out")
        print("  R         - Reset camera")
        print("  Space     - Pause/resume")
        print("  F1-F4     - Toggle debug overlays")
        print("  B         - Spawn random ball")
        print("  C         - Clear random balls")
        print("  G         - Toggle gravity")
        print("  ESC       - Exit")
        print()
        print("Joystick Controls:")
        print("  Left Stick - Control blue ball")
        print("  A Button   - Jump")
        print("  B Button   - Shake camera")
        print("  X Button   - Spawn ball")
        print("=" * 50)
        
        if args.duration > 0:
            print(f"Running for {args.duration} seconds...")
            
            # Set up timer to stop simulation
            import threading
            def stop_after_duration():
                time.sleep(args.duration)
                demo.stop()
            
            timer_thread = threading.Thread(target=stop_after_duration)
            timer_thread.start()
        
        # Run the simulation
        demo.run()
        
        print("\nDemo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Demo error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())