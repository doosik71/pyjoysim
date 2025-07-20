#!/usr/bin/env python3
"""
3D Simulation Demo for PyJoySim Phase 3.

This demo showcases the 3D rendering and physics systems working together,
demonstrating:
- 3D rendering with ModernGL
- 3D physics with PyBullet
- Camera system with multiple modes
- Model loading and primitive generation
- Basic lighting and shadows
- Joystick input integration
"""

import sys
import os
import math
import time
from typing import Optional, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import pygame
import numpy as np

from pyjoysim.core.logging import setup_logging, get_logger
from pyjoysim.config.config import Config
from pyjoysim.input.joystick_manager import JoystickManager
from pyjoysim.input.input_processor import InputProcessor
from pyjoysim.physics.physics3d import (
    Physics3D, PhysicsMode, Vector3D, Quaternion,
    create_box_3d, create_sphere_3d, PhysicsMaterial3D
)
from pyjoysim.rendering.renderer3d import (
    Renderer3D, RenderObject3D, Material3D, Light3D,
    create_render_object_from_physics
)
from pyjoysim.rendering.camera3d import Camera3D, CameraMode, Camera3DSettings
from pyjoysim.rendering.model_loader import ModelLoader, PrimitiveType

# Check for ModernGL availability
try:
    import moderngl as mgl
    import moderngl_window as mglw
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False


class Simple3DWindow(mglw.WindowConfig):
    """Simple 3D window using moderngl-window."""
    
    title = "PyJoySim 3D Demo"
    window_size = (1024, 768)
    aspect_ratio = None
    resizable = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.logger = get_logger("3d_demo")
        
        # Initialize 3D renderer
        self.renderer = Renderer3D(self.window_size[0], self.window_size[1])
        self.renderer.initialize_context(self.ctx)
        
        # Initialize camera
        camera_settings = Camera3DSettings()
        camera_settings.move_speed = 20.0
        camera_settings.mouse_sensitivity = 0.003
        self.camera = Camera3D(
            self.window_size[0], 
            self.window_size[1],
            camera_settings
        )
        self.camera.set_position(np.array([10.0, 10.0, 10.0]))
        self.camera.look_at(np.array([0.0, 0.0, 0.0]))
        
        # Initialize physics
        self.physics = Physics3D(PhysicsMode.HEADLESS)
        
        # Initialize model loader
        self.model_loader = ModelLoader()
        
        # Create demo scene
        self._create_demo_scene()
        
        # Input state
        self.keys_pressed = set()
        self.mouse_captured = False
        self.last_mouse_pos = (0, 0)
        
        # Demo state
        self.demo_objects = {}
        self.simulation_time = 0.0
        
        self.logger.info("3D Demo initialized successfully")
    
    def _create_demo_scene(self):
        """Create the demo scene with various 3D objects."""
        # Create ground plane
        ground = create_box_3d(
            "ground",
            Vector3D(20, 0.5, 20),
            Vector3D(0, -0.25, 0),
            0.0  # Static mass
        )
        ground.material = PhysicsMaterial3D(friction=0.8, restitution=0.1)
        self.physics.add_object(ground)
        
        # Create render object for ground
        ground_render = create_render_object_from_physics(ground, self.renderer, "default")
        ground_render.material.diffuse = (0.3, 0.7, 0.3)  # Green ground
        self.renderer.add_render_object(ground_render)
        
        # Create some boxes
        for i in range(5):
            box = create_box_3d(
                f"box_{i}",
                Vector3D(2, 2, 2),
                Vector3D(i * 3 - 6, 5 + i * 2, 0),
                2.0
            )
            box.material = PhysicsMaterial3D(friction=0.5, restitution=0.6)
            self.physics.add_object(box)
            
            # Create render object
            box_render = create_render_object_from_physics(box, self.renderer, "red")
            self.renderer.add_render_object(box_render)
            
            self.demo_objects[f"box_{i}"] = {"physics": box, "render": box_render}
        
        # Create some spheres
        for i in range(3):
            sphere = create_sphere_3d(
                f"sphere_{i}",
                1.5,
                Vector3D(i * 4 - 4, 10 + i * 3, 5),
                1.0
            )
            sphere.material = PhysicsMaterial3D(friction=0.3, restitution=0.8)
            self.physics.add_object(sphere)
            
            # Create render object
            sphere_render = create_render_object_from_physics(sphere, self.renderer, "blue")
            self.renderer.add_render_object(sphere_render)
            
            self.demo_objects[f"sphere_{i}"] = {"physics": sphere, "render": sphere_render}
        
        # Create a tower of boxes
        for i in range(6):
            tower_box = create_box_3d(
                f"tower_box_{i}",
                Vector3D(1.5, 1.5, 1.5),
                Vector3D(-8, 1 + i * 1.6, -8),
                1.5
            )
            tower_box.material = PhysicsMaterial3D(friction=0.7, restitution=0.2)
            self.physics.add_object(tower_box)
            
            # Alternate colors
            material_name = "green" if i % 2 == 0 else "red"
            tower_render = create_render_object_from_physics(tower_box, self.renderer, material_name)
            self.renderer.add_render_object(tower_render)
            
            self.demo_objects[f"tower_box_{i}"] = {"physics": tower_box, "render": tower_render}
        
        # Add some dynamic lights
        point_light = Light3D(
            name="point_light",
            light_type="point",
            position=Vector3D(5, 8, 5),
            diffuse=(1.0, 0.8, 0.6),
            ambient=(0.1, 0.1, 0.1)
        )
        self.renderer.add_light(point_light)
        
        self.logger.info("Demo scene created", extra={
            "physics_objects": len(self.physics.objects),
            "render_objects": len(self.renderer.render_objects)
        })
    
    def key_event(self, key, action, modifiers):
        """Handle keyboard input."""
        if action == self.wnd.keys.ACTION_PRESS:
            self.keys_pressed.add(key)
            
            # Camera mode switching
            if key == self.wnd.keys.F1:
                self.camera.set_mode(CameraMode.FREE)
                self.logger.info("Camera mode: FREE")
            elif key == self.wnd.keys.F2:
                # Follow first box
                if "box_0" in self.demo_objects:
                    target = self.demo_objects["box_0"]["physics"]
                    self.camera.set_mode(CameraMode.FOLLOW, target)
                    self.logger.info("Camera mode: FOLLOW")
            elif key == self.wnd.keys.F3:
                # Orbit around origin
                self.camera.set_mode(CameraMode.ORBIT, None)
                self.camera.target_position = np.array([0.0, 0.0, 0.0])
                self.logger.info("Camera mode: ORBIT")
            
            # Reset simulation
            elif key == self.wnd.keys.R:
                self._reset_simulation()
            
            # Toggle wireframe
            elif key == self.wnd.keys.Z:
                self.renderer.wireframe_mode = not self.renderer.wireframe_mode
                mode = "ON" if self.renderer.wireframe_mode else "OFF"
                self.logger.info(f"Wireframe mode: {mode}")
            
            # Throw projectile
            elif key == self.wnd.keys.SPACE:
                self._throw_projectile()
            
            # Mouse capture toggle
            elif key == self.wnd.keys.ESCAPE:
                self.mouse_captured = not self.mouse_captured
                self.wnd.mouse_exclusivity = self.mouse_captured
                
        elif action == self.wnd.keys.ACTION_RELEASE:
            self.keys_pressed.discard(key)
    
    def mouse_position_event(self, x, y, dx, dy):
        """Handle mouse movement."""
        if self.mouse_captured and self.camera.mode == CameraMode.FREE:
            self.camera.handle_mouse_movement(dx, dy)
        
        self.last_mouse_pos = (x, y)
    
    def mouse_scroll_event(self, x_offset, y_offset):
        """Handle mouse scroll."""
        self.camera.handle_scroll(y_offset)
    
    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.renderer.resize(width, height)
        self.camera.set_viewport_size(width, height)
    
    def _handle_camera_input(self, dt: float):
        """Handle camera input in free mode."""
        if self.camera.mode != CameraMode.FREE:
            return
        
        # Movement keys
        self.camera.move_forward = self.wnd.keys.W in self.keys_pressed
        self.camera.move_backward = self.wnd.keys.S in self.keys_pressed
        self.camera.move_left = self.wnd.keys.A in self.keys_pressed
        self.camera.move_right = self.wnd.keys.D in self.keys_pressed
        self.camera.move_up = self.wnd.keys.Q in self.keys_pressed
        self.camera.move_down = self.wnd.keys.E in self.keys_pressed
    
    def _throw_projectile(self):
        """Throw a projectile from camera position."""
        projectile_count = len([k for k in self.demo_objects.keys() if k.startswith("projectile")])
        
        # Create projectile
        projectile = create_sphere_3d(
            f"projectile_{projectile_count}",
            0.5,
            Vector3D(*self.camera.position),
            0.5
        )
        
        # Set initial velocity in camera forward direction
        throw_force = 25.0
        velocity = Vector3D(*(self.camera.forward_vector * throw_force))
        projectile.set_velocity(velocity)
        
        self.physics.add_object(projectile)
        
        # Create render object
        projectile_render = create_render_object_from_physics(projectile, self.renderer, "blue")
        projectile_render.material.diffuse = (1.0, 0.5, 0.0)  # Orange
        self.renderer.add_render_object(projectile_render)
        
        self.demo_objects[f"projectile_{projectile_count}"] = {
            "physics": projectile, 
            "render": projectile_render
        }
        
        self.logger.info("Projectile thrown", extra={
            "position": self.camera.position.tolist(),
            "velocity": velocity.to_tuple()
        })
    
    def _reset_simulation(self):
        """Reset the simulation to initial state."""
        # Remove all dynamic objects
        objects_to_remove = []
        for name, obj_data in self.demo_objects.items():
            if not name.startswith("ground"):
                objects_to_remove.append(name)
        
        for name in objects_to_remove:
            obj_data = self.demo_objects[name]
            self.physics.remove_object(obj_data["physics"].name)
            self.renderer.remove_render_object(obj_data["render"].name)
            del self.demo_objects[name]
        
        # Recreate scene
        self._create_demo_scene()
        
        self.logger.info("Simulation reset")
    
    def _update_render_objects(self):
        """Update render object transforms from physics objects."""
        for obj_data in self.demo_objects.values():
            physics_obj = obj_data["physics"]
            render_obj = obj_data["render"]
            
            # Update position and rotation
            render_obj.position = physics_obj.position
            render_obj.rotation = physics_obj.rotation
    
    def render(self, time: float, frame_time: float):
        """Main render loop."""
        self.simulation_time = time
        
        # Handle input
        self._handle_camera_input(frame_time)
        
        # Update camera
        self.camera.update(frame_time)
        
        # Step physics simulation
        self.physics.step(frame_time)
        
        # Update render objects from physics
        self._update_render_objects()
        
        # Render the scene
        self.renderer.render(self.camera)
        
        # Display info in window title
        fps = 1.0 / frame_time if frame_time > 0 else 0
        objects_count = len(self.demo_objects)
        camera_mode = self.camera.mode.value.upper()
        
        title = (f"PyJoySim 3D Demo - {objects_count} objects - "
                f"{camera_mode} camera - {fps:.1f} FPS - "
                f"[F1-F3: Camera] [SPACE: Throw] [R: Reset] [Z: Wireframe] [ESC: Mouse]")
        
        self.wnd.title = title


class PyGame3DDemo:
    """Alternative pygame-based 3D demo (fallback if moderngl-window not available)."""
    
    def __init__(self):
        self.logger = get_logger("3d_demo_pygame")
        
        # Initialize pygame
        pygame.init()
        
        # Create window
        self.width, self.height = 1024, 768
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("PyJoySim 3D Demo (Pygame)")
        
        # Initialize OpenGL context
        import moderngl as mgl
        self.ctx = mgl.create_context()
        
        # Initialize systems
        self.renderer = Renderer3D(self.width, self.height)
        self.renderer.initialize_context(self.ctx)
        
        self.camera = Camera3D(self.width, self.height)
        self.camera.set_position(np.array([10.0, 10.0, 10.0]))
        self.camera.look_at(np.array([0.0, 0.0, 0.0]))
        
        self.physics = Physics3D(PhysicsMode.HEADLESS)
        
        # Demo state
        self.running = True
        self.clock = pygame.time.Clock()
        self.demo_objects = {}
        
        # Create scene
        self._create_simple_scene()
        
        self.logger.info("Pygame 3D Demo initialized")
    
    def _create_simple_scene(self):
        """Create a simple test scene."""
        # Ground
        ground = create_box_3d("ground", Vector3D(10, 0.5, 10), Vector3D(0, 0, 0), 0.0)
        self.physics.add_object(ground)
        ground_render = create_render_object_from_physics(ground, self.renderer)
        self.renderer.add_render_object(ground_render)
        
        # A few test objects
        for i in range(3):
            box = create_box_3d(f"box_{i}", Vector3D(1, 1, 1), Vector3D(i*2-2, 3, 0), 1.0)
            self.physics.add_object(box)
            box_render = create_render_object_from_physics(box, self.renderer)
            self.renderer.add_render_object(box_render)
            self.demo_objects[f"box_{i}"] = {"physics": box, "render": box_render}
    
    def run(self):
        """Main game loop."""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # 60 FPS target
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # Update physics
            self.physics.step(dt)
            
            # Update render objects
            for obj_data in self.demo_objects.values():
                physics_obj = obj_data["physics"]
                render_obj = obj_data["render"]
                render_obj.position = physics_obj.position
                render_obj.rotation = physics_obj.rotation
            
            # Render
            self.renderer.render(self.camera)
            pygame.display.flip()
        
        pygame.quit()


def main():
    """Main function."""
    # Setup logging
    setup_logging()
    logger = get_logger("3d_demo_main")
    
    logger.info("Starting PyJoySim 3D Demo")
    
    if not MODERNGL_AVAILABLE:
        logger.error("ModernGL not available. Install with: pip install moderngl moderngl-window")
        return 1
    
    try:
        # Try to use moderngl-window for better experience
        try:
            mglw.run_window_config(Simple3DWindow)
        except ImportError:
            logger.warning("moderngl-window not available, using pygame fallback")
            demo = PyGame3DDemo()
            demo.run()
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error("Demo failed", extra={"error": str(e)})
        return 1
    
    logger.info("Demo completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())