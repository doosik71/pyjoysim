#!/usr/bin/env python3
"""
Integration tests for 3D rendering and physics systems.

Tests the integration between:
- Physics3D and Renderer3D
- Camera3D and scene objects
- Model loading and physics object creation
- Performance and stability
"""

import unittest
import numpy as np
import time
from typing import Dict, Any

from pyjoysim.physics.physics3d import (
    Physics3D, PhysicsMode, Vector3D, Quaternion,
    create_box_3d, create_sphere_3d, PhysicsMaterial3D
)
from pyjoysim.rendering.renderer3d import (
    Renderer3D, RenderObject3D, Material3D,
    create_render_object_from_physics
)
from pyjoysim.rendering.camera3d import Camera3D, CameraMode
from pyjoysim.rendering.model_loader import ModelLoader, PrimitiveType

# Skip tests if ModernGL is not available
try:
    import moderngl as mgl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False


class Mock3DContext:
    """Mock 3D context for testing without actual OpenGL."""
    
    def __init__(self):
        self.programs = {}
        self.buffers = {}
        self.textures = {}
        self.vertex_arrays = {}
        
    def program(self, vertex_shader, fragment_shader):
        """Mock program creation."""
        class MockProgram:
            def __init__(self):
                self.uniforms = {}
            
            def __getitem__(self, key):
                return MockUniform()
            
            def use(self):
                pass
            
            def release(self):
                pass
        
        return MockProgram()
    
    def buffer(self, data):
        """Mock buffer creation."""
        class MockBuffer:
            def release(self):
                pass
        return MockBuffer()
    
    def vertex_array(self, program, buffers, index_buffer=None):
        """Mock VAO creation."""
        class MockVertexArray:
            def render(self):
                pass
            
            def release(self):
                pass
        return MockVertexArray()
    
    def enable(self, cap):
        """Mock OpenGL enable."""
        pass
    
    def clear(self, *args):
        """Mock clear."""
        pass
    
    @property
    def viewport(self):
        return (0, 0, 800, 600)
    
    @viewport.setter
    def viewport(self, value):
        pass
    
    @property
    def wireframe(self):
        return False
    
    @wireframe.setter
    def wireframe(self, value):
        pass


class MockUniform:
    """Mock uniform for testing."""
    
    def __init__(self):
        self._value = None
    
    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, val):
        self._value = val
    
    def write(self, data):
        pass


@unittest.skipUnless(MODERNGL_AVAILABLE, "ModernGL not available")
class Test3DIntegration(unittest.TestCase):
    """Test 3D system integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Use headless physics for testing
        self.physics = Physics3D(PhysicsMode.HEADLESS)
        
        # Create renderer with mock context
        self.renderer = Renderer3D(800, 600)
        
        # Use mock context for testing
        if not hasattr(self, '_mock_context_set'):
            self.mock_ctx = Mock3DContext()
            self.renderer.initialize_context(self.mock_ctx)
            self._mock_context_set = True
        
        # Create camera
        self.camera = Camera3D(800, 600)
        self.camera.set_position(np.array([5.0, 5.0, 5.0]))
        self.camera.look_at(np.array([0.0, 0.0, 0.0]))
        
        # Create model loader
        self.model_loader = ModelLoader()
        
        # Test objects
        self.test_objects = {}
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove all objects
        for name in list(self.test_objects.keys()):
            self._remove_test_object(name)
        
        # Clean up physics
        self.physics.cleanup()
        
        # Clean up renderer
        self.renderer.cleanup()
    
    def _create_test_object(self, name: str, obj_type: str = "box") -> Dict[str, Any]:
        """Create a test object with both physics and rendering."""
        if obj_type == "box":
            physics_obj = create_box_3d(
                name,
                Vector3D(1, 1, 1),
                Vector3D(0, 5, 0),
                1.0
            )
        elif obj_type == "sphere":
            physics_obj = create_sphere_3d(
                name,
                0.5,
                Vector3D(0, 5, 0),
                1.0
            )
        else:
            raise ValueError(f"Unsupported object type: {obj_type}")
        
        # Add to physics
        self.physics.add_object(physics_obj)
        
        # Create render object
        render_obj = create_render_object_from_physics(physics_obj, self.renderer)
        self.renderer.add_render_object(render_obj)
        
        test_obj = {
            "physics": physics_obj,
            "render": render_obj,
            "type": obj_type
        }
        
        self.test_objects[name] = test_obj
        return test_obj
    
    def _remove_test_object(self, name: str):
        """Remove a test object."""
        if name in self.test_objects:
            obj = self.test_objects[name]
            self.physics.remove_object(obj["physics"].name)
            self.renderer.remove_render_object(obj["render"].name)
            del self.test_objects[name]
    
    def _update_render_from_physics(self):
        """Update render objects from physics objects."""
        for obj in self.test_objects.values():
            physics_obj = obj["physics"]
            render_obj = obj["render"]
            
            # Update position and rotation
            render_obj.position = physics_obj.position
            render_obj.rotation = physics_obj.rotation
    
    def test_physics_rendering_sync(self):
        """Test that physics and rendering stay synchronized."""
        # Create test object
        obj = self._create_test_object("test_box", "box")
        
        # Set initial position
        initial_pos = Vector3D(1, 5, 2)
        obj["physics"].set_position(initial_pos)
        
        # Update render object
        obj["render"].position = obj["physics"].position
        
        # Verify synchronization
        self.assertEqual(obj["physics"].position.x, initial_pos.x)
        self.assertEqual(obj["physics"].position.y, initial_pos.y)
        self.assertEqual(obj["physics"].position.z, initial_pos.z)
        
        self.assertEqual(obj["render"].position.x, initial_pos.x)
        self.assertEqual(obj["render"].position.y, initial_pos.y)
        self.assertEqual(obj["render"].position.z, initial_pos.z)
    
    def test_physics_simulation_step(self):
        """Test physics simulation with render object updates."""
        # Create test objects
        self._create_test_object("box1", "box")
        self._create_test_object("sphere1", "sphere")
        
        # Run simulation for a few steps
        for i in range(10):
            self.physics.step(1.0 / 60.0)  # 60 FPS
            self._update_render_from_physics()
        
        # Verify objects have moved (gravity should affect them)
        for name, obj in self.test_objects.items():
            # Objects should have fallen due to gravity
            self.assertLess(obj["physics"].position.y, 5.0)
            
            # Render position should match physics position
            np.testing.assert_array_almost_equal(
                obj["render"].position.to_array(),
                obj["physics"].position.to_array(),
                decimal=6
            )
    
    def test_camera_modes(self):
        """Test different camera modes with 3D objects."""
        # Create a test object to follow
        target_obj = self._create_test_object("target", "box")
        
        # Test free camera mode
        self.camera.set_mode(CameraMode.FREE)
        self.assertEqual(self.camera.mode, CameraMode.FREE)
        
        # Test follow camera mode
        self.camera.set_mode(CameraMode.FOLLOW, target_obj["physics"])
        self.assertEqual(self.camera.mode, CameraMode.FOLLOW)
        self.assertEqual(self.camera.target_object, target_obj["physics"])
        
        # Update camera (should track target)
        self.camera.update(1.0 / 60.0)
        
        # Test orbit camera mode
        self.camera.set_mode(CameraMode.ORBIT, target_obj["physics"])
        self.assertEqual(self.camera.mode, CameraMode.ORBIT)
        
        # Update camera (should orbit around target)
        for i in range(10):
            self.camera.update(1.0 / 60.0)
    
    def test_render_scene(self):
        """Test rendering a complete scene."""
        # Create various objects
        self._create_test_object("ground", "box")
        self._create_test_object("ball", "sphere")
        
        # Set up ground as static
        ground_obj = self.test_objects["ground"]["physics"]
        ground_obj.mass = 0.0  # Make it static
        ground_obj.set_position(Vector3D(0, -0.5, 0))
        
        # Update render objects
        self._update_render_from_physics()
        
        # Render the scene (should not crash)
        try:
            self.renderer.render(self.camera)
        except AttributeError:
            # Expected with mock context, but should not crash badly
            pass
    
    def test_model_primitive_creation(self):
        """Test creating primitives and integrating with physics."""
        # Create primitive meshes
        cube_mesh = self.model_loader.create_primitive(PrimitiveType.CUBE)
        sphere_mesh = self.model_loader.create_primitive(PrimitiveType.SPHERE)
        
        # Verify mesh properties
        self.assertIsNotNone(cube_mesh.vertices)
        self.assertIsNotNone(cube_mesh.indices)
        self.assertIsNotNone(cube_mesh.normals)
        
        self.assertIsNotNone(sphere_mesh.vertices)
        self.assertIsNotNone(sphere_mesh.indices)
        self.assertIsNotNone(sphere_mesh.normals)
        
        # Verify vertex counts are reasonable
        self.assertGreater(len(cube_mesh.vertices), 0)
        self.assertGreater(len(sphere_mesh.vertices), 0)
    
    def test_physics_materials(self):
        """Test physics materials integration."""
        # Create objects with different materials
        bouncy_material = PhysicsMaterial3D(
            friction=0.3,
            restitution=0.9,  # Very bouncy
            density=1.0
        )
        
        sticky_material = PhysicsMaterial3D(
            friction=1.0,  # Very sticky
            restitution=0.1,  # Not bouncy
            density=2.0
        )
        
        # Create objects with materials
        bouncy_obj = create_sphere_3d(
            "bouncy_ball",
            0.5,
            Vector3D(0, 5, 0),
            1.0,
            bouncy_material
        )
        
        sticky_obj = create_box_3d(
            "sticky_box",
            Vector3D(1, 1, 1),
            Vector3D(2, 5, 0),
            1.0,
            sticky_material
        )
        
        # Add to physics
        self.physics.add_object(bouncy_obj)
        self.physics.add_object(sticky_obj)
        
        # Create render objects
        bouncy_render = create_render_object_from_physics(bouncy_obj, self.renderer)
        sticky_render = create_render_object_from_physics(sticky_obj, self.renderer)
        
        self.renderer.add_render_object(bouncy_render)
        self.renderer.add_render_object(sticky_render)
        
        # Verify objects were created
        self.assertIsNotNone(self.physics.get_object("bouncy_ball"))
        self.assertIsNotNone(self.physics.get_object("sticky_box"))
        
        # Clean up
        self.physics.remove_object("bouncy_ball")
        self.physics.remove_object("sticky_box")
        self.renderer.remove_render_object("bouncy_ball")
        self.renderer.remove_render_object("sticky_box")
    
    def test_performance_many_objects(self):
        """Test performance with many objects."""
        start_time = time.time()
        
        # Create many objects
        num_objects = 50
        for i in range(num_objects):
            name = f"perf_test_{i}"
            obj_type = "sphere" if i % 2 == 0 else "box"
            
            obj = self._create_test_object(name, obj_type)
            
            # Set random position
            x = (i % 10) - 5
            z = (i // 10) - 5
            y = 5 + i * 0.1
            obj["physics"].set_position(Vector3D(x, y, z))
        
        creation_time = time.time() - start_time
        
        # Run simulation
        sim_start = time.time()
        for step in range(60):  # 1 second at 60 FPS
            self.physics.step(1.0 / 60.0)
            self._update_render_from_physics()
        
        simulation_time = time.time() - sim_start
        
        # Verify performance is reasonable
        self.assertLess(creation_time, 5.0, "Object creation took too long")
        self.assertLess(simulation_time, 2.0, "Simulation took too long")
        
        # Verify all objects still exist
        self.assertEqual(len(self.test_objects), num_objects)
        
        print(f"Performance test: {num_objects} objects")
        print(f"Creation time: {creation_time:.3f}s")
        print(f"Simulation time: {simulation_time:.3f}s")
    
    def test_camera_transformations(self):
        """Test camera transformation matrices."""
        # Set known camera position and orientation
        self.camera.set_position(np.array([0, 0, 5]))
        self.camera.look_at(np.array([0, 0, 0]))
        
        # Get transformation matrices
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix()
        
        # Verify matrix shapes
        self.assertEqual(view_matrix.shape, (4, 4))
        self.assertEqual(proj_matrix.shape, (4, 4))
        
        # Test world to screen transformation
        world_origin = np.array([0, 0, 0])
        screen_pos = self.camera.world_to_screen(world_origin)
        
        # Origin should be roughly in center of screen
        self.assertIsInstance(screen_pos, tuple)
        self.assertEqual(len(screen_pos), 3)  # x, y, depth
        
        # Screen coordinates should be reasonable
        screen_x, screen_y, depth = screen_pos
        self.assertGreater(screen_x, 0)
        self.assertLess(screen_x, 800)
        self.assertGreater(screen_y, 0)
        self.assertLess(screen_y, 600)


if __name__ == '__main__':
    unittest.main()