"""
Unit tests for the rendering system.

Tests the rendering engine, camera system, and visual output functionality.
"""

import unittest
import math
from unittest.mock import Mock, patch, MagicMock

import pygame

# Import the modules to test
from pyjoysim.rendering import (
    RenderEngine, Renderer2D, RenderEngineType, Color, StandardColors,
    BlendMode, Viewport, RenderStats, Camera2D, CameraController, CameraBounds,
    get_render_engine, create_render_engine, reset_render_engine
)
from pyjoysim.physics import Vector2D


class TestColor(unittest.TestCase):
    """Test the Color class."""
    
    def test_color_creation(self):
        """Test color creation and validation."""
        # Normal color
        color = Color(255, 128, 64, 200)
        self.assertEqual(color.r, 255)
        self.assertEqual(color.g, 128)
        self.assertEqual(color.b, 64)
        self.assertEqual(color.a, 200)
    
    def test_color_clamping(self):
        """Test that color values are clamped to valid ranges."""
        # Values over 255 should be clamped
        color = Color(300, -50, 128)
        self.assertEqual(color.r, 255)
        self.assertEqual(color.g, 0)
        self.assertEqual(color.b, 128)
        self.assertEqual(color.a, 255)  # Default alpha
    
    def test_color_conversions(self):
        """Test color conversion methods."""
        color = Color(255, 128, 64, 200)
        
        # RGB tuple
        rgb = color.to_tuple()
        self.assertEqual(rgb, (255, 128, 64))
        
        # RGBA tuple
        rgba = color.to_tuple_rgba()
        self.assertEqual(rgba, (255, 128, 64, 200))
    
    def test_color_with_alpha(self):
        """Test creating color with different alpha."""
        color = Color(255, 128, 64)
        new_color = color.with_alpha(100)
        
        self.assertEqual(new_color.r, 255)
        self.assertEqual(new_color.g, 128)
        self.assertEqual(new_color.b, 64)
        self.assertEqual(new_color.a, 100)
        
        # Original should be unchanged
        self.assertEqual(color.a, 255)
    
    def test_color_from_hex(self):
        """Test creating color from hex string."""
        # 6-digit hex
        color = Color.from_hex("#FF8040")
        self.assertEqual(color.r, 255)
        self.assertEqual(color.g, 128)
        self.assertEqual(color.b, 64)
        self.assertEqual(color.a, 255)
        
        # 8-digit hex with alpha
        color = Color.from_hex("#FF8040C8")
        self.assertEqual(color.r, 255)
        self.assertEqual(color.g, 128)
        self.assertEqual(color.b, 64)
        self.assertEqual(color.a, 200)
        
        # Without # prefix
        color = Color.from_hex("FF8040")
        self.assertEqual(color.r, 255)
        self.assertEqual(color.g, 128)
        self.assertEqual(color.b, 64)
    
    def test_invalid_hex(self):
        """Test that invalid hex strings raise errors."""
        with self.assertRaises(ValueError):
            Color.from_hex("invalid")
        
        with self.assertRaises(ValueError):
            Color.from_hex("#ZZ8040")


class TestViewport(unittest.TestCase):
    """Test the Viewport class."""
    
    def test_viewport_creation(self):
        """Test viewport creation."""
        viewport = Viewport(100, 50, 800, 600)
        self.assertEqual(viewport.x, 100)
        self.assertEqual(viewport.y, 50)
        self.assertEqual(viewport.width, 800)
        self.assertEqual(viewport.height, 600)
    
    def test_viewport_center(self):
        """Test viewport center calculation."""
        viewport = Viewport(100, 50, 800, 600)
        center = viewport.get_center()
        self.assertEqual(center.x, 500)  # 100 + 800/2
        self.assertEqual(center.y, 350)  # 50 + 600/2
    
    def test_aspect_ratio(self):
        """Test aspect ratio calculation."""
        viewport = Viewport(0, 0, 800, 600)
        ratio = viewport.get_aspect_ratio()
        self.assertAlmostEqual(ratio, 800/600, places=5)
        
        # Test division by zero protection
        viewport = Viewport(0, 0, 800, 0)
        ratio = viewport.get_aspect_ratio()
        self.assertEqual(ratio, 1.0)


class TestRenderStats(unittest.TestCase):
    """Test the RenderStats class."""
    
    def test_stats_initialization(self):
        """Test that stats are initialized correctly."""
        stats = RenderStats()
        self.assertEqual(stats.frame_count, 0)
        self.assertEqual(stats.total_render_time, 0.0)
        self.assertEqual(stats.fps, 0.0)
    
    def test_stats_update(self):
        """Test updating statistics."""
        stats = RenderStats()
        
        # First frame
        stats.update(0.016, objects=10, draw_calls=5)
        self.assertEqual(stats.frame_count, 1)
        self.assertEqual(stats.objects_rendered, 10)
        self.assertEqual(stats.draw_calls, 5)
        self.assertAlmostEqual(stats.fps, 1.0/0.016, places=1)
        
        # Second frame
        stats.update(0.020, objects=8, draw_calls=3)
        self.assertEqual(stats.frame_count, 2)
        self.assertEqual(stats.objects_rendered, 18)
        self.assertEqual(stats.draw_calls, 8)


class TestCamera2D(unittest.TestCase):
    """Test the Camera2D class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera = Camera2D(800, 600)
    
    def test_camera_initialization(self):
        """Test camera initialization."""
        self.assertEqual(self.camera.viewport_width, 800)
        self.assertEqual(self.camera.viewport_height, 600)
        self.assertEqual(self.camera.position.x, 0)
        self.assertEqual(self.camera.position.y, 0)
        self.assertEqual(self.camera.zoom, 1.0)
        self.assertEqual(self.camera.rotation, 0.0)
    
    def test_camera_position(self):
        """Test camera position setting."""
        new_pos = Vector2D(100, 50)
        self.camera.set_position(new_pos)
        self.assertEqual(self.camera.position.x, 100)
        self.assertEqual(self.camera.position.y, 50)
    
    def test_camera_movement(self):
        """Test camera movement."""
        initial_pos = self.camera.position
        offset = Vector2D(10, -5)
        self.camera.move(offset)
        
        self.assertEqual(self.camera.position.x, initial_pos.x + 10)
        self.assertEqual(self.camera.position.y, initial_pos.y - 5)
    
    def test_camera_zoom(self):
        """Test camera zoom functionality."""
        # Set zoom
        self.camera.set_zoom(2.0)
        self.assertEqual(self.camera.zoom, 2.0)
        
        # Zoom by factor
        self.camera.zoom_by(1.5)
        self.assertEqual(self.camera.zoom, 3.0)
        
        # Test zoom limits
        self.camera.set_zoom(0.01)  # Below minimum
        self.assertGreaterEqual(self.camera.zoom, self.camera.min_zoom)
        
        self.camera.set_zoom(50.0)  # Above maximum
        self.assertLessEqual(self.camera.zoom, self.camera.max_zoom)
    
    def test_camera_bounds(self):
        """Test camera bounds functionality."""
        bounds = CameraBounds(-100, 100, -50, 50)
        self.camera.set_bounds(bounds)
        
        # Try to move outside bounds
        self.camera.set_position(Vector2D(200, 100))  # Outside bounds
        self.assertLessEqual(self.camera.position.x, 100)
        self.assertLessEqual(self.camera.position.y, 50)
    
    def test_coordinate_transformation(self):
        """Test world to screen coordinate transformation."""
        # Simple case - no zoom, no offset
        world_pos = Vector2D(100, 50)
        screen_pos = self.camera.world_to_screen(world_pos)
        
        # Should center the coordinate system
        expected_x = 100 + 800 // 2
        expected_y = 600 // 2 - 50  # Y flipped
        self.assertEqual(screen_pos.x, expected_x)
        self.assertEqual(screen_pos.y, expected_y)
        
        # Test reverse transformation
        world_pos_back = self.camera.screen_to_world(screen_pos)
        self.assertAlmostEqual(world_pos_back.x, world_pos.x, places=5)
        self.assertAlmostEqual(world_pos_back.y, world_pos.y, places=5)
    
    def test_coordinate_transformation_with_zoom(self):
        """Test coordinate transformation with zoom."""
        self.camera.set_zoom(2.0)
        
        world_pos = Vector2D(50, 25)
        screen_pos = self.camera.world_to_screen(world_pos)
        
        # With 2x zoom, world coordinates should be scaled by 2
        expected_x = 50 * 2 + 800 // 2
        expected_y = 600 // 2 - 25 * 2
        self.assertEqual(screen_pos.x, expected_x)
        self.assertEqual(screen_pos.y, expected_y)
    
    def test_visibility_checks(self):
        """Test visibility checking methods."""
        # Point at center should be visible
        center_point = Vector2D(0, 0)
        self.assertTrue(self.camera.is_point_visible(center_point))
        
        # Point far away should not be visible
        far_point = Vector2D(1000, 1000)
        self.assertFalse(self.camera.is_point_visible(far_point))
        
        # Circle visibility
        self.assertTrue(self.camera.is_circle_visible(Vector2D(0, 0), 1.0))
        self.assertFalse(self.camera.is_circle_visible(Vector2D(1000, 1000), 1.0))
    
    def test_shake_effect(self):
        """Test camera shake functionality."""
        initial_pos = self.camera.position
        
        # Add shake
        self.camera.add_shake(10.0, 0.5)
        self.assertGreater(self.camera.shake_intensity, 0)
        self.assertGreater(self.camera.shake_time_remaining, 0)
        
        # Update should apply shake
        self.camera.update(0.1)
        # After update, shake should still be active but reduced
        self.assertGreater(self.camera.shake_time_remaining, 0)


class TestCameraController(unittest.TestCase):
    """Test the CameraController class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera = Camera2D(800, 600)
        self.controller = CameraController(self.camera)
    
    def test_controller_initialization(self):
        """Test controller initialization."""
        self.assertEqual(self.controller.camera, self.camera)
        self.assertFalse(self.controller.move_left)
        self.assertFalse(self.controller.move_right)
        self.assertFalse(self.controller.zoom_in)
        self.assertFalse(self.controller.zoom_out)
    
    def test_movement_control(self):
        """Test movement control."""
        initial_pos = self.camera.position
        
        # Enable movement
        self.controller.move_right = True
        self.controller.update(0.1)  # 0.1 second update
        
        # Camera should have moved right
        self.assertGreater(self.camera.position.x, initial_pos.x)
    
    def test_zoom_control(self):
        """Test zoom control."""
        initial_zoom = self.camera.zoom
        
        # Enable zoom in
        self.controller.zoom_in = True
        self.controller.update(0.1)
        
        # Camera should have zoomed in
        self.assertGreater(self.camera.zoom, initial_zoom)
    
    def test_reset_to_origin(self):
        """Test reset to origin functionality."""
        # Move camera away from origin
        self.camera.set_position(Vector2D(100, 50))
        self.camera.set_zoom(2.0)
        self.camera.set_rotation(0.5)
        
        # Reset
        self.controller.reset_to_origin()
        
        # Should be back at defaults
        self.assertEqual(self.camera.position.x, 0)
        self.assertEqual(self.camera.position.y, 0)
        self.assertEqual(self.camera.zoom, 1.0)
        self.assertEqual(self.camera.rotation, 0.0)


class TestRenderer2D(unittest.TestCase):
    """Test the Renderer2D class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock pygame to avoid actual window creation
        self.pygame_patcher = patch('pyjoysim.rendering.engine.pygame')
        self.mock_pygame = self.pygame_patcher.start()
        
        # Set up mock pygame objects
        self.mock_pygame.init.return_value = None
        self.mock_pygame.display.set_mode.return_value = Mock()
        self.mock_pygame.display.set_caption.return_value = None
        self.mock_pygame.font.init.return_value = None
        self.mock_pygame.time.Clock.return_value = Mock()
        
        self.renderer = Renderer2D()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.pygame_patcher.stop()
    
    def test_renderer_initialization(self):
        """Test renderer initialization."""
        self.assertFalse(self.renderer.is_initialized())
        
        # Test initialization
        success = self.renderer.initialize(800, 600, "Test")
        self.assertTrue(success)
        self.assertTrue(self.renderer.is_initialized())
    
    def test_frame_lifecycle(self):
        """Test frame begin/end lifecycle."""
        self.renderer.initialize(800, 600)
        
        # Should not raise errors
        self.renderer.begin_frame()
        self.renderer.end_frame()
    
    def test_clear_screen(self):
        """Test screen clearing."""
        self.renderer.initialize(800, 600)
        
        # Should not raise errors
        self.renderer.clear()
        self.renderer.clear(StandardColors.RED)
    
    def test_viewport_management(self):
        """Test viewport management."""
        viewport = Viewport(100, 50, 640, 480)
        self.renderer.set_viewport(viewport)
        
        retrieved_viewport = self.renderer.get_viewport()
        self.assertEqual(retrieved_viewport.x, 100)
        self.assertEqual(retrieved_viewport.y, 50)
        self.assertEqual(retrieved_viewport.width, 640)
        self.assertEqual(retrieved_viewport.height, 480)
    
    def test_background_color(self):
        """Test background color management."""
        color = StandardColors.BLUE
        self.renderer.set_background_color(color)
        
        retrieved_color = self.renderer.get_background_color()
        self.assertEqual(retrieved_color, color)
    
    def test_blend_mode(self):
        """Test blend mode management."""
        self.renderer.set_blend_mode(BlendMode.ALPHA)
        self.assertEqual(self.renderer.get_blend_mode(), BlendMode.ALPHA)


class TestRenderEngineFactory(unittest.TestCase):
    """Test the render engine factory functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset global state
        reset_render_engine()
    
    def tearDown(self):
        """Clean up test fixtures."""
        reset_render_engine()
    
    def test_create_render_engine(self):
        """Test render engine creation."""
        engine = create_render_engine(RenderEngineType.RENDERER_2D)
        self.assertIsInstance(engine, Renderer2D)
        
        # Should be set as global instance
        global_engine = get_render_engine()
        self.assertEqual(engine, global_engine)
    
    def test_unsupported_engine_type(self):
        """Test that unsupported engine types raise errors."""
        with self.assertRaises(ValueError):
            create_render_engine("unsupported_type")
    
    def test_reset_render_engine(self):
        """Test render engine reset."""
        # Create an engine
        engine = create_render_engine(RenderEngineType.RENDERER_2D)
        self.assertIsNotNone(get_render_engine())
        
        # Reset
        reset_render_engine()
        self.assertIsNone(get_render_engine())


class TestStandardColors(unittest.TestCase):
    """Test the StandardColors class."""
    
    def test_standard_colors_exist(self):
        """Test that standard colors are defined."""
        # Test a few key colors
        self.assertIsInstance(StandardColors.BLACK, Color)
        self.assertIsInstance(StandardColors.WHITE, Color)
        self.assertIsInstance(StandardColors.RED, Color)
        self.assertIsInstance(StandardColors.GREEN, Color)
        self.assertIsInstance(StandardColors.BLUE, Color)
        
        # Test specific values
        self.assertEqual(StandardColors.BLACK.r, 0)
        self.assertEqual(StandardColors.BLACK.g, 0)
        self.assertEqual(StandardColors.BLACK.b, 0)
        
        self.assertEqual(StandardColors.WHITE.r, 255)
        self.assertEqual(StandardColors.WHITE.g, 255)
        self.assertEqual(StandardColors.WHITE.b, 255)
        
        self.assertEqual(StandardColors.RED.r, 255)
        self.assertEqual(StandardColors.RED.g, 0)
        self.assertEqual(StandardColors.RED.b, 0)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)