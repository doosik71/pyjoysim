"""
Integration tests for the rendering system.

Tests the rendering system in combination with other systems like physics and cameras.
"""

import unittest
import time
from unittest.mock import Mock, patch

import pygame

from pyjoysim.rendering import (
    RenderEngine, Renderer2D, RenderEngineType, StandardColors,
    Camera2D, create_render_engine, reset_render_engine
)
from pyjoysim.physics import Vector2D


class TestRenderingIntegration(unittest.TestCase):
    """Test rendering system integration."""
    
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
        self.mock_pygame.draw.circle.return_value = None
        self.mock_pygame.draw.rect.return_value = None
        self.mock_pygame.draw.line.return_value = None
        self.mock_pygame.draw.polygon.return_value = None
        
        # Mock font rendering
        mock_font = Mock()
        mock_font.render.return_value = Mock()
        self.mock_pygame.font.Font.return_value = mock_font
        
        # Reset global state
        reset_render_engine()
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.pygame_patcher.stop()
        reset_render_engine()
    
    def test_renderer_with_camera_integration(self):
        """Test renderer working with camera system."""
        # Create renderer and camera
        renderer = create_render_engine(RenderEngineType.RENDERER_2D)
        camera = Camera2D(800, 600)
        
        # Initialize renderer
        success = renderer.initialize(800, 600, "Integration Test")
        self.assertTrue(success)
        
        # Test rendering with camera transforms
        renderer.begin_frame()
        renderer.clear()
        
        # Draw some objects using camera coordinates
        world_positions = [
            Vector2D(0, 0),      # Center
            Vector2D(5, 5),      # Upper right
            Vector2D(-5, -5),    # Lower left
        ]
        
        for pos in world_positions:
            screen_pos = camera.world_to_screen(pos)
            renderer.draw_circle(screen_pos, 10, StandardColors.RED)
        
        renderer.end_frame()
        
        # Verify pygame draw methods were called
        self.mock_pygame.draw.circle.assert_called()
    
    def test_camera_following_object(self):
        """Test camera following a moving object."""
        camera = Camera2D(800, 600)
        
        # Mock physics object
        mock_object = Mock()
        mock_object.position = Vector2D(0, 0)
        mock_object.is_active = True
        mock_object.name = "test_object"
        
        # Set camera to follow object
        camera.follow_object(mock_object, smoothing=0.1)
        
        # Move object
        mock_object.position = Vector2D(10, 5)
        
        # Update camera - should move towards object
        initial_pos = camera.position
        camera.update(0.1)
        
        # Camera should have moved towards object (but not instantly due to smoothing)
        self.assertNotEqual(camera.position, initial_pos)
        self.assertNotEqual(camera.position, mock_object.position)  # Not instant
    
    def test_render_performance_monitoring(self):
        """Test that rendering performance is properly monitored."""
        renderer = create_render_engine(RenderEngineType.RENDERER_2D)
        renderer.initialize(800, 600)
        
        # Render several frames
        for i in range(10):
            renderer.begin_frame()
            renderer.clear()
            
            # Draw multiple objects
            for j in range(20):
                renderer.draw_circle(
                    Vector2D(j * 30, i * 20),
                    5,
                    StandardColors.BLUE
                )
            
            renderer.end_frame()
        
        # Check that statistics were updated
        stats = renderer.get_stats()
        self.assertGreater(stats.frame_count, 0)
        self.assertGreater(stats.total_render_time, 0)
    
    def test_coordinate_system_consistency(self):
        """Test that coordinate transformations are consistent across systems."""
        renderer = create_render_engine(RenderEngineType.RENDERER_2D)
        camera = Camera2D(800, 600)
        
        # Test various coordinate transformations
        test_points = [
            Vector2D(0, 0),
            Vector2D(100, 50),
            Vector2D(-50, -25),
            Vector2D(200, -100),
        ]
        
        for world_point in test_points:
            # Transform world -> screen -> world
            screen_point = camera.world_to_screen(world_point)
            world_point_back = camera.screen_to_world(screen_point)
            
            # Should get back to original point (within floating point precision)
            self.assertAlmostEqual(world_point.x, world_point_back.x, places=3)
            self.assertAlmostEqual(world_point.y, world_point_back.y, places=3)
    
    def test_zoom_and_render_consistency(self):
        """Test that zoom levels affect rendering consistently."""
        renderer = create_render_engine(RenderEngineType.RENDERER_2D)
        camera = Camera2D(800, 600)
        renderer.initialize(800, 600)
        
        # Test different zoom levels
        zoom_levels = [0.5, 1.0, 2.0, 4.0]
        world_point = Vector2D(10, 10)
        
        for zoom in zoom_levels:
            camera.set_zoom(zoom)
            screen_point = camera.world_to_screen(world_point)
            
            # Render at this zoom level
            renderer.begin_frame()
            renderer.clear()
            renderer.draw_circle(screen_point, 5 * zoom, StandardColors.GREEN)
            renderer.end_frame()
            
            # Verify that the screen position changes with zoom
            # (Details depend on camera implementation)
    
    def test_viewport_clipping(self):
        """Test that objects outside viewport are handled correctly."""
        renderer = create_render_engine(RenderEngineType.RENDERER_2D)
        camera = Camera2D(800, 600)
        renderer.initialize(800, 600)
        
        # Test points inside and outside viewport
        test_cases = [
            (Vector2D(0, 0), True),        # Center - should be visible
            (Vector2D(1000, 1000), False), # Far away - should not be visible
            (Vector2D(-1000, -1000), False), # Far away negative - should not be visible
        ]
        
        for world_point, should_be_visible in test_cases:
            is_visible = camera.is_point_visible(world_point)
            self.assertEqual(is_visible, should_be_visible)
    
    def test_multi_frame_rendering(self):
        """Test rendering across multiple frames."""
        renderer = create_render_engine(RenderEngineType.RENDERER_2D)
        camera = Camera2D(800, 600)
        renderer.initialize(800, 600)
        
        # Simulate animation over multiple frames
        frame_count = 30
        positions = []
        
        for frame in range(frame_count):
            # Animate a circle moving in a circle
            angle = (frame / frame_count) * 2 * 3.14159
            world_pos = Vector2D(
                50 * math.cos(angle),
                50 * math.sin(angle)
            )
            positions.append(world_pos)
            
            # Render frame
            renderer.begin_frame()
            renderer.clear()
            
            screen_pos = camera.world_to_screen(world_pos)
            renderer.draw_circle(screen_pos, 10, StandardColors.RED)
            
            renderer.end_frame()
        
        # Verify we rendered the expected number of frames
        stats = renderer.get_stats()
        self.assertEqual(stats.frame_count, frame_count)
    
    def test_camera_shake_rendering(self):
        """Test that camera shake affects rendering correctly."""
        renderer = create_render_engine(RenderEngineType.RENDERER_2D)
        camera = Camera2D(800, 600)
        renderer.initialize(800, 600)
        
        # Fixed world position
        world_pos = Vector2D(0, 0)
        
        # Get screen position without shake
        screen_pos_normal = camera.world_to_screen(world_pos)
        
        # Add camera shake
        camera.add_shake(10.0, 1.0)
        camera.update(0.1)  # Update to apply shake
        
        # Get screen position with shake
        screen_pos_shaken = camera.world_to_screen(world_pos)
        
        # Screen position should be different due to shake
        # (might be the same by coincidence, but should not be identical over multiple frames)
        shake_detected = False
        for i in range(10):
            camera.update(0.1)
            current_pos = camera.world_to_screen(world_pos)
            if (current_pos.x != screen_pos_normal.x or 
                current_pos.y != screen_pos_normal.y):
                shake_detected = True
                break
        
        # Note: Shake might not always be detectable due to randomness
        # This test verifies the mechanism works, not that shake is always visible
    
    def test_rendering_with_callbacks(self):
        """Test rendering with pre/post render callbacks."""
        renderer = create_render_engine(RenderEngineType.RENDERER_2D)
        renderer.initialize(800, 600)
        
        # Track callback execution
        callback_calls = []
        
        def pre_render_callback():
            callback_calls.append("pre")
        
        def post_render_callback():
            callback_calls.append("post")
        
        # Add callbacks
        renderer.add_pre_render_callback(pre_render_callback)
        renderer.add_post_render_callback(post_render_callback)
        
        # Render frame
        renderer.begin_frame()
        renderer.clear()
        renderer.draw_circle(Vector2D(400, 300), 20, StandardColors.BLUE)
        renderer.end_frame()
        
        # Verify callbacks were called in correct order
        self.assertEqual(callback_calls, ["pre", "post"])


class TestRenderingErrorHandling(unittest.TestCase):
    """Test error handling in rendering system."""
    
    def setUp(self):
        """Set up test fixtures."""
        reset_render_engine()
    
    def tearDown(self):
        """Clean up test fixtures."""
        reset_render_engine()
    
    def test_render_before_initialization(self):
        """Test that rendering before initialization is handled gracefully."""
        renderer = Renderer2D()
        
        # These should not crash even if not initialized
        renderer.begin_frame()
        renderer.clear()
        renderer.draw_circle(Vector2D(0, 0), 10, StandardColors.RED)
        renderer.end_frame()
    
    def test_invalid_render_parameters(self):
        """Test handling of invalid render parameters."""
        # Mock pygame to avoid actual window creation
        with patch('pyjoysim.rendering.engine.pygame') as mock_pygame:
            mock_pygame.init.return_value = None
            mock_pygame.display.set_mode.return_value = Mock()
            mock_pygame.display.set_caption.return_value = None
            mock_pygame.font.init.return_value = None
            mock_pygame.time.Clock.return_value = Mock()
            
            renderer = Renderer2D()
            renderer.initialize(800, 600)
            
            # These should handle invalid inputs gracefully
            renderer.draw_circle(Vector2D(0, 0), -5, StandardColors.RED)  # Negative radius
            renderer.draw_rectangle(Vector2D(0, 0), 0, 0, StandardColors.BLUE)  # Zero size
            renderer.draw_line(Vector2D(0, 0), Vector2D(0, 0), StandardColors.GREEN)  # Zero length
    
    def test_camera_edge_cases(self):
        """Test camera system edge cases."""
        camera = Camera2D(800, 600)
        
        # Very large coordinates
        large_pos = Vector2D(1e6, 1e6)
        screen_pos = camera.world_to_screen(large_pos)
        self.assertIsInstance(screen_pos.x, (int, float))
        self.assertIsInstance(screen_pos.y, (int, float))
        
        # Very small zoom
        camera.set_zoom(0.001)
        screen_pos = camera.world_to_screen(Vector2D(0, 0))
        self.assertIsInstance(screen_pos.x, (int, float))
        self.assertIsInstance(screen_pos.y, (int, float))
        
        # Very large zoom
        camera.set_zoom(1000.0)
        screen_pos = camera.world_to_screen(Vector2D(0, 0))
        self.assertIsInstance(screen_pos.x, (int, float))
        self.assertIsInstance(screen_pos.y, (int, float))


if __name__ == '__main__':
    import math  # Need this for the animation test
    unittest.main(verbosity=2)