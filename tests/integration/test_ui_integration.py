"""
UI Integration Tests

This module provides comprehensive integration tests for the UI system:
- Main window and navigation
- Simulation selection and switching
- Control panel integration
- Settings management
- Performance monitoring
"""

import unittest
import time
import pygame
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

# Import UI components
from pyjoysim.ui.main_window import MainWindow, WindowState
from pyjoysim.ui.simulation_selector import SimulationSelector
from pyjoysim.ui.control_panel import ControlPanel
from pyjoysim.ui.simulation_switcher import SimulationSwitcher
from pyjoysim.ui.settings_manager import SettingsManager, GraphicsSettings


class TestMainWindowIntegration(unittest.TestCase):
    """Test main window integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock pygame to avoid creating actual windows
        self.pygame_patcher = patch('pyjoysim.ui.main_window.pygame')
        self.mock_pygame = self.pygame_patcher.start()
        
        # Mock display and surface
        self.mock_pygame.init.return_value = None
        self.mock_pygame.display.set_mode.return_value = Mock()
        self.mock_pygame.display.set_caption.return_value = None
        self.mock_pygame.time.Clock.return_value = Mock()
        self.mock_pygame.font.Font.return_value = Mock()
        
        # Mock screen surface
        mock_screen = Mock()
        mock_screen.fill.return_value = None
        mock_screen.blit.return_value = None
        mock_screen.get_size.return_value = (1024, 768)
        self.mock_pygame.display.set_mode.return_value = mock_screen
        
        # Mock surface creation
        mock_surface = Mock()
        mock_surface.set_alpha.return_value = None
        mock_surface.fill.return_value = None
        self.mock_pygame.Surface.return_value = mock_surface
        
        # Mock font rendering
        mock_font = Mock()
        mock_text_surface = Mock()
        mock_text_surface.get_rect.return_value = pygame.Rect(0, 0, 100, 20)
        mock_font.render.return_value = mock_text_surface
        self.mock_pygame.font.Font.return_value = mock_font
        
        # Mock input manager
        self.input_patcher = patch('pyjoysim.ui.main_window.InputManager')
        self.mock_input_manager = self.input_patcher.start()
        
        # Mock simulation manager
        self.sim_manager_patcher = patch('pyjoysim.ui.main_window.get_simulation_manager')
        self.mock_sim_manager = self.sim_manager_patcher.start()
        mock_manager = Mock()
        mock_manager.registry.get_all_simulations.return_value = {
            'test_sim': {'display_name': 'Test Simulation', 'category': 'test'}
        }
        self.mock_sim_manager.return_value = mock_manager
        
    def tearDown(self):
        """Clean up test environment."""
        self.pygame_patcher.stop()
        self.input_patcher.stop()
        self.sim_manager_patcher.stop()
    
    def test_main_window_creation(self):
        """Test main window creation and initialization."""
        window = MainWindow(800, 600)
        
        self.assertEqual(window.width, 800)
        self.assertEqual(window.height, 600)
        self.assertEqual(window.state, WindowState.MENU)
        self.assertTrue(window.running)
        self.assertIsNotNone(window.menu_buttons)
        self.assertEqual(len(window.menu_buttons), 4)  # Start, Settings, Help, Quit
    
    def test_state_transitions(self):
        """Test window state transitions."""
        window = MainWindow()
        
        # Test transition to simulation select
        window._change_state(WindowState.SIMULATION_SELECT)
        self.assertEqual(window.state, WindowState.SIMULATION_SELECT)
        self.assertEqual(window.previous_state, WindowState.MENU)
        
        # Test transition back to menu
        window._change_state(WindowState.MENU)
        self.assertEqual(window.state, WindowState.MENU)
        self.assertEqual(window.previous_state, WindowState.SIMULATION_SELECT)
    
    def test_button_actions(self):
        """Test button action execution."""
        window = MainWindow()
        
        # Test start simulation action
        window._execute_button_action("start_simulation")
        self.assertEqual(window.state, WindowState.SIMULATION_SELECT)
        
        # Test back action
        window._execute_button_action("back")
        self.assertEqual(window.state, WindowState.MENU)
        
        # Test quit action
        window._execute_button_action("quit")
        self.assertFalse(window.running)
    
    def test_simulation_creation(self):
        """Test simulation creation from main window."""
        window = MainWindow()
        
        # Mock successful simulation creation
        mock_simulation = Mock()
        mock_simulation.run.return_value = None
        window.simulation_manager.create_simulation.return_value = mock_simulation
        
        # Test simulation starting
        window._start_simulation("test_sim")
        self.assertEqual(window.state, WindowState.SIMULATION_RUNNING)
        self.assertEqual(window.current_simulation, mock_simulation)


class TestSimulationSelectorIntegration(unittest.TestCase):
    """Test simulation selector integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock pygame
        self.pygame_patcher = patch('pyjoysim.ui.simulation_selector.pygame')
        self.mock_pygame = self.pygame_patcher.start()
        
        # Mock font
        mock_font = Mock()
        mock_font.size.return_value = (100, 20)
        mock_font.render.return_value = Mock()
        self.mock_pygame.font.Font.return_value = mock_font
        
        # Mock simulation manager
        self.sim_manager_patcher = patch('pyjoysim.ui.simulation_selector.get_simulation_manager')
        self.mock_sim_manager = self.sim_manager_patcher.start()
        
        mock_registry = Mock()
        mock_registry.get_all_simulations.return_value = {
            'car_sim': {
                'display_name': 'Car Simulation',
                'category': 'vehicle',
                'difficulty': 'beginner',
                'description': 'Drive a car around tracks'
            },
            'robot_sim': {
                'display_name': 'Robot Arm',
                'category': 'robot',
                'difficulty': 'intermediate',
                'description': 'Control a robot arm'
            }
        }
        
        mock_manager = Mock()
        mock_manager.registry = mock_registry
        self.mock_sim_manager.return_value = mock_manager
        
        # Mock input manager
        self.input_patcher = patch('pyjoysim.ui.simulation_selector.InputManager')
        self.mock_input_manager = self.input_patcher.start()
        
    def tearDown(self):
        """Clean up test environment."""
        self.pygame_patcher.stop()
        self.sim_manager_patcher.stop()
        self.input_patcher.stop()
    
    def test_selector_creation(self):
        """Test simulation selector creation."""
        selector = SimulationSelector(1024, 768)
        
        self.assertEqual(selector.width, 1024)
        self.assertEqual(selector.height, 768)
        self.assertEqual(selector.active_filter, "all")
        self.assertIsNotNone(selector.simulation_cards)
        self.assertEqual(len(selector.simulation_cards), 3)  # 2 simulations + back button
    
    def test_category_filtering(self):
        """Test category filtering functionality."""
        selector = SimulationSelector()
        
        # Test filter to vehicle category
        selector.active_filter = "vehicle"
        selector._initialize_simulation_cards()
        
        # Should have 1 vehicle simulation + back button
        vehicle_cards = [card for card in selector.simulation_cards 
                        if not card.name.startswith("back")]
        self.assertEqual(len(vehicle_cards), 1)
        self.assertEqual(vehicle_cards[0].name, "car_sim")
    
    def test_simulation_selection(self):
        """Test simulation selection logic."""
        selector = SimulationSelector()
        
        # Test simulation selection
        selector.selected_simulation = "car_sim"
        selector._update_selection_states()
        
        # Check that correct card is marked as selected
        car_card = next((card for card in selector.simulation_cards 
                        if card.name == "car_sim"), None)
        self.assertIsNotNone(car_card)
        self.assertTrue(car_card.selected)
    
    def test_mouse_interaction(self):
        """Test mouse interaction handling."""
        selector = SimulationSelector()
        
        # Mock mouse click event
        mock_event = Mock()
        mock_event.type = pygame.MOUSEBUTTONDOWN
        mock_event.button = 1
        
        # Set mouse position over a simulation card
        if selector.simulation_cards:
            card = selector.simulation_cards[0]
            selector.mouse_pos = (card.position.centerx, card.position.centery)
            
            # Handle the event
            result = selector.handle_event(mock_event)
            
            # Should not immediately run simulation (single click)
            self.assertIsNone(result)


class TestControlPanelIntegration(unittest.TestCase):
    """Test control panel integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock pygame
        self.pygame_patcher = patch('pyjoysim.ui.control_panel.pygame')
        self.mock_pygame = self.pygame_patcher.start()
        
        # Mock font
        mock_font = Mock()
        mock_font.render.return_value = Mock()
        self.mock_pygame.font.Font.return_value = mock_font
        
        # Mock psutil for performance monitoring
        self.psutil_patcher = patch('pyjoysim.ui.control_panel.psutil')
        self.mock_psutil = self.psutil_patcher.start()
        
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        self.mock_psutil.Process.return_value = mock_process
        self.mock_psutil.cpu_percent.return_value = 25.0
        
    def tearDown(self):
        """Clean up test environment."""
        self.pygame_patcher.stop()
        self.psutil_patcher.stop()
    
    def test_control_panel_creation(self):
        """Test control panel creation."""
        panel = ControlPanel(300, 400)
        
        self.assertEqual(panel.width, 300)
        self.assertEqual(panel.height, 400)
        self.assertIsNotNone(panel.control_buttons)
        self.assertEqual(len(panel.control_buttons), 4)  # Pause, Resume, Reset, Stop
    
    def test_performance_metrics_update(self):
        """Test performance metrics updating."""
        panel = ControlPanel()
        
        # Update metrics with test data
        panel._update_performance_metrics(60.0)
        
        # Check that metrics were recorded
        self.assertEqual(len(panel.fps_history), 1)
        self.assertEqual(panel.fps_history[0], 60.0)
        self.assertEqual(len(panel.memory_history), 1)
        self.assertAlmostEqual(panel.memory_history[0], 100.0, places=1)
    
    def test_simulation_control(self):
        """Test simulation control functionality."""
        panel = ControlPanel()
        
        # Mock simulation
        mock_simulation = Mock()
        mock_simulation.is_running.return_value = True
        mock_simulation.is_paused = False
        
        panel.set_simulation(mock_simulation)
        panel._update_button_states()
        
        # Check button states
        pause_button = next((btn for btn in panel.control_buttons if btn.action == "pause"), None)
        self.assertIsNotNone(pause_button)
        self.assertTrue(pause_button.enabled)
        
        resume_button = next((btn for btn in panel.control_buttons if btn.action == "resume"), None)
        self.assertIsNotNone(resume_button)
        self.assertFalse(resume_button.enabled)  # Should be disabled when not paused


class TestSimulationSwitcherIntegration(unittest.TestCase):
    """Test simulation switcher integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock pygame
        self.pygame_patcher = patch('pyjoysim.ui.simulation_switcher.pygame')
        self.mock_pygame = self.pygame_patcher.start()
        
        # Mock font
        mock_font = Mock()
        mock_font.render.return_value = Mock()
        self.mock_pygame.font.Font.return_value = mock_font
        
        # Mock simulation manager
        self.sim_manager_patcher = patch('pyjoysim.ui.simulation_switcher.get_simulation_manager')
        self.mock_sim_manager = self.sim_manager_patcher.start()
        
        mock_registry = Mock()
        mock_registry.has_simulation.return_value = True
        mock_registry.get_simulation_metadata.return_value = {
            'display_name': 'Test Simulation'
        }
        
        mock_manager = Mock()
        mock_manager.registry = mock_registry
        mock_manager.create_simulation.return_value = Mock()
        self.mock_sim_manager.return_value = mock_manager
        
    def tearDown(self):
        """Clean up test environment."""
        self.pygame_patcher.stop()
        self.sim_manager_patcher.stop()
    
    def test_switcher_creation(self):
        """Test simulation switcher creation."""
        switcher = SimulationSwitcher(1024, 768)
        
        self.assertEqual(switcher.width, 1024)
        self.assertEqual(switcher.height, 768)
        self.assertFalse(switcher.is_switching())
    
    def test_simulation_switch_initiation(self):
        """Test simulation switch initiation."""
        switcher = SimulationSwitcher()
        
        # Test valid simulation switch
        result = switcher.switch_to_simulation("test_sim")
        self.assertTrue(result)
        self.assertTrue(switcher.is_switching())
        self.assertEqual(switcher.target_simulation_name, "test_sim")
    
    def test_switch_state_machine(self):
        """Test simulation switch state machine."""
        switcher = SimulationSwitcher()
        
        # Start a switch
        switcher.switch_to_simulation("test_sim")
        
        # Should be in loading state (no current simulation)
        self.assertEqual(switcher.state.name, "LOADING_NEW")
        
        # Update should progress through states
        switcher.update(0.016)
        
        # Should eventually complete
        # Note: In real scenario this would take multiple frames


class TestSettingsManagerIntegration(unittest.TestCase):
    """Test settings manager integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for settings
        self.temp_dir = Path(tempfile.mkdtemp())
        self.settings_manager = SettingsManager(config_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_settings_creation_and_defaults(self):
        """Test settings manager creation with defaults."""
        self.assertIsNotNone(self.settings_manager.graphics)
        self.assertIsNotNone(self.settings_manager.audio)
        self.assertIsNotNone(self.settings_manager.input)
        self.assertIsNotNone(self.settings_manager.simulation)
        self.assertIsNotNone(self.settings_manager.ui)
        
        # Check default values
        self.assertEqual(self.settings_manager.graphics.window_width, 1024)
        self.assertEqual(self.settings_manager.graphics.window_height, 768)
        self.assertEqual(self.settings_manager.audio.master_volume, 1.0)
    
    def test_setting_get_and_set(self):
        """Test getting and setting individual settings."""
        # Test getting default value
        width = self.settings_manager.get_setting("graphics.window_width")
        self.assertEqual(width, 1024)
        
        # Test setting new value
        result = self.settings_manager.set_setting("graphics.window_width", 1920)
        self.assertTrue(result)
        
        # Verify new value
        new_width = self.settings_manager.get_setting("graphics.window_width")
        self.assertEqual(new_width, 1920)
    
    def test_setting_validation(self):
        """Test setting value validation."""
        # Test valid value
        result = self.settings_manager.set_setting("graphics.window_width", 1920)
        self.assertTrue(result)
        
        # Test invalid value (too small)
        result = self.settings_manager.set_setting("graphics.window_width", 300)
        self.assertFalse(result)
        
        # Test invalid type
        result = self.settings_manager.set_setting("graphics.window_width", "invalid")
        self.assertFalse(result)
    
    def test_settings_persistence(self):
        """Test settings save and load."""
        # Modify settings
        self.settings_manager.set_setting("graphics.window_width", 1920)
        self.settings_manager.set_setting("audio.master_volume", 0.8)
        
        # Save settings
        result = self.settings_manager.save_settings()
        self.assertTrue(result)
        
        # Verify file was created
        self.assertTrue(self.settings_manager.config_file.exists())
        
        # Create new settings manager and load
        new_manager = SettingsManager(config_dir=self.temp_dir)
        
        # Verify loaded values
        self.assertEqual(new_manager.get_setting("graphics.window_width"), 1920)
        self.assertEqual(new_manager.get_setting("audio.master_volume"), 0.8)
    
    def test_settings_export_import(self):
        """Test settings export and import."""
        # Modify settings
        self.settings_manager.set_setting("graphics.window_width", 1600)
        self.settings_manager.set_setting("ui.theme", "light")
        
        # Export settings
        export_file = self.temp_dir / "exported_settings.json"
        result = self.settings_manager.export_settings(export_file)
        self.assertTrue(result)
        self.assertTrue(export_file.exists())
        
        # Reset to defaults
        self.settings_manager.reset_to_defaults()
        self.assertEqual(self.settings_manager.get_setting("graphics.window_width"), 1024)
        
        # Import settings
        result = self.settings_manager.import_settings(export_file)
        self.assertTrue(result)
        
        # Verify imported values
        self.assertEqual(self.settings_manager.get_setting("graphics.window_width"), 1600)
        self.assertEqual(self.settings_manager.get_setting("ui.theme"), "light")


class TestUIPerformance(unittest.TestCase):
    """Test UI performance and optimization."""
    
    def setUp(self):
        """Set up performance test environment."""
        # Mock pygame
        self.pygame_patcher = patch('pygame')
        self.mock_pygame = self.pygame_patcher.start()
        
        # Mock all necessary pygame components
        self.mock_pygame.init.return_value = None
        self.mock_pygame.display.set_mode.return_value = Mock()
        self.mock_pygame.display.set_caption.return_value = None
        self.mock_pygame.time.Clock.return_value = Mock()
        self.mock_pygame.font.Font.return_value = Mock()
        self.mock_pygame.Surface.return_value = Mock()
        
    def tearDown(self):
        """Clean up performance test environment."""
        self.pygame_patcher.stop()
    
    def test_ui_rendering_performance(self):
        """Test UI rendering performance."""
        # This would be a more comprehensive test in a real scenario
        # For now, just test that components can be created without performance issues
        
        start_time = time.time()
        
        # Create UI components
        with patch('pyjoysim.ui.main_window.InputManager'), \
             patch('pyjoysim.ui.main_window.get_simulation_manager'):
            window = MainWindow()
        
        with patch('pyjoysim.ui.simulation_selector.get_simulation_manager'), \
             patch('pyjoysim.ui.simulation_selector.InputManager'):
            selector = SimulationSelector()
        
        panel = ControlPanel()
        switcher = SimulationSwitcher()
        
        creation_time = time.time() - start_time
        
        # UI components should be created quickly (less than 1 second)
        self.assertLess(creation_time, 1.0)
    
    def test_memory_usage(self):
        """Test memory usage of UI components."""
        # This is a basic test - in production you'd use more sophisticated memory profiling
        import sys
        
        start_size = sys.getsizeof({})  # Baseline
        
        # Create components
        with patch('pyjoysim.ui.main_window.InputManager'), \
             patch('pyjoysim.ui.main_window.get_simulation_manager'):
            components = [MainWindow() for _ in range(10)]
        
        # Memory usage should be reasonable
        total_size = sum(sys.getsizeof(comp.__dict__) for comp in components)
        average_size = total_size / len(components)
        
        # Each component should use less than 1MB of direct memory
        self.assertLess(average_size, 1024 * 1024)


if __name__ == '__main__':
    # Configure test environment
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Prevent actual window creation
    
    unittest.main(verbosity=2)