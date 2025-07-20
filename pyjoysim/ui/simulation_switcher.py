"""
Simulation Switcher Implementation

This module provides seamless switching between simulations with:
- Graceful simulation lifecycle management
- State preservation and restoration
- Transition animations
- Resource cleanup and initialization
"""

import pygame
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from pyjoysim.simulation import (
    BaseSimulation, SimulationManager, get_simulation_manager, SimulationConfig
)
from pyjoysim.rendering import Color


class SwitchState(Enum):
    """Simulation switcher states."""
    IDLE = "idle"
    STOPPING_CURRENT = "stopping_current"
    LOADING_NEW = "loading_new"
    STARTING_NEW = "starting_new"
    TRANSITIONING = "transitioning"


@dataclass
class SimulationSnapshot:
    """Snapshot of simulation state for restoration."""
    name: str
    config: Dict[str, Any]
    state_data: Dict[str, Any]
    timestamp: float


@dataclass
class TransitionConfig:
    """Configuration for simulation transitions."""
    fade_duration: float = 1.0
    loading_timeout: float = 10.0
    cleanup_timeout: float = 5.0
    show_loading_screen: bool = True
    preserve_state: bool = False


class SimulationSwitcher:
    """
    Manages smooth transitions between simulations.
    
    Features:
    - Graceful simulation stopping and starting
    - State preservation and restoration
    - Loading screens and transition effects
    - Error handling and recovery
    - Resource management
    """
    
    def __init__(self, width: int = 1024, height: int = 768):
        self.width = width
        self.height = height
        
        # State management
        self.state = SwitchState.IDLE
        self.current_simulation = None
        self.target_simulation_name = None
        self.target_simulation = None
        
        # Transition configuration
        self.transition_config = TransitionConfig()
        
        # Simulation management
        self.simulation_manager = get_simulation_manager()
        self.snapshots = {}  # Dict[str, SimulationSnapshot]
        
        # Animation and timing
        self.transition_start_time = 0
        self.fade_alpha = 0
        self.loading_progress = 0
        self.error_message = None
        
        # Callbacks
        self.on_switch_complete = None
        self.on_switch_error = None
        
        # Colors and fonts
        self.colors = {
            'background': Color(20, 20, 30),
            'loading_bg': Color(40, 40, 50),
            'progress_bar': Color(64, 128, 255),
            'progress_bg': Color(60, 60, 70),
            'text': Color(255, 255, 255),
            'text_secondary': Color(180, 180, 180),
            'error': Color(244, 67, 54),
            'overlay': Color(0, 0, 0, 128)
        }
        
        self.fonts = {
            'title': pygame.font.Font(None, 36),
            'subtitle': pygame.font.Font(None, 24),
            'body': pygame.font.Font(None, 18)
        }
        
        # Performance tracking
        self.switch_start_time = 0
        self.switch_stats = {
            'total_switches': 0,
            'successful_switches': 0,
            'average_switch_time': 0,
            'last_switch_time': 0
        }
    
    def set_current_simulation(self, simulation: BaseSimulation) -> None:
        """Set the currently active simulation."""
        self.current_simulation = simulation
        self.state = SwitchState.IDLE
    
    def switch_to_simulation(self, simulation_name: str, 
                           config: Optional[SimulationConfig] = None,
                           preserve_current_state: bool = False) -> bool:
        """
        Initiate switch to a new simulation.
        
        Args:
            simulation_name: Name of simulation to switch to
            config: Optional configuration for the new simulation
            preserve_current_state: Whether to save current simulation state
            
        Returns:
            True if switch initiated successfully, False otherwise
        """
        if self.state != SwitchState.IDLE:
            print(f"Cannot switch: currently in state {self.state}")
            return False
        
        if not self.simulation_manager.registry.has_simulation(simulation_name):
            print(f"Simulation '{simulation_name}' not found")
            return False
        
        # Record switch statistics
        self.switch_start_time = time.time()
        self.switch_stats['total_switches'] += 1
        
        # Preserve current state if requested
        if preserve_current_state and self.current_simulation:
            self._create_snapshot(self.current_simulation)
        
        # Begin transition
        self.target_simulation_name = simulation_name
        self.transition_start_time = time.time()
        self.fade_alpha = 0
        self.loading_progress = 0
        self.error_message = None
        
        if self.current_simulation:
            self.state = SwitchState.STOPPING_CURRENT
        else:
            self.state = SwitchState.LOADING_NEW
        
        return True
    
    def _create_snapshot(self, simulation: BaseSimulation) -> None:
        """Create a snapshot of current simulation state."""
        try:
            snapshot = SimulationSnapshot(
                name=simulation.name,
                config=getattr(simulation, 'config', {}).__dict__ if hasattr(simulation, 'config') else {},
                state_data=self._extract_simulation_state(simulation),
                timestamp=time.time()
            )
            self.snapshots[simulation.name] = snapshot
            
        except Exception as e:
            print(f"Failed to create simulation snapshot: {e}")
    
    def _extract_simulation_state(self, simulation: BaseSimulation) -> Dict[str, Any]:
        """Extract serializable state data from simulation."""
        state_data = {}
        
        try:
            # Common state properties to preserve
            state_properties = [
                'physics_world', 'render_engine', 'input_events',
                'simulation_time', 'frame_count', 'statistics'
            ]
            
            for prop in state_properties:
                if hasattr(simulation, prop):
                    value = getattr(simulation, prop)
                    # Only include serializable data
                    if isinstance(value, (int, float, str, bool, list, dict)):
                        state_data[prop] = value
            
            # Simulation-specific state extraction
            if hasattr(simulation, 'get_state_data'):
                custom_state = simulation.get_state_data()
                if isinstance(custom_state, dict):
                    state_data.update(custom_state)
                    
        except Exception as e:
            print(f"Error extracting simulation state: {e}")
        
        return state_data
    
    def _restore_simulation_state(self, simulation: BaseSimulation, 
                                 snapshot: SimulationSnapshot) -> None:
        """Restore simulation state from snapshot."""
        try:
            for key, value in snapshot.state_data.items():
                if hasattr(simulation, key):
                    setattr(simulation, key, value)
            
            # Simulation-specific state restoration
            if hasattr(simulation, 'restore_state_data'):
                simulation.restore_state_data(snapshot.state_data)
                
        except Exception as e:
            print(f"Error restoring simulation state: {e}")
    
    def update(self, dt: float) -> None:
        """Update simulation switcher state machine."""
        current_time = time.time()
        
        if self.state == SwitchState.STOPPING_CURRENT:
            self._update_stopping_current(dt, current_time)
        elif self.state == SwitchState.LOADING_NEW:
            self._update_loading_new(dt, current_time)
        elif self.state == SwitchState.STARTING_NEW:
            self._update_starting_new(dt, current_time)
        elif self.state == SwitchState.TRANSITIONING:
            self._update_transitioning(dt, current_time)
        
        # Update fade animation
        if self.state != SwitchState.IDLE:
            transition_progress = min(1.0, (current_time - self.transition_start_time) / 
                                    self.transition_config.fade_duration)
            
            if self.state in [SwitchState.STOPPING_CURRENT, SwitchState.LOADING_NEW]:
                self.fade_alpha = int(255 * transition_progress)
            elif self.state == SwitchState.TRANSITIONING:
                self.fade_alpha = int(255 * (1.0 - transition_progress))
    
    def _update_stopping_current(self, dt: float, current_time: float) -> None:
        """Update stopping current simulation phase."""
        try:
            if self.current_simulation:
                # Gracefully stop current simulation
                if hasattr(self.current_simulation, 'stop'):
                    self.current_simulation.stop()
                
                # Clean up resources
                if hasattr(self.current_simulation, 'cleanup'):
                    self.current_simulation.cleanup()
            
            self.current_simulation = None
            self.state = SwitchState.LOADING_NEW
            self.loading_progress = 0.2  # 20% progress after stopping
            
        except Exception as e:
            self._handle_switch_error(f"Error stopping simulation: {e}")
    
    def _update_loading_new(self, dt: float, current_time: float) -> None:
        """Update loading new simulation phase."""
        try:
            # Simulate loading progress
            elapsed = current_time - self.transition_start_time
            self.loading_progress = min(0.8, 0.2 + (elapsed / 2.0) * 0.6)  # 20% to 80%
            
            if not self.target_simulation:
                # Create new simulation instance
                self.target_simulation = self.simulation_manager.create_simulation(
                    self.target_simulation_name
                )
                
                if not self.target_simulation:
                    self._handle_switch_error(f"Failed to create simulation: {self.target_simulation_name}")
                    return
            
            # Check if we should restore from snapshot
            if (self.transition_config.preserve_state and 
                self.target_simulation_name in self.snapshots):
                snapshot = self.snapshots[self.target_simulation_name]
                self._restore_simulation_state(self.target_simulation, snapshot)
            
            self.state = SwitchState.STARTING_NEW
            
        except Exception as e:
            self._handle_switch_error(f"Error loading simulation: {e}")
    
    def _update_starting_new(self, dt: float, current_time: float) -> None:
        """Update starting new simulation phase."""
        try:
            if self.target_simulation:
                # Initialize new simulation
                if hasattr(self.target_simulation, 'initialize'):
                    self.target_simulation.initialize()
                
                self.current_simulation = self.target_simulation
                self.target_simulation = None
                
                self.loading_progress = 1.0  # 100% complete
                self.state = SwitchState.TRANSITIONING
                
                # Record successful switch
                switch_time = time.time() - self.switch_start_time
                self.switch_stats['successful_switches'] += 1
                self.switch_stats['last_switch_time'] = switch_time
                
                # Update average switch time
                total_successful = self.switch_stats['successful_switches']
                current_avg = self.switch_stats['average_switch_time']
                self.switch_stats['average_switch_time'] = (
                    (current_avg * (total_successful - 1) + switch_time) / total_successful
                )
                
        except Exception as e:
            self._handle_switch_error(f"Error starting simulation: {e}")
    
    def _update_transitioning(self, dt: float, current_time: float) -> None:
        """Update transition fade-out phase."""
        elapsed = current_time - self.transition_start_time
        
        if elapsed >= self.transition_config.fade_duration:
            # Transition complete
            self.state = SwitchState.IDLE
            self.fade_alpha = 0
            
            if self.on_switch_complete:
                self.on_switch_complete(self.current_simulation)
    
    def _handle_switch_error(self, error_message: str) -> None:
        """Handle simulation switch errors."""
        print(f"Simulation switch error: {error_message}")
        self.error_message = error_message
        self.state = SwitchState.IDLE
        
        # Clean up
        if self.target_simulation:
            try:
                if hasattr(self.target_simulation, 'cleanup'):
                    self.target_simulation.cleanup()
            except:
                pass
            self.target_simulation = None
        
        if self.on_switch_error:
            self.on_switch_error(error_message)
    
    def render_transition_overlay(self, screen: pygame.Surface) -> None:
        """Render transition overlay effects."""
        if self.state == SwitchState.IDLE:
            return
        
        # Fade overlay
        if self.fade_alpha > 0:
            fade_surface = pygame.Surface((self.width, self.height))
            fade_surface.set_alpha(self.fade_alpha)
            fade_surface.fill(self.colors['overlay'].to_tuple()[:3])
            screen.blit(fade_surface, (0, 0))
        
        # Loading screen
        if (self.state in [SwitchState.LOADING_NEW, SwitchState.STARTING_NEW] and 
            self.transition_config.show_loading_screen):
            self._render_loading_screen(screen)
        
        # Error message
        if self.error_message:
            self._render_error_message(screen)
    
    def _render_loading_screen(self, screen: pygame.Surface) -> None:
        """Render loading screen."""
        # Loading panel
        panel_width = 400
        panel_height = 200
        panel_x = (self.width - panel_width) // 2
        panel_y = (self.height - panel_height) // 2
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(screen, self.colors['loading_bg'].to_tuple(), 
                        panel_rect, border_radius=10)
        pygame.draw.rect(screen, self.colors['text'].to_tuple(), 
                        panel_rect, 2, border_radius=10)
        
        # Loading title
        title_text = self.fonts['title'].render("로딩 중...", True, 
                                              self.colors['text'].to_tuple())
        title_rect = title_text.get_rect(center=(panel_x + panel_width // 2, 
                                                panel_y + 40))
        screen.blit(title_text, title_rect)
        
        # Simulation name
        if self.target_simulation_name:
            sim_metadata = self.simulation_manager.registry.get_simulation_metadata(
                self.target_simulation_name)
            display_name = sim_metadata.get('display_name', self.target_simulation_name) if sim_metadata else self.target_simulation_name
            
            name_text = self.fonts['subtitle'].render(display_name, True, 
                                                    self.colors['text_secondary'].to_tuple())
            name_rect = name_text.get_rect(center=(panel_x + panel_width // 2, 
                                                 panel_y + 80))
            screen.blit(name_text, name_rect)
        
        # Progress bar
        progress_width = 300
        progress_height = 20
        progress_x = panel_x + (panel_width - progress_width) // 2
        progress_y = panel_y + 120
        
        # Progress background
        progress_bg_rect = pygame.Rect(progress_x, progress_y, progress_width, progress_height)
        pygame.draw.rect(screen, self.colors['progress_bg'].to_tuple(), 
                        progress_bg_rect, border_radius=10)
        
        # Progress fill
        fill_width = int(progress_width * self.loading_progress)
        if fill_width > 0:
            progress_fill_rect = pygame.Rect(progress_x, progress_y, fill_width, progress_height)
            pygame.draw.rect(screen, self.colors['progress_bar'].to_tuple(), 
                           progress_fill_rect, border_radius=10)
        
        # Progress text
        progress_text = f"{int(self.loading_progress * 100)}%"
        progress_surface = self.fonts['body'].render(progress_text, True, 
                                                   self.colors['text'].to_tuple())
        progress_text_rect = progress_surface.get_rect(center=(
            progress_x + progress_width // 2, 
            progress_y + progress_height // 2
        ))
        screen.blit(progress_surface, progress_text_rect)
        
        # Status text
        status_texts = {
            SwitchState.STOPPING_CURRENT: "현재 시뮬레이션 종료 중...",
            SwitchState.LOADING_NEW: "새 시뮬레이션 로딩 중...",
            SwitchState.STARTING_NEW: "시뮬레이션 초기화 중..."
        }
        
        status_text = status_texts.get(self.state, "처리 중...")
        status_surface = self.fonts['body'].render(status_text, True, 
                                                 self.colors['text_secondary'].to_tuple())
        status_rect = status_surface.get_rect(center=(panel_x + panel_width // 2, 
                                                    panel_y + 160))
        screen.blit(status_surface, status_rect)
    
    def _render_error_message(self, screen: pygame.Surface) -> None:
        """Render error message overlay."""
        # Error panel
        panel_width = 450
        panel_height = 150
        panel_x = (self.width - panel_width) // 2
        panel_y = (self.height - panel_height) // 2
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(screen, self.colors['loading_bg'].to_tuple(), 
                        panel_rect, border_radius=10)
        pygame.draw.rect(screen, self.colors['error'].to_tuple(), 
                        panel_rect, 2, border_radius=10)
        
        # Error title
        title_text = self.fonts['title'].render("오류 발생", True, 
                                              self.colors['error'].to_tuple())
        title_rect = title_text.get_rect(center=(panel_x + panel_width // 2, 
                                                panel_y + 30))
        screen.blit(title_text, title_rect)
        
        # Error message
        if self.error_message:
            # Wrap error message
            words = self.error_message.split(' ')
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                if self.fonts['body'].size(test_line)[0] <= panel_width - 40:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Render error message lines
            start_y = panel_y + 70
            for i, line in enumerate(lines[:2]):  # Limit to 2 lines
                line_surface = self.fonts['body'].render(line, True, 
                                                       self.colors['text'].to_tuple())
                line_rect = line_surface.get_rect(center=(panel_x + panel_width // 2, 
                                                        start_y + i * 25))
                screen.blit(line_surface, line_rect)
    
    def get_switch_statistics(self) -> Dict[str, Any]:
        """Get simulation switching statistics."""
        return self.switch_stats.copy()
    
    def clear_snapshots(self) -> None:
        """Clear all simulation snapshots."""
        self.snapshots.clear()
    
    def is_switching(self) -> bool:
        """Check if currently switching simulations."""
        return self.state != SwitchState.IDLE


def create_simulation_switcher(width: int = 1024, height: int = 768) -> SimulationSwitcher:
    """Create and configure a simulation switcher."""
    return SimulationSwitcher(width, height)