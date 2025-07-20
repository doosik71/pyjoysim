"""
Control Panel Implementation

This module provides a real-time control panel for simulation management with:
- Live simulation status monitoring
- Performance metrics display
- Interactive controls (pause/resume/reset)
- Debug information overlay
"""

import pygame
import time
import psutil
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from pyjoysim.simulation import BaseSimulation
from pyjoysim.rendering import Color


class PanelState(Enum):
    """Control panel states."""
    MINIMIZED = "minimized"
    EXPANDED = "expanded"
    FLOATING = "floating"


@dataclass
class ControlButton:
    """Control button configuration."""
    text: str
    action: str
    position: pygame.Rect
    color: Color
    enabled: bool = True
    hover: bool = False


@dataclass
class MetricDisplay:
    """Performance metric display configuration."""
    name: str
    value: str
    unit: str
    color: Color
    position: tuple


class ControlPanel:
    """
    Real-time simulation control panel.
    
    Features:
    - Live performance monitoring (FPS, memory, CPU)
    - Simulation state controls
    - Debug information display
    - Interactive parameter adjustment
    """
    
    def __init__(self, width: int = 300, height: int = 400):
        self.width = width
        self.height = height
        
        # Panel state
        self.state = PanelState.EXPANDED
        self.position = pygame.Rect(10, 10, width, height)
        self.dragging = False
        self.drag_offset = (0, 0)
        
        # Colors
        self.colors = {
            'background': Color(20, 20, 30, 200),
            'header': Color(40, 40, 50),
            'button': Color(64, 128, 255),
            'button_hover': Color(80, 144, 255),
            'button_disabled': Color(100, 100, 100),
            'success': Color(76, 175, 80),
            'warning': Color(255, 193, 7),
            'error': Color(244, 67, 54),
            'text': Color(255, 255, 255),
            'text_secondary': Color(180, 180, 180)
        }
        
        # Fonts
        self.fonts = {
            'title': pygame.font.Font(None, 20),
            'body': pygame.font.Font(None, 16),
            'small': pygame.font.Font(None, 14)
        }
        
        # Control buttons
        self.control_buttons = []
        self.minimized_button = None
        
        # Performance metrics
        self.fps_history = []
        self.memory_history = []
        self.cpu_history = []
        self.max_history_length = 60  # 1 second at 60 FPS
        
        # Simulation reference
        self.simulation = None
        
        # Update timers
        self.last_update_time = time.time()
        self.metrics_update_interval = 0.1  # Update metrics every 100ms
        self.last_metrics_update = 0
        
        # Mouse state
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        
        self._initialize_buttons()
    
    def _initialize_buttons(self) -> None:
        """Initialize control buttons."""
        button_width = 80
        button_height = 25
        button_margin = 5
        start_x = 10
        start_y = 40
        
        self.control_buttons = [
            ControlButton(
                text="일시정지",
                action="pause",
                position=pygame.Rect(start_x, start_y, button_width, button_height),
                color=self.colors['warning']
            ),
            ControlButton(
                text="재시작",
                action="resume",
                position=pygame.Rect(start_x + button_width + button_margin, start_y, 
                                   button_width, button_height),
                color=self.colors['success']
            ),
            ControlButton(
                text="리셋",
                action="reset",
                position=pygame.Rect(start_x, start_y + button_height + button_margin, 
                                   button_width, button_height),
                color=self.colors['error']
            ),
            ControlButton(
                text="종료",
                action="stop",
                position=pygame.Rect(start_x + button_width + button_margin, 
                                   start_y + button_height + button_margin, 
                                   button_width, button_height),
                color=self.colors['button']
            )
        ]
        
        # Minimized toggle button
        self.minimized_button = ControlButton(
            text="●",
            action="toggle_minimize",
            position=pygame.Rect(10, 10, 30, 20),
            color=self.colors['button']
        )
    
    def set_simulation(self, simulation: BaseSimulation) -> None:
        """Set the simulation to monitor and control."""
        self.simulation = simulation
    
    def update(self, dt: float, current_fps: float) -> None:
        """Update control panel state and metrics."""
        current_time = time.time()
        
        # Update performance metrics
        if current_time - self.last_metrics_update > self.metrics_update_interval:
            self._update_performance_metrics(current_fps)
            self.last_metrics_update = current_time
        
        # Update button states based on simulation state
        self._update_button_states()
        
        self.last_update_time = current_time
    
    def _update_performance_metrics(self, current_fps: float) -> None:
        """Update performance metrics history."""
        # FPS
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.max_history_length:
            self.fps_history.pop(0)
        
        # Memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_history.append(memory_mb)
            if len(self.memory_history) > self.max_history_length:
                self.memory_history.pop(0)
        except:
            pass
        
        # CPU usage
        try:
            cpu_percent = psutil.cpu_percent()
            self.cpu_history.append(cpu_percent)
            if len(self.cpu_history) > self.max_history_length:
                self.cpu_history.pop(0)
        except:
            pass
    
    def _update_button_states(self) -> None:
        """Update button enabled states based on simulation state."""
        if not self.simulation:
            for button in self.control_buttons:
                button.enabled = False
            return
        
        is_running = self.simulation.is_running() if hasattr(self.simulation, 'is_running') else True
        is_paused = getattr(self.simulation, 'is_paused', False)
        
        for button in self.control_buttons:
            if button.action == "pause":
                button.enabled = is_running and not is_paused
                button.text = "일시정지" if not is_paused else "일시정지됨"
            elif button.action == "resume":
                button.enabled = is_running and is_paused
            elif button.action == "reset":
                button.enabled = is_running
            elif button.action == "stop":
                button.enabled = is_running
    
    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """
        Handle pygame events.
        
        Returns:
            Action string if an action should be taken, None otherwise.
        """
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.mouse_clicked = True
                return self._handle_mouse_click()
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION:
            self.mouse_pos = event.pos
            
            if self.dragging:
                # Update panel position while dragging
                new_x = event.pos[0] - self.drag_offset[0]
                new_y = event.pos[1] - self.drag_offset[1]
                self.position.x = max(0, min(new_x, 1024 - self.position.width))  # Constrain to screen
                self.position.y = max(0, min(new_y, 768 - self.position.height))
            else:
                self._update_hover_states()
        
        return None
    
    def _handle_mouse_click(self) -> Optional[str]:
        """Handle mouse click events."""
        if self.state == PanelState.MINIMIZED:
            # Only check minimized button
            if self.minimized_button.position.collidepoint(self.mouse_pos):
                self.state = PanelState.EXPANDED
                return None
        else:
            # Check if clicking on header (for dragging)
            header_rect = pygame.Rect(self.position.x, self.position.y, self.position.width, 30)
            if header_rect.collidepoint(self.mouse_pos):
                self.dragging = True
                self.drag_offset = (
                    self.mouse_pos[0] - self.position.x,
                    self.mouse_pos[1] - self.position.y
                )
                return None
            
            # Check minimize button
            minimize_btn_rect = pygame.Rect(
                self.position.x + self.position.width - 25, 
                self.position.y + 5, 20, 20
            )
            if minimize_btn_rect.collidepoint(self.mouse_pos):
                self.state = PanelState.MINIMIZED
                return None
            
            # Check control buttons
            for button in self.control_buttons:
                adjusted_rect = pygame.Rect(
                    self.position.x + button.position.x,
                    self.position.y + button.position.y,
                    button.position.width,
                    button.position.height
                )
                
                if adjusted_rect.collidepoint(self.mouse_pos) and button.enabled:
                    return button.action
        
        return None
    
    def _update_hover_states(self) -> None:
        """Update hover states for buttons."""
        if self.state == PanelState.MINIMIZED:
            return
        
        for button in self.control_buttons:
            adjusted_rect = pygame.Rect(
                self.position.x + button.position.x,
                self.position.y + button.position.y,
                button.position.width,
                button.position.height
            )
            button.hover = adjusted_rect.collidepoint(self.mouse_pos)
    
    def render(self, screen: pygame.Surface) -> None:
        """Render control panel."""
        if self.state == PanelState.MINIMIZED:
            self._render_minimized(screen)
        else:
            self._render_expanded(screen)
    
    def _render_minimized(self, screen: pygame.Surface) -> None:
        """Render minimized control panel."""
        # Small button to expand
        button_rect = self.minimized_button.position
        pygame.draw.rect(screen, self.colors['background'].to_tuple(), button_rect, border_radius=5)
        pygame.draw.rect(screen, self.colors['button'].to_tuple(), button_rect, 2, border_radius=5)
        
        # Icon
        icon_text = self.fonts['small'].render("◢", True, self.colors['text'].to_tuple())
        icon_rect = icon_text.get_rect(center=button_rect.center)
        screen.blit(icon_text, icon_rect)
    
    def _render_expanded(self, screen: pygame.Surface) -> None:
        """Render expanded control panel."""
        # Panel background
        pygame.draw.rect(screen, self.colors['background'].to_tuple(), 
                        self.position, border_radius=10)
        pygame.draw.rect(screen, self.colors['header'].to_tuple(), 
                        self.position, 2, border_radius=10)
        
        # Header
        header_rect = pygame.Rect(self.position.x, self.position.y, 
                                 self.position.width, 30)
        pygame.draw.rect(screen, self.colors['header'].to_tuple(), 
                        header_rect, border_radius=10)
        
        # Title
        title_text = self.fonts['title'].render("제어 패널", True, 
                                              self.colors['text'].to_tuple())
        screen.blit(title_text, (self.position.x + 10, self.position.y + 8))
        
        # Minimize button
        minimize_btn_rect = pygame.Rect(
            self.position.x + self.position.width - 25, 
            self.position.y + 5, 20, 20
        )
        pygame.draw.rect(screen, self.colors['button'].to_tuple(), 
                        minimize_btn_rect, border_radius=3)
        minimize_text = self.fonts['small'].render("—", True, 
                                                 self.colors['text'].to_tuple())
        minimize_text_rect = minimize_text.get_rect(center=minimize_btn_rect.center)
        screen.blit(minimize_text, minimize_text_rect)
        
        # Control buttons
        for button in self.control_buttons:
            self._render_control_button(screen, button)
        
        # Performance metrics
        self._render_performance_metrics(screen)
        
        # Simulation info
        self._render_simulation_info(screen)
    
    def _render_control_button(self, screen: pygame.Surface, button: ControlButton) -> None:
        """Render a control button."""
        adjusted_rect = pygame.Rect(
            self.position.x + button.position.x,
            self.position.y + button.position.y,
            button.position.width,
            button.position.height
        )
        
        # Determine button color
        if not button.enabled:
            color = self.colors['button_disabled']
        elif button.hover:
            color = self.colors['button_hover']
        else:
            color = button.color
        
        # Draw button
        pygame.draw.rect(screen, color.to_tuple(), adjusted_rect, border_radius=3)
        pygame.draw.rect(screen, self.colors['text'].to_tuple(), 
                        adjusted_rect, 1, border_radius=3)
        
        # Button text
        text_color = self.colors['text'] if button.enabled else self.colors['text_secondary']
        button_text = self.fonts['body'].render(button.text, True, text_color.to_tuple())
        text_rect = button_text.get_rect(center=adjusted_rect.center)
        screen.blit(button_text, text_rect)
    
    def _render_performance_metrics(self, screen: pygame.Surface) -> None:
        """Render performance metrics section."""
        metrics_y = self.position.y + 100
        
        # Section title
        title_text = self.fonts['body'].render("성능 지표", True, 
                                             self.colors['text'].to_tuple())
        screen.blit(title_text, (self.position.x + 10, metrics_y))
        metrics_y += 25
        
        # FPS
        if self.fps_history:
            current_fps = self.fps_history[-1]
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            fps_color = (self.colors['success'] if current_fps >= 50 else 
                        self.colors['warning'] if current_fps >= 30 else 
                        self.colors['error'])
            
            fps_text = f"FPS: {current_fps:.1f} (avg: {avg_fps:.1f})"
            fps_surface = self.fonts['small'].render(fps_text, True, fps_color.to_tuple())
            screen.blit(fps_surface, (self.position.x + 10, metrics_y))
            metrics_y += 20
        
        # Memory
        if self.memory_history:
            current_memory = self.memory_history[-1]
            memory_text = f"메모리: {current_memory:.1f} MB"
            memory_surface = self.fonts['small'].render(memory_text, True, 
                                                      self.colors['text_secondary'].to_tuple())
            screen.blit(memory_surface, (self.position.x + 10, metrics_y))
            metrics_y += 20
        
        # CPU
        if self.cpu_history:
            current_cpu = self.cpu_history[-1]
            cpu_color = (self.colors['success'] if current_cpu < 50 else 
                        self.colors['warning'] if current_cpu < 80 else 
                        self.colors['error'])
            
            cpu_text = f"CPU: {current_cpu:.1f}%"
            cpu_surface = self.fonts['small'].render(cpu_text, True, cpu_color.to_tuple())
            screen.blit(cpu_surface, (self.position.x + 10, metrics_y))
            metrics_y += 20
        
        # Mini performance graphs
        if len(self.fps_history) > 1:
            self._render_mini_graph(screen, self.fps_history, 
                                  (self.position.x + 10, metrics_y), 
                                  (self.position.width - 20, 30),
                                  self.colors['success'], "FPS")
    
    def _render_mini_graph(self, screen: pygame.Surface, data: List[float], 
                          position: tuple, size: tuple, color: Color, label: str) -> None:
        """Render a mini performance graph."""
        if len(data) < 2:
            return
        
        x, y = position
        width, height = size
        
        # Graph background
        graph_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(screen, Color(30, 30, 40).to_tuple(), graph_rect)
        pygame.draw.rect(screen, color.to_tuple(), graph_rect, 1)
        
        # Normalize data
        min_val = min(data)
        max_val = max(data)
        if max_val == min_val:
            max_val = min_val + 1
        
        # Draw data points
        points = []
        for i, value in enumerate(data):
            normalized = (value - min_val) / (max_val - min_val)
            point_x = x + (i / (len(data) - 1)) * width
            point_y = y + height - (normalized * height)
            points.append((point_x, point_y))
        
        if len(points) > 1:
            pygame.draw.lines(screen, color.to_tuple(), False, points, 2)
        
        # Label
        label_text = self.fonts['small'].render(label, True, color.to_tuple())
        screen.blit(label_text, (x + 5, y + 5))
    
    def _render_simulation_info(self, screen: pygame.Surface) -> None:
        """Render simulation-specific information."""
        info_y = self.position.y + 250
        
        # Section title
        title_text = self.fonts['body'].render("시뮬레이션 정보", True, 
                                             self.colors['text'].to_tuple())
        screen.blit(title_text, (self.position.x + 10, info_y))
        info_y += 25
        
        if self.simulation:
            # Simulation name
            sim_name = getattr(self.simulation, 'name', '알 수 없음')
            name_text = f"이름: {sim_name}"
            name_surface = self.fonts['small'].render(name_text, True, 
                                                    self.colors['text_secondary'].to_tuple())
            screen.blit(name_surface, (self.position.x + 10, info_y))
            info_y += 20
            
            # Simulation state
            is_running = getattr(self.simulation, 'is_running', lambda: True)()
            is_paused = getattr(self.simulation, 'is_paused', False)
            
            if is_paused:
                state_text = "상태: 일시정지"
                state_color = self.colors['warning']
            elif is_running:
                state_text = "상태: 실행 중"
                state_color = self.colors['success']
            else:
                state_text = "상태: 정지됨"
                state_color = self.colors['error']
            
            state_surface = self.fonts['small'].render(state_text, True, state_color.to_tuple())
            screen.blit(state_surface, (self.position.x + 10, info_y))
            info_y += 20
            
            # Runtime
            start_time = getattr(self.simulation, 'start_time', time.time())
            runtime = time.time() - start_time
            runtime_text = f"실행 시간: {runtime:.1f}초"
            runtime_surface = self.fonts['small'].render(runtime_text, True, 
                                                       self.colors['text_secondary'].to_tuple())
            screen.blit(runtime_surface, (self.position.x + 10, info_y))
        else:
            no_sim_text = "시뮬레이션이 연결되지 않음"
            no_sim_surface = self.fonts['small'].render(no_sim_text, True, 
                                                      self.colors['text_secondary'].to_tuple())
            screen.blit(no_sim_surface, (self.position.x + 10, info_y))


def create_control_panel(simulation: Optional[BaseSimulation] = None) -> ControlPanel:
    """Create and configure a control panel."""
    panel = ControlPanel()
    if simulation:
        panel.set_simulation(simulation)
    return panel