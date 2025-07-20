"""
Main Window Implementation

This module provides the main application window with:
- Menu system for navigation
- Simulation selection interface  
- Settings management
- Help and documentation system
"""

import pygame
import sys
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

from pyjoysim.simulation import (
    SimulationManager, get_simulation_manager, SimulationMetadata
)
from pyjoysim.input import InputManager, InputEvent, InputEventType
from pyjoysim.rendering import Color


class WindowState(Enum):
    """Main window states."""
    MENU = "menu"
    SIMULATION_SELECT = "simulation_select"
    SIMULATION_RUNNING = "simulation_running"
    SETTINGS = "settings"
    HELP = "help"


@dataclass
class MenuButton:
    """Menu button configuration."""
    text: str
    action: str
    position: pygame.Rect
    enabled: bool = True
    hover: bool = False


@dataclass
class UITheme:
    """UI theme configuration."""
    # Colors
    background: Color = Color(30, 30, 40)
    surface: Color = Color(45, 45, 55)
    primary: Color = Color(64, 128, 255)
    primary_hover: Color = Color(80, 144, 255)
    secondary: Color = Color(128, 128, 128)
    text: Color = Color(255, 255, 255)
    text_secondary: Color = Color(180, 180, 180)
    accent: Color = Color(255, 193, 7)
    success: Color = Color(76, 175, 80)
    warning: Color = Color(255, 152, 0)
    error: Color = Color(244, 67, 54)
    
    # Fonts
    title_size: int = 36
    subtitle_size: int = 24
    body_size: int = 18
    small_size: int = 14
    
    # Layout
    padding: int = 20
    margin: int = 10
    button_height: int = 50
    border_radius: int = 8


class MainWindow:
    """
    Main application window for PyJoySim.
    
    Provides the primary interface for simulation selection,
    settings management, and application navigation.
    """
    
    def __init__(self, width: int = 1024, height: int = 768):
        self.width = width
        self.height = height
        self.running = True
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("PyJoySim - 조이스틱 시뮬레이션 프로그램")
        self.clock = pygame.time.Clock()
        
        # UI theme
        self.theme = UITheme()
        
        # Window state
        self.state = WindowState.MENU
        self.previous_state = None
        
        # Input management
        self.input_manager = InputManager()
        
        # Simulation management
        self.simulation_manager = get_simulation_manager()
        self.current_simulation = None
        
        # UI components
        self.fonts = self._initialize_fonts()
        self.menu_buttons = self._create_menu_buttons()
        self.simulation_buttons = []
        
        # Mouse state
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        
        # Animation
        self.fade_alpha = 0
        self.transition_timer = 0
        
    def _initialize_fonts(self) -> Dict[str, pygame.font.Font]:
        """Initialize font objects for different text sizes."""
        return {
            'title': pygame.font.Font(None, self.theme.title_size),
            'subtitle': pygame.font.Font(None, self.theme.subtitle_size),
            'body': pygame.font.Font(None, self.theme.body_size),
            'small': pygame.font.Font(None, self.theme.small_size)
        }
    
    def _create_menu_buttons(self) -> List[MenuButton]:
        """Create main menu buttons."""
        button_width = 300
        button_height = self.theme.button_height
        start_y = self.height // 2 - 100
        center_x = self.width // 2 - button_width // 2
        
        buttons = [
            MenuButton(
                text="시뮬레이션 시작",
                action="start_simulation",
                position=pygame.Rect(center_x, start_y, button_width, button_height)
            ),
            MenuButton(
                text="설정",
                action="settings",
                position=pygame.Rect(center_x, start_y + 70, button_width, button_height)
            ),
            MenuButton(
                text="도움말",
                action="help",
                position=pygame.Rect(center_x, start_y + 140, button_width, button_height)
            ),
            MenuButton(
                text="종료",
                action="quit",
                position=pygame.Rect(center_x, start_y + 210, button_width, button_height)
            )
        ]
        
        return buttons
    
    def _create_simulation_buttons(self) -> List[MenuButton]:
        """Create simulation selection buttons."""
        buttons = []
        simulations = self.simulation_manager.registry.get_all_simulations()
        
        button_width = 250
        button_height = 80
        columns = 3
        rows = (len(simulations) + columns - 1) // columns
        
        start_x = (self.width - (columns * (button_width + 20))) // 2
        start_y = 150
        
        for i, (name, metadata) in enumerate(simulations.items()):
            row = i // columns
            col = i % columns
            
            x = start_x + col * (button_width + 20)
            y = start_y + row * (button_height + 20)
            
            button = MenuButton(
                text=metadata.get('display_name', name),
                action=f"run_simulation:{name}",
                position=pygame.Rect(x, y, button_width, button_height)
            )
            buttons.append(button)
        
        # Add back button
        buttons.append(MenuButton(
            text="← 뒤로",
            action="back",
            position=pygame.Rect(50, 50, 100, 40)
        ))
        
        return buttons
    
    def run(self) -> None:
        """Main application loop."""
        while self.running:
            dt = self.clock.tick(60) / 1000.0  # Delta time in seconds
            
            self._handle_events()
            self._update(dt)
            self._render()
            
        pygame.quit()
        sys.exit()
    
    def _handle_events(self) -> None:
        """Handle pygame events."""
        self.mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == WindowState.SIMULATION_RUNNING:
                        self._stop_current_simulation()
                    elif self.state != WindowState.MENU:
                        self._change_state(WindowState.MENU)
                    else:
                        self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.mouse_clicked = True
            
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = event.pos
            
            # Pass events to input manager for joystick handling
            self.input_manager.process_pygame_event(event)
        
        # Update button hover states
        self._update_button_hover_states()
    
    def _update_button_hover_states(self) -> None:
        """Update hover states for buttons based on mouse position."""
        if self.state == WindowState.MENU:
            buttons = self.menu_buttons
        elif self.state == WindowState.SIMULATION_SELECT:
            buttons = self.simulation_buttons
        else:
            buttons = []
        
        for button in buttons:
            button.hover = button.position.collidepoint(self.mouse_pos)
    
    def _update(self, dt: float) -> None:
        """Update main window state."""
        # Update input manager
        self.input_manager.update(dt)
        
        # Handle button clicks
        if self.mouse_clicked:
            self._handle_button_clicks()
        
        # Update transition animations
        if self.transition_timer > 0:
            self.transition_timer -= dt
            self.fade_alpha = max(0, min(255, 255 * (0.5 - self.transition_timer) / 0.5))
        
        # Update current simulation if running
        if self.state == WindowState.SIMULATION_RUNNING and self.current_simulation:
            try:
                if not self.current_simulation.is_running():
                    self._stop_current_simulation()
            except Exception as e:
                print(f"Simulation error: {e}")
                self._stop_current_simulation()
    
    def _handle_button_clicks(self) -> None:
        """Handle mouse clicks on buttons."""
        if self.state == WindowState.MENU:
            buttons = self.menu_buttons
        elif self.state == WindowState.SIMULATION_SELECT:
            buttons = self.simulation_buttons
        else:
            return
        
        for button in buttons:
            if button.hover and button.enabled:
                self._execute_button_action(button.action)
                break
    
    def _execute_button_action(self, action: str) -> None:
        """Execute button action."""
        if action == "start_simulation":
            self._change_state(WindowState.SIMULATION_SELECT)
            self.simulation_buttons = self._create_simulation_buttons()
            
        elif action == "settings":
            self._change_state(WindowState.SETTINGS)
            
        elif action == "help":
            self._change_state(WindowState.HELP)
            
        elif action == "quit":
            self.running = False
            
        elif action == "back":
            self._change_state(WindowState.MENU)
            
        elif action.startswith("run_simulation:"):
            simulation_name = action.split(":", 1)[1]
            self._start_simulation(simulation_name)
    
    def _change_state(self, new_state: WindowState) -> None:
        """Change window state with transition."""
        self.previous_state = self.state
        self.state = new_state
        self.transition_timer = 0.5  # Transition duration
        self.fade_alpha = 255
    
    def _start_simulation(self, simulation_name: str) -> None:
        """Start a simulation."""
        try:
            self.current_simulation = self.simulation_manager.create_simulation(
                simulation_name)
            
            if self.current_simulation:
                self._change_state(WindowState.SIMULATION_RUNNING)
                
                # Run simulation in a separate thread or process
                # For now, we'll run it in the main thread
                self.current_simulation.run()
            else:
                print(f"Failed to create simulation: {simulation_name}")
                
        except Exception as e:
            print(f"Error starting simulation {simulation_name}: {e}")
    
    def _stop_current_simulation(self) -> None:
        """Stop current simulation and return to menu."""
        if self.current_simulation:
            try:
                self.current_simulation.stop()
            except:
                pass
            finally:
                self.current_simulation = None
        
        self._change_state(WindowState.SIMULATION_SELECT)
    
    def _render(self) -> None:
        """Render the main window."""
        # Clear screen
        self.screen.fill(self.theme.background.to_tuple())
        
        if self.state == WindowState.MENU:
            self._render_main_menu()
        elif self.state == WindowState.SIMULATION_SELECT:
            self._render_simulation_select()
        elif self.state == WindowState.SETTINGS:
            self._render_settings()
        elif self.state == WindowState.HELP:
            self._render_help()
        elif self.state == WindowState.SIMULATION_RUNNING:
            self._render_simulation_running()
        
        # Apply transition fade effect
        if self.fade_alpha > 0:
            fade_surface = pygame.Surface((self.width, self.height))
            fade_surface.set_alpha(self.fade_alpha)
            fade_surface.fill((0, 0, 0))
            self.screen.blit(fade_surface, (0, 0))
        
        pygame.display.flip()
    
    def _render_main_menu(self) -> None:
        """Render main menu screen."""
        # Title
        title_text = self.fonts['title'].render(
            "PyJoySim", True, self.theme.text.to_tuple())
        title_rect = title_text.get_rect(
            center=(self.width // 2, self.height // 4))
        self.screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = self.fonts['subtitle'].render(
            "조이스틱을 이용한 교육용 시뮬레이션 프로그램", True, 
            self.theme.text_secondary.to_tuple())
        subtitle_rect = subtitle_text.get_rect(
            center=(self.width // 2, self.height // 4 + 50))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # Menu buttons
        for button in self.menu_buttons:
            self._render_button(button)
        
        # Version info
        version_text = self.fonts['small'].render(
            "Version 0.2.0 - Phase 2", True, 
            self.theme.text_secondary.to_tuple())
        version_rect = version_text.get_rect(
            bottomright=(self.width - 20, self.height - 20))
        self.screen.blit(version_text, version_rect)
    
    def _render_simulation_select(self) -> None:
        """Render simulation selection screen."""
        # Title
        title_text = self.fonts['title'].render(
            "시뮬레이션 선택", True, self.theme.text.to_tuple())
        title_rect = title_text.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title_text, title_rect)
        
        # Simulation buttons
        for button in self.simulation_buttons:
            self._render_simulation_button(button)
        
        # Instructions
        instruction_text = self.fonts['body'].render(
            "원하는 시뮬레이션을 선택하세요. ESC 키로 뒤로 갈 수 있습니다.", 
            True, self.theme.text_secondary.to_tuple())
        instruction_rect = instruction_text.get_rect(
            center=(self.width // 2, self.height - 60))
        self.screen.blit(instruction_text, instruction_rect)
    
    def _render_settings(self) -> None:
        """Render settings screen."""
        # Title
        title_text = self.fonts['title'].render(
            "설정", True, self.theme.text.to_tuple())
        title_rect = title_text.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title_text, title_rect)
        
        # Settings content (placeholder)
        content_lines = [
            "• 조이스틱 설정",
            "• 그래픽 품질 설정", 
            "• 오디오 설정",
            "• 키보드 단축키 설정",
            "• 시뮬레이션 매개변수 설정"
        ]
        
        y_start = 200
        for i, line in enumerate(content_lines):
            text = self.fonts['body'].render(line, True, self.theme.text.to_tuple())
            self.screen.blit(text, (100, y_start + i * 40))
        
        # Back instruction
        back_text = self.fonts['body'].render(
            "ESC 키를 눌러 메인 메뉴로 돌아가세요.", 
            True, self.theme.text_secondary.to_tuple())
        back_rect = back_text.get_rect(center=(self.width // 2, self.height - 60))
        self.screen.blit(back_text, back_rect)
    
    def _render_help(self) -> None:
        """Render help screen."""
        # Title
        title_text = self.fonts['title'].render(
            "도움말", True, self.theme.text.to_tuple())
        title_rect = title_text.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title_text, title_rect)
        
        # Help content
        help_sections = [
            ("조이스틱 연결", [
                "• 조이스틱을 컴퓨터에 연결하세요",
                "• 대부분의 USB 게임패드를 지원합니다",
                "• Xbox, PlayStation 컨트롤러 권장"
            ]),
            ("기본 조작법", [
                "• 좌 스틱: 주 제어 (조향, 이동)",
                "• 우 스틱: 보조 제어 (카메라, 미세 조정)",
                "• 트리거: 가속/제동",
                "• 버튼: 기능별 할당"
            ]),
            ("키보드 대체", [
                "• 화살표 키: 방향 제어",
                "• WASD: 카메라 이동",
                "• 스페이스바: 핸드브레이크/액션",
                "• ESC: 메뉴/종료"
            ])
        ]
        
        y_pos = 150
        for section_title, items in help_sections:
            # Section title
            section_text = self.fonts['subtitle'].render(
                section_title, True, self.theme.primary.to_tuple())
            self.screen.blit(section_text, (100, y_pos))
            y_pos += 40
            
            # Section items
            for item in items:
                item_text = self.fonts['body'].render(
                    item, True, self.theme.text.to_tuple())
                self.screen.blit(item_text, (120, y_pos))
                y_pos += 30
            
            y_pos += 20
        
        # Back instruction
        back_text = self.fonts['body'].render(
            "ESC 키를 눌러 메인 메뉴로 돌아가세요.", 
            True, self.theme.text_secondary.to_tuple())
        back_rect = back_text.get_rect(center=(self.width // 2, self.height - 60))
        self.screen.blit(back_text, back_rect)
    
    def _render_simulation_running(self) -> None:
        """Render simulation running screen."""
        # Show simulation status
        status_text = self.fonts['title'].render(
            "시뮬레이션 실행 중...", True, self.theme.text.to_tuple())
        status_rect = status_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(status_text, status_rect)
        
        # Instructions
        instruction_text = self.fonts['body'].render(
            "ESC 키를 눌러 시뮬레이션을 종료하세요.", 
            True, self.theme.text_secondary.to_tuple())
        instruction_rect = instruction_text.get_rect(
            center=(self.width // 2, self.height // 2 + 60))
        self.screen.blit(instruction_text, instruction_rect)
    
    def _render_button(self, button: MenuButton) -> None:
        """Render a menu button."""
        # Determine button color
        if button.hover and button.enabled:
            color = self.theme.primary_hover
        elif button.enabled:
            color = self.theme.primary
        else:
            color = self.theme.secondary
        
        # Draw button background
        pygame.draw.rect(self.screen, color.to_tuple(), button.position)
        pygame.draw.rect(self.screen, self.theme.text.to_tuple(), 
                        button.position, 2)
        
        # Draw button text
        text_color = self.theme.text if button.enabled else self.theme.text_secondary
        text = self.fonts['body'].render(button.text, True, text_color.to_tuple())
        text_rect = text.get_rect(center=button.position.center)
        self.screen.blit(text, text_rect)
    
    def _render_simulation_button(self, button: MenuButton) -> None:
        """Render a simulation selection button with metadata."""
        # Check if this is a simulation button or back button
        if button.action.startswith("run_simulation:"):
            simulation_name = button.action.split(":", 1)[1]
            metadata = self.simulation_manager.registry.get_simulation_metadata(simulation_name)
            
            # Determine button color based on category
            category = metadata.get('category', 'unknown') if metadata else 'unknown'
            if category == 'vehicle':
                base_color = Color(100, 150, 255)  # Blue for vehicles
            elif category == 'robot':
                base_color = Color(255, 150, 100)  # Orange for robots
            else:
                base_color = self.theme.primary
            
            color = base_color if not button.hover else Color(
                min(255, base_color.r + 30),
                min(255, base_color.g + 30), 
                min(255, base_color.b + 30)
            )
        else:
            # Back button
            color = self.theme.secondary if not button.hover else Color(
                self.theme.secondary.r + 30,
                self.theme.secondary.g + 30,
                self.theme.secondary.b + 30
            )
        
        # Draw button background
        pygame.draw.rect(self.screen, color.to_tuple(), button.position)
        pygame.draw.rect(self.screen, self.theme.text.to_tuple(), 
                        button.position, 2)
        
        # Draw button text
        text = self.fonts['body'].render(button.text, True, 
                                       self.theme.text.to_tuple())
        text_rect = text.get_rect(center=(
            button.position.centerx, 
            button.position.centery - 10
        ))
        self.screen.blit(text, text_rect)
        
        # Draw additional info for simulation buttons
        if button.action.startswith("run_simulation:") and metadata:
            # Difficulty indicator
            difficulty = metadata.get('difficulty', 'unknown')
            difficulty_text = self.fonts['small'].render(
                f"난이도: {difficulty}", True, 
                self.theme.text_secondary.to_tuple())
            difficulty_rect = difficulty_text.get_rect(center=(
                button.position.centerx,
                button.position.centery + 15
            ))
            self.screen.blit(difficulty_text, difficulty_rect)


def main():
    """Main entry point for the PyJoySim main window."""
    try:
        window = MainWindow()
        window.run()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())