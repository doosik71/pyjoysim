"""
Simulation Selector Implementation

This module provides an advanced simulation selection interface with:
- Detailed simulation information and previews
- Category filtering and search
- Joystick status checking
- Educational content display
"""

import pygame
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from pyjoysim.simulation import (
    SimulationManager, get_simulation_manager, SimulationMetadata
)
from pyjoysim.input import InputManager
from pyjoysim.rendering import Color


class SelectorState(Enum):
    """Simulation selector states."""
    BROWSING = "browsing"
    DETAILS = "details"
    LAUNCHING = "launching"


@dataclass
class SimulationCard:
    """Simulation card display configuration."""
    name: str
    metadata: Dict[str, Any]
    position: pygame.Rect
    thumbnail: Optional[pygame.Surface] = None
    hovered: bool = False
    selected: bool = False


@dataclass
class CategoryFilter:
    """Category filter configuration."""
    name: str
    display_name: str
    color: Color
    enabled: bool = True


class SimulationSelector:
    """
    Advanced simulation selection interface.
    
    Features:
    - Grid-based simulation browsing
    - Category filtering
    - Detailed information panels
    - Joystick compatibility checking
    - Educational content display
    """
    
    def __init__(self, width: int = 1024, height: int = 768):
        self.width = width
        self.height = height
        
        # State management
        self.state = SelectorState.BROWSING
        self.selected_simulation = None
        self.hovered_simulation = None
        
        # UI configuration
        self.card_width = 200
        self.card_height = 150
        self.cards_per_row = 4
        self.card_margin = 20
        self.header_height = 100
        self.sidebar_width = 250
        
        # Colors
        self.colors = {
            'background': Color(25, 25, 35),
            'card_bg': Color(45, 45, 55),
            'card_hover': Color(55, 55, 65),
            'card_selected': Color(64, 128, 255),
            'text': Color(255, 255, 255),
            'text_secondary': Color(180, 180, 180),
            'accent': Color(255, 193, 7),
            'vehicle': Color(100, 150, 255),
            'robot': Color(255, 150, 100),
            'game': Color(150, 255, 100),
            'educational': Color(255, 100, 150)
        }
        
        # Components
        self.simulation_manager = get_simulation_manager()
        self.input_manager = InputManager()
        
        # Fonts
        self.fonts = {
            'title': pygame.font.Font(None, 32),
            'subtitle': pygame.font.Font(None, 24),
            'body': pygame.font.Font(None, 18),
            'small': pygame.font.Font(None, 14)
        }
        
        # Category filters
        self.category_filters = [
            CategoryFilter("all", "전체", self.colors['text']),
            CategoryFilter("vehicle", "차량", self.colors['vehicle']),
            CategoryFilter("robot", "로봇", self.colors['robot']),
            CategoryFilter("game", "게임", self.colors['game']),
            CategoryFilter("educational", "교육", self.colors['educational'])
        ]
        self.active_filter = "all"
        
        # Simulation cards
        self.simulation_cards = []
        self.scroll_offset = 0
        self.max_scroll = 0
        
        # Input state
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False
        self.keys_pressed = set()
        
        # Animation
        self.animation_timer = 0
        self.card_scale_factors = {}
        
        self._initialize_simulation_cards()
    
    def _initialize_simulation_cards(self) -> None:
        """Initialize simulation cards from available simulations."""
        simulations = self.simulation_manager.registry.get_all_simulations()
        self.simulation_cards = []
        
        # Calculate grid layout
        content_width = self.width - self.sidebar_width - 40
        self.cards_per_row = max(1, (content_width + self.card_margin) // 
                               (self.card_width + self.card_margin))
        
        filtered_sims = self._get_filtered_simulations(simulations)
        
        for i, (name, metadata) in enumerate(filtered_sims.items()):
            row = i // self.cards_per_row
            col = i % self.cards_per_row
            
            x = self.sidebar_width + 20 + col * (self.card_width + self.card_margin)
            y = self.header_height + 20 + row * (self.card_height + self.card_margin) - self.scroll_offset
            
            card = SimulationCard(
                name=name,
                metadata=metadata,
                position=pygame.Rect(x, y, self.card_width, self.card_height),
                thumbnail=self._generate_thumbnail(name, metadata)
            )
            
            self.simulation_cards.append(card)
            self.card_scale_factors[name] = 1.0
        
        # Calculate max scroll
        total_rows = (len(filtered_sims) + self.cards_per_row - 1) // self.cards_per_row
        total_height = total_rows * (self.card_height + self.card_margin)
        visible_height = self.height - self.header_height - 40
        self.max_scroll = max(0, total_height - visible_height)
    
    def _get_filtered_simulations(self, simulations: Dict[str, Dict]) -> Dict[str, Dict]:
        """Get simulations filtered by active category."""
        if self.active_filter == "all":
            return simulations
        
        filtered = {}
        for name, metadata in simulations.items():
            if metadata.get('category') == self.active_filter:
                filtered[name] = metadata
        
        return filtered
    
    def _generate_thumbnail(self, name: str, metadata: Dict[str, Any]) -> pygame.Surface:
        """Generate a thumbnail for a simulation."""
        thumbnail = pygame.Surface((self.card_width - 20, 80))
        
        # Get category color
        category = metadata.get('category', 'educational')
        bg_color = self.colors.get(category, self.colors['educational'])
        
        # Fill background with category color (dimmed)
        dim_color = Color(bg_color.r // 3, bg_color.g // 3, bg_color.b // 3)
        thumbnail.fill(dim_color.to_tuple())
        
        # Draw category icon/pattern
        if category == 'vehicle':
            self._draw_vehicle_icon(thumbnail, bg_color)
        elif category == 'robot':
            self._draw_robot_icon(thumbnail, bg_color)
        else:
            self._draw_generic_icon(thumbnail, bg_color)
        
        return thumbnail
    
    def _draw_vehicle_icon(self, surface: pygame.Surface, color: Color) -> None:
        """Draw vehicle icon on thumbnail."""
        width, height = surface.get_size()
        
        # Simple car shape
        car_rect = pygame.Rect(width//4, height//2, width//2, height//4)
        pygame.draw.rect(surface, color.to_tuple(), car_rect)
        
        # Wheels
        wheel_radius = 8
        pygame.draw.circle(surface, color.to_tuple(), 
                         (width//3, height//2 + height//4), wheel_radius)
        pygame.draw.circle(surface, color.to_tuple(), 
                         (2*width//3, height//2 + height//4), wheel_radius)
    
    def _draw_robot_icon(self, surface: pygame.Surface, color: Color) -> None:
        """Draw robot icon on thumbnail."""
        width, height = surface.get_size()
        
        # Robot arm segments
        segments = [
            ((width//2, height*3//4), (width//2, height//2)),
            ((width//2, height//2), (width*3//4, height//3)),
            ((width*3//4, height//3), (width*5//6, height//4))
        ]
        
        for start, end in segments:
            pygame.draw.line(surface, color.to_tuple(), start, end, 4)
            pygame.draw.circle(surface, color.to_tuple(), start, 6)
            pygame.draw.circle(surface, color.to_tuple(), end, 6)
    
    def _draw_generic_icon(self, surface: pygame.Surface, color: Color) -> None:
        """Draw generic icon on thumbnail."""
        width, height = surface.get_size()
        
        # Simple geometric pattern
        center = (width//2, height//2)
        radius = min(width, height) // 4
        
        pygame.draw.circle(surface, color.to_tuple(), center, radius, 3)
        pygame.draw.circle(surface, color.to_tuple(), center, radius//2)
    
    def update(self, dt: float) -> None:
        """Update simulation selector."""
        self.animation_timer += dt
        
        # Update input manager
        self.input_manager.update(dt)
        
        # Update card animations
        for card in self.simulation_cards:
            target_scale = 1.1 if card.hovered else 1.0
            current_scale = self.card_scale_factors.get(card.name, 1.0)
            
            # Smooth animation towards target scale
            scale_diff = target_scale - current_scale
            self.card_scale_factors[card.name] = current_scale + scale_diff * dt * 5
    
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
        
        elif event.type == pygame.MOUSEMOTION:
            self.mouse_pos = event.pos
            self._update_hover_states()
        
        elif event.type == pygame.MOUSEWHEEL:
            # Handle scrolling
            scroll_amount = event.y * 30
            self.scroll_offset = max(0, min(self.max_scroll, 
                                          self.scroll_offset - scroll_amount))
            self._update_card_positions()
        
        elif event.type == pygame.KEYDOWN:
            self.keys_pressed.add(event.key)
            
            if event.key == pygame.K_ESCAPE:
                return "back"
            elif event.key == pygame.K_RETURN:
                if self.selected_simulation:
                    return f"run_simulation:{self.selected_simulation}"
        
        elif event.type == pygame.KEYUP:
            self.keys_pressed.discard(event.key)
        
        # Pass to input manager
        self.input_manager.process_pygame_event(event)
        
        return None
    
    def _handle_mouse_click(self) -> Optional[str]:
        """Handle mouse click events."""
        # Check category filter clicks
        filter_action = self._check_filter_clicks()
        if filter_action:
            return filter_action
        
        # Check simulation card clicks
        for card in self.simulation_cards:
            if card.position.collidepoint(self.mouse_pos):
                if self.selected_simulation == card.name:
                    # Double click - launch simulation
                    return f"run_simulation:{card.name}"
                else:
                    # Single click - select simulation
                    self.selected_simulation = card.name
                    self._update_selection_states()
                    return None
        
        # Click outside - deselect
        self.selected_simulation = None
        self._update_selection_states()
        return None
    
    def _check_filter_clicks(self) -> Optional[str]:
        """Check if any category filter was clicked."""
        filter_y = 20
        filter_height = 30
        
        for i, filter_item in enumerate(self.category_filters):
            filter_rect = pygame.Rect(20, filter_y + i * 40, self.sidebar_width - 40, filter_height)
            
            if filter_rect.collidepoint(self.mouse_pos):
                self.active_filter = filter_item.name
                self._initialize_simulation_cards()
                return None
        
        return None
    
    def _update_hover_states(self) -> None:
        """Update hover states for cards based on mouse position."""
        self.hovered_simulation = None
        
        for card in self.simulation_cards:
            card.hovered = card.position.collidepoint(self.mouse_pos)
            if card.hovered:
                self.hovered_simulation = card.name
    
    def _update_selection_states(self) -> None:
        """Update selection states for all cards."""
        for card in self.simulation_cards:
            card.selected = (card.name == self.selected_simulation)
    
    def _update_card_positions(self) -> None:
        """Update card positions based on scroll offset."""
        for i, card in enumerate(self.simulation_cards):
            row = i // self.cards_per_row
            col = i % self.cards_per_row
            
            x = self.sidebar_width + 20 + col * (self.card_width + self.card_margin)
            y = self.header_height + 20 + row * (self.card_height + self.card_margin) - self.scroll_offset
            
            card.position.x = x
            card.position.y = y
    
    def render(self, screen: pygame.Surface) -> None:
        """Render simulation selector interface."""
        # Clear background
        screen.fill(self.colors['background'].to_tuple())
        
        # Render sidebar
        self._render_sidebar(screen)
        
        # Render header
        self._render_header(screen)
        
        # Render simulation cards
        self._render_simulation_cards(screen)
        
        # Render details panel if simulation selected
        if self.selected_simulation:
            self._render_details_panel(screen)
    
    def _render_sidebar(self, screen: pygame.Surface) -> None:
        """Render category filter sidebar."""
        # Sidebar background
        sidebar_rect = pygame.Rect(0, 0, self.sidebar_width, self.height)
        pygame.draw.rect(screen, Color(35, 35, 45).to_tuple(), sidebar_rect)
        pygame.draw.line(screen, Color(55, 55, 65).to_tuple(), 
                        (self.sidebar_width, 0), (self.sidebar_width, self.height), 2)
        
        # Title
        title_text = self.fonts['subtitle'].render("카테고리", True, 
                                                  self.colors['text'].to_tuple())
        screen.blit(title_text, (20, 20))
        
        # Category filters
        filter_y = 60
        for filter_item in self.category_filters:
            # Filter background
            filter_rect = pygame.Rect(20, filter_y, self.sidebar_width - 40, 30)
            
            if filter_item.name == self.active_filter:
                pygame.draw.rect(screen, filter_item.color.to_tuple(), filter_rect)
                text_color = Color(20, 20, 20)
            else:
                pygame.draw.rect(screen, Color(50, 50, 60).to_tuple(), filter_rect)
                text_color = self.colors['text']
            
            # Filter text
            filter_text = self.fonts['body'].render(filter_item.display_name, True, 
                                                   text_color.to_tuple())
            text_rect = filter_text.get_rect(center=filter_rect.center)
            screen.blit(filter_text, text_rect)
            
            filter_y += 40
        
        # Joystick status
        joystick_y = self.height - 120
        status_title = self.fonts['body'].render("조이스틱 상태", True, 
                                                self.colors['text'].to_tuple())
        screen.blit(status_title, (20, joystick_y))
        
        # Check joystick connection
        joystick_count = self.input_manager.get_joystick_count()
        if joystick_count > 0:
            status_color = Color(76, 175, 80)  # Green
            status_text = f"{joystick_count}개 연결됨"
        else:
            status_color = Color(244, 67, 54)  # Red
            status_text = "연결되지 않음"
        
        status_surface = self.fonts['small'].render(status_text, True, 
                                                   status_color.to_tuple())
        screen.blit(status_surface, (20, joystick_y + 25))
        
        # Controls help
        help_y = joystick_y + 60
        help_text = self.fonts['small'].render("ESC: 뒤로", True, 
                                              self.colors['text_secondary'].to_tuple())
        screen.blit(help_text, (20, help_y))
    
    def _render_header(self, screen: pygame.Surface) -> None:
        """Render header section."""
        header_rect = pygame.Rect(self.sidebar_width, 0, 
                                 self.width - self.sidebar_width, self.header_height)
        pygame.draw.rect(screen, Color(40, 40, 50).to_tuple(), header_rect)
        pygame.draw.line(screen, Color(55, 55, 65).to_tuple(), 
                        (self.sidebar_width, self.header_height), 
                        (self.width, self.header_height), 2)
        
        # Title
        title_text = self.fonts['title'].render("시뮬레이션 선택", True, 
                                               self.colors['text'].to_tuple())
        screen.blit(title_text, (self.sidebar_width + 20, 20))
        
        # Instruction
        filtered_count = len(self.simulation_cards)
        instruction = f"{filtered_count}개의 시뮬레이션이 있습니다. 클릭하여 선택하고 더블클릭하여 실행하세요."
        instruction_text = self.fonts['body'].render(instruction, True, 
                                                   self.colors['text_secondary'].to_tuple())
        screen.blit(instruction_text, (self.sidebar_width + 20, 60))
    
    def _render_simulation_cards(self, screen: pygame.Surface) -> None:
        """Render simulation cards."""
        # Create clipping rectangle for cards area
        cards_rect = pygame.Rect(self.sidebar_width, self.header_height, 
                               self.width - self.sidebar_width, 
                               self.height - self.header_height)
        
        # Set clipping
        screen.set_clip(cards_rect)
        
        for card in self.simulation_cards:
            # Skip cards that are not visible
            if (card.position.bottom < self.header_height or 
                card.position.top > self.height):
                continue
            
            self._render_simulation_card(screen, card)
        
        # Reset clipping
        screen.set_clip(None)
    
    def _render_simulation_card(self, screen: pygame.Surface, card: SimulationCard) -> None:
        """Render a single simulation card."""
        # Get scale factor for animation
        scale = self.card_scale_factors.get(card.name, 1.0)
        
        # Calculate scaled position and size
        if scale != 1.0:
            scaled_width = int(self.card_width * scale)
            scaled_height = int(self.card_height * scale)
            scaled_x = card.position.x - (scaled_width - self.card_width) // 2
            scaled_y = card.position.y - (scaled_height - self.card_height) // 2
            card_rect = pygame.Rect(scaled_x, scaled_y, scaled_width, scaled_height)
        else:
            card_rect = card.position
        
        # Determine card color
        if card.selected:
            bg_color = self.colors['card_selected']
        elif card.hovered:
            bg_color = self.colors['card_hover']
        else:
            bg_color = self.colors['card_bg']
        
        # Draw card background
        pygame.draw.rect(screen, bg_color.to_tuple(), card_rect, border_radius=8)
        pygame.draw.rect(screen, self.colors['text'].to_tuple(), card_rect, 2, border_radius=8)
        
        # Draw thumbnail
        if card.thumbnail:
            thumb_rect = pygame.Rect(card_rect.x + 10, card_rect.y + 10, 
                                   card_rect.width - 20, 80)
            if scale != 1.0:
                # Scale thumbnail
                scaled_thumb = pygame.transform.scale(
                    card.thumbnail, (thumb_rect.width, thumb_rect.height))
                screen.blit(scaled_thumb, thumb_rect)
            else:
                screen.blit(card.thumbnail, thumb_rect)
        
        # Draw title
        display_name = card.metadata.get('display_name', card.name)
        title_text = self.fonts['body'].render(display_name, True, 
                                             self.colors['text'].to_tuple())
        title_rect = title_text.get_rect(centerx=card_rect.centerx, 
                                       y=card_rect.y + 100)
        screen.blit(title_text, title_rect)
        
        # Draw difficulty
        difficulty = card.metadata.get('difficulty', 'unknown')
        difficulty_color = {
            'beginner': Color(76, 175, 80),
            'intermediate': Color(255, 193, 7),
            'advanced': Color(244, 67, 54)
        }.get(difficulty, self.colors['text_secondary'])
        
        difficulty_text = self.fonts['small'].render(f"난이도: {difficulty}", True, 
                                                   difficulty_color.to_tuple())
        difficulty_rect = difficulty_text.get_rect(centerx=card_rect.centerx, 
                                                 y=card_rect.y + 125)
        screen.blit(difficulty_text, difficulty_rect)
    
    def _render_details_panel(self, screen: pygame.Surface) -> None:
        """Render details panel for selected simulation."""
        # Find selected simulation metadata
        selected_metadata = None
        for card in self.simulation_cards:
            if card.name == self.selected_simulation:
                selected_metadata = card.metadata
                break
        
        if not selected_metadata:
            return
        
        # Panel background
        panel_width = 400
        panel_height = 300
        panel_x = self.width - panel_width - 20
        panel_y = self.height - panel_height - 20
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(screen, Color(40, 40, 50, 230).to_tuple(), panel_rect, border_radius=10)
        pygame.draw.rect(screen, self.colors['text'].to_tuple(), panel_rect, 2, border_radius=10)
        
        # Panel content
        content_x = panel_x + 20
        content_y = panel_y + 20
        
        # Title
        display_name = selected_metadata.get('display_name', self.selected_simulation)
        title_text = self.fonts['subtitle'].render(display_name, True, 
                                                  self.colors['text'].to_tuple())
        screen.blit(title_text, (content_x, content_y))
        content_y += 40
        
        # Description
        description = selected_metadata.get('description', '설명이 없습니다.')
        desc_lines = self._wrap_text(description, self.fonts['body'], panel_width - 40)
        for line in desc_lines[:3]:  # Limit to 3 lines
            desc_text = self.fonts['body'].render(line, True, 
                                                self.colors['text_secondary'].to_tuple())
            screen.blit(desc_text, (content_x, content_y))
            content_y += 25
        
        content_y += 10
        
        # Educational topics
        topics = selected_metadata.get('educational_topics', [])
        if topics:
            topics_title = self.fonts['body'].render("학습 주제:", True, 
                                                   self.colors['accent'].to_tuple())
            screen.blit(topics_title, (content_x, content_y))
            content_y += 25
            
            for topic in topics[:3]:  # Limit to 3 topics
                topic_text = self.fonts['small'].render(f"• {topic}", True, 
                                                       self.colors['text'].to_tuple())
                screen.blit(topic_text, (content_x + 10, content_y))
                content_y += 20
        
        # Launch instruction
        content_y = panel_y + panel_height - 40
        launch_text = self.fonts['small'].render("더블클릭하거나 Enter 키를 눌러 실행", True, 
                                                self.colors['text_secondary'].to_tuple())
        launch_rect = launch_text.get_rect(centerx=panel_rect.centerx, y=content_y)
        screen.blit(launch_text, launch_rect)
    
    def _wrap_text(self, text: str, font: pygame.font.Font, max_width: int) -> List[str]:
        """Wrap text to fit within specified width."""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            # Test if adding this word would exceed max width
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Word itself is too long, add it anyway
                    lines.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines