import pygame
import numpy as np
from typing import Optional, List

NORMAL = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Material Design colors
COLOR_BG = (245, 245, 245)
COLOR_CARD = (255, 255, 255)
COLOR_PRIMARY = (33, 150, 243)
COLOR_SUCCESS = (76, 175, 80)
COLOR_WARNING = (255, 152, 0)
COLOR_ERROR = (244, 67, 54)
COLOR_TEXT = (33, 33, 33)
COLOR_TEXT_SECONDARY = (117, 117, 117)
COLOR_BORDER = (224, 224, 224)

KEY_START = pygame.K_s
KEY_QUIT_RECORDING = pygame.K_q
KEY_SAVE = pygame.K_a
KEY_DISCARD = pygame.K_b


class KBReset:
    def __init__(self, width: int = 450, height: int = 400):
        pygame.init()
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("YAM Data Collection")
        self.width = width
        self.height = height
        
        # Fonts
        self.font_title = pygame.font.SysFont('Arial', 20, bold=True)
        self.font_header = pygame.font.SysFont('Arial', 16, bold=True)
        self.font_normal = pygame.font.SysFont('Arial', 13)
        self.font_small = pygame.font.SysFont('Arial', 11)
        
        # Status data
        self.status = {
            'state': 'Initializing',  # Initializing, Waiting, Collecting, Saving
            'episode': 0,
            'total_episodes': 5,
            'frames_collected': 0,
            'fps': 0.0,
            'message': 'Initializing... Please wait',
            'left_arm': [0.0] * 7,
            'right_arm': [0.0] * 7,
        }
        
        # Whether the system is ready to accept commands
        self._ready = False
        
        # Options dialog state
        self.show_options_dialog = False
        self.options_title = ""
        self.options_list = []
        self.selected_option = 0
        self.user_choice = None
        
        self._set_color(NORMAL)
        self._draw_status()

    def update_status(self, **kwargs):
        """Update status information without redrawing."""
        self.status.update(kwargs)
        # Don't draw here - let update() handle drawing to avoid blocking
    
    def update_status_and_draw(self, **kwargs):
        """Update status and force redraw (use sparingly)."""
        self.status.update(kwargs)
        self._draw_status()
    
    def show_options(self, title: str, options: List[str]) -> int:
        """
        Show options dialog and wait for user selection.
        Returns selected index (0-based).
        """
        self.show_options_dialog = True
        self.options_title = title
        self.options_list = options
        self.selected_option = 0
        self.user_choice = None
        
        # Event loop for selection
        while self.user_choice is None:
            self._draw_options_dialog()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.selected_option = max(0, self.selected_option - 1)
                    elif event.key == pygame.K_DOWN:
                        self.selected_option = min(len(options) - 1, self.selected_option + 1)
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.user_choice = self.selected_option
        
        self.show_options_dialog = False
        self._draw_status()
        return self.user_choice

    def set_ready(self):
        """Mark the system as ready. Enables key input and shows 'Press S to start'."""
        # Flush any buffered key events from initialization phase
        pygame.event.clear()
        self._ready = True
        self.status['state'] = 'Waiting'
        self.status['message'] = 'Press S to start'
        self._draw_status()

    def update(self) -> str:
        """Check for keyboard input and update display."""
        pressed_last = self._get_pressed()
        
        # Draw status to keep display responsive
        self._draw_status()
        
        # Ignore key input if not ready
        if not self._ready:
            return "normal"
        
        state = self.status['state']
        
        # Only allow S in Waiting state
        if state == 'Waiting' and KEY_START in pressed_last:
            self.status['state'] = 'Collecting'
            self._set_color(BLUE)
            self._draw_status()
            return "start"
        # Only allow A/B in Collecting state
        if state == 'Collecting' and KEY_SAVE in pressed_last:
            self.status['state'] = 'Saving'
            self._set_color(GREEN)
            self._draw_status()
            return "save"
        if state == 'Collecting' and KEY_DISCARD in pressed_last:
            self.status['state'] = 'Waiting'
            self._set_color(RED)
            self._draw_status()
            return "discard"
        
        return "normal"

    def _get_pressed(self):
        pressed = []
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed.append(event.key)
        return pressed

    def _set_color(self, color):
        """Set background color (for state indication)."""
        pass  # Now handled by _draw_status()
    
    def _draw_status(self):
        """Draw status display."""
        if self.show_options_dialog:
            return  # Don't draw status when showing options
        
        # Background
        self._screen.fill(COLOR_BG)
        
        # Title bar
        title_rect = pygame.Rect(0, 0, self.width, 50)
        state = self.status['state']
        if state == 'Initializing':
            bg_color = COLOR_WARNING
        elif state == 'Collecting':
            bg_color = COLOR_PRIMARY
        elif state == 'Saving':
            bg_color = COLOR_SUCCESS
        elif state == 'Waiting':
            bg_color = COLOR_TEXT_SECONDARY
        else:
            bg_color = COLOR_TEXT
        
        pygame.draw.rect(self._screen, bg_color, title_rect)
        title_text = self.font_title.render(f"YAM Data Collection - {state}", True, WHITE)
        title_rect_center = title_text.get_rect(center=(self.width // 2, 25))
        self._screen.blit(title_text, title_rect_center)
        
        # Progress card
        card_y = 60
        self._draw_card(20, card_y, self.width - 40, 80, "Progress")
        
        # Progress bar
        progress = self.status['episode'] / max(1, self.status['total_episodes'])
        bar_width = self.width - 80
        bar_x = 40
        bar_y = card_y + 35
        
        # Background bar
        pygame.draw.rect(self._screen, COLOR_BORDER, (bar_x, bar_y, bar_width, 20), border_radius=10)
        # Progress bar
        filled_width = int(bar_width * progress)
        if filled_width > 0:
            pygame.draw.rect(self._screen, COLOR_PRIMARY, (bar_x, bar_y, filled_width, 20), border_radius=10)
        
        # Progress text
        progress_text = f"Episode {self.status['episode']}/{self.status['total_episodes']}  |  Frames: {self.status['frames_collected']}  |  FPS: {self.status['fps']:.1f}"
        progress_surf = self.font_small.render(progress_text, True, COLOR_TEXT_SECONDARY)
        self._screen.blit(progress_surf, (bar_x, bar_y + 25))
        
        # Robot state card
        card_y = 160
        self._draw_card(20, card_y, self.width - 40, 120, "Robot State")
        
        # Left arm
        left_text = self.font_normal.render("Left Arm:", True, COLOR_TEXT)
        self._screen.blit(left_text, (40, card_y + 35))
        self._draw_joint_bars(140, card_y + 30, self.status['left_arm'][:6], COLOR_PRIMARY)
        gripper_text = f"G: {self.status['left_arm'][6]:.2f}"
        gripper_surf = self.font_small.render(gripper_text, True, COLOR_TEXT_SECONDARY)
        self._screen.blit(gripper_surf, (350, card_y + 38))
        
        # Right arm
        right_text = self.font_normal.render("Right Arm:", True, COLOR_TEXT)
        self._screen.blit(right_text, (40, card_y + 75))
        self._draw_joint_bars(140, card_y + 70, self.status['right_arm'][:6], COLOR_SUCCESS)
        gripper_text = f"G: {self.status['right_arm'][6]:.2f}"
        gripper_surf = self.font_small.render(gripper_text, True, COLOR_TEXT_SECONDARY)
        self._screen.blit(gripper_surf, (350, card_y + 78))
        
        # Message bar
        msg_y = self.height - 80
        self._draw_card(20, msg_y, self.width - 40, 60, "Status")
        message_surf = self.font_normal.render(self.status['message'], True, COLOR_TEXT)
        self._screen.blit(message_surf, (40, msg_y + 30))
        
        pygame.display.flip()
    
    def _draw_card(self, x, y, w, h, title: Optional[str] = None):
        """Draw a card with optional title."""
        card_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self._screen, COLOR_CARD, card_rect, border_radius=8)
        pygame.draw.rect(self._screen, COLOR_BORDER, card_rect, width=2, border_radius=8)
        
        if title:
            title_surf = self.font_header.render(title, True, COLOR_TEXT)
            self._screen.blit(title_surf, (x + 10, y + 8))
    
    def _draw_joint_bars(self, x, y, joints, color):
        """Draw mini bar graphs for joint positions."""
        bar_width = 6
        bar_spacing = 10
        max_bar_height = 25
        
        for i, joint in enumerate(joints):
            # Normalize joint value (-π to π) to bar height
            normalized = (joint + np.pi) / (2 * np.pi)  # 0 to 1
            normalized = max(0, min(1, normalized))
            
            bar_height = int(normalized * max_bar_height)
            bar_x = x + i * (bar_width + bar_spacing)
            bar_y = y + max_bar_height - bar_height
            
            # Background
            pygame.draw.rect(self._screen, COLOR_BORDER, (bar_x, y, bar_width, max_bar_height))
            # Fill
            pygame.draw.rect(self._screen, color, (bar_x, bar_y, bar_width, bar_height))
    
    def _draw_options_dialog(self):
        """Draw options selection dialog."""
        # Background
        self._screen.fill(COLOR_BG)
        
        # Title
        title_surf = self.font_header.render(self.options_title, True, COLOR_TEXT)
        title_rect = title_surf.get_rect(center=(self.width // 2, 40))
        self._screen.blit(title_surf, title_rect)
        
        # Instructions
        inst_surf = self.font_small.render("Use UP/DOWN arrows, press ENTER to select", True, COLOR_TEXT_SECONDARY)
        inst_rect = inst_surf.get_rect(center=(self.width // 2, 70))
        self._screen.blit(inst_surf, inst_rect)
        
        # Options
        start_y = 100
        option_height = 50
        
        for i, option in enumerate(self.options_list):
            y = start_y + i * option_height
            is_selected = (i == self.selected_option)
            
            # Option card
            card_rect = pygame.Rect(30, y, self.width - 60, option_height - 10)
            bg_color = COLOR_PRIMARY if is_selected else COLOR_CARD
            pygame.draw.rect(self._screen, bg_color, card_rect, border_radius=8)
            pygame.draw.rect(self._screen, COLOR_BORDER, card_rect, width=2, border_radius=8)
            
            # Option text
            text_color = WHITE if is_selected else COLOR_TEXT
            option_surf = self.font_normal.render(option, True, text_color)
            option_rect = option_surf.get_rect(center=card_rect.center)
            self._screen.blit(option_surf, option_rect)
        
        pygame.display.flip()


def main():
    kb = KBReset()
    while True:
        state = kb.update()
        if state == "start":
            print("start")


if __name__ == "__main__":
    main()
