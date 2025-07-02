import pygame
import numpy as np
import sys
import os
import time
import math
import random
from typing import Dict, List, Tuple, Optional, Any, Set
import torch
from agents.dqn import act as dqn_act
from pathlib import Path
from pygame.locals import *

class GridWorldVisualizer:
    """Enhanced PyGame-based visualizer for the GridWorld environment and agents.
    
    This visualizer provides high-quality rendering with textures, animations, 
    particle effects, and a modern UI. Features include:
    
    1. Texture-based terrain rendering
    2. Smooth agent animations with directional sprites
    3. Particle effects for wind and goals
    4. Gradient-based value function visualization
    5. Enhanced policy arrows with confidence indicators
    6. Modern UI panel with custom fonts
    7. Performance optimizations (dirty rectangles, frame rate control)
    
    Usage:
        ```python
        # Create visualizer with default window size
        vis = GridWorldVisualizer()
        
        # Run simulation
        vis.simulate(env, agent, num_episodes=5)
        
        # For single-frame rendering
        vis.render(env, agent, episode=1, step=10, reward=5.0)
        ```
    
    Performance Tips:
    - Textures are cached for better performance
    - Animation speed is frame-rate independent
    - Particle effects scale based on available performance
    - Use render() with dirty rectangles for most efficient updating
    """
    
    # Color constants (backup in case textures aren't available)
    COLORS = {
        'background': (30, 30, 40),     # Dark blue-gray
        'grid': (100, 100, 120),        # Medium gray
        'agent': (255, 165, 0),         # Orange
        'goal': (0, 255, 100),          # Green
        'text': (240, 240, 255),        # Almost white
        'text_dark': (20, 20, 30),      # Dark text for light backgrounds
        'ice': (179, 224, 255),         # Light blue
        'mud': (139, 69, 19),           # Brown
        'quicksand': (194, 178, 128),   # Tan
        'wind': (135, 206, 235),        # Sky blue
        'value_high': (255, 50, 50, 180),  # Semi-transparent red
        'value_low': (255, 255, 50, 180)   # Semi-transparent yellow
    }
    
    # Wind direction arrows
    WIND_ARROWS = {
        (0, 1): '→',   # Right
        (0, -1): '←',  # Left
        (-1, 0): '↑',  # Up
        (1, 0): '↓',   # Down
        (-1, 1): '↗',  # Up-Right
        (-1, -1): '↖', # Up-Left
        (1, 1): '↘',   # Down-Right
        (1, -1): '↙'   # Down-Left
    }
    
    # Wind particle directions (for animation)
    WIND_VECTORS = {
        (0, 1): (1, 0),     # Right
        (0, -1): (-1, 0),   # Left
        (-1, 0): (0, -1),   # Up
        (1, 0): (0, 1),     # Down
        (-1, 1): (1, -1),   # Up-Right
        (-1, -1): (-1, -1), # Up-Left
        (1, 1): (1, 1),     # Down-Right
        (1, -1): (-1, 1)    # Down-Left
    }

    def __init__(self, window_size: Tuple[int, int] = (1024, 768)):
        """Initialize the enhanced PyGame visualizer.
        
        Args:
            window_size: Tuple of (width, height) for the window
        """
        # Initialize PyGame and display
        pygame.init()
        pygame.font.init()
        
        # Set up display with flags for smoother rendering
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("GridWorld RL Environment")
        
        # Find asset directories
        self.project_root = self._find_project_root()
        self.assets_dir = os.path.join(self.project_root, "assets")
        self.textures_dir = os.path.join(self.assets_dir, "textures")
        self.fonts_dir = os.path.join(self.assets_dir, "fonts")
        self.sounds_dir = os.path.join(self.assets_dir, "sounds")
        self.sprites_dir = os.path.join(self.assets_dir, "sprites")
        
        # Initialize texture cache
        self.textures = {}
        self.sprites = {}
        self.fonts = {}
        self.sounds = {}
        
        # Load assets
        self._load_assets()
        
        # Set up fonts - try to use custom fonts, fall back to system fonts
        self.title_font = self.fonts.get('title', pygame.font.SysFont('Arial', 24))
        self.font = self.fonts.get('main', pygame.font.SysFont('Arial', 20))
        self.small_font = self.fonts.get('small', pygame.font.SysFont('Arial', 16))
        
        # Initialize grid and visual parameters
        self.cell_size = None
        self.grid_offset = None
        self.info_panel_rect = None
        
        # Animation and simulation control
        self.running = False
        self.paused = False
        self.show_values = True
        self.show_policy = True
        self.animation_speed = 0.5  # seconds between steps
        
        # For value function visualization
        self.min_value = float('inf')
        self.max_value = float('-inf')
        
        # Animation state
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 60
        self.dt = 1.0 / self.fps
        
        # Particle effects
        self.particles = []  # List of active particles
        self.dirty_rects = []  # For efficient rendering
        
        # Agent animation state
        self.agent_visual_pos = None  # Smoothly interpolated position
        self.agent_target_pos = None  # Target grid position
        self.agent_move_speed = 5.0   # Grid cells per second
        
        # Goal animation state
        self.goal_pulse = 0
        self.goal_pulse_speed = 2.0
        
        # UI state
        self.hover_buttons = set()  # Set of buttons being hovered
    
    def _find_project_root(self) -> str:
        """Find the project root directory by looking for common markers."""
        # Start with the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to find project root by looking for common markers
        # (going up at most 3 levels from the current file)
        for _ in range(3):
            # Check if this looks like the project root
            if (os.path.isdir(os.path.join(current_dir, 'src')) and 
                os.path.isdir(os.path.join(current_dir, 'assets'))):
                return current_dir
                
            # Go up one level
            parent_dir = os.path.dirname(current_dir)
            if parent_dir == current_dir:  # We've reached the filesystem root
                break
            current_dir = parent_dir
        
        # If we can't find the project root, try to infer from the file structure
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def _load_assets(self) -> None:
        """Load textures, fonts, sprites and sounds."""
        # Create asset directories if they don't exist
        os.makedirs(self.textures_dir, exist_ok=True)
        os.makedirs(self.fonts_dir, exist_ok=True)
        os.makedirs(self.sprites_dir, exist_ok=True)
        os.makedirs(self.sounds_dir, exist_ok=True)
        
        # Generate textures if they don't exist yet
        self._ensure_textures_exist()
        
        # Load terrain textures
        self._load_textures()
        
        # Load fonts
        self._load_fonts()
        
        # Load sprites (for agent and effects)
        self._load_sprites()
        
        # Load sounds
        self._load_sounds()
    
    def _ensure_textures_exist(self) -> None:
        """Make sure texture files exist, generate them if needed."""
        try:
            from utils.texture_generator import TextureGenerator
            
            # Check if basic textures exist
            if not os.path.exists(os.path.join(self.textures_dir, "normal.png")):
                print("Generating textures...")
                generator = TextureGenerator(self.textures_dir)
                generator.generate_all()
                print("Textures generated successfully")
                
        except ImportError:
            print("Warning: texture_generator module not found, using fallback colors")
        except Exception as e:
            print(f"Warning: Failed to generate textures: {e}")
    
    def _load_textures(self) -> None:
        """Load terrain and UI textures."""
        # Define textures to load
        texture_files = [
            "normal.png",
            "ice.png",
            "mud.png", 
            "quicksand.png",
            "goal.png",
            "ui_panel.png",
            "ui_button.png",
            "ui_button_hover.png"
        ]
        
        # Load each texture
        for texture_file in texture_files:
            try:
                path = os.path.join(self.textures_dir, texture_file)
                if os.path.exists(path):
                    texture_name = os.path.splitext(texture_file)[0]
                    self.textures[texture_name] = pygame.image.load(path).convert_alpha()
                    print(f"Loaded texture: {texture_name}")
            except Exception as e:
                print(f"Warning: Failed to load texture '{texture_file}': {e}")
    
    def _load_fonts(self) -> None:
        """Load custom fonts."""
        # Try to find font files
        font_extensions = ['.ttf', '.otf']
        
        if os.path.exists(self.fonts_dir):
            for file in os.listdir(self.fonts_dir):
                for ext in font_extensions:
                    if file.lower().endswith(ext):
                        try:
                            font_path = os.path.join(self.fonts_dir, file)
                            font_name = os.path.splitext(file)[0].lower()
                            
                            # Load each font in different sizes
                            if "robot" in font_name.lower():
                                self.fonts['main'] = pygame.font.Font(font_path, 20)
                                self.fonts['small'] = pygame.font.Font(font_path, 16)
                            elif "orbit" in font_name.lower():
                                self.fonts['title'] = pygame.font.Font(font_path, 24)
                            elif "press" in font_name.lower() or "start" in font_name.lower():
                                self.fonts['pixel'] = pygame.font.Font(font_path, 16)
                                
                            print(f"Loaded font: {font_name}")
                        except Exception as e:
                            print(f"Warning: Failed to load font '{file}': {e}")
    
    def _load_sprites(self) -> None:
        """Load agent and effect sprites."""
        # Define sprite files to load
        sprite_files = [
            "agent.png", 
            "agent_up.png", 
            "agent_down.png", 
            "agent_left.png", 
            "agent_right.png",
            "particle_wind.png", 
            "particle_sparkle.png"
        ]
        
        # Check sprite folder and load sprites
        for sprite_file in sprite_files:
            try:
                path = os.path.join(self.textures_dir, sprite_file)
                if os.path.exists(path):
                    sprite_name = os.path.splitext(sprite_file)[0]
                    self.sprites[sprite_name] = pygame.image.load(path).convert_alpha()
                    print(f"Loaded sprite: {sprite_name}")
            except Exception as e:
                print(f"Warning: Failed to load sprite '{sprite_file}': {e}")
    
    def _load_sounds(self) -> None:
        """Load sound effects."""
        # Try to initialize mixer if it's not already
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except:
                print("Warning: Pygame mixer could not be initialized")
                return
        
        # Define sound files to load
        sound_files = {
            "move": "move.wav",
            "win": "win.wav",
            "collision": "collision.wav",
            "wind": "wind.wav"
        }
        
        # Check sound folder and load sounds
        for sound_name, sound_file in sound_files.items():
            try:
                path = os.path.join(self.sounds_dir, sound_file)
                if os.path.exists(path):
                    self.sounds[sound_name] = pygame.mixer.Sound(path)
            except Exception as e:
                print(f"Warning: Failed to load sound '{sound_file}': {e}")
    
    def _calculate_grid_dimensions(self, env) -> None:
        """Calculate grid cell size and offset based on environment size."""
        grid_height, grid_width = env.config["grid_size"]
        
        # Reserve 250px for info panel on the right (increased for nicer UI)
        info_panel_width = 250
        available_width = self.window_size[0] - info_panel_width
        available_height = self.window_size[1]
        
        # Calculate cell size to fit grid while maintaining square cells
        # Add some padding (0.9) to leave space around the grid
        cell_width = available_width / grid_width * 0.9
        cell_height = available_height / grid_height * 0.9
        self.cell_size = min(cell_width, cell_height)
        
        # Center the grid in available space
        total_grid_width = self.cell_size * grid_width
        total_grid_height = self.cell_size * grid_height
        
        self.grid_offset = (
            (available_width - total_grid_width) / 2,
            (available_height - total_grid_height) / 2
        )
        
        # Define info panel area
        self.info_panel_rect = pygame.Rect(
            available_width, 0, info_panel_width, self.window_size[1]
        )
        
        # Initialize agent visual position if it's not set yet
        if env.agent_pos is not None and self.agent_visual_pos is None:
            grid_pos = env.agent_pos
            self.agent_visual_pos = np.array([
                self.grid_offset[0] + (grid_pos[1] + 0.5) * self.cell_size,
                self.grid_offset[1] + (grid_pos[0] + 0.5) * self.cell_size
            ])
            self.agent_target_pos = self.agent_visual_pos.copy()

    def _draw_background(self) -> None:
        """Draw a gradient background."""
        # Create a vertical gradient from dark to slightly lighter
        top_color = (20, 20, 30)  # Dark blue-black
        bottom_color = (40, 40, 60)  # Slightly lighter blue-black
        
        # Get window dimensions
        width, height = self.window_size
        
        # Always fill with base color first to ensure no artifacts remain
        self.screen.fill(top_color)
        
        # Create a surface with the gradient that can be reused
        if not hasattr(self, 'gradient_surface') or self.frame_count <= 1:
            # Recreate the gradient surface on first frame or if not yet created
            self.gradient_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            for y in range(height):  # Draw every line for the stored gradient
                t = y / height
                r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
                g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
                b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
                pygame.draw.line(self.gradient_surface, (r, g, b), (0, y), (width, y))
        
        # Blit the stored gradient
        self.screen.blit(self.gradient_surface, (0, 0))
        
        # Mark the entire screen as needing update
        self.dirty_rects.append(pygame.Rect(0, 0, width, height))

    def _draw_grid(self, env) -> None:
        """Draw the grid structure with normal terrain cells."""
        grid_height, grid_width = env.config["grid_size"]
        
        # Draw normal terrain for all cells first
        for i in range(grid_height):
            for j in range(grid_width):
                cell_rect = self._get_cell_rect(i, j)
                
                # Use texture if available, otherwise use color
                if "normal" in self.textures:
                    # Scale the texture to fit the cell
                    scaled_texture = pygame.transform.scale(
                        self.textures["normal"], 
                        (int(self.cell_size), int(self.cell_size))
                    )
                    self.screen.blit(scaled_texture, cell_rect)
                else:
                    # Fallback to simple rectangle
                    pygame.draw.rect(
                        self.screen,
                        (50, 70, 50),  # Dark green
                        cell_rect
                    )
        
        # Draw grid lines
        for i in range(grid_height + 1):
            start_pos = (
                self.grid_offset[0],
                self.grid_offset[1] + i * self.cell_size
            )
            end_pos = (
                self.grid_offset[0] + grid_width * self.cell_size,
                self.grid_offset[1] + i * self.cell_size
            )
            # Draw semi-transparent grid lines
            pygame.draw.line(self.screen, (*self.COLORS['grid'], 100), start_pos, end_pos, 1)
            
        for j in range(grid_width + 1):
            start_pos = (
                self.grid_offset[0] + j * self.cell_size,
                self.grid_offset[1]
            )
            end_pos = (
                self.grid_offset[0] + j * self.cell_size,
                self.grid_offset[1] + grid_height * self.cell_size
            )
            pygame.draw.line(self.screen, (*self.COLORS['grid'], 100), start_pos, end_pos, 1)

    def _draw_terrain(self, env) -> None:
        """Draw different terrain types using textures."""
        for terrain_type, data in env.config.get("terrain", {}).items():
            # Choose the appropriate texture
            texture_key = terrain_type
            if texture_key not in self.textures:
                # Fall back to color if texture not available
                color = self.COLORS.get(terrain_type, self.COLORS['grid'])
                for pos in data["positions"]:
                    cell_rect = self._get_cell_rect(pos[0], pos[1])
                    s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    pygame.draw.rect(s, (*color[:3], 180), s.get_rect())
                    self.screen.blit(s, cell_rect)
                    
                    # Add terrain label
                    text = self.small_font.render(terrain_type[0].upper(), True, self.COLORS['text'])
                    text_rect = text.get_rect(center=cell_rect.center)
                    self.screen.blit(text, text_rect)
            else:
                # Use the texture
                texture = self.textures[texture_key]
                for pos in data["positions"]:
                    cell_rect = self._get_cell_rect(pos[0], pos[1])
                    # Scale texture to fit cell
                    scaled_texture = pygame.transform.scale(
                        texture, 
                        (int(self.cell_size), int(self.cell_size))
                    )
                    self.screen.blit(scaled_texture, cell_rect)
                    
                    # Add subtle label
                    text = self.small_font.render(terrain_type[0].upper(), True, self.COLORS['text'])
                    text_rect = text.get_rect(center=(cell_rect.centerx, cell_rect.centery + self.cell_size * 0.3))
                    self.screen.blit(text, text_rect)

    def _draw_wind_zones(self, env) -> None:
        """Draw wind zones with direction indicators and animated particles."""
        for wind_zone in env.config.get("wind_zones", []):
            direction = wind_zone["direction"]
            strength = wind_zone["strength"]
            arrow = self.WIND_ARROWS.get(direction, '•')
            
            for pos in wind_zone["area"]:
                cell_rect = self._get_cell_rect(pos[0], pos[1])
                
                # Draw wind zone background with subtle effect
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                
                # Use a slightly darker tint for wind areas
                wind_color = (100, 150, 255, 40)  # Light blue with low alpha
                pygame.draw.rect(s, wind_color, s.get_rect())
                self.screen.blit(s, cell_rect)
                
                # Add animated streaks in the wind direction
                vec = self.WIND_VECTORS.get(direction, (0, 0))
                
                # Create wind particle effects occasionally
                if random.random() < strength * 0.05:
                    self._create_wind_particle(cell_rect.center, direction, strength)
                
                # Draw wind indicator arrow
                arrow_color = (150, 200, 255)  # Light blue
                text = self.font.render(arrow, True, arrow_color)
                text_rect = text.get_rect(center=cell_rect.center)
                self.screen.blit(text, text_rect)
                
                # Draw wind strength indicator
                strength_text = self.small_font.render(f"{strength:.1f}", True, arrow_color)
                strength_rect = strength_text.get_rect(
                    center=(cell_rect.centerx, cell_rect.centery + 15)
                )
                self.screen.blit(strength_text, strength_rect)
                
    def _create_wind_particle(self, position, direction, strength):
        """Create a wind particle effect at the given position."""
        # Only create particles if we have the sprite
        if "particle_wind" not in self.sprites:
            return
            
        # Convert direction tuple to vector
        vec = self.WIND_VECTORS.get(direction, (0, 0))
        
        # Create particle with random lifetime and speed
        particle = {
            'pos': list(position),
            'vel': [vec[0] * (strength * 30 + random.uniform(-5, 5)), 
                   vec[1] * (strength * 30 + random.uniform(-5, 5))],
            'lifetime': random.uniform(0.5, 1.5),
            'age': 0,
            'sprite': self.sprites["particle_wind"],
            'size': random.uniform(0.5, 1.0)
        }
        
        self.particles.append(particle)
        
    def _update_particles(self, dt):
        """Update all particle effects."""
        # Update existing particles
        new_particles = []
        for p in self.particles:
            # Update position
            p['pos'][0] += p['vel'][0] * dt
            p['pos'][1] += p['vel'][1] * dt
            
            # Update age
            p['age'] += dt
            
            # Keep if still alive
            if p['age'] < p['lifetime']:
                new_particles.append(p)
            
        # Replace the particle list with updated particles
        self.particles = new_particles
        
    def _draw_particles(self):
        """Draw all active particle effects."""
        for p in self.particles:
            # Calculate alpha based on remaining lifetime
            life_ratio = 1.0 - (p['age'] / p['lifetime'])
            alpha = int(255 * life_ratio)
            
            # Scale sprite
            size = int(p['sprite'].get_width() * p['size'])
            scaled_sprite = pygame.transform.scale(p['sprite'], (size, size))
            
            # Apply alpha
            scaled_sprite.set_alpha(alpha)
            
            # Draw at position (centered)
            pos = (int(p['pos'][0] - size/2), int(p['pos'][1] - size/2))
            self.screen.blit(scaled_sprite, pos)
            
            # Add to dirty rects for efficient rendering
            self.dirty_rects.append(pygame.Rect(pos[0], pos[1], size, size))

    def _draw_value_heatmap(self, env, agent) -> None:
        """Draw value function or Q-value heatmap with visual enhancements."""
        if not self.show_values:
            return
            
        values = None
        if hasattr(agent, 'Q'):  # Q-Learning or SARSA
            values = np.max(agent.Q, axis=2)
        elif hasattr(agent, 'V'):  # Value/Policy Iteration
            values = agent.V
        elif hasattr(agent, 'online_net') or isinstance(agent, torch.nn.Module):  # DQN
            values = np.zeros(env.config["grid_size"])
            device = getattr(agent, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model = agent.online_net if hasattr(agent, 'online_net') else agent
            
            try:
                for i in range(env.grid_height):
                    for j in range(env.grid_width):
                        state = np.array([i, j])
                        state_tensor = torch.FloatTensor(state).to(device)
                        
                        # Try multiple input formats with error handling
                        try:
                            with torch.no_grad():
                                # Format 1: Direct input
                                q_values = model(state_tensor)
                                if q_values.dim() > 1:
                                    values[i, j] = q_values.max(dim=1).values.item()
                                else:
                                    values[i, j] = q_values.max().item()
                        except Exception:
                            try:
                                # Format 2: Unsqueezed input (batch dimension)
                                with torch.no_grad():
                                    q_values = model(state_tensor.unsqueeze(0))
                                    values[i, j] = q_values.max(dim=1).values.item()
                            except Exception as inner_e:
                                # If both attempts fail, print error and use a default value
                                print(f"Warning: Could not get Q-value at {i},{j}: {inner_e}")
                                values[i, j] = 0
            except Exception as e:
                # If visualization fails, print error but don't crash
                print(f"Warning: Error visualizing value function: {e}")
                return
        
        if values is not None:
            self.min_value = min(self.min_value, values.min())
            self.max_value = max(self.max_value, values.max())
            
            value_range = self.max_value - self.min_value
            if value_range > 0:
                for i in range(env.grid_height):
                    for j in range(env.grid_width):
                        normalized_value = (values[i, j] - self.min_value) / value_range
                        cell_rect = self._get_cell_rect(i, j)
                        
                        # Draw value heatmap with gradient
                        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        
                        # Use a fancy gradient color scheme
                        if normalized_value < 0.25:
                            # Blue to cyan gradient for low values
                            ratio = normalized_value / 0.25
                            r = int(0 + ratio * 0)
                            g = int(0 + ratio * 255)
                            b = int(255)
                            alpha = 150
                        elif normalized_value < 0.5:
                            # Cyan to green gradient for medium-low values
                            ratio = (normalized_value - 0.25) / 0.25
                            r = int(0)
                            g = int(255)
                            b = int(255 - ratio * 255)
                            alpha = 160
                        elif normalized_value < 0.75:
                            # Green to yellow gradient for medium-high values
                            ratio = (normalized_value - 0.5) / 0.25
                            r = int(ratio * 255)
                            g = int(255)
                            b = int(0)
                            alpha = 170
                        else:
                            # Yellow to red gradient for high values
                            ratio = (normalized_value - 0.75) / 0.25
                            r = int(255)
                            g = int(255 - ratio * 255)
                            b = int(0)
                            alpha = 180
                        
                        # Draw value cell with semi-transparency
                        color = (r, g, b, alpha)
                        pygame.draw.rect(s, color, s.get_rect())
                        
                        # Add inner glow for high-value cells
                        if normalized_value > 0.7:
                            glow_size = int(self.cell_size * 0.8)
                            glow_offset = (self.cell_size - glow_size) / 2
                            glow_rect = pygame.Rect(glow_offset, glow_offset, glow_size, glow_size)
                            glow_alpha = int(80 * normalized_value)
                            pygame.draw.rect(s, (r, g, b, glow_alpha), glow_rect)
                        
                        # Draw cell with value text
                        self.screen.blit(s, cell_rect)
                        
                        # Draw value number for important cells (either high or low values)
                        if normalized_value > 0.8 or normalized_value < 0.2:
                            value_text = f"{values[i, j]:.1f}"
                            text_color = (255, 255, 255) if normalized_value > 0.5 else (0, 0, 0)
                            value_label = self.small_font.render(value_text, True, text_color)
                            text_rect = value_label.get_rect(center=cell_rect.center)
                            self.screen.blit(value_label, text_rect)

    def _draw_policy_arrows(self, env, agent) -> None:
        """Draw policy arrows showing the best action in each state with visual enhancements."""
        if not self.show_policy:
            return
            
        action_arrows = ['↑', '→', '↓', '←']
        # Direction vectors for drawing arrow lines (dy, dx)
        action_vectors = [
            (0, -0.3),  # Up
            (0.3, 0),   # Right
            (0, 0.3),   # Down
            (-0.3, 0)   # Left
        ]
        
        for i in range(env.grid_height):
            for j in range(env.grid_width):
                if (i, j) == tuple(env.config["goal_pos"]):
                    continue
                    
                action = None
                q_values = None
                
                try:
                    if hasattr(agent, 'Q'):  # Q-Learning or SARSA
                        q_values = agent.Q[i, j]
                        action = np.argmax(q_values)
                    elif hasattr(agent, 'policy'):  # Value/Policy Iteration
                        action = agent.policy[i, j]
                    elif hasattr(agent, 'online_net') or isinstance(agent, torch.nn.Module):  # DQN
                        device = getattr(agent, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                        model = agent.online_net if hasattr(agent, 'online_net') else agent
                        # Create a properly formatted state tensor for DQN
                        state = torch.FloatTensor([i, j]).to(device)
                        
                        try:
                            # Try different input formats to handle various model architectures
                            with torch.no_grad():
                                # Format 1: Direct input
                                q_values_tensor = model(state)
                                if q_values_tensor.dim() > 1:
                                    action = q_values_tensor.argmax(dim=1).item()
                                    q_values = q_values_tensor[0].cpu().numpy()
                                else:
                                    action = q_values_tensor.argmax().item()
                                    q_values = q_values_tensor.cpu().numpy()
                        except Exception:
                            try:
                                # Format 2: Unsqueezed input (batch dimension)
                                with torch.no_grad():
                                    q_values_tensor = model(state.unsqueeze(0))
                                    action = q_values_tensor.argmax(dim=1).item()
                                    q_values = q_values_tensor[0].cpu().numpy()
                            except Exception:
                                # If all attempts fail, default to no action (will be skipped)
                                action = None
                except Exception as e:
                    # If visualization fails, print error but continue
                    print(f"Warning: Error drawing policy arrow: {e}")
                    continue
                
                if action is not None:
                    cell_rect = self._get_cell_rect(i, j)
                    cell_center = cell_rect.center
                    
                    # Draw fancy arrow based on action
                    # Change arrow color based on animation time for pulsing effect
                    pulse = (math.sin(self.frame_count * 0.05) + 1) * 0.5
                    base_color = (150, 200, 255)  # Light blue
                    arrow_color = (
                        int(base_color[0] + pulse * 50),
                        int(base_color[1] + pulse * 50),
                        int(base_color[2])
                    )
                    
                    # Draw arrow with larger size and glow effect
                    arrow_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                    
                    # Draw arrow glow (larger, semi-transparent)
                    glow_size = int(self.cell_size * 0.35)
                    pygame.draw.circle(
                        arrow_surface,
                        (*arrow_color, 60),  # Semi-transparent
                        (self.cell_size // 2, self.cell_size // 2),
                        glow_size
                    )
                    
                    # Draw arrow text
                    arrow_font = self.fonts.get('title', self.font)
                    text = arrow_font.render(action_arrows[action], True, arrow_color)
                    text_rect = text.get_rect(center=(self.cell_size // 2, self.cell_size // 2))
                    arrow_surface.blit(text, text_rect)
                    
                    # Draw a line in the direction of the action
                    dx, dy = action_vectors[action]
                    start_pos = (self.cell_size // 2, self.cell_size // 2)
                    end_pos = (int(self.cell_size * (0.5 + dx)), int(self.cell_size * (0.5 + dy)))
                    pygame.draw.line(arrow_surface, arrow_color, start_pos, end_pos, 3)
                    
                    # Draw a small circle at the end of the line
                    pygame.draw.circle(arrow_surface, arrow_color, end_pos, 5)
                    
                    # Apply to screen
                    self.screen.blit(arrow_surface, cell_rect)
                    
                    # If we have Q-values, visualize the action confidence
                    if q_values is not None and len(q_values) >= 4:
                        # Calculate action confidence (how much better is the best action)
                        # compared to the average of all other actions
                        best_q = q_values[action]
                        other_qs = [q_values[a] for a in range(len(q_values)) if a != action]
                        if other_qs:
                            avg_other_q = sum(other_qs) / len(other_qs)
                            confidence = max(0, min(1, (best_q - avg_other_q) / max(1, abs(best_q))))
                            
                            # Draw confidence indicator as a small bar below the arrow
                            bar_width = int(self.cell_size * 0.6 * confidence)
                            bar_height = 4
                            bar_rect = pygame.Rect(
                                cell_rect.centerx - bar_width // 2,
                                cell_rect.centery + self.cell_size // 4,
                                bar_width,
                                bar_height
                            )
                            pygame.draw.rect(self.screen, arrow_color, bar_rect)

    def _draw_goal(self, env) -> None:
        """Draw the goal with animated effects."""
        goal_pos = env.config["goal_pos"]
        goal_rect = self._get_cell_rect(*goal_pos)
        
        # Use texture if available
        if "goal" in self.textures:
            # Scale texture to fit cell
            goal_texture = pygame.transform.scale(
                self.textures["goal"], 
                (int(self.cell_size), int(self.cell_size))
            )
            
            # Add pulsing glow effect
            self.goal_pulse += self.dt * self.goal_pulse_speed
            pulse_scale = 1.0 + 0.1 * math.sin(self.goal_pulse)
            
            # Scale the texture slightly for pulsing effect
            pulse_size = int(self.cell_size * pulse_scale)
            pulse_offset = (pulse_size - self.cell_size) // 2
            
            # Create a glow surface
            if pulse_scale > 1.0:
                glow = pygame.Surface((pulse_size, pulse_size), pygame.SRCALPHA)
                glow_color = (100, 255, 100, 100)  # Semi-transparent green
                pygame.draw.circle(
                    glow, 
                    glow_color,
                    (pulse_size // 2, pulse_size // 2),
                    pulse_size // 2
                )
                
                # Blit glow under the goal
                self.screen.blit(
                    glow, 
                    (goal_rect.x - pulse_offset, goal_rect.y - pulse_offset)
                )
            
            # Blit the main goal texture
            self.screen.blit(goal_texture, goal_rect)
            
            # Add sparkle effects occasionally
            if "particle_sparkle" in self.sprites and random.random() < 0.05:
                self._create_sparkle_particle(goal_rect.center)
        else:
            # Fallback to simple rectangle if texture not available
            pygame.draw.rect(self.screen, self.COLORS['goal'], goal_rect)
            text = self.font.render("G", True, self.COLORS['text_dark'])
            text_rect = text.get_rect(center=goal_rect.center)
            self.screen.blit(text, text_rect)
    
    def _create_sparkle_particle(self, position):
        """Create a sparkle particle at the given position."""
        if "particle_sparkle" not in self.sprites:
            return
            
        # Create sparkle with random movement and lifetime
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(5, 20)
        particle = {
            'pos': list(position),
            'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
            'lifetime': random.uniform(0.3, 1.0),
            'age': 0,
            'sprite': self.sprites["particle_sparkle"],
            'size': random.uniform(0.4, 0.8),
            'rot_speed': random.uniform(-5, 5)  # Rotation speed in radians/sec
        }
        
        self.particles.append(particle)
    
    def _draw_agent(self, env) -> None:
        """Draw the agent with smooth animation."""
        # Get target position in grid coordinates
        grid_pos = env.agent_pos
        
        # Calculate target pixel position
        target_pos = np.array([
            self.grid_offset[0] + (grid_pos[1] + 0.5) * self.cell_size,
            self.grid_offset[1] + (grid_pos[0] + 0.5) * self.cell_size
        ])
        
        # Initialize agent position if needed
        if self.agent_visual_pos is None:
            self.agent_visual_pos = target_pos.copy()
            self.agent_target_pos = target_pos.copy()
            
        # Update target position
        self.agent_target_pos = target_pos.copy()
        
        # Smoothly animate the agent position
        if not np.array_equal(self.agent_visual_pos, self.agent_target_pos):
            # Calculate movement vector
            move_vec = self.agent_target_pos - self.agent_visual_pos
            distance = np.linalg.norm(move_vec)
            
            # Calculate maximum movement for this frame
            max_move = self.cell_size * self.agent_move_speed * self.dt
            
            if distance <= max_move:
                # We can reach the target this frame
                self.agent_visual_pos = self.agent_target_pos.copy()
            else:
                # Move towards target
                self.agent_visual_pos += move_vec * max_move / distance
        
        # Determine agent direction for appropriate sprite
        direction = "agent"  # Default
        if hasattr(self, 'previous_agent_pos') and self.previous_agent_pos is not None:
            # Calculate direction from previous to current position
            dy = grid_pos[0] - self.previous_agent_pos[0]
            dx = grid_pos[1] - self.previous_agent_pos[1]
            
            if abs(dy) > abs(dx):  # Vertical movement dominates
                direction = "agent_down" if dy > 0 else "agent_up"
            else:  # Horizontal movement dominates
                direction = "agent_right" if dx > 0 else "agent_left"
        
        # Update previous position
        self.previous_agent_pos = grid_pos.copy()
        
        # Get the appropriate sprite
        agent_sprite = self.sprites.get(direction, self.sprites.get("agent"))
        
        if agent_sprite is not None:
            # Scale sprite to appropriate size (80% of cell size)
            sprite_size = int(self.cell_size * 0.8)
            scaled_sprite = pygame.transform.scale(agent_sprite, (sprite_size, sprite_size))
            
            # Draw sprite centered on agent position
            sprite_pos = (
                int(self.agent_visual_pos[0] - sprite_size // 2),
                int(self.agent_visual_pos[1] - sprite_size // 2)
            )
            self.screen.blit(scaled_sprite, sprite_pos)
            
            # Add to dirty rects for efficient rendering
            self.dirty_rects.append(pygame.Rect(
                sprite_pos[0], sprite_pos[1], sprite_size, sprite_size
            ))
        else:
            # Fallback to simple circle if sprites aren't available
            pygame.draw.circle(
                self.screen, 
                self.COLORS['agent'],
                (int(self.agent_visual_pos[0]), int(self.agent_visual_pos[1])), 
                int(self.cell_size * 0.4)
            )
    
    def _draw_info_panel(self, env, episode: Optional[int] = None,
                         step: Optional[int] = None, reward: Optional[float] = None) -> None:
        """Draw a modern information panel on the right side."""
        # Draw panel background
        if "ui_panel" in self.textures:
            # Scale panel texture
            panel_texture = pygame.transform.scale(
                self.textures["ui_panel"],
                (self.info_panel_rect.width, self.info_panel_rect.height)
            )
            self.screen.blit(panel_texture, self.info_panel_rect)
        else:
            # Fallback to simple rectangle
            pygame.draw.rect(self.screen, (30, 30, 40), self.info_panel_rect)
            pygame.draw.rect(self.screen, (60, 60, 80), self.info_panel_rect, 2)  # Border
        
        # Initialize text position
        panel_x = self.info_panel_rect.x + 20
        y_offset = 30
        line_height = 30
        small_line_height = 25
        
        # Draw title
        title_font = self.fonts.get('title', self.font)
        title = title_font.render("GridWorld", True, (220, 220, 255))
        self.screen.blit(title, (panel_x, y_offset))
        y_offset += line_height + 10
        
        # Draw agent type
        agent_type = "Unknown Agent"
        if hasattr(env, 'agent_type'):
            agent_type = env.agent_type
        elif hasattr(env, '_agent_type'):
            agent_type = env._agent_type
        
        # Try to detect agent type from class name if not explicitly set
        if agent_type == "Unknown Agent":
            for name in ["QLearning", "SARSA", "DQN", "PolicyIteration", "ValueIteration"]:
                if name.lower() in str(type(env)).lower() or (
                    hasattr(env, 'agent') and name.lower() in str(type(env.agent)).lower()
                ):
                    agent_type = name
                    break
                    
        agent_text = self.font.render(f"Agent: {agent_type}", True, (180, 180, 220))
        self.screen.blit(agent_text, (panel_x, y_offset))
        y_offset += line_height
        
        # Draw stats in a box
        stats_box = pygame.Rect(panel_x - 10, y_offset, self.info_panel_rect.width - 30, 100)
        pygame.draw.rect(self.screen, (40, 40, 50), stats_box)
        pygame.draw.rect(self.screen, (80, 80, 100), stats_box, 1)
        
        stats_x = panel_x
        stats_y = y_offset + 10
        
        # Episode info
        if episode is not None:
            text = self.font.render(f"Episode: {episode}", True, (220, 220, 255))
            self.screen.blit(text, (stats_x, stats_y))
            stats_y += line_height
        
        # Step info
        if step is not None:
            text = self.font.render(f"Step: {step}", True, (220, 220, 255))
            self.screen.blit(text, (stats_x, stats_y))
            stats_y += line_height
        
        # Reward info with color based on value
        if reward is not None:
            reward_color = (220, 220, 255)  # Default color
            if reward > 0:
                # Green for positive rewards
                intensity = min(1.0, reward / 100.0)
                reward_color = (
                    int(220 * (1 - intensity)),
                    int(220 + 35 * intensity),
                    int(220 * (1 - intensity) + 35 * intensity)
                )
            elif reward < 0:
                # Red for negative rewards
                intensity = min(1.0, abs(reward) / 100.0)
                reward_color = (
                    int(220 + 35 * intensity),
                    int(220 * (1 - intensity)),
                    int(220 * (1 - intensity))
                )
                
            text = self.font.render(f"Reward: {reward:.1f}", True, reward_color)
            self.screen.blit(text, (stats_x, stats_y))
        
        y_offset += 120  # Move below stats box
        
        # Controls info with visual elements
        header = self.font.render("Controls", True, (180, 180, 220))
        self.screen.blit(header, (panel_x, y_offset))
        y_offset += line_height
        
        controls = [
            ("Space", "Pause"),
            ("V", "Toggle Values"),
            ("P", "Toggle Policy"),
            ("+", "Speed Up"),
            ("-", "Slow Down"),
            ("Q", "Quit")
        ]
        
        for key, action in controls:
            # Draw key in a small box
            key_text = self.small_font.render(key, True, (220, 220, 255))
            key_box = pygame.Rect(panel_x, y_offset, 30, 20)
            pygame.draw.rect(self.screen, (60, 60, 80), key_box)
            pygame.draw.rect(self.screen, (100, 100, 120), key_box, 1)
            
            # Center text in box
            key_rect = key_text.get_rect(center=key_box.center)
            self.screen.blit(key_text, key_rect)
            
            # Draw action description
            action_text = self.small_font.render(action, True, (180, 180, 220))
            self.screen.blit(action_text, (panel_x + 40, y_offset))
            
            y_offset += small_line_height
        
        # Add FPS counter at the bottom
        fps_text = self.small_font.render(f"FPS: {int(1 / max(0.001, self.dt))}", True, (150, 150, 180))
        self.screen.blit(fps_text, (panel_x, self.info_panel_rect.bottom - 30))

    def _get_cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Get the pygame Rect for a grid cell."""
        return pygame.Rect(
            int(self.grid_offset[0] + col * self.cell_size),
            int(self.grid_offset[1] + row * self.cell_size),
            int(self.cell_size),
            int(self.cell_size)
        )

    def _get_value_color(self, normalized_value: float) -> Tuple[int, int, int, int]:
        """Get color for value function visualization using an enhanced colormap."""
        # Use a more visually pleasing colormap (blue -> cyan -> green -> yellow -> red)
        if normalized_value < 0.25:
            # Blue to cyan
            t = normalized_value * 4
            r = 0
            g = int(255 * t)
            b = 255
        elif normalized_value < 0.5:
            # Cyan to green
            t = (normalized_value - 0.25) * 4
            r = 0
            g = 255
            b = int(255 * (1 - t))
        elif normalized_value < 0.75:
            # Green to yellow
            t = (normalized_value - 0.5) * 4
            r = int(255 * t)
            g = 255
            b = 0
        else:
            # Yellow to red
            t = (normalized_value - 0.75) * 4
            r = 255
            g = int(255 * (1 - t))
            b = 0
            
        # Add some alpha for a nice effect
        alpha = 170  # Semi-transparent
        return (r, g, b, alpha)

    def render(self, env, agent, episode: Optional[int] = None,
              step: Optional[int] = None, reward: Optional[float] = None) -> None:
        """Render the current state of the environment with visual enhancements.
        
        Args:
            env: GridWorld environment instance
            agent: RL agent instance
            episode: Current episode number
            step: Current step number
            reward: Current reward
        """
        # Make sure grid dimensions are calculated before rendering
        if self.cell_size is None or self.grid_offset is None:
            self._calculate_grid_dimensions(env)
        
        # Calculate frame time for smooth animations
        current_time = time.time()
        if hasattr(self, 'last_time'):
            self.dt = min(0.1, current_time - self.last_time)  # Cap at 0.1s to prevent large jumps
        self.last_time = current_time
        self.frame_count += 1
        
        # Reset dirty rects tracking for this frame
        self.dirty_rects = []
        
        # Update particle effects
        self._update_particles(self.dt)
        
        # ALWAYS ensure we draw a complete frame by filling the background first
        # This prevents the black screen issue by ensuring every pixel is drawn
        self.screen.fill((20, 20, 30))  # Fill with base background color
        
        # Draw gradient background on top
        self._draw_background()
        
        # Draw grid and terrain elements
        self._draw_grid(env)
        self._draw_terrain(env)
        self._draw_wind_zones(env)
        
        if agent is not None:
            self._draw_value_heatmap(env, agent)
            self._draw_policy_arrows(env, agent)
        
        # Draw goal and particles
        self._draw_goal(env)
        self._draw_particles()
        
        # Draw agent with animation
        self._draw_agent(env)
        
        # Draw info panel
        self._draw_info_panel(env, episode, step, reward)
        
        # IMPORTANT: Always use display.flip() to update the entire screen
        # This fixes the black screen issue by ensuring the full environment is always drawn
        pygame.display.flip()

    def simulate(self, env, agent, num_episodes: int = 1, max_steps: int = 200) -> None:
        """Run an interactive simulation of the environment.
        
        Args:
            env: GridWorld environment instance
            agent: RL agent instance
            num_episodes: Number of episodes to simulate
            max_steps: Maximum steps per episode
        """
        self.running = True
        episode = 0
        
        try:
            # Force an initial complete render of the environment before starting simulation
            # This ensures the grid is visible immediately upon startup
            self.render(env, agent, episode=1, step=0, reward=0.0)
            # Ensure display is properly initialized with two calls
            # (sometimes first flip doesn't display correctly due to driver issues)
            pygame.display.flip()
            pygame.time.wait(50)  # Short pause
            pygame.display.flip()  # Second flip to ensure display is refreshed
            pygame.time.wait(300)  # Pause to ensure first frame is visible
            
            while self.running and episode < num_episodes:
                obs = env.reset()
                done = False
                total_reward = 0
                step = 0
                
                # Render the initial state of each episode
                self.render(env, agent, episode=episode+1, step=0, reward=total_reward)
                pygame.display.flip()  # Full screen update for each new episode
                
                while not done and step < max_steps:
                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key in (pygame.K_q, pygame.K_ESCAPE):  # Add Escape key support
                                self.running = False
                                break
                            elif event.key == pygame.K_SPACE:
                                self.paused = not self.paused
                                # Force redraw when pausing to ensure UI updates
                                self.render(env, agent, episode+1, step, total_reward)
                                pygame.display.flip()
                            elif event.key == pygame.K_v:
                                self.show_values = not self.show_values
                                # Force redraw when toggling visualization features
                                self.render(env, agent, episode+1, step, total_reward)
                                pygame.display.flip()
                            elif event.key == pygame.K_p:
                                self.show_policy = not self.show_policy
                                # Force redraw when toggling visualization features
                                self.render(env, agent, episode+1, step, total_reward)
                                pygame.display.flip()
                            elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                                self.animation_speed = max(0.1, self.animation_speed - 0.1)
                                # Update UI to show changed animation speed
                                self.render(env, agent, episode+1, step, total_reward)
                                pygame.display.flip()
                            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                                self.animation_speed = min(2.0, self.animation_speed + 0.1)
                                # Update UI to show changed animation speed
                                self.render(env, agent, episode+1, step, total_reward)
                                pygame.display.flip()
                    
                    if not self.running:
                        break
                    
                    # Always render when paused to ensure display stays responsive
                    if self.paused:
                        self.render(env, agent, episode+1, step, total_reward)
                        pygame.display.flip()  # Ensure full screen update while paused
                        pygame.time.wait(100)  # Short delay when paused to reduce CPU usage
                        continue
                        
                    try:
                        # Get action based on agent type
                        if hasattr(agent, 'act'):
                            action = agent.act(obs)
                        elif hasattr(agent, 'online_net'):  # Agent is DQN agent
                            flat_obs = obs if not isinstance(obs, tuple) else obs[0]
                            action = dqn_act(flat_obs, agent.online_net, 0.01)
                        elif isinstance(agent, torch.nn.Module):  # Agent is direct DQN model
                            flat_obs = obs if not isinstance(obs, tuple) else obs[0]
                            action = dqn_act(flat_obs, agent, 0.01)
                        else:
                            # Default random action if agent type can't be determined
                            action = env.action_space.sample()
                            print("Warning: Using random action because agent type couldn't be determined")
                        
                        # Take step in environment
                        obs, reward, done, _ = env.step(action)
                        total_reward += reward
                        
                        # Render with full screen update for each step
                        self.render(env, agent, episode + 1, step + 1, total_reward)
                        # pygame.display.flip() is called inside render()
                        
                        # Ensure the frame rate is consistent
                        pygame.time.wait(int(self.animation_speed * 1000))
                        
                        step += 1
                    except Exception as e:
                        print(f"Error during simulation step: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next step instead of crashing
                
                if self.running:
                    episode += 1
        finally:
            # Always make sure to clean up PyGame resources
            self.cleanup()
        
    def cleanup(self):
        """Clean up PyGame resources."""
        try:
            pygame.quit()
        except Exception as e:
            print(f"Warning: Error during PyGame cleanup: {e}")

def main():
    """Example usage of the GridWorld visualizer."""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from envs.gridworld import GridWorldEnv
    from agents.q_learning import QLearningAgent
    from config import DEFAULT_ENV_CONFIG, QL_AGENT_CONFIG
    
    # Create environment and agent
    env = GridWorldEnv(DEFAULT_ENV_CONFIG)
    agent = QLearningAgent(env.observation_space, env.action_space,
                          env.config["grid_size"], QL_AGENT_CONFIG)
    
    # Create and run visualizer
    vis = GridWorldVisualizer()
    vis.simulate(env, agent)

if __name__ == "__main__":
    main()