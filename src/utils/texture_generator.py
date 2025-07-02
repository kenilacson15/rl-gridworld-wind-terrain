"""
Utility script to generate basic textures for the GridWorld environment.
This script will generate simple textures for different terrain types and save them to the assets folder.
"""

import pygame
import numpy as np
import os
import random
from pathlib import Path

class TextureGenerator:
    """Class to generate and save basic textures for the GridWorld environment."""
    
    def __init__(self, output_dir, base_size=128):
        """Initialize the texture generator.
        
        Args:
            output_dir: Directory where textures will be saved
            base_size: Base size for textures (will be square)
        """
        pygame.init()
        self.output_dir = Path(output_dir)
        self.base_size = base_size
        
        # Create directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _add_noise(self, surface, intensity=0.1, alpha=128):
        """Add noise to a surface to create texture variation.
        
        Args:
            surface: Pygame surface to add noise to
            intensity: Noise intensity (0.0-1.0)
            alpha: Alpha value for the noise (0-255)
        
        Returns:
            Surface with noise added
        """
        noise = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
        
        for y in range(surface.get_height()):
            for x in range(surface.get_width()):
                # Generate noise value
                noise_val = random.randint(-int(255 * intensity), int(255 * intensity))
                
                # Get original pixel color
                color = surface.get_at((x, y))
                
                # Add noise to each channel
                r = max(0, min(255, color[0] + noise_val))
                g = max(0, min(255, color[1] + noise_val))
                b = max(0, min(255, color[2] + noise_val))
                
                # Set new color with alpha
                noise.set_at((x, y), (r, g, b, alpha))
        
        # Combine surfaces
        result = surface.copy()
        result.blit(noise, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        return result

    def _generate_gradient(self, start_color, end_color, size):
        """Generate a gradient between two colors.
        
        Args:
            start_color: Starting color (r, g, b) or (r, g, b, a)
            end_color: Ending color (r, g, b) or (r, g, b, a)
            size: Tuple of (width, height)
        
        Returns:
            Surface with gradient
        """
        width, height = size
        surface = pygame.Surface(size, pygame.SRCALPHA)
        
        # Unpack colors, handle both RGB and RGBA
        sr, sg, sb = start_color[:3]
        sa = start_color[3] if len(start_color) > 3 else 255
        
        er, eg, eb = end_color[:3]
        ea = end_color[3] if len(end_color) > 3 else 255
        
        # Draw vertical gradient
        for y in range(height):
            # Calculate interpolation factor
            t = y / (height - 1) if height > 1 else 0
            
            # Interpolate color
            r = int(sr + (er - sr) * t)
            g = int(sg + (eg - sg) * t)
            b = int(sb + (eb - sb) * t)
            a = int(sa + (ea - sa) * t)
            
            # Draw horizontal line with this color
            pygame.draw.line(surface, (r, g, b, a), (0, y), (width, y))
        
        return surface

    def _add_pattern(self, surface, pattern_type='dots', color=(0, 0, 0, 50), density=0.1):
        """Add a pattern overlay to a surface.
        
        Args:
            surface: Pygame surface to add pattern to
            pattern_type: Type of pattern ('dots', 'lines', 'grid')
            color: Color for pattern
            density: Pattern density (0.0-1.0)
        
        Returns:
            Surface with pattern added
        """
        width, height = surface.get_width(), surface.get_height()
        pattern = pygame.Surface((width, height), pygame.SRCALPHA)
        
        if pattern_type == 'dots':
            # Calculate number of dots based on density
            num_dots = int(width * height * density / 100)
            for _ in range(num_dots):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                radius = random.randint(1, 3)
                pygame.draw.circle(pattern, color, (x, y), radius)
                
        elif pattern_type == 'lines':
            # Draw horizontal lines
            step = max(5, int(height * (1 - density) * 0.5))
            for y in range(0, height, step):
                pygame.draw.line(pattern, color, (0, y), (width, y), 1)
                
            # Draw some diagonal lines
            for i in range(-width, width * 2, step * 3):
                pygame.draw.line(pattern, color, (i, 0), (i + width, height), 1)
                
        elif pattern_type == 'grid':
            # Draw grid
            step_x = max(5, int(width * (1 - density) * 0.5))
            step_y = max(5, int(height * (1 - density) * 0.5))
            
            # Horizontal lines
            for y in range(0, height, step_y):
                pygame.draw.line(pattern, color, (0, y), (width, y), 1)
            
            # Vertical lines
            for x in range(0, width, step_x):
                pygame.draw.line(pattern, color, (x, 0), (x, height), 1)
        
        # Combine surfaces
        result = surface.copy()
        result.blit(pattern, (0, 0))
        return result

    def generate_normal_tile(self):
        """Generate a normal (grass) tile texture."""
        size = (self.base_size, self.base_size)
        
        # Create base grass color
        surface = pygame.Surface(size)
        base_color = (100, 180, 100)  # Green
        surface.fill(base_color)
        
        # Add noise for texture variation
        surface = self._add_noise(surface, intensity=0.2)
        
        # Add some darker green patches
        surface = self._add_pattern(surface, 'dots', (50, 120, 50, 100), 0.2)
        
        # Save the texture
        pygame.image.save(surface, self.output_dir / "normal.png")
        return surface

    def generate_ice_tile(self):
        """Generate an ice tile texture."""
        size = (self.base_size, self.base_size)
        
        # Create base ice color with gradient
        surface = self._generate_gradient(
            (200, 220, 255),  # Light blue
            (150, 180, 230),  # Slightly darker blue
            size
        )
        
        # Add some shine/reflection effects
        for _ in range(3):
            x = random.randint(0, size[0] - 1)
            y = random.randint(0, size[1] - 1)
            radius = random.randint(5, 20)
            highlight = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(highlight, (255, 255, 255, 100), (radius, radius), radius)
            surface.blit(highlight, (x - radius, y - radius))
        
        # Add cracks pattern
        surface = self._add_pattern(surface, 'lines', (220, 240, 255, 70), 0.05)
        
        # Save the texture
        pygame.image.save(surface, self.output_dir / "ice.png")
        return surface

    def generate_mud_tile(self):
        """Generate a mud tile texture."""
        size = (self.base_size, self.base_size)
        
        # Create base mud color
        surface = pygame.Surface(size)
        base_color = (139, 69, 19)  # Brown
        surface.fill(base_color)
        
        # Add noise for texture variation
        surface = self._add_noise(surface, intensity=0.3)
        
        # Add darker patches
        surface = self._add_pattern(surface, 'dots', (80, 40, 10, 100), 0.3)
        
        # Add some mud cracks
        surface = self._add_pattern(surface, 'grid', (160, 100, 60, 70), 0.05)
        
        # Save the texture
        pygame.image.save(surface, self.output_dir / "mud.png")
        return surface

    def generate_quicksand_tile(self):
        """Generate a quicksand tile texture."""
        size = (self.base_size, self.base_size)
        
        # Create base quicksand color
        surface = pygame.Surface(size)
        base_color = (194, 178, 128)  # Sand color
        surface.fill(base_color)
        
        # Add noise for texture variation
        surface = self._add_noise(surface, intensity=0.2)
        
        # Add swirl patterns to simulate quicksand movement
        center_x, center_y = size[0] // 2, size[1] // 2
        pattern = pygame.Surface(size, pygame.SRCALPHA)
        
        # Draw swirls
        for radius in range(10, size[0] // 2, 10):
            color = (150, 140, 100, 30)
            pygame.draw.circle(pattern, color, (center_x, center_y), radius, 2)
        
        surface.blit(pattern, (0, 0))
        
        # Save the texture
        pygame.image.save(surface, self.output_dir / "quicksand.png")
        return surface

    def generate_goal_tile(self):
        """Generate a goal tile texture."""
        size = (self.base_size, self.base_size)
        
        # Create base with gradient
        surface = self._generate_gradient(
            (50, 200, 50),  # Green
            (100, 255, 100),  # Lighter green
            size
        )
        
        # Add central highlight
        center = (size[0] // 2, size[1] // 2)
        
        # Draw outer circle
        pygame.draw.circle(surface, (200, 255, 200, 200), center, size[0] // 3)
        
        # Draw inner circle
        pygame.draw.circle(surface, (255, 255, 100, 200), center, size[0] // 5)
        
        # Save the texture
        pygame.image.save(surface, self.output_dir / "goal.png")
        return surface

    def generate_agent_sprite(self):
        """Generate a simple agent sprite (arrow-shaped)."""
        size = (self.base_size, self.base_size)
        
        # Create base sprite with transparency
        surface = pygame.Surface(size, pygame.SRCALPHA)
        
        # Draw agent as a colored circle with arrow
        center = (size[0] // 2, size[1] // 2)
        radius = size[0] // 3
        
        # Draw body
        pygame.draw.circle(surface, (255, 165, 0, 220), center, radius)  # Orange with alpha
        
        # Draw border
        pygame.draw.circle(surface, (0, 0, 0, 180), center, radius, 2)
        
        # Draw direction arrow (pointing up by default)
        arrow_points = [
            (center[0], center[1] - radius - 10),  # Tip
            (center[0] - 10, center[1] - radius + 5),  # Left corner
            (center[0] + 10, center[1] - radius + 5)   # Right corner
        ]
        pygame.draw.polygon(surface, (0, 0, 0, 200), arrow_points)
        
        # Save the base sprite
        pygame.image.save(surface, self.output_dir / "agent.png")
        
        # Create directional variants (rotate the original)
        directions = {
            'up': 0,
            'right': -90,  # Pygame rotation is clockwise (negative)
            'down': 180,
            'left': 90
        }
        
        # Save each directional sprite
        for direction, angle in directions.items():
            rotated = pygame.transform.rotate(surface, angle)
            pygame.image.save(rotated, self.output_dir / f"agent_{direction}.png")
        
        return surface

    def generate_particle_effects(self):
        """Generate particle textures for effects."""
        # Generate wind particle
        wind_particle = pygame.Surface((8, 8), pygame.SRCALPHA)
        pygame.draw.circle(wind_particle, (255, 255, 255, 150), (4, 4), 3)
        pygame.draw.circle(wind_particle, (200, 200, 255, 100), (4, 4), 4, 1)
        pygame.image.save(wind_particle, self.output_dir / "particle_wind.png")
        
        # Generate goal sparkle
        sparkle = pygame.Surface((16, 16), pygame.SRCALPHA)
        pygame.draw.line(sparkle, (255, 255, 100, 200), (8, 0), (8, 16), 2)
        pygame.draw.line(sparkle, (255, 255, 100, 200), (0, 8), (16, 8), 2)
        pygame.draw.line(sparkle, (255, 255, 100, 150), (2, 2), (14, 14), 1)
        pygame.draw.line(sparkle, (255, 255, 100, 150), (14, 2), (2, 14), 1)
        pygame.image.save(sparkle, self.output_dir / "particle_sparkle.png")
        
        return {
            'wind': wind_particle,
            'sparkle': sparkle
        }

    def generate_ui_elements(self):
        """Generate UI elements like buttons and panels."""
        # Generate panel background
        panel_bg = pygame.Surface((200, 400), pygame.SRCALPHA)
        panel_bg.fill((30, 30, 30, 180))  # Dark semi-transparent background
        pygame.draw.rect(panel_bg, (100, 100, 100, 200), panel_bg.get_rect(), 2)  # Border
        pygame.image.save(panel_bg, self.output_dir / "ui_panel.png")
        
        # Generate button
        button = pygame.Surface((120, 40), pygame.SRCALPHA)
        button.fill((80, 80, 80, 200))
        pygame.draw.rect(button, (120, 120, 120, 255), button.get_rect(), 2)  # Border
        pygame.image.save(button, self.output_dir / "ui_button.png")
        
        # Generate button hover state
        button_hover = pygame.Surface((120, 40), pygame.SRCALPHA)
        button_hover.fill((100, 100, 120, 220))
        pygame.draw.rect(button_hover, (150, 150, 200, 255), button_hover.get_rect(), 2)  # Border
        pygame.image.save(button_hover, self.output_dir / "ui_button_hover.png")
        
        return {
            'panel': panel_bg,
            'button': button,
            'button_hover': button_hover
        }

    def generate_all(self):
        """Generate all textures."""
        print("Generating textures...")
        self.generate_normal_tile()
        self.generate_ice_tile()
        self.generate_mud_tile()
        self.generate_quicksand_tile()
        self.generate_goal_tile()
        self.generate_agent_sprite()
        self.generate_particle_effects()
        self.generate_ui_elements()
        print(f"All textures generated and saved to {self.output_dir}")

def main():
    """Main function to generate all textures."""
    # Define paths relative to project root
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Get the project root directory (assuming this script is in src/utils)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    textures_dir = os.path.join(project_root, "assets", "textures")
    
    # Create generator and generate all textures
    generator = TextureGenerator(textures_dir)
    generator.generate_all()

if __name__ == "__main__":
    main()
