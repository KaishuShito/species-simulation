#!/usr/bin/env python3
"""
Pygame-based real-time visualizer for Predator-Prey Simulation.
"""

import sys
from typing import TYPE_CHECKING

import pygame
import numpy as np

if TYPE_CHECKING:
    from simulation import World, SimConfig


# Colors (RGB)
COLOR_BG = (20, 20, 30)
COLOR_GRASS = (34, 139, 34)
COLOR_GRASS_EMPTY = (15, 40, 15)
COLOR_HERBIVORE = (65, 105, 225)
COLOR_CARNIVORE = (220, 60, 60)
COLOR_TEXT = (220, 220, 220)
COLOR_PANEL_BG = (30, 30, 40)
COLOR_GRAPH_LINE_GRASS = (34, 139, 34)
COLOR_GRAPH_LINE_HERB = (65, 105, 225)
COLOR_GRAPH_LINE_CARN = (220, 60, 60)

# Day/Night colors
COLOR_DAY_OVERLAY = (255, 255, 200, 30)    # Warm yellow tint for day
COLOR_NIGHT_OVERLAY = (20, 20, 60, 80)     # Dark blue tint for night
COLOR_DAY_TEXT = (255, 200, 50)
COLOR_NIGHT_TEXT = (100, 100, 200)


class Visualizer:
    """Real-time pygame visualizer for the simulation."""

    def __init__(
        self,
        world: "World",
        cell_size: int = 8,
        target_fps: int = 30,
        steps_per_frame: int = 1,
    ):
        self.world = world
        self.cell_size = cell_size
        self.target_fps = target_fps
        self.steps_per_frame = steps_per_frame

        # Calculate dimensions
        self.grid_width = world.config.width * cell_size
        self.grid_height = world.config.height * cell_size
        self.panel_width = 300
        self.graph_height = 150
        self.window_width = self.grid_width + self.panel_width
        self.window_height = max(self.grid_height, 400)

        # State
        self.running = True
        self.paused = False
        self.step_count = 0

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Predator-Prey Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 18, bold=True)

        # Pre-render grid surfaces for performance
        self._init_grid_surface()

    def _init_grid_surface(self) -> None:
        """Initialize the grid surface."""
        self.grid_surface = pygame.Surface((self.grid_width, self.grid_height))
        self.overlay_surface = pygame.Surface((self.grid_width, self.grid_height), pygame.SRCALPHA)

    def _get_time_adjusted_color(self, base_color: tuple, is_day: bool) -> tuple:
        """Adjust color based on time of day."""
        if is_day:
            # Slightly brighter during day
            return tuple(min(255, int(c * 1.1)) for c in base_color)
        else:
            # Darker and bluer at night
            r, g, b = base_color
            return (int(r * 0.6), int(g * 0.6), min(255, int(b * 0.8 + 30)))

    def _draw_grid(self) -> None:
        """Draw the world grid with grass and creatures."""
        is_day = self.world.is_day

        # Adjust colors based on time
        grass_color = self._get_time_adjusted_color(COLOR_GRASS, is_day)
        grass_empty_color = self._get_time_adjusted_color(COLOR_GRASS_EMPTY, is_day)

        # Fill with empty grass color
        self.grid_surface.fill(grass_empty_color)

        # Draw grass
        for y in range(self.world.config.height):
            for x in range(self.world.config.width):
                if self.world.grass[y, x]:
                    rect = pygame.Rect(
                        x * self.cell_size,
                        y * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    pygame.draw.rect(self.grid_surface, grass_color, rect)

        # Adjust creature colors based on time
        herb_color = self._get_time_adjusted_color(COLOR_HERBIVORE, is_day)
        carn_color = self._get_time_adjusted_color(COLOR_CARNIVORE, is_day)

        # Draw herbivores
        for h in self.world.herbivores:
            if h.alive:
                center = (
                    h.x * self.cell_size + self.cell_size // 2,
                    h.y * self.cell_size + self.cell_size // 2,
                )
                radius = max(2, self.cell_size // 2 - 1)
                pygame.draw.circle(self.grid_surface, herb_color, center, radius)

        # Draw carnivores
        for c in self.world.carnivores:
            if c.alive:
                center = (
                    c.x * self.cell_size + self.cell_size // 2,
                    c.y * self.cell_size + self.cell_size // 2,
                )
                radius = max(3, self.cell_size // 2)
                pygame.draw.circle(self.grid_surface, carn_color, center, radius)

        self.screen.blit(self.grid_surface, (0, 0))

        # Apply day/night overlay
        if not is_day:
            self.overlay_surface.fill((20, 20, 80, 60))
            self.screen.blit(self.overlay_surface, (0, 0))

    def _draw_panel(self) -> None:
        """Draw the info panel on the right side."""
        panel_x = self.grid_width
        panel_rect = pygame.Rect(panel_x, 0, self.panel_width, self.window_height)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, panel_rect)

        y_offset = 20

        # Title
        title = self.font_large.render("Population Stats", True, COLOR_TEXT)
        self.screen.blit(title, (panel_x + 10, y_offset))
        y_offset += 35

        # Step counter
        step_text = self.font.render(f"Step: {self.step_count}", True, COLOR_TEXT)
        self.screen.blit(step_text, (panel_x + 10, y_offset))
        y_offset += 25

        # Status
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = (255, 200, 50) if self.paused else (50, 255, 50)
        status_text = self.font.render(f"Status: {status}", True, status_color)
        self.screen.blit(status_text, (panel_x + 10, y_offset))
        y_offset += 25

        # Day/Night indicator
        is_day = self.world.is_day
        dn_config = self.world.config.day_night
        time_in_cycle = self.world.current_step % dn_config.cycle_length
        time_label = "DAY" if is_day else "NIGHT"
        time_color = COLOR_DAY_TEXT if is_day else COLOR_NIGHT_TEXT
        time_text = self.font.render(f"Time: {time_label} ({time_in_cycle}/{dn_config.cycle_length})", True, time_color)
        self.screen.blit(time_text, (panel_x + 10, y_offset))
        y_offset += 35

        # Population counts
        grass_count = np.sum(self.world.grass)
        herb_count = len(self.world.herbivores)
        carn_count = len(self.world.carnivores)

        grass_text = self.font.render(f"Grass:      {grass_count:5d}", True, COLOR_GRASS)
        herb_text = self.font.render(f"Herbivores: {herb_count:5d}", True, COLOR_HERBIVORE)
        carn_text = self.font.render(f"Carnivores: {carn_count:5d}", True, COLOR_CARNIVORE)

        self.screen.blit(grass_text, (panel_x + 10, y_offset))
        y_offset += 22
        self.screen.blit(herb_text, (panel_x + 10, y_offset))
        y_offset += 22
        self.screen.blit(carn_text, (panel_x + 10, y_offset))
        y_offset += 40

        # Controls
        controls_title = self.font_large.render("Controls", True, COLOR_TEXT)
        self.screen.blit(controls_title, (panel_x + 10, y_offset))
        y_offset += 30

        controls = [
            "SPACE  - Pause/Resume",
            "UP/DOWN - Speed +/-",
            "R      - Reset",
            "S      - Single step",
            "G      - Save graph",
            "ESC/Q  - Quit",
        ]
        for ctrl in controls:
            ctrl_text = self.font.render(ctrl, True, (180, 180, 180))
            self.screen.blit(ctrl_text, (panel_x + 10, y_offset))
            y_offset += 20

        y_offset += 20

        # Speed info
        speed_text = self.font.render(f"Speed: {self.steps_per_frame} steps/frame", True, COLOR_TEXT)
        self.screen.blit(speed_text, (panel_x + 10, y_offset))
        y_offset += 20

        fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, COLOR_TEXT)
        self.screen.blit(fps_text, (panel_x + 10, y_offset))
        y_offset += 40

        # Mini graph
        self._draw_mini_graph(panel_x + 10, y_offset)

    def _draw_mini_graph(self, x: int, y: int) -> None:
        """Draw a mini population graph in the panel."""
        graph_width = self.panel_width - 20
        graph_height = self.graph_height

        # Background
        graph_rect = pygame.Rect(x, y, graph_width, graph_height)
        pygame.draw.rect(self.screen, (40, 40, 50), graph_rect)
        pygame.draw.rect(self.screen, (60, 60, 70), graph_rect, 1)

        history = self.world.history
        if len(history["grass"]) < 2:
            return

        # Get last N points
        max_points = graph_width
        grass_data = history["grass"][-max_points:]
        herb_data = history["herbivores"][-max_points:]
        carn_data = history["carnivores"][-max_points:]

        # Find max for scaling
        max_val = max(
            max(grass_data) if grass_data else 1,
            max(herb_data) if herb_data else 1,
            max(carn_data) if carn_data else 1,
            1,
        )

        def scale_points(data, color):
            if len(data) < 2:
                return
            points = []
            for i, val in enumerate(data):
                px = x + int(i * graph_width / len(data))
                py = y + graph_height - int(val * (graph_height - 10) / max_val) - 5
                points.append((px, py))
            if len(points) >= 2:
                pygame.draw.lines(self.screen, color, False, points, 1)

        scale_points(grass_data, COLOR_GRAPH_LINE_GRASS)
        scale_points(herb_data, COLOR_GRAPH_LINE_HERB)
        scale_points(carn_data, COLOR_GRAPH_LINE_CARN)

    def _handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_UP:
                    self.steps_per_frame = min(self.steps_per_frame + 1, 50)
                elif event.key == pygame.K_DOWN:
                    self.steps_per_frame = max(self.steps_per_frame - 1, 1)
                elif event.key == pygame.K_r:
                    self._reset()
                elif event.key == pygame.K_s:
                    # Single step when paused
                    if self.paused:
                        self.world.step()
                        self.step_count += 1
                elif event.key == pygame.K_g:
                    self.world.plot_history(output_path="population_graph.png", show=False)
                    print("Graph saved to population_graph.png")

    def _reset(self) -> None:
        """Reset the simulation."""
        from simulation import World
        self.world = World(self.world.config)
        self.step_count = 0
        self.paused = True

    def run(self) -> None:
        """Main visualization loop."""
        print("Visualizer started. Press ESC or Q to quit.")

        while self.running:
            self._handle_events()

            # Update simulation
            if not self.paused:
                for _ in range(self.steps_per_frame):
                    self.world.step()
                    self.step_count += 1

                    # Check for ecosystem collapse
                    if len(self.world.herbivores) == 0 and len(self.world.carnivores) == 0:
                        print(f"Ecosystem collapsed at step {self.step_count}")
                        self.paused = True
                        break

            # Draw
            self.screen.fill(COLOR_BG)
            self._draw_grid()
            self._draw_panel()

            pygame.display.flip()
            self.clock.tick(self.target_fps)

        # Cleanup
        pygame.quit()

        # Save final graph
        if len(self.world.history["grass"]) > 0:
            self.world.plot_history(output_path="population_graph.png", show=False)
            print(f"Final graph saved. Total steps: {self.step_count}")


def run_visual(world: "World", cell_size: int = 8, fps: int = 30, speed: int = 1) -> None:
    """Convenience function to run the visualizer."""
    viz = Visualizer(world, cell_size=cell_size, target_fps=fps, steps_per_frame=speed)
    viz.run()
