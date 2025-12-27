#!/usr/bin/env python3
"""
Predator-Prey Simulation MVP
2D grid world with grass, herbivores, and carnivores.
"""

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Callable

import numpy as np
import matplotlib.pyplot as plt
import yaml


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GrassConfig:
    initial_density: float = 0.3
    regrowth_rate: float = 0.05
    max_density: float = 0.5


@dataclass
class DayNightConfig:
    cycle_length: int = 100            # ステップ数で1日
    day_ratio: float = 0.6             # 昼の割合 (0-1)
    # 活動倍率 (視野・移動確率に影響)
    herbivore_day_activity: float = 1.0
    herbivore_night_activity: float = 0.5
    carnivore_day_activity: float = 0.7
    carnivore_night_activity: float = 1.0


@dataclass
class CreatureConfig:
    initial_count: int = 50
    max_count: int = 500
    initial_energy: int = 50
    max_energy: int = 100
    move_cost: int = 1
    eat_gain: int = 20
    reproduce_threshold: int = 70
    reproduce_cost: int = 40
    vision_range: int = 5
    # 草食動物の逃避能力
    flee_range: int = 0                # 0 = 逃げない、>0 = この距離内の捕食者から逃げる
    flee_speed: int = 1                # 逃走時の追加移動回数


@dataclass
class SimConfig:
    width: int = 100
    height: int = 100
    seed: int = 42
    steps: int = 1000
    display_interval: int = 100
    grass: GrassConfig = field(default_factory=GrassConfig)
    day_night: DayNightConfig = field(default_factory=DayNightConfig)
    herbivore: CreatureConfig = field(default_factory=lambda: CreatureConfig(
        initial_count=50, max_count=500, initial_energy=50, max_energy=100,
        move_cost=1, eat_gain=20, reproduce_threshold=70, reproduce_cost=40,
        vision_range=5, flee_range=4, flee_speed=1
    ))
    carnivore: CreatureConfig = field(default_factory=lambda: CreatureConfig(
        initial_count=20, max_count=200, initial_energy=80, max_energy=150,
        move_cost=2, eat_gain=50, reproduce_threshold=100, reproduce_cost=60,
        vision_range=7, flee_range=0, flee_speed=0
    ))

    @classmethod
    def from_yaml(cls, path: Path) -> "SimConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        world = data.get("world", {})
        sim = data.get("simulation", {})

        return cls(
            width=world.get("width", 100),
            height=world.get("height", 100),
            seed=world.get("seed", 42),
            steps=sim.get("steps", 1000),
            display_interval=sim.get("display_interval", 100),
            grass=GrassConfig(**data.get("grass", {})),
            day_night=DayNightConfig(**data.get("day_night", {})),
            herbivore=CreatureConfig(**data.get("herbivore", {})),
            carnivore=CreatureConfig(**data.get("carnivore", {})),
        )


# =============================================================================
# Pathfinding Interface (A*-ready)
# =============================================================================

class PathFinder(Protocol):
    """Protocol for pathfinding algorithms. Can be replaced with A*."""
    def find_next_step(
        self,
        start: tuple[int, int],
        targets: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        """Find the next step towards the nearest target."""
        ...


class GreedyPathFinder:
    """Simple greedy pathfinding: move towards the closest target."""

    def find_next_step(
        self,
        start: tuple[int, int],
        targets: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        if not targets:
            return None

        # Find nearest target (Manhattan distance)
        sx, sy = start
        nearest = min(targets, key=lambda t: abs(t[0] - sx) + abs(t[1] - sy))
        tx, ty = nearest

        # Determine direction
        dx = 0 if tx == sx else (1 if tx > sx else -1)
        dy = 0 if ty == sy else (1 if ty > sy else -1)

        # Try to move (prefer x, then y, then stay)
        candidates = []
        if dx != 0:
            candidates.append((sx + dx, sy))
        if dy != 0:
            candidates.append((sx, sy + dy))
        if dx != 0 and dy != 0:
            candidates.append((sx + dx, sy + dy))  # Diagonal

        for candidate in candidates:
            if candidate not in blocked:
                return candidate

        return None  # Can't move


# Placeholder for A* implementation
class AStarPathFinder:
    """A* pathfinding (stub for future implementation)."""

    def find_next_step(
        self,
        start: tuple[int, int],
        targets: list[tuple[int, int]],
        blocked: set[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        # TODO: Implement A* algorithm
        # For now, fall back to greedy
        return GreedyPathFinder().find_next_step(start, targets, blocked)


# =============================================================================
# Entities
# =============================================================================

@dataclass
class Creature:
    x: int
    y: int
    energy: int
    max_energy: int
    move_cost: int
    eat_gain: int
    reproduce_threshold: int
    reproduce_cost: int
    vision_range: int
    flee_range: int = 0
    flee_speed: int = 1
    alive: bool = True

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    def move_to(self, x: int, y: int, cost_multiplier: float = 1.0) -> None:
        self.x = x
        self.y = y
        self.energy -= int(self.move_cost * cost_multiplier)
        if self.energy <= 0:
            self.alive = False

    def eat(self) -> None:
        self.energy = min(self.energy + self.eat_gain, self.max_energy)

    def can_reproduce(self) -> bool:
        return self.energy >= self.reproduce_threshold

    def reproduce(self) -> "Creature":
        self.energy -= self.reproduce_cost
        return Creature(
            x=self.x,
            y=self.y,
            energy=self.reproduce_cost,
            max_energy=self.max_energy,
            move_cost=self.move_cost,
            eat_gain=self.eat_gain,
            reproduce_threshold=self.reproduce_threshold,
            reproduce_cost=self.reproduce_cost,
            vision_range=self.vision_range,
            flee_range=self.flee_range,
            flee_speed=self.flee_speed,
        )


# =============================================================================
# World
# =============================================================================

class World:
    def __init__(self, config: SimConfig, pathfinder: PathFinder = None):
        self.config = config
        self.pathfinder = pathfinder or GreedyPathFinder()
        self.rng = random.Random(config.seed)
        np.random.seed(config.seed)

        # Grass grid (True = has grass)
        self.grass = np.random.random((config.height, config.width)) < config.grass.initial_density

        # Creatures
        self.herbivores: list[Creature] = []
        self.carnivores: list[Creature] = []

        self._spawn_initial_creatures()

        # Day/Night cycle
        self.current_step = 0
        self.is_day = True

        # Statistics
        self.history = {
            "grass": [],
            "herbivores": [],
            "carnivores": [],
        }

    def _update_day_night(self) -> None:
        """Update day/night state based on current step."""
        dn = self.config.day_night
        time_in_cycle = self.current_step % dn.cycle_length
        day_steps = int(dn.cycle_length * dn.day_ratio)
        self.is_day = time_in_cycle < day_steps

    def _get_activity_multiplier(self, is_herbivore: bool) -> float:
        """Get activity multiplier based on time of day and creature type."""
        dn = self.config.day_night
        if is_herbivore:
            return dn.herbivore_day_activity if self.is_day else dn.herbivore_night_activity
        else:
            return dn.carnivore_day_activity if self.is_day else dn.carnivore_night_activity

    def _get_effective_vision(self, creature: Creature, is_herbivore: bool) -> int:
        """Get effective vision range based on activity level."""
        activity = self._get_activity_multiplier(is_herbivore)
        return max(1, int(creature.vision_range * activity))

    def _spawn_initial_creatures(self) -> None:
        """Spawn initial herbivores and carnivores at random positions."""
        positions = [
            (x, y)
            for x in range(self.config.width)
            for y in range(self.config.height)
        ]
        self.rng.shuffle(positions)

        hc = self.config.herbivore
        for i in range(min(hc.initial_count, len(positions))):
            x, y = positions[i]
            self.herbivores.append(Creature(
                x=x, y=y,
                energy=hc.initial_energy,
                max_energy=hc.max_energy,
                move_cost=hc.move_cost,
                eat_gain=hc.eat_gain,
                reproduce_threshold=hc.reproduce_threshold,
                reproduce_cost=hc.reproduce_cost,
                vision_range=hc.vision_range,
                flee_range=hc.flee_range,
                flee_speed=hc.flee_speed,
            ))

        cc = self.config.carnivore
        offset = hc.initial_count
        for i in range(min(cc.initial_count, len(positions) - offset)):
            x, y = positions[offset + i]
            self.carnivores.append(Creature(
                x=x, y=y,
                energy=cc.initial_energy,
                max_energy=cc.max_energy,
                move_cost=cc.move_cost,
                eat_gain=cc.eat_gain,
                reproduce_threshold=cc.reproduce_threshold,
                reproduce_cost=cc.reproduce_cost,
                vision_range=cc.vision_range,
                flee_range=cc.flee_range,
                flee_speed=cc.flee_speed,
            ))

    def _get_neighbors(self, x: int, y: int, radius: int) -> list[tuple[int, int]]:
        """Get all positions within radius (including diagonals)."""
        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.config.width and 0 <= ny < self.config.height:
                    neighbors.append((nx, ny))
        return neighbors

    def _find_grass_in_vision(self, creature: Creature, vision: int = None) -> list[tuple[int, int]]:
        """Find grass cells within creature's vision."""
        vision = vision if vision is not None else creature.vision_range
        targets = []
        for pos in self._get_neighbors(creature.x, creature.y, vision):
            if self.grass[pos[1], pos[0]]:
                targets.append(pos)
        return targets

    def _find_herbivores_in_vision(self, creature: Creature, vision: int = None) -> list[tuple[int, int]]:
        """Find herbivores within creature's vision."""
        vision = vision if vision is not None else creature.vision_range
        targets = []
        for h in self.herbivores:
            if not h.alive:
                continue
            dist = abs(h.x - creature.x) + abs(h.y - creature.y)
            if 0 < dist <= vision:
                targets.append((h.x, h.y))
        return targets

    def _find_carnivores_nearby(self, creature: Creature, distance: int) -> list[tuple[int, int]]:
        """Find carnivores within specified distance (for flee behavior)."""
        threats = []
        for c in self.carnivores:
            if not c.alive:
                continue
            dist = abs(c.x - creature.x) + abs(c.y - creature.y)
            if 0 < dist <= distance:
                threats.append((c.x, c.y))
        return threats

    def _find_flee_direction(self, creature: Creature, threats: list[tuple[int, int]],
                              occupied: set[tuple[int, int]]) -> Optional[tuple[int, int]]:
        """Find best direction to flee from threats."""
        if not threats:
            return None

        # Calculate average threat position
        avg_tx = sum(t[0] for t in threats) / len(threats)
        avg_ty = sum(t[1] for t in threats) / len(threats)

        # Move away from average threat position
        dx = -1 if avg_tx > creature.x else (1 if avg_tx < creature.x else 0)
        dy = -1 if avg_ty > creature.y else (1 if avg_ty < creature.y else 0)

        # Try to move (prefer diagonal away, then cardinal directions)
        candidates = []
        if dx != 0 and dy != 0:
            candidates.append((creature.x + dx, creature.y + dy))
        if dx != 0:
            candidates.append((creature.x + dx, creature.y))
        if dy != 0:
            candidates.append((creature.x, creature.y + dy))
        # Add perpendicular options as fallback
        if dx != 0:
            candidates.append((creature.x + dx, creature.y + 1))
            candidates.append((creature.x + dx, creature.y - 1))
        if dy != 0:
            candidates.append((creature.x + 1, creature.y + dy))
            candidates.append((creature.x - 1, creature.y + dy))

        for cx, cy in candidates:
            if (0 <= cx < self.config.width and
                0 <= cy < self.config.height and
                (cx, cy) not in occupied):
                return (cx, cy)

        return None

    def _get_occupied_positions(self) -> set[tuple[int, int]]:
        """Get all positions occupied by creatures."""
        positions = set()
        for h in self.herbivores:
            if h.alive:
                positions.add(h.pos)
        for c in self.carnivores:
            if c.alive:
                positions.add(c.pos)
        return positions

    def _step_herbivores(self) -> None:
        """Process herbivore actions."""
        self.rng.shuffle(self.herbivores)
        new_herbivores = []
        occupied = self._get_occupied_positions()
        activity = self._get_activity_multiplier(is_herbivore=True)

        for h in self.herbivores:
            if not h.alive:
                continue

            # Skip action based on activity level (night = less active)
            if activity < 1.0 and self.rng.random() > activity:
                continue

            # Check for nearby predators (flee behavior)
            fled = False
            if h.flee_range > 0:
                threats = self._find_carnivores_nearby(h, h.flee_range)
                if threats:
                    # Flee! Try to move multiple times based on flee_speed
                    for _ in range(h.flee_speed + 1):
                        flee_pos = self._find_flee_direction(h, threats, occupied)
                        if flee_pos:
                            occupied.discard(h.pos)
                            h.move_to(flee_pos[0], flee_pos[1], cost_multiplier=0.5)  # Flee is cheaper
                            occupied.add(h.pos)
                            fled = True
                            if not h.alive:
                                break
                        else:
                            break

            if not h.alive:
                occupied.discard(h.pos)
                continue

            # If didn't flee, do normal behavior
            if not fled:
                # Try to eat grass at current position
                if self.grass[h.y, h.x]:
                    h.eat()
                    self.grass[h.y, h.x] = False
                else:
                    # Find and move towards grass
                    effective_vision = self._get_effective_vision(h, is_herbivore=True)
                    targets = self._find_grass_in_vision(h, effective_vision)
                    if targets:
                        next_pos = self.pathfinder.find_next_step(h.pos, targets, occupied)
                        if next_pos:
                            occupied.discard(h.pos)
                            h.move_to(next_pos[0], next_pos[1])
                            occupied.add(h.pos)
                    else:
                        # Random walk
                        neighbors = self._get_neighbors(h.x, h.y, 1)
                        free = [n for n in neighbors if n not in occupied]
                        if free:
                            next_pos = self.rng.choice(free)
                            occupied.discard(h.pos)
                            h.move_to(next_pos[0], next_pos[1])
                            occupied.add(h.pos)
                        else:
                            h.energy -= h.move_cost  # Stuck, still costs energy

            if not h.alive:
                occupied.discard(h.pos)
                continue

            # Reproduce if possible
            if (h.can_reproduce() and
                len(self.herbivores) + len(new_herbivores) < self.config.herbivore.max_count):
                neighbors = self._get_neighbors(h.x, h.y, 1)
                free = [n for n in neighbors if n not in occupied]
                if free:
                    child = h.reproduce()
                    spawn_pos = self.rng.choice(free)
                    child.x, child.y = spawn_pos
                    new_herbivores.append(child)
                    occupied.add(spawn_pos)

        self.herbivores = [h for h in self.herbivores if h.alive] + new_herbivores

    def _step_carnivores(self) -> None:
        """Process carnivore actions."""
        self.rng.shuffle(self.carnivores)
        new_carnivores = []
        occupied = self._get_occupied_positions()
        herbivore_positions = {h.pos: h for h in self.herbivores if h.alive}
        activity = self._get_activity_multiplier(is_herbivore=False)

        for c in self.carnivores:
            if not c.alive:
                continue

            # Skip action based on activity level (day = less active for carnivores)
            if activity < 1.0 and self.rng.random() > activity:
                continue

            # Try to eat herbivore at current position
            if c.pos in herbivore_positions:
                prey = herbivore_positions[c.pos]
                prey.alive = False
                c.eat()
                del herbivore_positions[c.pos]
            else:
                # Find and move towards herbivores
                effective_vision = self._get_effective_vision(c, is_herbivore=False)
                targets = self._find_herbivores_in_vision(c, effective_vision)
                if targets:
                    next_pos = self.pathfinder.find_next_step(c.pos, targets, occupied - set(targets))
                    if next_pos:
                        occupied.discard(c.pos)
                        c.move_to(next_pos[0], next_pos[1])
                        occupied.add(c.pos)
                        # Check if caught prey
                        if c.pos in herbivore_positions:
                            prey = herbivore_positions[c.pos]
                            prey.alive = False
                            c.eat()
                            del herbivore_positions[c.pos]
                else:
                    # Random walk
                    neighbors = self._get_neighbors(c.x, c.y, 1)
                    free = [n for n in neighbors if n not in occupied]
                    if free:
                        next_pos = self.rng.choice(free)
                        occupied.discard(c.pos)
                        c.move_to(next_pos[0], next_pos[1])
                        occupied.add(c.pos)
                    else:
                        c.energy -= c.move_cost

            if not c.alive:
                occupied.discard(c.pos)
                continue

            # Reproduce if possible
            if (c.can_reproduce() and
                len(self.carnivores) + len(new_carnivores) < self.config.carnivore.max_count):
                neighbors = self._get_neighbors(c.x, c.y, 1)
                free = [n for n in neighbors if n not in occupied]
                if free:
                    child = c.reproduce()
                    spawn_pos = self.rng.choice(free)
                    child.x, child.y = spawn_pos
                    new_carnivores.append(child)
                    occupied.add(spawn_pos)

        # Update herbivores list
        self.herbivores = [h for h in self.herbivores if h.alive]
        self.carnivores = [c for c in self.carnivores if c.alive] + new_carnivores

    def _regrow_grass(self) -> None:
        """Regrow grass based on regrowth rate."""
        current_density = np.sum(self.grass) / (self.config.width * self.config.height)
        if current_density >= self.config.grass.max_density:
            return

        # Random regrowth
        regrow_mask = np.random.random((self.config.height, self.config.width)) < self.config.grass.regrowth_rate
        self.grass = self.grass | regrow_mask

        # Enforce max density
        current_density = np.sum(self.grass) / (self.config.width * self.config.height)
        if current_density > self.config.grass.max_density:
            excess = int((current_density - self.config.grass.max_density) * self.config.width * self.config.height)
            grass_positions = np.argwhere(self.grass)
            if len(grass_positions) > excess:
                remove_indices = np.random.choice(len(grass_positions), excess, replace=False)
                for idx in remove_indices:
                    y, x = grass_positions[idx]
                    self.grass[y, x] = False

    def step(self) -> None:
        """Execute one simulation step."""
        self._update_day_night()
        self._step_herbivores()
        self._step_carnivores()
        self._regrow_grass()

        # Record statistics
        self.history["grass"].append(np.sum(self.grass))
        self.history["herbivores"].append(len(self.herbivores))
        self.history["carnivores"].append(len(self.carnivores))

        self.current_step += 1

    def run(self, steps: int = None, verbose: bool = True) -> None:
        """Run simulation for given steps."""
        steps = steps or self.config.steps

        for i in range(steps):
            self.step()

            if verbose and (i + 1) % self.config.display_interval == 0:
                print(f"Step {i+1:5d}: Grass={self.history['grass'][-1]:5d}, "
                      f"Herbivores={self.history['herbivores'][-1]:4d}, "
                      f"Carnivores={self.history['carnivores'][-1]:4d}")

            # Early termination if ecosystem collapsed
            if len(self.herbivores) == 0 and len(self.carnivores) == 0:
                print(f"Ecosystem collapsed at step {i+1}")
                break

    def plot_history(self, output_path: str = None, show: bool = True) -> None:
        """Plot population history."""
        fig, ax = plt.subplots(figsize=(12, 6))

        steps = range(1, len(self.history["grass"]) + 1)

        ax.plot(steps, self.history["grass"], label="Grass", color="green", alpha=0.7)
        ax.plot(steps, self.history["herbivores"], label="Herbivores", color="blue", linewidth=2)
        ax.plot(steps, self.history["carnivores"], label="Carnivores", color="red", linewidth=2)

        ax.set_xlabel("Step")
        ax.set_ylabel("Count")
        ax.set_title("Predator-Prey Simulation")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Graph saved to {output_path}")

        if show:
            plt.show()

        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Predator-Prey Simulation")
    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"),
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("-s", "--steps", type=int, help="Override number of steps")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("-o", "--output", type=str, default="population_graph.png",
                        help="Output path for graph (default: population_graph.png)")
    parser.add_argument("--no-show", action="store_true", help="Don't display graph window")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress step output")
    parser.add_argument("-v", "--visual", action="store_true",
                        help="Run with pygame real-time visualization")
    parser.add_argument("--cell-size", type=int, default=8,
                        help="Cell size in pixels for visualization (default: 8)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target FPS for visualization (default: 30)")
    parser.add_argument("--speed", type=int, default=1,
                        help="Simulation steps per frame (default: 1)")
    args = parser.parse_args()

    # Load config
    if args.config.exists():
        config = SimConfig.from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = SimConfig()
        print("Using default config")

    # Override with CLI args
    if args.steps:
        config.steps = args.steps
    if args.seed:
        config.seed = args.seed

    print(f"World: {config.width}x{config.height}, Seed: {config.seed}, Steps: {config.steps}")
    print(f"Initial: Herbivores={config.herbivore.initial_count}, Carnivores={config.carnivore.initial_count}")
    print("-" * 60)

    # Create world
    world = World(config)

    if args.visual:
        # Run with pygame visualization
        try:
            from visualizer import run_visual
            run_visual(world, cell_size=args.cell_size, fps=args.fps, speed=args.speed)
        except ImportError as e:
            print(f"Error: pygame not installed. Run: pip install pygame")
            print(f"Details: {e}")
            return
    else:
        # Run headless simulation
        world.run(verbose=not args.quiet)

        print("-" * 60)
        print(f"Final: Grass={world.history['grass'][-1]}, "
              f"Herbivores={world.history['herbivores'][-1]}, "
              f"Carnivores={world.history['carnivores'][-1]}")

        # Plot results
        world.plot_history(output_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
