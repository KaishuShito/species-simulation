# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-25

### Added

- Initial release
- 2D grid world simulation with grass, herbivores, and carnivores
- YAML-based configuration system
- Day/night cycle affecting creature activity
- Herbivore flee behavior from predators
- Population time-series graph output (matplotlib)
- Real-time Pygame visualization with controls
- Reproducible simulations via random seed
- CLI with configurable options
- Two preset configurations: default and balanced

### Technical

- Greedy pathfinding (A* interface prepared)
- Population caps to prevent explosion
- Energy-based lifecycle mechanics
