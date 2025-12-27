# Contributing to Predator-Prey Simulation

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs

1. Check existing [Issues](../../issues) to avoid duplicates
2. Create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Configuration file used (if applicable)

### Suggesting Features

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. If possible, outline a proposed implementation

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run quality checks:
   ```bash
   black simulation.py visualizer.py
   isort simulation.py visualizer.py
   mypy simulation.py
   pytest  # if tests exist
   ```
5. Commit with clear messages: `feat: add omnivore species`
6. Push and create a Pull Request

## Code Style

- **Formatter**: Black (default settings)
- **Import sorting**: isort
- **Type hints**: Required for public interfaces
- **Naming**:
  - Modules/functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

## Commit Message Format

Follow conventional commits:

```
feat: add new feature
fix: correct bug in pathfinding
refactor: restructure World class
docs: update README
test: add unit tests for Creature
```

## Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] Changes are tested (manual or automated)
- [ ] Documentation updated if needed
- [ ] No unrelated changes included
- [ ] PR description explains what and why

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/predator-prey-simulation.git
cd predator-prey-simulation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev tools
pip install black isort mypy pytest
```

## Questions?

Feel free to open an issue for questions or discussions.
