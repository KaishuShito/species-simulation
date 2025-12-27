# Predator-Prey Simulation

A simple 2D predator-prey ecosystem simulation built entirely with [Claude Code](https://claude.ai/code).

## Inspiration

This project was inspired by [Import AI #438](https://importai.substack.com/p/import-ai-438-cyber-capability-overhang), where the author describes building a similar simulation with Claude Code in about 5 minutes:

> "I fired up Claude Code with Opus 4.5 and got it to build a predator-prey species simulation... producing in about 5 minutes something which I know took me several weeks to build a decade ago."

I wanted to try it myself. This is the result of that experiment.

## What It Does

- Grass grows on a 2D grid
- Herbivores eat grass and flee from predators
- Carnivores hunt herbivores
- Day/night cycle affects creature activity
- Watch population dynamics unfold in real-time

## Quick Start

```bash
pip install -r requirements.txt

# Run with visualization
python simulation.py -v

# Or headless (outputs a graph)
python simulation.py
```

## Controls (Visualization Mode)

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume |
| `UP/DOWN` | Speed +/- |
| `R` | Reset |
| `ESC` | Quit |

## Note

This is an experimental project made purely out of curiosity. No roadmap, no promises. Just vibes.

## License

MIT
