# Catantron - Catan AI Training Environment

A reinforcement learning environment for training AI agents to play Catan using PyTorch and PPO.

---

## Quick Start

### Play the Game (Human)
```bash
python3 huPlay.py
```

### Train AI (GPU - Recommended)
```bash
python3 train_gpu.py --episodes 5000 --model-name my_model --curriculum
```

---

## Project Structure

### Core Files
| File | Purpose |
|------|---------|
| `huPlay.py` | Full pygame GUI game for human play |
| `visual_ai_game.py` | Visual AI training with pygame |
| `train_gpu.py` | Main GPU training script |
| `game_system.py` | Core Catan game logic |
| `tile.py` | Hexagonal tile system |
| `ai_interface.py` | Headless AI training environment |
| `network_gpu.py` | Neural network with GPU support |
| `agent_gpu.py` | RL agent implementation |
| `trainer_gpu.py` | PPO trainer |
| `rule_based_ai.py` | Rule-based AI opponents |
| `catan_env_pytorch.py` | PyTorch environment wrapper |

### Deprecated Files
See `useless_files/` folder for old/unused scripts and documentation.

---

## Game Features

### Core Mechanics
- 4 players, 10 VP to win
- Initial placement (2 settlements + 2 roads each)
- Dice rolling, resource distribution
- Building: settlements, cities, roads
- Development cards (Knights, VP, Road Building, Year of Plenty, Monopoly)
- Robber mechanics (7 rolled = discard if 8+ cards)
- Trading (4:1 bank, 3:1/2:1 ports)
- Longest Road & Largest Army bonuses

### Building Costs
| Building | Cost |
|----------|------|
| Settlement | 1 wood, 1 brick, 1 wheat, 1 sheep |
| City | 2 wheat, 3 ore |
| Road | 1 wood, 1 brick |
| Dev Card | 1 wheat, 1 sheep, 1 ore |

---

## Training Guide

### Basic Commands
```bash
# Quick test (10 episodes)
python3 train_gpu.py --episodes 10 --model-name test

# Standard training (~40 min on RTX 2080)
python3 train_gpu.py --episodes 5000 --model-name catan_v1 --curriculum

# Extended training (~3-4 hours)
python3 train_gpu.py --episodes 25000 --model-name extended --curriculum --batch-size 1024
```

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 1000 | Number of training episodes |
| `--curriculum` | False | Enable curriculum learning |
| `--batch-size` | 256 | Batch size for updates |
| `--epochs` | 10 | PPO epochs per update |
| `--update-freq` | 20 | Episodes between updates |
| `--save-freq` | 500 | Episodes between saves |

### Curriculum Learning
Stages: VP targets [4, 5, 6, 7, 8, 10]
- Agent advances when achieving 90% of current target

---

## AI Interface

### Observation Format
```python
obs = {
    'is_my_turn': bool,
    'game_phase': str,  # 'INITIAL_PLACEMENT_1', 'INITIAL_PLACEMENT_2', 'NORMAL_PLAY'
    'my_resources': {'wood': 0, 'brick': 0, 'wheat': 0, 'sheep': 0, 'ore': 0},
    'my_settlements': int,
    'my_cities': int,
    'my_roads': int,
    'my_victory_points': int,
    'legal_actions': ['roll_dice', 'place_settlement', ...]
}
```

### Actions
| Action | When | Params |
|--------|------|--------|
| `roll_dice` | Start of turn | None |
| `place_settlement` | Initial/After rolling | `{'vertex': Vertex}` |
| `place_road` | After settlement | `{'edge': Edge}` |
| `build_settlement` | Your turn | `{'vertex': Vertex}` |
| `build_city` | Your turn | `{'vertex': Vertex}` |
| `build_road` | Your turn | `{'edge': Edge}` |
| `buy_dev_card` | Your turn | None |
| `end_turn` | After rolling | None |

---

## Reward System (Outcome-Based)

| Component | Value | Purpose |
|-----------|-------|---------|
| VP changes | +3.0 per VP | Main learning signal |
| PBRS | 10x multiplier | Strategic quality guidance |
| Win bonus | +20.0 | Terminal reward |
| Inaction penalty | -3.0 | Prevents passing when can build |

---

## Performance

### GPU Training Speed (RTX 2080 Super)
| Episodes | Time |
|----------|------|
| 1,000 | ~8 min |
| 5,000 | ~40 min |
| 25,000 | ~3-4 hours |

### Expected Learning
| Episodes | Expected VP |
|----------|-------------|
| 0-5k | 2.3-2.6 |
| 5k-15k | 2.6-2.8 |
| 15k-25k | 2.7-3.0 |

---

## Setup

### Requirements
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gymnasium numpy matplotlib pygame
```

### Verify GPU
```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

---

## Architecture

```
train_gpu.py
  ├── catan_env_pytorch.py (environment)
  ├── agent_gpu.py (RL agent)
  ├── trainer_gpu.py (PPO training)
  └── rule_based_ai.py (opponents)

game_system.py
  ├── tile.py (board/hexes)
  └── ai_interface.py (headless env)
```

---

## Tips

1. **Start simple**: Test with 10-50 episodes first
2. **Use curriculum**: Helps prevent getting stuck
3. **Monitor checkpoints**: Best model may not be final model
4. **Watch for exploitation**: Roads/game should be 20-35, not 100+
5. **GPU recommended**: 20-30x faster than CPU
