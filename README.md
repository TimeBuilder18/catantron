# Catan Game - Three Versions

This project provides **three different ways** to play/train Catan:

## 🎮 Three Versions

| File | Purpose | Use Case |
|------|---------|----------|
| **`main.py`** | Full pygame game with complete UI | Human play, full features with trading |
| **`visual_ai_game.py`** | Visual AI training environment | Watch AI train, simplified (no trading) |
| **`ai_interface.py`** | Headless AI training | Fast AI training, no visualization |

### Supporting Files
- `game_system.py` - Core game logic (players, buildings, resources, ports)
- `tile.py` - Hexagonal tile system
- `test_environment.py` - Test that environment works

---

## Quick Start

### 🎮 Option 1: Play the Full Game (Human Players)
```bash
python3 huPlay.py
```
- Full pygame UI with all features
- Supports trading, development cards, etc.
- Mouse + keyboard controls
- 4 players on one screen

### 🤖 Option 2: Visual AI Training (Watch AI Learn)
```bash
python3 visual_ai_game.py
```
- See AI agents play in real-time
- Simplified game (no trading)
- Better for AI training
- Manual controls for testing

### ⚡ Option 3: Headless AI Training (Maximum Speed)
```bash
# Test the environment
python3 test_environment.py

# Use in your code
from ai_interface import AIGameEnvironment

env = AIGameEnvironment()
observations = env.reset()  # Returns list of 4 observations

done = False
while not done:
    current_player = env.game.current_player_index
    obs = observations[current_player]

    # Your AI decides
    action = your_ai.choose_action(obs)
    params = your_ai.choose_params(obs)

    # Execute
    obs, done, info = env.step(current_player, action, params)
    observations[current_player] = obs

    # Calculate reward
    reward = your_reward_function(obs, info)
    your_ai.learn(obs, action, reward)
```

---

---

## Key Game Rules Implemented

### Robber & Discard Rule (7 Rolled)
When a **7** is rolled:
1. **All players with 8+ cards must discard half** (rounded down)
   - Example: 8 cards → discard 4, 9 cards → discard 4, 10 cards → discard 5
2. Players **choose which cards** to discard
3. After all discards complete, **robber is moved**
4. Current player steals from adjacent players

---

## Observation Format

Each observation contains:

```python
obs = {
    # Turn info
    'is_my_turn': bool,
    'current_player': int,
    'game_phase': str,  # 'INITIAL_PLACEMENT_1', 'INITIAL_PLACEMENT_2', 'NORMAL_PLAY'
    'turn_phase': str,  # 'ROLL_DICE', 'TRADE_BUILD', etc.

    # My private info
    'my_resources': {'wood': 0, 'brick': 0, 'wheat': 0, 'sheep': 0, 'ore': 0},
    'my_dev_cards': {...},
    'my_settlements': int,
    'my_cities': int,
    'my_roads': int,
    'my_victory_points': int,

    # Opponents (public info only)
    'opponents': [
        {'resource_count': int, 'settlements': int, ...},
        {...},
        {...}
    ],

    # Board state
    'tiles': [(q, r, resource, number), ...],
    'ports': [...],

    # What you can do
    'legal_actions': ['roll_dice', 'place_settlement', ...]
}
```

---

## Available Actions

| Action | When | Params |
|--------|------|--------|
| `'roll_dice'` | Start of turn | None |
| `'place_settlement'` | Initial placement / After rolling | `{'vertex': Vertex}` |
| `'place_road'` | After settlement | `{'edge': Edge}` |
| `'build_settlement'` | Your turn, after dice | `{'vertex': Vertex}` |
| `'build_city'` | Your turn, after dice | `{'vertex': Vertex}` |
| `'build_road'` | Your turn, after dice | `{'edge': Edge}` |
| `'buy_dev_card'` | Your turn, after dice | None |
| `'end_turn'` | After rolling | None |

---

## Game Rules (Standard Catan)

- **4 players** compete to reach **10 victory points**
- **Initial placement:** Each player places 2 settlements + 2 roads (forward then reverse order)
- **Turn structure:** Roll dice → Collect resources → Build/Trade → End turn
- **Building costs:**
  - Settlement: 1 wood, 1 brick, 1 wheat, 1 sheep
  - City: 2 wheat, 3 ore
  - Road: 1 wood, 1 brick
  - Dev Card: 1 wheat, 1 sheep, 1 ore
- **Trading:** 4:1 with bank, 3:1 with generic port, 2:1 with specialized port
- **Victory points:** Settlements (1), Cities (2), Longest Road (2), Largest Army (2), VP cards (1 each)

---

## Example: Random Agent

```python
import random
from scripts.ai_interface import AIGameEnvironment

env = AIGameEnvironment()
observations = env.reset()

done = False
while not done:
    current_player = env.game.current_player_index
    obs = observations[current_player]

    # Choose random legal action
    action = random.choice(obs['legal_actions'])

    # Get random valid parameters
    if action == 'place_settlement':
        # Find empty vertex (simplified - you'd want smarter logic)
        vertices = env.game.game_board.vertices
        empty = [v for v in vertices if v.structure is None]
        params = {'vertex': random.choice(empty)} if empty else None
    elif action == 'place_road':
        # Find empty edge connected to your infrastructure
        edges = env.game.game_board.edges
        empty = [e for e in edges if e.structure is None]
        params = {'edge': random.choice(empty)} if empty else None
    else:
        params = None

    # Execute
    obs, done, info = env.step(current_player, action, params)
    observations[current_player] = obs

    if done:
        print(f"Game over! Winner: Player {info['winner'] + 1}")
```

---

## Training Tips

1. **Start simple:** Get a random agent working first
2. **Action masking:** Use `obs['legal_actions']` to filter invalid actions
3. **Partial observability:** Each agent only sees their own hand
4. **Reward shaping:** Design rewards that guide learning (VPs, building, resource collection, etc.)
5. **Curriculum learning:** Train on simpler tasks first (e.g., just initial placement)

---

## Architecture for Multi-Agent RL

```
┌─────────────────────────────────┐
│   AIGameEnvironment (this repo) │
└─────────────────────────────────┘
         ↓         ↓         ↓         ↓
     [Agent 1] [Agent 2] [Agent 3] [Agent 4]
       (You)     (You)     (You)     (You)
```

**You implement:**
- Neural network architecture
- Action selection policy
- Reward function
- Training algorithm (PPO, DQN, A3C, etc.)

**We provide:**
- Game rules and state management
- Legal action masking
- Observation formatting
- Turn management

---

## Requirements

```bash
pip install pygame  # Only needed by game_system.py (legacy import)
```

Note: We don't actually use pygame for AI training, but `game_system.py` has a legacy import. You can remove it if you want to make the environment pygame-free.

---

## Questions?

Read the docstrings in `ai_interface.py` for detailed API documentation.

**Good luck training your agents!** 🤖
