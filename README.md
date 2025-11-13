# Catan AI Training Environment

Clean, minimal environment for training AI agents to play Settlers of Catan.

## What's Included

| File | Purpose |
|------|---------|
| `ai_interface.py` | **Main environment** - Use this to train your AI |
| `game_system.py` | Core game logic (players, buildings, resources, ports, etc.) |
| `tile.py` | Hexagonal tile system |
| `test_environment.py` | Verify the environment works |

**That's it!** No GUI, no networking, no extra complexity.

---

## Quick Start

```bash
# 1. Test the environment
python3 test_environment.py

# 2. Use in your code
from ai_interface import AIGameEnvironment

env = AIGameEnvironment()
observations = env.reset()  # Returns list of 4 observations (one per player)

# Game loop
done = False
while not done:
    current_player = env.game.current_player_index
    obs = observations[current_player]

    # Your AI chooses action
    action = your_ai.choose_action(obs)
    params = your_ai.choose_params(obs)

    # Execute action
    obs, done, info = env.step(current_player, action, params)
    observations[current_player] = obs

    # Calculate your reward
    reward = your_reward_function(obs, info)

    # Train your AI
    your_ai.learn(obs, action, reward)
```

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
from ai_interface import AIGameEnvironment

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
