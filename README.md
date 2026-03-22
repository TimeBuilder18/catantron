# Catantron — AI Agent for 1v1 Catan

A reinforcement learning AI agent trained with PPO and curriculum learning to play 1v1 Catan. The agent learns from scratch — starting against opponents that do literally nothing and gradually facing stronger opponents until it can hold its own against our best rule-based AI.

## How to Play

```bash
python main.py
```

This launches the game with a title screen where you can choose:
- **Player vs Player** — Local hotseat mode
- **Player vs AI** — Play against the trained neural network
- **Player vs Bot** — Play against rule-based AI at various difficulty levels
- **Watch AI** — Spectate the trained model playing

## Project Structure

```
catantron/
├── main.py                          # Entry point — launches the game
├── README.md
├── requirements.txt
│
├── game/                            # Core game engine
│   ├── game_system.py               # Full Catan rules: board, players, building, trading,
│   │                                #   dev cards, robber, victory detection
│   └── hexagon.py                   # Hexagonal tile system with axial coordinates
│
├── ai/                              # AI opponents
│   ├── scripted_opponents.py        # Rule-based AI (passive → strong difficulty levels)
│   ├── weighted_opponent.py         # Weighted random opponent (catanatron-style)
│   ├── specialist_opponents.py      # Strategy specialists (city rusher, road blocker, etc.)
│   └── opponent_pool.py             # Self-play pool with frozen past checkpoints
│
├── environment/                     # RL environment
│   ├── catan_env.py                 # Gymnasium wrapper (575-dim obs, 14-action hierarchical space)
│   ├── ai_interface.py              # Headless game interface for fast training
│   ├── pbrs_rewards.py              # Potential-based reward shaping wrapper
│   ├── simple_rewards.py            # Simpler reward modes (sparse, VP-only, simplified)
│   └── reward_utils.py              # Shared reward scoring utilities
│
├── model/                           # Neural network
│   ├── network.py                   # Multi-head architecture (action, vertex, edge, trade heads)
│   ├── model_loader.py              # Model loading/saving with backward compatibility
│   └── agent.py                     # RL agent with hierarchical action selection
│
├── training/                        # Training
│   └── trainer.py                   # Curriculum trainer (parallel games, adaptive phases, self-play)
│
├── evaluation/                      # Evaluation & diagnostics
│   ├── evaluator.py                 # Benchmark evaluation against all opponent levels
│   ├── quality_score.py             # Agent Quality Score (6-component composite metric)
│   └── replay_viewer.py             # Terminal-based game replay viewer
│
├── gui/                             # Visual game interface
│   ├── catan_game.py                # Pygame AI game viewer
│   └── gui_components.py            # Shared GUI drawing functions
│
├── models/                          # Trained model weights (.pt files)
└── archive/                         # Old/deprecated files kept for reference
```

## How the AI Works

### Training Pipeline

1. **Game Engine** (`game/game_system.py`) — Full Catan rules implementation
2. **Environment Wrapper** (`environment/catan_env.py`) — Converts the game state into a 575-dimensional observation vector with action masking
3. **Neural Network** (`model/network.py`) — Multi-head feedforward network with tile attention
4. **PPO Training** (`training/trainer.py`) — Proximal Policy Optimization with curriculum learning
5. **Evaluation** (`evaluation/evaluator.py`) — Benchmarks against rule-based opponents

### Curriculum Learning

The agent can't learn Catan from scratch against a strong opponent — it would never win and get no learning signal. So we use curriculum learning with 12+ phases:

| Phase | Opponent | What the Agent Learns |
|-------|----------|----------------------|
| 0 | Passive (does nothing) | Basic building mechanics |
| 1 | Truly random | Competing for limited resources |
| 2 | Weighted random | Building prioritization |
| 3-5 | Very weak → Weak → Medium | Board evaluation, resource management |
| 6-7 | Strong rule-based | Full strategic play |
| 8-11 | Specialist AIs | Handling diverse strategies |
| 12 | Self-play | General Catan mastery |

The agent advances to the next phase when it achieves a target average VP score, ensuring it's actually mastered each difficulty level before moving on.

### Reward Design

We use Potential-Based Reward Shaping (PBRS) — the agent gets small bonuses for improving its board position (better settlement spots, resource diversity, port access) on top of the main signals:

- **+10 per Victory Point gained** — Main learning signal
- **+100 for winning** — Must dominate all other rewards
- **-10 for losing** — Small to avoid overly defensive play
- **PBRS bonuses (±5 max)** — Guide early learning without distorting optimal play

The key lesson: **winning must dominate all other rewards**. Our first version had PBRS ±50 and win bonus +100, so the agent optimized board position instead of actually trying to win.

### Neural Network Architecture

Multi-head feedforward network with tile attention:

```
Observation (575 features)
    ├── Tile features → TileAttentionEncoder (attention over 19 hex tiles)
    └── Player/game context → Linear encoder
                    ↓
            Shared backbone (3 FC layers)
                    ↓
    ┌───────┬───────┬───────┬───────┬───────┐
  Action  Vertex  Edge   Trade   Trade  Value
  (14)    (54)    (72)   Give(5) Get(5)  (1)
```

We tried transformers for the full architecture but feedforward works better for Catan since each decision is basically independent — there's no sequence to attend to, unlike chess where move order matters.

### Agent Quality Score (AQS)

Win rate alone doesn't tell the full story. Our AQS system scores the agent across 6 dimensions:

1. **Outcome** (30%) — Win rate and VP margin
2. **Efficiency** (20%) — How quickly it wins
3. **Economy** (15%) — Resource management quality
4. **Tempo** (15%) — Turn utilization
5. **Dominance** (10%) — Board control metrics
6. **Adaptability** (10%) — Performance across opponent types

## Key Design Decisions

- **Why 1v1 instead of 4-player?** — 4x faster training, simpler strategy space, and the RL agent converges much more reliably. The game mechanics still apply.
- **Why feedforward over transformers?** — Catan decisions are mostly independent. Transformers add overhead without benefit since there's no meaningful sequence to attend to.
- **Why PPO instead of AlphaZero/MCTS?** — We tried AlphaZero but MCTS is too slow for Catan's large action space. PPO with curriculum learning trains faster and produces stronger agents.
- **Why curriculum learning?** — Essential. Without it, the agent can't learn anything — it never wins against a strong opponent, so it gets no reward signal and just learns to pass every turn.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pygame 2.5+
- gymnasium 0.29+
- numpy 1.24+

## Setup

```bash
pip install -r requirements.txt
python main.py
```

### Training

```bash
# Start curriculum training (the main training script)
python -m training.trainer --num-players 2

# Evaluate a trained model
python -m evaluation.evaluator --model models/your_model.pt --opponent all

# Watch the model play
python -m evaluation.replay_viewer --model models/your_model.pt
```
