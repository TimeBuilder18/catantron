# AlphaZero Curriculum Trainer - Improvements

## What's New?

The new `alphazero_trainer_curriculum.py` improves upon the base AlphaZero trainer with:

### 1. **Curriculum Learning** 🎓
Progressive difficulty ramp to help the AI learn faster:

- **Phase 0 (Games 0-200)**: 90% Random opponent
  - AI learns basic game mechanics
  - Easy wins build confidence

- **Phase 1 (Games 200-400)**: 75% Random opponent
  - Slightly harder
  - AI must start making better decisions

- **Phase 2 (Games 400-600)**: 50% Random/Smart mix
  - Balanced challenge
  - AI learns strategic thinking

- **Phase 3 (Games 600-800)**: 25% Random (mostly smart)
  - Difficult but beatable
  - AI refines tactics

- **Phase 4 (Games 800-1000)**: Rule-based AI
  - Competent opponent
  - AI must play well to win

- **Phase 5 (Games 1000+)**: Self-play
  - Both players use MCTS
  - AI plays against itself (AlphaZero style!)
  - Continuous improvement

### 2. **Increased MCTS Simulations** 🌲
- **Before**: 50 simulations per move
- **After**: 100 simulations per move (default)
- **Result**: Smarter decisions, better training data

### 3. **True Self-Play** 🪞
- After 1000 games, opponents also use MCTS
- Both players improve together
- Classic AlphaZero approach

### 4. **Better Metrics** 📊
- Tracks win rate by phase
- Shows current opponent type
- Phase-specific statistics
- Recent win rate (last 100 games)

### 5. **Command-Line Interface** 💻
Easy to configure without editing code!

## How to Use

### Basic Usage (Google Colab)

```python
from alphazero_trainer_curriculum import AlphaZeroCurriculumTrainer

# Create trainer
trainer = AlphaZeroCurriculumTrainer(
    num_simulations=100,  # MCTS simulations
    learning_rate=1e-3
)

# Train!
trainer.train(
    num_games=2000,
    save_frequency=100,
    save_path='models/my_alphazero'
)
```

### Command-Line Usage

```bash
# Default training (2000 games, 100 simulations)
python alphazero_trainer_curriculum.py

# Custom configuration
python alphazero_trainer_curriculum.py \
    --num-games 5000 \
    --simulations 150 \
    --lr 0.001 \
    --save-freq 200

# Load existing model and continue training
python alphazero_trainer_curriculum.py \
    --model-path models/alphazero_curriculum_game_1000.pt \
    --num-games 3000
```

### Available Arguments

- `--num-games`: Total games to play (default: 2000)
- `--simulations`: MCTS simulations per move (default: 100)
- `--batch-size`: Training batch size (default: auto-detect based on GPU)
- `--lr`: Learning rate (default: 0.001)
- `--save-freq`: Save checkpoint every N games (default: 100)
- `--model-path`: Path to load existing model (default: None)
- `--save-path`: Where to save models (default: 'models/alphazero_curriculum')

## Key Differences from Base AlphaZero

| Feature | Base AlphaZero | Curriculum AlphaZero |
|---------|---------------|---------------------|
| Opponent | Rule-based AI only | Progressive (random → smart → self-play) |
| MCTS Simulations | 50 | 100 (configurable) |
| Self-Play | No | Yes (after 1000 games) |
| Learning Curve | Steep | Gradual |
| Win Rate Tracking | Overall only | By phase + recent |
| CLI Arguments | No | Yes |

## Expected Performance

### Win Rate Progression

- **Phase 0-1** (Random opponents): 70-90% win rate
  - AI learns basics quickly

- **Phase 2-3** (Mixed opponents): 50-70% win rate
  - More challenging, AI refines strategy

- **Phase 4** (Rule-based): 30-50% win rate
  - Tough opponent, AI must be smart

- **Phase 5+** (Self-play): ~50% win rate
  - Both players equal strength
  - Continuous improvement through self-play

### Training Time

On A100 (40GB):
- Batch size: 8192 (optimized for high GPU utilization)
- Training steps: 50 per batch
- ~300-500 games/hour with 100 MCTS simulations
- 2000 games ≈ 4-7 hours

On RTX 2080 Super:
- Batch size: 3072-4096
- ~100-150 games/hour with 100 MCTS simulations
- 2000 games ≈ 13-20 hours

On CPU:
- ~10-20 games/hour
- 2000 games ≈ 100-200 hours (not recommended!)

## Tips for Best Results

1. **Start Fresh**: Don't load old models trained differently
2. **Use GPU**: CUDA dramatically speeds up training
3. **Be Patient**: Curriculum training takes time but learns better
4. **Monitor Phases**: Watch win rates drop at phase transitions (expected!)
5. **Self-Play is Key**: Phases 0-4 are just preparation for real learning in Phase 5

## Testing

Run quick test:
```bash
python test_alphazero_curriculum.py
```

This verifies:
- Curriculum opponents work
- Training step works
- No crashes

## Why This is Better

### Problem with Original Trainer:
- Jumped straight to hard opponents
- AI struggled to learn basics
- High variance in win rate
- Never implemented true self-play

### Solution with Curriculum:
- ✅ Learns fundamentals first (easy opponents)
- ✅ Gradually increases difficulty
- ✅ Reaches self-play for advanced learning
- ✅ More stable training
- ✅ Better final performance

## Example Output

```
======================================================================
ALPHAZERO CURRICULUM TRAINING
======================================================================
Device: cuda
Total games: 2000
MCTS simulations: 100
Batch size: 1024

Curriculum Phases:
  Phase 0 (0-200):     90% Random
  Phase 1 (200-400):   75% Random
  Phase 2 (400-600):   50% Random
  Phase 3 (600-800):   25% Random
  Phase 4 (800-1000):  Rule-based AI
  Phase 5 (1000+):     Self-play
======================================================================

Game    5/2000 | Phase: 0 (Random (90%)) | WR:  80.0% (recent:  80.0%, phase:  80.0%) | Buffer:    45 | Speed: 2.1 g/min
Game   10/2000 | Phase: 0 (Random (90%)) | WR:  75.0% (recent:  75.0%, phase:  75.0%) | Buffer:    92 | Speed: 2.3 g/min
  └─ Training: policy=2.1234, value=0.4567 (1.2s)
...
```

## Next Steps

1. Run training: `python alphazero_trainer_curriculum.py`
2. Monitor win rates by phase
3. After training, evaluate against rule-based AI
4. Compare to PPO trainer performance

---

**Remember**: AlphaZero learns through self-play. The curriculum just gets it to self-play faster! 🚀
