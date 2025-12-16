# Next Steps After Training Plateau

**Date**: 2025-12-16
**Status**: Training plateaued at 2.92 VP (stable_v1_BEST.pt)
**Training Completed**: 38,300 episodes from resumed checkpoint

---

## Current Situation

### Training History
1. **overnight_max** (26k episodes): Oscillated, collapsed to 2.35 VP
2. **stable_v1** (30k episodes): Stability fixes applied, peaked at 2.92 VP (ep 26,700)
3. **from_best_stable** (38k episodes): Resumed from stable_v1_BEST.pt, couldn't exceed 2.91 VP

### Best Model
- **File**: `models/stable_v1_BEST.pt`
- **Performance**: 2.92 VP (episode 26,700)
- **Status**: Near-optimal for current reward function

### Evidence of Plateau
- Resumed training never exceeded initial 2.91 VP
- Performance declining to 2.46 VP by episode 38,300
- Timeouts increasing (41.2%)
- Stability fixes prevented collapse but couldn't enable improvement

---

## Option 1: Use Current Model (RECOMMENDED)

### Why This Is Sufficient
- 2.92 VP is 73% of curriculum target (4 VP)
- Stable performance (no exploitation, no collapse)
- Represents peak of current training approach

### Next Actions
1. **Analyze behavior** to understand learned strategies:
   ```bash
   python analyze_model_performance.py --model models/stable_v1_BEST.pt --eval-episodes 50 --vp-target 4
   ```

2. **Test against rule-based AI** to measure true skill:
   ```bash
   python test_vs_rulebase.py --model models/stable_v1_BEST.pt --games 100
   ```

3. **Deploy for gameplay** if performance is acceptable

---

## Option 2: Break Through the Plateau

To exceed 2.92 VP requires fundamental changes to training approach:

### Change 1: Boost PBRS Even More (QUICK TEST)

**Current**: PBRS multiplier = 10.0x
**Try**: PBRS multiplier = 20.0x or 50.0x

**Rationale**: Agent may not be getting strong enough strategic signals to learn advanced play (cities, dev cards).

**Implementation**:
```python
# In catan_env_pytorch.py, line ~454
pbrs_reward = (current_potential - self.last_potential) * 50.0  # Increase from 10.0
```

**Test**: Train 15k episodes, see if cities/game increases

---

### Change 2: Modify Reward Function (MEDIUM EFFORT)

**Problem**: Current rewards may not incentivize VP > 3

**Solutions**:

#### 2a. Exponential VP Reward
```python
# Current: vp_reward = vp_diff * 3.0
# Proposed: Exponential growth for higher VP
vp_reward = vp_diff * (3.0 * (1.2 ** new_vp))  # VP 3→4 worth more than VP 0→1
```

#### 2b. Milestone Bonuses
```python
# Add bonus for reaching VP milestones
if new_vp >= 3 and old_vp < 3:
    vp_reward += 5.0  # Bonus for reaching VP 3
if new_vp >= 4 and old_vp < 4:
    vp_reward += 10.0  # Bigger bonus for VP 4
```

#### 2c. City/Dev Card Specific Bonuses
```python
# Current cities are barely used (0.3-0.6 per game)
# Add explicit bonus to overcome the learning difficulty
city_diff = new_cities - old_cities
dev_card_diff = new_dev_cards - old_dev_cards
strategic_bonus = city_diff * 0.5 + dev_card_diff * 0.3
```

---

### Change 3: Curriculum Forcing (QUICK TEST)

**Problem**: Agent not advancing past stage 1 (VP target 4)

**Solution**: Force advancement to expose agent to higher VP targets

```python
# In train_clean.py, modify curriculum logic
# Current: Advance when avg_vp >= 0.9 * target (90% mastery)
# Proposed: Advance when avg_vp >= 0.7 * target (70% mastery)
if avg_recent_vp >= 0.7 * current_vp_target:  # Lower threshold
```

**Rationale**: Agent may need exposure to higher VP targets to learn advanced strategies, even if not fully mastered.

---

### Change 4: Multi-Action Learning (HIGH EFFORT)

**Problem**: Single-action selection may not capture complex strategies

**Solution**: Implement hierarchical action space:
1. **High-level**: Strategy selection (expand, build cities, focus dev cards)
2. **Low-level**: Specific actions within strategy

**Effort**: Requires significant architectural changes (see MULTI_ACTION_ANALYSIS.md)

---

### Change 5: Longer Horizons with Opponent Modeling (HIGH EFFORT)

**Problem**: Agent plays against simple rule-based opponents

**Solution**:
- Self-play (agent vs copies of itself)
- More sophisticated opponents
- Longer training (100k-500k episodes)

**Effort**: Requires new training infrastructure

---

## Recommended Path Forward

### Conservative Approach (Use What Works)
1. Accept 2.92 VP as the best achievable with current setup
2. Analyze stable_v1_BEST.pt to document learned strategies
3. Use this model for gameplay/demonstration
4. Consider this phase 1 complete

### Aggressive Approach (Push Higher)
1. **Week 1**: Test PBRS boost (20x-50x) + curriculum forcing (70% threshold)
   - Train 15k episodes
   - Target: 3.0-3.2 VP, 1.0+ cities/game

2. **Week 2**: If no improvement, try exponential VP rewards + milestone bonuses
   - Train 20k episodes
   - Target: 3.2-3.5 VP

3. **Week 3**: If still plateaued, consider multi-action learning or self-play
   - Major architectural changes
   - Target: 3.5-4.0 VP

---

## Testing Protocol for Changes

When testing new approaches:

### Success Criteria
- ✅ VP > 3.0 sustained (beyond 2.92 plateau)
- ✅ Cities/game > 1.0 (currently 0.3-0.6)
- ✅ Dev cards used meaningfully
- ✅ Natural endings > 80%

### Failure Indicators
- ❌ Oscillation returns (road spam, trade spam)
- ❌ Performance worse than 2.92 VP baseline
- ❌ Timeouts > 30%
- ❌ No improvement after 15k episodes

### Always Compare to Baseline
- Run same evaluation on stable_v1_BEST.pt for comparison
- New approach must exceed 2.92 VP to be worth pursuing

---

## My Recommendation

**Start with Conservative**: Use stable_v1_BEST.pt as your production model. It's stable, exploit-free, and represents good performance.

**Then try Quick Tests**:
1. PBRS boost (20x or 50x)
2. Curriculum forcing (70% threshold)
3. Exponential VP rewards

Train each for 15k episodes (~3 hours). If any shows improvement beyond 2.92 VP, continue that direction. If none work after these tests, the plateau may be fundamental to the current architecture.

---

## Files to Modify for Quick Tests

### Test 1: PBRS Boost
**File**: `catan_env_pytorch.py`
**Line**: ~454
```python
pbrs_reward = (current_potential - self.last_potential) * 20.0  # Was 10.0
```

### Test 2: Curriculum Forcing
**File**: `train_clean.py`
**Line**: ~207
```python
if avg_recent_vp >= 0.7 * current_vp_target:  # Was 0.9
```

### Test 3: Exponential VP Rewards
**File**: `catan_env_pytorch.py`
**Line**: ~442
```python
vp_reward = vp_diff * (3.0 * (1.2 ** new_vp))  # Was vp_diff * 3.0
```

Run each test individually to isolate effects.

---

## Related Documents
- **STABILITY_FIXES.md** - Fixes that got us to stable 2.92 VP
- **OUTCOME_BASED_TESTING.md** - Current testing protocol
- **PBRS_AND_LOOKAHEAD.md** - Potential-based reward shaping details
- **MULTI_ACTION_ANALYSIS.md** - Advanced architectural changes

---

**The plateau at 2.92 VP is a natural stopping point. Pushing beyond requires either stronger learning signals (PBRS boost, exponential rewards) or architectural changes (multi-action, self-play).**
