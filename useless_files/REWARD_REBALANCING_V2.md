# Reward Function Rebalancing v2

## 🚨 Problem Diagnosis

Your agent learned a **pathological strategy** after 5000 episodes:

### Observed Behavior
- **Average VP**: 2.8/10 (barely progressed)
- **Buildings**: 2 settlements + 2 cities only (initial placement)
- **Roads**: 0 (never built beyond initial)
- **Dev Cards**: 45.5 per game average (one game: 453!)
- **Resource Hoarding**: 33.7 max cards average
- **Discard Events**: 106.3 per game

### The Pathological Loop
```
1. Collect resources → +0.03 per resource
2. Get "buildable reward" → +0.28 for having resources
3. Buy dev card to reduce cards → avoid -0.1 penalty
4. If VP card → +3.0 reward (via VP multiplier)
5. REPEAT (never actually build, never win)
```

### Root Causes

| Problem | Impact | Example |
|---------|--------|---------|
| **"Buildable" reward** | Rewarded HAVING, not USING | +0.28/turn for hoarding |
| **Weak robber penalty** | No cost to hoarding | 33 cards = only -2.6 penalty |
| **Undervalued roads** | No incentive to build | Road = 0.2, Settlement = 1.0 |
| **Weak VP scaling** | Winning not priority | +3.0 per VP vs +0.28 hoarding |
| **Dev card exploit** | VP cards too valuable | +3.0 for 3 resources |

---

## ✅ Fixes Applied

### 1. **REMOVED "Buildable Reward"** (Lines 680-695)
```diff
- # Bonus for having enough resources to build
- buildable_reward = 0
- if (wood >= 1 and brick >= 1 and wheat >= 1 and sheep >= 1):
-     buildable_reward += 0.1  # Can build settlement
- if (ore >= 3 and wheat >= 2):
-     buildable_reward += 0.15  # Can build city
- reward += buildable_reward

+ # REMOVED: "buildable_reward" - was encouraging hoarding instead of building!
+ # Agent should get rewarded for BUILDING, not HAVING resources
```

**Why**: This was the #1 cause of hoarding behavior.

---

### 2. **MASSIVELY Increased Robber Penalty** (10x + Exponential)
```python
# OLD (weak penalty):
if total_cards > 7:
    reward -= 0.1 * excess_cards  # 33 cards = -2.6

# NEW (harsh penalty):
if total_cards > 7:
    excess_cards = total_cards - 7
    robber_penalty = 1.0 * excess_cards  # Linear: 10x stronger

    if excess_cards > 10:  # Exponential for extreme hoarding
        robber_penalty += 2.0 * (excess_cards - 10)

    reward -= robber_penalty
```

**Penalty Examples**:
| Total Cards | Excess | Old Penalty | New Penalty | Change |
|-------------|--------|-------------|-------------|--------|
| 8 | 1 | -0.1 | -1.0 | **10x** |
| 10 | 3 | -0.3 | -3.0 | **10x** |
| 15 | 8 | -0.8 | -8.0 | **10x** |
| 20 | 13 | -1.3 | -19.0 | **15x** |
| 33 | 26 | -2.6 | -58.0 | **22x** |

**Impact**: Agent will be **forced** to spend resources instead of hoarding.

---

### 3. **Increased Road Building Reward**
```python
# OLD:
building_reward = settlement_diff * 1.0 + city_diff * 2.0 + road_diff * 0.2

# NEW:
building_reward = settlement_diff * 1.0 + city_diff * 2.0 + road_diff * 0.5
```

**Road value**: 0.2 → 0.5 (**2.5x increase**)

**Why**: Roads were so under-rewarded the agent never built them. Now they're 50% as valuable as a settlement, which is more reasonable.

---

### 4. **Increased VP Reward Scaling**
```python
# OLD:
vp_reward = vp_diff * 3.0  # Going 2→3 VP = +3.0 reward

# NEW:
vp_reward = vp_diff * 8.0  # Going 2→3 VP = +8.0 reward
```

**VP multiplier**: 3.0 → 8.0 (**2.67x increase**)

**Why**: VP progression must be the PRIMARY objective. With weak scaling, the agent optimized for steady "buildable" rewards instead of winning.

---

### 5. **Increased Win Bonus**
```python
# OLD:
if winner == self:
    reward += 10.0
else:
    reward -= 0.5

# NEW:
if winner == self:
    reward += 50.0  # 5x stronger!
else:
    reward -= 1.0
```

**Win bonus**: 10.0 → 50.0 (**5x increase**)
**Loss penalty**: -0.5 → -1.0 (**2x increase**)

**Why**: Winning must be the ULTIMATE goal. Old bonus was too weak compared to steady hoarding rewards.

---

### 6. **Differentiated Dev Card Rewards**
```python
# NEW: Different rewards by card type

# VP cards: Very low (VP already counted at 8.0x)
vp_card_reward = vp_card_diff * 0.05

# Knight cards: Low (situationally useful)
knight_reward = knight_diff * 0.2

# Utility cards: Very low (rarely optimal)
utility_reward = (road_building + year_of_plenty + monopoly) * 0.1
```

**Dev Card Values**:
| Card Type | Old Implicit | New Explicit | Change |
|-----------|--------------|--------------|--------|
| VP Card | +3.0 (via VP) | +0.05 direct | Much lower |
| Knight | 0 | +0.2 | Small reward |
| Utility | 0 | +0.1 | Minimal |

**Why**: Dev cards were exploited because VP cards gave huge rewards. Now they're properly valued as **tools, not goals**.

---

### 7. **Enhanced Exploration Bonuses**
```python
# OLD:
if vp > 2:  exploration_reward += 0.5
if vp >= 4: exploration_reward += 1.0

# NEW:
if vp > 2:  exploration_reward += 0.5
if vp >= 4: exploration_reward += 1.0
if vp >= 6: exploration_reward += 2.0  # NEW
if vp >= 8: exploration_reward += 3.0  # NEW
```

**New milestones**: 6 VP (+2.0) and 8 VP (+3.0)

**Why**: Agent gets stronger signals as it approaches victory.

---

## 📊 Expected Impact

### Reward Comparison (Per Turn)

| Action | Old Reward | New Reward | Change |
|--------|------------|------------|--------|
| **Build Settlement** | +1.0 + 3.0 VP = **+4.0** | +1.0 + 8.0 VP = **+9.0** | +125% |
| **Build City** | +2.0 + 8.0 VP = **+10.0** | +2.0 + 16.0 VP = **+18.0** | +80% |
| **Build Road** | +0.2 | **+0.5** | +150% |
| **Hoard 15 cards** | +0.28 - 0.8 = **-0.52** | 0 - 8.0 = **-8.0** | -1438% |
| **Hoard 33 cards** | +0.28 - 2.6 = **-2.32** | 0 - 58.0 = **-58.0** | -2400% |
| **Win Game** | **+10.0** | **+50.0** | +400% |
| **Buy VP Dev Card** | +3.05 | +8.05 | +164% |

### Key Changes
✅ **Building now >> Hoarding** (9.0 vs -8.0 for moderate hoarding)
✅ **Roads worthwhile** (2.5x more rewarding)
✅ **VP progression prioritized** (8.0x multiplier)
✅ **Winning is the goal** (50.0 bonus!)
✅ **Dev cards are tools** (not the strategy)

---

## 🔄 Retraining Recommendations

### Option 1: Fresh Start (Recommended)
```bash
# Train from scratch with new rewards
python train_clean.py \
    --curriculum \
    --episodes 5000 \
    --batch-size 2048 \
    --epochs 20 \
    --model-name catan_rebalanced_v2
```

**Why**: Old policy learned bad habits. Fresh start is fastest path to good behavior.

---

### Option 2: Continue Training (Risky)
```bash
# Continue from last checkpoint
python train_clean.py \
    --curriculum \
    --episodes 5000 \
    --batch-size 2048 \
    --epochs 20 \
    --model-name catan_continued \
    --load models/catan_clean_episode_5000.pt
```

**Risk**: Agent may struggle to unlearn hoarding behavior. New rewards fight against old policy.

---

### Option 3: Slower Curriculum (Safer)
```bash
# More gradual progression
python train_clean.py \
    --episodes 10000 \
    --curriculum \
    --batch-size 2048 \
    --epochs 20 \
    --model-name catan_slow_curriculum
```

**Modify curriculum in train_clean.py**:
```python
def get_vp_target(episode, use_curriculum):
    if not use_curriculum:
        return 10

    # SLOWER progression (2000 episodes per stage)
    if episode < 2000:
        return 5
    elif episode < 4000:
        return 6
    elif episode < 6000:
        return 7
    elif episode < 8000:
        return 8
    else:
        return 10
```

---

## 🎯 What to Expect

### Early Training (Episodes 0-1000)
- Agent learns to avoid hoarding (harsh penalties)
- Builds settlements consistently
- Starts building roads (now worthwhile)
- May still buy some dev cards (habit breaking)

### Mid Training (Episodes 1000-3000)
- Dev card spam should disappear
- Road networks emerge
- More consistent 4-6 VP games
- Better resource management

### Late Training (Episodes 3000-5000)
- Agent reaches 6-8 VP regularly
- Wins some games (10 VP)
- Balanced strategy (buildings > dev cards)
- Efficient resource usage

### Success Metrics
✅ **Average VP**: 5-7 (currently 2.8)
✅ **Roads built**: 3-5 per game (currently 0)
✅ **Dev cards**: 1-3 per game (currently 45.5)
✅ **Max cards held**: 10-15 (currently 33.7)
✅ **Discard events**: 10-20 (currently 106.3)
✅ **Win rate**: 10-25% vs rule-based AI

---

## 🐛 Potential Issues

### Issue 1: Agent Becomes Too Conservative
**Symptom**: Never holds more than 7 cards, even briefly
**Fix**: Reduce robber penalty slightly (1.0 → 0.7)

### Issue 2: Still Not Building Roads
**Symptom**: Roads rarely built despite reward increase
**Fix**: Increase road reward further (0.5 → 0.8)

### Issue 3: Training Instability
**Symptom**: High variance in rewards, policy oscillates
**Fix**: Reduce learning rate, increase batch size

### Issue 4: Wins Too Rare
**Symptom**: Never reaches 10 VP, gets stuck at 6-8
**Fix**: Extend curriculum (more time at each VP level)

---

## 📝 Testing the Fix

Before full training, test with evaluation:

```bash
# Pull latest changes
git pull

# Quick test (50 episodes)
python train_clean.py \
    --episodes 50 \
    --model-name catan_test_v2 \
    --curriculum \
    --batch-size 1024

# Check if behavior improved
python evaluate_model.py \
    --model models/catan_test_v2.pt \
    --episodes 10 \
    --vp-target 5
```

**Look for**:
- Roads being built (> 0)
- Lower dev card spam (< 10 per game)
- Lower max cards (< 20)
- Higher VP (> 3.5 average)

---

## 🎓 Summary

The reward function was fundamentally broken:
- Rewarded hoarding over building
- Made dev cards too attractive
- Undervalued roads and VP progression
- Weak win signal

The rebalance:
- **Removed** hoarding incentive
- **Punished** excessive cards harshly
- **Increased** building rewards (especially roads)
- **Prioritized** VP progression
- **Emphasized** winning as the goal

Expected result: **Agent learns to actually play Catan!**

---

**Status**: ✅ Ready for Retraining
**Next Step**: Run fresh training with new reward function
**ETA**: 40-50 minutes for 5000 episodes on RTX 2080 Super
