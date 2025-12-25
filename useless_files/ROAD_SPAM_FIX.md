# Road Spam Exploitation Fix

**Date Applied:** 2025-12-11
**Commit:** b9a9370
**Branch:** claude/analyze-project-changes-01BtSo7R1NVaCC4hNZePihJe

---

## 🚨 Critical Issue Discovered

The 20k episode training run revealed a catastrophic exploitation pattern where the agent learned to spam the `place_road` action, causing performance to **crash from VP 2.73 → 2.19** (worse than untrained random agents at 2.48 VP).

### Evidence of Road Spam:

| Episode | Roads/game | VP | Cities/game | Natural endings |
|---------|-----------|-----|-------------|-----------------|
| 2000 | 45 | 2.73 | 0.3 | 100% |
| 8000 | **73** | 2.73 | 0.5 | 87% |
| 12000 | 50 | 2.67 | 0.4 | 93% |
| 14000 | **95** | 2.33 ⬇️ | 0.2 ⬇️ | 83% |
| 18000 | **147** | 2.33 | 0.1 ⬇️ | 70% |
| 20000 | **265** | 2.19 💥 | 0.2 | 47% |

**Note:** The board only has 72 edges total!

### Analyzer Output (Episode 20000):
```
Top actions: place_road(7911), trade_with_bank(583), roll_dice(259)
```

The agent was attempting **7,911 roads in 30 games** = 264 roads per game!

---

## 🔍 Root Cause Analysis

### 1. Building Rewards Too High

From `catan_env_pytorch.py:442` (old):
```python
building_reward = settlement_diff * 1.0 + city_diff * 2.0 + road_diff * 1.5
```

The agent discovered it could:
- Spam `place_road` actions
- Get **+1.5 reward per road** (even if invalid, due to action masking)
- Accumulate small positive rewards through exploitation

### 2. No Action Diversity Enforcement

The agent could repeat the same action indefinitely with no penalty, leading to:
- Monotonous action sequences
- Exploitation of reward structures
- Poor generalization

### 3. Pattern Matches Previous Trade Spam

Same pathology as the original baseline:
- **Original:** Trade spam (4,854 trades) → crashed at ep 1500-2000
- **After first fix:** Road spam (7,911 roads) → crashed at ep 14000-20000

We fixed the symptom (trade spam) but not the disease (reward exploitation).

---

## ✅ Fixes Applied

### Fix 1: Drastically Reduce Building Rewards

**Before:**
```python
building_reward = settlement_diff * 1.0 + city_diff * 2.0 + road_diff * 1.5
```

**After:**
```python
# DRASTICALLY reduced to prevent exploitation (was 1.0, 2.0, 1.5)
# VP change (3.0x) and PBRS are primary signals now
building_reward = settlement_diff * 0.05 + city_diff * 0.1 + road_diff * 0.02
```

**Reduction:**
- Settlements: 1.0 → 0.05 (95% reduction)
- Cities: 2.0 → 0.1 (95% reduction)
- Roads: 1.5 → 0.02 (98.7% reduction)

**Rationale:**
- VP changes (+3.0 per VP) are now the primary signal
- PBRS provides strategic guidance
- Building rewards just provide slight preference, not exploitable magnitude

### Fix 2: Add Action Repetition Penalty

**New code in `catan_env_pytorch.py`:**

```python
# Track last 5 actions
self.last_actions = []  # In __init__
self.action_history_size = 5

# In step():
self.last_actions.append(action_name)
if len(self.last_actions) > self.action_history_size:
    self.last_actions.pop(0)

# In _calculate_reward():
if len(self.last_actions) >= 2:
    recent_same_actions = sum(1 for a in self.last_actions[-5:] if a == action_name)
    if recent_same_actions >= 3:  # 3+ same actions in last 5
        repetition_penalty = -0.5 * (recent_same_actions - 2)  # -0.5, -1.0, -1.5...
        reward += repetition_penalty
```

**Penalty scale:**
- 3 same actions in last 5: -0.5 penalty
- 4 same actions in last 5: -1.0 penalty
- 5 same actions in last 5: -1.5 penalty

**Rationale:**
- Discourages monotonous action sequences
- Allows some repetition (2 instances is fine)
- Prevents exploitation of any single action
- Encourages diverse, strategic gameplay

---

## 📊 Expected Results

### Primary Goal: Stop Exploitation

| Metric | Old (Exploited) | Expected (Fixed) |
|--------|----------------|------------------|
| **Roads/game** | 265 | 10-15 (realistic) |
| **VP trajectory** | 2.73 → 2.19 (crash) | 2.5 → 2.8+ (stable) |
| **Cities/game** | 0.1 → 0.2 | 0.5-1.0 |
| **Natural endings** | 100% → 47% | 80%+ sustained |
| **Action diversity** | 1-2 action types | 5-7 action types |

### Secondary Goal: Reach Curriculum Stage 2

With exploitation fixed, the agent should:
- Reach 3.6 VP needed to advance curriculum
- Progress to Stage 2 (VP target: 5)
- Continue improving past 20k episodes

### Tertiary Goal: Match 5k Run Performance

The 5k run accidentally avoided exploitation (stopped early):
- Peak: 2.87 VP at episode 3000
- Final: 2.73 VP at episode 5000

With fixes, 20k run should:
- Match or exceed 2.87 VP peak
- Maintain or improve beyond 10k episodes
- Never crash below 2.5 VP

---

## 🧪 Testing Plan

### Test 1: Quick Validation (5k episodes)

```bash
python train_clean.py --episodes 5000 --model-name spam_fix_test --curriculum --batch-size 512 --epochs 10 --save-freq 1000
```

**Success criteria:**
- ✅ Roads/game stays < 20 throughout
- ✅ VP reaches 2.5-2.7 by episode 5000
- ✅ No single action exceeds 50% of total actions
- ✅ Natural endings > 80%

### Test 2: Extended Run (20k episodes)

```bash
python train_clean.py --episodes 20000 --model-name spam_fix_extended --curriculum --batch-size 1024 --epochs 20 --save-freq 2000
```

**Success criteria:**
- ✅ NO REGRESSION after episode 10000
- ✅ VP stays ≥ 2.4 throughout
- ✅ Roads/game stays < 25 throughout
- ✅ Cities/game increases to 0.5-1.0
- ✅ Potentially advances to curriculum stage 2

### Test 3: Analyze Checkpoints

```bash
python analyze_model_performance.py --model-pattern "models/spam_fix_extended*.pt" --eval-episodes 30 --vp-target 4 --max-checkpoints 10
```

**Success criteria:**
- ✅ No action spam (no action > 40% of total)
- ✅ Diverse action distribution
- ✅ Roads/game realistic (10-20)
- ✅ Cities/game improving over training

---

## 🔬 Technical Details

### Why This Fix Is Better

**Compared to alternative approaches:**

| Approach | Pros | Cons |
|----------|------|------|
| **Action masking only** | Simple | Doesn't prevent spam of valid actions |
| **Remove building rewards** | Can't exploit what doesn't exist | Removes useful learning signal |
| **Add action diversity bonus** | Encourages variety | Complex to tune, may not prevent spam |
| **Our approach (reduce + penalize)** | Keeps signal, prevents exploitation | Requires both fixes together |

### Why 95-98% Reduction?

We want building rewards to provide **slight preference**, not **strong signal**:

- VP change: **+3.0 per VP** (primary signal)
- PBRS: **±2-5 per turn** (strategic signal)
- Building: **+0.02 to +0.1** (slight preference)

With this hierarchy, the agent:
- Learns to maximize VP (main goal)
- Uses PBRS for strategic positioning
- Slightly prefers building when ambiguous

### Why Repetition Penalty at Threshold 3?

- **0-2 repetitions:** Normal gameplay (e.g., building 2 roads in sequence)
- **3+ repetitions:** Likely exploitation or stuck behavior
- **Penalty scale:** Gentle (-0.5) initially, escalating if continues

---

## 📚 Lessons Learned

### 1. Reward Shaping Is Hard

- Fixed trade spam → got road spam
- Any exploitable reward structure will be found
- Need multiple defenses (reduction + penalty)

### 2. Long Training Exposes Hidden Issues

- 5k run looked fine (2.73 VP)
- 20k run revealed exploitation (crashed to 2.19 VP)
- Always test beyond "looks good" phase

### 3. Incremental Rewards Are Dangerous

Small per-action rewards like:
- `+1.5` per road
- `+1.0` per settlement

Can be exploited through high-frequency spam. Better to:
- Reward **outcomes** (VP changes)
- Use **potential functions** (PBRS)
- Keep incremental rewards **very small**

### 4. Action Diversity Matters

RL agents naturally find the single most rewarding action and spam it. Need explicit mechanisms to encourage:
- Varied action sequences
- Multi-step strategies
- Balanced gameplay

---

## 🎯 Next Steps

1. **Run Test 1** (5k episodes) to validate fixes work
2. **If successful:** Run Test 2 (20k episodes) for comprehensive test
3. **If still issues:** Consider adding multi-action learning from MULTI_ACTION_ANALYSIS.md
4. **If successful:** Train to 50k-100k episodes for full curriculum progression

---

## 🔗 Related Documents

- **PROJECT_ANALYSIS.md** - Original analysis identifying reward function issues
- **FIXES_TO_APPLY.md** - First round of fixes (addressed trade spam)
- **FIXES_APPLIED.md** - Documentation of first fix application
- **MULTI_ACTION_ANALYSIS.md** - Advanced fixes for multi-action learning
- **PBRS_AND_LOOKAHEAD.md** - PBRS enhancement proposals

---

**This fix addresses the root cause of reward exploitation, not just symptoms.**
