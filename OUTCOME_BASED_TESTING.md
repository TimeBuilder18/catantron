# Outcome-Based Learning - Testing Plan

**Date:** 2025-12-12
**Commit:** 92630a5
**Branch:** claude/analyze-project-changes-01BtSo7R1NVaCC4hNZePihJe

---

## 🎯 What Changed

### Complete Removal of Exploitable Rewards

This is the **most aggressive anti-exploitation fix** yet. The agent can ONLY learn from:
- **Outcomes** (VP changes)
- **Strategic quality** (PBRS potential)
- **Game states** (win/loss)

ALL incremental rewards that could be spammed have been removed.

---

## 📊 Current Reward Structure

### ✅ What the Agent GETS rewarded for:

| Component | Value | Exploitable? | Purpose |
|-----------|-------|--------------|---------|
| **VP changes** | +3.0 per VP | ❌ NO | Main outcome signal - can't spam VP |
| **PBRS (10x boost)** | ±5-15 per turn | ❌ NO | Rewards quality positions, not quantity |
| **VP state bonus** | +0.1 per current VP | ❌ NO | Steady progress signal |
| **Win bonus** | +20.0 | ❌ NO | Terminal state |
| **Loss penalty** | -1.0 | ❌ NO | Terminal state |
| **Inaction penalty** | -3.0 | ❌ NO | Prevents passing when can build |
| **Discard penalty** | -2.0 per card | ❌ NO | Discourages poor 7-roll management |

### ❌ What was REMOVED (previously exploitable):

| Component | Old Value | Why Removed |
|-----------|-----------|-------------|
| **Building rewards** | Settlement: 0.05<br>City: 0.1<br>Road: 0.02 | Even 0.02 × 300 spam = +6.0 exploitation |
| **Illegal action penalty** | -10.0 | Agent learned spam was "worth it" for occasional success |
| **Action repetition penalty** | -0.5 to -1.5 | Removed - economic cost handles this naturally |

---

## 🔍 Why This Approach Should Work

### Problem with Previous Approaches:

1. **Building rewards** (even tiny ones like 0.02) → Agent spammed roads/trades
2. **Illegal action penalty** (-10.0) → Agent thought spam was worth the occasional hit
3. **Action repetition penalty** → Band-aid solution, doesn't address root cause

### How Outcome-Based Learning Fixes This:

1. **No incremental rewards** → No value in spam
   - Spamming 300 roads = 0.0 reward (was +6.0)
   - Spamming invalid actions = 0.0 reward (was -10.0 gamble)

2. **PBRS rewards quality, not quantity**
   - Building on a 6/8/5 hex: +~10 PBRS (great position!)
   - Building on a 2/12/3 hex: +~2 PBRS (poor position)
   - Building 50 roads in bad spots: minimal PBRS gain
   - Building 5 roads in strategic spots: significant PBRS gain

3. **Economic disincentive built-in**
   - Roads cost resources (wood + brick)
   - Wasting resources → can't build settlements/cities → lower VP → negative outcome
   - Agent learns: "Spam prevents winning" through experience, not penalties

4. **VP is the only true goal**
   - +3.0 per VP is the strongest signal
   - Agent must learn strategic paths to VP
   - Can't game the system with spam

---

## 🧪 Testing Protocol

### Test 1: Quick Validation (2-3 hours)

```bash
python train_clean.py --episodes 15000 --model-name outcome_test_quick \
  --curriculum --batch-size 1024 --epochs 20 --update-freq 40 --save-freq 3000
```

**Success Criteria:**
- ✅ Roads/game < 25 throughout (realistic gameplay)
- ✅ VP trajectory: 2.3 → 2.7+ by episode 15000 (steady improvement)
- ✅ Natural endings > 70% (games complete properly)
- ✅ No single action exceeds 40% of total actions (diverse gameplay)

**Failure Indicators:**
- ❌ Roads/game > 50 (still spamming despite no reward)
- ❌ VP plateaus below 2.5 (not learning)
- ❌ Timeouts > 60% (agent stuck/confused)
- ❌ Any single action > 60% (found new exploit)

---

### Test 2: Extended Validation (5-6 hours)

```bash
python train_clean.py --episodes 25000 --model-name outcome_test_extended \
  --curriculum --batch-size 1024 --epochs 20 --update-freq 40 --save-freq 5000
```

**Success Criteria:**
- ✅ **No late-stage collapse** (VP doesn't drop after episode 18k)
- ✅ VP reaches 3.0+ by episode 20000 (approaching curriculum advancement)
- ✅ Cities/game improves: 0.1 → 0.5-1.0 (learning complex strategies)
- ✅ Roads/game stays realistic: 10-20 average
- ✅ Natural endings > 70% sustained

**Key Checkpoint Analysis:**

After training, run:
```bash
python analyze_model_performance.py --model-pattern "models/outcome_test_extended*.pt" \
  --eval-episodes 30 --vp-target 4 --max-checkpoints 6
```

Watch for:
1. **Episode 5000:** Learning basics (VP ~2.4-2.6)
2. **Episode 10000:** Building strategy emerges (VP ~2.6-2.8, cities appear)
3. **Episode 15000:** Peak performance (VP ~2.7-2.9)
4. **Episode 20000:** **CRITICAL - must maintain or improve** (VP ≥ 2.7)
5. **Episode 25000:** Final validation (VP ≥ 2.7, diverse actions)

---

## 📈 Expected Results vs Previous Runs

### Comparison to Previous Training Runs:

| Metric | Baseline (5k) | Road Spam (20k) | Expected (Outcome-Based) |
|--------|--------------|-----------------|-------------------------|
| **Peak VP** | 2.87 @ ep 3000 | 2.73 → 2.19 (crashed) | 2.8-3.0 @ ep 15-20k |
| **Final VP** | 2.73 | 2.19 (regressed) | 2.7-2.9 (stable) |
| **Roads/game** | 45 | 265 💥 | 10-20 (realistic) |
| **Cities/game** | 0.3 | 0.1-0.2 | 0.5-1.0 (improved) |
| **Natural endings** | 100% → 47% | 100% → 47% | 70-80% sustained |
| **Late-stage behavior** | Stable | Catastrophic collapse | **Should remain stable** |

### Why This Should Outperform Previous Approaches:

1. **No exploitation possible** → Stable learning throughout
2. **Quality over quantity** → Agent learns strategic gameplay
3. **Economic incentive alignment** → Spam naturally discouraged
4. **Cleaner signal** → VP and PBRS guide agent to real strategies

---

## 🚨 Warning Signs (What to Watch For)

### If the agent is STILL exploiting:

**Symptom:** Roads/game > 50 despite no building rewards

**Diagnosis:** Agent might be learning that:
- Invalid actions are "free" (0.0 cost)
- Spam actions = more chances for random success
- Time-wasting prolongs game for more PBRS rewards

**Fix:** May need to add time-based penalty or better terminal conditions

---

### If the agent isn't learning:

**Symptom:** VP plateaus at 2.3-2.4 for 10k+ episodes

**Diagnosis:**
- PBRS signal might be too weak (even at 10x)
- VP signal (+3.0) might not be enough
- Agent might need building rewards as "breadcrumbs"

**Fix:** Consider:
- Boosting PBRS further (1.0 → 2.0)
- Increasing VP reward (3.0 → 5.0)
- Adding VERY small building rewards (0.001) just for direction

---

### If training is unstable:

**Symptom:** Policy loss spikes, NaN values, gradient explosions

**Diagnosis:** Layer normalization might not be enough

**Fix:**
- Reduce learning rate (3e-4 → 1e-4)
- Increase gradient clipping (0.5 → 0.3)
- Add more aggressive value clipping

---

## 🎓 Learning Signals Analysis

### How the Agent Should Learn:

**Phase 1 (Episodes 0-5000): Fundamentals**
- Learn that building settlements → +VP → +3.0 reward
- Learn that PBRS increases with good positions
- Learn basic rules (when can build, what costs what)

**Phase 2 (Episodes 5000-10000): Strategy Emergence**
- Learn to prioritize high-production hexes (PBRS drives this)
- Learn to save resources for cities (PBRS for cities is 2x settlements)
- Learn that wasting resources → can't build → VP stagnates → bad outcome

**Phase 3 (Episodes 10000-15000): Optimization**
- Learn optimal settlement placements (PBRS optimization)
- Learn when to trade for city resources
- Learn efficient resource management

**Phase 4 (Episodes 15000-25000): Mastery**
- Consistent 2.7-2.9 VP performance
- Building 0.5-1.0 cities per game
- Strategic road placement for longest road (PBRS reward)
- **No regression** (this is the key test!)

---

## 📝 Results Documentation

### After running Test 1 (15k episodes):

Document in a new file `OUTCOME_TEST_RESULTS.md`:
1. Training curve (VP over episodes)
2. Final metrics (VP, roads/game, cities/game, timeouts%)
3. Top 5 actions from analyzer
4. Action distribution (% of each action type)
5. Comparison to baseline

### After running Test 2 (25k episodes):

Add to `OUTCOME_TEST_RESULTS.md`:
1. Late-stage performance (episodes 20-25k)
2. Checkpoint-by-checkpoint breakdown
3. Evidence of stability (no collapse)
4. Decision: ✅ Success or ❌ Needs more fixes

---

## 🎯 Success Definition

**Outcome-based learning is successful if:**

1. ✅ **No exploitation** - Roads/game < 25, no action spam
2. ✅ **Stable learning** - VP improves from 2.3 → 2.7-2.9 over 25k episodes
3. ✅ **No late-stage collapse** - Performance maintains or improves after episode 18k
4. ✅ **Strategic gameplay** - Cities appear (0.5-1.0/game), diverse actions
5. ✅ **Natural endings** - >70% games complete properly
6. ✅ **Curriculum readiness** - Approaching 3.6 VP needed to advance to stage 2

If all criteria are met: **This is the reward function foundation to build on**

If criteria are NOT met: **Need to reconsider approach** (see warning signs section)

---

## 🚀 Next Steps After Successful Testing

1. **Run longer training** (50k-100k episodes) to reach curriculum stage 2-3
2. **Add development card usage** (currently minimal)
3. **Improve trading strategy** (4:1 bank trades are expensive)
4. **Add opponent modeling** (PBRS could include opponent threat better)
5. **Consider multi-agent training** (current opponents are rule-based)

---

## 💾 Quick Commands Reference

```bash
# Test 1: Quick validation (2-3 hours)
python train_clean.py --episodes 15000 --model-name outcome_test_quick \
  --curriculum --batch-size 1024 --epochs 20 --update-freq 40 --save-freq 3000

# Test 2: Extended validation (5-6 hours)
python train_clean.py --episodes 25000 --model-name outcome_test_extended \
  --curriculum --batch-size 1024 --epochs 20 --update-freq 40 --save-freq 5000

# Analyze checkpoints
python analyze_model_performance.py --model-pattern "models/outcome_test_*" \
  --eval-episodes 30 --vp-target 4 --max-checkpoints 6

# Compare to baseline
python analyze_model_performance.py --model-pattern "models/catan_adaptive*.pt" \
  --eval-episodes 30 --vp-target 4 --max-checkpoints 3
```

---

**This is a critical test of the exploitation-proof reward design.**

If this works → We've solved the core learning stability problem
If this fails → We need fundamental architectural changes (e.g., multi-action learning, hierarchical RL)
