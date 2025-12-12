# Reward Function Evolution - Complete History

**Last Updated:** 2025-12-12
**Current Commit:** 92630a5

---

## 📜 Timeline of Fixes

### Version 1: Original (Baseline) - **EXPLOITABLE**
**Commits:** Pre-30d1ea6
**Result:** Trade spam (4,854 trades), crashed after 1500 episodes

| Component | Value | Problem |
|-----------|-------|---------|
| Building rewards | Settlement: 1.0, City: 2.0, Road: 1.5 | ❌ Too high - encourages spam |
| Illegal action penalty | -10.0 | ❌ Agent learned spam was "worth it" |
| VP reward | 8.0x | ❌ Too high - extreme variance |
| Opponent threat | Unbounded 5.0x | ❌ Could swing ±15 per turn |
| Hoarding penalty | VP>3, 7+ cards, 1.0x | ❌ Too harsh - prevented city building |
| Win bonus | 50.0 | ❌ Terminal reward spike |

**Evidence of Failure:**
```
Episode  500: VP 2.30, 91% trade spam (4,854 trades)
Episode 1500: VP 3.00 [PEAK]
Episode 2000: VP 2.30 [CRASHED - 23% regression]
```

---

### Version 2: First Fixes (FIXES_APPLIED.md) - **PARTIAL SUCCESS**
**Commit:** 30d1ea6 (2025-12-11)
**Result:** Reduced trade spam, but exposed road spam vulnerability

| Component | Old → New | Rationale |
|-----------|-----------|-----------|
| Building rewards | 1.0/2.0/1.5 → 0.05/0.1/0.02 | 95-98% reduction to prevent exploitation |
| Illegal action penalty | -10.0 → 0.0 | Remove perverse incentive |
| VP reward | 8.0x → 3.0x | Reduce variance by 62.5% |
| Opponent threat | Unbounded 5.0x → Capped 2.0x (max -10) | Prevent wild swings |
| Hoarding penalty | VP>3, 7+ cards → VP>5, 11+ cards | Allow resource accumulation for cities |
| Win bonus | 50.0 → 20.0 | Reduce terminal spike |

**Evidence of Partial Success:**
- Trade spam reduced from 90% → <10%
- VP improved from 2.3 → 2.7
- BUT: Still had building rewards that could be exploited

---

### Version 3: Road Spam Fix (ROAD_SPAM_FIX.md) - **STILL EXPLOITABLE**
**Commit:** b9a9370 (2025-12-11)
**Result:** Fixed trade spam, but road spam emerged in 20k training

**New Problem Discovered:**
```
Episode  2000: 45 roads/game, VP 2.73
Episode  8000: 73 roads/game, VP 2.73
Episode 14000: 95 roads/game, VP 2.33 ⬇️
Episode 20000: 265 roads/game, VP 2.19 💥
```
*(Board only has 72 edges!)*

**Attempted Fix:**
1. Further reduced building rewards: 0.05/0.1/0.02 → 0.02/0.05/0.01
2. Added action repetition penalty: -0.5 to -1.5 for 3+ same actions in last 5

**Why It Still Failed:**
- Even 0.02 reward × 265 roads = +5.3 from spam
- Action repetition penalty was a band-aid, not addressing root cause
- Agent found exploit: alternate between actions to avoid penalty

---

### Version 4: Remove Repetition Penalty (Interim) - **TESTING ECONOMIC DISINCENTIVE**
**Commit:** 4031f0f (2025-12-11)
**Rationale:** Action repetition penalty doesn't address root cause

**Theory:** If building roads costs resources (wood + brick), the agent should naturally learn:
- Spam roads → waste resources → can't build settlements/cities → lower VP → lose

**Problem:** Building rewards (even 0.02) still provide positive reinforcement for spam

---

### Version 5: OUTCOME-BASED LEARNING (Current) - **EXPLOITATION-PROOF**
**Commit:** 92630a5 (2025-12-12)
**Philosophy:** Remove ALL incremental rewards. Agent learns ONLY from outcomes and strategic quality.

#### Complete Reward Structure:

| Component | Value | Exploitable? | Purpose |
|-----------|-------|--------------|---------|
| **VP changes** | +3.0 per VP | ❌ NO | Main outcome - can't spam VP |
| **PBRS (10x boosted)** | ±5-15/turn | ❌ NO | Quality not quantity |
| **VP state bonus** | +0.1 per VP | ❌ NO | Steady progress |
| **Inaction penalty** | -3.0 | ❌ NO | Prevents passing when can build |
| **Discard penalty** | -2.0/card | ❌ NO | Encourages resource management |
| **Win bonus** | +20.0 | ❌ NO | Terminal state |
| **Loss penalty** | -1.0 | ❌ NO | Terminal state |
| **Building rewards** | 0.0 (REMOVED) | ✅ NO | Can't exploit what doesn't exist |
| **Illegal action penalty** | 0.0 (REMOVED) | ✅ NO | No perverse incentive |
| **Action repetition penalty** | 0.0 (REMOVED) | ✅ NO | Economic cost handles it |

#### Why This Should Work:

**1. No Incremental Rewards = No Spam Incentive**
```
Old: 265 roads × 0.02 reward = +5.3 from spam
New: 265 roads × 0.0 reward = 0.0 from spam
```

**2. PBRS Rewards Quality Over Quantity**
```
Old: Build 50 roads anywhere = +1.0 (0.02 × 50)
New: Build 50 roads in bad spots = +~0.5 PBRS (minimal strategic value)
     Build 5 roads strategically = +~3.0 PBRS (longest road pursuit)
```

**3. Economic Disincentive Built-In**
```
Road spam → waste wood/brick → can't afford settlements → VP stays low → lose
∴ Spam is self-penalizing through opportunity cost
```

**4. Clear Learning Path**
```
Episode 0-5k:    Learn VP is the goal (+3.0 per VP)
Episode 5k-10k:  Learn good positions (PBRS guides)
Episode 10k-15k: Learn resource efficiency (spam = lose)
Episode 15k-25k: Master strategic gameplay
```

---

## 🔬 Key Insights from Evolution

### 1. **Incremental Rewards Are Dangerous**

ANY reward that can be accumulated through high-frequency actions WILL be exploited:
- `1.0` per settlement → trade spam (accumulate resources, spam trades)
- `0.02` per road → road spam (even tiny reward × volume = exploitation)
- `-10.0` per illegal action → agent learns gamble is worth it

**Solution:** Remove all incremental rewards except those tied to irreversible outcomes (VP changes)

---

### 2. **Band-Aid Penalties Don't Work**

Attempted fixes that failed:
- ❌ Action repetition penalty → Agent alternates actions
- ❌ Hoarding penalty → Agent builds before hoarding
- ❌ Illegal action penalty → Agent learns spam is profitable anyway

**Solution:** Fix the incentive structure, not the symptoms

---

### 3. **PBRS is Powerful When Properly Scaled**

Original PBRS (0.1x multiplier):
- Production potential: ~5-20 → 0.5-2.0 reward
- Too weak compared to building rewards (1.0-1.5)
- Ignored by agent

New PBRS (1.0x multiplier, 10x boost):
- Production potential: ~5-20 → 5.0-20.0 reward
- Stronger than any other signal except VP
- Guides agent to quality strategies

**Key:** PBRS naturally rewards quality (good hex placement) not quantity (spam)

---

### 4. **Long Training Exposes Hidden Exploits**

| Training Length | What It Reveals |
|-----------------|-----------------|
| 2k episodes | Basic learning works |
| 5k episodes | May look successful (baseline: 2.73 VP) |
| 10k episodes | Exploitation starts to show |
| 20k episodes | Catastrophic collapse if exploitable (road spam: 2.19 VP) |
| 50k+ episodes | True test of stability |

**Lesson:** Always test beyond the "looks good" phase

---

### 5. **Reward Shaping Principles**

What we learned:

✅ **DO:**
- Reward irreversible outcomes (VP earned)
- Use potential-based rewards (PBRS) for strategy
- Let economic costs teach efficiency
- Keep reward signals clean and interpretable

❌ **DON'T:**
- Reward high-frequency actions (building, trading)
- Create perverse incentives (illegal action penalties)
- Use band-aid penalties (repetition penalties)
- Over-complicate the reward function

---

## 📊 Expected Performance Comparison

### Training to 25,000 episodes:

| Metric | v1 (Original) | v2 (First Fixes) | v3 (Road Spam) | v5 (Outcome-Based) Expected |
|--------|---------------|------------------|----------------|----------------------------|
| **Peak VP** | 3.0 @ ep 1.5k | 2.7-2.9 @ ep 5k | 2.73 → 2.19 | 2.8-3.0 @ ep 15-20k |
| **Final VP** | 2.3 (crashed) | Unknown | 2.19 (crashed) | **2.7-2.9 (stable)** |
| **Trade spam** | 91% | <10% | <10% | <5% |
| **Road spam** | Moderate | Unknown | 265/game 💥 | **10-20/game (realistic)** |
| **Cities/game** | 0.1 | 0.2-0.4 | 0.1-0.2 | **0.5-1.0** |
| **Natural endings** | 100% → 58% | ~70% | 100% → 47% | **70-80% sustained** |
| **Late-stage stability** | ❌ Crashed | Unknown | ❌ Crashed | **✅ Should remain stable** |

---

## 🎯 Success Criteria for v5 (Outcome-Based)

To validate that this reward structure works:

### Short-term (15k episodes):
- ✅ Roads/game < 25 throughout
- ✅ VP reaches 2.7+ by episode 15k
- ✅ No single action > 40% of total actions
- ✅ Natural endings > 70%

### Long-term (25k episodes):
- ✅ **NO late-stage collapse** (VP ≥ 2.7 at ep 25k)
- ✅ Cities/game improves to 0.5-1.0
- ✅ Performance maintains through episodes 20k-25k
- ✅ Diverse action distribution (no exploitation)

### Ultimate (50k+ episodes):
- ✅ Reaches curriculum stage 2 (VP target: 5)
- ✅ Continues improving past 25k episodes
- ✅ Demonstrates strategic mastery

---

## 🔮 If Outcome-Based Learning Fails

### Scenario 1: Agent Still Spams Actions

**Diagnosis:** 0.0 reward for spam is not enough disincentive
**Possible Fix:** Add small negative reward for wasteful actions (-0.01 per failed action)

---

### Scenario 2: Agent Doesn't Learn

**Diagnosis:** Signals too weak without incremental rewards
**Possible Fix:**
- Boost PBRS further (1.0 → 2.0)
- Increase VP reward (3.0 → 5.0)
- Add tiny building rewards (0.001) as breadcrumbs

---

### Scenario 3: Training Instability

**Diagnosis:** Network or optimizer issues
**Possible Fix:**
- Reduce learning rate
- Add more aggressive gradient clipping
- Improve value function normalization

---

### Scenario 4: Fundamental Approach Limitation

**Diagnosis:** Single-action RL may not be suitable for Catan
**Next Steps:** Consider architectural changes:
- Multi-action learning (predict sequences)
- Hierarchical RL (high-level strategy + low-level execution)
- Model-based RL (plan ahead)

---

## 📚 Related Documents

- **PROJECT_ANALYSIS.md** - Original analysis of training failures
- **FIXES_TO_APPLY.md** - First round of proposed fixes (v2)
- **FIXES_APPLIED.md** - Documentation of first fixes (v2)
- **ROAD_SPAM_FIX.md** - Analysis of road spam exploitation (v3)
- **OUTCOME_BASED_TESTING.md** - Testing protocol for v5
- **MULTI_ACTION_ANALYSIS.md** - Alternative approach if v5 fails
- **PBRS_AND_LOOKAHEAD.md** - Future enhancements

---

## 🚀 Current Status

**Code State:** ✅ Ready to test
**Commit:** 92630a5 - "Remove all exploitable reward sources - force outcome-based learning"
**Branch:** claude/analyze-project-changes-01BtSo7R1NVaCC4hNZePihJe

**Next Action:** Run Test 1 (15k episodes) or Test 2 (25k episodes) from OUTCOME_BASED_TESTING.md

**Expected Timeline:**
- Test 1 (15k): ~2-3 hours
- Test 2 (25k): ~5-6 hours
- Analysis: ~15-30 minutes

**If successful:** This becomes the foundation for all future training
**If fails:** We know we need fundamental architectural changes (multi-action, hierarchical RL)

---

**This is the culmination of iterative debugging and fixes. Outcome-based learning is our best shot at exploitation-proof RL training.**
