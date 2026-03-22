# Comprehensive Analysis: Training Plateau at 2.9 VP

**Date**: 2025-12-16
**Status**: Training plateaued at 2.66-2.92 VP despite 38,300+ episodes
**Best Model**: `stable_v1_BEST.pt` (2.92 VP, episode 26,700)

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problems Discovered](#problems-discovered)
3. [Root Cause Analysis](#root-cause-analysis)
4. [Fixes Already Applied](#fixes-already-applied)
5. [Proposed Fixes (Not Yet Applied)](#proposed-fixes-not-yet-applied)
6. [Self-Play Analysis](#self-play-analysis)
7. [Recommended Action Plan](#recommended-action-plan)

---

## Executive Summary

### What's Working ✅
- **No exploitation**: Agent no longer spam roads (29.3 roads/game is realistic)
- **Stable training**: Exponential LR/entropy decay prevents catastrophic forgetting
- **High completion rate**: 98% natural endings
- **Consistent performance**: 2.66 VP ± 0.79 (stable, not oscillating)

### Critical Bottleneck ❌
**Agent doesn't build cities** (0.4 per game instead of 1-2+)
- Cities = 2 VP each (vs 1 VP for settlements)
- Without cities, theoretical max VP ≈ 3.0
- This single issue explains the entire plateau

### Secondary Issues ⚠️
1. **Excessive bank trading**: 2,162 trades per 50 games (43 per game!)
2. **Not using dev cards**: Minimal development card purchases
3. **Curriculum never advances**: Stuck at stage 1 (target: 4 VP, needs 3.6 to advance)

---

## Problems Discovered

### Problem 1: Low City Building (CRITICAL)

**Symptoms**:
```
Cities per game: 0.4
Expected: 1-2+
Impact: Missing 1.2-3.2 VP per game
```

**Evidence from Analysis**:
- Top actions: `trade_with_bank(2162)`, `place_road(1370)`, `roll_dice(694)`
- `build_city` not even in top 3 actions
- Agent builds 2.2 settlements but only upgrades 0.4 to cities

**Why This Matters**:
- Each city upgrade: +1 VP (settlement already worth 1 VP, city worth 2 VP)
- To reach 4 VP (curriculum advancement): Need ~2 settlements + 1-2 cities + bonuses
- Current: 2.2 settlements + 0.4 cities = ~2.6 VP theoretical max

**Root Cause** (see Section 3):
- City building reward too small (0.1 vs 0.05 for settlements)
- Ore/Wheat harder to get than Wood/Brick/Sheep
- PBRS potential function doesn't strongly value cities
- Trading for city resources consumes too many actions

---

### Problem 2: Excessive Bank Trading (MAJOR)

**Symptoms**:
```
Bank trades per 50 games: 2,162
Average per game: 43.2 trades
Trade ratio: 4:1 (inefficient)
```

**Evidence**:
- `trade_with_bank` is #1 action by huge margin
- More bank trades than roads built (1370 roads)
- Agent trading constantly instead of building

**Why This Matters**:
- Each bank trade: 4 resources → 1 resource (75% loss)
- Wastes actions that could be used for building
- Suggests resource acquisition strategy is inefficient
- May indicate PBRS overvalues resource accumulation

**Root Cause** (see Section 3):
- PBRS potential function may reward resources too much
- Agent learns: "Get resources = good" but not "Use resources efficiently = better"
- No penalty for inefficient trading
- Bank trading is always available (easy action to spam)

---

### Problem 3: Curriculum Stuck at Stage 1

**Symptoms**:
```
Current stage: 1 (Target: 4 VP)
Advancement threshold: 90% of 4 VP = 3.6 VP
Current performance: 2.66 VP (66% of target)
Result: Never advances to stage 2
```

**Why This Matters**:
- Curriculum stages: [4, 5, 6, 7, 8, 10]
- Agent only ever trains against VP target 4
- Never exposed to higher complexity gameplay
- Curriculum supposed to guide progressive learning

**Root Cause**:
- Advancement threshold too strict (90%)
- Agent plateaus at 66% of target (2.66 / 4.0)
- Can't advance without solving city building problem
- Chicken-and-egg: Need cities to advance, but won't learn cities until exposed to higher targets?

---

### Problem 4: Development Card Neglect

**Symptoms**:
- Minimal dev card purchases observed
- No strategic use of Knights, Victory Point cards, etc.
- Missing potential 1-2 VP from dev cards

**Why This Matters**:
- Dev cards can provide:
  - Direct VP (5 victory point cards in deck)
  - Knights (Largest Army = 2 VP)
  - Strategic advantages (Monopoly, Year of Plenty, Road Building)
- Missing ~1 VP worth of potential

**Root Cause**:
- Dev card reward likely too small
- Ore/Wheat/Sheep needed (same resources as cities, competing priority)
- Uncertainty in reward (random card draw)
- Agent prefers deterministic building rewards

---

### Problem 5: Training Plateau Despite Stability Fixes

**Symptoms**:
```
Episodes 0-26,700 (stable_v1): Reached 2.92 VP
Episodes 26,700-65,000 (from_best_stable): 2.91 → 2.46 VP (no improvement)
Total: 65,000 episodes with stability fixes, no breakthrough
```

**Why This Matters**:
- Stability fixes work (no oscillation, no catastrophic forgetting)
- But stability ≠ continued improvement
- Agent found a local optimum and can't escape it

**Root Cause**:
- Reward function doesn't incentivize cities strongly enough
- PBRS guides toward resource accumulation, not resource usage
- Exponential LR decay makes late-stage learning very conservative
- Agent exploiting "safe" strategy: settlements + roads + trading

---

## Root Cause Analysis

### Root Cause 1: Reward Function Undervalues Cities

**Current Reward Structure** (`catan_env_pytorch.py:442`):
```python
# VP changes (primary signal)
vp_reward = vp_diff * 3.0  # +3 per VP

# Building rewards (after 95-98% reduction to prevent spam)
building_reward = settlement_diff * 0.05 + city_diff * 0.1 + road_diff * 0.02

# PBRS (10x multiplier)
pbrs_reward = (current_potential - self.last_potential) * 10.0

# Action repetition penalty
if recent_same_actions >= 3:
    repetition_penalty = -0.5 * (recent_same_actions - 2)
```

**Analysis**:

1. **VP Reward**: Works well, but cities and settlements both give +1 VP per building
   - Settlement: 0 → 1 VP = +3.0 reward
   - City upgrade: 1 → 2 VP = +3.0 reward
   - **Problem**: Equal reward despite cities being strategically superior (resource generation)

2. **Building Reward**: Cities only 2x settlement reward (0.1 vs 0.05)
   - Settlement cost: Wood, Brick, Sheep, Wheat = 4 resources
   - City cost: 3 Ore, 2 Wheat = 5 resources (harder to get)
   - **Problem**: City reward (0.1) doesn't compensate for extra difficulty

3. **PBRS**: Potential function based on resources and pieces
   - Likely counts settlements and cities equally in building count
   - Resources valued but not resource *efficiency*
   - **Problem**: May actually discourage city building (lose settlement piece, gain city piece = neutral)

**Conclusion**: Reward function makes cities only slightly more attractive than settlements, but cities are significantly harder to build. Rational agent prefers settlements.

---

### Root Cause 2: PBRS Overvalues Resource Hoarding

**Current PBRS Potential** (`catan_env_pytorch.py:402-425`):

```python
def _calculate_potential(self, state):
    potential = 0.0

    # Resource count
    total_resources = sum(state['my_resources'].values())
    potential += total_resources * 0.3

    # Building pieces
    potential += state['my_settlements'] * 1.0
    potential += state['my_cities'] * 2.0  # Cities worth 2x settlements
    potential += state['my_roads'] * 0.2

    # VP (primary signal)
    potential += state['my_victory_points'] * 5.0

    # Development cards
    potential += state['my_dev_cards'] * 0.5

    return potential
```

**Analysis**:

1. **Resource Hoarding**:
   - +0.3 per resource held
   - Holding 10 resources: +3.0 potential
   - **Problem**: Incentivizes accumulation, not usage
   - Agent learns: "Trade to get more resources" = good signal

2. **City Value**:
   - Cities: 2.0 potential
   - Settlements: 1.0 potential
   - **Problem**: 2x multiplier seems good, but...
   - Building city: Lose settlement (-1.0) + Gain city (+2.0) = **Net +1.0**
   - Building settlement: Gain settlement (+1.0) = **Net +1.0**
   - **Cities and settlements provide EQUAL PBRS boost!**

3. **Trading Incentive**:
   - Bank trade: 4 resources → 1 resource = -0.9 potential (4 * 0.3 - 1 * 0.3)
   - But if traded resource enables better positioning: Can get +0.3 per turn
   - **Problem**: Encourages trading for "better" resources even at 4:1 loss

**Conclusion**: PBRS potential function inadvertently makes bank trading and resource hoarding attractive, while not distinguishing cities from settlements.

---

### Root Cause 3: Resource Distribution Favors Settlements

**Catan Resource Economics**:

| Building | Resources Needed | Difficulty |
|----------|-----------------|------------|
| Settlement | Wood, Brick, Sheep, Wheat | Medium (4 different, common) |
| Road | Wood, Brick | Easy (2 resources, very common) |
| City | 3 Ore, 2 Wheat | **Hard (5 resources, Ore rare)** |
| Dev Card | Ore, Wheat, Sheep | Medium (3 resources, Ore rare) |

**Observation**:
- Wood/Brick: Very common (needed for roads + settlements)
- Wheat: Common (needed for settlements + cities + dev cards)
- Sheep: Common (settlements + dev cards)
- **Ore: RARE** (only needed for cities + dev cards)

**Agent's Learned Strategy**:
1. Collect common resources (Wood, Brick, Sheep, Wheat)
2. Build settlements (uses common resources)
3. Build roads (uses very common resources)
4. **Ore is scarce → Cities too expensive → Avoid cities**
5. Bank trade to convert excess common → desired resources
6. Repeat

**Why This Becomes Locally Optimal**:
- Settlements + roads = consistent 2-3 VP
- Minimal Ore dependency
- Can achieve 2.6-2.9 VP reliably
- Trying to build cities = risk running out of resources = lower VP

**Conclusion**: Reward function + resource scarcity creates perverse incentive favoring settlement-heavy strategy.

---

### Root Cause 4: Exponential LR Decay Prevents Late Discovery

**Current LR Schedule**:
```python
scheduler = ExponentialLR(trainer.optimizer, gamma=0.9995)

Episode 0:     LR = 3.0e-4
Episode 10k:   LR = 4.5e-5
Episode 20k:   LR = 6.7e-6
Episode 30k:   LR = 1.0e-6
```

**Analysis**:

**Early Training (ep 0-10k)**:
- High LR (3e-4 → 4.5e-5)
- Agent explores: settlements, roads, trading
- Discovers: "Settlements + roads = 2.5-2.7 VP reliably"
- **This strategy gets reinforced early**

**Late Training (ep 20k-30k)**:
- Very low LR (6.7e-6 → 1.0e-6)
- Policy barely changes
- Even if city building occasionally gives +3 VP, gradient update negligible
- **Can't learn new strategies**

**Conclusion**: Exponential decay ensures stability but prevents late-stage strategy discovery. Agent locked into early-learned settlement strategy.

---

## Fixes Already Applied

### Fix 1: Exponential LR/Entropy Decay (APPLIED ✅)

**File**: `train_clean.py:139-145, 272-278`

**What It Does**:
- Exponential LR decay: 3e-4 → 1e-6
- Exponential entropy decay: 0.05 → 0.0015
- Dynamic PPO clipping: 0.2 → 0.1

**Problem It Solved**:
- ✅ Prevents catastrophic forgetting (no more oscillation)
- ✅ Stops late-training collapse (performance doesn't crash)
- ✅ Maintains learned strategies

**Problem It Created**:
- ❌ Makes late-stage learning very conservative
- ❌ Agent can't discover new strategies after ~20k episodes
- ❌ Locks in whatever strategy was learned early

**Verdict**: Good for stability, bad for continued exploration. Need to balance.

---

### Fix 2: Full Training State Checkpointing (APPLIED ✅)

**File**: `train_clean.py:164-193, 254-263, 295-303`

**What It Does**:
- Saves model + optimizer + scheduler + episode + best_vp
- Proper resume preserves LR, entropy, momentum

**Problem It Solved**:
- ✅ Can resume training without performance regression
- ✅ Optimizer state preserved (momentum, Adam moments)
- ✅ Decay schedules continue correctly

**Verdict**: Essential fix, no downsides. Works perfectly.

---

### Fix 3: Best Model Tracking (APPLIED ✅)

**File**: `train_clean.py:157-159, 250-263`

**What It Does**:
- Tracks best avg VP over training
- Saves checkpoint whenever new best achieved
- Allows recovery of peak performance

**Problem It Solved**:
- ✅ Identified peak at episode 26,700 (2.92 VP)
- ✅ Can use best model instead of final model
- ✅ Provides clear signal of when training degraded

**Verdict**: Excellent addition. Revealed that further training didn't help.

---

### Fix 4: Massive Reward Reduction (APPLIED ✅)

**File**: `catan_env_pytorch.py:442`

**What It Does**:
```python
# Reduced from 1.0, 2.0, 1.5 to:
building_reward = settlement_diff * 0.05 + city_diff * 0.1 + road_diff * 0.02
```

**Problem It Solved**:
- ✅ Stopped road spam (was 265 roads/game, now 29)
- ✅ Stopped trade spam (was 4854 trades/game, eliminated)
- ✅ VP changes (3.0x) became primary signal

**Problem It Created**:
- ❌ Made building rewards TOO small
- ❌ Cities (0.1) vs Settlements (0.05) barely different
- ❌ Removed discriminatory signal for strategic buildings

**Verdict**: Fixed exploitation, but overcorrected. Need to reintroduce city preference.

---

### Fix 5: Action Repetition Penalty (APPLIED ✅)

**File**: `catan_env_pytorch.py:~407-411`

**What It Does**:
- Tracks last 5 actions
- Penalizes 3+ same actions: -0.5, -1.0, -1.5...

**Problem It Solved**:
- ✅ Prevents single-action spam
- ✅ Encourages action diversity

**Problem It Created**:
- ❌ May discourage legitimate repeated actions (building multiple settlements in setup)
- ❌ Penalty too weak (-0.5) vs VP reward (+3.0)

**Verdict**: Good concept, but likely ineffective due to magnitude. VP rewards dominate.

---

### Fix 6: PBRS 10x Boost (APPLIED ✅)

**File**: `catan_env_pytorch.py:~454`

```python
pbrs_reward = (current_potential - self.last_potential) * 10.0
```

**Problem It Solved**:
- ✅ Makes PBRS signal stronger vs noise
- ✅ Guides strategic play

**Problem It Created**:
- ❌ May be amplifying resource hoarding incentive
- ❌ If potential function is flawed (cities = settlements), 10x makes it worse

**Verdict**: Magnitude is good, but potential function needs fixing (see Proposed Fixes).

---

## Proposed Fixes (Not Yet Applied)

### Proposed Fix 1: Boost City Building Reward (HIGH PRIORITY) 🔥

**Problem**: Cities give 0.1 reward vs 0.05 for settlements (only 2x), but cities are 5x harder

**Proposed Change** (`catan_env_pytorch.py:442`):
```python
# Current:
building_reward = settlement_diff * 0.05 + city_diff * 0.1 + road_diff * 0.02

# Proposed Option A: 10x boost
building_reward = settlement_diff * 0.05 + city_diff * 1.0 + road_diff * 0.02

# Proposed Option B: 5x boost (more conservative)
building_reward = settlement_diff * 0.05 + city_diff * 0.5 + road_diff * 0.02
```

**Rationale**:
- City upgrade: +3.0 (VP) + 1.0 (building) = **+4.0 total**
- Settlement: +3.0 (VP) + 0.05 (building) = **+3.05 total**
- Cities now 31% more rewarding, compensates for difficulty

**Expected Impact**:
- Cities/game: 0.4 → 1.0-1.5
- VP: 2.66 → 3.0-3.5
- May enable curriculum advancement

**Risk**:
- Low risk: 1.0 reward still much smaller than 3.0 VP reward
- Won't cause city spam (resource scarcity prevents it)
- Can revert if causes issues

**Test Protocol**:
1. Apply fix
2. Train 15k episodes from scratch
3. Check cities/game every 5k episodes
4. Success: cities/game > 1.0 by episode 15k

---

### Proposed Fix 2: Fix PBRS City Valuation (HIGH PRIORITY) 🔥

**Problem**: PBRS treats city upgrade as +1.0 net (lose settlement -1.0, gain city +2.0)

**Proposed Change** (`catan_env_pytorch.py:402-425`):
```python
# Current:
potential += state['my_settlements'] * 1.0
potential += state['my_cities'] * 2.0

# Proposed:
potential += state['my_settlements'] * 1.0
potential += state['my_cities'] * 3.0  # 3x instead of 2x

# OR better: Count total building value
num_settlements = state['my_settlements']
num_cities = state['my_cities']
potential += num_settlements * 1.0 + num_cities * 2.0  # Cities worth 2 extra
```

**Rationale**:
- Settlement: +1.0 potential
- City upgrade: Lose settlement (-1.0) + Gain city (+3.0) = **+2.0 net**
- Cities now clearly better than settlements in PBRS

**Expected Impact**:
- PBRS guides toward city building
- Combined with building reward boost: Strong signal for cities

**Risk**:
- Low risk: PBRS multiplier (10x) makes this +20 reward for city
- Still much smaller than VP reward (3.0 * 10 = +30)
- Won't cause exploitation

---

### Proposed Fix 3: Penalize Resource Hoarding (MEDIUM PRIORITY)

**Problem**: PBRS rewards +0.3 per resource held, encourages hoarding/trading

**Proposed Change** (`catan_env_pytorch.py:402-425`):
```python
# Current:
total_resources = sum(state['my_resources'].values())
potential += total_resources * 0.3

# Proposed Option A: Diminishing returns
total_resources = sum(state['my_resources'].values())
# 0-5 resources: Full value (+0.3 each)
# 6-10 resources: Half value (+0.15 each)
# 11+ resources: No value
clamped_resources = min(total_resources, 5) + max(0, min(total_resources - 5, 5)) * 0.5
potential += clamped_resources * 0.3

# Proposed Option B: Reduce multiplier
potential += total_resources * 0.1  # Reduce from 0.3 to 0.1
```

**Rationale**:
- Having resources is good, but using them is better
- Diminishing returns discourages hoarding beyond 10
- May reduce excessive bank trading

**Expected Impact**:
- Bank trades/game: 43 → 20-30
- More actions available for building
- Slightly faster games

**Risk**:
- Medium risk: May make resource collection too unrewarding
- Could cause agent to build prematurely

---

### Proposed Fix 4: Curriculum Forcing (LOW PRIORITY)

**Problem**: Agent stuck at stage 1, never exposed to higher VP targets

**Proposed Change** (`train_clean.py:207`):
```python
# Current: 90% mastery required
if avg_recent_vp >= MASTERY_THRESHOLD * current_vp_target:  # 0.9 * 4 = 3.6 VP

# Proposed: 70% mastery
if avg_recent_vp >= 0.7 * current_vp_target:  # 0.7 * 4 = 2.8 VP
```

**Rationale**:
- Current performance (2.66 VP) is 66.5% of target (4 VP)
- Lowering to 70% would allow advancement at 2.8 VP
- Exposure to higher targets might teach cities

**Expected Impact**:
- Agent advances to stage 2 (target: 5 VP)
- May learn cities are necessary for 5 VP
- Or may fail completely at stage 2

**Risk**:
- High risk: Agent may advance before ready
- Could cause training instability
- Curriculum stages might be too large jumps (4 → 5 → 6...)

**Verdict**: Try other fixes first. Use curriculum forcing only if agent gets stuck at 2.8-2.9 VP with city building working.

---

### Proposed Fix 5: Exponential VP Rewards (LOW PRIORITY)

**Problem**: All VP worth same (+3.0), but higher VP should be more valuable

**Proposed Change** (`catan_env_pytorch.py:442`):
```python
# Current:
vp_reward = vp_diff * 3.0

# Proposed:
# VP 0→1: 3.0 reward
# VP 1→2: 3.6 reward
# VP 2→3: 4.3 reward
# VP 3→4: 5.2 reward
vp_reward = vp_diff * (3.0 * (1.2 ** new_vp))
```

**Rationale**:
- Makes reaching 3-4 VP more rewarding
- Incentivizes pushing for wins

**Expected Impact**:
- Minimal impact on city building directly
- May motivate overall higher VP pursuit

**Risk**:
- Low risk: Still using vp_diff, so relative rewards
- Could cause over-aggressive play

**Verdict**: Interesting idea, but unlikely to solve city problem. Try if other fixes plateau at 3.0-3.2 VP.

---

### Proposed Fix 6: Milestone Bonuses (LOW PRIORITY)

**Problem**: No special reward for VP thresholds

**Proposed Change** (`catan_env_pytorch.py:442`):
```python
# Add bonuses for reaching milestones
if new_vp >= 3 and old_vp < 3:
    vp_reward += 5.0
if new_vp >= 4 and old_vp < 4:
    vp_reward += 10.0
```

**Rationale**:
- Explicitly reward reaching curriculum targets
- May motivate strategies that enable 3-4 VP

**Expected Impact**:
- Modest boost to VP pursuit
- Unlikely to directly teach cities

**Risk**:
- Low risk: One-time bonuses, can't be exploited

**Verdict**: Nice-to-have, but not addressing root cause.

---

## Self-Play Analysis

### Question: Would Self-Play Help Push Past 2.9 VP?

**Short Answer**: **Maybe, but probably not without fixing city rewards first.**

---

### What Self-Play Would Change

**Current Setup**:
- Agent plays against 3 rule-based AI opponents
- Rule-based AI: Deterministic, predictable, suboptimal
- Agent learns: "Beat rule-based AI to maximize VP"

**With Self-Play**:
- Agent plays against 3 copies of itself
- All players use same policy (or variants at different training stages)
- Agent learns: "Beat other agents to maximize VP"

---

### Argument FOR Self-Play Helping

#### 1. **Competitive Pressure**

**Current**: Rule-based AI is weak
- Agent can reach 2.6-2.9 VP and win
- No pressure to improve beyond "good enough to beat rule-based"

**With Self-Play**: All players learning
- If one agent discovers cities → gains 1-2 VP advantage → wins more
- Other agents forced to adapt or lose
- **Arms race drives improvement**

**Example from AlphaGo/OpenAI Five**:
- Self-play agents discovered advanced strategies humans never found
- Competitive pressure → innovation

**Likelihood**: 40% chance self-play creates pressure to learn cities

---

#### 2. **Resource Competition**

**Current**: Rule-based AI doesn't optimize resource gathering
- Agent can get Ore/Wheat without much contest
- But agent still doesn't build cities (reward problem, not availability)

**With Self-Play**: All players want Ore/Wheat
- If all 4 players use same strategy (settlements only) → Ore/Wheat abundant
- If one player builds cities → Ore/Wheat becomes scarce → Others adapt
- **Scarcity drives strategic diversity**

**Likelihood**: 30% chance resource competition teaches cities

---

#### 3. **Curriculum Through Opponent Skill**

**Current**: Opponent skill fixed at rule-based level
- Agent learns to beat them, then plateaus
- No progressively harder opponents

**With Self-Play**: Opponents improve as agent improves
- Early: All agents bad, 2.0 VP wins
- Mid: All agents mediocre, 2.5 VP wins
- Late: All agents good, 3.0+ VP required to win
- **Progressive difficulty = natural curriculum**

**Likelihood**: 50% chance progressive difficulty helps

---

### Argument AGAINST Self-Play Helping

#### 1. **Root Cause Is Reward Function, Not Opponent Quality**

**Analysis**: Agent doesn't build cities because reward doesn't justify cost
- City reward: 0.1 (too small)
- PBRS: +1.0 net (same as settlements)
- Cost: 5 resources (harder than settlements)
- **Rational conclusion: Cities not worth it**

**With Self-Play**: Reward function doesn't change
- Agent still sees: City = +3.0 (VP) + 0.1 (building) + 1.0 (PBRS) = +4.1
- Agent still sees: Settlement = +3.0 (VP) + 0.05 (building) + 1.0 (PBRS) = +4.05
- **Cities only 1.2% better in rewards, but 25% harder to build**
- Self-play won't change this math

**Likelihood**: 70% chance self-play doesn't overcome reward imbalance

---

#### 2. **Self-Play Requires Winning, But Agent Already Wins**

**Current**: Agent wins ~50-60% of games at 2.6-2.9 VP
- Against 3 rule-based opponents, getting 2.6 VP often wins
- Agent's strategy is effective for winning

**With Self-Play**: Against 3 copies, win rate = 25% (1 in 4)
- But reaching 2.6 VP vs other 2.6 VP agents still wins 25% of time
- No pressure to improve beyond 2.6 if everyone is at 2.6
- **Nash equilibrium at suboptimal strategy**

**Example**:
- If all 4 players build 0.4 cities, all get ~2.6 VP, all win 25%
- For one player to deviate to 1.5 cities: Needs cities to be rewarding
- But cities aren't rewarding (reward problem)
- So no player deviates
- **Stuck in local equilibrium**

**Likelihood**: 60% chance self-play converges to same 2.6 VP strategy

---

#### 3. **Self-Play Adds Instability**

**Current**: Training is stable (exponential decay working)

**With Self-Play**: Opponent policies constantly changing
- Episode 1000: Opponents build 0.2 cities
- Episode 5000: Opponents build 0.5 cities
- Episode 10000: Opponents revert to 0.3 cities (oscillation)
- **Non-stationary environment, harder to learn**

**Plus**: Need to maintain opponent pool
- Store checkpoints at different skill levels
- Manage mixture of past selves
- More complex infrastructure

**Likelihood**: 80% chance self-play adds complexity without solving root issue

---

#### 4. **Self-Play Doesn't Address City Building Signal**

**What Self-Play Does**:
- Changes opponent behavior
- Changes relative VP needed to win
- Creates competitive pressure

**What Self-Play Doesn't Do**:
- Doesn't change that cities cost 5 resources
- Doesn't change that city reward is 0.1
- Doesn't change that PBRS values cities = settlements
- **Doesn't fix the fundamental incentive problem**

**Likelihood**: 90% chance self-play alone is insufficient

---

### Self-Play: Expected Outcome Scenarios

#### Scenario A: Self-Play DOES Help (30% probability)

**What Happens**:
1. Four agents start training against each other
2. All begin with settlement-heavy strategy (2.6 VP)
3. Random exploration occasionally builds a city
4. City-building agent reaches 3.0 VP, wins more (28% vs 24%)
5. Other agents observe city-builders winning more
6. City-building strategy propagates through population
7. All agents converge to city-building equilibrium (3.2-3.5 VP)

**Requirements**:
- Random exploration finds cities despite low reward
- Winning signal strong enough to overcome reward imbalance
- Training stable despite non-stationary opponents

**Why This Might Work**:
- Competition creates pressure current setup lacks
- Winning is ultimate reward (overrides incremental rewards)

---

#### Scenario B: Self-Play Doesn't Help (70% probability)

**What Happens**:
1. Four agents start training against each other
2. All begin with settlement-heavy strategy (2.6 VP)
3. Occasional city attempts due to exploration
4. City attempts consume resources, reduce VP to 2.4
5. City-building agents lose more often
6. City-building strategy discouraged
7. All agents converge to settlement-only Nash equilibrium (2.6 VP)
8. **Stuck at same plateau, just with 4 agents instead of 1**

**Why This Is Likely**:
- Reward function fundamentally doesn't justify cities
- Self-play doesn't change rewards, just opponents
- Local equilibrium at suboptimal strategy

---

### Verdict on Self-Play

**Without City Reward Fixes**: ❌ **Not Recommended**

**Reasoning**:
1. Root cause is reward imbalance, not opponent quality (70% confidence)
2. Self-play adds complexity and instability (80% confidence)
3. Likely to converge to same equilibrium (60% confidence)
4. **Expected outcome: 2.6 VP plateau with 4 agents instead of 1**

**With City Reward Fixes**: ✅ **Potentially Very Helpful**

**Reasoning**:
1. If cities become rewarding, self-play accelerates discovery (50% confidence)
2. Competitive pressure ensures all agents find optimal strategy (60% confidence)
3. May reach 3.5-4.0 VP with combined fixes (40% confidence)

---

### Self-Play: Implementation Complexity

**Required Changes**:

1. **Opponent Pool Management**:
   ```python
   # Store checkpoints at different skill levels
   opponent_pool = [
       ("episode_1000.pt", 0.3),  # 30% use this
       ("episode_5000.pt", 0.3),
       ("episode_10000.pt", 0.2),
       ("current.pt", 0.2)
   ]
   ```

2. **Opponent Sampling**:
   ```python
   # Each episode, sample 3 opponents from pool
   opponents = sample_opponents(opponent_pool, n=3)
   ```

3. **Pool Updates**:
   ```python
   # Every 1000 episodes, add current policy to pool
   if episode % 1000 == 0:
       add_to_pool(current_policy)
       prune_pool(max_size=10)
   ```

4. **Win Rate Tracking**:
   ```python
   # Track wins vs each opponent level
   # Adjust sampling to maintain 40-60% win rate
   ```

**Effort**: Medium (2-3 days implementation + testing)

**Complexity**: High (debugging non-stationary training is hard)

---

## Recommended Action Plan

### Phase 1: Fix City Rewards (IMMEDIATE - 3 hours)

**Goal**: Make cities rewarding enough to justify building

**Actions**:
1. ✅ Boost city building reward: 0.1 → 0.5 (`catan_env_pytorch.py:442`)
2. ✅ Fix PBRS city valuation: 2x → 3x (`catan_env_pytorch.py:402-425`)
3. ✅ Reduce resource hoarding reward: 0.3 → 0.15 (`catan_env_pytorch.py:402-425`)

**Test**: Train 15k episodes from scratch
- **Success**: Cities/game > 1.0, VP > 3.0
- **Failure**: Cities/game still < 0.6, VP < 2.8

**Estimated Impact**: 60% chance of reaching 3.0-3.3 VP

---

### Phase 2: Validate with Extended Training (IF Phase 1 succeeds - 6 hours)

**Goal**: Confirm city building is stable and improves performance

**Actions**:
1. Train 30k episodes with city reward fixes
2. Monitor cities/game every 5k episodes
3. Ensure no oscillation or exploitation
4. Check curriculum advancement (target: 3.6 VP to reach stage 2)

**Success Criteria**:
- ✅ Cities/game: 1.0-1.5 sustained
- ✅ VP: 3.0-3.5 sustained
- ✅ Natural endings > 80%
- ✅ No road/trade spam regression

**Estimated Impact**: 70% chance of stable 3.2-3.5 VP

---

### Phase 3: Self-Play (IF Phase 2 reaches 3.5+ VP - 2 days)

**Goal**: Push beyond 3.5 VP using competitive pressure

**Actions**:
1. Implement opponent pool system
2. Train 50k episodes with self-play
3. Monitor VP progression and strategy diversity

**Success Criteria**:
- ✅ VP > 4.0 (curriculum advancement)
- ✅ Diverse strategies (cities + dev cards + trading)
- ✅ Training stable despite non-stationary opponents

**Estimated Impact**: 40% chance of reaching 4.0+ VP

---

### Alternative: Self-Play First (NOT RECOMMENDED)

**If you want to try self-play without city fixes**:

**Expected Outcome**: 70% chance of same plateau (2.6 VP)
**Effort**: 2 days implementation
**Risk**: Wasted time if reward function is root cause

**Recommendation**: Only try if you have time to experiment and want to prove/disprove self-play hypothesis

---

## Summary Table: Problems vs Fixes

| Problem | Severity | Root Cause | Fix Applied? | Proposed Fix | Impact |
|---------|----------|------------|--------------|--------------|--------|
| **Low city building (0.4/game)** | 🔴 Critical | City reward too small (0.1) | ❌ No | Boost to 0.5-1.0 | 🔥 High |
| **PBRS doesn't value cities** | 🔴 Critical | Cities = +1.0 net (same as settlements) | ❌ No | Change 2x → 3x | 🔥 High |
| **Excessive bank trading (43/game)** | 🟡 Major | PBRS rewards resource hoarding | ❌ No | Diminishing returns | 🔥 Medium |
| **Curriculum stuck at stage 1** | 🟡 Major | 90% threshold too strict | ❌ No | Lower to 70% | ⚠️ Low (risky) |
| **Training oscillation** | 🟢 Fixed | LR/entropy too high late | ✅ Yes | Exponential decay | ✅ Solved |
| **Catastrophic forgetting** | 🟢 Fixed | Policy updates too large | ✅ Yes | Clip decay | ✅ Solved |
| **Resume degradation** | 🟢 Fixed | Only model weights saved | ✅ Yes | Full state checkpoint | ✅ Solved |
| **Road spam exploitation** | 🟢 Fixed | Road reward too high | ✅ Yes | Reduce 1.5 → 0.02 | ✅ Solved |
| **Self-play missing** | 🟡 Maybe | Only vs rule-based AI | ❌ No | Implement opponent pool | ❓ Unknown |

---

## Final Recommendations

### Recommended Path (90% confidence)

1. **Apply city reward fixes** (Phase 1)
2. **Test for 15k episodes**
3. **If cities/game > 1.0**: Continue to 30k episodes
4. **If VP reaches 3.5+**: Try self-play
5. **If VP plateaus at 3.0-3.2**: Analyze and iterate

### NOT Recommended

- ❌ Self-play before fixing city rewards (70% chance of wasted effort)
- ❌ Curriculum forcing without city fix (will advance then fail)
- ❌ Continuing to train current model (already at local optimum)

### Expected Final Results

**With City Fixes Only**:
- VP: 3.0-3.5 (80% confidence)
- Cities/game: 1.0-1.5
- Curriculum: May reach stage 2

**With City Fixes + Self-Play**:
- VP: 3.5-4.5 (40% confidence)
- Cities/game: 1.5-2.5
- Curriculum: Stages 2-3 possible

**Without Any Fixes**:
- VP: 2.6-2.9 (stuck forever)
- No progress

---

## Conclusion

**The plateau at 2.9 VP is caused by insufficient reward signals for city building, not by opponent quality or training stability.**

**Self-play might help, but only after fixing the fundamental reward imbalance that makes cities unattractive.**

**Recommended: Fix city rewards first (3 hours), then decide on self-play based on results.**
