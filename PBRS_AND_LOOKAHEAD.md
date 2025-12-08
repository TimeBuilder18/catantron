# PBRS and Look-Ahead Analysis

## Current State

### ✅ **You ALREADY Have PBRS Implemented!**

**Location:** `catan_env_pytorch.py:294-396`

```python
def _calculate_potential(self, player: Player):
    potential = 0.0

    # Production Potential (pips)
    # Strategic Assets (longest road, largest army, VP cards)
    # Opponent Threat (penalty for opponents close to winning)

    return potential

def _calculate_reward(self, old_obs, new_obs, step_info, old_potential, new_potential, debug=False):
    # PBRS Reward
    pbrs_reward = self.gamma * new_potential - old_potential  # Line 396
    reward += pbrs_reward
```

**This is correct PBRS implementation!** (Potential-Based Reward Shaping)

### ⚠️ **But There Are Issues With It**

#### Problem 1: Opponent Threat Penalty Too Aggressive (Line 385)

```python
# Current:
if opp_vp >= 8:
    potential -= (opp_vp - 7) * 5.0  # Can swing by ±15 per turn!
```

This creates **massive variance** in rewards:
- Opponent goes from 7 → 9 VP: Your potential drops by -10.0
- This variance destabilizes learning
- Agent learns to focus on defense instead of winning

**Fix (already in FIXES_TO_APPLY.md):**
```python
# Cap the penalty
if opp_vp >= 8:
    penalty = min((opp_vp - 7) * 2.0, 10.0)  # Max penalty -10
    potential -= penalty
```

#### Problem 2: Production Potential Might Be Too Small

```python
potential += production_potential * 0.1  # Line 373
```

With typical production potential of 20-40 pips:
- Production contributes: 2.0-4.0 to potential
- Opponent threat can swing: ±10.0
- **Opponent penalty dominates the signal**

**Consider increasing:**
```python
potential += production_potential * 0.3  # Increased from 0.1
```

---

## 🤔 **Do You Need Look-Ahead?**

Short answer: **Not yet, but it could help later.**

### What Is Look-Ahead?

**Look-ahead** means the agent simulates future states before choosing an action:

1. **Monte Carlo Tree Search (MCTS)** - Like AlphaGo
   - Simulate multiple games from current state
   - Pick action that leads to best average outcome
   - Very powerful but SLOW (100-1000x slower training)

2. **N-Step Returns** - Simple look-ahead
   - Instead of immediate reward, use sum of next N rewards
   - Already partially addressed by GAE in PPO (λ=0.95)

3. **Model-Based Planning**
   - Train a world model to predict next states
   - Use model to plan ahead without real environment
   - Complex to implement, requires separate model training

### Current PPO Already Does Some Look-Ahead

**Your trainer (trainer_gpu.py:33-55) uses GAE:**

```python
self.gae_lambda = 0.95  # Line 14

def compute_advantages(self, rewards, values, dones):
    # GAE looks ahead by bootstrapping from value function
    td_error = rewards[t] + self.gamma * next_value - values[t]
    advantage = td_error + self.gamma * self.gae_lambda * advantage
```

This effectively does **look-ahead through value function**:
- Value function predicts future returns
- GAE uses this to compute advantages
- Agent learns from multi-step trajectories

### When Would You Need More Look-Ahead?

**You would benefit from explicit look-ahead IF:**

1. ✅ **Agent reaches 7-8 VP consistently** (close to winning)
   - Strategic planning becomes critical
   - "Should I buy dev card or build city?" requires multi-turn thinking

2. ✅ **You switch to self-play** (all 4 players are RL agents)
   - Need to predict opponent responses
   - MCTS shines here

3. ❌ **Agent stuck at 2.8 VP** (current state)
   - Problem is immediate actions, not long-term planning
   - Fix reward function first

### Recommendation: **NOT YET**

**Priority now:**
1. Fix reward function (remove barriers to building)
2. Fix curriculum learning (prevent forgetting)
3. Add multi-action incentives (take multiple actions per turn)
4. **THEN** consider look-ahead if agent plateaus at 6-7 VP

**Look-ahead would help with:**
- "Should I save for city or build settlement now?"
- "Will opponent reach 10 VP before me?"
- "Should I block opponent's longest road?"

**But these questions don't matter if agent can't even reach 4 VP!**

---

## 🚀 **Recommended Enhancement: Improve PBRS Instead**

Rather than adding complex look-ahead, improve your existing PBRS:

### Enhanced Potential Function

**Location:** `catan_env_pytorch.py:359-387`

**REPLACE with:**

```python
def _calculate_potential(self, player: Player):
    potential = 0.0

    # 1. Production Potential (INCREASED weight)
    pip_map = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1}
    production_potential = 0
    for settlement in player.settlements:
        for tile in settlement.position.adjacent_tiles:
            if tile.number:
                production_potential += pip_map.get(tile.number, 0)
    for city in player.cities:
        for tile in city.position.adjacent_tiles:
             if tile.number:
                production_potential += 2 * pip_map.get(tile.number, 0)
    potential += production_potential * 0.3  # Increased from 0.1

    # 2. Resource Diversity (NEW - encourages balanced economy)
    resource_counts = [
        player.resources[ResourceType.WOOD],
        player.resources[ResourceType.BRICK],
        player.resources[ResourceType.WHEAT],
        player.resources[ResourceType.SHEEP],
        player.resources[ResourceType.ORE]
    ]
    # Entropy of resource distribution (higher = more diverse)
    total_resources = sum(resource_counts)
    if total_resources > 0:
        resource_entropy = 0
        for count in resource_counts:
            if count > 0:
                p = count / total_resources
                resource_entropy -= p * np.log(p + 1e-8)
        # Normalize to 0-1 range (max entropy for 5 resources is log(5) ≈ 1.6)
        resource_diversity = resource_entropy / 1.6
        potential += resource_diversity * 2.0  # Reward balanced economy

    # 3. Building Potential (NEW - rewards being "close" to building)
    # Can almost build city?
    wheat = player.resources[ResourceType.WHEAT]
    ore = player.resources[ResourceType.ORE]
    city_progress = (min(wheat, 3) + min(ore, 2)) / 5.0
    if len(player.settlements) > 0:  # Only if have settlements to upgrade
        potential += city_progress * 3.0

    # Can almost build settlement?
    wood = player.resources[ResourceType.WOOD]
    brick = player.resources[ResourceType.BRICK]
    sheep = player.resources[ResourceType.SHEEP]
    settlement_progress = (min(wood, 1) + min(brick, 1) +
                          min(wheat, 1) + min(sheep, 1)) / 4.0
    potential += settlement_progress * 2.0

    # 4. Strategic Assets
    if player.has_longest_road: potential += 4.0  # Increased from 2.0
    if player.has_largest_army: potential += 4.0  # Increased from 2.0
    potential += player.development_cards.get(DevelopmentCardType.VICTORY_POINT, 0) * 2.0

    # 5. Victory Points (NEW - directly encode progress)
    vp = player.calculate_victory_points()
    potential += vp * 5.0  # Strong signal for VP progress

    # 6. Opponent Threat (CAPPED to reduce variance)
    max_opp_vp = 0
    for i, opp in enumerate(self.game_env.game.players):
        if opp != player:
            opp_vp = opp.calculate_victory_points()
            max_opp_vp = max(max_opp_vp, opp_vp)

    # Only penalize if opponent is actually threatening (8+ VP)
    if max_opp_vp >= 8:
        threat_penalty = min((max_opp_vp - 7) * 2.0, 10.0)  # Capped
        potential -= threat_penalty

    return potential
```

### Why This Is Better Than Look-Ahead:

1. **Faster** - No extra rollouts needed
2. **More informative** - Guides agent toward productive states
3. **Fixes current issues** - Addresses why agent stuck at 2.8 VP
4. **Still theoretically sound** - PBRS with richer potential function

### New Potential Components:

| Component | Weight | What It Does |
|-----------|--------|--------------|
| Production pips | 0.3x | Rewards good board positions |
| Resource diversity | 2.0 | Prevents "all wheat, no brick" traps |
| City progress | 3.0 | Rewards being close to building city |
| Settlement progress | 2.0 | Rewards accumulating settlement resources |
| VP direct | 5.0 | Strong signal for VP gain |
| Longest road | 4.0 | Increased value (worth 2 VP) |
| Largest army | 4.0 | Increased value (worth 2 VP) |
| Opponent threat | -2.0x (capped) | Reduced variance |

---

## 📊 **Expected Impact: Enhanced PBRS**

### Current PBRS Issues:
- Agent at 2.8 VP gets potential ≈ 8-10
- Building 1 settlement: Potential increases by ~2-3
- **Not enough signal to overcome hoarding penalties**

### Enhanced PBRS:
- Agent at 2.8 VP gets potential ≈ 25-30 (includes 2.8*5 = 14 from VP)
- Accumulating resources for city:
  - 3 wheat, 2 ore: +3.0 from city_progress
  - Balanced resources: +2.0 from diversity
  - **Total: +5.0 from resource accumulation alone!**
- Building settlement:
  - VP increases: +5.0 (from 1 VP × 5.0)
  - Production increases: +1-2 (from new pips × 0.3)
  - **Total: +6-7 potential increase**

**This makes building MUCH more attractive than current weak rewards!**

---

## 🎯 **Recommendation Summary**

### ✅ **DO NOW:**
1. Fix PBRS opponent threat (cap penalty)
2. Enhance PBRS with new components (above)
3. Fix hoarding penalty (from FIXES_TO_APPLY.md)
4. Add multi-action incentives (from MULTI_ACTION_ANALYSIS.md)

### 🤔 **CONSIDER LATER (after agent reaches 6-7 VP):**
1. **Simple look-ahead (easiest):**
   - Add "end-of-turn value prediction"
   - Agent predicts: "If I end turn now, what's my expected value next turn?"
   - Helps with timing decisions

2. **Monte Carlo rollouts (medium complexity):**
   - Before choosing action, simulate 5-10 random continuations
   - Pick action with best average outcome
   - 5-10x slower but more strategic

3. **MCTS (hardest, most powerful):**
   - Full tree search like AlphaGo
   - Only needed for high-level play (8+ VP consistently)
   - Requires significant engineering

### ❌ **DON'T DO NOW:**
- Complex look-ahead systems
- Model-based planning
- Multi-agent reasoning

**The agent needs to learn to walk (reach 4 VP) before it learns to run (strategic planning).**

---

## 💡 **Alternative: Simple Look-Ahead Augmentation**

If you want to experiment with look-ahead WITHOUT major changes:

### "Greedy Rollout" Technique

**Add to `agent_gpu.py`:**

```python
def choose_action_with_rollout(self, obs, action_mask, vertex_mask, edge_mask, env, depth=3):
    """Choose action by simulating a few steps ahead"""

    # Get valid actions
    valid_actions = [i for i, m in enumerate(action_mask) if m > 0]

    best_action = None
    best_value = float('-inf')

    for action in valid_actions[:5]:  # Only try top 5 actions (speed)
        # Clone environment (if possible) or restore state after
        env_copy = copy.deepcopy(env)

        # Simulate taking this action
        total_reward = 0
        for step in range(depth):
            obs, reward, done, _, _ = env_copy.step(action, ...)
            total_reward += reward * (0.99 ** step)  # Discounted

            if done:
                break

            # Continue with greedy policy
            action = self.choose_action(obs, ...)

        if total_reward > best_value:
            best_value = total_reward
            best_action = action

    return best_action
```

**Pros:**
- Simple to implement
- 3-5x slower (acceptable)
- Helps with multi-step decisions

**Cons:**
- Requires env copying (might not work with current setup)
- Only explores greedy paths
- Still not as good as MCTS

**Verdict:** Try enhanced PBRS first. If still stuck at 5-6 VP after all fixes, THEN try rollouts.
