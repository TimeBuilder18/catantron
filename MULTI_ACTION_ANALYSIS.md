# Multiple Actions Per Turn - Analysis & Fixes

## Current State

**The environment DOES support multiple actions per turn**, but the agent might not be learning this properly.

### How It Works Now:

```python
# In train_clean.py (lines 178-208)
while not done and step_count < max_steps:
    action = agent.choose_action(obs, ...)
    next_obs, reward, terminated, truncated, info = env.step(action, ...)
    done = terminated or truncated
    obs = next_obs
    # Loop continues - agent can take another action!
```

The agent CAN:
1. Roll dice (action 0)
2. Build settlement (action 3)
3. Build road (action 5)
4. Build city (action 4)
5. End turn (action 7)

All in the same turn!

### The Problem:

**The agent doesn't have strong enough signals to learn this behavior.**

#### Evidence from your 2.8 VP plateau:

Your agent is stuck at 2.8 VP, which suggests:
- It's doing initial placement (2 VP)
- Maybe 1 additional settlement occasionally (0.8 avg)
- **NOT building multiple things per turn**
- Likely pattern: Roll → End Turn immediately

#### Why This Happens:

1. **No "actions per turn" reward bonus**
   - Agent gets same reward for "roll + end turn" vs "roll + build + build + end turn"
   - Except for building rewards (1.0, 2.0, 1.5) which are weak

2. **Inaction penalty is confusing**
   - Only triggers IF you end_turn when builds available
   - Doesn't encourage taking multiple actions BEFORE ending turn
   - Agent might learn: "Avoid end_turn" instead of "Do more actions before ending"

3. **No temporal credit assignment**
   - All rewards in a turn are treated equally
   - Agent doesn't learn: "I should do A, then B, then C, THEN end turn"

4. **Hoarding penalty sabotages multi-action turns**
   - Agent accumulates 6 cards (enough for settlement + road)
   - Gets penalized for having 6+ cards
   - Learns to spend/end turn immediately to avoid penalty
   - Never accumulates enough to do multiple builds

---

## 🔧 **FIXES FOR MULTI-ACTION LEARNING**

### Fix 1: Add "Actions Per Turn" Reward Bonus

**Location:** `catan_env_pytorch.py` - add to reward function

**Add after line 403:**

```python
# Track actions per turn to encourage multi-action turns
if not hasattr(self, '_turn_action_count'):
    self._turn_action_count = 0
    self._last_turn_player = None

# Reset counter when turn changes
current_turn_player = self.game_env.game.current_player_index
if current_turn_player != self._last_turn_player:
    self._turn_action_count = 0
    self._last_turn_player = current_turn_player

# Increment for productive actions
productive_actions = ['build_settlement', 'build_city', 'build_road',
                     'buy_dev_card', 'trade_with_bank']
if action_name in productive_actions:
    self._turn_action_count += 1
    # Bonus for multiple productive actions in same turn
    if self._turn_action_count >= 2:
        multi_action_bonus = 2.0 * (self._turn_action_count - 1)
        reward += multi_action_bonus
        reward_breakdown['multi_action_bonus'] = multi_action_bonus
```

This rewards:
- 2nd action in turn: +2.0
- 3rd action in turn: +4.0
- 4th action in turn: +6.0

### Fix 2: Make Inaction Penalty More Granular

**Location:** `catan_env_pytorch.py:405-411`

**REPLACE:**
```python
if action_name == 'end_turn':
    legal_actions = old_obs.get('legal_actions', [])
    build_actions = {'build_settlement', 'build_city', 'build_road', 'buy_dev_card'}
    if any(action in legal_actions for action in build_actions):
        inaction_penalty = -10.0
        reward += inaction_penalty
        reward_breakdown['inaction_penalty'] = inaction_penalty
```

**WITH:**
```python
if action_name == 'end_turn':
    legal_actions = old_obs.get('legal_actions', [])
    build_actions = {'build_settlement', 'build_city', 'build_road', 'buy_dev_card'}
    available_builds = [a for a in build_actions if a in legal_actions]

    if available_builds:
        # Graduated penalty based on HOW MUCH you could have done
        num_available = len(available_builds)

        # Check if agent has resources for high-value builds
        my_resources = old_obs['my_resources']

        # Can build city (high value)
        if 'build_city' in available_builds:
            inaction_penalty = -5.0  # Reduced from -10.0
        # Can build settlement (medium value)
        elif 'build_settlement' in available_builds:
            inaction_penalty = -3.0
        # Can only build road or dev card (lower value)
        else:
            inaction_penalty = -1.5

        reward += inaction_penalty
        reward_breakdown['inaction_penalty'] = inaction_penalty
```

### Fix 3: Add Turn Progress Tracking to Observation

**Location:** `catan_env_pytorch.py:155-251` (observation function)

The agent currently doesn't know:
- How many actions it has taken this turn
- Whether it's in "start of turn" vs "end of turn" phase

**Add to observation (after line 170):**

```python
# Add turn action tracking to observation
if not hasattr(self, '_turn_action_count_obs'):
    self._turn_action_count_obs = 0

# Track productive actions this turn
if hasattr(self, '_last_action_name'):
    productive = ['build_settlement', 'build_city', 'build_road',
                 'buy_dev_card', 'trade_with_bank']
    if self._last_action_name in productive:
        self._turn_action_count_obs += 1
    elif self._last_action_name == 'end_turn':
        self._turn_action_count_obs = 0

features.append(float(self._turn_action_count_obs))  # How many actions taken this turn
```

**Update observation size calculation (line 62-73):**
```python
size += 11  # Turn state
size += 5   # Resources
size += 3   # Buildings
size += 5   # Dev cards
size += 4   # VP and special
size += 1   # NEW: Actions this turn
size += 18  # Opponents
size += 57  # Tiles
size += 18  # Ports
return size  # Should now be 122 instead of 121
```

**Update network input size (network_gpu.py:20):**
```python
self.fc1 = nn.Linear(122, 768)  # Changed from 121
```

---

## 🎯 **Expected Impact**

With these fixes:

### Before:
```
Agent behavior per turn:
1. Roll dice
2. End turn immediately (or build 1 thing if lucky)
→ Result: 2.8 VP plateau
```

### After:
```
Agent behavior per turn:
1. Roll dice
2. Trade with bank (+0 reward)
3. Build settlement (+3.0 building + 2.0 multi-action = +5.0)
4. Build road (+1.0 building + 4.0 multi-action = +5.0)
5. End turn (no penalty, already built)
→ Result: Should break through 2.8 VP → 3.6+ VP
```

---

## 📊 **Testing Multi-Action Learning**

After applying fixes, run analyzer to check:

```python
# In analyze_model_performance.py output, look for:
# "Action Distribution (Latest Checkpoint)"

# BAD (current):
# end_turn: 45%
# roll_dice: 30%
# do_nothing: 15%
# build_settlement: 7%
# build_road: 3%

# GOOD (after fixes):
# roll_dice: 25%
# build_settlement: 20%
# build_road: 18%
# end_turn: 20%
# trade_with_bank: 10%
# build_city: 7%
```

Also track "Average actions per turn" metric:
- Current: ~1.5 actions/turn (just roll + end_turn)
- Target: ~3.5 actions/turn (roll + build + build + end_turn)

---

## ⚠️ **Important Notes**

1. **Apply hoarding penalty fix FIRST** - Otherwise agent still can't accumulate resources for multiple builds

2. **The multi-action bonus should be stronger than hoarding penalty** - Otherwise agent learns "spend immediately to avoid penalty" instead of "save for multiple builds"

3. **Network input size changes require retraining from scratch** - The observation size change (121→122) means old checkpoints won't work

---

## 🚀 **Implementation Priority**

1. **CRITICAL:** Fix hoarding penalty (from FIXES_TO_APPLY.md)
2. **HIGH:** Add multi-action reward bonus (Fix 1 above)
3. **HIGH:** Make inaction penalty granular (Fix 2 above)
4. **MEDIUM:** Add turn progress to observation (Fix 3 above - requires retrain)

Apply in order, test after each fix.
