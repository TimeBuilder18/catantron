# Catan RL Project Analysis

## Executive Summary

Your Catan reinforcement learning project has several critical issues causing the learning curve to plateau around 50k episodes and then decline. The model analyzer is also non-functional. Below is a comprehensive analysis with actionable recommendations.

---

## 🔴 CRITICAL ISSUES

### 1. Learning Curve Degradation (50k+ episodes)

**Problem:** Model performance improves until ~50k episodes, then curves downward.

**Root Causes:**

#### A. Reward Function Instability
**Location:** `catan_env_pytorch.py:389-462`

**Issues:**
1. **Conflicting Reward Signals:**
   - VP state bonus (+0.1 per VP) conflicts with building rewards
   - Hoarding penalty discourages necessary resource accumulation for cities (6 cards needed)
   - Inaction penalty (-10) is too harsh and might cause premature ending of turns

2. **PBRS Implementation Risk:**
   ```python
   pbrs_reward = self.gamma * new_potential - old_potential  # Line 396
   ```
   - The potential function includes opponent VP penalties which can swing wildly
   - Heavy penalty for opponents close to winning (-5.0 per VP over 7) creates high variance
   - Might cause the agent to prioritize defense over winning

3. **Hoarding Penalty Too Aggressive:**
   ```python
   if total_cards > 7:
       excess_cards = total_cards - 7
       hoarding_penalty = 1.0 * excess_cards
       if excess_cards > 10:
           hoarding_penalty += 2.0 * (excess_cards - 10)
   ```
   - Building a city requires 6 cards (3 wheat + 2 ore)
   - With hand of 8-10 cards, agent gets penalized for normal city-building strategy
   - Discourages optimal play patterns

#### B. Curriculum Learning Issues
**Location:** `train_clean.py:85-161`

**Issues:**
1. **Mastery Window Clearing:**
   ```python
   episode_vps.clear()  # Line 161 - DANGEROUS
   ```
   - Clears VP history when advancing stages
   - Causes sudden policy shift without proper transition
   - Can trigger catastrophic forgetting

2. **Fixed Mastery Threshold:**
   - 90% threshold might be too strict or too lenient depending on stage
   - No adaptive difficulty adjustment
   - Agent might get stuck at a stage or advance too quickly

3. **No Curriculum Decay:**
   - Same learning rate across all curriculum stages
   - No exploration decay as agent masters easier stages
   - Entropy bonus (0.05) stays constant, causing excessive exploration late in training

#### C. Hyperparameter Issues
**Location:** `train_clean.py:122-133`

**Problems:**
1. **Static Learning Rate:** 3e-4 never decays
2. **High Entropy Coefficient:** 0.05 is high for late-stage training
3. **No Gradient Clipping Adaptation:** Fixed at 0.5
4. **Batch Size:** 1024 might be too large for early learning

---

### 2. Non-Functional Model Analyzer

**Problem:** `analyze_checkpoints.py` only lists files, doesn't analyze performance.

**Current State:**
- Only shows checkpoint metadata
- No actual model evaluation
- No performance metrics
- No behavioral analysis
- Just prints file info and recommends manual evaluation

**What's Missing:**
- Automated performance evaluation
- Learning curve visualization
- Action distribution analysis
- Value function diagnostics
- Policy entropy tracking
- Win rate over time

---

## ⚠️ MAJOR ISSUES

### 3. Network Architecture Limitations
**Location:** `network_gpu.py:20-44`

**Issues:**
1. **No Normalization:**
   - No LayerNorm or BatchNorm
   - Can cause training instability
   - Harder to optimize deep networks

2. **Simple MLP:**
   - No skip connections (ResNet-style)
   - Deep network (768→768→512→256) prone to vanishing gradients
   - No attention mechanism for spatial information (board state)

3. **Fixed Architecture:**
   - 121-dimensional input might lose spatial relationships
   - All board positions flattened (19 tiles × 3 features = 57 dims)
   - Could benefit from convolutional or graph neural network layers

### 4. Experience Buffer Problems
**Location:** `agent_gpu.py:64-147`

**Issues:**
1. **No Size Limit:**
   - Buffer can grow indefinitely within an update cycle
   - Memory inefficient
   - No prioritization of important experiences

2. **Full Clearing After Update:**
   - All experiences discarded after PPO update
   - Recent successful strategies completely forgotten
   - No experience replay across updates

3. **No Off-Policy Data:**
   - Only uses data from current policy
   - Doesn't leverage historical good plays
   - Sample efficiency suffers

### 5. Training Instabilities

**Multiple Sources:**
1. **Value Function Divergence:**
   - MSE loss on returns can explode with high variance rewards
   - No value clipping like in modern PPO implementations

2. **Ratio Clipping:**
   - Standard PPO clipping (0.2) might not be appropriate for multi-stage curriculum
   - No adaptive clipping based on KL divergence

3. **Multi-Head Policy Gradients:**
   - Combines log probs from action, vertex, edge, trade heads
   - Different heads might learn at different rates
   - No per-head loss balancing

---

## 🟡 MODERATE ISSUES

### 6. Reward Shaping Details

**Specific Problems:**

1. **VP Differential Too High:**
   ```python
   vp_reward = vp_diff * 8.0  # Line 429 - Very high multiplier
   ```
   - 8.0 multiplier on VP gain might overshadow other rewards
   - Could cause agent to only focus on VP, ignoring strategy

2. **Building Rewards Imbalanced:**
   ```python
   building_reward = settlement_diff * 1.0 + city_diff * 2.0 + road_diff * 1.5
   ```
   - Cities worth only 2x settlements, but cost ~3x more resources
   - Roads worth 1.5 (more than settlements!) - encourages road spam

3. **Win Bonus Too Large:**
   ```python
   reward += 50.0  # Line 455 - Massive spike
   ```
   - 50.0 win bonus creates huge reward spike at end
   - Overshadows all intermediate rewards
   - Can destabilize value function

### 7. Action Masking Edge Cases

**Location:** `catan_env_pytorch.py:90-153`

**Issues:**
1. **Fallback to All Valid:**
   ```python
   if mask.sum() == 0:
       mask[:] = 1.0  # Lines 120, 152
   ```
   - When no valid actions, allows ALL actions
   - Should only allow "do_nothing" or "wait"
   - Can cause invalid action attempts

2. **Mask Inconsistency:**
   - Vertex and edge masks default to all 1s during waiting
   - Might confuse the network about valid spatial positions

### 8. Observation Space Issues

**Location:** `catan_env_pytorch.py:155-251`

**Issues:**
1. **Feature Normalization:**
   - Resources (0-20) and VPs (0-10) not normalized
   - Tile numbers (2-12) on different scale than resource types (1-6)
   - Network must learn different scales for different features

2. **Missing Features:**
   - No longest road length (only binary has_longest_road)
   - No largest army size (only binary has_largest_army)
   - No resource diversity/balance information
   - No turn number or game progress indicator

3. **Port Encoding:**
   - Only 2 features per port (type + access)
   - No information about port trade ratios
   - Doesn't encode which specific resources are available at 2:1 ports

---

## 💡 RECOMMENDATIONS

### Priority 1: Fix Reward Function (CRITICAL)

**File:** `catan_env_pytorch.py`

**Changes:**

1. **Reduce Reward Variance:**
   ```python
   # Scale down VP reward
   vp_reward = vp_diff * 3.0  # Instead of 8.0

   # Scale down win bonus
   reward += 20.0  # Instead of 50.0

   # Remove opponent VP penalties from PBRS potential
   # Or cap the maximum penalty
   ```

2. **Fix Hoarding Penalty:**
   ```python
   # Only penalize after VP threshold AND larger hand
   if new_obs['my_victory_points'] > 5:  # Increased from 3
       total_cards = sum(new_obs['my_resources'].values())
       if total_cards > 10:  # Increased from 7
           excess_cards = total_cards - 10
           hoarding_penalty = 0.5 * excess_cards  # Reduced from 1.0
           reward -= hoarding_penalty
   ```

3. **Reduce Inaction Penalty:**
   ```python
   # Make less harsh
   inaction_penalty = -3.0  # Instead of -10.0
   ```

4. **Add Value Clipping:**
   ```python
   # In trainer_gpu.py, add clipped value loss
   values_clipped = old_values + torch.clamp(
       state_values.squeeze() - old_values,
       -self.clip_epsilon,
       self.clip_epsilon
   )
   value_loss_unclipped = (state_values.squeeze() - batch_returns) ** 2
   value_loss_clipped = (values_clipped - batch_returns) ** 2
   value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
   ```

### Priority 2: Fix Curriculum Learning (CRITICAL)

**File:** `train_clean.py`

**Changes:**

1. **Don't Clear VP Window:**
   ```python
   # Line 161 - Comment out or remove
   # episode_vps.clear()  # REMOVED - causes catastrophic forgetting

   # Instead, track VP as percentage of target
   if len(episode_vps) == MASTERY_WINDOW:
       avg_recent_vp = np.mean(episode_vps)
       vp_percentage = avg_recent_vp / current_vp_target
       if vp_percentage >= MASTERY_THRESHOLD:
           # Advance curriculum
   ```

2. **Add Learning Rate Scheduling:**
   ```python
   # Add scheduler after optimizer creation
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
       trainer.optimizer,
       T_max=args.episodes,
       eta_min=1e-5
   )

   # Update after each episode or update
   scheduler.step()
   ```

3. **Decay Entropy Coefficient:**
   ```python
   # Calculate dynamic entropy coefficient
   progress = episode / args.episodes
   entropy_coef = 0.05 * (1.0 - progress * 0.8)  # Decays to 0.01
   trainer.entropy_coef = entropy_coef
   ```

### Priority 3: Create Functional Model Analyzer (HIGH)

**Create new:** `analyze_model_performance.py`

**Features Needed:**
- Load checkpoint
- Run evaluation games
- Track metrics over checkpoints
- Visualize learning curves
- Analyze action distributions
- Detect policy collapse
- Compare checkpoints

### Priority 4: Improve Network Architecture (MEDIUM)

**File:** `network_gpu.py`

**Changes:**

1. **Add Layer Normalization:**
   ```python
   self.ln1 = nn.LayerNorm(768)
   self.ln2 = nn.LayerNorm(768)
   self.ln3 = nn.LayerNorm(512)
   self.ln4 = nn.LayerNorm(256)

   # In forward:
   x = self.ln1(F.relu(self.fc1(obs)))
   x = self.ln2(F.relu(self.fc2(x)))
   # etc.
   ```

2. **Add Skip Connections:**
   ```python
   # Add residual connections
   x1 = F.relu(self.fc1(obs))
   x2 = F.relu(self.fc2(x1))
   x2 = x2 + x1  # Skip connection (requires matching dimensions)
   ```

3. **Consider Separate Value/Policy Networks:**
   - Current shared trunk might cause conflicts
   - Value network benefits from different features than policy

### Priority 5: Add Training Diagnostics (MEDIUM)

**What to Track:**
1. Policy entropy over time
2. Value function predictions vs actual returns
3. Advantage distribution (should be centered at 0)
4. Gradient norms (detect exploding/vanishing gradients)
5. KL divergence between old and new policies
6. Per-action selection frequencies
7. Episode length distribution
8. Resource accumulation patterns

### Priority 6: Improve Experience Collection (LOW)

**File:** `agent_gpu.py`

**Changes:**

1. **Add Buffer Size Limit:**
   ```python
   class ExperienceBuffer:
       def __init__(self, max_size=10000):
           self.max_size = max_size
           # ... existing code ...

       def store(self, ...):
           # ... existing code ...
           # Remove oldest if over limit
           if len(self.states) > self.max_size:
               for attr in [self.states, self.actions, ...]:
                   attr.pop(0)
   ```

2. **Consider Keeping Some Experiences:**
   - Don't clear entire buffer every update
   - Keep top 20% of experiences by reward
   - Mix old and new data

---

## 🔍 DEBUGGING RECOMMENDATIONS

### Immediate Actions:

1. **Add Detailed Logging:**
   ```python
   # In train_clean.py, log every 100 episodes:
   - Average reward per component (PBRS, VP, building, penalties)
   - Policy entropy
   - Value function mean/std
   - Gradient norms
   - Learning rate
   ```

2. **Create Checkpoint Comparison Script:**
   - Evaluate models at 10k, 20k, 30k, 40k, 50k, 60k episodes
   - Compare behavior differences
   - Identify when degradation starts

3. **Analyze Failed Episodes:**
   - Save episodes where model performs poorly
   - Review action sequences
   - Check if agent gets stuck in loops

### Testing Protocol:

1. **Test with Fixed Opponents:**
   - Same rule-based opponents every game
   - Reduces variance in evaluation
   - Easier to detect actual learning

2. **Ablation Studies:**
   - Train without curriculum
   - Train with simplified reward
   - Train with different entropy coefficients
   - Compare results

3. **Smaller Test Runs:**
   - Before 100k episode runs, test changes for 5k episodes
   - Faster iteration on fixes
   - Catch regressions early

---

## 📊 EXPECTED IMPROVEMENTS

After implementing these fixes:

1. **Learning Curve:** Should improve steadily to 80-100k episodes
2. **VP Achievement:** Should reach 6-8 VP average (for VP target of 10)
3. **Training Stability:** Less variance in episode rewards
4. **Natural Endings:** >60% natural game completions (vs timeouts)
5. **Strategic Behavior:** More city building, better resource management

---

## 🚀 IMPLEMENTATION PRIORITY

1. **Week 1:** Fix reward function + curriculum (Priority 1-2)
2. **Week 2:** Create model analyzer + diagnostics (Priority 3, 5)
3. **Week 3:** Improve network architecture (Priority 4)
4. **Week 4:** Refine experience collection (Priority 6)

---

## 📝 NOTES

- Your recent commits show "MAJOR: Rebalance reward function to fix pathological behavior" but issues remain
- Consider git bisect to find exactly when performance degraded
- The curriculum stages [4,5,6,7,8,10] are reasonable but might advance too quickly
- Rule-based AI opponents might be too weak/predictable - consider self-play later

---

## ⚠️ CRITICAL WARNING

**Do NOT train for 100k episodes with current setup!**

The reward function instabilities will likely cause:
- Catastrophic forgetting after 50k episodes
- Policy collapse (agent gets stuck in do_nothing loop)
- Value function divergence (predicted values become meaningless)

**First fix Priority 1-2 issues, then test for 10k episodes to verify stability.**
