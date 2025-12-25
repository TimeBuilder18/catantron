# Critical Fixes to Apply

This document contains specific code fixes for the main issues identified in the analysis.

## Fix 1: Reward Function (catan_env_pytorch.py)

### Issue: Reward instability causing performance degradation after 50k episodes

### Changes to make in `catan_env_pytorch.py`:

#### A. Fix hoarding penalty (lines 444-452)

**BEFORE:**
```python
if new_obs['my_victory_points'] > 3:
    total_cards = sum(new_obs['my_resources'].values())
    if total_cards > 7:
        excess_cards = total_cards - 7
        hoarding_penalty = 1.0 * excess_cards
        if excess_cards > 10:
            hoarding_penalty += 2.0 * (excess_cards - 10)
        reward -= hoarding_penalty
        reward_breakdown['hoarding_penalty'] = -hoarding_penalty
```

**AFTER:**
```python
# Only penalize excessive hoarding that prevents building
if new_obs['my_victory_points'] > 5:  # Changed from 3
    total_cards = sum(new_obs['my_resources'].values())
    if total_cards > 11:  # Changed from 7 - allows holding 10 cards (for city + extras)
        excess_cards = total_cards - 11
        hoarding_penalty = 0.3 * excess_cards  # Reduced from 1.0
        if excess_cards > 5:  # Reduced threshold
            hoarding_penalty += 0.5 * (excess_cards - 5)  # Reduced from 2.0
        reward -= hoarding_penalty
        reward_breakdown['hoarding_penalty'] = -hoarding_penalty
```

#### B. Reduce VP reward multiplier (line 429)

**BEFORE:**
```python
vp_reward = vp_diff * 8.0
```

**AFTER:**
```python
vp_reward = vp_diff * 3.0  # Reduced from 8.0 to reduce reward variance
```

#### C. Reduce inaction penalty (line 410)

**BEFORE:**
```python
inaction_penalty = -10.0
```

**AFTER:**
```python
inaction_penalty = -3.0  # Reduced from -10.0 - less harsh
```

#### D. Reduce win bonus (line 455)

**BEFORE:**
```python
reward += 50.0
```

**AFTER:**
```python
reward += 20.0  # Reduced from 50.0 to reduce terminal reward spike
```

#### E. Cap opponent threat penalty in potential function (line 385)

**BEFORE:**
```python
for i, opp in enumerate(self.game_env.game.players):
    if opp != player:
        opp_vp = opp.calculate_victory_points()
        if opp_vp >= 8:
            potential -= (opp_vp - 7) * 5.0 # Heavy penalty for opponents close to winning
```

**AFTER:**
```python
for i, opp in enumerate(self.game_env.game.players):
    if opp != player:
        opp_vp = opp.calculate_victory_points()
        if opp_vp >= 8:
            # Cap penalty to reduce variance - max penalty is -10 (when opp has 10 VP)
            penalty = min((opp_vp - 7) * 2.0, 10.0)  # Reduced multiplier from 5.0 to 2.0, capped at 10
            potential -= penalty
```

---

## Fix 2: Curriculum Learning (train_clean.py)

### Issue: VP window clearing causes catastrophic forgetting, no learning rate decay

### Changes to make in `train_clean.py`:

#### A. Don't clear VP window on curriculum advancement (line 161)

**BEFORE:**
```python
if avg_recent_vp >= MASTERY_THRESHOLD * current_vp_target:
    if current_stage_index < len(CURRICULUM_STAGES) - 1:
        current_stage_index += 1
        current_vp_target = CURRICULUM_STAGES[current_stage_index]
        GameConstants.VICTORY_POINTS_TO_WIN = current_vp_target
        print(f"\n🎉 MASTERY ACHIEVED! Advancing to VP Target: {current_vp_target}\n")
        episode_vps.clear() # Reset VP window for new stage
```

**AFTER:**
```python
if avg_recent_vp >= MASTERY_THRESHOLD * current_vp_target:
    if current_stage_index < len(CURRICULUM_STAGES) - 1:
        current_stage_index += 1
        current_vp_target = CURRICULUM_STAGES[current_stage_index]
        GameConstants.VICTORY_POINTS_TO_WIN = current_vp_target
        print(f"\n🎉 MASTERY ACHIEVED! Advancing to VP Target: {current_vp_target}\n")
        # DON'T clear episode_vps - keep history for smoother transition
        # Instead, track as percentage of target going forward
```

#### B. Add learning rate scheduler (after line 133)

**ADD AFTER creating trainer:**
```python
# Add learning rate scheduler for stable long-term training
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    trainer.optimizer,
    T_max=args.episodes,
    eta_min=1e-5  # Minimum learning rate
)
print(f"   Learning rate scheduler: CosineAnnealing (3e-4 → 1e-5)")
```

#### C. Add entropy decay (after line 147)

**ADD AFTER variable initialization:**
```python
# Store initial entropy coefficient for decay
initial_entropy_coef = trainer.entropy_coef
```

#### D. Update entropy and learning rate each episode (after line 244)

**ADD in the training loop after policy update:**
```python
if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
    # Update entropy coefficient (decay over time for less exploration late in training)
    progress = (episode + 1) / args.episodes
    trainer.entropy_coef = initial_entropy_coef * (1.0 - progress * 0.7)  # Decays to 30% of initial

    with SuppressOutput():
        metrics = trainer.update_policy(buffer)

    # Step learning rate scheduler
    scheduler.step()

    buffer.clear()
    if (episode + 1) % 500 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"         Policy updated | Loss: {metrics['policy_loss']:.4f} | LR: {current_lr:.2e} | Entropy: {trainer.entropy_coef:.4f}")
        sys.stdout.flush()
```

---

## Fix 3: Add Value Clipping (trainer_gpu.py)

### Issue: Value function can diverge without clipping

### Changes to make in `trainer_gpu.py`:

#### Modify value loss calculation (line 158)

**BEFORE:**
```python
value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)
```

**AFTER:**
```python
# Clipped value loss (like in modern PPO)
values_pred = state_values.squeeze()

# Need to track old values - add this to experience buffer storage
# For now, use simplified unclipped version but cap the returns
batch_returns_clipped = torch.clamp(batch_returns, -100, 100)  # Prevent extreme values
value_loss = nn.MSELoss()(values_pred, batch_returns_clipped)
```

**BETTER SOLUTION (requires storing old values in experience buffer):**
```python
# This requires modifying experience buffer to also store old values
# Clip value predictions to prevent large updates
values_pred = state_values.squeeze()
values_pred_clipped = old_values + torch.clamp(
    values_pred - old_values,
    -self.clip_epsilon,
    self.clip_epsilon
)

# Compute both clipped and unclipped value loss
value_loss_unclipped = (values_pred - batch_returns) ** 2
value_loss_clipped = (values_pred_clipped - batch_returns) ** 2

# Take the maximum (more conservative)
value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

---

## Fix 4: Add Layer Normalization (network_gpu.py)

### Issue: Deep network without normalization can be unstable

### Changes to make in `network_gpu.py`:

#### Add layer normalization (after line 29)

**ADD in __init__ after existing layers:**
```python
# Add layer normalization for training stability
self.ln1 = nn.LayerNorm(768)
self.ln2 = nn.LayerNorm(768)
self.ln3 = nn.LayerNorm(512)
self.ln4 = nn.LayerNorm(256)
```

#### Modify forward pass (lines 40-44)

**BEFORE:**
```python
x = F.relu(self.fc1(obs))
x = F.relu(self.fc2(x))
x = F.relu(self.fc3(x))
x = F.relu(self.fc4(x))
```

**AFTER:**
```python
x = F.relu(self.ln1(self.fc1(obs)))
x = F.relu(self.ln2(self.fc2(x)))
x = F.relu(self.ln3(self.fc3(x)))
x = F.relu(self.ln4(self.fc4(x)))
```

---

## Fix 5: Add Gradient Norm Logging (train_clean.py)

### Issue: No visibility into training dynamics

### Add diagnostic logging (in training loop, around line 240)

**ADD after policy update:**
```python
if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
    with SuppressOutput():
        metrics = trainer.update_policy(buffer)

    # Calculate gradient norms for diagnostics
    total_norm = 0.0
    for p in agent.policy.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    buffer.clear()
    if (episode + 1) % 500 == 0:
        print(f"         Policy updated | Loss: {metrics['policy_loss']:.4f} | Grad norm: {total_norm:.3f}")
        sys.stdout.flush()
```

---

## Fix 6: Add Training Metrics Tracking (train_clean.py)

### Issue: No detailed metrics for debugging

### Add after imports (around line 65):

**ADD:**
```python
# Track detailed metrics for analysis
training_metrics = {
    'episodes': [],
    'avg_rewards': [],
    'avg_vps': [],
    'policy_losses': [],
    'value_losses': [],
    'entropies': [],
    'gradient_norms': [],
    'learning_rates': [],
}
```

### Update metrics during training (in the loop):

**ADD when printing progress (around line 230):**
```python
if (episode + 1) % 100 == 0:
    # ... existing printing code ...

    # Store metrics
    training_metrics['episodes'].append(episode + 1)
    training_metrics['avg_rewards'].append(avg_reward)
    training_metrics['avg_vps'].append(avg_vp)
```

**ADD after policy update:**
```python
if (episode + 1) % args.update_freq == 0 and len(buffer) > 0:
    # ... update code ...

    # Store training metrics
    training_metrics['policy_losses'].append(metrics['policy_loss'])
    training_metrics['value_losses'].append(metrics['value_loss'])
    training_metrics['entropies'].append(metrics['entropy'])
    training_metrics['gradient_norms'].append(total_norm)
    training_metrics['learning_rates'].append(scheduler.get_last_lr()[0])
```

### Save metrics at end (around line 260):

**ADD:**
```python
# Save training metrics
import pickle
metrics_path = f"models/{args.model_name}_metrics.pkl"
with open(metrics_path, 'wb') as f:
    pickle.dump(training_metrics, f)
print(f"Metrics saved -> {metrics_path}")
```

---

## Implementation Priority

1. **CRITICAL - Apply immediately:**
   - Fix 1: Reward function changes
   - Fix 2: Curriculum learning changes

2. **HIGH - Apply before next long training run:**
   - Fix 3: Value clipping
   - Fix 5: Gradient logging

3. **MEDIUM - Apply when refactoring:**
   - Fix 4: Layer normalization (requires retraining from scratch)
   - Fix 6: Detailed metrics tracking

---

## Testing After Fixes

After applying Fixes 1-2:

1. **Short test run:**
   ```bash
   python train_clean.py --episodes 5000 --model-name test_fixes --curriculum
   ```

2. **Check for improvement:**
   - VP should steadily increase (no drop after 3000 episodes)
   - Reward variance should be lower
   - Natural game endings should increase

3. **If successful, run longer:**
   ```bash
   python train_clean.py --episodes 50000 --model-name fixed_training --curriculum
   ```

4. **Analyze checkpoints:**
   ```bash
   python analyze_model_performance.py --model-pattern "models/fixed_training*.pt" --eval-episodes 30
   ```

---

## Expected Results After Fixes

- **Learning stability:** No performance regression after 50k episodes
- **VP achievement:** Should reach 5-7 average VP for 10 VP target
- **Natural endings:** >50% games complete naturally (not timeout)
- **Building behavior:** More cities built (1-2 per game average)
- **Reward variance:** Lower standard deviation in episode rewards

---

## Notes

- These fixes address the root causes identified in PROJECT_ANALYSIS.md
- Some fixes require architectural changes (layer norm) - these need fresh training
- Monitor gradient norms - should stay in range [0.1, 10.0]
- If gradient norms exceed 10, reduce learning rate or increase gradient clipping
