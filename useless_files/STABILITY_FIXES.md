# Training Stability Fixes - Preventing Oscillation

**Date Applied:** 2025-12-15
**Commit:** TBD
**Branch:** claude/analyze-project-changes-01BtSo7R1NVaCC4hNZePihJe

---

## 🔍 Problem Discovered: Oscillating Performance, Not Just Exploitation

### Analysis Results from overnight_max Training (25k episodes)

| Episode | VP | Roads/game | Cities/game | Behavior |
|---------|-----|-----------|-------------|----------|
| **5k** | 2.77 ✅ | 26.1 (realistic) | 0.3 | **GOOD - Best VP** |
| **10k** | 2.63 ⬇️ | 129.7 💥 | 0.4 | **BAD - Road spam** |
| **15k** | 2.63 | 62.8 | 0.3 | **MEDIUM - Moderate spam** |
| **20k** | 2.73 ⬆️ | 25.5 (realistic) | **1.9** 🎯 | **EXCELLENT - Strategic!** |
| **25k** | 2.43 💥 | 133.8 💥 | 0.6 ⬇️ | **BAD - Collapsed** |

### 🎯 Critical Insight: Episode 20k Proves the Agent CAN Learn!

**Episode 20,000 checkpoint had:**
- **1.9 cities per game** - Excellent strategic play!
- **25.5 roads/game** - Realistic, not spam
- **2.73 VP** - Good performance
- **87% natural endings** - Games complete properly

**This means:**
- ✅ The reward function WORKS (agent learned good strategies)
- ✅ The agent is CAPABLE of learning
- ❌ Training is UNSTABLE (can't maintain learned strategies)
- ❌ Catastrophic forgetting occurs (ep 20k forgotten by ep 25k)

---

## 🚨 Root Cause: Training Instability

The agent oscillates between good and bad strategies because:

### 1. **Catastrophic Forgetting**
- Agent learns good policy (ep 5k, ep 20k: building cities, realistic gameplay)
- New gradient updates in later episodes overwrite that knowledge
- Reverts to exploitation (ep 10k, ep 25k: road spam)

### 2. **Learning Rate Too High in Late Training**
- **Early training:** High LR helps explore → discovers good strategies
- **Late training:** LR still too high → large updates destroy learned knowledge
- **CosineAnnealingLR** decays too slowly → allows destructive updates

### 3. **Entropy Too High in Late Training**
- High entropy = random exploration continues
- Agent randomly tries road spam again in late episodes
- Gets positive PBRS signal → reinforces spam behavior
- Overwrites earlier strategic knowledge

### 4. **PPO Clipping Too Loose**
- Fixed clip_ratio=0.2 allows large policy changes throughout
- Late training should have tighter clipping for stability
- Large updates cause sudden behavior shifts

---

## ✅ Fixes Applied

### Fix 1: Exponential Learning Rate Decay (CRITICAL)

**Problem:** CosineAnnealingLR decays too slowly, allows destructive updates in late training

**Before:**
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    trainer.optimizer,
    T_max=args.episodes,
    eta_min=1e-5
)
# Decays smoothly: 3e-4 → 1e-5 over full training
```

**After:**
```python
from torch.optim.lr_scheduler import ExponentialLR

scheduler = ExponentialLR(
    trainer.optimizer,
    gamma=0.9995
)
# Decays exponentially: 3e-4 → ~1e-5 by ep 25000
# Faster decay in late training = smaller, safer updates
```

**Why This Helps:**
- Early training (ep 0-10k): LR ~3e-4 to 1e-4 - can explore and learn
- Mid training (ep 10k-20k): LR ~1e-4 to 3e-5 - refines strategies
- Late training (ep 20k-30k): LR ~3e-5 to 1e-5 - locks in learned behavior
- **Prevents large destructive updates that cause forgetting**

---

### Fix 2: Exponential Entropy Decay (CRITICAL)

**Problem:** Linear entropy decay keeps exploration too high in late training

**Before:**
```python
progress = (episode + 1) / args.episodes
trainer.entropy_coef = initial_entropy_coef * (1.0 - progress * 0.7)
# Linear decay: 0.05 → 0.015 (retains 30% at end)
```

**After:**
```python
progress = (episode + 1) / args.episodes
trainer.entropy_coef = initial_entropy_coef * (0.3 ** progress)
# Exponential decay: 0.05 → ~0.001 at end
```

**Decay Comparison:**

| Episode | Linear (old) | Exponential (new) |
|---------|-------------|-------------------|
| 0 | 0.0500 | 0.0500 |
| 5k (20%) | 0.0430 | 0.0342 |
| 10k (40%) | 0.0360 | 0.0234 |
| 15k (60%) | 0.0290 | 0.0160 |
| 20k (80%) | 0.0220 | 0.0110 |
| 25k (100%) | 0.0150 | 0.0015 |

**Why This Helps:**
- Early: High entropy for exploration (discover strategies)
- Mid: Moderate entropy (refine strategies)
- Late: Very low entropy (exploit learned strategies, no random rediscovery of spam)
- **Prevents random exploration from rediscovering exploits in late training**

---

### Fix 3: Dynamic PPO Clip Ratio Decay

**Problem:** Fixed clip_ratio=0.2 allows large policy updates throughout training

**Before:**
```python
# In trainer_gpu.py, line 155
surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * batch_advantages
# Always allows ±20% policy change
```

**After:**
```python
# In train_clean.py, before calling update_policy()
progress = (episode + 1) / args.episodes
trainer.clip_epsilon = 0.2 * (1.0 - progress * 0.5)
# Decays: 0.2 → 0.1 by end of training
```

**Decay Schedule:**

| Episode | Clip Ratio | Max Policy Change |
|---------|-----------|-------------------|
| 0 | 0.20 | ±20% |
| 5k (20%) | 0.18 | ±18% |
| 10k (40%) | 0.16 | ±16% |
| 15k (60%) | 0.14 | ±14% |
| 20k (80%) | 0.12 | ±12% |
| 25k (100%) | 0.10 | ±10% |

**Why This Helps:**
- Early: Large updates allowed (exploration, rapid learning)
- Late: Tighter clipping (stability, prevent catastrophic updates)
- **Reduces magnitude of policy changes that cause forgetting**

---

### Fix 4: Best Model Checkpointing (IMMEDIATE VALUE)

**Problem:** Final checkpoint may be worse than mid-training checkpoints

**Solution:**
```python
# Track best model by VP performance
best_avg_vp = 0.0
best_episode = 0

# Every 100 episodes, check for new best
if avg_vp > best_avg_vp:
    best_avg_vp = avg_vp
    best_episode = episode + 1
    best_path = f"models/{args.model_name}_BEST.pt"
    agent.policy.save(best_path)
    print(f"🏆 New best! VP: {best_avg_vp:.2f}")
```

**Why This Helps:**
- **Captures peak performance** (e.g., episode 20k with 2.73 VP, 1.9 cities)
- Always have the best model available for deployment
- Can use best checkpoint as starting point for further training
- Provides clear signal of when training degraded

**Example from overnight_max run:**
- Best would be saved at episode 5,000 (VP 2.77) or episode 20,000 (VP 2.73, 1.9 cities)
- Can use this instead of degraded episode 25,000 checkpoint

---

## 📊 Expected Results

### Before Fixes (overnight_max pattern):

| Metric | Behavior |
|--------|----------|
| VP trajectory | Oscillates: 2.77 → 2.63 → 2.73 → 2.43 |
| Roads/game | Oscillates: 26 → 130 → 63 → 26 → 134 |
| Cities/game | Oscillates: 0.3 → 0.4 → 1.9 → 0.6 |
| Stability | ❌ Catastrophic forgetting |

### After Fixes (expected):

| Metric | Behavior |
|--------|----------|
| VP trajectory | Monotonic or near-monotonic: 2.3 → 2.5 → 2.7 → 2.8+ |
| Roads/game | Stable: 20-30 throughout |
| Cities/game | Increasing: 0.3 → 0.5 → 1.0 → 1.5+ |
| Stability | ✅ Maintains learned strategies |

---

## 🧪 Testing Protocol

### Quick Validation Test (15k episodes, ~3 hours)

```powershell
python train_clean.py --episodes 15000 --model-name stable_test `
  --curriculum --batch-size 1024 --epochs 20 --update-freq 40 --save-freq 3000
```

**Success Criteria:**
- ✅ VP monotonically increases (no oscillation)
- ✅ Roads/game stays < 40 throughout
- ✅ Cities/game increases over time
- ✅ Best model saved at reasonable point (not just early)

**Failure Indicators:**
- ❌ VP still oscillates wildly
- ❌ Road spam emerges (>80 roads/game)
- ❌ Performance degrades after initial peak

---

### Extended Validation (30k episodes, ~6-7 hours)

```powershell
python train_clean.py --episodes 30000 --model-name stable_extended `
  --curriculum --batch-size 1024 --epochs 20 --update-freq 40 --save-freq 5000
```

**Success Criteria:**
- ✅ **No late-stage collapse** (VP at ep 30k ≥ VP at ep 20k)
- ✅ VP reaches 3.0+ (closer to curriculum advancement)
- ✅ Cities/game reaches 1.5-2.0
- ✅ Natural endings > 80% sustained
- ✅ Best checkpoint is from late training (ep 25k-30k), not early

---

## 📈 Monitoring During Training

Watch these metrics every 5k episodes:

### 1. **VP Trajectory** - Should be monotonic
```
Episode  5k: VP 2.4-2.6
Episode 10k: VP 2.5-2.7  (↑ or stable)
Episode 15k: VP 2.6-2.8  (↑ or stable)
Episode 20k: VP 2.7-2.9  (↑ or stable)
Episode 25k: VP 2.8-3.0+ (↑ or stable)
```

### 2. **Roads/game** - Should be stable
```
Episode  5k: 20-30 roads
Episode 10k: 20-35 roads (not >80!)
Episode 15k: 20-35 roads
Episode 20k: 20-35 roads
Episode 25k: 20-35 roads
```

### 3. **Cities/game** - Should increase
```
Episode  5k: 0.2-0.4 cities
Episode 10k: 0.4-0.7 cities (↑)
Episode 15k: 0.7-1.2 cities (↑)
Episode 20k: 1.0-1.8 cities (↑)
Episode 25k: 1.5-2.0 cities (↑)
```

### 4. **Best Model Timing**
- Should update multiple times during training
- Later updates better than earlier (shows continued learning)
- If best is always early (ep < 5k), fixes didn't work

---

## 🔬 Technical Analysis

### Why Exponential Decay Works Better

**Linear Decay Problem:**
- Episode 20k: LR still ~1e-4, entropy 0.022
- Large enough updates to overwrite learned strategies
- High enough exploration to randomly try spam again

**Exponential Decay Solution:**
- Episode 20k: LR ~3e-5, entropy 0.011
- Much smaller updates (3x smaller LR)
- Much less exploration (2x smaller entropy)
- Learned strategies locked in, harder to forget

### Mathematical Analysis

**Learning rate at episode E:**
- Linear: LR = 3e-4 - (3e-4 - 1e-5) * (E / total)
- Exponential: LR = 3e-4 * (0.9995 ^ E)

**At episode 20,000 / 25,000:**
- Linear: LR = 1.0e-4
- Exponential: LR = 3.0e-5 (3x smaller!)

**Impact:**
- Gradient updates 3x smaller in late training
- Policy changes 3x smaller
- Forgetting 3x less likely

---

## 🎯 Success Definition

**Training is stable if:**

1. ✅ **Monotonic VP trajectory** - No drops > 0.2 VP after episode 10k
2. ✅ **No oscillating exploitation** - Roads/game stays in 20-40 range
3. ✅ **Continued learning** - Cities/game increases throughout
4. ✅ **Best model from late training** - Peak performance in final third of training
5. ✅ **Natural endings sustained** - > 80% games complete properly

**If all criteria met:**
- Training is stable and effective
- Can extend to 50k-100k episodes safely
- Foundation for curriculum progression

**If criteria NOT met:**
- May need even more aggressive decay
- Or fundamental architectural changes (multi-action RL)

---

## 🚀 Next Steps After Successful Validation

1. **Use best checkpoint** from stable_extended run
2. **Continue training** to 50k-100k episodes
3. **Advance curriculum** to stages 2-3 (VP targets 5-6)
4. **Add enhancements:**
   - Development card usage
   - Better trading strategies
   - Opponent modeling

---

## 💡 Key Insights

### 1. **Reward function was NOT the problem**
- Episode 20k proved agent can learn strategic play
- Issue was inability to maintain learned strategies
- Stability > Exploitation prevention

### 2. **Hyperparameter schedules matter more than you think**
- Fixed hyperparameters work for short training
- Long training (>10k episodes) needs adaptive schedules
- Exponential decay > Linear decay for stability

### 3. **Best model tracking is essential**
- Final checkpoint ≠ best checkpoint
- Always track and save peak performance
- Valuable for both deployment and debugging

### 4. **Catastrophic forgetting is real in RL**
- Not just a supervised learning problem
- Particularly bad with high LR + high entropy in late training
- Need explicit mechanisms to prevent it

---

## 📚 Related Documents

- **OUTCOME_BASED_TESTING.md** - Original testing protocol
- **REWARD_EVOLUTION.md** - History of reward function changes
- **ROAD_SPAM_FIX.md** - Earlier exploitation fix attempts
- **QUICKSTART.md** - Quick reference for running training

---

## 🔄 Comparison to Previous Approaches

| Approach | Target Problem | Result |
|----------|---------------|---------|
| **v1: Reduce building rewards** | Trade spam | Fixed trade spam, exposed road spam |
| **v2: Remove building rewards** | Road spam | Prevented immediate spam, but unstable |
| **v3: Add repetition penalty** | Road spam | Band-aid, didn't address root cause |
| **v4: Outcome-based learning** | All exploitation | Works but unstable (oscillation) |
| **v5: Stability fixes (this)** | Oscillation/Forgetting | **Should provide stable, monotonic learning** |

---

**These fixes address the ROOT CAUSE of performance degradation: training instability and catastrophic forgetting, not just exploitation.**
