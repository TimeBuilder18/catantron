# Quick Start - Outcome-Based Learning Testing

**Status:** ✅ Code ready to test (Commit: 92630a5)
**Goal:** Validate exploitation-proof reward function

---

## 🚀 TL;DR - Run This Now

### Option 1: Quick Test (~2-3 hours)
```bash
python train_clean.py --episodes 15000 --model-name outcome_test_quick \
  --curriculum --batch-size 1024 --epochs 20 --update-freq 40 --save-freq 3000
```

### Option 2: Full Test (~5-6 hours, RECOMMENDED)
```bash
python train_clean.py --episodes 25000 --model-name outcome_test_extended \
  --curriculum --batch-size 1024 --epochs 20 --update-freq 40 --save-freq 5000
```

Then analyze:
```bash
python analyze_model_performance.py --model-pattern "models/outcome_test_*.pt" \
  --eval-episodes 30 --vp-target 4 --max-checkpoints 6
```

---

## ✅ What to Look For (Success)

During training:
- ✅ VP steadily increases: 2.3 → 2.7-2.9
- ✅ NO late-stage collapse after episode 18k
- ✅ Natural endings stay > 70%

In analyzer:
- ✅ Roads/game: 10-20 (realistic, NOT 100+)
- ✅ Cities/game: 0.5-1.0 (learning complex strategies)
- ✅ Top action < 40% of total (diverse gameplay)
- ✅ No single action spam pattern

---

## ❌ What to Watch For (Failure)

During training:
- ❌ VP plateaus below 2.5
- ❌ Performance crashes after episode 15k
- ❌ Timeouts > 60%

In analyzer:
- ❌ Roads/game > 50 (still spamming!)
- ❌ Any action > 60% of total
- ❌ Cities/game < 0.3 (not learning)

---

## 📊 What Changed

**Old (Exploitable):**
- Building rewards: 0.05/0.1/0.02 → Agent spammed roads (265/game!)
- Illegal action penalty: -10.0 → Agent thought spam was worth it

**New (Outcome-Based):**
- Building rewards: **0.0 (REMOVED)** → No incentive to spam
- Illegal action penalty: **0.0 (REMOVED)** → No perverse incentive
- PBRS: **10x boost** (0.1 → 1.0) → Rewards quality strategies
- VP changes: +3.0 → Main outcome signal

**Theory:** Agent can ONLY learn from outcomes (VP) and strategic quality (PBRS), making exploitation impossible.

---

## 🎯 Expected Results

| Metric | Previous Runs | Expected (Outcome-Based) |
|--------|---------------|--------------------------|
| Peak VP | 2.73 → 2.19 (crashed) | 2.8-3.0 (stable) |
| Roads/game | 265 💥 | 10-20 (realistic) |
| Cities/game | 0.1-0.2 | 0.5-1.0 |
| Natural endings | 100% → 47% | 70-80% sustained |
| Late-stage | ❌ Catastrophic collapse | ✅ Stable |

---

## 📁 Documentation

Full details in:
- **OUTCOME_BASED_TESTING.md** - Complete testing protocol
- **REWARD_EVOLUTION.md** - History of all reward function changes
- **ROAD_SPAM_FIX.md** - Analysis of previous exploitation issues

---

## 🔄 Quick Decision Tree

After training completes:

1. **Check final VP:**
   - VP ≥ 2.7 → ✅ Proceed to step 2
   - VP < 2.5 → ❌ Agent not learning, see OUTCOME_BASED_TESTING.md warning signs

2. **Run analyzer:**
   - Roads/game < 25 → ✅ No exploitation, proceed to step 3
   - Roads/game > 50 → ❌ Still exploiting, need deeper fixes

3. **Check late-stage stability:**
   - VP stable from ep 18k-25k → ✅ **SUCCESS!**
   - VP crashed after ep 18k → ❌ Stability issue, adjust LR/entropy

4. **If all success:**
   - Document results in OUTCOME_TEST_RESULTS.md
   - Run longer training (50k episodes)
   - Advance to curriculum stage 2

5. **If any failure:**
   - See OUTCOME_BASED_TESTING.md "Warning Signs" section
   - Consider MULTI_ACTION_ANALYSIS.md approach

---

## 💻 Environment Setup (If Needed)

If you get "No module named 'torch'" error:

```bash
# Install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install gymnasium numpy matplotlib

# Verify
python3 -c 'import torch; print(f"GPU Ready: {torch.cuda.is_available()}")'
```

Or use the automated setup:
```bash
bash setup_desktop.sh
```

---

## ⏱️ Time Estimates

| Training | Episodes | GPU Time | CPU Time |
|----------|----------|----------|----------|
| Quick test | 15k | 2-3 hours | 8-12 hours |
| Full test | 25k | 5-6 hours | 15-20 hours |
| Extended | 50k | 10-12 hours | 30-40 hours |

Analysis: ~5-15 minutes

---

## 🎓 What This Tests

1. **Can the agent learn WITHOUT building rewards?**
   - Previous: Even 0.02 reward × volume = exploitation
   - Now: 0.0 reward = no spam incentive

2. **Is PBRS (10x boost) enough to guide strategy?**
   - Previous: PBRS too weak (0.1x)
   - Now: PBRS is primary signal (1.0x)

3. **Will economic cost prevent spam?**
   - Theory: Wasting resources → can't build → lose
   - Test: Does agent learn this without explicit penalties?

4. **Is late-stage training stable?**
   - Previous: Crashed at episodes 18k-22k
   - Now: Should maintain performance

---

## 📞 Quick Reference

**What am I testing?**
Exploitation-proof reward function

**How long will it take?**
5-6 hours for full test (25k episodes)

**What does success look like?**
VP 2.7-2.9, roads 10-20/game, no late-stage collapse

**What if it fails?**
See OUTCOME_BASED_TESTING.md warning signs section

**Where are the results?**
models/outcome_test_*.pt (checkpoints)
Analyze with analyze_model_performance.py

---

**This is the final test of the anti-exploitation approach. If this works, we have a stable foundation. If not, we need architectural changes (multi-action RL, hierarchical RL).**
