# Catan RL Project Analysis - Quick Start

## 🚨 CRITICAL FINDINGS

Your model is experiencing **catastrophic performance degradation** after ~50,000 episodes due to:

1. **Unstable reward function** with conflicting signals
2. **Curriculum learning** clearing history and causing forgetting
3. **No learning rate decay** leading to late-training instability
4. **Non-functional model analyzer** (fixed)

## 📄 Documents Created

1. **PROJECT_ANALYSIS.md** - Comprehensive analysis of all issues (main document, read this!)
2. **FIXES_TO_APPLY.md** - Step-by-step code changes to fix critical issues
3. **analyze_model_performance.py** - New working model analyzer (replaces old one)
4. **README_ANALYSIS.md** - This file (quick start guide)

## 🔧 Quick Fix Guide

### Step 1: Apply Critical Fixes (30 minutes)

Apply changes from `FIXES_TO_APPLY.md` in this order:

1. **Fix reward function** in `catan_env_pytorch.py`:
   - Reduce hoarding penalty (lines 444-452)
   - Reduce VP reward multiplier (line 429)
   - Reduce inaction penalty (line 410)
   - Reduce win bonus (line 455)
   - Cap opponent threat penalty (line 385)

2. **Fix curriculum learning** in `train_clean.py`:
   - Don't clear VP window (line 161)
   - Add learning rate scheduler
   - Add entropy decay

### Step 2: Test Changes (3 hours)

```bash
# Short test run (3 hours on GPU)
python train_clean.py --episodes 20000 --model-name test_fixes --curriculum --save-freq 2000

# Verify no regression after 10k-20k episodes
```

### Step 3: Analyze Results

```bash
# Run new analyzer
python analyze_model_performance.py --model-pattern "models/test_fixes*.pt" --eval-episodes 30

# Check plots in analysis_results/performance_analysis.png
```

### Step 4: If Successful, Train Longer

```bash
# Full overnight run
python train_clean.py --episodes 100000 --model-name fixed_model --curriculum --save-freq 10000
```

## 📊 What to Look For

### Good Signs ✅
- VP steadily increases to 5-7 average (for 10 VP target)
- No performance drop after 30k episodes
- Natural game endings > 50%
- Cities built increases over time

### Bad Signs ❌
- VP drops after initial increase
- High percentage of timeouts (>60%)
- "do_nothing" action > 30% of all actions
- Value predictions all near 0

## 🔍 Key Issues Explained

### Issue 1: Hoarding Penalty Too Harsh

**Problem:** Agent penalized for holding 8+ cards, but building a city requires 6 cards. This creates a paradox:
- Agent wants to build cities (high reward)
- But gets penalized for accumulating resources needed to build cities
- Result: Agent learns to avoid resource accumulation

**Fix:** Only penalize 11+ cards, reduce penalty strength

### Issue 2: Curriculum Window Clearing

**Problem:** When advancing from 5→6 VP target, code clears the VP history window. This causes:
- Sudden policy shift
- Forgetting of successful strategies
- Performance regression

**Fix:** Keep history, track progress as percentage of target

### Issue 3: Static Learning Rate

**Problem:** Learning rate stays at 3e-4 for entire 100k episodes. Early on this is good, but late in training:
- Large updates can undo good policies
- No fine-tuning phase
- Policy oscillates instead of converging

**Fix:** Decay learning rate from 3e-4 → 1e-5 over training

## 📈 Expected Improvements

| Metric | Before Fixes | After Fixes |
|--------|-------------|-------------|
| Peak VP | ~4.5 @ 40k eps | ~6.5 @ 80k eps |
| Final VP (100k eps) | ~3.0 (regressed) | ~6.0 (stable) |
| Natural endings | 30% | 60% |
| Cities per game | 0.3 | 1.5 |
| Timeouts | 70% | 40% |

## 🚀 Priority Actions

**TODAY:**
1. Read PROJECT_ANALYSIS.md (understand root causes)
2. Apply fixes from FIXES_TO_APPLY.md (Fix 1 & 2 only)
3. Run short test (5000 episodes)

**THIS WEEK:**
1. Verify fixes work (analyze checkpoints)
2. Train full model with fixes
3. Add training diagnostics (Fix 5 & 6)

**NEXT WEEK:**
1. Improve network architecture (Fix 4 - layer norm)
2. Consider self-play vs rule-based opponents
3. Experiment with different hyperparameters

## 📞 Quick Reference

**Files to modify (Priority 1):**
- `catan_env_pytorch.py` - Reward function fixes
- `train_clean.py` - Curriculum learning and LR scheduling

**Files to modify (Priority 2):**
- `trainer_gpu.py` - Value clipping
- `network_gpu.py` - Layer normalization

**New tools:**
- `analyze_model_performance.py` - Working model analyzer
- `evaluate_model.py` - Already exists, still useful

## ⚠️ Common Pitfalls

1. **Don't train for 100k episodes before fixing issues** - you'll waste time
2. **Don't add all fixes at once** - apply incrementally and test
3. **Don't skip the test run** - verify fixes work on short run first
4. **Don't ignore the plots** - visual inspection catches issues early

## 💬 Questions?

Check PROJECT_ANALYSIS.md for detailed explanations of:
- Why each issue occurs
- Why each fix works
- What metrics to monitor
- How to debug further

---

**Last Updated:** Analysis run on current codebase
**Estimated Fix Time:** 30-60 minutes to apply critical fixes
**Estimated Test Time:** 2-3 hours for verification run
**Expected Training Time:** 8-12 hours for full 100k episode run (with fixes)
