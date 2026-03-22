# Fix Git Pull in Google Colab

## The Problem
You're getting "Host key verification failed" because Git is trying to use SSH, but Colab doesn't have your SSH keys.

## Quick Fix (In Colab)

Run these commands in your Colab notebook:

```python
%cd /content/drive/MyDrive/catantron

# Check current remote (probably shows git@github.com - that's SSH)
!git remote -v

# Switch to HTTPS
!git remote set-url origin https://github.com/TimeBuilder18/catantron.git

# Verify it changed
!git remote -v
# Should now show: https://github.com/TimeBuilder18/catantron.git

# Now pull the fixes from your branch
!git fetch origin claude/review-catan-ai-PD3MQ
!git checkout claude/review-catan-ai-PD3MQ
!git pull origin claude/review-catan-ai-PD3MQ

# Verify you got the fixes
!git log --oneline -3
# Should show recent commits with "Fix critical bugs..."
```

---

## Alternative: Fresh Clone (Easier!)

If the above doesn't work, just clone fresh with HTTPS:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive

# Backup old directory (optional)
!mv catantron catantron_backup_$(date +%Y%m%d)

# Clone with HTTPS (NOT SSH!)
!git clone https://github.com/TimeBuilder18/catantron.git

%cd catantron

# Switch to the branch with fixes
!git checkout claude/review-catan-ai-PD3MQ

# Verify
!ls FIXES_APPLIED.md  # Should exist
```

---

## Verify Fixes Are Applied

```python
# Check critical files exist
!ls -la FIXES_APPLIED.md UPDATE_COLAB.md

# Check a specific fix
!grep -A 2 "Standardize returns" curriculum_trainer_v2.py | head -5

# Should show:
#   # Standardize returns (preserve magnitude differences between games)
#   returns = np.array(returns)
#   if len(returns) > 1:

print("✅ Fixes are installed!")
```

---

## Start Training!

```python
# Check GPU
import torch
print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Run training with all fixes
!python curriculum_trainer_v2.py --games-per-phase 300
```

---

## What Changed

Your updated files now have:

1. ✅ **Better learning algorithm** (policy gradient with advantages)
2. ✅ **Exploration** (entropy bonus)
3. ✅ **Stability** (reward clipping, return standardization)
4. ✅ **Balanced penalties** (trade penalty reduced)
5. ✅ **Multi-player MCTS** (4-player support)
6. ✅ **Better AlphaZero** (discounted values)

---

## Expected Output

You should see:

```
======================================================================
CURRICULUM TRAINING V2 (Using Full Reward System)
======================================================================
Device: cuda
Games per phase: 300
Batch size: 2048
======================================================================

======================================================================
PHASE 1: Random opponents
======================================================================
  Game  10/300 | WR:  30.0% | VP: 3.5 | Reward: 8.2 | Speed: 76.3 g/min
    └─ Train: policy=1.2345, value=0.5678, entropy=1.8234
  Game  20/300 | WR:  35.0% | VP: 3.8 | Reward: 9.1 | Speed: 77.1 g/min
    └─ Train: policy=1.1234, value=0.4567, entropy=1.7123
```

Key signs it's working:
- **Entropy** shows up in training output
- **Win rate increases** through phases
- **VP climbs** (3.0 → 4.0 → 5.0)
- **No crashes** or errors

---

## Troubleshooting

### Still getting SSH error?
```python
# Make absolutely sure remote is HTTPS
!git remote remove origin
!git remote add origin https://github.com/TimeBuilder18/catantron.git
!git fetch origin
!git checkout claude/review-catan-ai-PD3MQ
```

### "Already on master"?
```python
# You're on the wrong branch, switch to the fixed branch
!git fetch origin
!git checkout claude/review-catan-ai-PD3MQ
!git pull origin claude/review-catan-ai-PD3MQ
```

### Want to check what branch you're on?
```python
!git branch -a
# The one with * is your current branch
# You want: * claude/review-catan-ai-PD3MQ
```

---

## Summary

**Problem:** SSH doesn't work in Colab
**Solution:** Use HTTPS instead
**Command:** `git remote set-url origin https://github.com/TimeBuilder18/catantron.git`

Then pull the fixed branch:
```python
!git checkout claude/review-catan-ai-PD3MQ
!git pull
```

Done! 🚀
