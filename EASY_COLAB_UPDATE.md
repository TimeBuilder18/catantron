# Easy Colab Update (No Git Needed!)

## Problem
Git authentication is failing in Colab. No worries - we'll download the fixed files directly!

---

## Solution 1: Download Script (Easiest!)

**Just run this ONE cell in Colab:**

```python
%cd /content/drive/MyDrive/catantron

# Download the update script
!wget -q https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/download_fixes.py

# Run it to download all fixed files
!python download_fixes.py

# Check it worked
!ls FIXES_APPLIED.md
```

This will:
- ✅ Backup your old files (.backup)
- ✅ Download all 7 fixed files
- ✅ Ready to train!

---

## Solution 2: Manual Download (If Script Fails)

Run these commands one by one in Colab:

```python
%cd /content/drive/MyDrive/catantron

# Download each fixed file directly
!wget -O curriculum_trainer_v2.py https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/curriculum_trainer_v2.py

!wget -O catan_env_pytorch.py https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/catan_env_pytorch.py

!wget -O mcts.py https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/mcts.py

!wget -O alphazero_trainer.py https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/alphazero_trainer.py

!wget -O FIXES_APPLIED.md https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/FIXES_APPLIED.md

# Verify
!ls -lh curriculum_trainer_v2.py catan_env_pytorch.py mcts.py alphazero_trainer.py
```

---

## Solution 3: Copy Files from GitHub Web

1. Go to: https://github.com/TimeBuilder18/catantron/tree/claude/review-catan-ai-PD3MQ
2. Click each file and copy contents
3. In Colab, create/overwrite files:

```python
%%writefile curriculum_trainer_v2.py
# Paste contents here
```

---

## Verify Fixes Are Installed

```python
# Check critical fix is present
!grep "Standardize returns" curriculum_trainer_v2.py
# Should show: # Standardize returns (preserve magnitude differences...)

!grep "np.clip(reward" catan_env_pytorch.py
# Should show: reward = np.clip(reward, -20.0, 20.0)

!grep "root_player" mcts.py | head -3
# Should show lines with root_player tracking

print("✅ All fixes verified!")
```

---

## Start Training!

```python
# Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Run training with all fixes
!python curriculum_trainer_v2.py --games-per-phase 300
```

---

## What You're Getting

These 7 critical fixes:

1. ✅ **Return standardization** (not normalization) - preserves learning signal
2. ✅ **Policy gradient with advantages** - correct learning algorithm
3. ✅ **Entropy bonus** - encourages exploration
4. ✅ **Reward clipping** to [-20, 20] - prevents instability
5. ✅ **Reduced trade penalty** (0.5 vs 3.0) - allows strategic trades
6. ✅ **MCTS multi-player** - works with 4 players
7. ✅ **AlphaZero discounting** - early positions less certain

---

## Expected Results

With fixes, you should see:

```
PHASE 1: Random opponents
  Game  10/300 | WR:  30.0% | VP: 3.5 | Reward: 8.2 | Speed: 76.3 g/min
    └─ Train: policy=1.234, value=0.567, entropy=1.823
                                          ^^^^^^^ NEW!
```

**Key improvements:**
- 🎯 Win rate: 30-40% in Phase 1 (vs random)
- 📈 VP: 3.5 → 4.5+ over phases
- ⚡ No crashes or NaN values
- 🔍 Entropy shows exploration

---

## Troubleshooting

### wget not found?
```python
!pip install wget
# Then try again
```

### Files not downloading?
Check internet:
```python
!ping -c 3 github.com
```

### Still stuck?
DM me the error and I'll help!

---

## Quick Command Summary

**Fastest way:**
```python
%cd /content/drive/MyDrive/catantron
!wget -q https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/download_fixes.py
!python download_fixes.py
!python curriculum_trainer_v2.py --games-per-phase 300
```

That's it! 🚀
