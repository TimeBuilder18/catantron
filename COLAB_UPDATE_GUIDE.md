# How to Update Colab with Training Fixes

## Quick Update (Copy-Paste into Colab)

### Method 1: Using the Update Script (Recommended)

```bash
# Run this in a Colab code cell
!cd /content/catantron && bash update_colab.sh
```

That's it! The script will:
- Fetch the latest changes
- Switch to the fixed training branch
- Pull all updates
- Show you what's new

---

### Method 2: Manual Git Commands

If the script doesn't work, use these commands:

```bash
# In Colab code cell:
!cd /content/catantron && \
  git stash && \
  git fetch origin claude/review-catan-ai-PD3MQ && \
  git checkout claude/review-catan-ai-PD3MQ && \
  git pull origin claude/review-catan-ai-PD3MQ
```

---

### Method 3: Fresh Clone (If You're Starting Fresh)

```bash
# Remove old version (if it exists)
!rm -rf /content/catantron

# Clone the fixed version
!git clone -b claude/review-catan-ai-PD3MQ \
  https://github.com/TimeBuilder18/catantron.git /content/catantron

# Navigate to the directory
%cd /content/catantron
```

---

## Verify the Update

Run this to confirm you have the new files:

```python
import os
os.chdir('/content/catantron')

# Check for new files
files_to_check = [
    'curriculum_trainer_v2_fixed.py',
    'TRAINING_FIXES_ANALYSIS.md',
    'update_colab.sh'
]

print("✅ Checking for updated files:\n")
for file in files_to_check:
    exists = "✅ Found" if os.path.exists(file) else "❌ Missing"
    print(f"{exists}: {file}")

print("\n" + "="*50)
print("If all files are found, you're ready to train!")
print("="*50)
```

---

## Run the Fixed Trainer

### Quick Test (100 games per phase)
```bash
!python curriculum_trainer_v2_fixed.py --games-per-phase 100
```

### Full Training (1000 games per phase, ~90 minutes)
```bash
!python curriculum_trainer_v2_fixed.py --games-per-phase 1000
```

### Monitor Training

The output should show:
- ✅ Win rate increasing (especially in Phase 1)
- ✅ Entropy staying above 0.5 (not collapsing to 0.0003)
- ✅ VP scores reaching 4-6+ (not stuck at 2.0)
- ✅ Stable losses (no spikes to 57.27)

**Phase 1 expectations:**
- Old version: ~0.2% win rate, 2.3 VP
- Fixed version: ~15-25% win rate, 4-5 VP

---

## Read the Full Analysis

```bash
!cat TRAINING_FIXES_ANALYSIS.md
```

Or read specific sections:

```python
# Show just the summary
!head -n 50 TRAINING_FIXES_ANALYSIS.md

# Show the critical issues
!grep -A 10 "^### Issue #" TRAINING_FIXES_ANALYSIS.md
```

---

## Complete Colab Setup (From Scratch)

If you need to set up everything from scratch, here's the full sequence:

```bash
# 1. Install dependencies
!pip install torch numpy

# 2. Clone the repo with fixes
!git clone -b claude/review-catan-ai-PD3MQ \
  https://github.com/TimeBuilder18/catantron.git /content/catantron

# 3. Navigate to directory
%cd /content/catantron

# 4. Verify GPU (optional but recommended)
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 5. Run training
!python curriculum_trainer_v2_fixed.py --games-per-phase 1000
```

---

## Troubleshooting

### Error: "Not in a git repository"

**Solution**: Navigate to the catantron directory first:
```bash
%cd /content/catantron
!bash update_colab.sh
```

### Error: "branch 'claude/review-catan-ai-PD3MQ' not found"

**Solution**: Fetch all branches first:
```bash
!cd /content/catantron && \
  git fetch --all && \
  git checkout claude/review-catan-ai-PD3MQ
```

### Error: Module not found (e.g., 'catan_env_pytorch')

**Solution**: Make sure you're in the catantron directory:
```bash
%cd /content/catantron
!python curriculum_trainer_v2_fixed.py --games-per-phase 100
```

### Files are there but can't import

**Solution**: Add to Python path:
```python
import sys
sys.path.insert(0, '/content/catantron')

# Now you can import
from curriculum_trainer_v2_fixed import CurriculumTrainerV2
```

### Training is slow

**Check**: Make sure GPU is being used:
```python
import torch
print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
```

If it says 'cpu', go to `Runtime` → `Change runtime type` → Select `GPU` (T4 or A100)

---

## Expected Training Time

With GPU (A100):
- 100 games/phase (5 phases) = ~10 minutes
- 1000 games/phase (5 phases) = ~90 minutes
- Batch size: 2048
- Speed: ~60 games/min

Without GPU (CPU only):
- Will be much slower (~5-10 games/min)
- Consider reducing batch size to 256

---

## What's Different in the Fixed Version?

The fixed trainer (`curriculum_trainer_v2_fixed.py`) has:

1. **No return normalization** - preserves learning signal
2. **Higher entropy coefficient** - 0.1 instead of 0.01
3. **Lower value loss weight** - 0.1 instead of 0.5
4. **Better random opponent** - plays 90% instead of 60%
5. **More training** - 20 steps every 5 games (4x total)

See `TRAINING_FIXES_ANALYSIS.md` for full technical details.

---

## Quick Reference

| Task | Command |
|------|---------|
| Update code | `!cd /content/catantron && bash update_colab.sh` |
| Quick test | `!python curriculum_trainer_v2_fixed.py --games-per-phase 100` |
| Full training | `!python curriculum_trainer_v2_fixed.py --games-per-phase 1000` |
| Read analysis | `!cat TRAINING_FIXES_ANALYSIS.md` |
| Check GPU | `import torch; print(torch.cuda.is_available())` |

---

## Next Steps After Training

1. Check the saved models in `models/` directory
2. Test the trained model against rule-based AI
3. Compare with the old broken trainer's results
4. If still not learning well, increase games-per-phase to 2000

Good luck! 🎲🤖
