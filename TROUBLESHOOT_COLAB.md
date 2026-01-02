# Training Not Starting? Troubleshoot Here!

## Quick Diagnosis

**Run this in Colab to see what's wrong:**

```python
%cd /content/drive/MyDrive/catantron

# Download and run diagnostic
!wget -q -O diagnose_training.py https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/diagnose_training.py
!python diagnose_training.py
```

This will tell you exactly what's missing or broken.

---

## Common Issues & Fixes

### Issue 1: "Nothing happens" when running training

**Symptoms:** Script runs but no output appears

**Cause:** Usually stuck on imports or missing dependencies

**Fix:**
```python
# Check if it's actually running
import time
import subprocess

# Run with timeout to see if it hangs
proc = subprocess.Popen(
    ['python', 'curriculum_trainer_v2.py', '--games-per-phase', '10'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait 30 seconds
try:
    stdout, stderr = proc.communicate(timeout=30)
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
except subprocess.TimeoutExpired:
    print("⚠️  Script is stuck! Likely hanging on imports.")
    proc.kill()
```

---

### Issue 2: Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Fix:**
```python
# Install all dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy gymnasium
```

Then try again.

---

### Issue 3: Files Not Downloaded

**Symptoms:** `FileNotFoundError` or `ModuleNotFoundError` for local files

**Fix:** Download all required files:
```python
%cd /content/drive/MyDrive/catantron

# Download ALL fixed files
files = [
    'curriculum_trainer_v2.py',
    'catan_env_pytorch.py',
    'mcts.py',
    'alphazero_trainer.py',
    'network_wrapper.py',
]

base_url = "https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ"

for f in files:
    !wget -q -O {f} {base_url}/{f}
    print(f"✅ Downloaded {f}")
```

---

### Issue 4: Import Errors from Old Files

**Error:** Various import errors or AttributeErrors

**Cause:** Old code still cached

**Fix:**
```python
# Restart Python kernel
import os
os._exit(0)  # Forces kernel restart

# Then in new cell:
%cd /content/drive/MyDrive/catantron
!python curriculum_trainer_v2.py --games-per-phase 10
```

---

### Issue 5: Script Starts But Crashes Immediately

**Symptoms:** Starts to run then exits with error

**Fix:** Run with error output:
```python
# See full error traceback
!python -u curriculum_trainer_v2.py --games-per-phase 10 2>&1

# Or in Python to catch errors
import sys
sys.path.insert(0, '/content/drive/MyDrive/catantron')

try:
    from curriculum_trainer_v2 import CurriculumTrainerV2
    trainer = CurriculumTrainerV2()
    trainer.train(games_per_phase=10)
except Exception as e:
    import traceback
    print("ERROR:", e)
    traceback.print_exc()
```

---

### Issue 6: Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Fix:** Reduce batch size:
```python
# Edit curriculum_trainer_v2.py or run manually:
from curriculum_trainer_v2 import CurriculumTrainerV2

# Smaller batch size for limited memory
trainer = CurriculumTrainerV2(batch_size=512)  # Down from 2048
trainer.train(games_per_phase=100)
```

---

## Step-by-Step Debug Process

### Step 1: Verify Setup
```python
# Check you're in right directory
!pwd
# Should show: /content/drive/MyDrive/catantron

# Check files exist
!ls -lh curriculum_trainer_v2.py catan_env_pytorch.py network_gpu.py
```

### Step 2: Test Imports
```python
# Test each import individually
import torch
print(f"✅ torch {torch.__version__}")

import numpy
print(f"✅ numpy {numpy.__version__}")

from curriculum_trainer_v2 import CurriculumTrainerV2
print(f"✅ curriculum_trainer_v2 imports")
```

### Step 3: Test Environment
```python
from catan_env_pytorch import CatanEnv

env = CatanEnv()
print("✅ Environment created")

obs, info = env.reset()
print(f"✅ Reset works, obs shape: {obs['observation'].shape}")
```

### Step 4: Test Trainer
```python
from curriculum_trainer_v2 import CurriculumTrainerV2

print("Creating trainer...")
trainer = CurriculumTrainerV2(batch_size=32)
print(f"✅ Trainer created on {trainer.device}")
```

### Step 5: Run Mini Training
```python
# Very short test run
from curriculum_trainer_v2 import CurriculumTrainerV2

trainer = CurriculumTrainerV2(batch_size=32)
trainer.train(games_per_phase=5)  # Just 5 games per phase for testing
```

---

## Working Example (Copy This)

This should definitely work:

```python
# === COMPLETE WORKING SETUP ===
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/catantron

# Download diagnostic
!wget -q -O diagnose_training.py https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ/diagnose_training.py

# Run diagnostic
!python diagnose_training.py

# If diagnostic passes, run mini test
!python curriculum_trainer_v2.py --games-per-phase 10

# If that works, run full training
# !python curriculum_trainer_v2.py --games-per-phase 300
```

---

## Expected Output When Working

You should see:

```
======================================================================
CURRICULUM TRAINING V2 (Using Full Reward System)
======================================================================
Device: cuda
Games per phase: 10
Batch size: 2048
======================================================================

======================================================================
PHASE 1: Random opponents
======================================================================
  Game   2/10 | WR:  50.0% | VP: 3.2 | Reward: 5.4 | Speed: 76.2 g/min
  Game   4/10 | WR:  50.0% | VP: 3.5 | Reward: 6.1 | Speed: 77.3 g/min
    └─ Train: policy=1.234, value=0.567, entropy=1.823
  Game   6/10 | WR:  50.0% | VP: 3.3 | Reward: 5.8 | Speed: 75.8 g/min
    └─ Train: policy=1.123, value=0.534, entropy=1.756
```

If you see this, it's working! 🎉

---

## Still Stuck?

If none of this helps:

1. **Share the output of:**
   ```python
   !python diagnose_training.py
   ```

2. **Share any error messages**

3. **Check you downloaded the fixes:**
   ```python
   !grep "Standardize returns" curriculum_trainer_v2.py
   # Should find the line
   ```

---

## Quick Reset (Nuclear Option)

If nothing works, start completely fresh:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive

# Remove old directory
!rm -rf catantron_old
!mv catantron catantron_old

# Clone fresh
!git clone https://github.com/TimeBuilder18/catantron.git
%cd catantron

# Download fixes manually
base = "https://raw.githubusercontent.com/TimeBuilder18/catantron/claude/review-catan-ai-PD3MQ"
!wget -q -O curriculum_trainer_v2.py {base}/curriculum_trainer_v2.py
!wget -q -O catan_env_pytorch.py {base}/catan_env_pytorch.py
!wget -q -O mcts.py {base}/mcts.py
!wget -q -O alphazero_trainer.py {base}/alphazero_trainer.py

# Test
!python curriculum_trainer_v2.py --games-per-phase 10
```

---

Good luck! 🚀
