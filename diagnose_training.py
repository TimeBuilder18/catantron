"""
Diagnostic script to check what's preventing training from starting
"""

print("=" * 70)
print("DIAGNOSING TRAINING ISSUES")
print("=" * 70)

# Test 1: Check Python version
print("\n[1] Python Version")
import sys
print(f"    Python {sys.version}")

# Test 2: Check critical imports
print("\n[2] Checking Imports...")
imports_ok = True

try:
    import torch
    print(f"    ✅ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"    ❌ PyTorch missing: {e}")
    imports_ok = False

try:
    import numpy
    print(f"    ✅ NumPy {numpy.__version__}")
except ImportError as e:
    print(f"    ❌ NumPy missing: {e}")
    imports_ok = False

try:
    import gymnasium
    print(f"    ✅ Gymnasium {gymnasium.__version__}")
except ImportError as e:
    print(f"    ❌ Gymnasium missing: {e}")
    imports_ok = False

# Test 3: Check files exist
print("\n[3] Checking Files...")
import os

required_files = [
    'curriculum_trainer_v2.py',
    'catan_env_pytorch.py',
    'network_gpu.py',
    'game_system.py',
    'rule_based_ai.py',
    'ai_interface.py',
]

files_ok = True
for f in required_files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"    ✅ {f} ({size:,} bytes)")
    else:
        print(f"    ❌ {f} MISSING!")
        files_ok = False

# Test 4: Check GPU
print("\n[4] Checking GPU...")
if imports_ok:
    import torch
    if torch.cuda.is_available():
        print(f"    ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print(f"    ✅ MPS available (Apple Silicon)")
    else:
        print(f"    ⚠️  No GPU - will use CPU (slow!)")

# Test 5: Try importing curriculum trainer
print("\n[5] Testing Import of curriculum_trainer_v2...")
try:
    import curriculum_trainer_v2
    print(f"    ✅ curriculum_trainer_v2.py imports successfully")
except Exception as e:
    print(f"    ❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    files_ok = False

# Test 6: Try creating environment
print("\n[6] Testing Environment Creation...")
try:
    from catan_env_pytorch import CatanEnv
    env = CatanEnv()
    print(f"    ✅ CatanEnv created successfully")
    obs, info = env.reset()
    print(f"    ✅ Environment reset works")
    print(f"    Observation shape: {obs['observation'].shape}")
except Exception as e:
    print(f"    ❌ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    files_ok = False

# Test 7: Try creating trainer
print("\n[7] Testing Trainer Creation...")
try:
    from curriculum_trainer_v2 import CurriculumTrainerV2
    print(f"    Creating trainer (this may take a few seconds)...")
    trainer = CurriculumTrainerV2(batch_size=32)
    print(f"    ✅ CurriculumTrainerV2 created successfully")
    print(f"    Device: {trainer.device}")
    print(f"    Batch size: {trainer.batch_size}")
except Exception as e:
    print(f"    ❌ Trainer creation failed: {e}")
    import traceback
    traceback.print_exc()
    files_ok = False

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if imports_ok and files_ok:
    print("✅ All checks passed!")
    print("\nYour environment is ready. Training should work.")
    print("\nTo start training, run:")
    print("  python curriculum_trainer_v2.py --games-per-phase 50")
    print("\nTip: Start with --games-per-phase 50 for a quick test")
else:
    print("❌ Some checks failed.")
    if not imports_ok:
        print("\n⚠️  MISSING DEPENDENCIES")
        print("Run this to install:")
        print("  pip install torch numpy gymnasium")
    if not files_ok:
        print("\n⚠️  MISSING FILES")
        print("Make sure you downloaded all fixed files!")
        print("See EASY_COLAB_UPDATE.md for instructions")

print("\n" + "=" * 70)
