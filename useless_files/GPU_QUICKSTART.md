# 🎮 GPU-Ready Catan RL Training - Quick Reference

## 📁 Files Created

| File | Purpose | Use On |
|------|---------|--------|
| `network_gpu.py` | GPU-ready neural network | Mac + Desktop |
| `agent_gpu.py` | GPU-ready agent | Mac + Desktop |
| `trainer_gpu.py` | GPU-ready PPO trainer | Mac + Desktop |
| `train_gpu.py` | Main training script | Mac + Desktop |
| `benchmark_speed.py` | CPU vs GPU speed test | Desktop |
| `setup_desktop.sh` | Auto-setup script | Desktop |
| `TRANSFER_WORKFLOW.md` | Complete workflow guide | Reference |

---

## 🚀 Quick Start

### On Desktop (First Time)

```bash
# 1. Setup environment
chmod +x setup_desktop.sh
./setup_desktop.sh

# 2. Test GPU works
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# 3. Run speed benchmark
python3 benchmark_speed.py

# 4. Quick test (10 episodes)
python3 train_gpu.py --episodes 10 --model-name test

# 5. Full training!
python3 train_gpu.py --episodes 5000 --model-name catan_v1
```

---

## 💡 Key Features

### Auto Device Detection
```python
# Automatically uses GPU if available, CPU otherwise
agent = CatanAgent()  # Auto-detects
trainer = PPOTrainer(policy=agent.policy)  # Uses same device
```

### Larger Batch Size on GPU
- **CPU**: 64 samples per batch
- **GPU**: 256 samples per batch (4x faster updates!)

### Model Save/Load
```python
# Save model (works on any device)
agent.policy.save('models/my_model.pt')

# Load model (automatically handles device transfer)
agent.policy.load('models/my_model.pt', device='cpu')  # For Mac
agent.policy.load('models/my_model.pt', device='cuda')  # For Desktop
```

---

## ⚡ Expected Performance

### Speed Comparison (RTX 2080 Super)

| Task | CPU (Mac) | GPU (Desktop) | Speedup |
|------|-----------|---------------|---------|
| 10 episodes | ~2 min | ~5 sec | ~24x |
| 100 episodes | ~20 min | ~50 sec | ~24x |
| 1000 episodes | ~3.3 hours | ~8 min | ~25x |
| 5000 episodes | ~16 hours | ~40 min | ~24x |

*Your RTX 2080 Super should give 20-30x speedup!*

---

## 📊 Training Commands

### Quick Test (Development)
```bash
# On Mac (CPU) - test your changes
python3 train_gpu.py --episodes 10 --device cpu

# On Desktop (GPU) - verify it works
python3 train_gpu.py --episodes 50
```

### Real Training Sessions

```bash
# Short training (1-2 hours on GPU)
python3 train_gpu.py --episodes 2000 --update-freq 20 --save-freq 200

# Full training (2-3 hours on GPU)
python3 train_gpu.py --episodes 5000 --update-freq 20 --save-freq 500

# Overnight training (6-8 hours on GPU)
python3 train_gpu.py --episodes 20000 --update-freq 50 --save-freq 2000
```

---

## 🔧 Troubleshooting

### GPU Not Detected

**Check NVIDIA drivers:**
```bash
nvidia-smi
```

**Check PyTorch can see GPU:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

**Reinstall PyTorch with CUDA:**
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

---

### Out of Memory Error

**Reduce batch size in `trainer_gpu.py`:**
```python
trainer = PPOTrainer(
    policy=agent.policy,
    batch_size=128  # Reduce from 256
)
```

Or in `train_gpu.py`:
```python
batch_size = 128 if device.type == 'cuda' else 64
```

---

### Model Won't Load on Mac

**Always specify device when loading:**
```python
# On Mac (CPU)
policy.load('models/trained_on_gpu.pt', device='cpu')

# On Desktop (GPU)
policy.load('models/trained_on_gpu.pt', device='cuda')
```

Our code handles this automatically!

---

## 📈 Monitoring Training

### Terminal Output
```
Episode 100/5000 | Reward: 28.45 | VP: 3.2 | Length: 145 | Time: 0.8s | Speed: 75.0 eps/min
   GPU Memory: 1.23GB allocated, 2.00GB cached

📊 Updating policy (buffer size: 1450)...
   Policy loss: -0.0045
   Value loss: 2.8934
   Entropy: 1.1823
   Update time: 2.34s
```

### Watch GPU Usage
```bash
# On Desktop, in another terminal
watch -n 1 nvidia-smi
```

---

## 📦 What Gets Synced

### Via Git (Recommended)

**Always sync:**
- ✅ All `.py` files
- ✅ `models/*.pt` (use Git LFS for large files)
- ✅ Training plots (`*.png`)
- ✅ Documentation

**Never sync:**
- ❌ `__pycache__/`
- ❌ `.pyc` files
- ❌ Large datasets
- ❌ Temporary files

### `.gitignore` contents:
```
__pycache__/
*.pyc
.DS_Store
*.log
```

---

## 🎯 Development Workflow

### Day 1: Initial Training
```
Mac: Code + test (10 eps) → Git push
Desktop: Git pull → Train (5000 eps) → Git push models
Mac: Git pull → Analyze results
```

### Day 2: Improvements
```
Mac: Improve code + test → Git push
Desktop: Git pull → Train new version → Git push
Mac: Compare v1 vs v2
```

### Day 3: Fine-tuning
```
Mac: Tune hyperparameters → Git push
Desktop: Train multiple versions in parallel
Mac: Select best model → Continue
```

---

## 🏆 Training Goals

### Short-term (Episodes 0-1000)
- **Goal**: Learn basic gameplay
- **Expected VP**: 2-4 points
- **Time on GPU**: ~20 minutes
- **Success**: Agent can place settlements and roads

### Mid-term (Episodes 1000-5000)
- **Goal**: Strategic building
- **Expected VP**: 4-7 points
- **Time on GPU**: ~1.5 hours
- **Success**: Agent competes with rule-based AI

### Long-term (Episodes 5000-20000)
- **Goal**: Master-level play
- **Expected VP**: 7-10 points
- **Time on GPU**: ~6-8 hours
- **Success**: Agent wins games consistently

---

## 🔗 Useful Links

**Documentation:**
- Full workflow: `TRANSFER_WORKFLOW.md`
- Game features: `/mnt/project/GAME_FEATURES.md`
- Project README: `/mnt/project/README.md`

**Tools:**
- PyTorch CUDA: https://pytorch.org/get-started/locally/
- NVIDIA Drivers: https://www.nvidia.com/drivers
- Git LFS: https://git-lfs.github.com/

---

## 💪 Next Steps

1. ✅ Copy all GPU files to your Desktop
2. ✅ Run `setup_desktop.sh` on Desktop
3. ✅ Run `benchmark_speed.py` to see speedup
4. ✅ Start training: `python3 train_gpu.py --episodes 5000`
5. ✅ Monitor progress and analyze results
6. ✅ Iterate and improve!

---

**Questions?** Check `TRANSFER_WORKFLOW.md` for detailed instructions!

**Ready to train? Let's go! 🚀**
