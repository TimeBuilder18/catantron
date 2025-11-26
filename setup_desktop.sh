#!/bin/bash
# Desktop GPU Setup Script for Catan RL Training
# Run this on your Desktop to set up everything automatically

echo "======================================================================"
echo "🚀 Catan RL - Desktop GPU Setup"
echo "======================================================================"
echo ""

# Check if NVIDIA GPU exists
echo "Step 1: Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ NVIDIA GPU detected!"
else
    echo "⚠️  Warning: nvidia-smi not found. GPU training may not work."
    echo "   Install NVIDIA drivers: https://www.nvidia.com/drivers"
    read -p "   Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Step 2: Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo ""
echo "Step 3: Installing PyTorch with CUDA support..."
echo "   This may take a few minutes..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
if [ $? -eq 0 ]; then
    echo "✅ PyTorch installed"
else
    echo "❌ Failed to install PyTorch. Check your internet connection."
    exit 1
fi

echo ""
echo "Step 4: Installing other dependencies..."
pip3 install gymnasium numpy matplotlib --quiet
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies."
    exit 1
fi

echo ""
echo "Step 5: Verifying GPU access..."
python3 << EOF
import torch
if torch.cuda.is_available():
    #print(f"✅ CUDA available: {torch.cuda.is_available()}")
    #print(f"   GPU: {torch.cuda.get_device_name(0)}")
    #print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    #print("⚠️  CUDA not available. Training will use CPU (slow).")
    #print("   Check NVIDIA drivers and CUDA installation.")
EOF

echo ""
echo "Step 6: Creating directories..."
mkdir -p models
mkdir -p plots
echo "✅ Directories created"

echo ""
echo "======================================================================"
echo "✅ SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Transfer your code to this machine"
echo "2. Run a test: python3 train_gpu.py --episodes 10"
echo "3. Start real training: python3 train_gpu.py --episodes 5000"
echo ""
echo "Quick Test Command:"
echo "  python3 -c 'import torch; print(f\"GPU Ready: {torch.cuda.is_available()}\")'"
echo ""
echo "======================================================================"
