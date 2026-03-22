#!/bin/bash
# Update Colab with Fixed Training Code
# Run this in Colab to get the latest training fixes

set -e  # Exit on error

echo "=============================================="
echo "Updating Catan AI with Training Fixes"
echo "=============================================="
echo ""

# Check if we're in the catantron directory
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    echo "Please run this from the catantron directory"
    exit 1
fi

# Save any local changes
echo "📦 Stashing any local changes..."
git stash

# Fetch latest changes
echo "📥 Fetching latest changes..."
git fetch origin claude/review-catan-ai-PD3MQ

# Switch to the branch with fixes
echo "🔄 Switching to fixed training branch..."
git checkout claude/review-catan-ai-PD3MQ

# Pull latest changes
echo "⬇️  Pulling latest changes..."
git pull origin claude/review-catan-ai-PD3MQ

echo ""
echo "=============================================="
echo "✅ Update Complete!"
echo "=============================================="
echo ""
echo "New files available:"
echo "  • curriculum_trainer_v2_fixed.py"
echo "  • TRAINING_FIXES_ANALYSIS.md"
echo ""
echo "Key fixes:"
echo "  1. Removed return normalization (enables learning!)"
echo "  2. Increased entropy coefficient (prevents collapse)"
echo "  3. Reduced value loss weight (stabilizes training)"
echo "  4. Fixed random opponent (easier curriculum)"
echo "  5. 4x more training steps"
echo ""
echo "Expected: 0.4% → 8-12% win rate"
echo ""
echo "=============================================="
echo "Next Steps:"
echo "=============================================="
echo ""
echo "To read the full analysis:"
echo "  cat TRAINING_FIXES_ANALYSIS.md"
echo ""
echo "To run the fixed trainer:"
echo "  python curriculum_trainer_v2_fixed.py --games-per-phase 1000"
echo ""
echo "To compare with original (optional):"
echo "  python curriculum_trainer_v2.py --games-per-phase 1000"
echo ""
