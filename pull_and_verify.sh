#!/bin/bash
# Script to pull latest changes and verify

echo "=================================================="
echo "Pulling Latest Port Fix from Git"
echo "=================================================="

# Show current branch
echo -e "\n1. Current branch:"
git branch --show-current

# Fetch latest
echo -e "\n2. Fetching from remote..."
git fetch origin

# Show what will change
echo -e "\n3. Changes that will be pulled:"
git log HEAD..origin/claude/pygame-catan-ai-game-011CUcyT6G6W4JXTypiahi1d --oneline

# Pull the changes
echo -e "\n4. Pulling changes..."
git pull origin claude/pygame-catan-ai-game-011CUcyT6G6W4JXTypiahi1d

# Verify the new code is there
echo -e "\n5. Verifying new port generation code..."
if grep -q "🚢 PORT PLACEMENT SYSTEM - FOR FLAT-TOP HEXAGONS" game_system.py; then
    echo "✅ NEW CODE FOUND - Ports will be placed correctly!"
else
    echo "❌ OLD CODE STILL PRESENT - Pull may have failed"
fi

# Clear Python cache
echo -e "\n6. Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "✅ Cache cleared"

echo -e "\n=================================================="
echo "Ready to test! Run: python3 play_multiplayer.py"
echo "=================================================="
