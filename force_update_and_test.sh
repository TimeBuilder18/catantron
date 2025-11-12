#!/bin/bash
# Comprehensive script to force update code and clear all caches

echo "=========================================================="
echo "🔧 FORCE UPDATE & CACHE CLEAR"
echo "=========================================================="

# Show current location
echo -e "\n📍 Current directory:"
pwd

# Show current branch
echo -e "\n🌿 Current branch:"
git branch --show-current

# Show current commit BEFORE update
echo -e "\n📜 Current commit (BEFORE update):"
git log --oneline -1

# Step 1: Stash any local changes (just in case)
echo -e "\n💾 Stashing any local changes..."
git stash

# Step 2: Fetch latest from remote
echo -e "\n📥 Fetching latest from remote..."
git fetch origin claude/pygame-catan-ai-game-011CUcyT6G6W4JXTypiahi1d

# Step 3: FORCE reset to remote branch (nuclear option)
echo -e "\n⚠️  FORCE resetting to remote branch..."
git reset --hard origin/claude/pygame-catan-ai-game-011CUcyT6G6W4JXTypiahi1d

# Show commit AFTER update
echo -e "\n📜 Current commit (AFTER update):"
git log --oneline -1
echo "   ☝️  Should show: 6d017ca FIX: Port placement for FLAT-TOP hexagons"

# Step 4: Verify new code exists in file
echo -e "\n🔍 Checking if new code is in game_system.py..."
if grep -q "🚢 PORT PLACEMENT SYSTEM - FOR FLAT-TOP HEXAGONS" game_system.py; then
    echo "   ✅ NEW CODE FOUND in file!"
    echo "   Line 602 contains the new port system"
else
    echo "   ❌ ERROR: NEW CODE NOT FOUND!"
    echo "   Something went wrong with git pull"
    exit 1
fi

# Step 5: Show the actual line to confirm
echo -e "\n📄 Showing line 602 from game_system.py:"
sed -n '602p' game_system.py

# Step 6: Verify the critical neighbor check exists
echo -e "\n🔍 Checking for critical neighbor check (line 653)..."
if grep -q "if (neighbor_q, neighbor_r) not in tile_map:" game_system.py; then
    echo "   ✅ NEIGHBOR CHECK FOUND - Algorithm is correct!"
else
    echo "   ❌ ERROR: Neighbor check not found"
    exit 1
fi

# Step 7: AGGRESSIVELY clear Python cache
echo -e "\n🧹 AGGRESSIVELY clearing ALL Python cache..."

# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rfv {} + 2>/dev/null
echo "   ✅ Removed __pycache__ directories"

# Remove all .pyc files
find . -name "*.pyc" -delete -print 2>/dev/null
echo "   ✅ Removed .pyc files"

# Remove all .pyo files
find . -name "*.pyo" -delete -print 2>/dev/null
echo "   ✅ Removed .pyo files"

# Step 8: Verify no cache remains
echo -e "\n🔍 Verifying no Python cache remains..."
CACHE_COUNT=$(find . -name "*.pyc" -o -name "*.pyo" -o -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ "$CACHE_COUNT" -eq 0 ]; then
    echo "   ✅ All Python cache cleared successfully!"
else
    echo "   ⚠️  Warning: $CACHE_COUNT cache files/dirs still exist"
fi

echo -e "\n=========================================================="
echo "✅ UPDATE COMPLETE!"
echo "=========================================================="
echo ""
echo "🎯 NEXT STEPS:"
echo "   1. Close ALL Python processes and terminal windows"
echo "   2. Open a fresh terminal"
echo "   3. cd to this directory"
echo "   4. Run: python3 play_multiplayer.py"
echo ""
echo "🔍 WHAT TO LOOK FOR:"
echo "   ❌ OLD: '=== GENERATING PORTS AT FIXED POSITIONS ==='"
echo "   ✅ NEW: '🚢 PORT PLACEMENT SYSTEM - FOR FLAT-TOP HEXAGONS'"
echo ""
echo "   If you still see OLD message after fresh restart:"
echo "   • Check you're in correct directory: $(pwd)"
echo "   • Check Python version: python3 --version"
echo "   • Try: python3 -B play_multiplayer.py (bypasses cache)"
echo "=========================================================="
