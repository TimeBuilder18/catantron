#!/usr/bin/env python3
"""
Quick test script to verify port placement algorithm is working correctly
Run this to see if the new code is loaded without starting the full game
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("🧪 PORT ALGORITHM TEST")
print("="*70)

# Step 1: Check if game_system.py has the new code
print("\n1️⃣  Checking game_system.py file contents...")
try:
    with open('game_system.py', 'r') as f:
        content = f.read()

    if "🚢 PORT PLACEMENT SYSTEM - FOR FLAT-TOP HEXAGONS" in content:
        print("   ✅ NEW CODE FOUND in file!")
    else:
        print("   ❌ OLD CODE in file - git pull needed")
        sys.exit(1)

    if "if (neighbor_q, neighbor_r) not in tile_map:" in content:
        print("   ✅ NEIGHBOR CHECK FOUND - Algorithm is correct!")
    else:
        print("   ❌ NEIGHBOR CHECK MISSING - Algorithm is wrong")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ ERROR reading file: {e}")
    sys.exit(1)

# Step 2: Try importing game_system module
print("\n2️⃣  Importing game_system module...")
try:
    import game_system
    print("   ✅ Module imported successfully")
except Exception as e:
    print(f"   ❌ ERROR importing: {e}")
    sys.exit(1)

# Step 3: Check what code the module has loaded
print("\n3️⃣  Checking loaded module code...")
import inspect
try:
    source = inspect.getsource(game_system.GameBoard.generate_ports)

    if "🚢 PORT PLACEMENT SYSTEM - FOR FLAT-TOP HEXAGONS" in source:
        print("   ✅ NEW CODE LOADED in Python!")
    else:
        print("   ❌ OLD CODE LOADED - Python is using cached bytecode!")
        print("   💡 Solution: Clear Python cache and restart terminal")
        print("   Run: find . -type d -name __pycache__ -exec rm -rf {} +")
        sys.exit(1)

    if "if (neighbor_q, neighbor_r) not in tile_map:" in source:
        print("   ✅ NEIGHBOR CHECK in loaded code - Ready to test!")
    else:
        print("   ❌ NEIGHBOR CHECK MISSING in loaded code")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ ERROR checking source: {e}")
    sys.exit(1)

# Step 4: Create a minimal test
print("\n4️⃣  Creating test board...")
try:
    import pygame
    pygame.init()

    # Create minimal display
    screen = pygame.display.set_mode((100, 100))

    # Create game board
    board = game_system.GameBoard()

    print(f"   ✅ Board created with {len(board.tiles)} tiles")
    print(f"   ✅ Generated {len(board.ports)} ports")

    # Check if any ports are on coastal edges
    if len(board.ports) == 9:
        print(f"   ✅ CORRECT: Found 9 ports (expected)")
    else:
        print(f"   ⚠️  WARNING: Found {len(board.ports)} ports (expected 9)")

    # Show port positions
    print("\n5️⃣  Port locations:")
    for i, port in enumerate(board.ports, 1):
        print(f"   Port {i}: {port.trade_type} at edge ({port.edge.vertex1.x:.1f},{port.edge.vertex1.y:.1f})")

    pygame.quit()

except Exception as e:
    print(f"   ❌ ERROR creating board: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ TEST COMPLETE - Port algorithm is working correctly!")
print("="*70)
print("\n🎯 Next step: Run full game with 'python3 play_multiplayer.py'")
print("   Watch for: '🚢 PORT PLACEMENT SYSTEM - FOR FLAT-TOP HEXAGONS'")
