# Catan Multi-Window Mode - User Guide

## Overview

The multi-window mode creates **4 separate Pygame windows**, one for each player, allowing you to play Catan locally just like on Colonist.io!

## How to Run

```bash
python multi_window.py
```

## What You'll See

**4 windows will open automatically:**
- **Player 1 (Red)** - Top-left corner of screen
- **Player 2 (Blue)** - Top-right corner of screen
- **Player 3 (Yellow)** - Bottom-left corner of screen
- **Player 4 (White)** - Bottom-right corner of screen

Each window is **1000x700 pixels**.

## How It Works

### Architecture

```
┌─────────────────────────────────────┐
│   Shared Game State (GameSystem)    │
│   - One copy in memory              │
│   - All windows read/write to it    │
│   - Thread-safe with locks          │
└─────────────────────────────────────┘
         ↑         ↑         ↑         ↑
         │         │         │         │
    Window 1   Window 2   Window 3   Window 4
   (Player 1) (Player 2) (Player 3) (Player 4)
```

### What Each Window Shows

**Every window shows:**
- ✅ **Same shared board** - all hexes, roads, settlements, cities, ports
- ✅ **Whose turn it is** - highlighted at top
- ✅ **All players' buildings** - color-coded

**Each window shows PRIVATELY:**
- 🔒 **Only their own resources** - other players can't see
- 🔒 **Trade offers directed at them**
- 🔒 **Their own messages/events**

### Privacy Model

This is like Colonist.io:
- Everyone sees the **public board**
- Only you see your **private hand** (resources, cards)
- Trade offers visible only to the recipient

## Controls

**On your turn only:**
- `D` - Roll dice
- `T` - End turn
- `5` - Toggle trade mode (coming soon)

**Any time:**
- `Y` - Accept trade offer (if you have one)
- `N` - Reject trade offer (if you have one)

## How Turns Work

1. **Only one player can act at a time** (the current player)
2. Other windows are **read-only** until it's their turn
3. When you end turn, the next window becomes active
4. All windows update instantly when actions happen

## Technical Implementation

### Files Created

- `multi_window.py` - Main launcher, creates 4 windows
- `player_window.py` - Individual player window class

### Thread Model

- **4 threads** - one per player window
- **Shared memory** - all threads access same `GameSystem` object
- **Thread locks** - prevent race conditions when updating state
- **Real-time sync** - windows refresh at 30 FPS

### State Synchronization

When Player 1 rolls dice:
```python
1. Player 1 window locks the game state
2. Updates GameSystem (roll dice)
3. Unlocks the game state
4. Other windows read updated state on next frame
5. All windows show the new dice result
```

## Future Features (To Be Added)

- [ ] Full trade interface with offers/counters
- [ ] Building placement (click to build)
- [ ] Development cards UI
- [ ] Robber movement
- [ ] Chat between windows
- [ ] Network support (play across computers)

## How to Exit

- Close any window: closes only that window
- Press `Ctrl+C` in terminal: closes all windows
- Close all 4 windows: game ends

## Troubleshooting

**Windows overlap?**
- They auto-position, but may need manual repositioning if your screen is small

**Can't see other players' actions?**
- Make sure all windows are visible
- Updates happen every frame (30 FPS)

**Window not responding?**
- Only the current player's window accepts input
- Wait for your turn!

## Comparison to Original main.py

| Feature | Single Window (`main.py`) | Multi-Window (`multi_window.py`) |
|---------|---------------------------|----------------------------------|
| Players | 1 window, all players visible | 4 windows, one per player |
| Privacy | None - everyone sees everything | Each player sees only their hand |
| Turn enforcement | Same screen | Separate windows |
| Local multiplayer | Not ideal | Perfect! |
| Like Colonist.io | No | Yes! |

## Next Steps

I'm adding the **trade system** next so players can:
1. Propose trades to other players
2. See offers appear in recipient's window
3. Accept/reject/counter offers
4. Complete trades with state sync

This will make it a **full local multiplayer experience!** 🎲
