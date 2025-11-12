# How to Run Multi-Window Catan on macOS

## The Problem You Encountered

```
pygame.error: NSWindow should only be instantiated on the main thread!
```

**Why this happens:**
- macOS requires **all GUI windows** to be created on the **main thread**
- The original `multi_window.py` uses **threading** (4 threads in 1 process)
- Each thread tried to create a window → macOS said NO!

## The Solution

Use **multiprocessing** instead of threading:
- **4 separate processes** instead of 4 threads
- Each process has its own main thread
- Each process can create its own window
- macOS is happy! ✅

## How to Run

### Step 1: Install Pygame (if needed)
```bash
pip install pygame
```

### Step 2: Run the macOS version
```bash
cd /path/to/catantron
python multi_window_macos.py
```

### Step 3: You'll see 4 windows open!
```
┌─────────────┐  ┌─────────────┐
│  Player 1   │  │  Player 2   │
│   (Red)     │  │   (Blue)    │
│  TOP-LEFT   │  │  TOP-RIGHT  │
└─────────────┘  └─────────────┘

┌─────────────┐  ┌─────────────┐
│  Player 3   │  │  Player 4   │
│  (Yellow)   │  │  (White)    │
│ BOTTOM-LEFT │  │BOTTOM-RIGHT │
└─────────────┘  └─────────────┘
```

## Current Status

**✅ What Works:**
- 4 separate windows open
- Each window is its own process
- Windows are positioned automatically
- Basic keyboard input (D, T, 5)
- Messages display in each window

**⚠️ Not Yet Implemented:**
- Shared game state between processes
- Actual game board rendering
- Turn-based gameplay
- Trading system

## Architecture Difference

### Threading (doesn't work on macOS):
```
┌─────────────────────────────┐
│      One Process             │
│  ┌──────┐ ┌──────┐          │
│  │Thread│ │Thread│ ← macOS   │
│  │  1   │ │  2   │   blocks! │
│  └──────┘ └──────┘          │
└─────────────────────────────┘
```

### Multiprocessing (works on macOS):
```
┌──────────┐  ┌──────────┐
│Process 1 │  │Process 2 │
│ (window) │  │ (window) │
└──────────┘  └──────────┘
     ↓             ↓
  Shared State File or IPC
```

## Next Steps to Complete

To make this a full game, we need to add:

1. **Shared State Management**
   - Option A: JSON file (simple but slower)
   - Option B: Shared memory (faster, more complex)
   - Option C: Socket server (most flexible)

2. **Game Board Rendering**
   - Each process loads the board
   - Reads shared state to show buildings

3. **Turn Management**
   - Only current player can act
   - Others wait and watch

4. **Trade System**
   - Propose trades via shared state
   - Notifications appear in target's window

## Recommended Next Approach

**Option: Socket-Based Server** (Best for multiple windows)

```
         ┌─────────────┐
         │Game Server  │
         │  (Python)   │
         └─────────────┘
              ↓  ↑
       ┌──────┴──┴─────┐
       ↓      ↓    ↓   ↓
    Window Window Window Window
      1      2      3     4
```

Benefits:
- Clean separation
- Easy to add network play later
- Works perfectly on macOS
- Can add web clients later

Would you like me to implement this next?

## Why Can't We Just Use Threading?

**Technical Reason:**
macOS uses Cocoa (Apple's GUI framework)
- Cocoa is **not thread-safe** for window operations
- Apple enforces: "All NSWindow operations MUST be on main thread"
- Python Pygame uses Cocoa on macOS
- Therefore: One window per main thread only

**Windows/Linux:**
- Use different GUI systems (Win32, X11)
- These ARE thread-safe for windows
- Multiple windows in multiple threads works fine

**Bottom Line:**
- macOS is more strict
- Multiprocessing is the only solution
- This is why Electron apps are popular (each window = separate process)

## Alternative: Web-Based UI

If multiprocessing is too complex, consider:

**Flask/FastAPI + HTML/CSS/JS**
- Python server hosts game logic
- 4 browser tabs (one per player)
- JavaScript handles UI
- WebSocket for real-time updates
- Works on ANY platform
- Looks more modern

Let me know which direction you want to go!
