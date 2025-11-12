# Catan Multiplayer Mode - Complete Guide

## 🎮 What's New?

You now have **fully synchronized 4-player Catan** with shared game state!

### Architecture

```
         ┌─────────────┐
         │Game Server  │  ← Manages shared GameSystem
         │ (Socket)    │
         └─────────────┘
              ↑  ↓
       ┌──────┴──┴─────┐
       ↓      ↓    ↓   ↓
    Client Client Client Client
      1      2      3     4
    (Red)  (Blue)(Yellow)(White)
```

**How it works:**
1. **Game Server** runs in its own process, managing the `GameSystem`
2. **4 Client Windows** connect via sockets (TCP port 5555)
3. Server sends game state to each client 30 times/second
4. Clients send actions (roll dice, end turn) to server
5. Server updates state, broadcasts to all clients instantly

## 🚀 How to Run

### Quick Start

```bash
python3 play_multiplayer.py
```

That's it! The launcher will:
1. Start the game server
2. Launch 4 player windows
3. Connect everything automatically

### What You'll See

**5 separate processes:**
- 1 server process (runs in terminal)
- 4 window processes (one per player)

**4 windows positioned automatically:**
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

## 🎯 Gameplay

### Controls

**On Your Turn:**
- `D` - Roll dice
- `T` - End turn

**Any Time:**
- Close window or press `Ctrl+C` to quit

### Game Flow

1. **Player 1 starts** - their window shows "YOUR TURN"
2. **Other players wait** - their windows show "P1's turn"
3. **Roll dice (D)** - dice result appears in ALL windows
4. **Resources distributed** - only your window shows your new resources
5. **End turn (T)** - turn passes to next player
6. **All windows update** - new player's window shows "YOUR TURN"

### What's Synchronized

**✅ Everyone sees:**
- Current player's turn
- Dice rolls
- All roads, settlements, cities
- The hexagonal board
- Port locations
- Robber position

**🔒 Only you see:**
- Your resource cards (private hand)
- Your messages

## 🔧 Technical Details

### Files

| File | Purpose |
|------|---------|
| `play_multiplayer.py` | Main launcher (start here!) |
| `game_server.py` | Server that manages game state |
| `client_window.py` | Client window that connects to server |
| `game_system.py` | Core game logic (shared by server) |

### Network Protocol

**Port:** 5555 (localhost only)
**Protocol:** TCP with JSON messages
**Message Format:** Newline-delimited JSON

**Client → Server Actions:**
```
"ROLL_DICE"
"END_TURN"
```

**Server → Client State:**
```json
{
  "current_turn": 0,
  "dice_rolled": false,
  "my_resources": {"wood": 3, "brick": 2, ...},
  "players": [...],
  "tiles": [...],
  "settlements": [...],
  "cities": [...],
  "roads": [...],
  "ports": [...]
}
```

### Why This Works on macOS

**Problem with Threading:**
- macOS Cocoa requires all NSWindow operations on main thread
- Threading = multiple windows in one process = BLOCKED

**Solution with Multiprocessing:**
- Each window runs in separate OS process
- Each process has its own main thread
- macOS is happy! ✅

**Bonus:** This architecture also supports network play in the future!

## 🎲 Current Features

**✅ Implemented:**
- [x] 4 separate player windows
- [x] Synchronized game state
- [x] Turn-based gameplay
- [x] Dice rolling with resource distribution
- [x] Full board rendering (hexes, ports, numbers)
- [x] Private resource hands
- [x] Roads, settlements, cities display
- [x] Port visualization
- [x] Real-time state updates (30 FPS)

**⏳ Coming Next:**
- [ ] Building placement (click to build)
- [ ] Trading system
- [ ] Development cards
- [ ] Robber movement
- [ ] Victory point tracking
- [ ] Longest road / largest army

## 🆚 Comparison to Old Versions

| Feature | `main.py` | `multi_window_macos.py` | `play_multiplayer.py` |
|---------|-----------|-------------------------|----------------------|
| Architecture | Single window | 4 windows (no sync) | 4 windows + server (synced) |
| Game state | Shared in-memory | None (separate) | Shared via server |
| Privacy | None | None | Each player sees own hand |
| macOS compatible | Yes | Yes | Yes |
| Turn enforcement | No | No | **Yes** |
| Multiplayer ready | No | No | **Yes** |
| Like Colonist.io | No | No | **Yes!** |

## 🐛 Troubleshooting

**"Connection refused"**
- Server may not have started yet
- Wait 2 seconds and try again

**Windows overlap?**
- Reposition manually if your screen is small
- Windows auto-position at: (50,50), (1070,50), (50,780), (1070,780)

**Can't see other players' actions?**
- Make sure all windows are visible
- State updates at 30 FPS - should be instant

**Server crashes?**
- Check terminal for error messages
- Ensure port 5555 is available

**macOS "Python quit unexpectedly"?**
- Use `python3` not `python`
- Make sure Pygame is installed: `pip3 install pygame`

## 🚀 Next Steps

To complete the game, I'll be adding:

1. **Click-to-Build System**
   - Click vertices to place settlements/cities
   - Click edges to place roads
   - Resource checking and deduction

2. **Trading Interface**
   - Propose trades to specific players
   - Accept/reject/counter offers
   - Trade notifications appear in recipient's window

3. **Development Cards**
   - Buy cards
   - Play cards (knight, road building, etc.)
   - Victory point cards

4. **Robber & Stealing**
   - Move robber on 7
   - Steal from adjacent players
   - Blocking hex production

5. **Win Conditions**
   - Track victory points
   - Longest road calculation
   - Largest army tracking
   - Winner announcement

## 🎉 Try It Now!

```bash
python3 play_multiplayer.py
```

Have 4 player windows + server running in under 5 seconds! 🚀
