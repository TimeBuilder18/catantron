# macOS Setup Guide

## First Time Running - Window Permission Popup

When you run `python3 play_multiplayer.py` for the first time, macOS will show a popup asking:

```
"Python" would like to open multiple windows
Do you want to allow this?
```

**✅ Click "Allow"**

This is normal! The game launches 5 separate processes:
- 1 game server
- 4 player windows

macOS asks for permission to create multiple windows for security reasons.

## If You See Black Screens

If you see windows but they're all black:

1. **Wait 2-3 seconds** - Server and clients need time to connect
2. **Look for debug text** in bottom-right corner:
   - Should say "✓ Connected to server"
   - Should say "✓ Game active (19 tiles)"
3. **Check the terminal** - Look for error messages

## Troubleshooting

### "Connection refused"
```bash
# Make sure you're using python3 (not python)
python3 play_multiplayer.py

# If still having issues, try running server manually first:
python3 game_server.py &
sleep 2
python3 play_multiplayer.py
```

### Windows Don't Appear
- Make sure you clicked "Allow" on the popup
- Try running again
- Check System Preferences > Security & Privacy > Privacy > Screen Recording

### "Module not found: pygame"
```bash
pip3 install pygame
```

### Windows Overlap
- If your screen is small (<1920x1080), windows might overlap
- Manually drag windows to arrange them
- Or edit `client_window.py` line 54-60 to adjust positions

## Recommended Screen Size

For best experience, use a screen at least **1920x1080** so all 4 windows fit comfortably:

```
Screen Layout:
┌─────────────┐  ┌─────────────┐
│  Player 1   │  │  Player 2   │
│ (50, 50)    │  │ (1070, 50)  │
└─────────────┘  └─────────────┘
┌─────────────┐  ┌─────────────┐
│  Player 3   │  │  Player 4   │
│ (50, 780)   │  │ (1070, 780) │
└─────────────┘  └─────────────┘
```

## Quick Start

```bash
# Install pygame if needed
pip3 install pygame

# Run the game
python3 play_multiplayer.py

# Click "Allow" on macOS popup

# Wait 2-3 seconds for everything to connect

# Click on Player 1's window and press D to roll dice!
```

## What You Should See

**In Terminal:**
```
CATAN GAME SERVER STARTED
Listening on 127.0.0.1:5555
Waiting for 4 players to connect...

✓ Player 1 connected
✓ Player 2 connected
✓ Player 3 connected
✓ Player 4 connected

✓ All 4 players connected! Game starting...
```

**In Windows:**
- Header with player name and color
- "YOUR TURN" indicator (Player 1 starts)
- Resource list on left
- Hexagonal board in center with colored hexes
- Port circles around the edges
- Number tokens on each hex
- Controls guide in bottom-left
- Connection status in bottom-right

## Having Issues?

1. Check the terminal for error messages
2. Make sure port 5555 isn't in use: `lsof -i :5555`
3. Kill old processes if needed: `killall python3`
4. Try again!

Enjoy playing Catan! 🎲
