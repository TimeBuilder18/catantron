"""
Catan Multiplayer Launcher - macOS Compatible
Starts game server + 4 client windows using multiprocessing
"""

import time
import sys
from multiprocessing import Process
from game_server import GameServer
from client_window import run_client_window


def run_server():
    """Run the game server in a separate process"""
    server = GameServer()
    try:
        server.start()
        # Keep server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.stop()


def main():
    """Launch server + 4 client windows"""
    print("=" * 60)
    print("CATAN - MULTIPLAYER MODE (macOS Compatible)")
    print("=" * 60)
    print("\nStarting 4-player networked game...")
    print("Architecture: 1 server + 4 client windows\n")

    # Player configurations
    players_config = [
        (0, "Player 1 (Red)", (255, 50, 50)),
        (1, "Player 2 (Blue)", (50, 50, 255)),
        (2, "Player 3 (Yellow)", (255, 255, 50)),
        (3, "Player 4 (White)", (255, 255, 255))
    ]

    processes = []

    # Start server process
    print("1. Starting game server...")
    server_process = Process(target=run_server)
    server_process.start()
    processes.append(server_process)
    time.sleep(2)  # Give server time to start

    # Start 4 client window processes
    print("2. Launching 4 player windows...")
    for player_index, player_name, player_color in players_config:
        p = Process(
            target=run_client_window,
            args=(player_index, player_name, player_color)
        )
        p.start()
        processes.append(p)
        time.sleep(0.5)  # Stagger window creation

    print(f"\n✓ Launched server + {len(players_config)} player windows!")
    print("  Player 1 (Red): Top-left")
    print("  Player 2 (Blue): Top-right")
    print("  Player 3 (Yellow): Bottom-left")
    print("  Player 4 (White): Bottom-right")
    print("\n" + "=" * 60)
    print("GAME STARTED!")
    print("=" * 60)
    print("\nControls (on your turn):")
    print("  D - Roll dice")
    print("  T - End turn")
    print("\nAll windows share the same game state!")
    print("Actions in one window appear in all windows instantly.")
    print("\nPress Ctrl+C to quit all windows.\n")

    # Wait for all processes
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\nShutting down all processes...")
        for p in processes:
            p.terminate()
            p.join()

    print("\n✓ All processes closed. Game ended.")


if __name__ == "__main__":
    main()
