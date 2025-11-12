"""
Multi-Window Catan Game
Launches 4 separate Pygame windows, one for each player
"""

import pygame
import threading
import sys
from tile import Tile
from game_system import (Player, Robber, GameBoard, GameSystem, ResourceType)
from player_window import PlayerWindow

# Standard Catan setup
NUMBER_TOKENS = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
RESOURCES = ["forest"] * 4 + ["hill"] * 3 + ["field"] * 4 + ["mountain"] * 3 + ["pasture"] * 4 + ["desert"]


def create_hexagonal_board(size, radius=2):
    """Create a hexagonal board"""
    tiles = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            tiles.append(Tile(q, r, size))
    return tiles


def assign_resources_numbers(tiles, robber):
    """Assign resources and numbers to tiles"""
    import random

    resources = RESOURCES.copy()
    random.shuffle(resources)

    for i, tile in enumerate(tiles):
        if i < len(resources):
            tile.resource = resources[i]

    # Place robber on desert
    desert_tile = None
    for tile in tiles:
        if tile.resource == "desert":
            tile.number = None
            desert_tile = tile
            robber.move_to_tile(tile)
            break

    # Assign numbers to non-desert tiles
    non_desert_tiles = [t for t in tiles if t.resource != "desert"]
    nums = NUMBER_TOKENS.copy()
    random.shuffle(nums)

    for i, tile in enumerate(non_desert_tiles):
        if i < len(nums):
            tile.number = nums[i]


def compute_center_offset(tiles, screen_w, screen_h):
    """Compute offset to center the board"""
    if not tiles:
        return (screen_w // 2, screen_h // 2)

    min_x = min(t.x for t in tiles)
    max_x = max(t.x for t in tiles)
    min_y = min(t.y for t in tiles)
    max_y = max(t.y for t in tiles)

    board_center_x = (min_x + max_x) / 2
    board_center_y = (min_y + max_y) / 2

    offset_x = screen_w // 2 - board_center_x
    offset_y = screen_h // 2 - board_center_y

    return (offset_x, offset_y)


def run_player_window(window):
    """Thread function to run a player window"""
    try:
        window.run()
    except Exception as e:
        print(f"Error in {window.player.name}'s window: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function - sets up 4-player game with separate windows"""

    print("=" * 60)
    print("CATAN - MULTI-WINDOW MODE")
    print("=" * 60)
    print("\nStarting 4-player game...")
    print("Each player will get their own window.")
    print("\nControls:")
    print("  D - Roll dice (on your turn)")
    print("  T - End turn")
    print("  5 - Toggle trade mode")
    print("\n" + "=" * 60)

    # Create game board
    tile_size = 50
    tiles = create_hexagonal_board(tile_size, radius=2)

    for t in tiles:
        t.find_neighbors(tiles)

    robber = Robber()
    assign_resources_numbers(tiles, robber)

    game_board = GameBoard(tiles)

    # Create 4 players
    players = [
        Player("Player 1 (Red)", (255, 50, 50)),
        Player("Player 2 (Blue)", (50, 50, 255)),
        Player("Player 3 (Yellow)", (255, 255, 50)),
        Player("Player 4 (White)", (255, 255, 255))
    ]

    # Give each player starting resources for testing
    for player in players:
        player.add_resource(ResourceType.WOOD, 3)
        player.add_resource(ResourceType.BRICK, 3)
        player.add_resource(ResourceType.WHEAT, 2)
        player.add_resource(ResourceType.SHEEP, 2)
        player.add_resource(ResourceType.ORE, 1)

    # Create shared game system
    game_system = GameSystem(game_board, players)
    game_system.robber = robber

    # Compute board offset (same for all windows)
    offset = compute_center_offset(tiles, 600, 600)

    # Create player windows
    windows = []
    for i, player in enumerate(players):
        window = PlayerWindow(player, game_system, game_board, offset, i)
        windows.append(window)

    # Launch windows in separate threads
    threads = []
    for window in windows:
        thread = threading.Thread(target=run_player_window, args=(window,))
        thread.daemon = True  # Exit when main thread exits
        thread.start()
        threads.append(thread)

    print(f"\n✓ Launched {len(windows)} player windows!")
    print("  Player 1: Top-left (Red)")
    print("  Player 2: Top-right (Blue)")
    print("  Player 3: Bottom-left (Yellow)")
    print("  Player 4: Bottom-right (White)")
    print("\nGame started! It's Player 1's turn.")
    print("\nPress Ctrl+C to quit all windows.\n")

    # Keep main thread alive
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n\nShutting down all windows...")
        for window in windows:
            window.running = False


if __name__ == "__main__":
    main()
