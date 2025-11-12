"""
Multi-Window Catan Game - macOS Compatible Version
Uses multiprocessing instead of threading to avoid macOS window restrictions
"""

import pygame
import sys
import json
import time
import os
from pathlib import Path
from tile import Tile
from game_system import (Player, Robber, GameBoard, GameSystem, ResourceType,
                         Settlement, City)

# Standard Catan setup
NUMBER_TOKENS = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
RESOURCES = ["forest"] * 4 + ["hill"] * 3 + ["field"] * 4 + ["mountain"] * 3 + ["pasture"] * 4 + ["desert"]

RESOURCE_COLORS = {
    "forest": (34, 139, 34),
    "hill": (178, 34, 34),
    "field": (218, 165, 32),
    "mountain": (169, 169, 169),
    "pasture": (144, 238, 144),
    "desert": (237, 201, 175),
    "wood": (34, 139, 34),
    "brick": (178, 34, 34),
    "wheat": (218, 165, 32),
    "ore": (169, 169, 169),
    "sheep": (144, 238, 144)
}

STATE_FILE = "game_state.json"


def create_hexagonal_board(size, radius=2):
    """Create a hexagonal board"""
    tiles = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            tiles.append(Tile(q, r, size))
    return tiles


def run_player_window(player_index, player_name, player_color):
    """Run a single player window - this runs in a separate process"""

    # Position window based on player index
    positions = [
        (50, 50),      # Player 1: top-left
        (1070, 50),    # Player 2: top-right
        (50, 780),     # Player 3: bottom-left
        (1070, 780)    # Player 4: bottom-right
    ]

    x, y = positions[player_index]
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((1000, 700))
    pygame.display.set_caption(f"Catan - {player_name}")

    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 18)
    title_font = pygame.font.Font(None, 32)

    clock = pygame.time.Clock()
    running = True

    messages = []

    def add_message(text, color=(255, 255, 255)):
        messages.append((text, color, pygame.time.get_ticks()))
        if len(messages) > 5:
            messages.pop(0)

    print(f"✓ {player_name} window started (PID: {os.getpid()})")
    add_message(f"Welcome {player_name}!", (100, 255, 100))

    # Main game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    add_message("Roll dice (D pressed)", (255, 255, 0))
                elif event.key == pygame.K_t:
                    add_message("End turn (T pressed)", (100, 255, 100))
                elif event.key == pygame.K_5:
                    add_message("Toggle trade mode", (255, 200, 255))

        # Clear screen
        screen.fill((20, 20, 30))

        # Draw header
        pygame.draw.rect(screen, (30, 30, 40), (0, 0, 1000, 80))

        # Player color indicator
        color_box = pygame.Rect(20, 20, 40, 40)
        pygame.draw.rect(screen, player_color, color_box)
        pygame.draw.rect(screen, (255, 255, 255), color_box, 2)

        # Player name
        name_text = title_font.render(player_name, True, (255, 255, 255))
        screen.blit(name_text, (70, 25))

        # Instructions
        y_pos = 100
        instructions = [
            "Controls:",
            "  D - Roll dice",
            "  T - End turn",
            "  5 - Trade mode",
            "",
            "Each window is a separate process!",
            f"Process ID: {os.getpid()}"
        ]

        for line in instructions:
            text = small_font.render(line, True, (200, 200, 200))
            screen.blit(text, (20, y_pos))
            y_pos += 20

        # Draw messages
        y_pos = 600
        current_time = pygame.time.get_ticks()
        for message, color, timestamp in messages:
            age = current_time - timestamp
            if age < 5000:
                text = small_font.render(message, True, color)
                screen.blit(text, (20, y_pos))
                y_pos += 20

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    print(f"✓ {player_name} window closed")


def main():
    """Main function - launches 4 separate processes"""

    print("=" * 60)
    print("CATAN - MULTI-WINDOW MODE (macOS Compatible)")
    print("=" * 60)
    print("\nStarting 4-player game using multiprocessing...")
    print("Each player window will be a separate process.\n")

    # Player configurations
    players_config = [
        (0, "Player 1 (Red)", (255, 50, 50)),
        (1, "Player 2 (Blue)", (50, 50, 255)),
        (2, "Player 3 (Yellow)", (255, 255, 50)),
        (3, "Player 4 (White)", (255, 255, 255))
    ]

    # Import multiprocessing
    from multiprocessing import Process

    # Create and start processes
    processes = []
    for player_index, player_name, player_color in players_config:
        p = Process(target=run_player_window, args=(player_index, player_name, player_color))
        p.start()
        processes.append(p)
        time.sleep(0.3)  # Stagger window creation

    print(f"\n✓ Launched {len(processes)} player windows!")
    print("  Player 1 (Red): Top-left")
    print("  Player 2 (Blue): Top-right")
    print("  Player 3 (Yellow): Bottom-left")
    print("  Player 4 (White): Bottom-right")
    print("\nEach window is running in its own process.")
    print("Close any window or press Ctrl+C to exit.\n")

    # Wait for all processes
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\n\nShutting down all windows...")
        for p in processes:
            p.terminate()
            p.join()

    print("\n✓ All windows closed. Game ended.")


if __name__ == "__main__":
    main()
