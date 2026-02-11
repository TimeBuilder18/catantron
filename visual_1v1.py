"""
Visual 1v1 Catan - Watch AI agents play or play against AI

USAGE:
------
# Watch trained model vs rule-based AI
python visual_1v1.py --model models/v6_pc_phase5.pt --opponent medium

# Watch two models fight
python visual_1v1.py --model models/v6_pc_phase5.pt --model2 models/v6_pc_game1024.pt

# Play against trained AI (you are Player 1)
python visual_1v1.py --model models/v6_pc_phase5.pt --human

# Watch random vs random (fast)
python visual_1v1.py --speed 0.1
"""

import pygame
import sys
import random
import math
import argparse
import torch
import numpy as np
from tile import Tile
from game_system import (Player, Robber, GameBoard, GameSystem, ResourceType,
                         Settlement, City, Road, DevelopmentCardType, GameConstants, PortType)
from network_wrapper import NetworkWrapper
from simplified_reward_wrapper import SimplifiedRewardWrapper

# Standard Catan setup
NUMBER_TOKENS = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
RESOURCES = ["forest"] * 4 + ["hill"] * 3 + ["field"] * 4 + ["mountain"] * 3 + ["pasture"] * 4 + ["desert"]

RESOURCE_COLORS = {
    "forest": (34, 139, 34),
    "hill": (178, 34, 34),
    "field": (218, 165, 32),
    "mountain": (169, 169, 169),
    "pasture": (144, 238, 144),
    "desert": (237, 201, 175)
}

# Optimized screen size for 1v1
SCREEN_W, SCREEN_H = 1000, 700


def create_hexagonal_board(size, radius=2):
    tiles = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            tiles.append(Tile(q, r, size))
    return tiles


def assign_resources_numbers(tiles, robber):
    all_resources = RESOURCES.copy()
    random.shuffle(all_resources)

    for i, tile in enumerate(tiles):
        if i < len(all_resources):
            tile.resource = all_resources[i]
        else:
            tile.resource = "forest"

    for tile in tiles:
        if tile.resource == "desert":
            tile.number = None
            robber.move_to_tile(tile)
            break

    nums = NUMBER_TOKENS.copy()
    random.shuffle(nums)
    non_desert = [t for t in tiles if t.resource != "desert"]
    for i, tile in enumerate(non_desert):
        if i < len(nums):
            tile.number = nums[i]


def draw_board(screen, game_board, offset):
    """Draw game board optimized for 1v1"""
    # Draw tiles
    for tile in game_board.tiles:
        color = RESOURCE_COLORS.get(tile.resource, RESOURCE_COLORS["desert"])
        tile.draw(screen, fill_color=color, offset=offset, show_coords=False)

    # Draw edges (roads)
    for edge in game_board.edges:
        x1, y1 = edge.vertex1.x + offset[0], edge.vertex1.y + offset[1]
        x2, y2 = edge.vertex2.x + offset[0], edge.vertex2.y + offset[1]

        if edge.structure:
            pygame.draw.line(screen, edge.structure.player.color, (x1, y1), (x2, y2), 6)
        else:
            pygame.draw.line(screen, (80, 80, 80), (x1, y1), (x2, y2), 1)

    # Draw ports
    PORT_COLORS = {
        PortType.GENERIC: (200, 200, 200),
        PortType.WOOD: (34, 139, 34),
        PortType.BRICK: (178, 34, 34),
        PortType.WHEAT: (218, 165, 32),
        PortType.SHEEP: (144, 238, 144),
        PortType.ORE: (169, 169, 169),
    }

    for port in game_board.ports:
        x1, y1 = port.vertex1.x + offset[0], port.vertex1.y + offset[1]
        x2, y2 = port.vertex2.x + offset[0], port.vertex2.y + offset[1]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

        port_color = PORT_COLORS.get(port.port_type, (200, 200, 200))
        pygame.draw.circle(screen, port_color, (int(mid_x), int(mid_y)), 8)
        pygame.draw.circle(screen, (0, 0, 0), (int(mid_x), int(mid_y)), 8, 2)

    # Draw vertices (settlements/cities)
    for vertex in game_board.vertices:
        x = vertex.x + offset[0]
        y = vertex.y + offset[1]

        if vertex.structure:
            player_color = vertex.structure.player.color
            if isinstance(vertex.structure, Settlement):
                pygame.draw.circle(screen, player_color, (int(x), int(y)), 12)
                pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 12, 2)
            elif isinstance(vertex.structure, City):
                pygame.draw.rect(screen, player_color, (x - 14, y - 14, 28, 28))
                pygame.draw.rect(screen, (0, 0, 0), (x - 14, y - 14, 28, 28), 2)
        else:
            pygame.draw.circle(screen, (60, 60, 60), (int(x), int(y)), 3)


def draw_player_panel(screen, player, x, y, width, is_current, font, small_font):
    """Draw compact player info panel"""
    height = 120

    # Background
    bg_color = (50, 50, 60) if is_current else (30, 30, 40)
    pygame.draw.rect(screen, bg_color, (x, y, width, height), border_radius=10)

    # Border highlight for current player
    if is_current:
        pygame.draw.rect(screen, player.color, (x, y, width, height), 3, border_radius=10)

    # Player name and color
    pygame.draw.circle(screen, player.color, (x + 20, y + 20), 10)
    name_text = font.render(player.name, True, player.color)
    screen.blit(name_text, (x + 40, y + 12))

    # Victory Points - BIG
    vp = player.calculate_victory_points()
    vp_color = (255, 215, 0) if vp >= 8 else (255, 255, 255)
    vp_text = font.render(f"VP: {vp}", True, vp_color)
    screen.blit(vp_text, (x + width - 70, y + 12))

    # Resources (compact)
    res_y = y + 45
    res_names = ['W', 'B', 'G', 'S', 'O']  # Wood, Brick, Grain, Sheep, Ore
    res_types = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT, ResourceType.SHEEP, ResourceType.ORE]

    for i, (name, res_type) in enumerate(zip(res_names, res_types)):
        amt = player.resources[res_type]
        color = (255, 255, 255) if amt > 0 else (80, 80, 80)
        text = small_font.render(f"{name}:{amt}", True, color)
        screen.blit(text, (x + 15 + i * 40, res_y))

    # Buildings (compact)
    build_y = y + 70
    build_text = small_font.render(
        f"S:{len(player.settlements)} C:{len(player.cities)} R:{len(player.roads)}",
        True, (200, 200, 200)
    )
    screen.blit(build_text, (x + 15, build_y))

    # Dev cards
    total_dev = sum(player.development_cards.values())
    if total_dev > 0:
        dev_text = small_font.render(f"Dev:{total_dev}", True, (200, 200, 200))
        screen.blit(dev_text, (x + 130, build_y))

    # Special achievements
    achieve_y = y + 95
    if player.has_longest_road:
        lr_text = small_font.render("Longest Road", True, (255, 215, 0))
        screen.blit(lr_text, (x + 15, achieve_y))
    if player.has_largest_army:
        la_text = small_font.render("Largest Army", True, (255, 215, 0))
        screen.blit(la_text, (x + 120, achieve_y))


def draw_game_info(screen, game, font, small_font, vp_to_win):
    """Draw game state info at bottom"""
    y = SCREEN_H - 80

    # Background
    pygame.draw.rect(screen, (30, 30, 40), (0, y, SCREEN_W, 80))
    pygame.draw.line(screen, (100, 100, 100), (0, y), (SCREEN_W, y), 2)

    # Phase info
    if game.is_initial_placement_phase():
        phase_text = f"SETUP: {game.get_current_player_needs()}"
        phase_color = (255, 255, 0)
    else:
        phase_text = f"Turn {game.turn_number} - {game.turn_phase}"
        phase_color = (100, 255, 100)

    phase_surface = font.render(phase_text, True, phase_color)
    screen.blit(phase_surface, (20, y + 15))

    # Dice roll
    if game.last_dice_roll:
        d1, d2, total = game.last_dice_roll
        dice_color = (255, 100, 100) if total == 7 else (255, 255, 255)
        dice_text = font.render(f"Dice: {total} ({d1}+{d2})", True, dice_color)
        screen.blit(dice_text, (20, y + 45))

    # VP target
    vp_text = small_font.render(f"First to {vp_to_win} VP wins!", True, (200, 200, 200))
    screen.blit(vp_text, (SCREEN_W - 150, y + 30))


def find_closest_vertex(game_board, mouse_pos, offset, max_distance=20):
    mouse_x, mouse_y = mouse_pos
    closest_vertex = None
    closest_distance = max_distance

    for vertex in game_board.vertices:
        x = vertex.x + offset[0]
        y = vertex.y + offset[1]
        distance = ((mouse_x - x) ** 2 + (mouse_y - y) ** 2) ** 0.5

        if distance < closest_distance:
            closest_distance = distance
            closest_vertex = vertex

    return closest_vertex


def find_closest_edge(game_board, mouse_pos, offset, max_distance=15):
    mouse_x, mouse_y = mouse_pos
    closest_edge = None
    closest_distance = max_distance

    for edge in game_board.edges:
        x1, y1 = edge.vertex1.x + offset[0], edge.vertex1.y + offset[1]
        x2, y2 = edge.vertex2.x + offset[0], edge.vertex2.y + offset[1]

        # Midpoint distance
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        distance = ((mouse_x - mid_x) ** 2 + (mouse_y - mid_y) ** 2) ** 0.5

        if distance < closest_distance:
            closest_distance = distance
            closest_edge = edge

    return closest_edge


def play_random_turn(game, player_id):
    """Random AI - makes random valid moves"""
    player = game.players[player_id]

    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            if game.last_settlement_vertex:
                edges = game.game_board.edges
                valid = [e for e in edges
                        if e.structure is None and
                        (e.vertex1 == game.last_settlement_vertex or
                         e.vertex2 == game.last_settlement_vertex)]
                if valid:
                    game.try_place_initial_road(random.choice(valid), player)
        else:
            vertices = game.game_board.vertices
            valid = [v for v in vertices if v.structure is None and
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                game.try_place_initial_settlement(random.choice(valid), player)
        return True

    if game.can_roll_dice():
        game.roll_dice()
        return True

    if game.can_trade_or_build():
        actions = []
        res = player.resources

        if (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
            res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1):
            v = game.get_buildable_vertices_for_settlements()
            if v: actions.append(('sett', v))

        if res[ResourceType.WHEAT] >= 2 and res[ResourceType.ORE] >= 3:
            v = game.get_buildable_vertices_for_cities()
            if v: actions.append(('city', v))

        if res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1:
            e = game.get_buildable_edges()
            if e: actions.append(('road', e))

        if (res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1 and
            res[ResourceType.ORE] >= 1 and not game.dev_deck.is_empty()):
            actions.append(('dev', None))

        if actions and random.random() < 0.85:
            action_type, locs = random.choice(actions)
            if action_type == 'sett':
                player.try_build_settlement(random.choice(locs))
            elif action_type == 'city':
                player.try_build_city(random.choice(locs))
            elif action_type == 'road':
                player.try_build_road(random.choice(locs))
            elif action_type == 'dev':
                player.try_buy_development_card(game.dev_deck)
            return True

    if game.can_end_turn():
        game.end_turn()
        return True

    return False


def play_rule_based_turn(game, player_id, difficulty='medium'):
    """Rule-based AI opponent"""
    try:
        from rule_based_ai import play_rule_based_turn as rb_turn

        class MinimalEnv:
            def __init__(self, g):
                self.game_env = type('obj', (object,), {'game': g})()

        return rb_turn(MinimalEnv(game), player_id, difficulty=difficulty)
    except ImportError:
        return play_random_turn(game, player_id)


def play_neural_turn(game_env, obs, network):
    """Neural network AI turn"""
    try:
        with torch.no_grad():
            action_probs, vertex_probs, edge_probs, _, _, _ = network.policy.forward(
                torch.FloatTensor(obs['observation']).unsqueeze(0),
                torch.FloatTensor(obs['action_mask']).unsqueeze(0),
                torch.FloatTensor(obs['vertex_mask']).unsqueeze(0),
                torch.FloatTensor(obs['edge_mask']).unsqueeze(0)
            )

            action_dist = torch.distributions.Categorical(action_probs)
            action_id = action_dist.sample().item()

            vertex_dist = torch.distributions.Categorical(vertex_probs)
            vertex_id = vertex_dist.sample().item()

            edge_dist = torch.distributions.Categorical(edge_probs)
            edge_id = edge_dist.sample().item()

        next_obs, reward, terminated, truncated, info = game_env.step(
            action_id, vertex_id, edge_id, 0, 0
        )
        return next_obs, terminated or truncated
    except Exception as e:
        print(f"Neural network error: {e}")
        return obs, False


def main():
    parser = argparse.ArgumentParser(description='Visual 1v1 Catan')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model for Player 1')
    parser.add_argument('--model2', type=str, default=None, help='Path to trained model for Player 2')
    parser.add_argument('--opponent', type=str, default='random',
                       choices=['random', 'very_weak', 'weak', 'medium', 'strong'],
                       help='AI difficulty for Player 2 (if no model2)')
    parser.add_argument('--human', action='store_true', help='Play as Player 1 (human vs AI)')
    parser.add_argument('--speed', type=float, default=0.3, help='Game speed (seconds per move)')
    parser.add_argument('--vp', type=int, default=10, help='Victory points to win')
    args = parser.parse_args()

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Catan 1v1")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 26)
    small_font = pygame.font.SysFont(None, 20)
    big_font = pygame.font.SysFont(None, 48)

    # Center board
    offset = (SCREEN_W // 2 - 50, SCREEN_H // 2 - 30)

    # Create game environment
    game_env = SimplifiedRewardWrapper(player_id=0, victory_points_to_win=args.vp, num_players=2)
    obs, _ = game_env.reset()
    game = game_env.game_env.game

    # Load models
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    network1 = None
    network2 = None

    if args.model and not args.human:
        print(f"Loading Player 1 model: {args.model}")
        try:
            network1 = NetworkWrapper(model_path=args.model, device=device)
            print(f"Player 1: Neural Network on {device}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    if args.model2:
        print(f"Loading Player 2 model: {args.model2}")
        try:
            network2 = NetworkWrapper(model_path=args.model2, device=device)
            print(f"Player 2: Neural Network on {device}")
        except Exception as e:
            print(f"Failed to load model2: {e}")

    # Set player names
    if args.human:
        game.players[0].name = "YOU"
        game.players[0].color = (50, 255, 50)  # Green for human
    elif network1:
        game.players[0].name = "Neural Net"
    else:
        game.players[0].name = "Random AI"

    if network2:
        game.players[1].name = "Neural Net 2"
    else:
        diff_name = args.opponent.replace('_', ' ').title()
        game.players[1].name = f"{diff_name} AI"

    print(f"\n{'='*50}")
    print(f"1v1 CATAN - First to {args.vp} VP wins!")
    print(f"{'='*50}")
    print(f"Player 1: {game.players[0].name}")
    print(f"Player 2: {game.players[1].name}")
    if args.human:
        print("\nControls:")
        print("  D - Roll dice")
        print("  T - End turn")
        print("  Click - Build (settlement/city/road)")
        print("  1/2/3 - Switch build mode")
    print("  ESC - Quit")
    print(f"{'='*50}\n")

    running = True
    build_mode = "SETTLEMENT"
    last_move_time = pygame.time.get_ticks()
    game_over = False
    winner = None

    while running:
        current_player = game.get_current_player()
        current_id = game.players.index(current_player)
        is_human_turn = args.human and current_id == 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if is_human_turn and not game_over:
                    if event.key == pygame.K_d and game.can_roll_dice():
                        game.roll_dice()
                    elif event.key == pygame.K_t and game.can_end_turn():
                        game.end_turn()
                    elif event.key == pygame.K_1:
                        build_mode = "SETTLEMENT"
                    elif event.key == pygame.K_2:
                        build_mode = "CITY"
                    elif event.key == pygame.K_3:
                        build_mode = "ROAD"

            elif event.type == pygame.MOUSEBUTTONDOWN and is_human_turn and not game_over:
                mouse_pos = event.pos

                if game.is_initial_placement_phase():
                    if not game.waiting_for_road:
                        vertex = find_closest_vertex(game.game_board, mouse_pos, offset)
                        if vertex:
                            game.try_place_initial_settlement(vertex)
                    else:
                        edge = find_closest_edge(game.game_board, mouse_pos, offset)
                        if edge:
                            game.try_place_initial_road(edge)
                elif game.can_trade_or_build():
                    if build_mode == "SETTLEMENT":
                        vertex = find_closest_vertex(game.game_board, mouse_pos, offset)
                        if vertex:
                            current_player.try_build_settlement(vertex, False)
                    elif build_mode == "CITY":
                        vertex = find_closest_vertex(game.game_board, mouse_pos, offset)
                        if vertex:
                            current_player.try_build_city(vertex)
                    elif build_mode == "ROAD":
                        edge = find_closest_edge(game.game_board, mouse_pos, offset)
                        if edge:
                            current_player.try_build_road(edge)

        # AI turns (non-human)
        if not game_over and not is_human_turn:
            current_time = pygame.time.get_ticks()
            if current_time - last_move_time >= args.speed * 1000:
                last_move_time = current_time

                if current_id == 0 and network1:
                    # Player 1 neural network
                    obs, done = play_neural_turn(game_env, obs, network1)
                    if done:
                        game_over = True
                elif current_id == 1 and network2:
                    # Player 2 neural network (need separate env)
                    play_rule_based_turn(game, current_id, args.opponent)
                    obs = game_env.env._get_obs()
                elif current_id == 0:
                    # Player 1 random/rule-based
                    play_random_turn(game, current_id)
                else:
                    # Player 2 rule-based
                    if args.opponent == 'random':
                        play_random_turn(game, current_id)
                    else:
                        play_rule_based_turn(game, current_id, args.opponent)
                    obs = game_env.env._get_obs()

        # Check for winner
        if not game_over:
            winner = game.check_victory_conditions()
            if winner:
                game_over = True

        # Draw everything
        screen.fill((20, 40, 60))

        # Draw board
        draw_board(screen, game.game_board, offset)

        # Draw player panels at top
        panel_width = 220
        draw_player_panel(screen, game.players[0], 20, 20, panel_width,
                         current_id == 0, font, small_font)
        draw_player_panel(screen, game.players[1], SCREEN_W - panel_width - 20, 20, panel_width,
                         current_id == 1, font, small_font)

        # Draw game info at bottom
        draw_game_info(screen, game, font, small_font, args.vp)

        # Human mode indicators
        if is_human_turn and not game_over:
            # Build mode indicator
            mode_text = small_font.render(f"Build: {build_mode} (1/2/3 to change)", True, (255, 255, 0))
            screen.blit(mode_text, (SCREEN_W // 2 - 100, SCREEN_H - 70))

            # Action hints
            if game.can_roll_dice():
                hint = font.render("Press D to roll dice", True, (255, 100, 100))
                screen.blit(hint, (SCREEN_W // 2 - 80, 160))
            elif game.can_end_turn():
                hint = small_font.render("T to end turn", True, (100, 255, 100))
                screen.blit(hint, (SCREEN_W // 2 - 50, SCREEN_H - 50))

        # Victory screen
        if game_over and winner:
            # Overlay
            overlay = pygame.Surface((SCREEN_W, SCREEN_H))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            # Victory box
            box_w, box_h = 400, 200
            box_x = (SCREEN_W - box_w) // 2
            box_y = (SCREEN_H - box_h) // 2

            pygame.draw.rect(screen, (30, 30, 50), (box_x, box_y, box_w, box_h), border_radius=15)
            pygame.draw.rect(screen, (255, 215, 0), (box_x, box_y, box_w, box_h), 4, border_radius=15)

            # Winner text
            win_text = big_font.render("VICTORY!", True, (255, 215, 0))
            win_rect = win_text.get_rect(center=(SCREEN_W // 2, box_y + 50))
            screen.blit(win_text, win_rect)

            name_text = font.render(f"{winner.name} wins!", True, winner.color)
            name_rect = name_text.get_rect(center=(SCREEN_W // 2, box_y + 100))
            screen.blit(name_text, name_rect)

            vp_text = font.render(f"{winner.calculate_victory_points()} Victory Points", True, (255, 255, 255))
            vp_rect = vp_text.get_rect(center=(SCREEN_W // 2, box_y + 140))
            screen.blit(vp_text, vp_rect)

            hint_text = small_font.render("Press ESC to exit", True, (150, 150, 150))
            hint_rect = hint_text.get_rect(center=(SCREEN_W // 2, box_y + 175))
            screen.blit(hint_text, hint_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
