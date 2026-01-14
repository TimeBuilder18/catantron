"""
Visual AI Training Environment for Catan

Combines:
- AI training interface (observations, actions, step function)
- Pygame visualization (watch the AI play)
- Simplified game (NO TRADING - too complex for AI)

USAGE:
------
# Watch AI play with trained model
python visual_ai_game.py --model models/curriculum_v3_stable_phase5.pt --ai-difficulty medium

# Watch random agents
python visual_ai_game.py --ai-difficulty random

# Manual control mode
python visual_ai_game.py --manual
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
                         Settlement, City, Road, DevelopmentCardType, GameConstants)
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
    """Assign resources and numbers"""
    all_resources = RESOURCES.copy()
    random.shuffle(all_resources)

    for i, tile in enumerate(tiles):
        if i < len(all_resources):
            tile.resource = all_resources[i]
        else:
            tile.resource = "forest"

    # Place robber on desert
    for tile in tiles:
        if tile.resource == "desert":
            tile.number = None
            robber.move_to_tile(tile)
            break

    # Assign numbers to non-desert tiles
    nums = NUMBER_TOKENS.copy()
    random.shuffle(nums)
    non_desert = [t for t in tiles if t.resource != "desert"]
    for i, tile in enumerate(non_desert):
        if i < len(nums):
            tile.number = nums[i]


def compute_center_offset(tiles, screen_w, screen_h):
    """Compute offset to center the board"""
    if not tiles:
        return 0, 0
    xs = [t.x for t in tiles]
    ys = [t.y for t in tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    board_cx, board_cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    return screen_w / 2 - board_cx, screen_h / 2 - board_cy


def draw_game_board(screen, game_board, offset):
    """Draw the game board with tiles, vertices, and edges"""
    # Draw tiles
    for tile in game_board.tiles:
        color = RESOURCE_COLORS.get(tile.resource, RESOURCE_COLORS["desert"])
        tile.draw(screen, fill_color=color, offset=offset, show_coords=False)

    # Draw edges (roads)
    for edge in game_board.edges:
        x1, y1 = edge.vertex1.x + offset[0], edge.vertex1.y + offset[1]
        x2, y2 = edge.vertex2.x + offset[0], edge.vertex2.y + offset[1]

        if edge.structure:
            player_color = edge.structure.player.color
            pygame.draw.line(screen, player_color, (x1, y1), (x2, y2), 5)
        else:
            pygame.draw.line(screen, (150, 150, 150), (x1, y1), (x2, y2), 1)

    # Draw vertices (settlements/cities)
    for vertex in game_board.vertices:
        x = vertex.x + offset[0]
        y = vertex.y + offset[1]

        if vertex.structure:
            player_color = vertex.structure.player.color
            if isinstance(vertex.structure, Settlement):
                pygame.draw.circle(screen, player_color, (int(x), int(y)), 10)
                pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 10, 2)
            elif isinstance(vertex.structure, City):
                pygame.draw.rect(screen, player_color, (x - 12, y - 12, 24, 24))
                pygame.draw.rect(screen, (0, 0, 0), (x - 12, y - 12, 24, 24), 2)
        else:
            pygame.draw.circle(screen, (100, 100, 100), (int(x), int(y)), 3)


def draw_player_info(screen, game_system, font, small_font):
    """Draw player information panel on the right side"""
    panel_x = 820
    panel_y = 20

    title = font.render("PLAYERS", True, (255, 255, 255))
    screen.blit(title, (panel_x, panel_y))
    panel_y += 35

    for i, player in enumerate(game_system.players):
        # Player header
        is_current = (player == game_system.get_current_player())
        bg_color = (60, 60, 60) if is_current else (30, 30, 30)
        pygame.draw.rect(screen, bg_color, (panel_x, panel_y, 360, 130), border_radius=8)

        if is_current:
            pygame.draw.rect(screen, player.color, (panel_x, panel_y, 360, 130), 3, border_radius=8)

        # Player name and color indicator
        pygame.draw.circle(screen, player.color, (panel_x + 15, panel_y + 15), 8)
        name_text = font.render(player.name, True, player.color)
        screen.blit(name_text, (panel_x + 30, panel_y + 8))

        # Victory points (BIG)
        vp = player.calculate_victory_points()
        vp_text = font.render(f"VP: {vp}", True, (255, 255, 0))
        screen.blit(vp_text, (panel_x + 280, panel_y + 8))

        y = panel_y + 35

        # Resources
        total_resources = player.get_total_resources()
        res_text = small_font.render(f"Resources: {total_resources}", True, (200, 200, 200))
        screen.blit(res_text, (panel_x + 10, y))
        y += 22

        # Buildings
        buildings_text = small_font.render(
            f"🏘️ {len(player.settlements)}  🏰 {len(player.cities)}  🛤️  {len(player.roads)}",
            True, (200, 200, 200)
        )
        screen.blit(buildings_text, (panel_x + 10, y))
        y += 22

        # Dev cards
        total_dev = sum(player.development_cards.values())
        dev_text = small_font.render(f"Dev Cards: {total_dev}", True, (200, 200, 200))
        screen.blit(dev_text, (panel_x + 10, y))
        y += 22

        # Special achievements
        if player.has_longest_road:
            lr_text = small_font.render("🏆 Longest Road", True, (255, 215, 0))
            screen.blit(lr_text, (panel_x + 10, y))
            y += 20
        if player.has_largest_army:
            la_text = small_font.render("⚔️  Largest Army", True, (255, 215, 0))
            screen.blit(la_text, (panel_x + 10, y))

        panel_y += 145


def draw_game_state(screen, game_system, font, small_font):
    """Draw current game state info"""
    panel_x = 820
    panel_y = 620

    # Background
    pygame.draw.rect(screen, (40, 40, 40), (panel_x, panel_y, 360, 180), border_radius=8)
    pygame.draw.rect(screen, (100, 100, 100), (panel_x, panel_y, 360, 180), 2, border_radius=8)

    y = panel_y + 15

    # Phase
    phase_text = font.render(f"Phase: {game_system.game_phase}", True, (255, 255, 255))
    screen.blit(phase_text, (panel_x + 15, y))
    y += 30

    # Turn phase
    if game_system.game_phase == "NORMAL_PLAY":
        turn_text = small_font.render(f"Turn Phase: {game_system.turn_phase}", True, (200, 200, 200))
        screen.blit(turn_text, (panel_x + 15, y))
        y += 25

    # Last dice roll
    if game_system.last_dice_roll:
        dice = game_system.last_dice_roll
        dice_text = font.render(f"🎲 Rolled: {dice[2]} ({dice[0]}+{dice[1]})", True, (255, 255, 100))
        screen.blit(dice_text, (panel_x + 15, y))
        y += 30

    # Current action needed
    if game_system.is_initial_placement_phase():
        action_text = small_font.render(game_system.get_current_player_needs(), True, (100, 255, 100))
        screen.blit(action_text, (panel_x + 15, y))
    elif game_system.can_roll_dice():
        action_text = small_font.render("⏳ Waiting for dice roll (D key)", True, (255, 100, 100))
        screen.blit(action_text, (panel_x + 15, y))
    elif game_system.can_end_turn():
        action_text = small_font.render("✓ Can end turn (T key)", True, (100, 255, 100))
        screen.blit(action_text, (panel_x + 15, y))


def draw_controls(screen, font, small_font):
    """Draw control instructions"""
    panel_x = 820
    panel_y = 820

    title = font.render("CONTROLS", True, (255, 255, 255))
    screen.blit(title, (panel_x, panel_y))
    y = panel_y + 30

    controls = [
        "D - Roll Dice",
        "T - End Turn",
        "1 - Settlement Mode",
        "2 - City Mode",
        "3 - Road Mode",
        "X - Buy Dev Card",
        "Click - Build"
    ]

    for ctrl in controls:
        text = small_font.render(ctrl, True, (180, 180, 180))
        screen.blit(text, (panel_x, y))
        y += 22


def find_closest_vertex(game_board, mouse_pos, offset, max_distance=15):
    """Find the closest vertex to mouse position"""
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


def find_closest_edge(game_board, mouse_pos, offset, max_distance=10):
    """Find the closest edge to mouse position"""
    mouse_x, mouse_y = mouse_pos
    closest_edge = None
    closest_distance = max_distance

    for edge in game_board.edges:
        x1, y1 = edge.vertex1.x + offset[0], edge.vertex1.y + offset[1]
        x2, y2 = edge.vertex2.x + offset[0], edge.vertex2.y + offset[1]

        # Calculate distance from point to line segment
        A = mouse_x - x1
        B = mouse_y - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D

        if len_sq == 0:
            distance = ((mouse_x - x1) ** 2 + (mouse_y - y1) ** 2) ** 0.5
        else:
            param = dot / len_sq
            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D

            distance = ((mouse_x - xx) ** 2 + (mouse_y - yy) ** 2) ** 0.5

        if distance < closest_distance:
            closest_distance = distance
            closest_edge = edge

    return closest_edge


class VisualAIEnvironment:
    """
    Visual AI Training Environment
    - Shows game state with pygame
    - Provides AI interface (observations, actions)
    - Simplified: NO TRADING
    """

    def __init__(self, screen, offset, font, small_font):
        self.screen = screen
        self.offset = offset
        self.font = font
        self.small_font = small_font

        # Create game
        tile_size = 50
        tiles = create_hexagonal_board(tile_size, radius=2)

        for t in tiles:
            t.find_neighbors(tiles)

        self.robber = Robber()
        assign_resources_numbers(tiles, self.robber)

        game_board = GameBoard(tiles)

        # Create 4 AI players
        players = [
            Player("AI 1", (255, 50, 50)),
            Player("AI 2", (50, 50, 255)),
            Player("AI 3", (255, 255, 50)),
            Player("AI 4", (255, 255, 255))
        ]

        self.game = GameSystem(game_board, players)
        self.game.robber = self.robber

        #print("✅ Visual AI Environment Ready!")

    def draw(self):
        """Draw the game state"""
        self.screen.fill((20, 20, 20))

        draw_game_board(self.screen, self.game.game_board, self.offset)
        draw_player_info(self.screen, self.game, self.font, self.small_font)
        draw_game_state(self.screen, self.game, self.font, self.small_font)
        draw_controls(self.screen, self.font, self.small_font)

        pygame.display.flip()

    def get_observation(self, player_index):
        """Get observation for AI agent (same as ai_interface.py)"""
        player = self.game.players[player_index]
        current_player_idx = self.game.players.index(self.game.get_current_player())

        obs = {
            'current_player': current_player_idx,
            'is_my_turn': (current_player_idx == player_index),
            'game_phase': self.game.game_phase,
            'turn_phase': self.game.turn_phase,
            'dice_rolled': self.game.dice_rolled,
            'last_roll': self.game.last_dice_roll,

            'my_resources': dict(player.resources),
            'my_dev_cards': dict(player.development_cards),
            'my_settlements': len(player.settlements),
            'my_cities': len(player.cities),
            'my_roads': len(player.roads),
            'my_victory_points': player.calculate_victory_points(),

            'opponents': [
                {
                    'resource_count': sum(p.resources.values()),
                    'settlements': len(p.settlements),
                    'cities': len(p.cities),
                    'roads': len(p.roads),
                    'victory_points': p.calculate_victory_points()
                }
                for i, p in enumerate(self.game.players) if i != player_index
            ],

            'tiles': [(t.q, t.r, t.resource, t.number) for t in self.game.game_board.tiles],
            'legal_actions': self.get_legal_actions(player_index)
        }

        return obs

    def get_legal_actions(self, player_index):
        """Get legal actions (NO TRADING)"""
        player = self.game.players[player_index]
        current_player = self.game.get_current_player()

        if player != current_player:
            return ['wait']

        actions = []

        if self.game.is_initial_placement_phase():
            if self.game.waiting_for_road:
                actions.append('place_road')
            else:
                actions.append('place_settlement')
        else:
            if self.game.can_roll_dice():
                actions.append('roll_dice')

            if self.game.can_trade_or_build():
                actions.append('build_settlement')
                actions.append('build_city')
                actions.append('build_road')
                actions.append('buy_dev_card')

            if self.game.can_end_turn():
                actions.append('end_turn')

        return actions


def play_random_turn(game, player_id):
    """Random AI opponent"""
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


def play_opponent_turn(game, player_id, ai_difficulty='random'):
    """Play opponent turn with specified AI difficulty"""
    if ai_difficulty == 'random':
        return play_random_turn(game, player_id)
    else:
        try:
            from rule_based_ai import play_rule_based_turn
            class MinimalEnv:
                def __init__(self, g):
                    self.game_env = type('obj', (object,), {'game': g})()
            return play_rule_based_turn(MinimalEnv(game), player_id, difficulty=ai_difficulty)
        except ImportError:
            print(f"Warning: rule_based_ai not found, using random AI instead")
            return play_random_turn(game, player_id)


def main():
    """Main game loop"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visual Catan AI Game')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--ai-difficulty', type=str, default='random',
                       choices=['random', 'very_weak', 'weak', 'medium', 'strong'],
                       help='AI difficulty for opponents')
    parser.add_argument('--manual', action='store_true', help='Manual control mode')
    parser.add_argument('--speed', type=float, default=0.5, help='Game speed (seconds per move)')
    args = parser.parse_args()

    pygame.init()
    pygame.font.init()

    SCREEN_W, SCREEN_H = 1200, 1000
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Visual AI Training Environment - Catan")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 28)
    small_font = pygame.font.SysFont(None, 22)

    # Center the board on left side
    offset = (400, SCREEN_H / 2)

    if args.manual:
        # Original manual control mode
        env = VisualAIEnvironment(screen, offset, font, small_font)
        build_mode = "SETTLEMENT"
        print("\n" + "="*60)
        print("MANUAL CONTROL MODE")
        print("="*60)
        print("Controls: D (dice), T (end turn), 1/2/3 (build mode), X (dev card), Click (build)")
        print("="*60 + "\n")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        if env.game.can_roll_dice():
                            result = env.game.roll_dice()
                            if result:
                                #print(f"🎲 Rolled: {result[2]} ({result[0]}+{result[1]})")
                                pass

                    elif event.key == pygame.K_t:
                        success, msg = env.game.end_turn()
                        if success:
                            #print(f"✓ {msg}")
                            # Check for winner
                            winner = env.game.check_victory_conditions()
                            if winner:
                                #print(f"\n🏆 {winner.name} WINS with {winner.victory_points} points!")
                                pass

                    elif event.key == pygame.K_1:
                        build_mode = "SETTLEMENT"
                        #print("Mode: Settlement")

                    elif event.key == pygame.K_2:
                        build_mode = "CITY"
                        #print("Mode: City")

                    elif event.key == pygame.K_3:
                        build_mode = "ROAD"
                        #print("Mode: Road")

                    elif event.key == pygame.K_x:
                        if env.game.can_trade_or_build():
                            success, msg = env.game.get_current_player().try_buy_development_card(env.game.dev_deck)
                            print(msg)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos

                    # Only handle clicks in game area (left side)
                    if mouse_pos[0] < 800:
                        if env.game.is_initial_placement_phase():
                            if not env.game.waiting_for_road:
                                vertex = find_closest_vertex(env.game.game_board, mouse_pos, offset)
                                if vertex:
                                    success, msg = env.game.try_place_initial_settlement(vertex)
                                    print(msg)
                            else:
                                edge = find_closest_edge(env.game.game_board, mouse_pos, offset)
                                if edge:
                                    success, msg = env.game.try_place_initial_road(edge)
                                    print(msg)
                        else:
                            if env.game.can_trade_or_build():
                                current_player = env.game.get_current_player()

                                if build_mode == "SETTLEMENT":
                                    vertex = find_closest_vertex(env.game.game_board, mouse_pos, offset)
                                    if vertex:
                                        success, msg = current_player.try_build_settlement(vertex, False)
                                        print(msg)

                                elif build_mode == "CITY":
                                    vertex = find_closest_vertex(env.game.game_board, mouse_pos, offset)
                                    if vertex:
                                        success, msg = current_player.try_build_city(vertex)
                                        print(msg)

                                elif build_mode == "ROAD":
                                    edge = find_closest_edge(env.game.game_board, mouse_pos, offset)
                                    if edge:
                                        success, msg = current_player.try_build_road(edge)
                                        print(msg)

            # Draw everything
            env.draw()
            clock.tick(60)

    else:
        # AI mode - watch agents play
        print("\n" + "="*60)
        print("AI VISUALIZATION MODE")
        print("="*60)
        if args.model:
            print(f"Loading model: {args.model}")
        print(f"Opponent AI: {args.ai_difficulty}")
        print(f"Game speed: {args.speed}s per move")
        print("Press ESC to quit")
        print("="*60 + "\n")

        # Create environment wrapper for proper observations (needed for model testing)
        game_env = SimplifiedRewardWrapper(player_id=0, victory_points_to_win=10)
        obs, _ = game_env.reset()

        # Load model if provided
        network = None
        if args.model:
            # Use GPU if available for better performance
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

            print(f"Loading model on {device}...")
            try:
                network = NetworkWrapper(model_path=args.model, device=device)
                print(f"✅ Model loaded on {device}")

                # Test the model with actual observation
                print("Testing model output...")
                with torch.no_grad():
                    action_probs, vertex_probs, edge_probs, _, _, value = network.policy.forward(
                        torch.FloatTensor(obs['observation']).unsqueeze(0),
                        torch.FloatTensor(obs['action_mask']).unsqueeze(0),
                        torch.FloatTensor(obs['vertex_mask']).unsqueeze(0),
                        torch.FloatTensor(obs['edge_mask']).unsqueeze(0)
                    )

                    # Check for NaN in any output
                    has_nan = (torch.isnan(action_probs).any() or
                              torch.isnan(vertex_probs).any() or
                              torch.isnan(edge_probs).any() or
                              torch.isnan(value).any())

                    if has_nan:
                        print("⚠️  WARNING: Model outputs NaN values!")
                        print(f"   Action probs NaN: {torch.isnan(action_probs).any()}")
                        print(f"   Vertex probs NaN: {torch.isnan(vertex_probs).any()}")
                        print(f"   Edge probs NaN: {torch.isnan(edge_probs).any()}")
                        print(f"   Value NaN: {torch.isnan(value).any()}")
                        print("   Model weights are corrupted - falling back to random AI")
                        network = None
                    else:
                        print("✅ Model test passed - outputs are valid")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to random AI for Player 0")
                network = None

        # Create visual environment with the same game instance
        visual_env = VisualAIEnvironment(screen, offset, font, small_font)
        visual_env.game = game_env.game_env.game

        # Update player names to show AI types
        if network:
            visual_env.game.players[0].name = "🧠 Neural Net (RED)"
        else:
            visual_env.game.players[0].name = "🎲 Random (RED)"

        ai_type = args.ai_difficulty.capitalize() if args.ai_difficulty != 'random' else 'Random'
        visual_env.game.players[1].name = f"🤖 {ai_type} (BLUE)"
        visual_env.game.players[2].name = f"🤖 {ai_type} (YELLOW)"
        visual_env.game.players[3].name = f"🤖 {ai_type} (WHITE)"

        print("🎮 Game started! Player 0 is " + ("Neural Network" if network else "Random AI"))

        running = True
        moves = 0
        max_moves = 500
        last_move_time = pygame.time.get_ticks()

        while running and moves < max_moves:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # AI turn logic (throttled by speed setting)
            current_time = pygame.time.get_ticks()
            if current_time - last_move_time >= args.speed * 1000:
                last_move_time = current_time

                game = game_env.game_env.game
                current_player = game.get_current_player()
                current_id = game.players.index(current_player)

                if current_id == 0:
                    # Player 0 - Neural network or random
                    moves += 1
                    if network:
                        try:
                            # Get action from network
                            action_probs, vertex_probs, edge_probs, _, _, _ = network.policy.forward(
                                torch.FloatTensor(obs['observation']).unsqueeze(0),
                                torch.FloatTensor(obs['action_mask']).unsqueeze(0),
                                torch.FloatTensor(obs['vertex_mask']).unsqueeze(0),
                                torch.FloatTensor(obs['edge_mask']).unsqueeze(0)
                            )

                            # Sample actions
                            action_dist = torch.distributions.Categorical(action_probs)
                            action_id = action_dist.sample().item()

                            vertex_dist = torch.distributions.Categorical(vertex_probs)
                            vertex_id = vertex_dist.sample().item()

                            edge_dist = torch.distributions.Categorical(edge_probs)
                            edge_id = edge_dist.sample().item()

                            # Step environment
                            next_obs, reward, terminated, truncated, info = game_env.step(
                                action_id, vertex_id, edge_id, 0, 0
                            )
                            obs = next_obs

                            if terminated or truncated:
                                winner = game.check_victory_conditions()
                                if winner:
                                    print(f"\n🏆 {winner.name} WINS with {winner.victory_points} points!")
                                running = False
                        except Exception as e:
                            print(f"Error in neural network turn: {e}")
                            import traceback
                            traceback.print_exc()
                            # Fallback to random
                            play_random_turn(game, current_id)
                    else:
                        # No model loaded, use random AI
                        play_random_turn(game, current_id)
                else:
                    # Opponent AI (random or rule-based)
                    success = play_opponent_turn(game, current_id, args.ai_difficulty)
                    if not success and game.can_end_turn():
                        game.end_turn()

                    # Update observation for player 0 after opponent turn
                    # game_env.env is CatanEnv, which has _get_obs() method
                    obs = game_env.env._get_obs()

                    winner = game.check_victory_conditions()
                    if winner:
                        print(f"\n🏆 {winner.name} WINS with {winner.victory_points} points!")
                        running = False

            # Draw everything
            visual_env.draw()
            clock.tick(60)

        if moves >= max_moves:
            print(f"\n⏱️  Game ended after {max_moves} moves (max limit reached)")

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
