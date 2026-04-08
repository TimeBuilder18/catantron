"""
Catantron — Polished Catan Game GUI
Entry point with title screen, game modes, and visual gameplay.

Usage:
    python main.py
"""

import pygame
import sys
import random
import math
import time
from enum import Enum

try:
    import torch
    from model.model_loader import NetworkWrapper
    from environment.catan_env import CatanEnv
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from game.hexagon import Tile
from game.game_system import (Player, Robber, GameBoard, GameSystem, ResourceType,
                         Settlement, City, Road, DevelopmentCardType, GameConstants,
                         PortType)
from gui import gui_components
from gui.gui_components import (
    BG_COLOR, PANEL_BG, PANEL_BORDER,
    TEXT_COLOR, TEXT_DIM, GOLD, WHITE, BLACK,
    RESOURCE_COLORS, RESOURCE_DISPLAY_NAMES, RESOURCE_TYPE_NAMES,
    RESOURCE_CARD_COLORS, BUILD_COSTS, DICE_PIPS,
    get_font, draw_game_board, draw_buildable_highlights,
    draw_resource_panel, draw_player_info_box, draw_vp_tracker,
    draw_button, draw_game_state_panel, draw_game_log,
    draw_title_screen, draw_difficulty_screen, set_screen_size,
    draw_robber_overlay, draw_dev_card_popup, draw_dev_cards_panel,
    draw_discard_overlay, draw_trade_overlay,
    draw_watch_select_screen,
)


# --- Standard Catan Setup ---
NUMBER_TOKENS = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
RESOURCES = (["forest"] * 4 + ["hill"] * 3 + ["field"] * 4 +
             ["mountain"] * 3 + ["pasture"] * 4 + ["desert"])


# ===========================================================================
# Game State
# ===========================================================================

class GameState(Enum):
    TITLE = "title"
    DIFFICULTY_SELECT = "difficulty"
    WATCH_SELECT = "watch_select"
    PLAYING = "playing"
    VICTORY = "victory"


class GameMode(Enum):
    PVP = "pvp"
    VS_AI = "vs_ai"
    VS_BOT = "vs_bot"
    WATCH_AI = "watch_ai"


# ===========================================================================
# Board Setup (reused from visual_ai_game.py)
# ===========================================================================

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
        tile.resource = all_resources[i] if i < len(all_resources) else "forest"

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


def compute_center_offset(tiles, target_x, target_y):
    if not tiles:
        return (0, 0)
    xs = [t.x for t in tiles]
    ys = [t.y for t in tiles]
    board_cx = (min(xs) + max(xs)) / 2
    board_cy = (min(ys) + max(ys)) / 2
    return (target_x - board_cx, target_y - board_cy)


# ===========================================================================
# Interaction Helpers (reused from visual_ai_game.py)
# ===========================================================================

def find_closest_vertex(game_board, mouse_pos, offset, max_distance=15):
    mx, my = mouse_pos
    closest = None
    best_dist = max_distance
    for v in game_board.vertices:
        x = v.x + offset[0]
        y = v.y + offset[1]
        d = math.sqrt((mx - x) ** 2 + (my - y) ** 2)
        if d < best_dist:
            best_dist = d
            closest = v
    return closest


def find_closest_edge(game_board, mouse_pos, offset, max_distance=12):
    mx, my = mouse_pos
    closest = None
    best_dist = max_distance
    for edge in game_board.edges:
        x1, y1 = edge.vertex1.x + offset[0], edge.vertex1.y + offset[1]
        x2, y2 = edge.vertex2.x + offset[0], edge.vertex2.y + offset[1]
        A, B = mx - x1, my - y1
        C, D = x2 - x1, y2 - y1
        dot = A * C + B * D
        len_sq = C * C + D * D
        if len_sq == 0:
            d = math.sqrt((mx - x1) ** 2 + (my - y1) ** 2)
        else:
            param = dot / len_sq
            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx, yy = x1 + param * C, y1 + param * D
            d = math.sqrt((mx - xx) ** 2 + (my - yy) ** 2)
        if d < best_dist:
            best_dist = d
            closest = edge
    return closest


def find_closest_tile(game_board, mouse_pos, offset, max_distance=40):
    """Find the closest tile to mouse position."""
    mx, my = mouse_pos
    closest = None
    best_dist = max_distance
    for tile in game_board.tiles:
        x = tile.x + offset[0]
        y = tile.y + offset[1]
        d = math.sqrt((mx - x) ** 2 + (my - y) ** 2)
        if d < best_dist:
            best_dist = d
            closest = tile
    return closest


# ===========================================================================
# AI Helpers (reused from visual_ai_game.py)
# ===========================================================================

def _find_best_robber_tile(game, my_player_id):
    from game.game_system import City as CityClass
    pip_values = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    my_player = game.players[my_player_id]
    current_robber_tile = game.robber.position
    best_tile = None
    best_score = -1
    for tile in game.game_board.tiles:
        if tile == current_robber_tile or tile.resource is None:
            continue
        if tile.resource == "desert":
            continue
        score = 0
        pips = pip_values.get(tile.number, 0) if tile.number else 0
        for vertex in game.game_board.vertices:
            if tile in vertex.adjacent_tiles and vertex.structure:
                if vertex.structure.player != my_player:
                    mult = 2 if isinstance(vertex.structure, CityClass) else 1
                    score += pips * mult
        if score > best_score:
            best_score = score
            best_tile = tile
    return best_tile


def _auto_discard_only(game):
    """Handle discards only (no robber placement). Used when human will place robber."""
    if not game.waiting_for_discards:
        return
    for player in game.players_must_discard:
        if player in game.players_discarded:
            continue
        total = sum(player.resources.values())
        num_to_discard = total // 2
        all_cards = []
        for res_type, count in player.resources.items():
            all_cards.extend([res_type] * count)
        if all_cards and num_to_discard > 0:
            cards_to_discard = random.sample(all_cards, min(num_to_discard, len(all_cards)))
            discard_dict = {}
            for res_type in set(cards_to_discard):
                discard_dict[res_type] = cards_to_discard.count(res_type)
            game.discard_cards(player, discard_dict)
    game.waiting_for_discards = False
    game.players_must_discard = []
    game.players_discarded = set()


def _auto_discard_for_game(game, current_player_id=0):
    if not game.waiting_for_discards:
        return
    for player in game.players_must_discard:
        if player in game.players_discarded:
            continue
        total = sum(player.resources.values())
        num_to_discard = total // 2
        all_cards = []
        for res_type, count in player.resources.items():
            all_cards.extend([res_type] * count)
        if all_cards and num_to_discard > 0:
            cards_to_discard = random.sample(all_cards, min(num_to_discard, len(all_cards)))
            discard_dict = {}
            for res_type in set(cards_to_discard):
                discard_dict[res_type] = cards_to_discard.count(res_type)
            game.discard_cards(player, discard_dict)
    game.waiting_for_discards = False
    game.players_must_discard = []
    game.players_discarded = set()
    best_tile = _find_best_robber_tile(game, current_player_id)
    if best_tile:
        game.move_robber_to_tile(best_tile)
    else:
        available = [t for t in game.game_board.tiles
                     if t != game.robber.position and t.resource]
        if available:
            game.move_robber_to_tile(random.choice(available))


def play_random_turn(game, player_id, log=None):
    player = game.players[player_id]
    if log is None:
        log = lambda msg, color=None: None
    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            if game.last_settlement_vertex:
                valid = [e for e in game.game_board.edges
                        if e.structure is None and
                        (e.vertex1 == game.last_settlement_vertex or
                         e.vertex2 == game.last_settlement_vertex)]
                if valid:
                    game.try_place_initial_road(random.choice(valid), player)
                    log(f"{player.name} placed a road", player.color)
        else:
            valid = [v for v in game.game_board.vertices
                    if v.structure is None and
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                game.try_place_initial_settlement(random.choice(valid), player)
                log(f"{player.name} placed a settlement", player.color)
        return True

    if game.can_roll_dice():
        result = game.roll_dice()
        if result:
            log(f"{player.name} rolled {result[2]}", player.color)
        if result and result[2] == 7 and game.waiting_for_discards:
            _auto_discard_for_game(game, player_id)
        return True

    if game.can_trade_or_build():
        actions = []
        res = player.resources
        if (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
            res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1):
            v = game.get_buildable_vertices_for_settlements()
            if v:
                actions.append(('sett', v))
        if res[ResourceType.WHEAT] >= 2 and res[ResourceType.ORE] >= 3:
            v = game.get_buildable_vertices_for_cities()
            if v:
                actions.append(('city', v))
        if res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1:
            e = game.get_buildable_edges()
            if e:
                actions.append(('road', e))
        if (res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1 and
            res[ResourceType.ORE] >= 1 and not game.dev_deck.is_empty()):
            actions.append(('dev', None))
        if actions and random.random() < 0.85:
            action_type, locs = random.choice(actions)
            if action_type == 'sett':
                player.try_build_settlement(random.choice(locs))
                log(f"{player.name} built a settlement", player.color)
            elif action_type == 'city':
                player.try_build_city(random.choice(locs))
                log(f"{player.name} upgraded to a city", player.color)
            elif action_type == 'road':
                player.try_build_road(random.choice(locs))
                game.update_longest_road()
                log(f"{player.name} built a road", player.color)
            elif action_type == 'dev':
                player.try_buy_development_card(game.dev_deck)
                log(f"{player.name} bought a dev card", player.color)
            return True

    if game.can_end_turn():
        game.end_turn()
        return True
    return False


def play_neural_turn(game, player_id, network, device, catan_env, log=None):
    """Play one AI action using the neural network.

    Uses CatanEnv for observation building, then executes the chosen action
    directly on the GameSystem.
    """
    if log is None:
        log = lambda msg, color=None: None
    if not HAS_TORCH:
        return play_random_turn(game, player_id, log=log)

    try:
        player = game.players[player_id]

        # Sync env's game reference to our game
        catan_env.game_env.game = game

        # Verify env perspective matches the player taking the turn
        if catan_env.player_id != player_id:
            catan_env.player_id = player_id

        # Build observation + masks
        obs = catan_env._get_obs()

        with torch.no_grad():
            action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, _ = \
                network.policy.forward(
                    torch.FloatTensor(obs['observation']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(device),
                    torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(device)
                )

        action_id = int(torch.argmax(action_probs[0]).item())
        vertex_id = int(torch.argmax(vertex_probs[0]).item())
        edge_id = int(torch.argmax(edge_probs[0]).item())
        give_idx = int(torch.argmax(trade_give_probs[0]).item())
        get_idx = int(torch.argmax(trade_get_probs[0]).item())
        if give_idx == get_idx:
            get_idx = (give_idx + 1) % 5

        action_names = [
            'roll_dice', 'place_settlement', 'place_road',
            'build_settlement', 'build_city', 'build_road',
            'buy_dev_card', 'end_turn', 'wait', 'trade_with_bank',
            'do_nothing', 'play_knight', 'play_monopoly', 'play_year_of_plenty'
        ]
        action_name = action_names[action_id]
        board = game.game_board
        resource_map = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT,
                        ResourceType.SHEEP, ResourceType.ORE]

        if action_name == 'roll_dice':
            result = game.roll_dice()
            if result:
                log(f"{player.name} rolled {result[2]}", player.color)
            if result and result[2] == 7 and game.waiting_for_discards:
                _auto_discard_for_game(game, player_id)
            return True

        elif action_name == 'place_settlement':
            if vertex_id < len(board.vertices):
                game.try_place_initial_settlement(board.vertices[vertex_id], player)
                log(f"{player.name} placed a settlement", player.color)
            return True

        elif action_name == 'place_road':
            if edge_id < len(board.edges):
                game.try_place_initial_road(board.edges[edge_id], player)
                log(f"{player.name} placed a road", player.color)
            return True

        elif action_name == 'build_settlement':
            built = False
            if vertex_id < len(board.vertices):
                result = player.try_build_settlement(board.vertices[vertex_id])
                built = result[0] if isinstance(result, tuple) else bool(result)
            if built:
                log(f"{player.name} built a settlement", player.color)
            elif game.can_end_turn():
                game.end_turn()
            return True

        elif action_name == 'build_city':
            built = False
            if vertex_id < len(board.vertices):
                result = player.try_build_city(board.vertices[vertex_id])
                built = result[0] if isinstance(result, tuple) else bool(result)
            if built:
                log(f"{player.name} upgraded to a city", player.color)
            elif game.can_end_turn():
                game.end_turn()
            return True

        elif action_name == 'build_road':
            built = False
            if edge_id < len(board.edges):
                result = player.try_build_road(board.edges[edge_id])
                built = result[0] if isinstance(result, tuple) else bool(result)
                if built:
                    game.update_longest_road()
            if built:
                log(f"{player.name} built a road", player.color)
            elif game.can_end_turn():
                game.end_turn()
            return True

        elif action_name == 'buy_dev_card':
            result = player.try_buy_development_card(game.dev_deck)
            bought = result[0] if isinstance(result, tuple) else bool(result)
            if bought:
                log(f"{player.name} bought a dev card", player.color)
            elif game.can_end_turn():
                game.end_turn()
            return True

        elif action_name == 'end_turn':
            if game.can_end_turn():
                game.end_turn()
            return True

        elif action_name == 'trade_with_bank':
            give_res = resource_map[give_idx]
            get_res = resource_map[get_idx]
            success, msg = game.execute_bank_trade(player, give_res, get_res)
            if success:
                log(f"{player.name} traded {give_res.name} for {get_res.name}", player.color)
            else:
                # Trade failed — end turn to avoid infinite loop
                if game.can_end_turn():
                    game.end_turn()
            return True

        elif action_name == 'play_knight':
            success, _ = game.play_knight_card(player)
            if success:
                log(f"{player.name} played a Knight", player.color)
                # Move robber to a tile with opponent buildings
                available = [t for t in board.tiles
                             if t != game.robber.position and t.resource is not None]
                if available:
                    # Pick tile adjacent to opponents
                    best = None
                    for t in available:
                        for v in board.vertices:
                            if t in v.adjacent_tiles and v.structure and v.structure.player != player:
                                best = t
                                break
                        if best:
                            break
                    game.move_robber_to_tile(best or random.choice(available))
                    # Steal from opponent on that tile
                    for v in board.vertices:
                        tile = best or available[0]
                        if tile in v.adjacent_tiles and v.structure and v.structure.player != player:
                            game.steal_random_resource(player, v.structure.player)
                            break
            return True

        elif action_name in ('play_monopoly', 'play_year_of_plenty'):
            # Use trade_give/get indices as resource choices
            if action_name == 'play_monopoly':
                game.play_monopoly_card(player, resource_map[give_idx])
                log(f"{player.name} played Monopoly on {resource_map[give_idx].name}", player.color)
            else:
                game.play_year_of_plenty_card(player, resource_map[give_idx], resource_map[get_idx])
                log(f"{player.name} played Year of Plenty", player.color)
            return True

        else:  # wait, do_nothing
            if game.can_end_turn():
                game.end_turn()
            return True

    except Exception as e:
        print(f"[Neural AI] ERROR for player {player_id}: {e}")
        import traceback
        traceback.print_exc()
        print(f"[Neural AI] Falling back to random for player {player_id}")
        return play_random_turn(game, player_id, log=log)


def play_opponent_turn(game, player_id, ai_difficulty='random'):
    if ai_difficulty == 'random':
        return play_random_turn(game, player_id)
    try:
        from ai.scripted_opponents import play_rule_based_turn
        class MinimalEnv:
            def __init__(self, g):
                self.game_env = type('obj', (object,), {'game': g})()
        return play_rule_based_turn(MinimalEnv(game), player_id, difficulty=ai_difficulty)
    except ImportError:
        return play_random_turn(game, player_id)


# ===========================================================================
# Main Game Application
# ===========================================================================

class CatantronApp:
    def __init__(self):
        pygame.init()
        pygame.font.init()

        # Detect monitor size and scale window to fit
        info = pygame.display.Info()
        max_w = min(info.current_w - 60, 1600)
        max_h = min(info.current_h - 100, 1000)
        # Use uniform scale to keep aspect ratio
        self.scale = min(max_w / 1600, max_h / 1000)
        self.screen_w = int(1600 * self.scale)
        self.screen_h = int(1000 * self.scale)
        set_screen_size(self.screen_w, self.screen_h)

        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("Catantron")
        self.clock = pygame.time.Clock()

        self.font = get_font(max(16, int(22 * self.scale)), bold=True)
        self.small_font = get_font(max(12, int(16 * self.scale)))
        self.big_font = get_font(max(24, int(36 * self.scale)), bold=True)

        self.state = GameState.TITLE
        self.mode = None
        self.difficulty = None
        self.buttons_hovered = {}

        # Game state
        self.game = None
        self.game_board = None
        self.offset = (0, 0)
        self.build_mode = "SETTLEMENT"
        self.messages = []  # (text, color) pairs

        # Robber mode
        self.robber_mode = False

        # Discard mode
        self.discard_mode = False
        self.discard_selection = {}  # {ResourceType: count_to_discard}
        self.discard_required = 0
        self.discard_buttons = []  # clickable rects from draw_discard_overlay

        # Trade mode
        self.trade_mode = False
        self.trade_give = None   # ResourceType to give
        self.trade_get = None    # ResourceType to receive
        self.trade_buttons = []  # clickable rects from draw_trade_overlay

        # Dev card popup
        self.dev_card_popup = None  # {"card_name": str, "description": str, "show_until": float}

        # Neural network AI
        self.neural_network = None
        self.neural_device = None
        self.neural_env = None

        # Players: which indices are human
        self.human_players = set()

    def add_message(self, text, color=TEXT_COLOR):
        self.messages.append((text, color))
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]

    def setup_game(self, mode, difficulty=None):
        """Initialize a new game for the given mode."""
        self.mode = mode
        self.difficulty = difficulty
        self.messages = []
        self.build_mode = "SETTLEMENT"

        tile_size = max(30, int(50 * self.scale))
        tiles = create_hexagonal_board(tile_size, radius=2)
        for t in tiles:
            t.find_neighbors(tiles)

        robber = Robber()
        assign_resources_numbers(tiles, robber)

        game_board = GameBoard(tiles)
        self.game_board = game_board

        # Center board in left portion of screen
        self.offset = compute_center_offset(tiles, int(460 * self.scale),
                                            self.screen_h // 2)
        self.robber_mode = False
        self.discard_mode = False
        self.discard_selection = {}
        self.discard_required = 0
        self.discard_buttons = []
        self.trade_mode = False
        self.trade_give = None
        self.trade_get = None
        self.trade_buttons = []
        self.dev_card_popup = None

        if mode == GameMode.PVP:
            p1 = Player("Player 1", (220, 60, 60))    # Red
            p2 = Player("Player 2", (60, 100, 220))    # Blue
            players = [p1, p2]
            self.human_players = {0, 1}
            self.add_message("Player vs Player — Place your first settlement!", GOLD)
        elif mode == GameMode.VS_BOT:
            p1 = Player("You", (220, 60, 60))
            diff_label = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}.get(difficulty, difficulty)
            p2 = Player(f"Bot ({diff_label})", (60, 100, 220))
            players = [p1, p2]
            self.human_players = {0}
            self.add_message(f"You vs {diff_label} Bot — Place your first settlement!", GOLD)
        elif mode == GameMode.VS_AI:
            p1 = Player("You", (220, 60, 60))
            p2 = Player("Neural AI", (60, 100, 220))
            players = [p1, p2]
            self.human_players = {0}
            self.add_message("You vs Neural AI — Place your first settlement!", GOLD)
        elif mode == GameMode.WATCH_AI:
            self.watch_opponent_difficulty = difficulty
            if difficulty == "ai_vs_ai":
                p1 = Player("Neural AI 1", (220, 60, 60))
                p2 = Player("Neural AI 2", (60, 100, 220))
                self.add_message("Watching: Neural AI vs Neural AI", GOLD)
            else:
                diff_label = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}.get(difficulty, difficulty)
                p1 = Player("Neural AI", (220, 60, 60))
                p2 = Player(f"Bot ({diff_label})", (60, 100, 220))
                self.add_message(f"Watching: Neural AI vs {diff_label} Bot", GOLD)
            players = [p1, p2]
            self.human_players = set()
        else:
            return

        self.game = GameSystem(game_board, players)
        self.game.robber = robber

        # Load neural network for VS_AI / WATCH_AI modes
        self.neural_network = None
        self.neural_device = None
        self.neural_env = None
        if mode in (GameMode.VS_AI, GameMode.WATCH_AI) and HAS_TORCH:
            self._load_neural_ai()

        self.state = GameState.PLAYING

    def _load_neural_ai(self):
        """Load neural network model for VS_AI mode."""
        import os
        model_path = "models/v3_selfplay_game61024.pt"
        if not os.path.exists(model_path):
            # Try to find any .pt model in models/
            candidates = sorted(
                [f for f in os.listdir("models") if f.endswith(".pt")],
                key=lambda f: os.path.getmtime(os.path.join("models", f)),
                reverse=True
            )
            if candidates:
                model_path = os.path.join("models", candidates[0])
                self.add_message(f"Model not found, using {candidates[0]}", (255, 200, 100))
            else:
                self.add_message("No model found — AI will play randomly", (255, 100, 100))
                return

        try:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

            self.neural_device = device
            self.neural_network = NetworkWrapper(model_path=model_path, device=device, hidden_dim=512)

            # Create CatanEnv per player so each gets correct perspective
            self.neural_envs = {}
            for pid in range(len(self.game.players)):
                env = CatanEnv(player_id=pid, num_players=2)
                env.game_env.game = self.game
                self.neural_envs[pid] = env
            # Keep neural_env for backward compat (VS_AI uses player 1)
            self.neural_env = self.neural_envs.get(1, list(self.neural_envs.values())[0])

            model_name = model_path.split('/')[-1] if '/' in model_path else model_path
            print(f"[Neural AI] Loaded model: {model_path} on {device}")
            self.add_message(f"Neural AI loaded: {model_name} ({device})", (100, 220, 100))
        except Exception as e:
            print(f"Failed to load neural AI: {e}")
            import traceback
            traceback.print_exc()
            self.add_message("AI model failed to load — using random", (255, 100, 100))
            self.neural_network = None

    def is_human_turn(self):
        if not self.game:
            return False
        idx = self.game.players.index(self.game.get_current_player())
        return idx in self.human_players

    def handle_human_click(self, mouse_pos):
        """Handle mouse click on the game board for the current human player."""
        if not self.game or mouse_pos[0] >= gui_components.PANEL_X - 20:
            return

        game = self.game
        player = game.get_current_player()

        if game.is_initial_placement_phase():
            if not game.waiting_for_road:
                vertex = find_closest_vertex(game.game_board, mouse_pos, self.offset)
                if vertex:
                    success, msg = game.try_place_initial_settlement(vertex, player)
                    if success:
                        self.add_message(f"{player.name} placed a settlement", player.color)
                    else:
                        self.add_message(msg, (255, 100, 100))
            else:
                edge = find_closest_edge(game.game_board, mouse_pos, self.offset)
                if edge:
                    success, msg = game.try_place_initial_road(edge, player)
                    if success:
                        self.add_message(f"{player.name} placed a road", player.color)
                    else:
                        self.add_message(msg, (255, 100, 100))
        elif game.can_trade_or_build():
            if self.build_mode == "SETTLEMENT":
                vertex = find_closest_vertex(game.game_board, mouse_pos, self.offset)
                if vertex:
                    success, msg = player.try_build_settlement(vertex, False)
                    if success:
                        self.add_message(f"{player.name} built a settlement", player.color)
                    else:
                        self.add_message(msg, (255, 100, 100))
            elif self.build_mode == "CITY":
                vertex = find_closest_vertex(game.game_board, mouse_pos, self.offset)
                if vertex:
                    success, msg = player.try_build_city(vertex)
                    if success:
                        self.add_message(f"{player.name} upgraded to a city", player.color)
                    else:
                        self.add_message(msg, (255, 100, 100))
            elif self.build_mode == "ROAD":
                edge = find_closest_edge(game.game_board, mouse_pos, self.offset)
                if edge:
                    success, msg = player.try_build_road(edge)
                    if success:
                        game.update_longest_road()
                        self.add_message(f"{player.name} built a road", player.color)
                    else:
                        self.add_message(msg, (255, 100, 100))

    def _handle_seven_rolled(self, game, player):
        """Handle rolling a 7: discard (manual for humans), then robber placement."""
        self.add_message("Seven! Robber activated!", (255, 100, 100))

        if game.waiting_for_discards:
            # Auto-discard for AI/bot players
            for p in list(game.players_must_discard):
                if p in game.players_discarded:
                    continue
                idx = game.players.index(p)
                if idx not in self.human_players:
                    total = sum(p.resources.values())
                    num_to_discard = total // 2
                    all_cards = []
                    for res_type, count in p.resources.items():
                        all_cards.extend([res_type] * count)
                    if all_cards and num_to_discard > 0:
                        cards = random.sample(all_cards, min(num_to_discard, len(all_cards)))
                        discard_dict = {}
                        for rt in set(cards):
                            discard_dict[rt] = cards.count(rt)
                        game.discard_cards(p, discard_dict)

            # Check if human player needs to discard
            if game.player_must_discard(player):
                total = sum(player.resources.values())
                self.discard_required = total // 2
                self.discard_selection = {rt: 0 for rt in ResourceType}
                self.discard_mode = True
                self.add_message(f"Discard {self.discard_required} cards!", (255, 100, 100))
                return  # Don't enter robber mode yet — discard first

            # All discards done, clear state
            game.waiting_for_discards = False
            game.players_must_discard = []
            game.players_discarded = set()

        # Enter robber mode
        self.robber_mode = True
        self.add_message("Click a tile to move the robber!", (255, 200, 80))

    def _handle_robber_click(self, mouse_pos):
        """Handle a click during robber placement mode."""
        game = self.game
        player = game.get_current_player()
        tile = find_closest_tile(game.game_board, mouse_pos, self.offset)
        if not tile:
            return
        # Can't place on current robber tile or desert
        if tile == game.robber.position:
            self.add_message("Robber is already there!", (255, 100, 100))
            return
        if tile.resource == "desert":
            self.add_message("Can't place robber on desert!", (255, 100, 100))
            return

        # Move robber
        game.move_robber_to_tile(tile)
        self.add_message(f"Robber moved to {tile.resource} ({tile.number})", player.color)

        # Steal from opponent on that tile
        current_idx = game.players.index(player)
        adjacent_players = game.get_players_on_tile(tile)
        opponents = [p for p in adjacent_players if p != player]
        stole = False
        for opp in opponents:
            if sum(opp.resources.values()) > 0:
                success, msg = game.steal_random_resource(player, opp)
                if success:
                    self.add_message(f"Stole a resource from {opp.name}!", player.color)
                    stole = True
                break
        if opponents and not stole:
            self.add_message("No resources to steal", TEXT_DIM)

        self.robber_mode = False

    def _handle_discard_click(self, mouse_pos):
        """Handle a click during discard mode."""
        for rect, action in self.discard_buttons:
            if rect.collidepoint(mouse_pos):
                if action[0] == "+" and len(action) == 2:
                    res_type = action[1]
                    have = self.game.get_current_player().resources.get(res_type, 0)
                    total_sel = sum(self.discard_selection.values())
                    if self.discard_selection[res_type] < have and total_sel < self.discard_required:
                        self.discard_selection[res_type] += 1
                elif action[0] == "-" and len(action) == 2:
                    res_type = action[1]
                    if self.discard_selection[res_type] > 0:
                        self.discard_selection[res_type] -= 1
                elif action[0] == "confirm":
                    total_sel = sum(self.discard_selection.values())
                    if total_sel == self.discard_required:
                        game = self.game
                        player = game.get_current_player()
                        discard_dict = {rt: cnt for rt, cnt in self.discard_selection.items() if cnt > 0}
                        game.discard_cards(player, discard_dict)
                        self.discard_mode = False
                        self.add_message(f"Discarded {self.discard_required} cards", player.color)
                        # Clear remaining discard state
                        game.waiting_for_discards = False
                        game.players_must_discard = []
                        game.players_discarded = set()
                        # Now enter robber mode
                        self.robber_mode = True
                        self.add_message("Click a tile to move the robber!", (255, 200, 80))
                break

    def _handle_trade_click(self, mouse_pos):
        """Handle a click during trade mode."""
        for rect, action in self.trade_buttons:
            if rect.collidepoint(mouse_pos):
                if action[0] == "give":
                    self.trade_give = action[1]
                    # Reset get if same as give
                    if self.trade_get == self.trade_give:
                        self.trade_get = None
                elif action[0] == "get":
                    res = action[1]
                    if res != self.trade_give:
                        self.trade_get = res
                elif action[0] == "trade":
                    if self.trade_give and self.trade_get and self.trade_give != self.trade_get:
                        game = self.game
                        player = game.get_current_player()
                        success, msg = game.execute_bank_trade(player, self.trade_give, self.trade_get)
                        if success:
                            self.add_message(msg, player.color)
                            self.trade_mode = False
                        else:
                            self.add_message(msg, (255, 100, 100))
                elif action[0] == "cancel":
                    self.trade_mode = False
                break

    def _show_dev_card_popup(self, buy_message):
        """Show a popup for the purchased dev card."""
        # Parse card name from message like "Bought Knight"
        card_name = buy_message.replace("Bought ", "").strip() if buy_message.startswith("Bought") else buy_message
        DEV_CARD_INFO = {
            "Knight": "Move the robber and steal a resource",
            "Victory Point": "+1 hidden victory point",
            "Road Building": "Build 2 roads for free",
            "Year Of Plenty": "Take any 2 resources from the bank",
            "Monopoly": "Take all of one resource type from opponents",
        }
        desc = DEV_CARD_INFO.get(card_name, "")
        self.dev_card_popup = {
            "card_name": card_name,
            "description": desc,
            "show_until": time.time() + 2.5,
        }

    def _get_neural_env(self, player_idx):
        """Get the CatanEnv for the given player index."""
        if hasattr(self, 'neural_envs') and player_idx in self.neural_envs:
            return self.neural_envs[player_idx]
        return self.neural_env

    def handle_ai_turn(self):
        """Execute one AI action."""
        game = self.game
        current_idx = game.players.index(game.get_current_player())
        player = game.get_current_player()

        log = self.add_message

        if self.mode == GameMode.VS_BOT:
            diff_map = {"easy": "weak", "medium": "medium", "hard": "strong"}
            ai_diff = diff_map.get(self.difficulty, "medium")
            success = play_opponent_turn(game, current_idx, ai_diff)
            if not success and game.can_end_turn():
                game.end_turn()

            # Handle stuck discards
            if game.waiting_for_discards:
                _auto_discard_for_game(game, current_idx)
                if game.waiting_for_discards:
                    game.waiting_for_discards = False
                    game.players_must_discard = []
                    game.players_discarded = set()
        elif self.mode == GameMode.WATCH_AI:
            difficulty = getattr(self, 'watch_opponent_difficulty', 'ai_vs_ai')
            env = self._get_neural_env(current_idx)
            if difficulty == "ai_vs_ai":
                # Both players use neural network
                if self.neural_network and env:
                    play_neural_turn(game, current_idx, self.neural_network,
                                     self.neural_device, env, log=log)
                else:
                    play_random_turn(game, current_idx, log=log)
            else:
                # Player 0 = neural AI, Player 1 = rule-based bot
                if current_idx == 0:
                    if self.neural_network and env:
                        play_neural_turn(game, current_idx, self.neural_network,
                                         self.neural_device, env, log=log)
                    else:
                        play_random_turn(game, current_idx, log=log)
                else:
                    diff_map = {"easy": "weak", "medium": "medium", "hard": "strong"}
                    ai_diff = diff_map.get(difficulty, "medium")
                    success = play_opponent_turn(game, current_idx, ai_diff)
                    if not success and game.can_end_turn():
                        game.end_turn()

            # Handle stuck discards
            if game.waiting_for_discards:
                _auto_discard_for_game(game, current_idx)
                if game.waiting_for_discards:
                    game.waiting_for_discards = False
                    game.players_must_discard = []
                    game.players_discarded = set()
        else:
            # VS_AI mode: use neural network if available, else random
            env = self._get_neural_env(current_idx)
            if self.neural_network and env:
                play_neural_turn(game, current_idx, self.neural_network,
                                 self.neural_device, env, log=log)
            else:
                play_random_turn(game, current_idx, log=log)

            # Handle stuck discards for AI
            if game.waiting_for_discards:
                _auto_discard_for_game(game, current_idx)
                if game.waiting_for_discards:
                    game.waiting_for_discards = False
                    game.players_must_discard = []
                    game.players_discarded = set()

    def draw_playing(self):
        """Draw the game screen."""
        self.screen.fill(BG_COLOR)

        # Draw board
        draw_game_board(self.screen, self.game.game_board, self.offset)

        # Draw buildable highlights for human players
        if self.is_human_turn():
            draw_buildable_highlights(self.screen, self.game.game_board,
                                    self.game, self.offset, self.build_mode)

        # --- Right Panel ---
        x = gui_components.PANEL_X
        y = 15
        pw = gui_components.PANEL_W

        # Player info boxes
        for i, player in enumerate(self.game.players):
            is_current = (player == self.game.get_current_player())
            h = draw_player_info_box(self.screen, player, x, y, pw,
                                    is_current, self.font, self.small_font)
            y += h + 8

        # VP tracker for current player
        current = self.game.get_current_player()
        vp = current.calculate_victory_points()
        draw_vp_tracker(self.screen, vp, GameConstants.VICTORY_POINTS_TO_WIN,
                       x, y, width=pw)
        y += 24

        # Resource cards for current player (always visible in watch/PvP, human turn otherwise)
        if self.is_human_turn() or self.mode in (GameMode.PVP, GameMode.WATCH_AI):
            draw_resource_panel(self.screen, current, x, y)
            y += 82

        # Dev card inventory
        if sum(current.development_cards.values()) > 0:
            h = draw_dev_cards_panel(self.screen, current, x, y, pw, self.small_font)
            y += h + 4

        # Build mode indicator
        mode_font = get_font(14, bold=True)
        mode_text = f"Build Mode: {self.build_mode.title()}"
        mode_surf = mode_font.render(mode_text, True, GOLD)
        self.screen.blit(mode_surf, (x, y))
        y += 18
        # Show hint if no valid spots for current build mode
        game = self.game
        if self.is_human_turn() and game.can_trade_or_build():
            if self.build_mode == "SETTLEMENT" and not game.get_buildable_vertices_for_settlements():
                hint = get_font(12).render("No valid spots — build roads first!", True, (255, 100, 100))
                self.screen.blit(hint, (x, y))
                y += 14
            elif self.build_mode == "CITY" and not game.get_buildable_vertices_for_cities():
                hint = get_font(12).render("No settlements to upgrade!", True, (255, 100, 100))
                self.screen.blit(hint, (x, y))
                y += 14
        y += 4

        # Action buttons (2 columns)
        btn_w = (pw - 10) // 2
        btn_h = 38
        btn_font = get_font(16, bold=True)
        mouse_pos = pygame.mouse.get_pos()

        can_build = game.can_trade_or_build()
        player = game.get_current_player()

        not_blocked = self.is_human_turn() and not self.robber_mode and not self.discard_mode and not self.trade_mode
        human_can_build = can_build and not_blocked
        buttons_data = [
            ("Roll Dice", "D", game.can_roll_dice() and not_blocked, None, "roll"),
            ("End Turn", "T", game.can_end_turn() and not_blocked, None, "end_turn"),
            ("Settlement", "1", human_can_build and player.can_afford(BUILD_COSTS["settlement"]), BUILD_COSTS["settlement"], "sett"),
            ("City", "2", human_can_build and player.can_afford(BUILD_COSTS["city"]), BUILD_COSTS["city"], "city"),
            ("Road", "3", human_can_build and player.can_afford(BUILD_COSTS["road"]), BUILD_COSTS["road"], "road"),
            ("Dev Card", "X", human_can_build and player.can_afford(BUILD_COSTS["dev_card"]), BUILD_COSTS["dev_card"], "dev"),
            ("Trade", "B", can_build and not_blocked, None, "trade"),
            ("", "", False, None, None),  # empty slot to keep 2-column grid even
        ]

        self.action_buttons = []
        for i, (label, shortcut, enabled, cost, action) in enumerate(buttons_data):
            if not label:
                continue  # skip empty placeholder
            col = i % 2
            row = i // 2
            bx = x + col * (btn_w + 10)
            by = y + row * (btn_h + 6)
            rect = pygame.Rect(bx, by, btn_w, btn_h)
            hovered = rect.collidepoint(mouse_pos) and enabled
            drawn_rect = draw_button(self.screen, (bx, by, btn_w, btn_h), label,
                                    btn_font, enabled=enabled, hovered=hovered,
                                    shortcut=shortcut, cost=cost)
            self.action_buttons.append((drawn_rect, action, enabled))

        y += 4 * (btn_h + 6) + 10

        # Game state panel
        h = draw_game_state_panel(self.screen, game, x, y, pw,
                                  self.font, self.small_font)
        y += h + 10

        # AI thinking indicator
        if not self.is_human_turn():
            thinking_font = get_font(20, bold=True)
            ai_name = game.get_current_player().name
            thinking_surf = thinking_font.render(f"{ai_name} is thinking...", True, (255, 200, 80))
            self.screen.blit(thinking_surf, (x, y))
            y += 30

        # Game log
        log_h = max(80, gui_components.SCREEN_H - y - 10)
        draw_game_log(self.screen, self.messages, x, y, pw, log_h, self.small_font)

        # Trade overlay (drawn on top of everything)
        if self.trade_mode:
            current = self.game.get_current_player()
            self.trade_buttons = draw_trade_overlay(
                self.screen, current, self.game.game_board,
                self.trade_give, self.trade_get)

        # Discard overlay (drawn on top of everything)
        if self.discard_mode:
            current = self.game.get_current_player()
            self.discard_buttons = draw_discard_overlay(
                self.screen, current, self.discard_selection, self.discard_required)

        # Robber overlay (drawn on top of everything)
        if self.robber_mode:
            draw_robber_overlay(self.screen, self.game.game_board, self.offset,
                               self.game.robber.position)

        # Dev card popup (drawn on top of everything)
        if self.dev_card_popup:
            if time.time() < self.dev_card_popup["show_until"]:
                draw_dev_card_popup(self.screen,
                                   self.dev_card_popup["card_name"],
                                   self.dev_card_popup["description"])
            else:
                self.dev_card_popup = None

    def draw_victory(self, winner):
        """Draw victory overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        cx = self.screen_w // 2

        # Victory text
        title_font = get_font(64, bold=True)
        title = title_font.render("VICTORY!", True, GOLD)
        self.screen.blit(title, title.get_rect(center=(cx, 250)))

        # Winner name
        winner_font = get_font(36, bold=True)
        winner_surf = winner_font.render(f"{winner.name} Wins!", True, winner.color)
        self.screen.blit(winner_surf, winner_surf.get_rect(center=(cx, 330)))

        # Stats
        stats_font = get_font(22)
        vp = winner.calculate_victory_points()
        stats = [
            f"Victory Points: {vp}",
            f"Settlements: {len(winner.settlements)}  Cities: {len(winner.cities)}  Roads: {len(winner.roads)}",
        ]
        if winner.has_longest_road:
            stats.append("Longest Road (+2 VP)")
        if winner.has_largest_army:
            stats.append("Largest Army (+2 VP)")

        for i, s in enumerate(stats):
            surf = stats_font.render(s, True, TEXT_COLOR)
            self.screen.blit(surf, surf.get_rect(center=(cx, 400 + i * 35)))

        # Buttons
        btn_font = get_font(22, bold=True)
        btn_w, btn_h = 200, 50
        mouse_pos = pygame.mouse.get_pos()

        play_again_rect = pygame.Rect(cx - btn_w - 20, 580, btn_w, btn_h)
        menu_rect = pygame.Rect(cx + 20, 580, btn_w, btn_h)

        pa_hover = play_again_rect.collidepoint(mouse_pos)
        m_hover = menu_rect.collidepoint(mouse_pos)

        draw_button(self.screen, play_again_rect, "Play Again", btn_font,
                   enabled=True, hovered=pa_hover)
        draw_button(self.screen, menu_rect, "Main Menu", btn_font,
                   enabled=True, hovered=m_hover)

        self.victory_buttons = [
            (play_again_rect, "play_again"),
            (menu_rect, "main_menu"),
        ]

    def run(self):
        """Main application loop."""
        running = True
        ai_last_move = 0
        ai_delay_ms = 400  # ms between AI actions
        winner = None

        while running:
            mouse_pos = pygame.mouse.get_pos()

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state == GameState.PLAYING:
                            self.state = GameState.TITLE
                            self.game = None
                        else:
                            running = False

                    elif self.state == GameState.PLAYING and self.is_human_turn():
                        game = self.game
                        player = game.get_current_player()

                        # Block all keys during discard, robber, or trade mode
                        if self.discard_mode or self.robber_mode or self.trade_mode:
                            pass
                        elif event.key == pygame.K_d:
                            if game.can_roll_dice():
                                result = game.roll_dice()
                                if result:
                                    self.add_message(
                                        f"{player.name} rolled {result[2]} ({result[0]}+{result[1]})",
                                        GOLD)
                                    if result[2] == 7:
                                        self._handle_seven_rolled(game, player)

                        elif event.key == pygame.K_t:
                            if game.can_end_turn():
                                success, msg = game.end_turn()
                                if success:
                                    self.add_message("Turn ended", TEXT_DIM)
                                    winner = game.check_victory_conditions()
                                    if winner:
                                        self.state = GameState.VICTORY
                                        self.add_message(
                                            f"{winner.name} wins!",
                                            winner.color)

                        elif event.key == pygame.K_1:
                            if game.can_trade_or_build() and player.can_afford(BUILD_COSTS["settlement"]):
                                self.build_mode = "SETTLEMENT"
                        elif event.key == pygame.K_2:
                            if game.can_trade_or_build() and player.can_afford(BUILD_COSTS["city"]):
                                self.build_mode = "CITY"
                        elif event.key == pygame.K_3:
                            if game.can_trade_or_build() and player.can_afford(BUILD_COSTS["road"]):
                                self.build_mode = "ROAD"
                        elif event.key == pygame.K_x:
                            if game.can_trade_or_build() and player.can_afford(BUILD_COSTS["dev_card"]):
                                success, msg = player.try_buy_development_card(game.dev_deck)
                                if success:
                                    self._show_dev_card_popup(msg)
                                    self.add_message(f"{player.name} bought a dev card", player.color)
                        elif event.key == pygame.K_b:
                            if game.can_trade_or_build():
                                self.trade_mode = True
                                self.trade_give = None
                                self.trade_get = None

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == GameState.TITLE:
                        # Check title buttons
                        button_rects = draw_title_screen(self.screen, self.buttons_hovered)
                        for rect, mode in button_rects:
                            if rect.collidepoint(event.pos):
                                if mode == "pvp":
                                    self.setup_game(GameMode.PVP)
                                elif mode == "vs_ai":
                                    # Phase C: neural AI. For now, use random.
                                    self.setup_game(GameMode.VS_AI)
                                elif mode == "vs_bot":
                                    self.state = GameState.DIFFICULTY_SELECT
                                elif mode == "watch_ai":
                                    self.state = GameState.WATCH_SELECT

                    elif self.state == GameState.WATCH_SELECT:
                        button_rects = draw_watch_select_screen(self.screen, self.buttons_hovered)
                        for rect, opt in button_rects:
                            if rect.collidepoint(event.pos):
                                if opt == "back":
                                    self.state = GameState.TITLE
                                else:
                                    self.setup_game(GameMode.WATCH_AI, difficulty=opt)

                    elif self.state == GameState.DIFFICULTY_SELECT:
                        button_rects = draw_difficulty_screen(self.screen, self.buttons_hovered)
                        for rect, diff in button_rects:
                            if rect.collidepoint(event.pos):
                                if diff == "back":
                                    self.state = GameState.TITLE
                                else:
                                    self.setup_game(GameMode.VS_BOT, difficulty=diff)

                    elif self.state == GameState.PLAYING and self.is_human_turn():
                        game = self.game
                        player = game.get_current_player()

                        # Discard mode: only accept discard overlay clicks
                        if self.discard_mode:
                            self._handle_discard_click(event.pos)
                        # Trade mode: only accept trade overlay clicks
                        elif self.trade_mode:
                            self._handle_trade_click(event.pos)
                        # Robber mode: only accept board tile clicks
                        elif self.robber_mode:
                            self._handle_robber_click(event.pos)
                        else:
                            # Check action button clicks
                            btn_handled = False
                            if hasattr(self, 'action_buttons'):
                                for rect, action, enabled in self.action_buttons:
                                    if enabled and rect.collidepoint(event.pos):
                                        btn_handled = True
                                        if action == "roll":
                                            if game.can_roll_dice():
                                                result = game.roll_dice()
                                                if result:
                                                    self.add_message(
                                                        f"{player.name} rolled {result[2]} ({result[0]}+{result[1]})",
                                                        GOLD)
                                                    if result[2] == 7:
                                                        self._handle_seven_rolled(game, player)
                                        elif action == "end_turn":
                                            if game.can_end_turn():
                                                success, msg = game.end_turn()
                                                if success:
                                                    self.add_message("Turn ended", TEXT_DIM)
                                                    winner = game.check_victory_conditions()
                                                    if winner:
                                                        self.state = GameState.VICTORY
                                        elif action == "sett":
                                            self.build_mode = "SETTLEMENT"
                                        elif action == "city":
                                            self.build_mode = "CITY"
                                        elif action == "road":
                                            self.build_mode = "ROAD"
                                        elif action == "dev":
                                            if game.can_trade_or_build():
                                                success, msg = player.try_buy_development_card(game.dev_deck)
                                                if success:
                                                    self._show_dev_card_popup(msg)
                                                    self.add_message(f"{player.name} bought a dev card", player.color)
                                        elif action == "trade":
                                            if game.can_trade_or_build():
                                                self.trade_mode = True
                                                self.trade_give = None
                                                self.trade_get = None
                                        break  # Only handle one button click

                            # Board clicks (if no button was clicked)
                            if not btn_handled:
                                self.handle_human_click(event.pos)

                        # Dismiss dev card popup on click
                        if self.dev_card_popup:
                            self.dev_card_popup = None

                    elif self.state == GameState.VICTORY:
                        if hasattr(self, 'victory_buttons'):
                            for rect, action in self.victory_buttons:
                                if rect.collidepoint(event.pos):
                                    if action == "play_again":
                                        self.setup_game(self.mode, self.difficulty)
                                        winner = None
                                    elif action == "main_menu":
                                        self.state = GameState.TITLE
                                        self.game = None
                                        winner = None

            # --- AI Turn Logic ---
            if (self.state == GameState.PLAYING and
                self.game and not self.is_human_turn()):
                now = pygame.time.get_ticks()
                if now - ai_last_move >= ai_delay_ms:
                    ai_last_move = now
                    self.handle_ai_turn()
                    # Check victory
                    w = self.game.check_victory_conditions()
                    if w:
                        winner = w
                        self.state = GameState.VICTORY
                        self.add_message(f"{w.name} wins!", w.color)

            # --- Drawing ---
            if self.state == GameState.TITLE:
                # Update hover state
                self.buttons_hovered = {}
                temp_rects = draw_title_screen(self.screen, {})
                for rect, mode in temp_rects:
                    self.buttons_hovered[mode] = rect.collidepoint(mouse_pos)
                draw_title_screen(self.screen, self.buttons_hovered)

            elif self.state == GameState.DIFFICULTY_SELECT:
                self.buttons_hovered = {}
                temp_rects = draw_difficulty_screen(self.screen, {})
                for rect, diff in temp_rects:
                    self.buttons_hovered[diff] = rect.collidepoint(mouse_pos)
                draw_difficulty_screen(self.screen, self.buttons_hovered)

            elif self.state == GameState.WATCH_SELECT:
                self.buttons_hovered = {}
                temp_rects = draw_watch_select_screen(self.screen, {})
                for rect, opt in temp_rects:
                    self.buttons_hovered[opt] = rect.collidepoint(mouse_pos)
                draw_watch_select_screen(self.screen, self.buttons_hovered)

            elif self.state == GameState.PLAYING:
                self.draw_playing()

            elif self.state == GameState.VICTORY:
                self.draw_playing()
                if winner:
                    self.draw_victory(winner)

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = CatantronApp()
    app.run()
