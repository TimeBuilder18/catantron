"""
Visual Catan Game - Play against AI or watch AI matches

MODES:
------
1. Human vs AI - You play against 3 AI agents
2. AI vs AI - Watch 4 AI agents play (with speed control)
3. Human only - 4 human players

CONTROLS:
---------
- Click to select vertices/edges for building
- D: Roll dice
- T: End turn
- B: Build settlement
- C: Build city
- R: Build road
- V: Buy dev card
- TAB: Toggle debug overlay
- SPACE: Pause/Resume (AI mode)
- +/-: Speed up/slow down AI moves
- ESC: Quit
"""

import pygame
import sys
import random
from ai_interface import AIGameEnvironment
from game_system import ResourceType, PortType
from tile import Tile

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Resource colors
RESOURCE_COLORS = {
    "forest": (34, 139, 34),
    "hill": (139, 69, 19),
    "field": (255, 215, 0),
    "mountain": (128, 128, 128),
    "pasture": (144, 238, 144),
    "desert": (210, 180, 140)
}


class DummyAI:
    """Simple random AI for testing"""
    def __init__(self, player_index):
        self.player_index = player_index

    def choose_action(self, obs):
        """Choose random legal action"""
        legal_actions = obs['legal_actions']
        if not legal_actions or legal_actions == ['wait']:
            return 'wait'
        return random.choice(legal_actions)

    def choose_params(self, action, env):
        """Choose random valid parameters for action"""
        if action in ['roll_dice', 'end_turn', 'buy_dev_card', 'wait']:
            return None

        if action == 'place_settlement' or action == 'build_settlement':
            vertices = env.game.game_board.vertices
            valid = [v for v in vertices if v.structure is None]
            if valid:
                return {'vertex': random.choice(valid)}

        elif action == 'place_road' or action == 'build_road':
            edges = env.game.game_board.edges
            valid = [e for e in edges if e.structure is None]
            if valid:
                return {'edge': random.choice(empty)}

        elif action == 'build_city':
            player = env.game.players[self.player_index]
            vertices = env.game.game_board.vertices
            valid = [v for v in vertices if v.structure and
                    v.structure.player == player and
                    hasattr(v.structure, '__class__') and
                    v.structure.__class__.__name__ == 'Settlement']
            if valid:
                return {'vertex': random.choice(valid)}

        return None


class VisualCatanGame:
    """Visual game interface with human + AI support"""

    def __init__(self, mode='human_vs_ai'):
        """
        mode: 'human_vs_ai', 'ai_vs_ai', or 'human_only'
        """
        pygame.init()

        # Screen setup
        self.screen_width = 1400
        self.screen_height = 900
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Settlers of Catan - AI Training")

        # Mode setup BEFORE creating environment
        self.mode = mode
        self.human_player_index = 0  # Player 1 is human in human_vs_ai mode

        # Game setup - create environment with appropriate player names
        self.env = self._create_environment()
        self.observations = self.env.reset()

        # AI agents
        self.ai_agents = [DummyAI(i) for i in range(4)]

        # View settings - center the board in the remaining space
        # Left panel is 350px wide, so center in remaining space
        remaining_width = self.screen_width - 350
        self.offset_x = 350 + remaining_width // 2  # Center horizontally in remaining space
        self.offset_y = self.screen_height // 2      # Center vertically
        self.zoom = 1.0

        # UI state
        self.selected_vertex = None
        self.selected_edge = None
        self.hover_vertex = None
        self.hover_edge = None
        self.build_mode = None
        self.show_debug = True
        self.message = ""
        self.message_timer = 0

        # AI control
        self.ai_paused = False
        self.ai_speed = 1.0
        self.ai_timer = 0

        # Fonts
        self.font_small = pygame.font.SysFont(None, 20)
        self.font_medium = pygame.font.SysFont(None, 28)
        self.font_large = pygame.font.SysFont(None, 36)
        self.font_title = pygame.font.SysFont(None, 48)

        self.clock = pygame.time.Clock()
        self.running = True

    def _create_environment(self):
        """Create environment with appropriate player names based on mode"""
        # We need to create a custom environment with proper names
        from game_system import GameSystem, GameBoard, Player
        from tile import Tile
        import random

        # Import the board creation functions from ai_interface
        from ai_interface import create_hexagonal_board, assign_resources_numbers

        tile_size = 50
        tiles = create_hexagonal_board(tile_size, radius=2)

        for t in tiles:
            t.find_neighbors(tiles)

        from game_system import Robber
        robber = Robber()
        assign_resources_numbers(tiles, robber)

        game_board = GameBoard(tiles)

        # Create players with appropriate names
        if self.mode == 'human_vs_ai':
            players = [
                Player("You (Human)", (255, 50, 50)),      # Red
                Player("AI Bot 1", (50, 50, 255)),         # Blue
                Player("AI Bot 2", (255, 255, 50)),        # Yellow
                Player("AI Bot 3", (255, 255, 255))        # White
            ]
        elif self.mode == 'human_only':
            players = [
                Player("Player 1", (255, 50, 50)),         # Red
                Player("Player 2", (50, 50, 255)),         # Blue
                Player("Player 3", (255, 255, 50)),        # Yellow
                Player("Player 4", (255, 255, 255))        # White
            ]
        else:  # ai_vs_ai
            players = [
                Player("AI Bot 1", (255, 50, 50)),         # Red
                Player("AI Bot 2", (50, 50, 255)),         # Blue
                Player("AI Bot 3", (255, 255, 50)),        # Yellow
                Player("AI Bot 4", (255, 255, 255))        # White
            ]

        # Create a minimal environment object
        class CustomEnv:
            def __init__(self, game_board, players):
                self.game = GameSystem(game_board, players)
                self.game.robber = robber

            def reset(self):
                # Return observations for all players
                return [self._get_observation(i) for i in range(4)]

            def _get_observation(self, player_index):
                """Get observation for a player"""
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
                            'dev_card_count': sum(p.development_cards.values()),
                            'settlements': len(p.settlements),
                            'cities': len(p.cities),
                            'roads': len(p.roads),
                            'victory_points': p.calculate_victory_points()
                        }
                        for i, p in enumerate(self.game.players) if i != player_index
                    ],
                    'tiles': [(t.q, t.r, t.resource, t.number) for t in self.game.game_board.tiles],
                    'ports': [(p.port_type.name, p.vertex1.x, p.vertex1.y) for p in self.game.game_board.ports],
                    'legal_actions': self._get_legal_actions(player_index)
                }
                return obs

            def _get_legal_actions(self, player_index):
                """Get legal actions for a player"""
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

            def step(self, player_index, action, action_params=None):
                """Execute an action"""
                player = self.game.players[player_index]

                if player != self.game.get_current_player():
                    return self._get_observation(player_index), False, {'error': 'Not your turn'}

                info = {}

                if action == 'roll_dice':
                    success, message = self.game.roll_dice_action()
                    info['success'] = success
                    info['message'] = message
                elif action == 'end_turn':
                    success, message = self.game.end_turn()
                    info['success'] = success
                    info['message'] = message
                elif action == 'place_settlement':
                    vertex = action_params.get('vertex') if action_params else None
                    if vertex:
                        success, message = self.game.try_place_initial_settlement(vertex, player)
                        info['success'] = success
                        info['message'] = message
                    else:
                        info['success'] = False
                        info['message'] = 'No vertex provided'
                elif action == 'place_road':
                    edge = action_params.get('edge') if action_params else None
                    if edge:
                        success, message = self.game.try_place_initial_road(edge, player)
                        info['success'] = success
                        info['message'] = message
                    else:
                        info['success'] = False
                        info['message'] = 'No edge provided'
                elif action == 'build_settlement':
                    vertex = action_params.get('vertex') if action_params else None
                    if vertex:
                        success, message = self.game.try_build_settlement(vertex, player)
                        info['success'] = success
                        info['message'] = message
                    else:
                        info['success'] = False
                        info['message'] = 'No vertex provided'
                elif action == 'build_road':
                    edge = action_params.get('edge') if action_params else None
                    if edge:
                        success, message = self.game.try_build_road(edge, player)
                        info['success'] = success
                        info['message'] = message
                    else:
                        info['success'] = False
                        info['message'] = 'No edge provided'
                elif action == 'build_city':
                    vertex = action_params.get('vertex') if action_params else None
                    if vertex:
                        success, message = self.game.try_build_city(vertex, player)
                        info['success'] = success
                        info['message'] = message
                    else:
                        info['success'] = False
                        info['message'] = 'No vertex provided'
                elif action == 'buy_dev_card':
                    success, message = self.game.try_buy_development_card(player)
                    info['success'] = success
                    info['message'] = message

                winner = self.game.check_victory_conditions()
                done = (winner is not None)

                if done:
                    if winner:
                        info['winner'] = self.game.players.index(winner)
                        info['result'] = 'game_over'

                obs = self._get_observation(player_index)
                return obs, done, info

        return CustomEnv(game_board, players)

    def run(self):
        """Main game loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0

            self.handle_events()
            self.update(dt)
            self.draw()

            pygame.display.flip()

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.handle_click(event.pos)

            elif event.type == pygame.MOUSEMOTION:
                self.update_hover(event.pos)

    def handle_keypress(self, key):
        """Handle keyboard input"""
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_TAB:
            self.show_debug = not self.show_debug
        elif key == pygame.K_SPACE and self.mode == 'ai_vs_ai':
            self.ai_paused = not self.ai_paused
        elif key == pygame.K_EQUALS or key == pygame.K_PLUS:
            self.ai_speed = min(10.0, self.ai_speed * 1.5)
            self.show_message(f"AI Speed: {self.ai_speed:.1f}x")
        elif key == pygame.K_MINUS:
            self.ai_speed = max(0.1, self.ai_speed / 1.5)
            self.show_message(f"AI Speed: {self.ai_speed:.1f}x")

        # Arrow keys to pan view
        elif key == pygame.K_LEFT:
            self.offset_x -= 50
        elif key == pygame.K_RIGHT:
            self.offset_x += 50
        elif key == pygame.K_UP:
            self.offset_y -= 50
        elif key == pygame.K_DOWN:
            self.offset_y += 50

        # Human player controls
        if self.mode != 'ai_vs_ai' and self.is_human_turn():
            if key == pygame.K_d:
                self.try_roll_dice()
            elif key == pygame.K_t:
                self.try_end_turn()
            elif key == pygame.K_b:
                self.build_mode = 'settlement'
                self.show_message("Click a vertex to build settlement")
            elif key == pygame.K_c:
                self.build_mode = 'city'
                self.show_message("Click a vertex to upgrade to city")
            elif key == pygame.K_r:
                self.build_mode = 'road'
                self.show_message("Click an edge to build road")
            elif key == pygame.K_v:
                self.try_buy_dev_card()

    def handle_click(self, pos):
        """Handle mouse clicks"""
        if self.mode == 'ai_vs_ai':
            return  # No clicking in AI mode

        if self.mode == 'human_vs_ai' and not self.is_human_turn():
            return  # Not your turn in human_vs_ai mode

        # In human_only mode, any player can click on their turn

        clicked_vertex = self.get_vertex_at_pos(pos)
        if clicked_vertex:
            self.handle_vertex_click(clicked_vertex)
            return

        clicked_edge = self.get_edge_at_pos(pos)
        if clicked_edge:
            self.handle_edge_click(clicked_edge)
            return

    def handle_vertex_click(self, vertex):
        """Handle clicking on a vertex"""
        if self.mode == 'human_only':
            player_idx = self.env.game.current_player_index  # Current player in human_only
        else:
            player_idx = self.human_player_index  # Player 0 in human_vs_ai

        obs = self.observations[player_idx]

        # Auto-detect what to do based on legal actions
        if 'place_settlement' in obs['legal_actions']:
            # Initial placement - just click to place
            self.try_place_settlement(vertex, player_idx)
        elif self.build_mode == 'settlement' and 'build_settlement' in obs['legal_actions']:
            self.try_build_settlement(vertex, player_idx)
        elif self.build_mode == 'city' and 'build_city' in obs['legal_actions']:
            self.try_build_city(vertex, player_idx)
        else:
            self.selected_vertex = vertex
            # Give feedback about what mode you need to be in
            if 'build_settlement' in obs['legal_actions']:
                self.show_message("Press B to enter settlement build mode", duration=2)

    def handle_edge_click(self, edge):
        """Handle clicking on an edge"""
        if self.mode == 'human_only':
            player_idx = self.env.game.current_player_index  # Current player in human_only
        else:
            player_idx = self.human_player_index  # Player 0 in human_vs_ai

        obs = self.observations[player_idx]

        # Auto-detect what to do based on legal actions
        if 'place_road' in obs['legal_actions']:
            # Initial placement - just click to place
            self.try_place_road(edge, player_idx)
        elif self.build_mode == 'road' and 'build_road' in obs['legal_actions']:
            self.try_build_road(edge, player_idx)
        else:
            self.selected_edge = edge
            # Give feedback about what mode you need to be in
            if 'build_road' in obs['legal_actions']:
                self.show_message("Press R to enter road build mode", duration=2)

    def update(self, dt):
        """Update game state"""
        if self.message_timer > 0:
            self.message_timer -= dt
            if self.message_timer <= 0:
                self.message = ""

        # Handle AI turns
        if self.mode == 'ai_vs_ai' or (self.mode == 'human_vs_ai' and not self.is_human_turn()):
            if not self.ai_paused:
                self.ai_timer += dt * self.ai_speed
                if self.ai_timer >= 1.0:
                    self.ai_timer = 0
                    self.execute_ai_move()

    def execute_ai_move(self):
        """Execute one AI move"""
        current_player = self.env.game.current_player_index
        obs = self.observations[current_player]

        action = self.ai_agents[current_player].choose_action(obs)
        params = self.ai_agents[current_player].choose_params(action, self.env)

        new_obs, done, info = self.env.step(current_player, action, params)
        self.observations[current_player] = new_obs

        player_name = self.env.game.players[current_player].name
        if info.get('success'):
            self.show_message(f"{player_name}: {info.get('message', action)}")

        if done:
            winner = self.env.game.players[info['winner']]
            self.show_message(f"GAME OVER! {winner.name} wins!", duration=10)

    def try_roll_dice(self):
        """Try to roll dice"""
        player_idx = self.env.game.current_player_index if self.mode == 'human_only' else self.human_player_index
        obs, done, info = self.env.step(player_idx, 'roll_dice', None)
        self.observations[player_idx] = obs
        if info.get('success'):
            self.show_message(info['message'])

    def try_end_turn(self):
        """Try to end turn"""
        player_idx = self.env.game.current_player_index if self.mode == 'human_only' else self.human_player_index
        obs, done, info = self.env.step(player_idx, 'end_turn', None)
        self.observations[player_idx] = obs
        if info.get('success'):
            self.show_message(info['message'])
            self.build_mode = None

    def try_place_settlement(self, vertex, player_idx):
        """Try to place settlement"""
        obs, done, info = self.env.step(player_idx, 'place_settlement', {'vertex': vertex})
        self.observations[player_idx] = obs
        self.show_message(info['message'])
        if info.get('success'):
            self.build_mode = None

    def try_build_settlement(self, vertex, player_idx):
        """Try to build settlement"""
        obs, done, info = self.env.step(player_idx, 'build_settlement', {'vertex': vertex})
        self.observations[player_idx] = obs
        self.show_message(info['message'])
        if info.get('success'):
            self.build_mode = None

    def try_build_city(self, vertex, player_idx):
        """Try to build city"""
        obs, done, info = self.env.step(player_idx, 'build_city', {'vertex': vertex})
        self.observations[player_idx] = obs
        self.show_message(info['message'])
        if info.get('success'):
            self.build_mode = None

    def try_place_road(self, edge, player_idx):
        """Try to place road"""
        obs, done, info = self.env.step(player_idx, 'place_road', {'edge': edge})
        self.observations[player_idx] = obs
        self.show_message(info['message'])
        if info.get('success'):
            self.build_mode = None

    def try_build_road(self, edge, player_idx):
        """Try to build road"""
        obs, done, info = self.env.step(player_idx, 'build_road', {'edge': edge})
        self.observations[player_idx] = obs
        self.show_message(info['message'])
        if info.get('success'):
            self.build_mode = None

    def try_buy_dev_card(self):
        """Try to buy development card"""
        player_idx = self.env.game.current_player_index if self.mode == 'human_only' else self.human_player_index
        obs, done, info = self.env.step(player_idx, 'buy_dev_card', None)
        self.observations[player_idx] = obs
        self.show_message(info['message'])

    def is_human_turn(self):
        """Check if it's a human player's turn"""
        if self.mode == 'human_only':
            return True  # All players are human
        elif self.mode == 'human_vs_ai':
            return self.env.game.current_player_index == self.human_player_index
        else:  # ai_vs_ai
            return False

    def show_message(self, message, duration=3.0):
        """Show a temporary message"""
        self.message = message
        self.message_timer = duration
        print(f"[Game] {message}")

    def update_hover(self, pos):
        """Update hover states"""
        self.hover_vertex = self.get_vertex_at_pos(pos)
        self.hover_edge = self.get_edge_at_pos(pos)

    def get_vertex_at_pos(self, pos):
        """Find vertex at mouse position"""
        mouse_x, mouse_y = pos
        for vertex in self.env.game.game_board.vertices:
            vx = vertex.x + self.offset_x
            vy = vertex.y + self.offset_y
            dist = ((mouse_x - vx) ** 2 + (mouse_y - vy) ** 2) ** 0.5
            if dist < 15:
                return vertex
        return None

    def get_edge_at_pos(self, pos):
        """Find edge at mouse position"""
        mouse_x, mouse_y = pos
        for edge in self.env.game.game_board.edges:
            ex = edge.x + self.offset_x
            ey = edge.y + self.offset_y
            dist = ((mouse_x - ex) ** 2 + (mouse_y - ey) ** 2) ** 0.5
            if dist < 15:
                return edge
        return None

    def draw(self):
        """Draw everything"""
        self.screen.fill(LIGHT_GRAY)

        self.draw_board()
        self.draw_roads()
        self.draw_settlements_and_cities()
        self.draw_ports()
        self.draw_hover_highlights()
        self.draw_ui()

        if self.show_debug:
            self.draw_debug_overlay()

    def draw_board(self):
        """Draw the hexagonal board"""
        for tile in self.env.game.game_board.tiles:
            color = RESOURCE_COLORS.get(tile.resource, WHITE)
            tile.draw(self.screen, color, offset=(self.offset_x, self.offset_y))

    def draw_roads(self):
        """Draw all roads"""
        for edge in self.env.game.game_board.edges:
            if edge.structure:
                player = edge.structure.player
                color = player.color
                x1 = edge.vertex1.x + self.offset_x
                y1 = edge.vertex1.y + self.offset_y
                x2 = edge.vertex2.x + self.offset_x
                y2 = edge.vertex2.y + self.offset_y
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 6)

    def draw_settlements_and_cities(self):
        """Draw all settlements and cities"""
        for vertex in self.env.game.game_board.vertices:
            if vertex.structure:
                player = vertex.structure.player
                color = player.color
                x = vertex.x + self.offset_x
                y = vertex.y + self.offset_y

                if vertex.structure.__class__.__name__ == 'Settlement':
                    size = 12
                    rect = pygame.Rect(x - size/2, y - size/2, size, size)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, BLACK, rect, 2)
                else:  # City
                    size = 18
                    rect = pygame.Rect(x - size/2, y - size/2, size, size)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, BLACK, rect, 2)

    def draw_ports(self):
        """Draw port indicators"""
        for port in self.env.game.game_board.ports:
            x1 = port.vertex1.x + self.offset_x
            y1 = port.vertex1.y + self.offset_y
            x2 = port.vertex2.x + self.offset_x
            y2 = port.vertex2.y + self.offset_y

            pygame.draw.line(self.screen, BLUE, (x1, y1), (x2, y2), 4)

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            text = self.font_small.render(port.port_type.value, True, BLACK)
            text_rect = text.get_rect(center=(mid_x, mid_y))
            pygame.draw.rect(self.screen, WHITE, text_rect.inflate(4, 4))
            self.screen.blit(text, text_rect)

    def draw_hover_highlights(self):
        """Draw highlights for hovered elements"""
        if self.hover_vertex:
            x = self.hover_vertex.x + self.offset_x
            y = self.hover_vertex.y + self.offset_y
            pygame.draw.circle(self.screen, YELLOW, (int(x), int(y)), 10, 3)

        if self.hover_edge:
            x = self.hover_edge.x + self.offset_x
            y = self.hover_edge.y + self.offset_y
            pygame.draw.circle(self.screen, YELLOW, (int(x), int(y)), 10, 3)

    def draw_ui(self):
        """Draw UI elements"""
        panel_rect = pygame.Rect(0, 0, 350, self.screen_height)
        pygame.draw.rect(self.screen, DARK_GRAY, panel_rect)

        y_offset = 20

        title = self.font_title.render("CATAN AI", True, WHITE)
        self.screen.blit(title, (20, y_offset))
        y_offset += 60

        # Show what action is expected
        current = self.env.game.current_player_index
        if self.mode != 'ai_vs_ai':  # Show for human_vs_ai and human_only
            current_player_name = self.env.game.players[current].name

            # Only show action prompt if it's actually your turn
            if (self.mode == 'human_only') or (self.mode == 'human_vs_ai' and current == self.human_player_index):
                obs = self.observations[current]
                legal_actions = obs['legal_actions']

                if 'place_settlement' in legal_actions:
                    action_text = f"{current_player_name}: Click vertex to place settlement"
                    action_color = GREEN
                elif 'place_road' in legal_actions:
                    action_text = f"{current_player_name}: Click edge to place road"
                    action_color = GREEN
                elif 'roll_dice' in legal_actions:
                    action_text = f"{current_player_name}: Press D to roll dice"
                    action_color = GREEN
                elif 'end_turn' in legal_actions:
                    action_text = f"{current_player_name}: Press T to end turn"
                    action_color = GREEN
                else:
                    action_text = f"{current_player_name}'s turn"
                    action_color = YELLOW

                action_surf = self.font_medium.render(action_text, True, action_color)
                self.screen.blit(action_surf, (20, y_offset))
                y_offset += 40
            elif self.mode == 'human_vs_ai':
                # Show AI's turn
                action_text = f"{current_player_name} is thinking..."
                action_surf = self.font_medium.render(action_text, True, GRAY)
                self.screen.blit(action_surf, (20, y_offset))
                y_offset += 40

        phase_text = f"Phase: {self.env.game.game_phase}"
        phase = self.font_medium.render(phase_text, True, WHITE)
        self.screen.blit(phase, (20, y_offset))
        y_offset += 35

        current = self.env.game.current_player_index
        current_player = self.env.game.players[current]
        player_text = f"Current: {current_player.name}"
        player = self.font_medium.render(player_text, True, current_player.color)
        self.screen.blit(player, (20, y_offset))
        y_offset += 40

        if self.env.game.last_dice_roll:
            die1, die2, total = self.env.game.last_dice_roll
            dice_text = f"Last Roll: {die1} + {die2} = {total}"
            dice = self.font_medium.render(dice_text, True, WHITE)
            self.screen.blit(dice, (20, y_offset))
        y_offset += 40

        for i, player in enumerate(self.env.game.players):
            is_current = (i == current)
            color = player.color if not is_current else YELLOW

            name = self.font_medium.render(f"{player.name}", True, color)
            self.screen.blit(name, (20, y_offset))
            y_offset += 30

            vp_text = f"  VP: {player.calculate_victory_points()}"
            vp = self.font_small.render(vp_text, True, WHITE)
            self.screen.blit(vp, (20, y_offset))
            y_offset += 25

            if self.mode == 'human_only':
                # Show for current player in human_only mode, or all if debug
                if i == current or self.show_debug:
                    total_resources = sum(player.resources.values())
                    res_text = f"  Resources: {total_resources}"
                    res = self.font_small.render(res_text, True, WHITE)
                    self.screen.blit(res, (20, y_offset))
                    y_offset += 25
            elif i == self.human_player_index or self.show_debug:
                # In human_vs_ai, show for player 0 or all if debug
                total_resources = sum(player.resources.values())
                res_text = f"  Resources: {total_resources}"
                res = self.font_small.render(res_text, True, WHITE)
                self.screen.blit(res, (20, y_offset))
                y_offset += 25

            y_offset += 10

        y_offset += 20
        controls_title = self.font_medium.render("Controls:", True, WHITE)
        self.screen.blit(controls_title, (20, y_offset))
        y_offset += 30

        controls = [
            "D - Roll Dice",
            "T - End Turn",
            "B - Build Settlement",
            "C - Build City",
            "R - Build Road",
            "V - Buy Dev Card",
            "",
            "ARROWS - Pan View",
            "TAB - Toggle Debug",
            "ESC - Quit"
        ]

        if self.mode == 'ai_vs_ai':
            controls.append("SPACE - Pause AI")
            controls.append("+/- - Speed")

        for control in controls:
            text = self.font_small.render(control, True, WHITE)
            self.screen.blit(text, (20, y_offset))
            y_offset += 25

        if self.message:
            msg_rect = pygame.Rect(360, self.screen_height - 80,
                                   self.screen_width - 370, 70)
            pygame.draw.rect(self.screen, WHITE, msg_rect)
            pygame.draw.rect(self.screen, BLACK, msg_rect, 2)

            msg = self.font_medium.render(self.message, True, BLACK)
            msg_rect_center = msg.get_rect(center=msg_rect.center)
            self.screen.blit(msg, msg_rect_center)

    def draw_debug_overlay(self):
        """Draw debug information"""
        panel_width = 400
        panel_height = 300
        panel_x = self.screen_width - panel_width - 10
        panel_y = 10

        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        s = pygame.Surface((panel_width, panel_height))
        s.set_alpha(180)
        s.fill(BLACK)
        self.screen.blit(s, (panel_x, panel_y))
        pygame.draw.rect(self.screen, WHITE, panel_rect, 2)

        y = panel_y + 10

        title = self.font_medium.render("DEBUG INFO", True, YELLOW)
        self.screen.blit(title, (panel_x + 10, y))
        y += 35

        current = self.env.game.current_player_index
        player = self.env.game.players[current]
        obs = self.observations[current]

        debug_info = [
            f"Player: {player.name}",
            f"Phase: {self.env.game.game_phase}",
            f"Turn Phase: {self.env.game.turn_phase}",
            "",
            "Resources:",
            f"  Wood: {player.resources[ResourceType.WOOD]}",
            f"  Brick: {player.resources[ResourceType.BRICK]}",
            f"  Wheat: {player.resources[ResourceType.WHEAT]}",
            f"  Ore: {player.resources[ResourceType.ORE]}",
            f"  Sheep: {player.resources[ResourceType.SHEEP]}",
            "",
            "Legal Actions:",
        ]

        for line in debug_info:
            text = self.font_small.render(line, True, WHITE)
            self.screen.blit(text, (panel_x + 10, y))
            y += 20

        for action in obs['legal_actions'][:5]:
            text = self.font_small.render(f"  - {action}", True, GREEN)
            self.screen.blit(text, (panel_x + 10, y))
            y += 20


def main():
    """Main entry point"""
    print("="*60)
    print("CATAN VISUAL GAME")
    print("="*60)
    print("\nSelect game mode:")
    print("1. Human vs 3 AI (You are Player 1)")
    print("2. Watch 4 AI play")
    print("3. Human only (4 human players)")
    print()

    choice = input("Enter choice (1-3): ").strip()

    if choice == '1':
        mode = 'human_vs_ai'
        print("\n🎮 Starting Human vs AI mode")
        print("You are Player 1 (Red)")
    elif choice == '2':
        mode = 'ai_vs_ai'
        print("\n🤖 Starting AI vs AI mode")
        print("Press SPACE to pause, +/- to control speed")
    else:
        mode = 'human_only'
        print("\n👥 Starting Human only mode")
        print("4 human players take turns")

    print("\nStarting game...")
    print("="*60)

    game = VisualCatanGame(mode=mode)
    game.run()

    pygame.quit()
    print("\nGame ended. Thanks for playing!")


if __name__ == "__main__":
    main()