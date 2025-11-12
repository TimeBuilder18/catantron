"""
Client Window - Pygame window that connects to game server
Each player gets their own window showing shared board + private hand
"""

import pygame
import socket
import json
import threading
import os

# Resource colors for board rendering
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


class ClientWindow:
    """Player window that connects to game server"""

    def __init__(self, player_index, player_name, player_color, server_host='127.0.0.1', server_port=5555):
        self.player_index = player_index
        self.player_name = player_name
        self.player_color = player_color
        self.server_host = server_host
        self.server_port = server_port

        # Network
        self.socket = None
        self.game_state = None
        self.state_lock = threading.Lock()

        # Pygame - EXACT same size as main.py
        self.width = 1600
        self.height = 1000
        self.screen = None
        self.running = False

        # Messages
        self.messages = []

        # UI state (from main.py)
        self.show_buildable = False
        self.build_mode = None
        self.show_coords = False

    def position_window(self):
        """Position window based on player index - adjusted for 1600x1000 windows"""
        # These windows are much larger, so we may need to overlap or stack them
        # For now, just offset them slightly so you can see all titles
        positions = [
            (0, 0),         # Player 1: top-left corner
            (50, 50),       # Player 2: slightly offset
            (100, 100),     # Player 3: more offset
            (150, 150)      # Player 4: most offset
        ]

        if self.player_index < len(positions):
            x, y = positions[self.player_index]
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

    def connect_to_server(self):
        """Connect to the game server"""
        print(f"[{self.player_name}] Connecting to server...")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.server_host, self.server_port))
        print(f"[{self.player_name}] ✓ Connected to server!")

        # Start thread to receive game state
        thread = threading.Thread(target=self.receive_game_state)
        thread.daemon = True
        thread.start()

    def receive_game_state(self):
        """Background thread to receive game state updates from server"""
        buffer = ""
        while self.running:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    print(f"[{self.player_name}] Server disconnected")
                    break

                buffer += data

                # Process complete JSON messages (newline-delimited)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            with self.state_lock:
                                self.game_state = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"[{self.player_name}] JSON error: {e}")

            except Exception as e:
                if self.running:
                    print(f"[{self.player_name}] Receive error: {e}")
                break

    def send_action(self, action):
        """Send an action to the server"""
        try:
            self.socket.sendall((action + '\n').encode('utf-8'))
        except Exception as e:
            print(f"[{self.player_name}] Send error: {e}")

    def add_message(self, text, color=(255, 255, 255)):
        """Add a message to this window's message log"""
        self.messages.append((text, color, pygame.time.get_ticks()))
        if len(self.messages) > 5:
            self.messages.pop(0)

    def is_my_turn(self):
        """Check if it's this player's turn"""
        with self.state_lock:
            if self.game_state:
                return self.game_state['current_turn'] == self.player_index
        return False

    def init_window(self):
        """Initialize Pygame window"""
        self.position_window()

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Catan - {self.player_name}")

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 32)

    def draw_messages(self):
        """Draw messages at top of screen - EXACT same as main.py"""
        current_time = pygame.time.get_ticks()
        msg_y = 10
        for msg_text, msg_color, msg_time in self.messages[:]:
            if current_time - msg_time < 5000:  # Show for 5 seconds
                alpha = min(255, 255 - (current_time - msg_time) // 20)
                msg_surface = self.font.render(msg_text, True, msg_color)
                msg_surface.set_alpha(alpha)
                msg_rect = msg_surface.get_rect(center=(400, msg_y))

                # Background for readability
                bg_rect = pygame.Rect(msg_rect.x - 10, msg_rect.y - 5, msg_rect.width + 20, msg_rect.height + 10)
                bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
                bg_surface.set_alpha(150)
                bg_surface.fill((0, 0, 0))
                self.screen.blit(bg_surface, bg_rect)

                self.screen.blit(msg_surface, msg_rect)
                msg_y += 40
            else:
                self.messages.remove((msg_text, msg_color, msg_time))

    def draw_right_panel(self):
        """Draw right side panel - EXACT same as main.py"""
        if not self.game_state:
            return

        from game_system import ResourceType

        panel_x = 1200
        panel_width = 400

        # Panel background
        pygame.draw.rect(self.screen, (30, 30, 40), (panel_x, 0, panel_width, self.height))
        pygame.draw.line(self.screen, (100, 100, 100), (panel_x, 0), (panel_x, self.height), 3)

        y_pos = 20

        # Current Player - Large and prominent
        player_title_font = pygame.font.SysFont(None, 32)
        player_title = player_title_font.render(self.player_name, True, self.player_color)
        self.screen.blit(player_title, (panel_x + 20, y_pos))
        y_pos += 45

        with self.state_lock:
            my_resources = self.game_state.get('my_resources', {})
            players = self.game_state.get('players', [])
            current_turn = self.game_state.get('current_turn', 0)
            dice_rolled = self.game_state.get('dice_rolled', False)
            last_roll = self.game_state.get('last_roll')

            # Get my player data
            if self.player_index < len(players):
                my_player_data = players[self.player_index]
                vp_value = my_player_data.get('victory_points', 0)
            else:
                vp_value = 0

            # Victory Points - Big and clear
            vp_text = self.font.render(f"Victory Points: {vp_value}/10", True,
                                      (255, 215, 0) if vp_value >= 8 else (255, 255, 255))
            self.screen.blit(vp_text, (panel_x + 20, y_pos))
            y_pos += 35

            pygame.draw.line(self.screen, (100, 100, 100), (panel_x + 20, y_pos), (panel_x + panel_width - 20, y_pos), 2)
            y_pos += 15

            # Turn indicator - handle both initial placement and normal play
            game_phase = self.game_state.get('game_phase', 'NORMAL_PLAY')
            turn_phase = self.game_state.get('turn_phase', 'ROLL_DICE')

            if current_turn == self.player_index:
                phase_text = "YOUR TURN"
                phase_color = (100, 255, 100)

                # Show different instructions based on game phase
                if game_phase == "INITIAL_PLACEMENT_1":
                    phase_desc = "▶ Place first settlement (click vertex)"
                elif game_phase == "INITIAL_PLACEMENT_2":
                    phase_desc = "▶ Place second settlement (click vertex)"
                elif not dice_rolled:
                    phase_desc = "▶ Roll dice (D)"
                else:
                    phase_desc = "▶ Trade & Build (T=End)"
            else:
                player_names = ["Player 1", "Player 2", "Player 3", "Player 4"]
                if current_turn < len(player_names):
                    phase_text = f"{player_names[current_turn]}'s Turn"
                else:
                    phase_text = "Waiting..."
                phase_color = (200, 200, 200)
                phase_desc = "Please wait..."

            phase_surface = self.font.render(phase_text, True, phase_color)
            self.screen.blit(phase_surface, (panel_x + 20, y_pos))
            y_pos += 25

            desc_surface = self.small_font.render(phase_desc, True, (200, 200, 200))
            self.screen.blit(desc_surface, (panel_x + 20, y_pos))
            y_pos += 30

            # Dice Roll Display
            if last_roll:
                pygame.draw.rect(self.screen, (50, 50, 60), (panel_x + 20, y_pos, panel_width - 40, 65), border_radius=10)
                pygame.draw.rect(self.screen, (255, 215, 0), (panel_x + 20, y_pos, panel_width - 40, 65), 3, border_radius=10)

                dice_title = self.small_font.render("LAST ROLL", True, (255, 215, 0))
                self.screen.blit(dice_title, (panel_x + 30, y_pos + 8))

                if isinstance(last_roll, (list, tuple)) and len(last_roll) >= 3:
                    die1, die2, total = last_roll[0], last_roll[1], last_roll[2]
                else:
                    die1, die2, total = 0, 0, last_roll if isinstance(last_roll, int) else 0

                # Draw dice visually
                dice_x = panel_x + 30
                dice_y = y_pos + 30

                # Die 1
                pygame.draw.rect(self.screen, (255, 255, 255), (dice_x, dice_y, 30, 30), border_radius=5)
                die1_text = self.font.render(str(die1), True, (0, 0, 0))
                self.screen.blit(die1_text, (dice_x + 10, dice_y + 5))

                # Plus sign
                plus_text = self.small_font.render("+", True, (255, 255, 255))
                self.screen.blit(plus_text, (dice_x + 38, dice_y + 5))

                # Die 2
                pygame.draw.rect(self.screen, (255, 255, 255), (dice_x + 60, dice_y, 30, 30), border_radius=5)
                die2_text = self.font.render(str(die2), True, (0, 0, 0))
                self.screen.blit(die2_text, (dice_x + 70, dice_y + 5))

                # Equals sign
                equals_text = self.small_font.render("=", True, (255, 255, 255))
                self.screen.blit(equals_text, (dice_x + 98, dice_y + 5))

                # Total
                total_color = (255, 0, 0) if total in [6, 8] else (255, 255, 255)
                total_text = self.font.render(str(total), True, total_color)
                self.screen.blit(total_text, (dice_x + 120, dice_y + 5))

                y_pos += 70

            pygame.draw.line(self.screen, (100, 100, 100), (panel_x + 20, y_pos), (panel_x + panel_width - 20, y_pos), 2)
            y_pos += 15

            # Resources Section
            resources_title = self.font.render("RESOURCES", True, (100, 200, 255))
            self.screen.blit(resources_title, (panel_x + 20, y_pos))
            y_pos += 30

            for res_name, amount in my_resources.items():
                color = (255, 255, 255) if amount > 0 else (100, 100, 100)

                # Resource name and count
                text = self.small_font.render(f"{res_name.capitalize()}: {amount}", True, color)
                self.screen.blit(text, (panel_x + 30, y_pos))

                # Draw small bar to visualize amount
                if amount > 0:
                    bar_width = min(amount * 20, 200)
                    pygame.draw.rect(self.screen, color, (panel_x + 150, y_pos + 5, bar_width, 10))

                y_pos += 20

            y_pos += 5
            pygame.draw.line(self.screen, (100, 100, 100), (panel_x + 20, y_pos), (panel_x + panel_width - 20, y_pos), 2)
            y_pos += 10

    def draw_board(self):
        """Draw the game board - exactly like main.py"""
        if not self.game_state:
            # Show "waiting for server" message
            waiting_text = self.font.render("Connecting to server...", True, (255, 255, 100))
            self.screen.blit(waiting_text, (600, 500))
            return

        with self.state_lock:
            # Import Tile to reconstruct board
            from tile import Tile

            # Reconstruct tiles for proper rendering
            tiles = []
            for tile_data in self.game_state['tiles']:
                tile = Tile(tile_data['q'], tile_data['r'], 50, tile_data['resource'], tile_data['number'])
                tiles.append(tile)

            # Compute center offset - center in LEFT area (before panel at x=1200)
            # Same as main.py: center board in the main view area
            if tiles:
                xs = [t.x for t in tiles]
                ys = [t.y for t in tiles]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                board_cx, board_cy = (min_x + max_x) / 2, (min_y + max_y) / 2
                # Center in the left 1200px area (600px is center of 1200px)
                offset = (600 - board_cx, self.height / 2 - board_cy)
            else:
                offset = (0, 0)

            # Draw tiles (same as main.py)
            for tile in tiles:
                color = RESOURCE_COLORS.get(tile.resource, RESOURCE_COLORS["desert"])
                tile.draw(self.screen, fill_color=color, offset=offset, show_coords=False)

            # Draw vertices (from game state)
            if 'vertices' in self.game_state:
                for vertex_data in self.game_state['vertices']:
                    x = vertex_data['x'] + offset[0]
                    y = vertex_data['y'] + offset[1]
                    pygame.draw.circle(self.screen, (100, 100, 100), (int(x), int(y)), 3)

            # Draw edges (from game state)
            if 'edges' in self.game_state:
                for edge_data in self.game_state['edges']:
                    x1 = edge_data['x1'] + offset[0]
                    y1 = edge_data['y1'] + offset[1]
                    x2 = edge_data['x2'] + offset[0]
                    y2 = edge_data['y2'] + offset[1]
                    pygame.draw.line(self.screen, (150, 150, 150), (x1, y1), (x2, y2), 1)

            # Draw settlements (same as main.py)
            for settlement in self.game_state['settlements']:
                x = settlement['x'] + offset[0]
                y = settlement['y'] + offset[1]
                player_idx = settlement['player_index']
                player_color = self.game_state['players'][player_idx]['color']

                pygame.draw.circle(self.screen, player_color, (int(x), int(y)), 8)
                pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 8, 2)

            # Draw cities (same as main.py)
            for city in self.game_state['cities']:
                x = city['x'] + offset[0]
                y = city['y'] + offset[1]
                player_idx = city['player_index']
                player_color = self.game_state['players'][player_idx]['color']

                pygame.draw.rect(self.screen, player_color, (x - 10, y - 10, 20, 20))
                pygame.draw.rect(self.screen, (0, 0, 0), (x - 10, y - 10, 20, 20), 2)

            # Draw roads (same as main.py)
            for road in self.game_state['roads']:
                x1 = road['x1'] + offset[0]
                y1 = road['y1'] + offset[1]
                x2 = road['x2'] + offset[0]
                y2 = road['y2'] + offset[1]
                player_idx = road['player_index']
                player_color = self.game_state['players'][player_idx]['color']

                pygame.draw.line(self.screen, player_color, (x1, y1), (x2, y2), 4)

            # Draw ports (same as main.py)
            port_font = pygame.font.Font(None, 20)
            port_label_font = pygame.font.Font(None, 16)
            for port in self.game_state['ports']:
                x1 = port['x1'] + offset[0]
                y1 = port['y1'] + offset[1]
                x2 = port['x2'] + offset[0]
                y2 = port['y2'] + offset[1]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                port_type = port['port_type']

                # Determine port color, text, and label (same as main.py)
                if port_type == "GENERIC":
                    port_color = (200, 200, 200)
                    port_text = "3:1"
                    resource_label = None
                else:
                    resource_name = port_type.lower()
                    port_color = RESOURCE_COLORS.get(resource_name, (255, 255, 255))
                    port_text = "2:1"
                    resource_label = resource_name.capitalize()

                # Draw thicker edge line for the port edge
                pygame.draw.line(self.screen, port_color, (x1, y1), (x2, y2), 5)

                # Draw port icon at midpoint
                pygame.draw.circle(self.screen, port_color, (int(mid_x), int(mid_y)), 14)
                pygame.draw.circle(self.screen, (0, 0, 0), (int(mid_x), int(mid_y)), 14, 2)

                # Draw port ratio text inside circle
                text_surface = port_font.render(port_text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(int(mid_x), int(mid_y)))
                self.screen.blit(text_surface, text_rect)

                # Draw resource label below circle for specialized ports
                if resource_label:
                    label_surface = port_label_font.render(resource_label, True, port_color)
                    label_rect = label_surface.get_rect(center=(int(mid_x), int(mid_y) + 20))
                    # Draw background for better visibility
                    bg_rect = label_rect.inflate(4, 2)
                    pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
                    pygame.draw.rect(self.screen, port_color, bg_rect, 1)
                    self.screen.blit(label_surface, label_rect)


    def handle_events(self):
        """Handle Pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            # Only allow actions on your turn
            if not self.is_my_turn():
                continue

            if event.type == pygame.KEYDOWN:
                self.handle_keypress(event.key)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouseclick(event.pos)

        return True

    def handle_mouseclick(self, pos):
        """Handle mouse clicks for building placement"""
        # TODO: Implement building placement
        # For now just show a message
        with self.state_lock:
            game_phase = self.game_state.get('game_phase', 'NORMAL_PLAY') if self.game_state else 'NORMAL_PLAY'

            if game_phase in ["INITIAL_PLACEMENT_1", "INITIAL_PLACEMENT_2"]:
                self.add_message("Settlement placement: Coming soon!", (255, 200, 100))

    def handle_keypress(self, key):
        """Handle keyboard input"""
        # Roll dice
        if key == pygame.K_d:
            with self.state_lock:
                if self.game_state and not self.game_state.get('dice_rolled', False):
                    self.send_action("ROLL_DICE")
                    self.add_message("Rolling dice...", (255, 255, 0))

        # End turn
        elif key == pygame.K_t:
            self.send_action("END_TURN")
            self.add_message("Ending turn...", (100, 255, 100))

    def run(self):
        """Main loop for this client window"""
        self.running = True
        self.init_window()
        self.connect_to_server()

        clock = pygame.time.Clock()

        print(f"[{self.player_name}] ✓ Window started! Waiting for game state...")

        while self.running:
            if not self.handle_events():
                break

            # Clear screen - same color as main.py
            self.screen.fill((20, 50, 80))

            # Draw in exact same order as main.py
            self.draw_board()           # Draw board first
            self.draw_messages()        # Messages at top
            self.draw_right_panel()     # Right panel with all UI

            pygame.display.flip()
            clock.tick(30)  # 30 FPS

        pygame.quit()
        if self.socket:
            self.socket.close()
        print(f"[{self.player_name}] Window closed")


def run_client_window(player_index, player_name, player_color):
    """Entry point for running a client window in a separate process"""
    client = ClientWindow(player_index, player_name, player_color)
    client.run()


if __name__ == "__main__":
    # For testing - run as Player 1
    run_client_window(0, "Player 1 (Red)", (255, 50, 50))
