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

        # Pygame
        self.width = 1000
        self.height = 700
        self.screen = None
        self.running = False

        # Messages
        self.messages = []

    def position_window(self):
        """Position window based on player index"""
        positions = [
            (50, 50),      # Player 1: top-left
            (1070, 50),    # Player 2: top-right
            (50, 780),     # Player 3: bottom-left
            (1070, 780)    # Player 4: bottom-right
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

    def draw_header(self):
        """Draw window header"""
        pygame.draw.rect(self.screen, (30, 30, 40), (0, 0, self.width, 80))

        # Player color indicator
        color_box = pygame.Rect(20, 20, 40, 40)
        pygame.draw.rect(self.screen, self.player_color, color_box)
        pygame.draw.rect(self.screen, (255, 255, 255), color_box, 2)

        # Player name
        name_text = self.title_font.render(self.player_name, True, (255, 255, 255))
        self.screen.blit(name_text, (70, 25))

        # Turn indicator
        with self.state_lock:
            if self.game_state:
                current_turn = self.game_state['current_turn']
                if current_turn == self.player_index:
                    turn_text = self.font.render("YOUR TURN", True, (100, 255, 100))
                    self.screen.blit(turn_text, (self.width - 150, 30))
                else:
                    player_names = ["P1", "P2", "P3", "P4"]
                    turn_text = self.small_font.render(
                        f"{player_names[current_turn]}'s turn",
                        True, (150, 150, 150)
                    )
                    self.screen.blit(turn_text, (self.width - 120, 35))

    def draw_resources(self):
        """Draw player's private resources"""
        y_pos = 90

        title = self.font.render("YOUR RESOURCES:", True, (255, 255, 0))
        self.screen.blit(title, (20, y_pos))
        y_pos += 30

        with self.state_lock:
            if self.game_state and 'my_resources' in self.game_state:
                resources = self.game_state['my_resources']

                for resource_name, amount in resources.items():
                    color = (255, 255, 255) if amount > 0 else (100, 100, 100)
                    text = self.small_font.render(
                        f"{resource_name.capitalize()}: {amount}",
                        True, color
                    )
                    self.screen.blit(text, (20, y_pos))
                    y_pos += 20

    def draw_board(self):
        """Draw the game board - exactly like main.py"""
        if not self.game_state:
            # Show "waiting for server" message
            waiting_text = self.font.render("Connecting to server...", True, (255, 255, 100))
            self.screen.blit(waiting_text, (self.width // 2 - 100, self.height // 2))
            return

        with self.state_lock:
            # Import Tile to reconstruct board
            from tile import Tile
            from game_system import Settlement, City, Vertex, Edge, Port, PortType

            # Reconstruct tiles for proper rendering
            tiles = []
            for tile_data in self.game_state['tiles']:
                tile = Tile(tile_data['q'], tile_data['r'], 50, tile_data['resource'], tile_data['number'])
                tiles.append(tile)

            # Compute center offset (same as main.py)
            if tiles:
                xs = [t.x for t in tiles]
                ys = [t.y for t in tiles]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                board_cx, board_cy = (min_x + max_x) / 2, (min_y + max_y) / 2
                offset = (self.width / 2 - board_cx, self.height / 2 - board_cy)
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

    def draw_messages(self):
        """Draw recent messages"""
        y_pos = self.height - 120
        current_time = pygame.time.get_ticks()

        for message, color, timestamp in self.messages:
            age = current_time - timestamp
            if age < 5000:  # Show for 5 seconds
                text = self.small_font.render(message, True, color)
                self.screen.blit(text, (20, y_pos))
                y_pos += 20

    def draw_controls(self):
        """Draw control hints"""
        y_pos = self.height - 180

        hints = [
            "Controls:",
            "  D - Roll dice",
            "  T - End turn"
        ]

        for hint in hints:
            text = self.small_font.render(hint, True, (150, 150, 150))
            self.screen.blit(text, (20, y_pos))
            y_pos += 18

    def draw_debug_info(self):
        """Draw debug information"""
        debug_y = self.height - 30

        # Connection status
        if self.socket:
            status_text = self.small_font.render("✓ Connected to server", True, (100, 255, 100))
        else:
            status_text = self.small_font.render("✗ Not connected", True, (255, 100, 100))

        self.screen.blit(status_text, (self.width - 200, debug_y))

        # Game state status
        with self.state_lock:
            if self.game_state:
                state_text = self.small_font.render(f"✓ Game active ({len(self.game_state.get('tiles', []))} tiles)", True, (100, 255, 100))
            else:
                state_text = self.small_font.render("Waiting for game state...", True, (255, 255, 100))

        self.screen.blit(state_text, (self.width - 200, debug_y - 20))

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

        return True

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

            # Clear screen
            self.screen.fill((20, 20, 30))

            # Draw UI components
            self.draw_header()
            self.draw_resources()
            self.draw_board()
            self.draw_controls()
            self.draw_messages()
            self.draw_debug_info()

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
