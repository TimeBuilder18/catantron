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
        """Draw the game board"""
        if not self.game_state:
            return

        board_x = 250
        board_y = 100

        with self.state_lock:
            # Draw hexagonal tiles
            for tile_data in self.game_state['tiles']:
                x = tile_data['x'] + board_x
                y = tile_data['y'] + board_y
                resource = tile_data['resource']
                number = tile_data['number']

                # Draw hex
                color = RESOURCE_COLORS.get(resource, RESOURCE_COLORS['desert'])
                self.draw_hexagon(x, y, 50, color)

                # Draw number
                if number:
                    # Highlight 6 and 8 (best numbers)
                    num_color = (255, 100, 100) if number in [6, 8] else (255, 255, 255)
                    num_text = self.font.render(str(number), True, num_color)
                    num_rect = num_text.get_rect(center=(int(x), int(y)))

                    # Background circle
                    pygame.draw.circle(self.screen, (50, 50, 50), (int(x), int(y)), 18)
                    self.screen.blit(num_text, num_rect)

            # Draw settlements
            for settlement in self.game_state['settlements']:
                x = settlement['x'] + board_x
                y = settlement['y'] + board_y
                player_idx = settlement['player_index']
                player_color = self.game_state['players'][player_idx]['color']

                pygame.draw.circle(self.screen, player_color, (int(x), int(y)), 8)
                pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 8, 2)

            # Draw cities
            for city in self.game_state['cities']:
                x = city['x'] + board_x
                y = city['y'] + board_y
                player_idx = city['player_index']
                player_color = self.game_state['players'][player_idx]['color']

                pygame.draw.rect(self.screen, player_color, (x - 10, y - 10, 20, 20))
                pygame.draw.rect(self.screen, (0, 0, 0), (x - 10, y - 10, 20, 20), 2)

            # Draw roads
            for road in self.game_state['roads']:
                x1 = road['x1'] + board_x
                y1 = road['y1'] + board_y
                x2 = road['x2'] + board_x
                y2 = road['y2'] + board_y
                player_idx = road['player_index']
                player_color = self.game_state['players'][player_idx]['color']

                pygame.draw.line(self.screen, player_color, (x1, y1), (x2, y2), 6)

            # Draw ports
            port_font = pygame.font.Font(None, 14)
            for port in self.game_state['ports']:
                x1 = port['x1'] + board_x
                y1 = port['y1'] + board_y
                x2 = port['x2'] + board_x
                y2 = port['y2'] + board_y
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                port_type = port['port_type']

                if port_type == "GENERIC":
                    port_color = (200, 200, 200)
                    port_text = "3:1"
                else:
                    resource_name = port_type.lower()
                    port_color = RESOURCE_COLORS.get(resource_name, (255, 255, 255))
                    port_text = "2:1"

                pygame.draw.circle(self.screen, port_color, (int(mid_x), int(mid_y)), 10)
                pygame.draw.circle(self.screen, (0, 0, 0), (int(mid_x), int(mid_y)), 10, 2)

                text_surface = port_font.render(port_text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(int(mid_x), int(mid_y)))
                self.screen.blit(text_surface, text_rect)

    def draw_hexagon(self, cx, cy, size, color):
        """Draw a hexagon at (cx, cy)"""
        import math
        points = []
        for i in range(6):
            angle = math.pi / 3 * i
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            points.append((x, y))

        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (0, 0, 0), points, 2)

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

        # Wait for initial game state
        print(f"[{self.player_name}] Waiting for game state...")
        while self.running and not self.game_state:
            import time
            time.sleep(0.1)

        print(f"[{self.player_name}] ✓ Game state received! Starting window...")

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
