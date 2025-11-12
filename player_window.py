"""
Player Window - Individual Pygame window for each player
Shows the shared board + player's private hand
"""

import pygame
import threading
from game_system import ResourceType, Settlement, City

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

class PlayerWindow:
    """Individual window for one player"""

    def __init__(self, player, game_system, game_board, offset, player_index):
        self.player = player
        self.game_system = game_system
        self.game_board = game_board
        self.offset = offset
        self.player_index = player_index

        # Window settings
        self.width = 1000
        self.height = 700
        self.running = False
        self.screen = None

        # Lock for thread-safe state access
        self.state_lock = threading.Lock()

        # UI state
        self.show_buildable = False
        self.build_mode = "SETTLEMENT"
        self.trade_mode = False
        self.selected_trade_partner = 0

        # Trade amounts
        self.offering_resources = {r: 0 for r in ResourceType}
        self.requesting_resources = {r: 0 for r in ResourceType}

        # Messages
        self.messages = []

    def position_window(self):
        """Position window based on player index"""
        positions = [
            (50, 50),      # Player 1: top-left
            (1050, 50),    # Player 2: top-right
            (50, 750),     # Player 3: bottom-left
            (1050, 750)    # Player 4: bottom-right
        ]

        if self.player_index < len(positions):
            x, y = positions[self.player_index]
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x},{y}"

    def init_window(self):
        """Initialize Pygame window for this player"""
        self.position_window()

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Catan - {self.player.name}")

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 32)

    def add_message(self, text, color=(255, 255, 255)):
        """Add a message to this window's message log"""
        self.messages.append((text, color, pygame.time.get_ticks()))
        # Keep only last 5 messages
        if len(self.messages) > 5:
            self.messages.pop(0)

    def is_my_turn(self):
        """Check if it's this player's turn"""
        with self.state_lock:
            return self.game_system.get_current_player() == self.player

    def draw_header(self):
        """Draw window header with player info"""
        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), (0, 0, self.width, 80))

        # Player name and color indicator
        color_box = pygame.Rect(20, 20, 40, 40)
        pygame.draw.rect(self.screen, self.player.color, color_box)
        pygame.draw.rect(self.screen, (255, 255, 255), color_box, 2)

        name_text = self.title_font.render(self.player.name, True, (255, 255, 255))
        self.screen.blit(name_text, (70, 25))

        # Turn indicator
        with self.state_lock:
            current_player = self.game_system.get_current_player()
            if current_player == self.player:
                turn_text = self.font.render("YOUR TURN", True, (100, 255, 100))
                self.screen.blit(turn_text, (self.width - 150, 30))
            else:
                turn_text = self.small_font.render(f"{current_player.name}'s turn", True, (150, 150, 150))
                self.screen.blit(turn_text, (self.width - 150, 35))

    def draw_resources(self):
        """Draw player's resources (PRIVATE - only this player sees)"""
        y_pos = 90

        title = self.font.render("YOUR RESOURCES:", True, (255, 255, 0))
        self.screen.blit(title, (20, y_pos))
        y_pos += 30

        resource_names = {
            ResourceType.WOOD: "Wood",
            ResourceType.BRICK: "Brick",
            ResourceType.WHEAT: "Wheat",
            ResourceType.SHEEP: "Sheep",
            ResourceType.ORE: "Ore"
        }

        for resource_type, name in resource_names.items():
            amount = self.player.resources[resource_type]
            color = (255, 255, 255) if amount > 0 else (100, 100, 100)
            text = self.small_font.render(f"{name}: {amount}", True, color)
            self.screen.blit(text, (20, y_pos))
            y_pos += 20

    def draw_messages(self):
        """Draw recent messages"""
        y_pos = self.height - 120
        current_time = pygame.time.get_ticks()

        for message, color, timestamp in self.messages:
            # Fade out old messages
            age = current_time - timestamp
            if age < 5000:  # Show for 5 seconds
                alpha = max(0, 255 - (age / 5000 * 255))
                text = self.small_font.render(message, True, color)
                self.screen.blit(text, (20, y_pos))
                y_pos += 20

    def handle_events(self):
        """Handle Pygame events for this window"""
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
                if self.game_system.can_roll_dice():
                    result = self.game_system.roll_dice()
                    if result:
                        self.add_message(f"Rolled: {result[2]} ({result[0]}+{result[1]})", (255, 255, 0))

        # End turn
        elif key == pygame.K_t:
            with self.state_lock:
                success, message = self.game_system.end_turn()
                self.add_message(message, (100, 255, 100) if success else (255, 100, 100))

        # Toggle trade mode
        elif key == pygame.K_5:
            self.trade_mode = not self.trade_mode
            self.add_message("Trade mode: " + ("ON" if self.trade_mode else "OFF"), (255, 200, 255))

    def draw_board(self):
        """Draw the shared game board (all players see same board)"""
        # Board area - center of screen
        board_x = 250
        board_y = 100
        board_offset = (board_x + self.offset[0], board_y + self.offset[1])

        with self.state_lock:
            # Draw tiles
            for tile in self.game_board.tiles:
                color = RESOURCE_COLORS.get(tile.resource, RESOURCE_COLORS["desert"])
                tile.draw(self.screen, fill_color=color, offset=board_offset, show_coords=False)

            # Draw vertices (settlements/cities)
            for vertex in self.game_board.vertices:
                x = vertex.x + board_offset[0]
                y = vertex.y + board_offset[1]

                if vertex.structure:
                    player_color = vertex.structure.player.color
                    if isinstance(vertex.structure, Settlement):
                        pygame.draw.circle(self.screen, player_color, (int(x), int(y)), 8)
                        pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), 8, 2)
                    elif isinstance(vertex.structure, City):
                        pygame.draw.rect(self.screen, player_color, (x - 10, y - 10, 20, 20))
                        pygame.draw.rect(self.screen, (0, 0, 0), (x - 10, y - 10, 20, 20), 2)
                else:
                    pygame.draw.circle(self.screen, (100, 100, 100), (int(x), int(y)), 3)

            # Draw edges (roads)
            for edge in self.game_board.edges:
                x1 = edge.vertex1.x + board_offset[0]
                y1 = edge.vertex1.y + board_offset[1]
                x2 = edge.vertex2.x + board_offset[0]
                y2 = edge.vertex2.y + board_offset[1]

                if edge.structure:
                    player_color = edge.structure.player.color
                    pygame.draw.line(self.screen, player_color, (x1, y1), (x2, y2), 4)
                else:
                    pygame.draw.line(self.screen, (150, 150, 150), (x1, y1), (x2, y2), 1)

            # Draw ports
            port_font = pygame.font.Font(None, 16)
            for port in self.game_board.ports:
                x1 = port.vertex1.x + board_offset[0]
                y1 = port.vertex1.y + board_offset[1]
                x2 = port.vertex2.x + board_offset[0]
                y2 = port.vertex2.y + board_offset[1]
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                if port.port_type.name == "GENERIC":
                    port_color = (200, 200, 200)
                    port_text = "3:1"
                else:
                    resource_name = port.port_type.name.lower()
                    port_color = RESOURCE_COLORS.get(resource_name, (255, 255, 255))
                    port_text = "2:1"

                pygame.draw.circle(self.screen, port_color, (int(mid_x), int(mid_y)), 10)
                pygame.draw.circle(self.screen, (0, 0, 0), (int(mid_x), int(mid_y)), 10, 2)

                text_surface = port_font.render(port_text, True, (0, 0, 0))
                text_rect = text_surface.get_rect(center=(int(mid_x), int(mid_y)))
                self.screen.blit(text_surface, text_rect)

    def run(self):
        """Main loop for this window"""
        self.running = True
        self.init_window()
        clock = pygame.time.Clock()

        while self.running:
            if not self.handle_events():
                break

            # Clear screen
            self.screen.fill((20, 20, 30))

            # Draw UI components
            self.draw_header()
            self.draw_resources()
            self.draw_board()  # Draw shared board in center
            self.draw_messages()

            # TODO: Draw trade interface if active

            pygame.display.flip()
            clock.tick(30)  # 30 FPS

        pygame.quit()
