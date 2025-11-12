"""
Catan Game Server - Manages shared game state
Runs in a separate process and handles client connections via sockets
"""

import socket
import threading
import json
import random
from tile import Tile
from game_system import (Player, Robber, GameBoard, GameSystem, ResourceType,
                         Settlement, City, Road)

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
    resources = RESOURCES.copy()
    random.shuffle(resources)

    for i, tile in enumerate(tiles):
        if i < len(resources):
            tile.resource = resources[i]

    # Place robber on desert
    for tile in tiles:
        if tile.resource == "desert":
            tile.number = None
            robber.move_to_tile(tile)
            break

    # Assign numbers to non-desert tiles
    non_desert_tiles = [t for t in tiles if t.resource != "desert"]
    nums = NUMBER_TOKENS.copy()
    random.shuffle(nums)

    for i, tile in enumerate(non_desert_tiles):
        if i < len(nums):
            tile.number = nums[i]


class GameServer:
    """Socket server that manages game state and handles client connections"""

    def __init__(self, host='127.0.0.1', port=5555):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []  # List of (client_socket, player_index, address)
        self.game_system = None
        self.game_board = None
        self.running = False
        self.state_lock = threading.Lock()

    def initialize_game(self):
        """Set up the game board and system"""
        print("Initializing game board...")

        # Create board
        tile_size = 50
        tiles = create_hexagonal_board(tile_size, radius=2)

        for t in tiles:
            t.find_neighbors(tiles)

        robber = Robber()
        assign_resources_numbers(tiles, robber)

        self.game_board = GameBoard(tiles)

        # Create 4 players
        players = [
            Player("Player 1 (Red)", (255, 50, 50)),
            Player("Player 2 (Blue)", (50, 50, 255)),
            Player("Player 3 (Yellow)", (255, 255, 50)),
            Player("Player 4 (White)", (255, 255, 255))
        ]

        # Give starting resources for testing
        for player in players:
            player.add_resource(ResourceType.WOOD, 3)
            player.add_resource(ResourceType.BRICK, 3)
            player.add_resource(ResourceType.WHEAT, 2)
            player.add_resource(ResourceType.SHEEP, 2)
            player.add_resource(ResourceType.ORE, 1)

        # Create game system
        self.game_system = GameSystem(self.game_board, players)
        self.game_system.robber = robber

        print("✓ Game initialized!")
        print(f"  {len(tiles)} tiles created")
        print(f"  {len(self.game_board.ports)} ports generated")
        print(f"  4 players ready")

    def serialize_game_state(self, player_index):
        """Convert game state to JSON for a specific player"""
        with self.state_lock:
            # Get player-specific data
            player = self.game_system.players[player_index]
            current_player_index = self.game_system.players.index(
                self.game_system.get_current_player()
            )

            state = {
                'current_turn': current_player_index,
                'dice_rolled': self.game_system.dice_rolled,
                'last_roll': self.game_system.last_roll,

                # Player's private resources
                'my_resources': {
                    'wood': player.resources[ResourceType.WOOD],
                    'brick': player.resources[ResourceType.BRICK],
                    'wheat': player.resources[ResourceType.WHEAT],
                    'sheep': player.resources[ResourceType.SHEEP],
                    'ore': player.resources[ResourceType.ORE]
                },

                # All players' public info
                'players': [
                    {
                        'name': p.name,
                        'color': p.color,
                        'victory_points': p.victory_points,
                        'resource_count': sum(p.resources.values())
                    }
                    for p in self.game_system.players
                ],

                # Board state
                'tiles': [
                    {
                        'q': t.q,
                        'r': t.r,
                        'resource': t.resource,
                        'number': t.number,
                        'x': t.x,
                        'y': t.y
                    }
                    for t in self.game_board.tiles
                ],

                # Structures
                'settlements': [
                    {
                        'x': v.x,
                        'y': v.y,
                        'player_index': self.game_system.players.index(v.structure.player)
                    }
                    for v in self.game_board.vertices
                    if v.structure and isinstance(v.structure, Settlement)
                ],

                'cities': [
                    {
                        'x': v.x,
                        'y': v.y,
                        'player_index': self.game_system.players.index(v.structure.player)
                    }
                    for v in self.game_board.vertices
                    if v.structure and isinstance(v.structure, City)
                ],

                'roads': [
                    {
                        'x1': e.vertex1.x,
                        'y1': e.vertex1.y,
                        'x2': e.vertex2.x,
                        'y2': e.vertex2.y,
                        'player_index': self.game_system.players.index(e.structure.player)
                    }
                    for e in self.game_board.edges
                    if e.structure
                ],

                'ports': [
                    {
                        'x1': p.vertex1.x,
                        'y1': p.vertex1.y,
                        'x2': p.vertex2.x,
                        'y2': p.vertex2.y,
                        'port_type': p.port_type.name
                    }
                    for p in self.game_board.ports
                ]
            }

            return json.dumps(state)

    def handle_client(self, client_socket, player_index, address):
        """Handle a single client connection"""
        print(f"✓ Player {player_index + 1} connected from {address}")

        try:
            while self.running:
                # Send current game state
                state_json = self.serialize_game_state(player_index)
                client_socket.sendall((state_json + '\n').encode('utf-8'))

                # Receive action from client (non-blocking with timeout)
                client_socket.settimeout(0.1)
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break

                    message = data.decode('utf-8').strip()
                    if message:
                        self.process_action(player_index, message)

                except socket.timeout:
                    pass  # No data received, continue loop

        except Exception as e:
            print(f"Client {player_index + 1} error: {e}")
        finally:
            client_socket.close()
            print(f"✓ Player {player_index + 1} disconnected")

    def process_action(self, player_index, action):
        """Process an action from a client"""
        with self.state_lock:
            player = self.game_system.players[player_index]
            current_player = self.game_system.get_current_player()

            # Only allow actions from current player
            if player != current_player:
                return

            if action == "ROLL_DICE":
                if self.game_system.can_roll_dice():
                    result = self.game_system.roll_dice()
                    if result:
                        print(f"  Player {player_index + 1} rolled: {result[2]} ({result[0]}+{result[1]})")

            elif action == "END_TURN":
                success, message = self.game_system.end_turn()
                if success:
                    new_player = self.game_system.get_current_player()
                    new_index = self.game_system.players.index(new_player)
                    print(f"  Turn ended. Now: Player {new_index + 1}'s turn")

    def start(self):
        """Start the server"""
        self.initialize_game()
        self.running = True

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(4)

        print(f"\n{'='*60}")
        print(f"CATAN GAME SERVER STARTED")
        print(f"{'='*60}")
        print(f"Listening on {self.host}:{self.port}")
        print(f"Waiting for 4 players to connect...\n")

        # Accept 4 client connections
        for i in range(4):
            client_socket, address = self.server_socket.accept()
            self.clients.append((client_socket, i, address))

            # Start handler thread for this client
            thread = threading.Thread(
                target=self.handle_client,
                args=(client_socket, i, address)
            )
            thread.daemon = True
            thread.start()

        print(f"\n✓ All 4 players connected! Game starting...")
        print(f"It's Player 1's turn.\n")

    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        for client_socket, _, _ in self.clients:
            client_socket.close()


def main():
    """Run the game server"""
    server = GameServer()
    try:
        server.start()
        # Keep server running
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.stop()


if __name__ == "__main__":
    main()
