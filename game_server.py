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
    """Assign resources and numbers with constraints - EXACT copy from main.py"""
    all_resources = RESOURCES.copy()
    random.shuffle(all_resources)

    for i, tile in enumerate(tiles):
        if i < len(all_resources):
            tile.resource = all_resources[i]
        else:
            tile.resource = "forest"

    desert_tile = None
    for tile in tiles:
        if tile.resource == "desert":
            tile.number = None
            desert_tile = tile
            robber.move_to_tile(tile)
            break

    non_desert_tiles = [t for t in tiles if t.resource != "desert"]
    assign_numbers_with_constraints(non_desert_tiles)


def assign_numbers_with_constraints(tiles):
    """Assign numbers ensuring 6s and 8s are not adjacent - EXACT copy from main.py"""
    nums = NUMBER_TOKENS.copy()
    red_numbers = [6, 8]

    sixes = [n for n in nums if n == 6]
    eights = [n for n in nums if n == 8]
    red_nums_to_place = sixes + eights

    max_attempts = 1000
    for attempt in range(max_attempts):
        for tile in tiles:
            tile.number = None

        success = True
        remaining_nums = nums.copy()

        for red_num in red_nums_to_place:
            valid_tiles = []
            for tile in tiles:
                if tile.number is None and can_place_red_number(tile, red_num):
                    valid_tiles.append(tile)

            if not valid_tiles:
                success = False
                break

            chosen_tile = random.choice(valid_tiles)
            chosen_tile.number = red_num
            remaining_nums.remove(red_num)

        if not success:
            continue

        # Shuffle and assign remaining numbers
        random.shuffle(remaining_nums)
        remaining_tiles = [t for t in tiles if t.number is None]

        for i, tile in enumerate(remaining_tiles):
            if i < len(remaining_nums):
                tile.number = remaining_nums[i]

        if all(t.number is not None for t in tiles if t.resource != "desert"):
            if is_board_valid(tiles):
                return

    print("Warning: Could not assign numbers without red adjacency.")
    random.shuffle(nums)
    for i, tile in enumerate(tiles):
        if i < len(nums):
            tile.number = nums[i]


def can_place_red_number(tile, red_number):
    """Check if a red number (6 or 8) can be placed on a tile - EXACT copy from main.py"""
    red_numbers = [6, 8]

    for neighbor in tile.neighbors:
        if neighbor.number in red_numbers:
            return False

    return True


def validate_board(tiles):
    """Validate that no red numbers (6,8) are adjacent - EXACT copy from main.py"""
    red_numbers = [6, 8]
    violations = []

    for tile in tiles:
        if tile.number in red_numbers:
            for neighbor in tile.neighbors:
                if neighbor.number in red_numbers:
                    violations.append(f"Red numbers adjacent: Tile ({tile.q},{tile.r}) has {tile.number}")

    return violations


def is_board_valid(tiles):
    """Check if board is valid - EXACT copy from main.py"""
    return len(validate_board(tiles)) == 0


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
        print("[SERVER] Initializing game board...")
        import time
        start = time.time()

        # Create board
        print("[SERVER] Creating hexagonal tiles...")
        tile_size = 50
        tiles = create_hexagonal_board(tile_size, radius=2)
        print(f"[SERVER] ✓ Created {len(tiles)} tiles in {time.time()-start:.2f}s")

        for t in tiles:
            t.find_neighbors(tiles)

        robber = Robber()
        assign_resources_numbers(tiles, robber)

        self.game_board = GameBoard(tiles)

        # Debug: Print port info
        print(f"\n[SERVER] Port generation:")
        print(f"  Total edges in board: {len(self.game_board.edges)}")
        print(f"  Ports generated: {len(self.game_board.ports)}")

        # Check if ports are on coastal edges
        tile_map = {(tile.q, tile.r): tile for tile in tiles}
        HEX_DIRECTIONS = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]

        coastal_edge_count = 0
        for tile in tiles:
            for dir_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
                neighbor_q = tile.q + dq
                neighbor_r = tile.r + dr
                if (neighbor_q, neighbor_r) not in tile_map:
                    coastal_edge_count += 1

        print(f"  Coastal edges available: {coastal_edge_count}")
        print(f"  Ports should only be on coastal edges (where land meets ocean)\n")

        # Create 4 players - EXACT same as main.py
        players = [
            Player("Player 1", (255, 50, 50)),
            Player("Player 2", (50, 50, 255)),
            Player("Player 3", (255, 255, 50)),
            Player("Player 4", (255, 255, 255))
        ]

        # DO NOT give starting resources - players start with 0 resources
        # Resources are earned by placing settlements and rolling dice

        # Create game system - EXACT same as main.py
        self.game_system = GameSystem(self.game_board, players)
        self.game_system.robber = robber

        # DO NOT skip initial placement - game should start with settlement placement
        # GameSystem defaults to "INITIAL_PLACEMENT_1" phase which is correct

        print("✓ Game initialized!")
        print(f"  {len(tiles)} tiles created")
        print(f"  {len(self.game_board.ports)} ports generated")
        print(f"  4 players ready")
        print(f"  Game phase: {self.game_system.game_phase}")
        print(f"  Turn phase: {self.game_system.turn_phase}")

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
                'last_roll': self.game_system.last_dice_roll,
                'game_phase': self.game_system.game_phase,
                'turn_phase': self.game_system.turn_phase,
                'waiting_for_road': self.game_system.waiting_for_road,  # For initial placement

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

                # Vertices (for rendering)
                'vertices': [
                    {
                        'x': v.x,
                        'y': v.y
                    }
                    for v in self.game_board.vertices
                ],

                # Edges (for rendering)
                'edges': [
                    {
                        'x1': e.vertex1.x,
                        'y1': e.vertex1.y,
                        'x2': e.vertex2.x,
                        'y2': e.vertex2.y
                    }
                    for e in self.game_board.edges
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
        """Process an action from a client - EXACT logic from main.py"""
        with self.state_lock:
            player = self.game_system.players[player_index]
            current_player = self.game_system.get_current_player()

            # Only allow actions from current player
            if player != current_player:
                print(f"  [DEBUG] Player {player_index + 1} tried to act, but it's Player {self.game_system.players.index(current_player) + 1}'s turn")
                return

            print(f"  [DEBUG] Processing action '{action}' from Player {player_index + 1}")

            # Parse action (format: "ACTION:x,y" or just "ACTION")
            parts = action.split(':')
            action_type = parts[0]
            coords = parts[1] if len(parts) > 1 else None

            if action_type == "ROLL_DICE":
                can_roll = self.game_system.can_roll_dice()
                print(f"  [DEBUG] can_roll_dice() = {can_roll}, dice_rolled = {self.game_system.dice_rolled}")
                if can_roll:
                    success, message = self.game_system.roll_dice_action()
                    if success:
                        roll = self.game_system.last_dice_roll
                        print(f"  ✓ Player {player_index + 1} rolled: {roll[2]} ({roll[0]}+{roll[1]})")
                    else:
                        print(f"  ✗ roll_dice_action() failed: {message}")
                else:
                    print(f"  ✗ Cannot roll dice (already rolled or wrong phase)")

            elif action_type == "END_TURN":
                success, message = self.game_system.end_turn()
                print(f"  [DEBUG] end_turn() = {success}, message = '{message}'")
                if success:
                    new_player = self.game_system.get_current_player()
                    new_index = self.game_system.players.index(new_player)
                    print(f"  ✓ Turn ended. Now: Player {new_index + 1}'s turn")
                else:
                    print(f"  ✗ Could not end turn: {message}")

            elif action_type == "PLACE_SETTLEMENT" and coords:
                try:
                    x, y = map(float, coords.split(','))
                    vertex = self.find_vertex(x, y)
                    if vertex:
                        if self.game_system.is_initial_placement_phase():
                            success, message = self.game_system.try_place_initial_settlement(vertex, player)
                        else:
                            success, message = self.game_system.try_build_settlement(vertex, player)
                        print(f"  {'✓' if success else '✗'} Place settlement: {message}")
                    else:
                        print(f"  ✗ Vertex not found at ({x}, {y})")
                except Exception as e:
                    print(f"  ✗ Error placing settlement: {e}")

            elif action_type == "PLACE_ROAD" and coords:
                try:
                    coords_parts = coords.split(',')
                    x1, y1, x2, y2 = map(float, coords_parts)
                    edge = self.find_edge(x1, y1, x2, y2)
                    if edge:
                        if self.game_system.is_initial_placement_phase():
                            success, message = self.game_system.try_place_initial_road(edge, player)
                        else:
                            success, message = self.game_system.try_build_road(edge, player)
                        print(f"  {'✓' if success else '✗'} Place road: {message}")
                    else:
                        print(f"  ✗ Edge not found")
                except Exception as e:
                    print(f"  ✗ Error placing road: {e}")

            elif action_type == "PLACE_CITY" and coords:
                try:
                    x, y = map(float, coords.split(','))
                    vertex = self.find_vertex(x, y)
                    if vertex:
                        success, message = self.game_system.try_build_city(vertex, player)
                        print(f"  {'✓' if success else '✗'} Place city: {message}")
                    else:
                        print(f"  ✗ Vertex not found at ({x}, {y})")
                except Exception as e:
                    print(f"  ✗ Error placing city: {e}")

    def find_vertex(self, x, y, tolerance=1.0):
        """Find vertex by coordinates with tolerance"""
        for vertex in self.game_board.vertices:
            if abs(vertex.x - x) < tolerance and abs(vertex.y - y) < tolerance:
                return vertex
        return None

    def find_edge(self, x1, y1, x2, y2, tolerance=1.0):
        """Find edge by endpoint coordinates with tolerance"""
        for edge in self.game_board.edges:
            # Check both orderings
            if ((abs(edge.vertex1.x - x1) < tolerance and abs(edge.vertex1.y - y1) < tolerance and
                 abs(edge.vertex2.x - x2) < tolerance and abs(edge.vertex2.y - y2) < tolerance) or
                (abs(edge.vertex1.x - x2) < tolerance and abs(edge.vertex1.y - y2) < tolerance and
                 abs(edge.vertex2.x - x1) < tolerance and abs(edge.vertex2.y - y1) < tolerance)):
                return edge
        return None

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
