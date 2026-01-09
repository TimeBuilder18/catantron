import random
import math
import pygame
from enum import Enum

# ==================== ENUMS AND CONSTANTS ====================

class ResourceType(Enum):
    WOOD = "forest"
    BRICK = "hill"
    WHEAT = "field"
    ORE = "mountain"
    SHEEP = "pasture"


class DevelopmentCardType(Enum):
    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"


class PortType(Enum):
    """Types of trading ports"""
    GENERIC = "3:1"  # Trade any 3 of one resource for 1 of another
    WOOD = "wood_2:1"  # Trade 2 wood for 1 of any resource
    BRICK = "brick_2:1"
    WHEAT = "wheat_2:1"
    SHEEP = "sheep_2:1"
    ORE = "ore_2:1"


class GameConstants:
    """Game constants for Catan"""
    # Victory conditions
    VICTORY_POINTS_TO_WIN = 10 # 10
    LARGEST_ARMY_MIN_KNIGHTS = 3
    LONGEST_ROAD_MIN_LENGTH = 5

    # Player limits
    MAX_SETTLEMENTS = 5
    MAX_CITIES = 4
    MAX_ROADS = 15

    # Resource limits
    ROBBER_DISCARD_LIMIT = 7


# ==================== STRUCTURES ====================

class Structure:
    def __init__(self, player, position):
        self.player = player
        self.position = position


class Settlement(Structure):
    def __init__(self, player, vertex_position):
        super().__init__(player, vertex_position)
        self.is_port = False
        self.port_type = None

    @staticmethod
    def get_cost():
        return {
            ResourceType.WOOD: 1,
            ResourceType.BRICK: 1,
            ResourceType.WHEAT: 1,
            ResourceType.SHEEP: 1
        }


class City(Structure):
    def __init__(self, player, vertex_position):
        super().__init__(player, vertex_position)

    @staticmethod
    def get_cost():
        return {
            ResourceType.WHEAT: 2,
            ResourceType.ORE: 3
        }


class Road(Structure):
    def __init__(self, player, edge_position):
        super().__init__(player, edge_position)

    @staticmethod
    def get_cost():
        return {
            ResourceType.WOOD: 1,
            ResourceType.BRICK: 1
        }


# ==================== PORTS ====================

class Port:
    """Represents a trading port on the board"""

    def __init__(self, port_type, vertex1, vertex2):
        self.port_type = port_type  # PortType enum
        self.vertex1 = vertex1  # First vertex position
        self.vertex2 = vertex2  # Second vertex position

    def get_trade_ratio(self, resource_type=None):
        """Get the trade ratio for this port"""
        if self.port_type == PortType.GENERIC:
            return 3  # 3:1 for any resource
        else:
            # Check if this is the specialized resource for this port
            port_resource = self._get_port_resource()
            if resource_type == port_resource:
                return 2  # 2:1 for specialized resource
            else:
                return 4  # Regular 4:1 if not the right resource

    def _get_port_resource(self):
        """Get the resource type this port specializes in"""
        port_map = {
            PortType.WOOD: ResourceType.WOOD,
            PortType.BRICK: ResourceType.BRICK,
            PortType.WHEAT: ResourceType.WHEAT,
            PortType.SHEEP: ResourceType.SHEEP,
            PortType.ORE: ResourceType.ORE
        }
        return port_map.get(self.port_type, None)

    def can_player_use(self, player):
        """Check if player has a settlement/city on this port"""
        for vertex in [self.vertex1, self.vertex2]:
            if vertex and vertex.structure and vertex.structure.player == player:
                return True
        return False


# ==================== TRADE NEGOTIATION ====================

class TradeOfferStatus(Enum):
    """Status of a trade offer"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"
    EXPIRED = "expired"


class TradeOffer:
    """Represents a trade offer between players"""

    def __init__(self, offering_player, target_player, offered_resources, requested_resources):
        self.offering_player = offering_player
        self.target_player = target_player
        self.offered_resources = offered_resources.copy()  # Dict of ResourceType -> amount
        self.requested_resources = requested_resources.copy()  # Dict of ResourceType -> amount
        self.status = TradeOfferStatus.PENDING
        self.counter_offer = None  # Can store a counter-offer TradeOffer

    def get_offer_summary(self):
        """Get a human-readable summary of the offer"""
        offered = ", ".join([f"{amt} {res.name.lower()}" for res, amt in self.offered_resources.items() if amt > 0])
        requested = ", ".join([f"{amt} {res.name.lower()}" for res, amt in self.requested_resources.items() if amt > 0])
        return f"{self.offering_player.name} offers {offered} for {requested}"

    def is_valid(self):
        """Check if both players still have the resources for this trade"""
        # Check offering player has offered resources
        for resource_type, amount in self.offered_resources.items():
            if amount > 0 and self.offering_player.resources[resource_type] < amount:
                return False
        # Check target player has requested resources
        for resource_type, amount in self.requested_resources.items():
            if amount > 0 and self.target_player.resources[resource_type] < amount:
                return False
        return True


# ==================== DEVELOPMENT CARDS ====================

class DevelopmentCard:
    def __init__(self, card_type):
        self.card_type = card_type
        self.played = False


class DevelopmentCardDeck:
    def __init__(self):
        self.cards = []
        self.initialize_deck()

    def initialize_deck(self):
        """Create the standard development card deck"""
        # 14 Knight cards
        for _ in range(14):
            self.cards.append(DevelopmentCard(DevelopmentCardType.KNIGHT))

        # 5 Victory Point cards
        for _ in range(5):
            self.cards.append(DevelopmentCard(DevelopmentCardType.VICTORY_POINT))

        # 2 Road Building cards
        for _ in range(2):
            self.cards.append(DevelopmentCard(DevelopmentCardType.ROAD_BUILDING))

        # 2 Year of Plenty cards
        for _ in range(2):
            self.cards.append(DevelopmentCard(DevelopmentCardType.YEAR_OF_PLENTY))

        # 2 Monopoly cards
        for _ in range(2):
            self.cards.append(DevelopmentCard(DevelopmentCardType.MONOPOLY))

        random.shuffle(self.cards)

    def draw_card(self):
        """Draw a card from the deck"""
        if self.cards:
            return self.cards.pop()
        return None

    def is_empty(self):
        return len(self.cards) == 0

    @staticmethod
    def get_cost():
        """Get the cost to buy a development card"""
        return {
            ResourceType.WHEAT: 1,
            ResourceType.SHEEP: 1,
            ResourceType.ORE: 1
        }


# ==================== ROBBER ====================

class Robber:
    def __init__(self):
        self.position = None

    def move_to_tile(self, tile):
        """Move robber to a new tile"""
        if self.position:
            self.position.has_robber = False

        self.position = tile
        tile.has_robber = True

    def get_affected_players(self):
        """Get players who have structures adjacent to the robber"""
        if not self.position:
            return []

        affected_players = set()
        # Implementation would depend on vertex system
        return list(affected_players)


# ==================== PLAYER ====================

class Player:
    def __init__(self, name, color):
        self.name = name
        self.color = color

        # Resources (hand)
        self.resources = {
            ResourceType.WOOD: 0,
            ResourceType.BRICK: 0,
            ResourceType.WHEAT: 0,
            ResourceType.ORE: 0,
            ResourceType.SHEEP: 0
        }

        # Development cards
        self.development_cards = {
            DevelopmentCardType.KNIGHT: 0,
            DevelopmentCardType.VICTORY_POINT: 0,
            DevelopmentCardType.ROAD_BUILDING: 0,
            DevelopmentCardType.YEAR_OF_PLENTY: 0,
            DevelopmentCardType.MONOPOLY: 0
        }

        # Structures built
        self.settlements = []
        self.cities = []
        self.roads = []

        # Game stats
        self.knights_played = 0
        self.has_largest_army = False
        self.has_longest_road = False
        self.victory_points = 0

    def add_resource(self, resource_type, amount=1):
        """Add resources to player's hand"""
        self.resources[resource_type] += amount

    def remove_resource(self, resource_type, amount=1):
        """Remove resources from player's hand"""
        if self.resources[resource_type] >= amount:
            self.resources[resource_type] -= amount
            return True
        return False

    def get_total_resources(self):
        """Get total number of resource cards"""
        return sum(self.resources.values())

    def can_afford(self, cost):
        """Check if player can afford a cost (dict of resource types and amounts)"""
        for resource_type, amount in cost.items():
            if self.resources[resource_type] < amount:
                return False
        return True

    def pay_cost(self, cost):
        """Pay a cost if possible"""
        if self.can_afford(cost):
            for resource_type, amount in cost.items():
                self.remove_resource(resource_type, amount)
            return True
        return False

    def calculate_victory_points(self):
        """Calculate current victory points with detailed breakdown"""
        points = 0
        breakdown = {}

        # Settlements (1 point each)
        settlement_points = len(self.settlements)
        if settlement_points > 0:
            breakdown['Settlements'] = settlement_points
        points += settlement_points

        # Cities (2 points each)
        city_points = len(self.cities) * 2
        if city_points > 0:
            breakdown['Cities'] = city_points
        points += city_points

        # Victory point development cards
        vp_card_points = self.development_cards[DevelopmentCardType.VICTORY_POINT]
        if vp_card_points > 0:
            breakdown['VP Cards'] = vp_card_points
        points += vp_card_points

        # Largest army (2 points)
        if self.has_largest_army:
            breakdown['Largest Army'] = 2
            points += 2

        # Longest road (2 points)
        if self.has_longest_road:
            breakdown['Longest Road'] = 2
            points += 2

        self.victory_points = points
        self.vp_breakdown = breakdown
        return points

    def get_victory_point_breakdown(self):
        """Get detailed breakdown of victory points"""
        self.calculate_victory_points()
        return self.vp_breakdown if hasattr(self, 'vp_breakdown') else {}

    # ==================== BUILDING ACTIONS ====================

    def try_build_settlement(self, vertex, ignore_road_rule=False):
        """Attempt to build a settlement"""
        cost = Settlement.get_cost()

        # Check resources with detailed message
        if not self.can_afford(cost):
            missing = []
            for resource, amount in cost.items():
                if self.resources[resource] < amount:
                    missing.append(f"{resource.name.lower()}")
            return False, f"Need: 1 wood + 1 brick + 1 wheat + 1 sheep (Missing: {', '.join(missing)})"

        if not vertex.can_build_settlement(self, ignore_road_rule):
            return False, "Cannot build settlement (too close or no road connection)"

        self.pay_cost(cost)
        vertex.build_settlement(self)
        return True, "Settlement built"

    def try_build_city(self, vertex):
        """Attempt to build a city"""
        cost = City.get_cost()

        # Check resources with detailed message
        if not self.can_afford(cost):
            missing = []
            for resource, amount in cost.items():
                if self.resources[resource] < amount:
                    missing.append(f"{resource.name.lower()}")
            return False, f"Need: 2 wheat + 3 ore (Missing: {', '.join(missing)})"

        if not vertex.can_build_city(self):
            return False, "Cannot upgrade (must have a settlement here)"

        self.pay_cost(cost)
        vertex.build_city(self)
        return True, "City built"

    def try_build_road(self, edge):
        """Attempt to build a road"""
        cost = Road.get_cost()

        # Check resources with detailed message
        if not self.can_afford(cost):
            missing = []
            for resource, amount in cost.items():
                if self.resources[resource] < amount:
                    missing.append(f"{resource.name.lower()}")
            return False, f"Need: 1 wood + 1 brick (Missing: {', '.join(missing)})"

        if not edge.can_build_road(self):
            return False, "Road must connect to your existing road/settlement"

        self.pay_cost(cost)
        edge.build_road(self)
        return True, "Road built"

    def try_buy_development_card(self, deck):
        """Attempt to buy a development card"""
        if not self.can_afford(DevelopmentCardDeck.get_cost()):
            return False, "Not enough resources"

        if deck.is_empty():
            return False, "No development cards left"

        self.pay_cost(DevelopmentCardDeck.get_cost())
        card = deck.draw_card()
        if card:
            self.development_cards[card.card_type] += 1
            return True, f"Bought {card.card_type.value.replace('_', ' ').title()}"

        return False, "Failed to draw card"


# ==================== BOARD ELEMENTS ====================

class Vertex:
    """Represents a vertex where settlements/cities can be built"""

    def __init__(self, x, y, adjacent_tiles=None):
        self.x = x
        self.y = y
        self.adjacent_tiles = adjacent_tiles or []
        self.structure = None
        self.adjacent_vertices = []
        self.connected_edges = []

    def can_build_settlement(self, player, ignore_road_rule=False):
        """Check if player can build a settlement here"""
        if self.structure is not None:
            return False

        # Distance rule: no settlements within 2 edges
        for adj_vertex in self.adjacent_vertices:
            if adj_vertex.structure is not None:
                return False

        # Must have a connected road (except during initial placement)
        if not ignore_road_rule and not self.has_connected_road(player):
            return False

        return True

    def can_build_city(self, player):
        """Check if player can build a city here"""
        return (self.structure is not None and
                isinstance(self.structure, Settlement) and
                self.structure.player == player)

    def has_connected_road(self, player):
        """Check if this vertex has a road connected to it owned by player"""
        for edge in self.connected_edges:
            if edge.structure and edge.structure.player == player:
                return True
        return False

    def build_settlement(self, player):
        """Build a settlement at this vertex"""
        self.structure = Settlement(player, self)
        player.settlements.append(self.structure)
        return True

    def build_city(self, player):
        """Upgrade settlement to city"""
        # Remove old settlement from player's list
        player.settlements.remove(self.structure)
        # Replace with city
        self.structure = City(player, self)
        player.cities.append(self.structure)
        return True


class Edge:
    """Represents an edge where roads can be built"""

    def __init__(self, vertex1, vertex2):
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.structure = None
        self.x = (vertex1.x + vertex2.x) / 2
        self.y = (vertex1.y + vertex2.y) / 2

        # Connect this edge to its vertices
        vertex1.connected_edges.append(self)
        vertex2.connected_edges.append(self)

    def can_build_road(self, player):
        """Check if player can build a road here"""
        if self.structure is not None:
            return False

        return self.connects_to_player_infrastructure(player)

    def connects_to_player_infrastructure(self, player):
        """Check if this edge connects to player's existing roads or settlements"""
        for vertex in [self.vertex1, self.vertex2]:
            # Check if vertex has player's structure
            if (vertex.structure and vertex.structure.player == player):
                return True

            # Check if vertex has player's roads
            for edge in vertex.connected_edges:
                if (edge != self and edge.structure and
                        edge.structure.player == player):
                    return True
        return False

    def build_road(self, player):
        """Build a road on this edge"""
        self.structure = Road(player, self)
        player.roads.append(self.structure)
        return True


# ==================== GAME BOARD ====================

class GameBoard:
    """Game board with tiles, vertices, and edges"""

    def __init__(self, tiles):
        self.tiles = tiles
        self.vertices = []
        self.edges = []
        self.ports = []
        self.generate_vertices_and_edges()
        self.generate_ports()

    def generate_vertices_and_edges(self):
        """Generate vertices and edges based on tile positions"""
        vertex_dict = {}

        # Generate vertices for each tile
        for tile in self.tiles:
            corners = tile.get_corners()
            for i, (x, y) in enumerate(corners):
                x, y = round(x, 1), round(y, 1)

                if (x, y) not in vertex_dict:
                    vertex = Vertex(x, y)
                    vertex_dict[(x, y)] = vertex
                    self.vertices.append(vertex)

                vertex_dict[(x, y)].adjacent_tiles.append(tile)

        # Generate edges between adjacent vertices
        edge_set = set()

        for tile in self.tiles:
            corners = tile.get_corners()
            for i in range(len(corners)):
                x1, y1 = round(corners[i][0], 1), round(corners[i][1], 1)
                x2, y2 = round(corners[(i + 1) % 6][0], 1), round(corners[(i + 1) % 6][1], 1)

                edge_key = tuple(sorted([(x1, y1), (x2, y2)]))

                if edge_key not in edge_set:
                    vertex1 = vertex_dict[(x1, y1)]
                    vertex2 = vertex_dict[(x2, y2)]
                    edge = Edge(vertex1, vertex2)
                    self.edges.append(edge)
                    edge_set.add(edge_key)

        # Set up adjacent vertices (for distance rule)
        for edge in self.edges:
            edge.vertex1.adjacent_vertices.append(edge.vertex2)
            edge.vertex2.adjacent_vertices.append(edge.vertex1)


    def generate_ports(self):
        """Generate 9 trading ports on COASTAL edges - ALGORITHMIC VERSION FOR FLAT-TOP HEXAGONS"""

        ###print("\n" + "="*70)
        ####print("🚢 PORT PLACEMENT SYSTEM - FOR FLAT-TOP HEXAGONS")
        ###print("="*70)

        # Hex neighbor offsets in axial coordinates for FLAT-TOP hexagons
        # Direction 0=E, 1=SE, 2=SW, 3=W, 4=NW, 5=NE (based on which neighbor hex)
        HEX_DIRECTIONS = [
            (+1, 0),   # East - right neighbor
            (+0, +1),  # Southeast - lower-right neighbor
            (-1, +1),  # Southwest - lower-left neighbor
            (-1, 0),   # West - left neighbor
            (0, -1),   # Northwest - upper-left neighbor
            (+1, -1)   # Northeast - upper-right neighbor
        ]

        # For FLAT-TOP hexagons with corners: 0=top, 1=upper-right, 2=lower-right, 3=bottom, 4=lower-left, 5=upper-left
        # Map direction (which neighbor is missing) to which edge faces that direction
        # If East neighbor is missing, the edge between corners 1-2 faces East
        DIR_TO_CORNER = [1, 2, 3, 4, 5, 0]  # Maps direction index to starting corner of the edge

        # Create lookup map of tile positions
        tile_map = {(tile.q, tile.r): tile for tile in self.tiles}

        # Step 1: Calculate board radius
        max_distance = 0
        for tile in self.tiles:
            distance = max(abs(tile.q), abs(tile.r), abs(tile.q + tile.r))
            max_distance = max(max_distance, distance)

        ##print(f"\n📊 Board Analysis:")
        ##print(f"   • Total tiles: {len(self.tiles)}")
        ##print(f"   • Board radius: {max_distance}")

        # Step 2: Identify outer ring tiles
        outer_ring_tiles = []
        for tile in self.tiles:
            distance = max(abs(tile.q), abs(tile.r), abs(tile.q + tile.r))
            if distance == max_distance:
                outer_ring_tiles.append(tile)

        ##print(f"   • Outer ring tiles: {len(outer_ring_tiles)}")

        # Step 3: Find coastal edges (edges facing outward from outer ring)
        coastal_edges = []

        for tile in outer_ring_tiles:
            # Check each of the 6 possible neighbor directions
            for dir_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
                neighbor_q = tile.q + dq
                neighbor_r = tile.r + dr

                # If NO hex exists in this direction, this edge faces the ocean
                if (neighbor_q, neighbor_r) not in tile_map:
                    # Get the two vertices for this edge
                    corners = tile.get_corners()
                    corner_idx = DIR_TO_CORNER[dir_idx]
                    v1_x, v1_y = round(corners[corner_idx][0], 1), round(corners[corner_idx][1], 1)
                    v2_x, v2_y = round(corners[(corner_idx + 1) % 6][0], 1), round(corners[(corner_idx + 1) % 6][1], 1)

                    # Find the edge object matching these vertices
                    for edge in self.edges:
                        e1_x, e1_y = round(edge.vertex1.x, 1), round(edge.vertex1.y, 1)
                        e2_x, e2_y = round(edge.vertex2.x, 1), round(edge.vertex2.y, 1)

                        if ((e1_x, e1_y) == (v1_x, v1_y) and (e2_x, e2_y) == (v2_x, v2_y)) or \
                           ((e1_x, e1_y) == (v2_x, v2_y) and (e2_x, e2_y) == (v1_x, v1_y)):
                            coastal_edges.append((edge, tile, dir_idx))
                            break

        ###print(f"\n🏖️  Coastal Edge Detection:")
        ##print(f"   • Coastal edges found: {len(coastal_edges)}")

        if len(coastal_edges) < 9:
            ##print(f"\n ERROR: Only {len(coastal_edges)} coastal edges found!")
            return

        # Step 4: Sort by angle for even distribution
        def get_angle(edge_tuple):
            edge = edge_tuple[0]
            mid_x = (edge.vertex1.x + edge.vertex2.x) / 2
            mid_y = (edge.vertex1.y + edge.vertex2.y) / 2
            return math.atan2(mid_y, mid_x)

        coastal_edges.sort(key=get_angle)

        # Step 5: Select 9 evenly distributed positions
        num_coastal = len(coastal_edges)
        step = num_coastal / 9.0
        port_indices = [int(i * step) for i in range(9)]
        selected_coastal_edges = [coastal_edges[i] for i in port_indices]

        # Step 6: Create port types
        port_types = [
            PortType.GENERIC, PortType.GENERIC, PortType.GENERIC, PortType.GENERIC,
            PortType.WOOD, PortType.BRICK, PortType.WHEAT, PortType.SHEEP, PortType.ORE
        ]
        random.shuffle(port_types)

        # Step 7: Create and place ports
        ###print(f"\n🚢 Port Placement Details:")
        ##print(f"   {'#':<4} {'Type':<15} {'Hex':<12} {'Distance':<10}")
        ##print(f"   {'-'*4} {'-'*15} {'-'*12} {'-'*10}")

        for i, (edge, tile, direction) in enumerate(selected_coastal_edges):
            port = Port(port_types[i], edge.vertex1, edge.vertex2)
            self.ports.append(port)

            tile_dist = max(abs(tile.q), abs(tile.r), abs(tile.q + tile.r))
            ##print(f"   {i+1:<4} {port_types[i].value:<15} ({tile.q:2d},{tile.r:2d}){'':<5} {tile_dist:<10}")

        ##print(f"\n Port Placement Complete!")
        ##print(f"   • {len(self.ports)} ports successfully placed on OUTER PERIMETER")
        ###print("="*70 + "\n")

    def get_player_ports(self, player):
        """Get all ports a player has access to"""
        return [port for port in self.ports if port.can_player_use(player)]

    def get_best_trade_ratio(self, player, resource_type):
        """Get the best trade ratio for a player for a specific resource"""
        player_ports = self.get_player_ports(player)

        if not player_ports:
            return 4  # Default 4:1 with no ports

        best_ratio = 4
        for port in player_ports:
            ratio = port.get_trade_ratio(resource_type)
            if ratio < best_ratio:
                best_ratio = ratio

        return best_ratio


# ==================== DICE AND RESOURCE DISTRIBUTION ====================

class DiceRoller:
    """Handles dice rolling and resource distribution"""

    @staticmethod
    def roll_dice():
        """Roll dice only when explicitly called by user during normal play"""
        import inspect

        # Get the calling context
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            if caller_frame:
                # Check if we're being called during game setup/initial placement
                caller_locals = caller_frame.f_locals

                # If 'self' exists and it's a GameSystem, check the phase
                if 'self' in caller_locals and hasattr(caller_locals['self'], 'game_phase'):
                    game_system = caller_locals['self']
                    if hasattr(game_system, 'game_phase') and game_system.game_phase != "NORMAL_PLAY":
                        return 1, 1, 2  # Return fake result to prevent crash during initial placement
        finally:
            del frame

        # Only roll real dice if we get here
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        return die1, die2, die1 + die2

    @staticmethod
    def distribute_resources(dice_sum, game_board, players):
        """Distribute resources to players based on dice roll (NO AUTO-ROLLING)"""
        if dice_sum == 7:
            return {}  # Robber moves, no resources distributed

        resource_gains = {player: {} for player in players}

        # Find all tiles with the rolled number
        for tile in game_board.tiles:
            if tile.number == dice_sum and tile.produces_resources():
                resource_type = tile.get_resource_type()
                if resource_type:
                    # Find all vertices adjacent to this tile
                    for vertex in game_board.vertices:
                        if tile in vertex.adjacent_tiles and vertex.structure:
                            player = vertex.structure.player

                            # Calculate resource amount based on structure type
                            if isinstance(vertex.structure, Settlement):
                                amount = 1
                            elif isinstance(vertex.structure, City):
                                amount = 2
                            else:
                                continue

                            # Add resources to player
                            player.add_resource(resource_type, amount)

                            # Track for display
                            if resource_type not in resource_gains[player]:
                                resource_gains[player][resource_type] = 0
                            resource_gains[player][resource_type] += amount

        return resource_gains


# ==================== GAME SYSTEM ====================

class GameSystem:
    """Main game system that manages all game state and actions"""

    def __init__(self, game_board, players, victory_points_to_win=10):
        self.game_board = game_board
        self.players = players
        self.victory_points_to_win = victory_points_to_win  # Customizable VP threshold
        self.current_player_index = 0
        self.game_phase = "INITIAL_PLACEMENT_1"  # INITIAL_PLACEMENT_1, INITIAL_PLACEMENT_2, NORMAL_PLAY

        # Turn management
        self.turn_phase = "ROLL_DICE"  # ROLL_DICE, TRADE_BUILD, TURN_COMPLETE
        self.dice_rolled = False
        self.last_dice_roll = None
        self.last_resource_gains = None
        self.turn_number = 1

        # Game objects
        self.robber = Robber()
        self.dev_deck = DevelopmentCardDeck()

        # Initial placement tracking
        self.initial_settlements_placed = 0
        self.initial_roads_placed = 0
        self.player_initial_placements = {player: {"settlements": 0, "roads": 0} for player in players}
        self.waiting_for_road = False
        self.last_settlement_vertex = None

        # Development card effects
        self.free_roads_remaining = 0  # For Road Building card
        self.longest_road_player = None
        self.largest_army_player = None

        # Trade negotiation
        self.pending_trade_offers = []  # List of TradeOffer objects

        # Discard phase (when 7 is rolled)
        self.waiting_for_discards = False
        self.players_must_discard = []  # List of players who need to discard
        self.players_discarded = set()  # Track who has discarded

    def can_roll_dice(self):
        """Check if current player can roll dice"""
        return (self.game_phase == "NORMAL_PLAY" and
                self.turn_phase == "ROLL_DICE" and
                not self.dice_rolled)

    def can_trade_or_build(self):
        """Check if current player can trade or build"""
        return (self.game_phase == "NORMAL_PLAY" and
                self.turn_phase == "TRADE_BUILD")

    def can_end_turn(self):
        """Check if current player can end their turn"""
        return (self.game_phase == "NORMAL_PLAY" and
                (self.turn_phase == "TRADE_BUILD" or self.turn_phase == "TURN_COMPLETE"))

    def get_turn_phase_description(self):
        """Get description of current turn phase"""
        if self.game_phase == "NORMAL_PLAY":
            if self.turn_phase == "ROLL_DICE":
                return "Must roll dice to start turn"
            elif self.turn_phase == "TRADE_BUILD":
                return "Trade resources and/or build"
            elif self.turn_phase == "TURN_COMPLETE":
                return "Turn complete - press T to end turn"
        return self.get_current_player_needs()

    def is_initial_placement_phase(self):
        """Check if we're in initial placement phase"""
        return self.game_phase in ["INITIAL_PLACEMENT_1", "INITIAL_PLACEMENT_2"]

    def get_current_player_needs(self):
        """Get what the current player needs to place"""
        if not self.is_initial_placement_phase():
            return "Normal play"

        current_player = self.get_current_player()
        placements = self.player_initial_placements[current_player]

        if self.waiting_for_road:
            return "Must place ROAD connected to last settlement"
        elif placements["settlements"] < (1 if self.game_phase == "INITIAL_PLACEMENT_1" else 2):
            return "Must place SETTLEMENT"
        elif placements["roads"] < placements["settlements"]:
            return "Must place ROAD connected to settlement"
        else:
            return "Ready for next player"

    def get_current_player(self):
        return self.players[self.current_player_index]

    def next_player(self):
        """Legacy method - use end_turn() instead"""
        return self.end_turn()

    def give_starting_resources_for_player(self, player):
        """Gives a player their starting resources based on their second settlement."""
        if len(player.settlements) >= 2:
            second_settlement = player.settlements[1]
            vertex = second_settlement.position
            for tile in vertex.adjacent_tiles:
                if tile.resource != "desert":
                    resource_type = tile.get_resource_type()
                    if resource_type:
                        player.add_resource(resource_type, 1)

    def roll_dice(self):
        """Roll dice and distribute resources"""
        if not self.can_roll_dice():
            return None

        die1, die2, total = DiceRoller.roll_dice()
        self.last_dice_roll = (die1, die2, total)
        self.dice_rolled = True
        self.turn_phase = "TRADE_BUILD"

        if total == 7:
            # When 7 is rolled, players with 8+ cards must discard half
            self.players_must_discard = self.get_players_who_must_discard()
            if self.players_must_discard:
                self.waiting_for_discards = True
                # Game pauses until all discards are done, then robber moves
            return (die1, die2, total)
        else:
            # Distribute resources
            self.last_resource_gains = DiceRoller.distribute_resources(
                total, self.game_board, self.players
            )
            return (die1, die2, total)

    def end_turn(self):
        """End current player's turn and move to next player"""
        if self.game_phase == "NORMAL_PLAY":
            if not self.can_end_turn():
                return False, "Cannot end turn yet - must roll dice first"

            # Reset turn state for next player
            self.dice_rolled = False
            self.turn_phase = "ROLL_DICE"
            self.last_dice_roll = None
            self.last_resource_gains = None

            # Clear any pending trade offers
            self.clear_expired_offers()

            # Move to next player
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            self.turn_number += 1

            ##print(f"\n=== TURN {self.turn_number} ===")
            ##print(f"{self.get_current_player().name}'s turn")
            ###print("Press 'D' to roll dice and start your turn")

            return True, f"Turn ended. Now {self.get_current_player().name}'s turn"
        else:
            # Initial placement logic
            return self.advance_initial_placement()

    def advance_initial_placement(self):
        """Handle initial placement turn advancement"""
        current_player = self.get_current_player()
        placements = self.player_initial_placements[current_player]

        expected_settlements = 1 if self.game_phase == "INITIAL_PLACEMENT_1" else 2

        if placements["settlements"] >= expected_settlements and placements["roads"] >= expected_settlements:
            # Player completed their placements
            if self.game_phase == "INITIAL_PLACEMENT_1":
                # In first round, go to next player normally
                self.current_player_index = (self.current_player_index + 1) % len(self.players)

                # Check if all players finished first round
                if all(self.player_initial_placements[p]["settlements"] >= 1 and
                       self.player_initial_placements[p]["roads"] >= 1 for p in self.players):
                    self.game_phase = "INITIAL_PLACEMENT_2"
                    # In second round, go in reverse order (last player goes first)
                    self.current_player_index = len(self.players) - 1
                    self.waiting_for_road = False  # Reset waiting_for_road for new round
                    self.give_starting_resources_for_player(self.get_current_player())


            elif self.game_phase == "INITIAL_PLACEMENT_2":
                # In second round, go in reverse order
                self.current_player_index -= 1
                if self.current_player_index >= 0:
                    self.give_starting_resources_for_player(self.get_current_player())


                if self.current_player_index < 0:

                    self.game_phase = "NORMAL_PLAY"
                    self.turn_phase = "ROLL_DICE"
                    self.current_player_index = 0
                    self.turn_number = 1
                    self.dice_rolled = False
                    self.last_dice_roll = None
                    self.last_resource_gains = None

            return True, f"{self.get_current_player().name}'s turn"

        return False, f"{current_player.name} must complete placement first"

    def clear_expired_offers(self):
        """Remove expired trade offers"""
        # Simple implementation - remove all pending offers
        self.pending_trade_offers = []

    # ==================== ROBBER & DISCARD SYSTEM ====================

    def get_players_who_must_discard(self):
        """Get list of players with 8+ cards who must discard"""
        players_to_discard = []
        for player in self.players:
            if player.get_total_resources() >= 8:
                players_to_discard.append(player)
        return players_to_discard

    def player_must_discard(self, player):
        """Check if a specific player must discard"""
        return (self.waiting_for_discards and
                player in self.players_must_discard and
                player not in self.players_discarded)

    def discard_cards(self, player, cards_to_discard):
        """
        Player discards cards when 7 is rolled

        Args:
            player: The player discarding
            cards_to_discard: Dict of {ResourceType: amount} to discard

        Returns:
            (success, message)
        """
        if not self.player_must_discard(player):
            return False, "You don't need to discard"

        total_resources = player.get_total_resources()
        num_to_discard = total_resources // 2  # Half, rounded down

        # Count total cards being discarded
        total_discarding = sum(cards_to_discard.values())

        if total_discarding != num_to_discard:
            return False, f"Must discard exactly {num_to_discard} cards (have {total_resources})"

        # Check player has all the cards they want to discard
        for resource_type, amount in cards_to_discard.items():
            if amount > 0:
                if player.resources[resource_type] < amount:
                    return False, f"Don't have {amount} {resource_type.value}"

        # Execute discard
        for resource_type, amount in cards_to_discard.items():
            if amount > 0:
                player.remove_resource(resource_type, amount)

        # Mark player as having discarded
        self.players_discarded.add(player)

        # Check if all players have discarded
        if len(self.players_discarded) == len(self.players_must_discard):
            # All discards complete - ready to move robber
            self.waiting_for_discards = False
            self.players_must_discard = []
            self.players_discarded = set()

        return True, f"Discarded {total_discarding} cards"

    def all_players_discarded(self):
        """Check if all required players have discarded"""
        return not self.waiting_for_discards

    # ==================== BUILDING ACTIONS ====================

    def try_build_settlement(self, vertex, player=None):
        """Try to build a settlement"""
        if player is None:
            player = self.get_current_player()

        if self.is_initial_placement_phase():
            return False, "Use initial placement method during setup"

        if not self.can_trade_or_build():
            return False, "Cannot build now - roll dice first"

        return player.try_build_settlement(vertex)

    def try_build_city(self, vertex, player=None):
        """Try to build a city"""
        if player is None:
            player = self.get_current_player()

        if self.is_initial_placement_phase():
            return False, "Cannot build cities during initial placement"

        if not self.can_trade_or_build():
            return False, "Cannot build now - roll dice first"

        return player.try_build_city(vertex)

    def try_build_road(self, edge, player=None):
        """Try to build a road"""
        if player is None:
            player = self.get_current_player()

        if self.is_initial_placement_phase():
            return False, "Use initial placement method during setup"

        if not self.can_trade_or_build():
            return False, "Cannot build now - roll dice first"

        return player.try_build_road(edge)

    # ==================== DEVELOPMENT CARDS ====================

    def try_buy_development_card(self, player=None):
        """Try to buy a development card"""
        if player is None:
            player = self.get_current_player()

        if not self.can_trade_or_build():
            return False, "Cannot buy cards now"

        return player.try_buy_development_card(self.dev_deck)

    def can_play_development_card(self, player, card_type):
        """Check if player can play a development card"""
        if player.development_cards[card_type] <= 0:
            return False, "You don't have this card"

        if not self.can_trade_or_build():
            return False, "Can only play cards during your turn"

        return True, "Can play card"

    def play_knight_card(self, player):
        """Play a Knight card"""
        success, message = self.can_play_development_card(player, DevelopmentCardType.KNIGHT)
        if not success:
            return False, message

        player.development_cards[DevelopmentCardType.KNIGHT] -= 1
        player.knights_played += 1
        self.update_largest_army()

        return True, "Knight played - move the robber and steal from a player"

    def play_year_of_plenty_card(self, player, resource1, resource2):
        """Play Year of Plenty card - take 2 resources"""
        success, message = self.can_play_development_card(player, DevelopmentCardType.YEAR_OF_PLENTY)
        if not success:
            return False, message

        player.development_cards[DevelopmentCardType.YEAR_OF_PLENTY] -= 1
        player.add_resource(resource1, 1)
        player.add_resource(resource2, 1)

        return True, f"Year of Plenty: Gained 1 {resource1.value} and 1 {resource2.value}"

    def play_road_building_card(self, player):
        """Play Road Building card - build 2 free roads"""
        success, message = self.can_play_development_card(player, DevelopmentCardType.ROAD_BUILDING)
        if not success:
            return False, message

        player.development_cards[DevelopmentCardType.ROAD_BUILDING] -= 1
        self.free_roads_remaining = 2
        return True, "Road Building: Click 2 edges to build free roads"

    def try_build_free_road(self, edge, player=None):
        """Build a free road from Road Building card"""
        if player is None:
            player = self.get_current_player()

        if self.free_roads_remaining <= 0:
            return False, "No free roads available"

        if edge.structure is not None:
            return False, "Road already exists here"

        if not edge.can_build_road(player):
            return False, "Cannot build road here"

        # Build the road without cost
        road = Road(player, edge)
        edge.structure = road
        player.roads.append(road)
        self.free_roads_remaining -= 1

        return True, f"Free road built! ({self.free_roads_remaining} remaining)"

    def play_monopoly_card(self, player, resource_type):
        """Play Monopoly card - take all of one resource from all players"""
        success, message = self.can_play_development_card(player, DevelopmentCardType.MONOPOLY)
        if not success:
            return False, message

        player.development_cards[DevelopmentCardType.MONOPOLY] -= 1
        total_stolen = 0

        for other_player in self.players:
            if other_player != player:
                amount = other_player.resources[resource_type]
                if amount > 0:
                    other_player.resources[resource_type] = 0
                    player.add_resource(resource_type, amount)
                    total_stolen += amount

        return True, f"Monopoly: Stole {total_stolen} {resource_type.value} from other players"

    # ==================== ROBBER SYSTEM ====================

    def move_robber_to_tile(self, tile):
        """Move robber to a new tile"""
        if self.robber.position:
            self.robber.position.has_robber = False

        self.robber.move_to_tile(tile)
        return True, f"Robber moved to {tile.resource} tile"

    def get_players_on_tile(self, tile):
        """Get players who have structures adjacent to a tile"""
        adjacent_players = set()

        for vertex in self.game_board.vertices:
            if tile in vertex.adjacent_tiles and vertex.structure:
                adjacent_players.add(vertex.structure.player)

        return list(adjacent_players)

    def steal_random_resource(self, stealing_player, target_player):
        """Steal a random resource from target player"""
        available_resources = []
        for resource_type, amount in target_player.resources.items():
            available_resources.extend([resource_type] * amount)

        if not available_resources:
            return False, f"{target_player.name} has no resources to steal"

        stolen_resource = random.choice(available_resources)
        target_player.remove_resource(stolen_resource, 1)
        stealing_player.add_resource(stolen_resource, 1)

        return True, f"Stole 1 resource from {target_player.name}"

    def check_victory_conditions(self):
        """Check if any player has won and return the winner"""
        for player in self.players:
            if player.calculate_victory_points() >= self.victory_points_to_win:
                return player
        return None

    def update_longest_road(self):
        """Calculate and award longest road to the player with the longest continuous road"""
        MIN_LENGTH = GameConstants.LONGEST_ROAD_MIN_LENGTH

        longest_length = 0
        longest_player = None

        for player in self.players:
            road_length = self.calculate_longest_road_for_player(player)
            if road_length >= MIN_LENGTH and road_length > longest_length:
                longest_length = road_length
                longest_player = player

        # Update longest road status
        for player in self.players:
            player.has_longest_road = (player == longest_player)

        return longest_player, longest_length

    def calculate_longest_road_for_player(self, player):
        """Calculate the longest continuous road for a player using DFS"""
        # Build adjacency graph of player's roads
        road_graph = {}
        for road in player.roads:
            v1, v2 = road.position.vertex1, road.position.vertex2

            if v1 not in road_graph:
                road_graph[v1] = []
            if v2 not in road_graph:
                road_graph[v2] = []

            # Only connect if no enemy settlement/city breaks the path
            if not (v1.structure and v1.structure.player != player):
                road_graph[v1].append(v2)
            if not (v2.structure and v2.structure.player != player):
                road_graph[v2].append(v1)

        if not road_graph:
            return 0

        # DFS from each vertex to find longest path
        max_length = 0
        for start_vertex in road_graph.keys():
            length = self._dfs_longest_path(start_vertex, road_graph, set())
            max_length = max(max_length, length)

        return max_length

    def _dfs_longest_path(self, vertex, graph, visited):
        """DFS helper to find longest path from a vertex"""
        visited.add(vertex)
        max_length = 0

        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                length = 1 + self._dfs_longest_path(neighbor, graph, visited)
                max_length = max(max_length, length)

        visited.remove(vertex)
        return max_length

    def update_largest_army(self):
        """Update largest army based on knights played"""
        MIN_KNIGHTS = GameConstants.LARGEST_ARMY_MIN_KNIGHTS

        most_knights = 0
        army_player = None

        for player in self.players:
            if player.knights_played >= MIN_KNIGHTS and player.knights_played > most_knights:
                most_knights = player.knights_played
                army_player = player

        # Update largest army status
        for player in self.players:
            player.has_largest_army = (player == army_player)

        return army_player, most_knights

    def try_place_initial_settlement(self, vertex, player=None):
        """Place settlement during initial placement"""
        if player is None:
            player = self.get_current_player()

        if not self.is_initial_placement_phase():
            return False, "Not in initial placement phase"

        if self.waiting_for_road:
            ##print(f"  [DEBUG] Cannot place settlement - waiting_for_road is True")
            return False, "Must place road first"

        placements = self.player_initial_placements[player]
        expected_settlements = 1 if self.game_phase == "INITIAL_PLACEMENT_1" else 2

        ##print(f"  [DEBUG] Phase: {self.game_phase}, Player has {placements['settlements']} settlements, expected: {expected_settlements}")

        if placements["settlements"] >= expected_settlements:
            return False, f"Already placed {expected_settlements} settlement(s) this round"

        # Check distance rule only
        if vertex.structure is not None:
            return False, "Vertex already occupied"

        for adj_vertex in vertex.adjacent_vertices:
            if adj_vertex.structure is not None:
                return False, "Too close to another settlement"

        # Place settlement (free during initial placement)
        vertex.build_settlement(player)
        placements["settlements"] += 1
        self.initial_settlements_placed += 1
        self.waiting_for_road = True
        self.last_settlement_vertex = vertex

        return True, "Settlement placed - now place a road"

    def try_place_initial_road(self, edge, player=None):
        """Place road during initial placement"""
        if player is None:
            player = self.get_current_player()

        if not self.is_initial_placement_phase():
            return False, "Not in initial placement phase"

        if not self.waiting_for_road:
            return False, "Must place settlement first"

        if edge.structure is not None:
            return False, "Edge already has a road"

        # Must connect to the last settlement placed
        if (edge.vertex1 != self.last_settlement_vertex and
                edge.vertex2 != self.last_settlement_vertex):
            return False, "Road must connect to your settlement"

        # Place road (free during initial placement)
        edge.build_road(player)
        placements = self.player_initial_placements[player]
        placements["roads"] += 1
        self.initial_roads_placed += 1
        self.waiting_for_road = False
        self.last_settlement_vertex = None

        success, msg = self.advance_initial_placement()

        return True, "Road placed"

    # ==================== TRADING SYSTEM ====================

    def execute_bank_trade(self, player, give_resource, get_resource, amount_to_give=None):
        """Execute a bank trade using best available port ratio"""
        if not self.can_trade_or_build():
            return False, "Cannot trade now - roll dice first"

        # Get the best trade ratio for the resource being given
        ratio = self.game_board.get_best_trade_ratio(player, give_resource)

        if amount_to_give is None:
            amount_to_give = ratio

        # Check if player has enough resources
        if player.resources[give_resource] < amount_to_give:
            return False, f"Need {amount_to_give} {give_resource.value} (only have {player.resources[give_resource]})"

        # Execute trade
        player.remove_resource(give_resource, amount_to_give)
        player.add_resource(get_resource, 1)

        return True, f"Traded {amount_to_give} {give_resource.value} for 1 {get_resource.value} (ratio {ratio}:1)"

    def get_available_trade_partners(self, player):
        """Get list of other players who can trade"""
        return [p for p in self.players if p != player]

    def propose_player_trade(self, offering_player, target_player, offered_resources, requested_resources):
        """Propose a trade to another player"""
        if not self.can_trade_or_build():
            return False, None, "Cannot trade now - roll dice first"

        # Validate offering player has the offered resources
        for resource_type, amount in offered_resources.items():
            if amount > 0 and offering_player.resources[resource_type] < amount:
                return False, None, f"You don't have enough {resource_type.value}"

        # Create trade offer
        trade_offer = TradeOffer(offering_player, target_player, offered_resources, requested_resources)
        self.pending_trade_offers.append(trade_offer)

        return True, trade_offer, f"Trade proposed to {target_player.name}"

    def get_pending_offers_for_player(self, player):
        """Get all pending trade offers for a player"""
        return [offer for offer in self.pending_trade_offers
                if offer.target_player == player and offer.status == TradeOfferStatus.PENDING]

    def accept_trade_offer(self, trade_offer):
        """Accept a trade offer"""
        if not trade_offer.is_valid():
            return False, "Trade no longer valid - players don't have resources"

        # Execute the trade
        offering_player = trade_offer.offering_player
        target_player = trade_offer.target_player

        # Transfer offered resources
        for resource_type, amount in trade_offer.offered_resources.items():
            if amount > 0:
                offering_player.remove_resource(resource_type, amount)
                target_player.add_resource(resource_type, amount)

        # Transfer requested resources
        for resource_type, amount in trade_offer.requested_resources.items():
            if amount > 0:
                target_player.remove_resource(resource_type, amount)
                offering_player.add_resource(resource_type, amount)

        trade_offer.status = TradeOfferStatus.ACCEPTED
        self.pending_trade_offers.remove(trade_offer)

        return True, "Trade completed!"

    def reject_trade_offer(self, trade_offer):
        """Reject a trade offer"""
        trade_offer.status = TradeOfferStatus.REJECTED
        self.pending_trade_offers.remove(trade_offer)
        return True, "Trade rejected"

    # ==================== BUILDING HELPERS ====================

    def get_buildable_vertices_for_settlements(self):
        """Get all vertices where current player can build settlements"""
        if not self.can_trade_or_build():
            return []

        current_player = self.get_current_player()
        buildable = []

        for vertex in self.game_board.vertices:
            if vertex.can_build_settlement(current_player, ignore_road_rule=False):
                buildable.append(vertex)

        return buildable

    def get_buildable_vertices_for_cities(self):
        """Get all vertices where current player can upgrade to cities"""
        if not self.can_trade_or_build():
            return []

        current_player = self.get_current_player()
        buildable = []

        for vertex in self.game_board.vertices:
            if vertex.can_build_city(current_player):
                buildable.append(vertex)

        return buildable

    def get_buildable_edges(self):
        """Get all edges where current player can build roads"""
        if not self.can_trade_or_build():
            return []

        current_player = self.get_current_player()
        buildable = []

        for edge in self.game_board.edges:
            if edge.can_build_road(current_player):
                buildable.append(edge)

        return buildable
