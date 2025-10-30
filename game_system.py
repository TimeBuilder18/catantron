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


class GameConstants:
    """Game constants for Catan"""
    # Victory conditions
    VICTORY_POINTS_TO_WIN = 10
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
        if not self.can_afford(Settlement.get_cost()):
            return False, "Not enough resources"

        if not vertex.can_build_settlement(self, ignore_road_rule):
            return False, "Cannot build settlement here"

        self.pay_cost(Settlement.get_cost())
        vertex.build_settlement(self)
        return True, "Settlement built"

    def try_build_city(self, vertex):
        """Attempt to build a city"""
        if not self.can_afford(City.get_cost()):
            return False, "Not enough resources"

        if not vertex.can_build_city(self):
            return False, "Cannot build city here"

        self.pay_cost(City.get_cost())
        vertex.build_city(self)
        return True, "City built"

    def try_build_road(self, edge):
        """Attempt to build a road"""
        if not self.can_afford(Road.get_cost()):
            return False, "Not enough resources"

        if not edge.can_build_road(self):
            return False, "Cannot build road here"

        self.pay_cost(Road.get_cost())
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
        self.generate_vertices_and_edges()

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
                        print("ERROR: Preventing dice roll during initial placement/setup!")
                        print(f"ERROR: Current phase: {game_system.game_phase}")
                        print("ERROR: Dice should only roll during normal play when user presses 'D'")
                        return 1, 1, 2  # Return fake result to prevent crash

                print(f"DEBUG: Dice roll called from: {caller_frame.f_code.co_filename}:{caller_frame.f_lineno}")
                print(f"DEBUG: In function: {caller_frame.f_code.co_name}")
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

    def __init__(self, game_board, players):
        self.game_board = game_board
        self.players = players
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

    def give_starting_resources_step2(self):
        """Step 2: Absolute minimum - just iterate players"""
        print("\n=== STARTING RESOURCES - STEP 2 (MINIMAL) ===")

        for player in self.players:
            print(f"Player name: {player.name}")
            # Don't touch settlements, resources, or anything else

        print("=== STEP 2 COMPLETE - ONLY TOUCHED PLAYER NAMES ===\n")
        """Roll dice and distribute resources"""
        if self.dice_rolled:
            return None

        die1, die2, total = DiceRoller.roll_dice()
        self.last_dice_roll = (die1, die2, total)
        self.dice_rolled = True

        if total == 7:
            self.last_resource_gains = {}
        else:
            self.last_resource_gains = DiceRoller.distribute_resources(
                total, self.game_board, self.players
            )

        return self.last_dice_roll

    def roll_dice(self):
        """Roll dice and distribute resources - ONLY CALL THIS FROM USER INPUT"""
        if not self.can_roll_dice():
            print("ERROR: Cannot roll dice right now")
            return None

        print(f"DEBUG: {self.get_current_player().name} is rolling dice via game_system.roll_dice()")

        # Call the dice roller directly instead of through the class method
        die1 = random.randint(1, 6)
        die2 = random.randint(1, 6)
        total = die1 + die2

        self.last_dice_roll = (die1, die2, total)

        # CRITICAL: Update both flags to prevent multiple rolls
        self.dice_rolled = True
        self.turn_phase = "TRADE_BUILD"

        print(f"DEBUG: Dice result = {total}, turn_phase = {self.turn_phase}, dice_rolled = {self.dice_rolled}")

        if total == 7:
            self.last_resource_gains = {}
            print("Rolled 7! Robber activates (not implemented yet)")
        else:
            self.last_resource_gains = DiceRoller.distribute_resources(
                total, self.game_board, self.players
            )

        return self.last_dice_roll

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

            # Move to next player
            self.current_player_index = (self.current_player_index + 1) % len(self.players)
            self.turn_number += 1

            print(f"\n=== TURN {self.turn_number} ===")
            print(f"{self.get_current_player().name}'s turn")
            print("Press 'D' to roll dice and start your turn")

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
                    print("=== INITIAL PLACEMENT ROUND 2 ===")
                    print("Players place in reverse order")

            elif self.game_phase == "INITIAL_PLACEMENT_2":
                # In second round, go in reverse order
                self.current_player_index -= 1
                if self.current_player_index < 0:
                    print("DEBUG: About to transition to normal play...")
                    self.game_phase = "NORMAL_PLAY"
                    self.turn_phase = "ROLL_DICE"
                    self.current_player_index = 0
                    self.turn_number = 1
                    self.dice_rolled = False
                    self.last_dice_roll = None
                    self.last_resource_gains = None

                    # Give proper starting resources from second settlements (safe version)
                    print("DEBUG: Giving proper starting resources from second settlements...")
                    for i in range(len(self.players)):
                        player = self.players[i]
                        print(f"Processing {player.name}...")

                        # Check if player has at least 2 settlements
                        if len(player.settlements) >= 2:
                            # Get the second settlement (last one placed)
                            second_settlement = player.settlements[1]
                            vertex = second_settlement.position

                            print(
                                f"  {player.name}'s second settlement has {len(vertex.adjacent_tiles)} adjacent tiles")

                            # Check each adjacent tile for resources
                            for tile in vertex.adjacent_tiles:
                                tile_resource = tile.resource
                                print(f"    Tile resource: {tile_resource}")

                                if tile_resource == "forest":
                                    player.resources[ResourceType.WOOD] += 1
                                    print(f"      Added 1 wood to {player.name}")
                                elif tile_resource == "hill":
                                    player.resources[ResourceType.BRICK] += 1
                                    print(f"      Added 1 brick to {player.name}")
                                elif tile_resource == "field":
                                    player.resources[ResourceType.WHEAT] += 1
                                    print(f"      Added 1 wheat to {player.name}")
                                elif tile_resource == "mountain":
                                    player.resources[ResourceType.ORE] += 1
                                    print(f"      Added 1 ore to {player.name}")
                                elif tile_resource == "pasture":
                                    player.resources[ResourceType.SHEEP] += 1
                                    print(f"      Added 1 sheep to {player.name}")
                                elif tile_resource == "desert":
                                    print(f"      Desert tile - no resource")
                        else:
                            print(f"  {player.name} has no second settlement")

                    print("Starting resources complete!")
                    print(f"Turn {self.turn_number} - {self.get_current_player().name}")
                    print("Press 'D' to roll dice and begin your turn!")

            self.waiting_for_road = False
            self.last_settlement_vertex = None

            if self.game_phase != "NORMAL_PLAY":
                print(f"{self.get_current_player().name}'s turn - {self.get_current_player_needs()}")

            return True, "Advanced to next player"
        else:
            return False, "Must complete settlement and road placement first"
        """Get all vertices where player can build settlements"""
        if player is None:
            player = self.get_current_player()

        return [v for v in self.game_board.vertices if v.can_build_settlement(player, self.ignore_road_rule)]

    def get_buildable_vertices_for_settlements(self, player=None):
        """Get all vertices where player can build settlements"""
        if player is None:
            player = self.get_current_player()

        if self.is_initial_placement_phase():
            # During initial placement, only check distance rule
            return [v for v in self.game_board.vertices
                    if v.structure is None and
                    all(adj.structure is None for adj in v.adjacent_vertices)]
        else:
            # Normal play - check all rules
            return [v for v in self.game_board.vertices if v.can_build_settlement(player, False)]

    def get_buildable_edges_for_initial_roads(self, player=None):
        """Get edges where player can place initial roads (must connect to last settlement)"""
        if player is None:
            player = self.get_current_player()

        if not self.waiting_for_road or not self.last_settlement_vertex:
            return []

        return [e for e in self.game_board.edges
                if e.structure is None and
                (e.vertex1 == self.last_settlement_vertex or e.vertex2 == self.last_settlement_vertex)]
        """Get all vertices where player can build cities"""
        if player is None:
            player = self.get_current_player()

        return [v for v in self.game_board.vertices if v.can_build_city(player)]

    def get_buildable_edges(self, player=None):
        """Get all edges where player can build roads"""
        if player is None:
            player = self.get_current_player()

        return [e for e in self.game_board.edges if e.can_build_road(player)]

    def debug_reset_turn_state(self):
        """Debug method to reset turn state"""
        self.dice_rolled = False
        self.turn_phase = "ROLL_DICE"
        self.last_dice_roll = None
        self.last_resource_gains = None
        print(f"DEBUG: Turn state reset for {self.get_current_player().name}")
        print("You can now roll dice with 'D'")

    # ==================== ENHANCED TRADING SYSTEM ====================

    def get_available_trade_partners(self, current_player):
        """Get list of other players for trading"""
        return [p for p in self.players if p != current_player]

    def can_afford_bank_trade(self, player, offering_resource, trade_ratio=4):
        """Check if player can afford a bank trade"""
        return player.resources[offering_resource] >= trade_ratio

    def execute_bank_trade(self, player, offering_resource, requesting_resource, trade_ratio=4):
        """Execute a bank trade"""
        if not self.can_trade_or_build():
            return False, "Can only trade during trade/build phase"

        if not self.can_afford_bank_trade(player, offering_resource, trade_ratio):
            return False, f"Need {trade_ratio} {offering_resource.value} to trade"

        # Execute trade
        player.remove_resource(offering_resource, trade_ratio)
        player.add_resource(requesting_resource, 1)

        return True, f"Traded {trade_ratio} {offering_resource.value} for 1 {requesting_resource.value}"

    def propose_player_trade(self, offering_player, target_player, offered_resources, requested_resources):
        """Propose a trade between players"""
        if not self.can_trade_or_build():
            return False, "Can only trade during trade/build phase"

        # Validate offering player has resources
        for resource_type, amount in offered_resources.items():
            if amount > 0 and offering_player.resources[resource_type] < amount:
                return False, f"You don't have {amount} {resource_type.value}"

        # Validate target player has requested resources
        for resource_type, amount in requested_resources.items():
            if amount > 0 and target_player.resources[resource_type] < amount:
                return False, f"{target_player.name} doesn't have {amount} {resource_type.value}"

        return True, f"Trade proposal valid with {target_player.name}"

    def execute_player_trade(self, offering_player, target_player, offered_resources, requested_resources):
        """Execute a trade between players (auto-accept for now)"""
        success, message = self.propose_player_trade(offering_player, target_player, offered_resources,
                                                     requested_resources)
        if not success:
            return False, message

        # Execute the trade
        for resource_type, amount in offered_resources.items():
            if amount > 0:
                offering_player.remove_resource(resource_type, amount)
                target_player.add_resource(resource_type, amount)

        for resource_type, amount in requested_resources.items():
            if amount > 0:
                target_player.remove_resource(resource_type, amount)
                offering_player.add_resource(resource_type, amount)

        return True, f"Trade completed with {target_player.name}"

    # ==================== DEVELOPMENT CARD SYSTEM ====================

    def can_play_development_card(self, player, card_type):
        """Check if player can play a development card"""
        if not self.can_trade_or_build():
            return False, "Can only play cards during trade/build phase"

        if player.development_cards[card_type] <= 0:
            return False, f"No {card_type.value} cards available"

        return True, "Can play card"

    def play_knight_card(self, player):
        """Play a knight development card"""
        success, message = self.can_play_development_card(player, DevelopmentCardType.KNIGHT)
        if not success:
            return False, message

        # Remove card and track knights played
        player.development_cards[DevelopmentCardType.KNIGHT] -= 1
        player.knights_played += 1

        # Check for largest army (3+ knights and most among all players)
        if player.knights_played >= GameConstants.LARGEST_ARMY_MIN_KNIGHTS:
            current_largest = max(p.knights_played for p in self.players)
            if player.knights_played >= current_largest:
                # Remove largest army from other players
                for p in self.players:
                    p.has_largest_army = False
                # Give to current player
                player.has_largest_army = True

        return True, f"Knight played! Move robber and steal from adjacent player. Knights played: {player.knights_played}"

    def play_year_of_plenty_card(self, player, resource1, resource2):
        """Play Year of Plenty card - take 2 resources from bank"""
        success, message = self.can_play_development_card(player, DevelopmentCardType.YEAR_OF_PLENTY)
        if not success:
            return False, message

        player.development_cards[DevelopmentCardType.YEAR_OF_PLENTY] -= 1
        player.add_resource(resource1, 1)
        player.add_resource(resource2, 1)

        return True, f"Year of Plenty: Gained {resource1.value} and {resource2.value}"

    def play_road_building_card(self, player):
        """Play Road Building card - build 2 free roads"""
        success, message = self.can_play_development_card(player, DevelopmentCardType.ROAD_BUILDING)
        if not success:
            return False, message

        player.development_cards[DevelopmentCardType.ROAD_BUILDING] -= 1
        return True, "Road Building: Click 2 edges to build free roads"

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

        return True, f"Stole 1 resource from {target_player.name}"

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
            points = player.calculate_victory_points()
            if points >= GameConstants.VICTORY_POINTS_TO_WIN:
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
        """Check if any player has won"""
        for player in self.players:
            if player.calculate_victory_points() >= GameConstants.VICTORY_POINTS_TO_WIN:
                return player
        return None

    def try_place_initial_settlement(self, vertex, player=None):
        """Place settlement during initial placement"""
        if player is None:
            player = self.get_current_player()

        if not self.is_initial_placement_phase():
            return False, "Not in initial placement phase"

        if self.waiting_for_road:
            return False, "Must place road first"

        placements = self.player_initial_placements[player]
        expected_settlements = 1 if self.game_phase == "INITIAL_PLACEMENT_1" else 2

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

        return True, "Road placed"