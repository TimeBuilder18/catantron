"""
AI Training Interface for Catan

ENVIRONMENT ONLY - You implement the AI agents yourself!

This provides a clean interface to the Catan game for training 4 AI agents.
No GUI, no networking, no sockets - just pure game logic.
"""

import random
from game_system import GameSystem, GameBoard, Player, Robber
from rule_based_ai import RuleBasedAI
from tile import Tile

# Same board setup as huPlay.py
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
    """Assign resources and numbers"""
    all_resources = RESOURCES.copy()
    random.shuffle(all_resources)

    for i, tile in enumerate(tiles):
        if i < len(all_resources):
            tile.resource = all_resources[i]

    # Place robber on desert
    for tile in tiles:
        if tile.resource == "desert":
            tile.number = None
            robber.move_to_tile(tile)
            break

    # Assign numbers to non-desert tiles
    nums = NUMBER_TOKENS.copy()
    random.shuffle(nums)
    non_desert = [t for t in tiles if t.resource != "desert"]
    for i, tile in enumerate(non_desert):
        if i < len(nums):
            tile.number = nums[i]


class AIGameEnvironment:
    """
    Game environment for AI training
    Manages 4 AI agents playing Catan
    """

    def __init__(self):
        """Initialize game for 4 AI agents"""
        # Create board
        tile_size = 50
        tiles = create_hexagonal_board(tile_size, radius=2)

        for t in tiles:
            t.find_neighbors(tiles)

        robber = Robber()
        assign_resources_numbers(tiles, robber)

        game_board = GameBoard(tiles)

        # Create 4 AI players
        players = [
            Player("AI Agent 1", (255, 50, 50)),
            Player("AI Agent 2", (50, 50, 255)),
            Player("AI Agent 3", (255, 255, 50)),
            Player("AI Agent 4", (255, 255, 255))
        ]

        self.game = GameSystem(game_board, players)

        # Create rule-based AIs for players 1-3
        self.ai_players = [None, RuleBasedAI(), RuleBasedAI(), RuleBasedAI()]
        self.game.robber = robber

    def get_observation(self, player_index):
        """
        Get observation for a specific AI agent
        This is what the agent "sees" - only their private info + public board state
        """
        player = self.game.players[player_index]
        current_player_idx = self.game.players.index(self.game.get_current_player())

        # Calculate discard information
        must_discard = player in self.game.players_must_discard
        if must_discard:
            total_cards = sum(player.resources.values())
            cards_to_discard = total_cards // 2
        else:
            cards_to_discard = 0

        obs = {
            # Game state
            'current_player': current_player_idx,
            'is_my_turn': (current_player_idx == player_index),
            'game_phase': self.game.game_phase,
            'turn_phase': self.game.turn_phase,
            'dice_rolled': self.game.dice_rolled,
            'last_roll': self.game.last_dice_roll,

            # Discard state (new - for 7 rolled discard feature)
            'waiting_for_discards': self.game.waiting_for_discards,
            'must_discard': must_discard,
            'must_discard_count': cards_to_discard,
            'players_discarding': len(self.game.players_must_discard),

            # My private info
            'my_resources': dict(player.resources),
            'my_dev_cards': dict(player.development_cards),
            'my_settlements': len(player.settlements),
            'my_cities': len(player.cities),
            'my_roads': len(player.roads),
            'my_victory_points': player.calculate_victory_points(),

            # Public info about opponents (can't see their cards)
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

            # Board state
            'tiles': [(t.q, t.r, t.resource, t.number) for t in self.game.game_board.tiles],
            'ports': [(p.port_type.name, p.vertex1.x, p.vertex1.y) for p in self.game.game_board.ports],

            # Available actions
            'legal_actions': self.get_legal_actions(player_index)
        }

        return obs

    def get_legal_actions(self, player_index):
        """Get list of legal actions for this player"""
        player = self.game.players[player_index]
        current_player = self.game.get_current_player()

        if player != current_player:
            return ['wait']  # Not your turn

        actions = []

        # Initial placement phase
        if self.game.is_initial_placement_phase():
            if self.game.waiting_for_road:
                actions.append('place_road')
            else:
                actions.append('place_settlement')

        # Normal play
        else:
            if self.game.can_roll_dice():
                actions.append('roll_dice')

            if self.game.can_trade_or_build():
                from game_system import ResourceType
                res = player.resources

                # Check for buildable locations AND resources
                if (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
                    res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1 and
                    self.game.get_buildable_vertices_for_settlements()):
                    actions.append('build_settlement')

                if (res[ResourceType.ORE] >= 3 and res[ResourceType.WHEAT] >= 2 and
                    self.game.get_buildable_vertices_for_cities()):
                    actions.append('build_city')

                if (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
                    self.game.get_buildable_edges()):
                    actions.append('build_road')

                if (res[ResourceType.ORE] >= 1 and res[ResourceType.WHEAT] >= 1 and
                    res[ResourceType.SHEEP] >= 1):
                    actions.append('buy_dev_card')

                if sum(res.values()) >= 4:
                    actions.append('trade_with_bank')

            if self.game.can_end_turn():
                actions.append('end_turn')
        
        # Fallback action if no other actions are available
        if not actions:
            actions.append('end_turn')

        return actions

    def _auto_play_other_players(self, target_player_index):
        """Auto-play other players' turns using rule-based AI"""
        max_iterations = 100  # Safety limit
        iterations = 0

        while self.game.current_player_index != target_player_index and iterations < max_iterations:
            iterations += 1
            current_idx = self.game.current_player_index

            # Use rule-based AI for players 1-3
            if current_idx != 0 and self.ai_players[current_idx]:
                success = self.ai_players[current_idx].play_turn(self.game, current_idx)
                if not success:
                    break  # AI couldn't make a move, exit
            else:
                break  # Player 0 or no AI available

    def _handle_automatic_discards(self):
        """
        Automatically discard cards for all players with >7 cards when 7 is rolled
        This simplifies the AI training by not requiring agents to learn discard strategy
        """
        if not self.game.waiting_for_discards:
            return

        from game_system import ResourceType
        import random

        for player in self.game.players_must_discard:
            if player in self.game.players_discarded:
                continue  # Already discarded

            total_resources = player.get_total_resources()
            num_to_discard = total_resources // 2

            # Build list of all cards
            all_cards = []
            for res_type in ResourceType:
                count = player.resources[res_type]
                all_cards.extend([res_type] * count)

            # Randomly select cards to discard
            if all_cards and num_to_discard > 0:
                cards_to_discard_list = random.sample(all_cards, min(num_to_discard, len(all_cards)))

                # Count discards by type
                discard_dict = {}
                for res_type in ResourceType:
                    discard_dict[res_type] = cards_to_discard_list.count(res_type)

                # Perform discard
                self.game.discard_cards(player, discard_dict)

        # Clear waiting flag after all discards and move robber automatically
        if len(self.game.players_discarded) >= len(self.game.players_must_discard):
            self.game.waiting_for_discards = False
            self.game.players_must_discard = []
            self.game.players_discarded = set()

            # Automatically move robber to a random tile (not current position)
            # This simplifies AI training by not requiring robber movement strategy
            import random
            available_tiles = [t for t in self.game.game_board.tiles if t != self.game.robber.position]
            if available_tiles:
                new_tile = random.choice(available_tiles)
                self.game.move_robber_to_tile(new_tile)

    def step(self, player_index, action, action_params=None):
        """
        Execute an action for a player
        Returns: (observation, done, info)

        NOTE: No reward calculation - you implement your own reward function!
        """
        player = self.game.players[player_index]

        # Validate it's player's turn
        if player != self.game.get_current_player():
            return self.get_observation(player_index), False, {'error': 'Not your turn'}

        info = {}

        # Execute action
        if action == 'roll_dice':
            dice_result = self.game.roll_dice()
            success = dice_result is not None
            message = f"Rolled {dice_result[2]}" if success else "Cannot roll dice"
            info['success'] = success
            info['message'] = message

            # Handle automatic discards when 7 is rolled
            if success and dice_result[2] == 7:
                self._handle_automatic_discards()

        elif action == 'end_turn':
            success, message = self.game.end_turn()
            info['success'] = success
            info['message'] = message

        elif action == 'place_settlement':
            vertex = action_params.get('vertex') if action_params else None
            if vertex:
                if self.game.is_initial_placement_phase():
                    success, message = self.game.try_place_initial_settlement(vertex, player)
                else:
                    success, message = self.game.try_build_settlement(vertex, player)
                info['success'] = success
                info['message'] = message
            else:
                info['success'] = False
                info['message'] = 'No vertex provided'

        elif action == 'place_road':
            edge = action_params.get('edge') if action_params else None
            if edge:
                if self.game.is_initial_placement_phase():
                    success, message = self.game.try_place_initial_road(edge, player)
                else:
                    success, message = self.game.try_build_road(edge, player)
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

        elif action == 'build_city':
            vertex = action_params.get('vertex') if action_params else None
            if vertex:
                success, message = self.game.try_build_city(vertex, player)
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

        elif action == 'buy_dev_card':
            success, message = self.game.try_buy_development_card(player)
            info['success'] = success
            info['message'] = message

        # Check for victory
        winner = self.game.check_victory_conditions()
        done = (winner is not None)

        if done:
            if winner:
                info['winner'] = self.game.players.index(winner)
                info['result'] = 'game_over'

        # IMPORTANT: Auto-play other players' turns after agent's action
        # if not done:
        #     self._auto_play_other_players(player_index)

        obs = self.get_observation(player_index)
        return obs, done, info

    def reset(self):
        """Reset game for new episode"""
        self.__init__()
        return [self.get_observation(i) for i in range(4)]

    def get_game_state(self):
        """
        Get raw game state for advanced use cases
        Returns the actual GameSystem object if you need low-level access
        """
        return {
            'game': self.game,
            'game_board': self.game.game_board,
            'players': self.game.players,
            'current_player_index': self.game.current_player_index,
            'game_phase': self.game.game_phase,
            'turn_phase': self.game.turn_phase
        }
