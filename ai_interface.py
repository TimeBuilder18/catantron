"""
AI Training Interface for Catan
Allows 4 AI agents to play against each other using the existing game engine
No networking, no multiple windows - just pure game logic
"""

import random
from game_system import GameSystem, GameBoard, Player, Robber
from tile import Tile

# Same board setup as main.py
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
        self.game.robber = robber

        print("✅ AI Training Environment Ready!")
        print(f"   • 4 AI agents initialized")
        print(f"   • {len(tiles)} tiles")
        print(f"   • {len(game_board.ports)} ports")
        print(f"   • Game phase: {self.game.game_phase}")

    def get_observation(self, player_index):
        """
        Get observation for a specific AI agent
        This is what the agent "sees" - only their private info + public board state
        """
        player = self.game.players[player_index]
        current_player_idx = self.game.players.index(self.game.get_current_player())

        obs = {
            # Game state
            'current_player': current_player_idx,
            'is_my_turn': (current_player_idx == player_index),
            'game_phase': self.game.game_phase,
            'turn_phase': self.game.turn_phase,
            'dice_rolled': self.game.dice_rolled,
            'last_roll': self.game.last_dice_roll,

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
                # Can place road connected to last settlement
                actions.append('place_road')
            else:
                # Can place settlement
                actions.append('place_settlement')

        # Normal play
        else:
            if self.game.can_roll_dice():
                actions.append('roll_dice')

            if self.game.can_trade_or_build():
                actions.append('build_settlement')
                actions.append('build_city')
                actions.append('build_road')
                actions.append('buy_dev_card')
                actions.append('trade_with_bank')
                # TODO: Add player-to-player trading

            if self.game.can_end_turn():
                actions.append('end_turn')

        return actions

    def step(self, player_index, action, action_params=None):
        """
        Execute an action for a player
        Returns: (observation, reward, done, info)
        """
        player = self.game.players[player_index]

        # Validate it's player's turn
        if player != self.game.get_current_player():
            return self.get_observation(player_index), 0, False, {'error': 'Not your turn'}

        reward = 0
        info = {}

        # Execute action
        if action == 'roll_dice':
            success, message = self.game.roll_dice_action()
            info['message'] = message

        elif action == 'end_turn':
            success, message = self.game.end_turn()
            info['message'] = message

        elif action == 'place_settlement':
            vertex = action_params.get('vertex')
            if vertex:
                if self.game.is_initial_placement_phase():
                    success, message = self.game.try_place_initial_settlement(vertex, player)
                else:
                    success, message = self.game.try_build_settlement(vertex, player)
                info['message'] = message
                if success:
                    reward = 1  # Reward for successful building

        elif action == 'place_road':
            edge = action_params.get('edge')
            if edge:
                if self.game.is_initial_placement_phase():
                    success, message = self.game.try_place_initial_road(edge, player)
                else:
                    success, message = self.game.try_build_road(edge, player)
                info['message'] = message
                if success:
                    reward = 0.5

        # Check for victory
        winner = self.game.check_victory_conditions()
        done = (winner is not None)

        if done:
            if winner == player:
                reward = 100  # Big reward for winning!
                info['result'] = 'victory'
            else:
                reward = -10  # Penalty for losing
                info['result'] = 'defeat'

        obs = self.get_observation(player_index)
        return obs, reward, done, info

    def reset(self):
        """Reset game for new episode"""
        self.__init__()
        return [self.get_observation(i) for i in range(4)]


# ==================== EXAMPLE: Random AI Agent ====================

class RandomAI:
    """Simple random agent for testing"""

    def __init__(self, player_index):
        self.player_index = player_index

    def choose_action(self, observation):
        """Choose random legal action"""
        legal_actions = observation['legal_actions']
        return random.choice(legal_actions)

    def choose_action_params(self, action, env):
        """Choose random parameters for action (e.g., which vertex to build on)"""
        # TODO: Implement smart action parameter selection
        # For now, return None (agents would need to select specific vertices/edges)
        return None


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("="*60)
    print("CATAN AI TRAINING ENVIRONMENT - DEMO")
    print("="*60)

    # Create environment
    env = AIGameEnvironment()

    # Create 4 random agents
    agents = [RandomAI(i) for i in range(4)]

    # Get initial observations
    observations = env.reset()

    print("\n📊 Initial observations:")
    for i, obs in enumerate(observations):
        print(f"   Agent {i+1}: {obs['my_victory_points']} VP, Turn: {obs['is_my_turn']}")
        print(f"              Legal actions: {obs['legal_actions']}")

    print("\n✅ Environment ready for AI training!")
    print("   Use this interface to train reinforcement learning agents")
    print("   (PyTorch, Stable-Baselines3, etc.)")
