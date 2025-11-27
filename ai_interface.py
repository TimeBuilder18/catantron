"""
AI Training Interface for Catan

ENVIRONMENT ONLY - You implement the AI agents yourself!

This provides a clean interface to the Catan game for training 4 AI agents.
No GUI, no networking, no sockets - just pure game logic.

USAGE EXAMPLE:
--------------
from ai_interface import AIGameEnvironment

# 1. Create the environment
env = AIGameEnvironment()

# 2. Get initial observations (one per agent)
observations = env.reset()  # Returns list of 4 observations

# 3. Game loop
done = False
while not done:
    current_player = env.game.current_player_index

    # 4. Your AI decides action based on observation
    obs = observations[current_player]
    action = your_ai_agent.choose_action(obs)
    action_params = your_ai_agent.choose_params(obs)

    # 5. Execute action (NO REWARD - you calculate that yourself!)
    obs, done, info = env.step(current_player, action, action_params)
    observations[current_player] = obs

    # 6. Calculate your own reward based on obs
    reward = your_reward_function(obs, info)

    # 7. Your AI learns
    your_ai_agent.learn(obs, action, reward, done)

# 7. Reset for next game
observations = env.reset()

OBSERVATION FORMAT:
-------------------
obs = {
    'is_my_turn': bool,
    'game_phase': str,
    'my_resources': dict,        # YOUR private hand
    'my_victory_points': int,
    'opponents': list,            # Public info only (can't see their cards)
    'legal_actions': list,        # What you can do right now
    'tiles': list,                # Board state
    ...
}

ACTIONS:
--------
- 'roll_dice' (no params needed)
- 'place_settlement' (params: {'vertex': Vertex object})
- 'place_road' (params: {'edge': Edge object})
- 'build_settlement' (params: {'vertex': Vertex object})
- 'build_city' (params: {'vertex': Vertex object})
- 'build_road' (params: {'edge': Edge object})
- 'buy_dev_card' (no params)
- 'end_turn' (no params)

NOW GO BUILD YOUR AI! This is just the game environment.
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

        pass  # print("✅ AI Training Environment Ready!")
        pass  # print(f"   • 4 AI agents initialized")
        pass  # print(f"   • {len(tiles)} tiles")
        pass  # print(f"   • {len(game_board.ports)} ports")
        pass  # print(f"   • Game phase: {self.game.game_phase}")

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
                # Only add build actions if player has the resources!
                from game_system import ResourceType
                res = player.resources

                # Settlement: Wood=1, Brick=1, Wheat=1, Sheep=1
                if (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
                    res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1):
                    actions.append('build_settlement')

                # City: Ore=3, Wheat=2
                if res[ResourceType.ORE] >= 3 and res[ResourceType.WHEAT] >= 2:
                    actions.append('build_city')

                # Road: Wood=1, Brick=1
                if res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1:
                    actions.append('build_road')

                # Dev card: Ore=1, Wheat=1, Sheep=1
                if (res[ResourceType.ORE] >= 1 and res[ResourceType.WHEAT] >= 1 and
                    res[ResourceType.SHEEP] >= 1):
                    actions.append('buy_dev_card')

                # Can always trade with bank (4:1 or with ports)
                if sum(res.values()) >= 4:  # Need at least 4 resources to trade
                    actions.append('trade_with_bank')

            if self.game.can_end_turn():
                actions.append('end_turn')

        return actions

    def _auto_play_other_players(self, target_player_index):
        """Auto-play other players' turns using rule-based AI"""
        max_iterations = 100  # Safety limit
        iterations = 0

        print(f"[AUTO-PLAY] Starting: current={self.game.current_player_index}, target={target_player_index}, phase={self.game.game_phase}")

        while self.game.current_player_index != target_player_index and iterations < max_iterations:
            iterations += 1
            current_idx = self.game.current_player_index

            # Use rule-based AI for players 1-3
            if current_idx != 0 and self.ai_players[current_idx]:
                if hasattr(self, '_debug_autoplay'):
                    print(f"[AUTO-PLAY] Calling AI for player {current_idx}")
                success = self.ai_players[current_idx].play_turn(self.game, current_idx)
                if not success:
                    if hasattr(self, '_debug_autoplay'):
                        print(f"[AUTO-PLAY] AI returned False, stopping")
                    break  # AI couldn't make a move, exit
            else:
                break  # Player 0 or no AI available

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

        # Check for victory
        winner = self.game.check_victory_conditions()
        done = (winner is not None)

        if done:
            if winner:
                info['winner'] = self.game.players.index(winner)
                info['result'] = 'game_over'

        # IMPORTANT: Auto-play other players' turns after agent's action
        # This ensures initial placement and normal play progresses
        if not done:
            self._auto_play_other_players(player_index)

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


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    pass  # print("="*60)
    pass  # print("CATAN AI TRAINING ENVIRONMENT - READY")
    pass  # print("="*60)

    # Create environment
    env = AIGameEnvironment()

    # Get initial observations for all 4 players
    observations = env.reset()

    pass  # print("\n📊 Initial State:")
    for i, obs in enumerate(observations):
        pass  # print(f"   Player {i+1}:")
        pass  # print(f"      • Victory Points: {obs['my_victory_points']}")
        pass  # print(f"      • My Turn: {obs['is_my_turn']}")
        pass  # print(f"      • Legal Actions: {obs['legal_actions']}")
        pass  # print(f"      • Resources: {sum(obs['my_resources'].values())}")

    pass  # print("\n" + "="*60)
    pass  # print("✅ ENVIRONMENT READY FOR YOUR AI IMPLEMENTATION")
    pass  # print("="*60)
    pass  # print("\nHow to use:")
    pass  # print("1. Create your AI agents (4 agents)")
    pass  # print("2. Each agent calls: obs = env.get_observation(player_index)")
    pass  # print("3. Agent decides action from obs['legal_actions']")
    pass  # print("4. Execute: obs, reward, done, info = env.step(player_index, action, params)")
    pass  # print("5. Repeat until done == True")
    pass  # print("\n" + "="*60)
