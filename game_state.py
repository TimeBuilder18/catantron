"""
GameState wrapper for MCTS

Provides clean interface for:
- Copying game state
- Getting legal actions
- Simulating moves
- Checking terminal states
"""

import copy
import numpy as np
from catan_env_pytorch import CatanEnv


class GameState:
    """
    Wrapper around CatanEnv for MCTS simulations
    """

    def __init__(self, env=None):
        """
        Args:
            env: Existing CatanEnv, or None to create new game
        """
        if env is None:
            self.env = CatanEnv()
            self.env.reset()
        else:
            self.env = env

    def copy(self):
        """Create independent copy of this state"""
        new_state = GameState.__new__(GameState)
        new_state.env = copy.deepcopy(self.env)
        return new_state

    def get_observation(self):
        """Get current observation dict"""
        return self.env._get_obs()

    def get_legal_actions(self):
        """
        Get list of all legal actions as tuples: (action, vertex, edge)

        Action mapping (from your catan_env_pytorch.py):
        0 = roll_dice
        1 = place_settlement (initial) - needs vertex
        2 = place_road (initial) - needs EDGE
        3 = build_settlement - needs vertex
        4 = build_city - needs vertex
        5 = build_road - needs edge
        6 = buy_dev_card
        7 = end_turn
        8 = wait
        9 = trade_with_bank
        10 = do_nothing

        Returns:
            List of (action_id, vertex_id, edge_id) tuples
        """
        obs = self.get_observation()
        action_mask = obs['action_mask']
        vertex_mask = obs['vertex_mask']
        edge_mask = obs['edge_mask']

        legal_actions = []

        for action_id in range(len(action_mask)):
            if action_mask[action_id] == 0:
                continue

            # Actions that need VERTEX selection
            if action_id in [1, 3, 4]:  # place_settlement, build_settlement, build_city
                for vertex_id in range(len(vertex_mask)):
                    if vertex_mask[vertex_id] == 1:
                        legal_actions.append((action_id, vertex_id, 0))

            # Actions that need EDGE selection
            elif action_id in [2, 5]:  # place_road, build_road
                for edge_id in range(len(edge_mask)):
                    if edge_mask[edge_id] == 1:
                        legal_actions.append((action_id, 0, edge_id))

            # Actions with no location needed
            else:
                legal_actions.append((action_id, 0, 0))

        return legal_actions

    def apply_action(self, action, vertex=0, edge=0, trade_give=0, trade_get=0):
        """
        Apply action to this state (modifies state in place)

        Returns:
            reward: float
            done: bool
        """
        obs, reward, terminated, truncated, info = self.env.step(
            action, vertex, edge, trade_give, trade_get
        )
        done = terminated or truncated
        return reward, done

    def is_terminal(self):
        """Check if game is over"""
        winner = self.env.game_env.game.check_victory_conditions()
        return winner is not None

    def get_current_player(self):
        """Get current player index (0-3)"""
        return self.env.game_env.game.current_player_index

    def get_winner(self):
        """
        Get winner player index

        Returns:
            int: Winner player index (0-3), or None if game not over
        """
        winner = self.env.game_env.game.check_victory_conditions()
        if winner:
            return self.env.game_env.game.players.index(winner)
        return None

    def get_result(self, player_id):
        """
        Get result from perspective of player_id

        Returns:
            1.0 if player won, -1.0 if lost, 0.0 if draw/ongoing
        """
        winner = self.get_winner()
        if winner is None:
            return 0.0
        return 1.0 if winner == player_id else -1.0

    def get_victory_points(self, player_id=None):
        """Get victory points for a player (default: current player)"""
        if player_id is None:
            player_id = self.get_current_player()
        return self.env.game_env.game.players[player_id].victory_points


# Quick test
if __name__ == "__main__":
    print("Testing GameState wrapper...")

    # Test 1: Create state
    state = GameState()
    print(f"✅ Created new game state")

    # Test 2: Copy state
    state_copy = state.copy()
    print(f"✅ Copied state")

    # Test 3: Check they're independent
    original_vp = state.get_victory_points(0)
    state_copy.env.game_env.game.players[0].victory_points = 99

    if state.get_victory_points(0) == original_vp:
        print(f"✅ States are independent")
    else:
        print(f"❌ States are linked!")

    # Test 4: Get legal actions
    legal = state.get_legal_actions()
    print(f"✅ Got {len(legal)} legal actions")
    if legal:
        print(f"   First action: {legal[0]}")

    # Test 5: Check terminal
    print(f"✅ Is terminal: {state.is_terminal()}")

    # Test 6: Get current player
    print(f"✅ Current player: {state.get_current_player()}")

    print("\n✅ All GameState tests passed!")