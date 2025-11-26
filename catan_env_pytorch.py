"""
Catan Gymnasium Environment for PyTorch PPO Training

Designed to work with custom PyTorch PPO implementation.
Provides clean observation/action spaces and proper masking.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys

sys.path.append('/mnt/project')

from ai_interface import AIGameEnvironment
from game_system import ResourceType
from game_system import DevelopmentCardType
class CatanEnv(gym.Env):
    """
    Gymnasium environment for Catan - optimized for PyTorch PPO

    Key features:
    - Flat observation vector for neural network
    - Action masking for invalid actions
    - Proper reward shaping
    - Self-play ready (4 players)
    """

    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, player_id=0):
        """
        Args:
            player_id: Which player this environment controls (0-3)
        """
        super().__init__()

        self.player_id = player_id
        self.game_env = AIGameEnvironment()

        # Action space: 9 discrete actions
        # 0: roll_dice
        # 1: place_settlement (initial)
        # 2: place_road (initial)
        # 3: build_settlement
        # 4: build_city
        # 5: build_road
        # 6: buy_dev_card
        # 7: end_turn
        # 8: wait (not your turn)
        self.action_space = spaces.Discrete(9)

        # Observation space: flat vector + action mask
        # We'll create a comprehensive feature vector
        obs_size = self._calculate_obs_size()

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-10, high=100,
                shape=(obs_size,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(9,),
                dtype=np.int8
            )
        })

        #print(f"✅ CatanEnv initialized for Player {player_id}")
        #print(f"   Observation size: {obs_size}")
        #print(f"   Action space: {self.action_space.n}")

    def _calculate_obs_size(self):
        """Calculate total observation vector size"""
        size = 0

        # Game state: 10 features
        size += 11

        # My resources: 5 features
        size += 5

        # My buildings: 3 features
        size += 3

        # My dev cards: 5 features
        size += 5

        # My stats: 3 features (VP, knights, has_longest_road, has_largest_army)
        size += 4

        # Opponents (3 opponents x 6 features each): 18 features
        size += 18

        # Board tiles (19 tiles x 3 features): 57 features
        size += 57

        # Ports (9 ports x 2 features): 18 features
        size += 18

        # Total: 10+5+3+5+4+18+57+18 = 120 features
        return size

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Reset game
        self.game_env = AIGameEnvironment()

        # Get initial observation
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_obs(self):
        """
        Convert game state to flat observation vector

        Returns:
            dict with 'observation' (flat vector) and 'action_mask'
        """
        raw_obs = self.game_env.get_observation(self.player_id)

        features = []

        # === GAME STATE (10 features) ===
        # === GAME STATE (11 features) ===  # Changed from 10 to 11
        features.extend([
            1.0 if raw_obs['is_my_turn'] else 0.0,
            float(raw_obs['current_player']),
            1.0 if raw_obs['game_phase'] == 'INITIAL_PLACEMENT_1' else 0.0,
            1.0 if raw_obs['game_phase'] == 'INITIAL_PLACEMENT_2' else 0.0,
            1.0 if raw_obs['game_phase'] == 'NORMAL_PLAY' else 0.0,
            1.0 if raw_obs['dice_rolled'] else 0.0,
            float(raw_obs['last_roll'][2] if raw_obs['last_roll'] else 0),
            1.0 if raw_obs['turn_phase'] == 'ROLL_DICE' else 0.0,
            1.0 if raw_obs['turn_phase'] == 'TRADE_BUILD' else 0.0,
            1.0 if raw_obs['turn_phase'] == 'TURN_COMPLETE' else 0.0,
            1.0 if self.game_env.game.waiting_for_road else 0.0,  # ← ADD THIS!
        ])

        # === MY RESOURCES (5 features) ===
        resources = raw_obs['my_resources']
        features.extend([
            float(resources[ResourceType.WOOD]),
            float(resources[ResourceType.BRICK]),
            float(resources[ResourceType.WHEAT]),
            float(resources[ResourceType.SHEEP]),
            float(resources[ResourceType.ORE])
        ])

        # === MY BUILDINGS (3 features) ===
        features.extend([
            float(raw_obs['my_settlements']),
            float(raw_obs['my_cities']),
            float(raw_obs['my_roads'])
        ])

        # === MY DEV CARDS (5 features) ===
        dev_cards = raw_obs['my_dev_cards']

        features.extend([
            float(dev_cards.get(DevelopmentCardType.KNIGHT, 0)),
            float(dev_cards.get(DevelopmentCardType.VICTORY_POINT, 0)),
            float(dev_cards.get(DevelopmentCardType.ROAD_BUILDING, 0)),
            float(dev_cards.get(DevelopmentCardType.YEAR_OF_PLENTY, 0)),
            float(dev_cards.get(DevelopmentCardType.MONOPOLY, 0))
        ])

        # === MY STATS (4 features) ===
        player = self.game_env.game.players[self.player_id]
        features.extend([
            float(raw_obs['my_victory_points']),
            float(player.knights_played),
            1.0 if player.has_longest_road else 0.0,
            1.0 if player.has_largest_army else 0.0
        ])

        # === OPPONENTS (3 opponents x 6 features = 18 features) ===
        for opp in raw_obs['opponents']:
            features.extend([
                float(opp['resource_count']),
                float(opp['settlements']),
                float(opp['cities']),
                float(opp['roads']),
                float(opp['victory_points']),
                float(opp.get('dev_card_count', 0))
            ])
        # Pad if less than 3 opponents (shouldn't happen but safe)
        while len(features) < 10 + 5 + 3 + 5 + 4 + 18:
            features.append(0.0)

        # === BOARD TILES (19 tiles x 3 features = 57 features) ===
        resource_encoding = {
            'forest': 1.0, 'hill': 2.0, 'field': 3.0,
            'mountain': 4.0, 'pasture': 5.0, 'desert': 0.0
        }

        for q, r, resource, number in raw_obs['tiles']:
            features.extend([
                resource_encoding.get(resource, 0.0),
                float(number if number else 0),
                1.0 if self._tile_has_robber(q, r) else 0.0
            ])

        # === PORTS (9 ports x 2 features = 18 features) ===
        # Encode port type and whether player has access
        port_type_encoding = {
            'GENERIC': 1.0,
            'WOOD': 2.0,
            'BRICK': 3.0,
            'WHEAT': 4.0,
            'SHEEP': 5.0,
            'ORE': 6.0
        }

        for port_type, vx, vy in raw_obs['ports']:
            features.extend([
                port_type_encoding.get(port_type, 0.0),
                1.0 if self._player_has_port_access(vx, vy) else 0.0
            ])

        # === ACTION MASK (9 actions) ===
        # === ACTION MASK (9 actions) ===
        legal_actions = raw_obs['legal_actions']
        action_mask = np.zeros(9, dtype=np.int8)
        # Handle initial placement phase specially
        if self.game_env.game.is_initial_placement_phase():
            if self.game_env.game.waiting_for_road:
                action_mask[2] = 1  # place_road
            else:
                action_mask[1] = 1  # place_settlement
        else:
            # Normal play - map legal actions to indices
            action_map = {
                'roll_dice': 0,
                'build_settlement': 3,
                'build_city': 4,
                'build_road': 5,
                'buy_dev_card': 6,
                'end_turn': 7,
                'wait': 8
            }

            for action_name in legal_actions:
                if action_name in action_map:
                    action_mask[action_map[action_name]] = 1

        # Convert to numpy array
        observation = np.array(features, dtype=np.float32)

        return {
            'observation': observation,
            'action_mask': action_mask
        }

    def _tile_has_robber(self, q, r):
        """Check if tile at (q,r) has the robber"""
        robber_pos = self.game_env.game.robber.position
        if robber_pos:
            return robber_pos.q == q and robber_pos.r == r
        return False

    def _player_has_port_access(self, vx, vy):
        """Check if player has a settlement/city at port vertex"""
        player = self.game_env.game.players[self.player_id]

        # Check all player's settlements and cities
        for settlement in player.settlements:
            if abs(settlement.position.x - vx) < 0.1 and abs(settlement.position.y - vy) < 0.1:
                return True

        for city in player.cities:
            if abs(city.position.x - vx) < 0.1 and abs(city.position.y - vy) < 0.1:
                return True

        return False

    def _get_info(self):
        """Get additional info"""
        raw_obs = self.game_env.get_observation(self.player_id)
        return {
            'player_id': self.player_id,
            'is_my_turn': raw_obs['is_my_turn'],
            'game_phase': raw_obs['game_phase'],
            'victory_points': raw_obs['my_victory_points']
        }

    def step(self, action):
        raw_obs = self.game_env.get_observation(self.player_id)

        if not raw_obs['is_my_turn']:
            # Return observation with ALL actions masked (can't do anything)
            obs = self._get_obs()
            obs['action_mask'] = np.zeros(9, dtype=np.int8)  # ← Mask ALL actions
            obs['action_mask'][8] = 1  # Only 'wait' is valid
            info = self._get_info()
            return obs, 0.0, False, False, info

        # Get current action mask
        current_obs = self._get_obs()
        action_mask = current_obs['action_mask']

        # Check if action is legal
        if action_mask[action] == 0:
            obs = self._get_obs()
            info = self._get_info()
            info['illegal_action'] = True
            return obs, -1.0, False, False, info

        # ... rest of your step() method stays the same

        # Map action index to action name
        action_names = [
            'roll_dice', 'place_settlement', 'place_road',
            'build_settlement', 'build_city', 'build_road',
            'buy_dev_card', 'end_turn', 'wait'
        ]
        action_name = action_names[action]

        # Get action parameters (if needed)
        action_params = self._get_action_params(action_name)

        # Execute action in game
        new_obs, done, step_info = self.game_env.step(
            self.player_id,
            action_name,
            action_params
        )

        # Calculate reward
        reward = self._calculate_reward(raw_obs, new_obs, step_info)

        # Get formatted observation
        obs = self._get_obs()
        info = self._get_info()
        info.update(step_info)

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info
    def _get_action_params(self, action_name):
        """
        Get parameters for actions that need them
        Uses random selection for now - will be improved with policy network
        """
        if action_name == 'place_settlement':
            vertices = self.game_env.game.game_board.vertices
            player = self.game_env.game.players[self.player_id]

            if self.game_env.game.is_initial_placement_phase():
                # Initial placement: find valid spots
                valid = []
                for v in vertices:
                    if v.structure is None:
                        # Check distance rule
                        too_close = any(adj.structure is not None for adj in v.adjacent_vertices)
                        if not too_close:
                            valid.append(v)
            else:
                valid = self.game_env.game.get_buildable_vertices_for_settlements()

            if valid:
                return {'vertex': np.random.choice(valid)}

        elif action_name == 'place_road':
            edges = self.game_env.game.game_board.edges

            if self.game_env.game.is_initial_placement_phase():
                last_settlement = self.game_env.game.last_settlement_vertex
                if last_settlement:
                    valid = [e for e in edges
                             if e.structure is None and
                             (e.vertex1 == last_settlement or e.vertex2 == last_settlement)]
                else:
                    valid = []
            else:
                valid = self.game_env.game.get_buildable_edges()

            if valid:
                return {'edge': np.random.choice(valid)}

        elif action_name == 'build_settlement':
            valid = self.game_env.game.get_buildable_vertices_for_settlements()
            if valid:
                return {'vertex': np.random.choice(valid)}

        elif action_name == 'build_city':
            valid = self.game_env.game.get_buildable_vertices_for_cities()
            if valid:
                return {'vertex': np.random.choice(valid)}

        elif action_name == 'build_road':
            valid = self.game_env.game.get_buildable_edges()
            if valid:
                return {'edge': np.random.choice(valid)}

        return None

    def _calculate_reward(self, old_obs, new_obs, step_info):
        """
        Reward shaping for Catan

        Goals:
        - Win the game (10 VP)
        - Gain victory points
        - Build structures
        - Collect resources
        """
        reward = 0.0

        # Victory points (main goal)
        vp_diff = new_obs['my_victory_points'] - old_obs['my_victory_points']
        reward += vp_diff * 10.0

        # Building rewards
        settlement_diff = new_obs['my_settlements'] - old_obs['my_settlements']
        city_diff = new_obs['my_cities'] - old_obs['my_cities']
        road_diff = new_obs['my_roads'] - old_obs['my_roads']

        reward += settlement_diff * 2.0
        reward += city_diff * 3.0
        reward += road_diff * 0.5

        # Resource collection
        old_resources = sum(old_obs['my_resources'].values())
        new_resources = sum(new_obs['my_resources'].values())
        resource_diff = new_resources - old_resources
        reward += resource_diff * 0.1

        # Small negative reward each step (encourage efficiency)
        reward -= 0.01

        # Big win/loss bonus
        if step_info.get('result') == 'game_over':
            winner_id = step_info.get('winner')
            if winner_id == self.player_id:
                reward += 100.0  # Win!
            else:
                reward -= 10.0  # Loss

        # Illegal action penalty
        if not step_info.get('success', True):
            reward -= 0.5

        return reward

    def render(self):
        """Render the environment (text mode)"""
        raw_obs = self.game_env.get_observation(self.player_id)

        #print(f"\n{'=' * 50}")
        #print(f"Player {self.player_id} | VP: {raw_obs['my_victory_points']}/10")
        #print(f"{'=' * 50}")
        #print(f"Phase: {raw_obs['game_phase']}")
        #print(f"Resources: {sum(raw_obs['my_resources'].values())}")
        #print(f"Buildings: S:{raw_obs['my_settlements']} C:{raw_obs['my_cities']} R:{raw_obs['my_roads']}")
        #print(f"Legal Actions: {raw_obs['legal_actions']}")
        #print(f"{'=' * 50}\n")

    def close(self):
        """Cleanup"""
        pass


# ==================== TEST ====================

if __name__ == "__main__":
    #print("=" * 60)
    #print("TESTING CATAN GYMNASIUM ENVIRONMENT (PYTORCH)")
    #print("=" * 60)

    # Create environment
    env = CatanEnv(player_id=0)

    # Check spaces
    #print(f"\n✅ Observation Space: {env.observation_space}")
    #print(f"✅ Action Space: {env.action_space}")

    # Reset
    obs, info = env.reset()

    #print(f"\n✅ Observation shape: {obs['observation'].shape}")
    #print(f"✅ Action mask: {obs['action_mask']}")
    #print(f"✅ Info: {info}")

    # Take random actions
    #print("\n🎮 Taking 10 random actions...")
    total_reward = 0.0

    for step in range(10):
        # Get valid actions
        valid_actions = np.where(obs['action_mask'] == 1)[0]

        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            #print(f"Step {step + 1}: Action={action}, Reward={reward:.3f}, VP={info['victory_points']}")

            if terminated or truncated:
                #print("   Game ended!")
                break

    #print(f"\n✅ Total reward: {total_reward:.3f}")
    #print("✅ Environment test complete!")