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
        self._episode_count = 0  # Track episodes for debug output

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
        self._episode_count += 1  # Increment episode counter

        # Get initial observation
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_vertex_mask(self):
        """
        Create mask for valid vertices based on current game state

        Returns:
            np.array of shape [54]: 1 = valid vertex, 0 = invalid
        """
        mask = np.zeros(54, dtype=np.float32)

        # Get all vertices on the board
        all_vertices = self.game_env.game.game_board.vertices

        # Determine which vertices are valid based on current phase
        if self.game_env.game.is_initial_placement_phase():
            if not self.game_env.game.waiting_for_road:
                # Initial settlement placement
                valid_vertices = []
                for v in all_vertices:
                    if v.structure is None:
                        # Check distance rule
                        too_close = any(adj.structure is not None for adj in v.adjacent_vertices)
                        if not too_close:
                            valid_vertices.append(v)
            else:
                # Waiting for road, no vertices needed
                return np.ones(54, dtype=np.float32)  # All ones (will be ignored)
        else:
            # Normal play - settlements or cities
            player = self.game_env.game.players[self.player_id]
            current_player = self.game_env.game.get_current_player()

            if player == current_player and self.game_env.game.can_trade_or_build():
                # Can build settlements
                valid_vertices = self.game_env.game.get_buildable_vertices_for_settlements()

                # Can also upgrade to cities
                city_vertices = self.game_env.game.get_buildable_vertices_for_cities()
                valid_vertices = list(set(valid_vertices + city_vertices))
            else:
                valid_vertices = []

        # Convert vertex objects to indices
        for vertex in valid_vertices:
            try:
                idx = all_vertices.index(vertex)
                if 0 <= idx < 54:
                    mask[idx] = 1.0
            except (ValueError, IndexError):
                continue

        # If no valid vertices, allow all (prevents NaN in softmax)
        if mask.sum() == 0:
            mask[:] = 1.0

        return mask

    def _get_edge_mask(self):
        """
        Create mask for valid edges based on current game state

        Returns:
            np.array of shape [72]: 1 = valid edge, 0 = invalid
        """
        mask = np.zeros(72, dtype=np.float32)

        # Get all edges on the board
        all_edges = self.game_env.game.game_board.edges

        # Determine which edges are valid
        if self.game_env.game.is_initial_placement_phase():
            if self.game_env.game.waiting_for_road:
                # Initial road placement - must connect to last settlement
                last_settlement = self.game_env.game.last_settlement_vertex
                if last_settlement:
                    valid_edges = [e for e in all_edges
                                   if e.structure is None and
                                   (e.vertex1 == last_settlement or e.vertex2 == last_settlement)]
                else:
                    valid_edges = []
            else:
                # Waiting for settlement, no edges needed
                return np.ones(72, dtype=np.float32)  # All ones (will be ignored)
        else:
            # Normal play - buildable roads
            player = self.game_env.game.players[self.player_id]
            current_player = self.game_env.game.get_current_player()

            if player == current_player and self.game_env.game.can_trade_or_build():
                valid_edges = self.game_env.game.get_buildable_edges()
            else:
                valid_edges = []

        # Convert edge objects to indices
        for edge in valid_edges:
            try:
                idx = all_edges.index(edge)
                if 0 <= idx < 72:
                    mask[idx] = 1.0
            except (ValueError, IndexError):
                continue

        # If no valid edges, allow all (prevents NaN in softmax)
        if mask.sum() == 0:
            mask[:] = 1.0

        return mask
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

        # Get location masks for current state
        vertex_mask = self._get_vertex_mask()
        edge_mask = self._get_edge_mask()

        return {
            'observation': observation,
            'action_mask': action_mask,
            'vertex_mask': vertex_mask,  # NEW: Valid vertices
            'edge_mask': edge_mask  # NEW: Valid edges
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

    def step(self, action, vertex_idx=None, edge_idx=None):
        """
        Execute action and return observation

        FIXED: Now properly detects game ending!
        """
        raw_obs = self.game_env.get_observation(self.player_id)

        # If not our turn, return wait action
        if not raw_obs['is_my_turn']:
            obs = self._get_obs()
            obs['action_mask'] = np.zeros(9, dtype=np.int8)
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
            return obs, -10.0, False, False, info

        # Map action index to action name
        action_names = [
            'roll_dice', 'place_settlement', 'place_road',
            'build_settlement', 'build_city', 'build_road',
            'buy_dev_card', 'end_turn', 'wait'
        ]
        action_name = action_names[action]

        # Track old state for reward calculation
        old_dice_rolled = raw_obs.get('dice_rolled', False)

        if action_name in ['build_settlement', 'build_city', 'build_road']:
            current_player = self.game_env.game.players[self.player_id]
            resources = current_player.resources
            print(f"[DEBUG] Player {self.player_id} attempting {action_name}")
            print(f"        Resources: Wood={resources[ResourceType.WOOD]}, "
                  f"Brick={resources[ResourceType.BRICK]}, "
                  f"Wheat={resources[ResourceType.WHEAT]}, "
                  f"Sheep={resources[ResourceType.SHEEP]}, "
                  f"Ore={resources[ResourceType.ORE]}")
        # Get action parameters (if needed)
        action_params = self._get_action_params(action_name, vertex_idx, edge_idx)

        # Execute action in game
        new_obs, done, step_info = self.game_env.step(
            self.player_id,
            action_name,
            action_params
        )

        # ✅ CRITICAL FIX: Check for victory AFTER every action!
        winner = self.game_env.game.check_victory_conditions()

        if winner is not None:
            # Game ended naturally - someone won!
            done = True
            winner_id = self.game_env.game.players.index(winner)
            step_info['winner'] = winner_id
            step_info['result'] = 'game_over'

            # Debug print (remove later if you want)
            winner_vp = winner.calculate_victory_points()
            print(f"\n🏆 GAME END: Player {winner_id} won with {winner_vp} VP!\n")

        # Calculate reward (AFTER checking victory so win bonus applies)
        # Enable debug every 50 episodes
        debug_reward = (hasattr(self, '_episode_count') and self._episode_count % 50 == 0)
        reward = self._calculate_reward(raw_obs, new_obs, step_info, debug=debug_reward)

        # Print reward breakdown for debug episodes
        if debug_reward and hasattr(self, '_last_reward_breakdown'):
            print(f"  [REWARD BREAKDOWN] Total: {reward:.1f}")
            for key, value in self._last_reward_breakdown.items():
                if value != 0:
                    print(f"    {key}: {value:.1f}")

        # Get formatted observation
        obs = self._get_obs()
        info = self._get_info()
        info.update(step_info)

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def _get_action_params(self, action_name, vertex_idx=None, edge_idx=None):
        """
        Get parameters for actions using agent's hierarchical choices

        Args:
            action_name: Name of action ('build_settlement', 'place_road', etc.)
            vertex_idx: Index [0-53] chosen by agent
            edge_idx: Index [0-71] chosen by agent

        Returns:
            Dictionary with 'vertex' or 'edge' key, or None
        """
        all_vertices = self.game_env.game.game_board.vertices
        all_edges = self.game_env.game.game_board.edges

        # === SETTLEMENT ACTIONS ===
        if action_name == 'place_settlement':
            if vertex_idx is not None and 0 <= vertex_idx < len(all_vertices):
                # Use agent's choice!
                return {'vertex': all_vertices[vertex_idx]}
            else:
                # Fallback to random (shouldn't happen)
                vertices = self.game_env.game.game_board.vertices
                player = self.game_env.game.players[self.player_id]
                valid = []
                for v in vertices:
                    if v.structure is None:
                        too_close = any(adj.structure is not None for adj in v.adjacent_vertices)
                        if not too_close:
                            valid.append(v)
                if valid:
                    return {'vertex': np.random.choice(valid)}

        elif action_name == 'build_settlement':
            if vertex_idx is not None and 0 <= vertex_idx < len(all_vertices):
                # Use agent's choice!
                return {'vertex': all_vertices[vertex_idx]}
            else:
                # Fallback
                valid = self.game_env.game.get_buildable_vertices_for_settlements()
                if valid:
                    return {'vertex': np.random.choice(valid)}

        # === CITY ACTIONS ===
        elif action_name == 'build_city':
            if vertex_idx is not None and 0 <= vertex_idx < len(all_vertices):
                # Use agent's choice!
                return {'vertex': all_vertices[vertex_idx]}
            else:
                # Fallback
                valid = self.game_env.game.get_buildable_vertices_for_cities()
                if valid:
                    return {'vertex': np.random.choice(valid)}

        # === ROAD ACTIONS ===
        elif action_name == 'place_road':
            if edge_idx is not None and 0 <= edge_idx < len(all_edges):
                # Use agent's choice!
                return {'edge': all_edges[edge_idx]}
            else:
                # Fallback
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

        elif action_name == 'build_road':
            if edge_idx is not None and 0 <= edge_idx < len(all_edges):
                # Use agent's choice!
                return {'edge': all_edges[edge_idx]}
            else:
                # Fallback
                valid = self.game_env.game.get_buildable_edges()
                if valid:
                    return {'edge': np.random.choice(valid)}

        return None

    def _calculate_reward(self, old_obs, new_obs, step_info, debug=False):
        """
        Reward function with scaled values for stable PPO training

        REBALANCED v2: Fixed pathological hoarding/dev card spam behavior
        - Removed "buildable" reward (was encouraging hoarding)
        - Increased robber penalty 10x + exponential (discourage excess cards)
        - Increased road rewards (0.2 → 0.5)
        - Increased VP scaling (3.0 → 8.0)
        - Increased win bonus (10.0 → 50.0)
        - Differentiated dev card rewards by type
        - Added rewards for USING dev cards
        """
        reward = 0.0
        reward_breakdown = {}  # Track where rewards come from

        # Check game phase
        is_initial = self.game_env.game.is_initial_placement_phase()

        # ===== VICTORY POINTS =====
        vp_diff = new_obs['my_victory_points'] - old_obs['my_victory_points']
        if is_initial:
            vp_reward = vp_diff * 0.01  # Minimal during setup
        else:
            vp_reward = vp_diff * 8.0  # INCREASED: 3.0 → 8.0 (make VP THE priority)
        reward += vp_reward
        reward_breakdown['vp'] = vp_reward

        # ===== BUILDINGS =====
        settlement_diff = new_obs['my_settlements'] - old_obs['my_settlements']
        city_diff = new_obs['my_cities'] - old_obs['my_cities']
        road_diff = new_obs['my_roads'] - old_obs['my_roads']

        if is_initial:
            # Initial placement - small rewards
            building_reward = settlement_diff * 0.005 + road_diff * 0.002
            reward += building_reward
            reward_breakdown['building'] = building_reward

            # Reward good initial placements
            tile_quality_bonus = 0
            if settlement_diff > 0:
                player = self.game_env.game.players[self.player_id]
                if player.settlements:
                    last_settlement = player.settlements[-1]
                    settlement_pos = last_settlement.position

                    # Check adjacent tiles for their numbers
                    for tile in self.game_env.game.game_board.tiles:
                        corners = tile.get_corners()
                        for corner_x, corner_y in corners:
                            if abs(corner_x - settlement_pos.x) < 0.1 and abs(corner_y - settlement_pos.y) < 0.1:
                                if tile.number in [6, 8]:
                                    tile_quality_bonus += 0.3  # Scaled from 30 → 0.3
                                elif tile.number in [5, 9]:
                                    tile_quality_bonus += 0.2  # Scaled from 20 → 0.2
                                elif tile.number in [4, 10]:
                                    tile_quality_bonus += 0.1  # Scaled from 10 → 0.1
                                elif tile.number in [3, 11]:
                                    tile_quality_bonus += 0.05  # Scaled from 5 → 0.05
                                break

            reward += tile_quality_bonus
            reward_breakdown['tile_quality'] = tile_quality_bonus
        else:
            # Normal play - scaled rewards
            building_reward = settlement_diff * 1.0 + city_diff * 2.0 + road_diff * 0.5  # INCREASED: road 0.2 → 0.5
            reward += building_reward
            reward_breakdown['building'] = building_reward

        # ===== RESOURCES =====
        old_resources = sum(old_obs['my_resources'].values())
        new_resources = sum(new_obs['my_resources'].values())
        resource_diff = new_resources - old_resources
        resource_collection_reward = resource_diff * 0.03  # Scaled from 3.0 → 0.03
        reward += resource_collection_reward
        reward_breakdown['resource_collection'] = resource_collection_reward

        # Bonus for resource diversity
        diversity_reward = 0
        if not is_initial:
            from game_system import ResourceType
            res = new_obs['my_resources']
            resource_types_owned = sum([
                1 if res[ResourceType.WOOD] > 0 else 0,
                1 if res[ResourceType.BRICK] > 0 else 0,
                1 if res[ResourceType.WHEAT] > 0 else 0,
                1 if res[ResourceType.SHEEP] > 0 else 0,
                1 if res[ResourceType.ORE] > 0 else 0
            ])
            if resource_types_owned >= 4:
                diversity_reward = 0.15  # Scaled from 15 → 0.15
            elif resource_types_owned >= 3:
                diversity_reward = 0.08  # Scaled from 8 → 0.08
        reward += diversity_reward
        reward_breakdown['diversity'] = diversity_reward

        # REMOVED: "buildable_reward" - was encouraging hoarding instead of building!
        # Agent should get rewarded for BUILDING, not HAVING resources

        # ===== ROBBER RISK =====
        # REBALANCED: Massively increased penalty to prevent hoarding
        total_cards = sum(new_obs['my_resources'].values())
        robber_penalty = 0
        if total_cards > 7:
            excess_cards = total_cards - 7
            # Linear penalty: 1.0 per card (10x stronger than before)
            robber_penalty = 1.0 * excess_cards

            # Exponential penalty for extreme hoarding (10+ excess)
            if excess_cards > 10:
                extreme_excess = excess_cards - 10
                robber_penalty += 2.0 * extreme_excess  # Additional harsh penalty

            # Examples:
            # 8 cards = -1.0
            # 10 cards = -3.0
            # 15 cards = -8.0
            # 20 cards = -13.0 + -6.0 = -19.0 (brutal)
            reward -= robber_penalty
            reward_breakdown['robber_penalty'] = -robber_penalty

        # ===== DEVELOPMENT CARDS =====
        # Differentiate rewards by card type (low rewards - dev cards are a means, not the goal)
        dev_card_reward = 0
        if not is_initial:
            from game_system import DevelopmentCardType
            old_dev = old_obs['my_dev_cards']
            new_dev = new_obs['my_dev_cards']

            # VP cards: Very low immediate reward (VP reward already counted in vp_diff above)
            vp_card_diff = new_dev.get(DevelopmentCardType.VICTORY_POINT, 0) - old_dev.get(DevelopmentCardType.VICTORY_POINT, 0)
            dev_card_reward += vp_card_diff * 0.05  # Very low - VP already counted at 8.0x

            # Knight cards: Low reward (useful for robber control)
            knight_diff = new_dev.get(DevelopmentCardType.KNIGHT, 0) - old_dev.get(DevelopmentCardType.KNIGHT, 0)
            dev_card_reward += knight_diff * 0.2  # Low reward

            # Utility cards: Very low reward (situational value)
            road_building_diff = new_dev.get(DevelopmentCardType.ROAD_BUILDING, 0) - old_dev.get(DevelopmentCardType.ROAD_BUILDING, 0)
            year_of_plenty_diff = new_dev.get(DevelopmentCardType.YEAR_OF_PLENTY, 0) - old_dev.get(DevelopmentCardType.YEAR_OF_PLENTY, 0)
            monopoly_diff = new_dev.get(DevelopmentCardType.MONOPOLY, 0) - old_dev.get(DevelopmentCardType.MONOPOLY, 0)
            dev_card_reward += (road_building_diff + year_of_plenty_diff + monopoly_diff) * 0.1

        reward += dev_card_reward
        reward_breakdown['dev_cards'] = dev_card_reward

        # ===== WIN/LOSS =====
        if step_info.get('result') == 'game_over':
            if step_info.get('winner') == self.player_id:
                reward += 50.0  # INCREASED: 10.0 → 50.0 (WINNING is the ultimate goal!)
                reward_breakdown['win_bonus'] = 50.0
            else:
                reward -= 1.0  # INCREASED: -0.5 → -1.0 (losing hurts more)
                reward_breakdown['loss_penalty'] = -1.0

        # ===== ILLEGAL ACTIONS =====
        if not step_info.get('success', True):
            reward -= 0.1  # Scaled from -5 → -0.1

        # ===== EXPLORATION BONUS =====
        exploration_reward = 0
        current_vp = new_obs['my_victory_points']
        if not is_initial and current_vp > 2:
            exploration_reward += 0.5  # Scaled from 50 → 0.5
        if not is_initial and current_vp >= 4:
            exploration_reward += 1.0  # Scaled from 100 → 1.0
        if not is_initial and current_vp >= 6:
            exploration_reward += 2.0  # NEW: Bonus for getting close to winning
        if not is_initial and current_vp >= 8:
            exploration_reward += 3.0  # NEW: Big bonus for being very close
        reward += exploration_reward
        reward_breakdown['exploration'] = exploration_reward

        # Store breakdown for debugging
        if debug:
            self._last_reward_breakdown = reward_breakdown

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