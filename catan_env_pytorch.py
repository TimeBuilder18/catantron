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
from game_system import DevelopmentCardType, Player

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

    def __init__(self, player_id=0, victory_points_to_win=10):
        """
        Args:
            player_id: Which player this environment controls (0-3)
            victory_points_to_win: VP needed to win (default 10, can be lowered for easier games)
        """
        super().__init__()

        self.player_id = player_id
        self.victory_points_to_win = victory_points_to_win
        self.game_env = AIGameEnvironment(victory_points_to_win=victory_points_to_win)
        self._episode_count = 0  # Track episodes for debug output
        self.gamma = 0.99 # Discount factor for PBRS

        # Track game state for phase-aware rewards
        self._turn_count = 0
        self._bank_trades_this_game = 0
        self._last_city_count = 0
        self._resources_spent_on_trades = 0

        # Action space: 11 discrete actions
        self.action_space = spaces.Discrete(11)

        # Observation space: flat vector + action mask
        obs_size = self._calculate_obs_size()

        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-10, high=100,
                shape=(obs_size,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(11,),
                dtype=np.int8
            )
        })

    def _calculate_obs_size(self):
        """Calculate total observation vector size"""
        size = 0
        size += 11
        size += 5
        size += 3
        size += 5
        size += 4
        size += 18
        size += 57
        size += 18
        return size

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.game_env = AIGameEnvironment(victory_points_to_win=self.victory_points_to_win)
        self._episode_count += 1

        # Reset game state tracking
        self._turn_count = 0
        self._bank_trades_this_game = 0
        self._last_city_count = 0
        self._resources_spent_on_trades = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_vertex_mask(self):
        mask = np.zeros(54, dtype=np.float32)
        all_vertices = self.game_env.game.game_board.vertices
        if self.game_env.game.is_initial_placement_phase():
            if not self.game_env.game.waiting_for_road:
                valid_vertices = []
                for v in all_vertices:
                    if v.structure is None:
                        too_close = any(adj.structure is not None for adj in v.adjacent_vertices)
                        if not too_close:
                            valid_vertices.append(v)
            else:
                return np.ones(54, dtype=np.float32)
        else:
            player = self.game_env.game.players[self.player_id]
            current_player = self.game_env.game.get_current_player()
            if player == current_player and self.game_env.game.can_trade_or_build():
                valid_vertices = self.game_env.game.get_buildable_vertices_for_settlements()
                city_vertices = self.game_env.game.get_buildable_vertices_for_cities()
                valid_vertices = list(set(valid_vertices + city_vertices))
            else:
                valid_vertices = []
        for vertex in valid_vertices:
            try:
                idx = all_vertices.index(vertex)
                if 0 <= idx < 54:
                    mask[idx] = 1.0
            except (ValueError, IndexError):
                continue
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask

    def _get_edge_mask(self):
        mask = np.zeros(72, dtype=np.float32)
        all_edges = self.game_env.game.game_board.edges
        if self.game_env.game.is_initial_placement_phase():
            if self.game_env.game.waiting_for_road:
                last_settlement = self.game_env.game.last_settlement_vertex
                if last_settlement:
                    valid_edges = [e for e in all_edges
                                   if e.structure is None and
                                   (e.vertex1 == last_settlement or e.vertex2 == last_settlement)]
                else:
                    valid_edges = []
            else:
                return np.zeros(72, dtype=np.float32)
        else:
            player = self.game_env.game.players[self.player_id]
            current_player = self.game_env.game.get_current_player()
            if player == current_player and self.game_env.game.can_trade_or_build():
                valid_edges = self.game_env.game.get_buildable_edges()
            else:
                valid_edges = []
        for edge in valid_edges:
            try:
                idx = all_edges.index(edge)
                if 0 <= idx < 72:
                    mask[idx] = 1.0
            except (ValueError, IndexError):
                continue
        if mask.sum() == 0:
            mask[:] = 1.0
        return mask

    def _get_obs(self):
        raw_obs = self.game_env.get_observation(self.player_id)
        features = []
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
            1.0 if self.game_env.game.waiting_for_road else 0.0,
        ])
        resources = raw_obs['my_resources']
        features.extend([
            float(resources[ResourceType.WOOD]),
            float(resources[ResourceType.BRICK]),
            float(resources[ResourceType.WHEAT]),
            float(resources[ResourceType.SHEEP]),
            float(resources[ResourceType.ORE])
        ])
        features.extend([
            float(raw_obs['my_settlements']),
            float(raw_obs['my_cities']),
            float(raw_obs['my_roads'])
        ])
        dev_cards = raw_obs['my_dev_cards']
        features.extend([
            float(dev_cards.get(DevelopmentCardType.KNIGHT, 0)),
            float(dev_cards.get(DevelopmentCardType.VICTORY_POINT, 0)),
            float(dev_cards.get(DevelopmentCardType.ROAD_BUILDING, 0)),
            float(dev_cards.get(DevelopmentCardType.YEAR_OF_PLENTY, 0)),
            float(dev_cards.get(DevelopmentCardType.MONOPOLY, 0))
        ])
        player = self.game_env.game.players[self.player_id]
        features.extend([
            float(raw_obs['my_victory_points']),
            float(player.knights_played),
            1.0 if player.has_longest_road else 0.0,
            1.0 if player.has_largest_army else 0.0
        ])
        for opp in raw_obs['opponents']:
            features.extend([
                float(opp['resource_count']),
                float(opp['settlements']),
                float(opp['cities']),
                float(opp['roads']),
                float(opp['victory_points']),
                float(opp.get('dev_card_count', 0))
            ])
        while len(features) < 10 + 5 + 3 + 5 + 4 + 18:
            features.append(0.0)
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
        port_type_encoding = {
            'GENERIC': 1.0, 'WOOD': 2.0, 'BRICK': 3.0,
            'WHEAT': 4.0, 'SHEEP': 5.0, 'ORE': 6.0
        }
        for port_type, vx, vy in raw_obs['ports']:
            features.extend([
                port_type_encoding.get(port_type, 0.0),
                1.0 if self._player_has_port_access(vx, vy) else 0.0
            ])
        legal_actions = raw_obs['legal_actions']
        action_mask = np.zeros(11, dtype=np.int8)
        if self.game_env.game.is_initial_placement_phase():
            if self.game_env.game.waiting_for_road:
                action_mask[2] = 1
            else:
                action_mask[1] = 1
        else:
            action_map = {
                'roll_dice': 0, 'build_settlement': 3, 'build_city': 4,
                'build_road': 5, 'buy_dev_card': 6, 'end_turn': 7,
                'wait': 8, 'trade_with_bank': 9, 'do_nothing': 10
            }
            for action_name in legal_actions:
                if action_name in action_map:
                    action_mask[action_map[action_name]] = 1
        observation = np.array(features, dtype=np.float32)
        vertex_mask = self._get_vertex_mask()
        edge_mask = self._get_edge_mask()
        return {
            'observation': observation, 'action_mask': action_mask,
            'vertex_mask': vertex_mask, 'edge_mask': edge_mask
        }

    def _tile_has_robber(self, q, r):
        robber_pos = self.game_env.game.robber.position
        if robber_pos:
            return robber_pos.q == q and robber_pos.r == r
        return False

    def _player_has_port_access(self, vx, vy):
        player = self.game_env.game.players[self.player_id]
        for settlement in player.settlements:
            if abs(settlement.position.x - vx) < 0.1 and abs(settlement.position.y - vy) < 0.1:
                return True
        for city in player.cities:
            if abs(city.position.x - vx) < 0.1 and abs(city.position.y - vy) < 0.1:
                return True
        return False

    def _get_info(self):
        raw_obs = self.game_env.get_observation(self.player_id)
        return {
            'player_id': self.player_id,
            'is_my_turn': raw_obs['is_my_turn'],
            'game_phase': raw_obs['game_phase'],
            'victory_points': raw_obs['my_victory_points']
        }

    def step(self, action, vertex_idx=None, edge_idx=None, trade_give_idx=None, trade_get_idx=None):
        raw_obs = self.game_env.get_observation(self.player_id)
        if not raw_obs['is_my_turn']:
            obs = self._get_obs()
            obs['action_mask'] = np.zeros(11, dtype=np.int8)
            obs['action_mask'][8] = 1
            info = self._get_info()
            return obs, 0.0, False, False, info
        current_obs = self._get_obs()
        action_mask = current_obs['action_mask']
        if action_mask[action] == 0:
            # Masked action - PENALIZE to discourage illegal action spam
            obs = self._get_obs()
            info = self._get_info()
            info['illegal_action'] = True
            illegal_penalty = -2.0  # Strong penalty for trying illegal actions
            return obs, illegal_penalty, False, False, info

        old_potential = self._calculate_potential(self.game_env.game.players[self.player_id])

        action_names = [
            'roll_dice', 'place_settlement', 'place_road',
            'build_settlement', 'build_city', 'build_road',
            'buy_dev_card', 'end_turn', 'wait', 'trade_with_bank', 'do_nothing'
        ]
        action_name = action_names[action]
        step_info = {'action_name': action_name}

        if action_name == 'do_nothing':
            new_obs, done, _ = self.game_env.step(self.player_id, 'wait', {})
        elif action_name == 'trade_with_bank':
            player = self.game_env.game.players[self.player_id]
            resource_map = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT, ResourceType.SHEEP, ResourceType.ORE]
            give_res = resource_map[trade_give_idx]
            get_res = resource_map[trade_get_idx]
            success, message = self.game_env.game.execute_bank_trade(player, give_res, get_res)
            step_info['success'] = success
            step_info['message'] = message
            step_info['bank_trade'] = True
            # Track bank trades for efficiency penalty
            if success:
                self._bank_trades_this_game += 1
                self._resources_spent_on_trades += 4  # 4:1 trade ratio
            new_obs, done, _ = self.game_env.step(self.player_id, 'wait', {})
        else:
            action_params = self._get_action_params(action_name, vertex_idx, edge_idx)
            new_obs, done, step_info_from_env = self.game_env.step(
                self.player_id, action_name, action_params
            )
            step_info.update(step_info_from_env)

            # Track turn count for phase awareness
            if action_name == 'end_turn':
                self._turn_count += 1

            # Track city building for rewards - ONLY if build actually succeeded!
            if action_name == 'build_city' and step_info.get('success', False):
                step_info['built_city'] = True

            # Track settlement building for rewards - ONLY if build actually succeeded!
            if action_name == 'build_settlement' and step_info.get('success', False):
                step_info['built_settlement'] = True

        new_potential = self._calculate_potential(self.game_env.game.players[self.player_id])
        winner = self.game_env.game.check_victory_conditions()
        if winner is not None:
            done = True
            winner_id = self.game_env.game.players.index(winner)
            step_info['winner'] = winner_id
            step_info['result'] = 'game_over'

        new_obs = self.game_env.get_observation(self.player_id) # Get final observation after all changes
        debug_reward = (hasattr(self, '_episode_count') and self._episode_count % 50 == 0)
        reward = self._calculate_reward(raw_obs, new_obs, step_info, old_potential, new_potential, debug=debug_reward)
        obs = self._get_obs()
        info = self._get_info()
        info.update(step_info)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def _get_action_params(self, action_name, vertex_idx=None, edge_idx=None):
        all_vertices = self.game_env.game.game_board.vertices
        all_edges = self.game_env.game.game_board.edges
        if action_name in ['place_settlement', 'build_settlement', 'build_city']:
            if vertex_idx is not None and 0 <= vertex_idx < len(all_vertices):
                return {'vertex': all_vertices[vertex_idx]}
        elif action_name in ['place_road', 'build_road']:
            if edge_idx is not None and 0 <= edge_idx < len(all_edges):
                return {'edge': all_edges[edge_idx]}
        return None

    def _calculate_potential(self, player: Player):
        potential = 0.0

        # ========== PRODUCTION POTENTIAL ==========
        pip_map = {2:1, 3:2, 4:3, 5:4, 6:5, 8:5, 9:4, 10:3, 11:2, 12:1}
        resource_type_map = {
            'forest': 'wood', 'hill': 'brick', 'field': 'wheat',
            'mountain': 'ore', 'pasture': 'sheep', 'desert': None
        }

        production_potential = 0
        ore_wheat_production = 0  # Track city-enabling resource production

        for settlement in player.settlements:
            for tile in settlement.position.adjacent_tiles:
                if tile.number:
                    pips = pip_map.get(tile.number, 0)
                    production_potential += pips
                    # Bonus for ore/wheat production (city resources)
                    resource = resource_type_map.get(tile.resource, None)
                    if resource in ('ore', 'wheat'):
                        ore_wheat_production += pips

        for city in player.cities:
            for tile in city.position.adjacent_tiles:
                if tile.number:
                    pips = pip_map.get(tile.number, 0)
                    production_potential += 2 * pips
                    resource = resource_type_map.get(tile.resource, None)
                    if resource in ('ore', 'wheat'):
                        ore_wheat_production += 2 * pips

        potential += production_potential * 1.0
        # Ore/wheat production bonus - encourages city-enabling positions
        potential += ore_wheat_production * 0.3

        # ========== SETTLEMENT BUILDING INCENTIVE (CAPPED AT 5) ==========
        # Settlements are crucial - need them to upgrade to cities!
        # Max 5 settlements per player in Catan
        num_settlements = min(len(player.settlements), 5)  # Cap at 5 (Catan max)
        # Bonus for each settlement beyond starting 2
        # 3rd settlement: +5, 4th: +6, 5th: +7
        if num_settlements > 2:
            extra_settlements = num_settlements - 2
            settlement_bonus = sum(5.0 + 1.0 * i for i in range(extra_settlements))
            potential += settlement_bonus
        # Small bonus for having settlements (encourages not losing them all to cities too fast)
        potential += num_settlements * 2.0

        # ========== SETTLEMENT READINESS BONUS (NEW) ==========
        # Reward for being close to building a settlement (wood, brick, sheep, wheat)
        wood_count = player.resources.get(ResourceType.WOOD, 0)
        brick_count = player.resources.get(ResourceType.BRICK, 0)
        sheep_count = player.resources.get(ResourceType.SHEEP, 0)
        wheat_count_settle = player.resources.get(ResourceType.WHEAT, 0)
        # How close are we to settlement resources?
        wood_ok = min(wood_count, 1)
        brick_ok = min(brick_count, 1)
        sheep_ok = min(sheep_count, 1)
        wheat_ok = min(wheat_count_settle, 1)
        settlement_readiness = (wood_ok + brick_ok + sheep_ok + wheat_ok) / 4.0
        potential += settlement_readiness * 3.0  # Up to +3 when ready to build

        # ========== CITY BUILDING INCENTIVE (CAPPED AT 4) ==========
        # MASSIVE bonus for cities - this is the KEY to winning
        # Max 4 cities per player in Catan
        num_cities = min(len(player.cities), 4)  # Cap at 4 (Catan max)
        # First city: +8, Second: +9, Third: +10, Fourth: +11 (compound bonus)
        city_bonus = sum(8.0 + 1.0 * i for i in range(num_cities))
        potential += city_bonus

        # ========== CITY READINESS BONUS (STRONGER) ==========
        # Reward for being close to building a city
        ore_count = player.resources.get(ResourceType.ORE, 0)
        wheat_count = player.resources.get(ResourceType.WHEAT, 0)
        # How close are we to city resources? (3 ore, 2 wheat needed)
        ore_progress = min(ore_count / 3.0, 1.0)
        wheat_progress = min(wheat_count / 2.0, 1.0)
        # Only give readiness bonus if we have a settlement to upgrade
        if len(player.settlements) > 0:
            city_readiness = ore_progress * wheat_progress  # 0 to 1
            potential += city_readiness * 5.0  # Up to +5 when ready to build (was 2)

        # ========== ROAD VALUE (CAPPED AT 15) ==========
        # Roads are valuable for expansion and longest road!
        # Max 15 roads per player in Catan - cap bonuses there
        num_roads = min(len(player.roads), 15)  # Cap at 15 (Catan max)
        # Bonus for each road - roads enable expansion
        potential += num_roads * 0.3
        # Extra bonus for longest road progress
        if num_roads >= 5:
            potential += 1.0
        if num_roads >= 8:
            potential += 1.0  # Getting close to longest road
        if num_roads >= 10:
            potential += 1.5  # Strong longest road contender
        if num_roads >= 13:
            potential += 2.0  # Near max roads - dominating the board

        # ========== STRATEGIC ASSET POTENTIAL ==========
        if player.has_longest_road: potential += 2.0
        if player.has_largest_army: potential += 2.0
        potential += player.development_cards.get(DevelopmentCardType.VICTORY_POINT, 0) * 1.5

        # ========== DEVELOPMENT CARD BONUS (CAPPED) ==========
        # Knights are valuable for largest army AND robber control
        # 14 knights in deck total, cap bonus at reasonable amount
        knights = min(player.development_cards.get(DevelopmentCardType.KNIGHT, 0), 10)
        potential += knights * 0.5
        # Total dev cards encourage buying (25 in deck total, cap at 15)
        total_dev = min(sum(player.development_cards.values()), 15)
        potential += total_dev * 0.3

        # ========== OPPONENT THREAT POTENTIAL ==========
        for opp in self.game_env.game.players:
            if opp != player:
                opp_vp = opp.calculate_victory_points()
                if opp_vp >= 8:
                    penalty = min((opp_vp - 7) * 2.0, 10.0)
                    potential -= penalty

        # ========== EXCESSIVE TRADING PENALTY (PBRS) ==========
        # Penalize states where agent has wasted resources on trades
        # This shapes long-term behavior to avoid trading addiction
        if self._bank_trades_this_game > 3:
            trade_waste_penalty = 0.5 * (self._bank_trades_this_game - 3)
            potential -= trade_waste_penalty

        return potential

    def _calculate_reward(self, old_obs, new_obs, step_info, old_potential, new_potential, debug=False):
        reward = 0.0
        reward_breakdown = {}
        is_initial = self.game_env.game.is_initial_placement_phase()
        action_name = step_info.get('action_name')

        # ========== PBRS REWARD ==========
        pbrs_reward = self.gamma * new_potential - old_potential
        reward += pbrs_reward
        reward_breakdown['pbrs'] = pbrs_reward

        # ========== VP REWARDS ==========
        vp_diff = new_obs['my_victory_points'] - old_obs['my_victory_points']
        if is_initial:
            vp_reward = vp_diff * 0.01
        else:
            vp_reward = vp_diff * 3.0
        reward += vp_reward
        reward_breakdown['vp'] = vp_reward

        # VP State Bonus (encourages maintaining high VP)
        vp_state_bonus = new_obs['my_victory_points'] * 0.1
        reward += vp_state_bonus
        reward_breakdown['vp_state_bonus'] = vp_state_bonus

        # ========== CITY BUILDING BONUS (CRITICAL FIX v2) ==========
        # MASSIVELY increased reward - cities are THE key to winning
        if step_info.get('built_city') or (action_name == 'build_city' and vp_diff > 0):
            # Determine game phase for phase-aware bonus
            turn = self._turn_count
            if turn < 15:
                # Early game: still reward cities (don't wait!)
                phase_multiplier = 1.5
            elif turn < 40:
                # Mid game: HUGE city bonus - this is when cities matter most!
                phase_multiplier = 2.0
            else:
                # Late game: still very valuable
                phase_multiplier = 1.5

            # Base city bonus TRIPLED: 15.0 instead of 5.0
            city_bonus = 15.0 * phase_multiplier
            reward += city_bonus
            reward_breakdown['city_bonus'] = city_bonus

        # ========== SETTLEMENT BUILDING BONUS (NEW) ==========
        # Settlements are critical - you need them to upgrade to cities!
        if step_info.get('built_settlement') or (action_name == 'build_settlement' and vp_diff > 0):
            # Each new settlement expands your resource generation AND gives upgrade potential
            num_settlements = new_obs.get('my_settlements', 0)
            # Higher bonus for 3rd, 4th, 5th settlement (beyond starting 2)
            settlement_bonus = 8.0  # Strong base bonus
            if num_settlements > 2:
                settlement_bonus += 3.0 * (num_settlements - 2)  # Extra for expansion
            reward += settlement_bonus
            reward_breakdown['settlement_bonus'] = settlement_bonus

        # ========== BANK TRADE PENALTY (BALANCED) ==========
        # Discourage excessive trading but allow strategic trades
        # Some trades are necessary, especially early game
        if step_info.get('bank_trade') and step_info.get('success'):
            trades_so_far = self._bank_trades_this_game
            # Small base penalty - trading is inefficient but sometimes needed
            trade_penalty = 0.5
            # Only escalate after many trades (>8 is excessive)
            if trades_so_far > 8:
                trade_penalty += 0.5 * (trades_so_far - 8)
            # Strong penalty after 15 trades - clearly inefficient play
            if trades_so_far > 15:
                trade_penalty += 1.0 * (trades_so_far - 15)

            reward -= trade_penalty
            reward_breakdown['bank_trade_penalty'] = -trade_penalty

        # ========== INACTION PENALTY (STRONGER) ==========
        if action_name == 'end_turn':
            legal_actions = old_obs.get('legal_actions', [])
            build_actions = {'build_settlement', 'build_city', 'build_road', 'buy_dev_card'}
            if any(action in legal_actions for action in build_actions):
                # MASSIVE penalty if city was available but not built
                if 'build_city' in legal_actions:
                    inaction_penalty = -15.0  # HUGE penalty - you MUST build cities!
                elif 'build_settlement' in legal_actions:
                    inaction_penalty = -5.0  # Strong penalty for not building settlement
                else:
                    inaction_penalty = -1.0  # Small penalty for other builds
                reward += inaction_penalty
                reward_breakdown['inaction_penalty'] = inaction_penalty

        # Roads are valuable - no penalty for building them
        # Agent will learn to prioritize through VP rewards

        # ========== STRATEGIC TRADE BONUS ==========
        if step_info.get('trade_led_to_build_opportunity'):
            reward += 5.0
            reward_breakdown['strategic_trade'] = 5.0

        # ========== DISCARD PENALTY (7 ROLLED) ==========
        was_seven_rolled = new_obs.get('last_roll') and new_obs['last_roll'][2] == 7
        if was_seven_rolled:
            old_card_count = sum(old_obs['my_resources'].values())
            if old_card_count > 7:
                new_card_count = sum(new_obs['my_resources'].values())
                cards_discarded = old_card_count - new_card_count
                if cards_discarded > 0:
                    discard_event_penalty = -2.0 * cards_discarded
                    reward += discard_event_penalty
                    reward_breakdown['discard_event_penalty'] = discard_event_penalty

        # ========== RESOURCE HOARDING PENALTY ==========
        # Diminishing returns on holding resources
        total_cards = sum(new_obs['my_resources'].values())
        if total_cards > 10:
            # Penalize hoarding beyond 10 cards (risky for 7s anyway)
            excess_cards = total_cards - 10
            hoarding_penalty = 0.2 * excess_cards
            if excess_cards > 5:
                hoarding_penalty += 0.3 * (excess_cards - 5)
            reward -= hoarding_penalty
            reward_breakdown['hoarding_penalty'] = -hoarding_penalty

        # ========== GAME END REWARDS ==========
        if step_info.get('result') == 'game_over':
            if step_info.get('winner') == self.player_id:
                # Win bonus scales with VP (higher VP wins are better)
                final_vp = new_obs['my_victory_points']
                win_bonus = 15.0 + final_vp * 1.0  # e.g., 10VP win = +25
                reward += win_bonus
                reward_breakdown['win_bonus'] = win_bonus
            else:
                # Small loss penalty
                reward -= 1.0
                reward_breakdown['loss_penalty'] = -1.0

        if debug:
            self._last_reward_breakdown = reward_breakdown

        # Clip reward to prevent extreme values that destabilize training
        # Range allows for strong signals while preventing explosions
        reward = np.clip(reward, -20.0, 20.0)

        return reward

    def render(self):
        pass

    def close(self):
        pass
