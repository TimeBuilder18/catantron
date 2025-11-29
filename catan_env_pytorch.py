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

        self.game_env = AIGameEnvironment()
        self._episode_count += 1

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
                return np.ones(72, dtype=np.float32)
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
            obs = self._get_obs()
            info = self._get_info()
            info['illegal_action'] = True
            return obs, -10.0, False, False, info
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
            pre_trade_actions = self.game_env.get_legal_actions(self.player_id)
            resource_map = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT, ResourceType.SHEEP, ResourceType.ORE]
            give_res = resource_map[trade_give_idx]
            get_res = resource_map[trade_get_idx]
            success, message = self.game_env.game.execute_bank_trade(player, give_res, get_res)
            step_info['success'] = success
            step_info['message'] = message
            if success:
                post_trade_actions = self.game_env.get_legal_actions(self.player_id)
                build_actions = {'build_settlement', 'build_city', 'build_road', 'buy_dev_card'}
                pre_buildable = any(a in pre_trade_actions for a in build_actions)
                post_buildable = any(a in post_trade_actions for a in build_actions)
                if post_buildable and not pre_buildable:
                    step_info['trade_led_to_build_opportunity'] = True
            new_obs, done, _ = self.game_env.step(self.player_id, 'wait', {})
        else:
            action_params = self._get_action_params(action_name, vertex_idx, edge_idx)
            new_obs, done, step_info_from_env = self.game_env.step(
                self.player_id, action_name, action_params
            )
            step_info.update(step_info_from_env)

        winner = self.game_env.game.check_victory_conditions()
        if winner is not None:
            done = True
            winner_id = self.game_env.game.players.index(winner)
            step_info['winner'] = winner_id
            step_info['result'] = 'game_over'
        debug_reward = (hasattr(self, '_episode_count') and self._episode_count % 50 == 0)
        reward = self._calculate_reward(raw_obs, new_obs, step_info, debug=debug_reward)
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

    def _calculate_reward(self, old_obs, new_obs, step_info, debug=False):
        reward = 0.0
        reward_breakdown = {}
        is_initial = self.game_env.game.is_initial_placement_phase()
        action_name = step_info.get('action_name')
        if action_name == 'end_turn':
            legal_actions = old_obs.get('legal_actions', [])
            build_actions = {'build_settlement', 'build_city', 'build_road', 'buy_dev_card'}
            if any(action in legal_actions for action in build_actions):
                inaction_penalty = -10.0
                reward += inaction_penalty
                reward_breakdown['inaction_penalty'] = inaction_penalty
        if step_info.get('trade_led_to_build_opportunity'):
            reward += 5.0
            reward_breakdown['strategic_trade'] = 5.0
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
        vp_diff = new_obs['my_victory_points'] - old_obs['my_victory_points']
        if is_initial:
            vp_reward = vp_diff * 0.01
        else:
            vp_reward = vp_diff * 8.0
        reward += vp_reward
        reward_breakdown['vp'] = vp_reward
        settlement_diff = new_obs['my_settlements'] - old_obs['my_settlements']
        city_diff = new_obs['my_cities'] - old_obs['my_cities']
        road_diff = new_obs['my_roads'] - old_obs['my_roads']
        if is_initial:
            building_reward = settlement_diff * 0.005 + road_diff * 0.002
            reward += building_reward
            reward_breakdown['building'] = building_reward
        else:
            building_reward = settlement_diff * 1.0 + city_diff * 2.0 + road_diff * 1.5
            reward += building_reward
            reward_breakdown['building'] = building_reward
        player = self.game_env.game.players[self.player_id]
        if player.has_longest_road and not old_obs.get('has_longest_road'):
            reward += 5.0
            reward_breakdown['longest_road_bonus'] = 5.0
        if player.has_largest_army and not old_obs.get('has_largest_army'):
            reward += 5.0
            reward_breakdown['largest_army_bonus'] = 5.0
        if new_obs['my_victory_points'] > 3:
            total_cards = sum(new_obs['my_resources'].values())
            if total_cards > 7:
                excess_cards = total_cards - 7
                hoarding_penalty = 1.0 * excess_cards
                if excess_cards > 10:
                    hoarding_penalty += 2.0 * (excess_cards - 10)
                reward -= hoarding_penalty
                reward_breakdown['hoarding_penalty'] = -hoarding_penalty
        if step_info.get('result') == 'game_over':
            if step_info.get('winner') == self.player_id:
                reward += 50.0
                reward_breakdown['win_bonus'] = 50.0
            else:
                reward -= 1.0
                reward_breakdown['loss_penalty'] = -1.0
        if debug:
            self._last_reward_breakdown = reward_breakdown
        return reward

    def render(self):
        pass

    def close(self):
        pass
