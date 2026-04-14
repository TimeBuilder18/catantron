"""
Gymnasium environment wrapper for 1v1 Catan. Converts the game state into
a flat 575-dimensional observation vector that the neural network processes,
and handles the hierarchical action space (14 action types, each optionally
needing a vertex/edge/trade selection).

The hardest part was getting the action masking right — Catan has different
action types and some need location info, so we use separate masks for
actions, vertices, and edges. Without proper masking the agent wastes
forever trying illegal moves.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from environment.ai_interface import AIGameEnvironment
from game.game_system import ResourceType, PortType
from game.game_system import DevelopmentCardType, Player
from game.game_system import Settlement, City, Road, DevelopmentCardDeck

# Dice pip counts (probability weight) for each number token
DICE_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# One-hot resource encoding for tiles
RESOURCE_ONEHOT = {
    'forest':   [1, 0, 0, 0, 0, 0],
    'hill':     [0, 1, 0, 0, 0, 0],
    'field':    [0, 0, 1, 0, 0, 0],
    'mountain': [0, 0, 0, 1, 0, 0],
    'pasture':  [0, 0, 0, 0, 1, 0],
    'desert':   [0, 0, 0, 0, 0, 1],
}

# Tile resource string -> ResourceType mapping (skip desert)
RESOURCE_STR_TO_TYPE = {rt.value: rt for rt in ResourceType}

class CatanEnv(gym.Env):
    """
    Gymnasium environment for Catan - optimized for PyTorch PPO

    Key features:
    - Flat observation vector for neural network
    - Action masking for invalid actions
    - Proper reward shaping
    - Supports 2-4 players (2 for 1v1 training, 4 for standard)
    """

    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(self, player_id=0, victory_points_to_win=10, num_players=4):
        """
        Args:
            player_id: Which player this environment controls (0 for learning agent)
            victory_points_to_win: VP needed to win (default 10, can be lowered for easier games)
            num_players: Number of players (2 for 1v1, 4 for standard)
        """
        super().__init__()

        self.player_id = player_id
        self.victory_points_to_win = victory_points_to_win
        self.num_players = num_players
        self.game_env = AIGameEnvironment(victory_points_to_win=victory_points_to_win, num_players=num_players)
        self._episode_count = 0  # Track episodes for debug output
        self.gamma = 0.99 # Discount factor for PBRS

        # Track game state for phase-aware rewards
        self._turn_count = 0
        self._bank_trades_this_game = 0
        self._last_city_count = 0
        self._resources_spent_on_trades = 0
        self._actions_this_turn = 0

        # PERFORMANCE: Cache for port access checks (cleared on settlement/city build)
        self._port_access_cache = None
        self._port_access_cache_turn = -1

        # Action space: 11 discrete actions
        self.action_space = spaces.Discrete(14)

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
        # Non-tile features (61):
        #   game state (11) + resources (5) + structures (3) + dev cards (5)
        #   + VP/stats (4) + opponents (18) + strategic (15)
        # Tile features (190): 19 tiles × 10 features
        # Port features (18): 9 ports × 2 features
        size = 11 + 5 + 3 + 5 + 4 + 18 + 15 + 190 + 18  # = 269

        # Positional features (306):
        #   my_settlements (54) + my_cities (54) + opp_buildings (54)
        #   + my_roads (72) + opp_roads (72)
        size += 54 + 54 + 54 + 72 + 72  # = 306

        return size  # = 575

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.game_env = AIGameEnvironment(victory_points_to_win=self.victory_points_to_win, num_players=self.num_players)
        self._episode_count += 1

        # Reset game state tracking
        self._turn_count = 0
        self._bank_trades_this_game = 0
        self._last_city_count = 0
        self._resources_spent_on_trades = 0
        self._actions_this_turn = 0

        # Reset port access cache for new game
        self._port_access_cache = None
        self._port_access_cache_turn = None

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _get_vertex_mask(self):
        """Get mask for valid vertex positions.

        Returns separate masks for settlements vs cities since they need different vertices.
        """
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
                # Get BOTH types but store separately for action-aware selection
                self._settlement_vertices = self.game_env.game.get_buildable_vertices_for_settlements()
                self._city_vertices = self.game_env.game.get_buildable_vertices_for_cities()
                # Combine for mask (network sees all options)
                valid_vertices = list(set(self._settlement_vertices + self._city_vertices))
            else:
                valid_vertices = []
                self._settlement_vertices = []
                self._city_vertices = []
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
                # Not waiting for road during initial placement - no valid edges
                # Don't early return - let it flow through to the fallback logic
                valid_edges = []
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
        """Build the 575-dim observation vector. It's a flat array with sections:
        [0:11]    game state (phase, turn, dice)
        [11:16]   my resources (wood, brick, wheat, sheep, ore)
        [16:19]   my structure counts
        [19:24]   my dev cards
        [24:28]   VP and stats
        [28:46]   opponent info (padded for multi-player compat)
        [46:61]   strategic features (port access, longest road, etc)
        [61:251]  tile features (19 tiles * 10 features each)
        [251:269] port features
        [269:575] positional features (54 vertices * ~5-6 features each)
        """
        raw_obs = self.game_env.get_observation(self.player_id)
        features = []
        # Game state features — what phase/turn are we in?
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
        # Pad opponent features to always have 3 opponents worth (18 features)
        # This ensures consistent observation size for both 1v1 and 4-player
        # 11 (game) + 5 (resources) + 3 (structures) + 5 (dev cards) + 4 (VP) + 18 (3 opponents) = 46
        while len(features) < 11 + 5 + 3 + 5 + 4 + 18:
            features.append(0.0)

        # === Strategic features (15) ===
        # Skip for legacy self-play opponents — these get discarded by the
        # 575→427 adapter anyway, and calculate_longest_road is expensive.
        if getattr(self, '_skip_strategic', False):
            features.extend([0.0] * 15)
        else:
            game = self.game_env.game

            # Per-resource pip production rate (5 features)
            resource_pips = {rt: 0 for rt in ResourceType}
            for settlement in player.settlements:
                for tile in settlement.position.adjacent_tiles:
                    rt = RESOURCE_STR_TO_TYPE.get(tile.resource)
                    if rt and tile.number and not tile.has_robber:
                        resource_pips[rt] += DICE_PIPS.get(tile.number, 0)
            for city in player.cities:
                for tile in city.position.adjacent_tiles:
                    rt = RESOURCE_STR_TO_TYPE.get(tile.resource)
                    if rt and tile.number and not tile.has_robber:
                        resource_pips[rt] += DICE_PIPS.get(tile.number, 0) * 2
            for rt in [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT,
                       ResourceType.SHEEP, ResourceType.ORE]:
                features.append(resource_pips[rt] / 10.0)

            # Turn number (1 feature)
            features.append(self._turn_count / 200.0)

            # My longest road length (1 feature)
            my_road_length = game.calculate_longest_road_for_player(player)
            features.append(my_road_length / 15.0)

            # Opponent max longest road (1 feature)
            opp_max_road = 0
            for i, p in enumerate(game.players):
                if i != self.player_id:
                    opp_max_road = max(opp_max_road, game.calculate_longest_road_for_player(p))
            features.append(opp_max_road / 15.0)

            # Opponent max knights played (1 feature)
            opp_max_knights = max((p.knights_played for i, p in enumerate(game.players) if i != self.player_id), default=0)
            features.append(opp_max_knights / 5.0)

            # VP gap (1 feature)
            opp_max_vp = max((p.calculate_victory_points() for i, p in enumerate(game.players) if i != self.player_id), default=0)
            features.append((raw_obs['my_victory_points'] - opp_max_vp) / 10.0)

            # Cards in hand total (1 feature)
            features.append(sum(player.resources.values()) / 15.0)

            # Can afford checks (4 features)
            features.append(1.0 if player.can_afford(Settlement.get_cost()) else 0.0)
            features.append(1.0 if player.can_afford(City.get_cost()) else 0.0)
            features.append(1.0 if player.can_afford(Road.get_cost()) else 0.0)
            features.append(1.0 if player.can_afford(DevelopmentCardDeck.get_cost()) else 0.0)

        # === Tile features (19 tiles × 10 = 190) ===
        # Pre-compute building strength per tile
        tile_building_strength = {}
        for settlement in player.settlements:
            for tile in settlement.position.adjacent_tiles:
                key = (tile.q, tile.r)
                tile_building_strength[key] = tile_building_strength.get(key, 0) + 1
        for city in player.cities:
            for tile in city.position.adjacent_tiles:
                key = (tile.q, tile.r)
                tile_building_strength[key] = tile_building_strength.get(key, 0) + 2

        for q, r, resource, number in raw_obs['tiles']:
            features.extend(RESOURCE_ONEHOT.get(resource, [0, 0, 0, 0, 0, 0]))
            features.append(float(number) / 12.0 if number else 0.0)
            features.append(DICE_PIPS.get(number, 0) / 5.0)
            features.append(1.0 if self._tile_has_robber(q, r) else 0.0)
            features.append(tile_building_strength.get((q, r), 0) / 2.0)
        port_type_encoding = {
            'GENERIC': 1.0, 'WOOD': 2.0, 'BRICK': 3.0,
            'WHEAT': 4.0, 'SHEEP': 5.0, 'ORE': 6.0
        }
        for port_type, vx, vy in raw_obs['ports']:
            features.extend([
                port_type_encoding.get(port_type, 0.0),
                1.0 if self._player_has_port_access(vx, vy) else 0.0
            ])

        # === POSITIONAL FEATURES (critical for strategic play) ===
        # These tell the model WHERE buildings are, not just HOW MANY
        all_vertices = self.game_env.game.game_board.vertices
        all_edges = self.game_env.game.game_board.edges
        player = self.game_env.game.players[self.player_id]

        # My settlement positions (54 binary features)
        my_settlement_set = set(player.settlements)
        for vertex in all_vertices:
            features.append(1.0 if vertex in my_settlement_set else 0.0)

        # My city positions (54 binary features)
        my_city_set = set(player.cities)
        for vertex in all_vertices:
            features.append(1.0 if vertex in my_city_set else 0.0)

        # Opponent building positions (54 binary features - any opponent)
        opp_buildings = set()
        for i, p in enumerate(self.game_env.game.players):
            if i != self.player_id:
                opp_buildings.update(p.settlements)
                opp_buildings.update(p.cities)
        for vertex in all_vertices:
            features.append(1.0 if vertex in opp_buildings else 0.0)

        # My road positions (72 binary features)
        my_road_set = set(player.roads)
        for edge in all_edges:
            features.append(1.0 if edge in my_road_set else 0.0)

        # Opponent road positions (72 binary features)
        opp_roads = set()
        for i, p in enumerate(self.game_env.game.players):
            if i != self.player_id:
                opp_roads.update(p.roads)
        for edge in all_edges:
            features.append(1.0 if edge in opp_roads else 0.0)

        legal_actions = raw_obs['legal_actions']
        action_mask = np.zeros(14, dtype=np.int8)
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
            # Dev card play actions (not in legal_actions from game_env, computed manually)
            # Respect real Catan rules: can't play a card bought this turn; only one per turn
            player = self.game_env.game.players[self.player_id]
            game = self.game_env.game
            can_play_dev = (
                game.can_trade_or_build() and
                not game.dev_card_bought_this_turn and
                not game.dev_card_played_this_turn
            )
            if can_play_dev:
                if player.development_cards.get(DevelopmentCardType.KNIGHT, 0) > 0:
                    action_mask[11] = 1  # play_knight
                if player.development_cards.get(DevelopmentCardType.MONOPOLY, 0) > 0:
                    action_mask[12] = 1  # play_monopoly
                if player.development_cards.get(DevelopmentCardType.YEAR_OF_PLENTY, 0) > 0:
                    action_mask[13] = 1  # play_year_of_plenty
        observation = np.array(features, dtype=np.float32)
        vertex_mask = self._get_vertex_mask()
        edge_mask = self._get_edge_mask()
        return {
            'observation': observation, 'action_mask': action_mask,
            'vertex_mask': vertex_mask, 'edge_mask': edge_mask,
            'legal_actions': legal_actions,  # CRITICAL: Needed for inaction penalty!
            'my_resources': raw_obs['my_resources'],  # For reward calculations
            'my_victory_points': raw_obs['my_victory_points'],
            'my_settlements': raw_obs['my_settlements'],
            'my_roads': raw_obs['my_roads']
        }

    def _tile_has_robber(self, q, r):
        robber_pos = self.game_env.game.robber.position
        if robber_pos:
            return robber_pos.q == q and robber_pos.r == r
        return False

    def _player_has_port_access(self, vx, vy):
        """Check if player has access to port at (vx, vy). Uses caching for performance."""
        # PERFORMANCE: Build port access cache, invalidate when buildings change
        player = self.game_env.game.players[self.player_id]
        current_building_count = len(player.settlements) + len(player.cities)

        # Check if cache needs rebuild (buildings changed)
        cache_key = (self._turn_count, current_building_count)
        if self._port_access_cache is None or self._port_access_cache_turn != cache_key:
            self._port_access_cache = set()
            # Pre-compute all accessible positions
            for settlement in player.settlements:
                # Round to 1 decimal for consistent lookup
                pos_key = (round(settlement.position.x, 1), round(settlement.position.y, 1))
                self._port_access_cache.add(pos_key)
            for city in player.cities:
                pos_key = (round(city.position.x, 1), round(city.position.y, 1))
                self._port_access_cache.add(pos_key)
            self._port_access_cache_turn = cache_key

        # O(1) lookup instead of O(settlements + cities)
        port_key = (round(vx, 1), round(vy, 1))
        return port_key in self._port_access_cache

    def _get_info(self, raw_obs=None):
        """Get info dict. Optionally accepts cached raw_obs to avoid redundant calls."""
        if raw_obs is None:
            raw_obs = self.game_env.get_observation(self.player_id)
        return {
            'player_id': self.player_id,
            'is_my_turn': raw_obs['is_my_turn'],
            'game_phase': raw_obs['game_phase'],
            'victory_points': raw_obs['my_victory_points']
        }

    def step(self, action, vertex_idx=None, edge_idx=None, trade_give_idx=None, trade_get_idx=None):
        # PERFORMANCE FIX: Cache observation to avoid redundant calls (was 6+ calls, now 2)
        raw_obs = self.game_env.get_observation(self.player_id)

        if not raw_obs['is_my_turn']:
            obs = self._get_obs()
            obs['action_mask'] = np.zeros(14, dtype=np.int8)
            obs['action_mask'][8] = 1
            info = self._get_info(raw_obs)  # Reuse cached raw_obs
            return obs, 0.0, False, False, info

        # Build observation once, reuse for action mask check
        current_obs = self._get_obs()
        action_mask = current_obs['action_mask']

        if action_mask[action] == 0:
            # Masked action - PENALIZE to discourage illegal action spam
            info = self._get_info(raw_obs)  # Reuse cached raw_obs
            info['illegal_action'] = True
            illegal_penalty = -5.0  # Strong penalty for trying illegal actions
            return current_obs, illegal_penalty, False, False, info  # Reuse current_obs

        old_potential = self._calculate_potential(self.game_env.game.players[self.player_id])

        action_names = [
            'roll_dice', 'place_settlement', 'place_road',
            'build_settlement', 'build_city', 'build_road',
            'buy_dev_card', 'end_turn', 'wait', 'trade_with_bank', 'do_nothing',
            'play_knight', 'play_monopoly', 'play_year_of_plenty'
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
            # Get actual trade ratio (2:1 with specific port, 3:1 generic, 4:1 no port)
            actual_ratio = self.game_env.game.game_board.get_best_trade_ratio(player, give_res)
            success, message = self.game_env.game.execute_bank_trade(player, give_res, get_res)
            step_info['success'] = success
            step_info['message'] = message
            step_info['bank_trade'] = True
            step_info['trade_ratio'] = actual_ratio
            # Track bank trades for efficiency penalty
            if success:
                self._bank_trades_this_game += 1
                self._resources_spent_on_trades += actual_ratio
            new_obs, done, _ = self.game_env.step(self.player_id, 'wait', {})
        elif action_name == 'play_knight':
            import random
            player = self.game_env.game.players[self.player_id]
            # Snapshot resources before steal to detect "completed a build"
            res_before = dict(player.resources)
            success, message = self.game_env.game.play_knight_card(player)
            step_info['success'] = success
            step_info['message'] = message
            if success:
                # Move robber to best opponent hex
                best_tile = self.game_env._find_best_robber_tile(self.player_id)
                if best_tile:
                    self.game_env.game.move_robber_to_tile(best_tile)
                    opponents_on_tile = []
                    for vertex in self.game_env.game.game_board.vertices:
                        if best_tile in vertex.adjacent_tiles and vertex.structure:
                            owner = vertex.structure.player
                            if owner != player:
                                opponents_on_tile.append(owner)
                    if opponents_on_tile:
                        victim = max(opponents_on_tile, key=lambda p: sum(p.resources.values()))
                        stealable = [r for r, amt in victim.resources.items() if amt > 0]
                        if stealable:
                            stolen = random.choice(stealable)
                            victim.resources[stolen] -= 1
                            player.resources[stolen] = player.resources.get(stolen, 0) + 1
                            step_info['stolen_resource'] = str(stolen)
                else:
                    available = [t for t in self.game_env.game.game_board.tiles
                                 if t != self.game_env.game.robber.position and t.resource is not None]
                    if available:
                        self.game_env.game.move_robber_to_tile(random.choice(available))

                # Always steal: if no steal from tile, steal from richest opponent globally
                if not step_info.get('stolen_resource'):
                    all_opponents = [p for p in self.game_env.game.players if p != player]
                    if all_opponents:
                        richest = max(all_opponents, key=lambda p: sum(p.resources.values()))
                        stealable = [r for r, amt in richest.resources.items() if amt > 0]
                        if stealable:
                            stolen = random.choice(stealable)
                            richest.resources[stolen] -= 1
                            player.resources[stolen] = player.resources.get(stolen, 0) + 1
                            step_info['stolen_resource'] = str(stolen)

                # Check if stolen resource completed a build the player was one card short of
                if step_info.get('stolen_resource'):
                    stolen_type = None
                    for rt in ResourceType:
                        if str(rt) == step_info['stolen_resource'] or rt.value == step_info['stolen_resource'].lower().replace('resourcetype.', ''):
                            stolen_type = rt
                            break
                    if stolen_type is not None:
                        build_costs = [
                            {ResourceType.ORE: 3, ResourceType.WHEAT: 2},                          # city
                            {ResourceType.WOOD: 1, ResourceType.BRICK: 1,
                             ResourceType.WHEAT: 1, ResourceType.SHEEP: 1},                       # settlement
                            {ResourceType.ORE: 1, ResourceType.WHEAT: 1, ResourceType.SHEEP: 1},  # dev card
                            {ResourceType.WOOD: 1, ResourceType.BRICK: 1},                        # road
                        ]
                        res_after = player.resources
                        for cost in build_costs:
                            # Can afford after steal but couldn't before
                            can_now = all(res_after.get(r, 0) >= amt for r, amt in cost.items())
                            could_before = all(res_before.get(r, 0) >= amt for r, amt in cost.items())
                            if can_now and not could_before:
                                step_info['steal_completes_build'] = True
                                break

            new_obs, done, _ = self.game_env.step(self.player_id, 'wait', {})
        elif action_name == 'play_monopoly':
            player = self.game_env.game.players[self.player_id]
            resource_map = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT,
                            ResourceType.SHEEP, ResourceType.ORE]
            resource = resource_map[trade_give_idx] if trade_give_idx is not None else ResourceType.ORE
            success, message = self.game_env.game.play_monopoly_card(player, resource)
            step_info['success'] = success
            step_info['message'] = message
            # Parse stolen count from message: "Monopoly: Stole 5 wood from other players"
            if success:
                try:
                    stolen_count = int(message.split('Stole ')[1].split(' ')[0])
                    step_info['stolen_count'] = stolen_count
                except (IndexError, ValueError):
                    step_info['stolen_count'] = 0
            new_obs, done, _ = self.game_env.step(self.player_id, 'wait', {})
        elif action_name == 'play_year_of_plenty':
            player = self.game_env.game.players[self.player_id]
            resource_map = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT,
                            ResourceType.SHEEP, ResourceType.ORE]
            res1 = resource_map[trade_give_idx] if trade_give_idx is not None else ResourceType.ORE
            res2 = resource_map[trade_get_idx] if trade_get_idx is not None else ResourceType.WHEAT
            success, message = self.game_env.game.play_year_of_plenty_card(player, res1, res2)
            step_info['success'] = success
            step_info['message'] = message
            new_obs, done, _ = self.game_env.step(self.player_id, 'wait', {})
        else:
            action_params = self._get_action_params(action_name, vertex_idx, edge_idx)
            new_obs, done, step_info_from_env = self.game_env.step(
                self.player_id, action_name, action_params
            )
            step_info.update(step_info_from_env)

            # Track actions and turns
            self._actions_this_turn += 1
            if action_name == 'end_turn':
                self._turn_count += 1
                self._actions_this_turn = 0

            # Track city building for rewards - ONLY if build actually succeeded!
            if action_name == 'build_city' and step_info.get('success', False):
                step_info['built_city'] = True

            # Track settlement building for rewards - ONLY if build actually succeeded!
            if action_name == 'build_settlement' and step_info.get('success', False):
                step_info['built_settlement'] = True

            # Detect VP dev card purchase: VP cards auto-apply on buy ("Bought Victory Point")
            if action_name == 'buy_dev_card' and step_info.get('success', False):
                msg = step_info.get('message', '')
                if 'Victory Point' in msg or 'victory_point' in msg.lower():
                    step_info['got_vp_card'] = True

        new_potential = self._calculate_potential(self.game_env.game.players[self.player_id])
        winner = self.game_env.game.check_victory_conditions()
        if winner is not None:
            done = True
            winner_id = self.game_env.game.players.index(winner)
            step_info['winner'] = winner_id
            step_info['result'] = 'game_over'

        # PERFORMANCE FIX: Get final observation once, reuse for reward calc and return
        new_obs = self.game_env.get_observation(self.player_id)
        debug_reward = (hasattr(self, '_episode_count') and self._episode_count % 50 == 0)
        reward = self._calculate_reward(raw_obs, new_obs, step_info, old_potential, new_potential, debug=debug_reward)

        # Build final obs (includes masks) - single call instead of separate _get_obs() + _get_info()
        obs = self._get_obs()
        info = self._get_info(new_obs)  # Reuse new_obs instead of another get_observation call
        info.update(step_info)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def _get_action_params(self, action_name, vertex_idx=None, edge_idx=None):
        """Get action parameters, ensuring valid vertex/edge selection.

        CRITICAL FIX: Select from action-appropriate valid positions.
        - build_settlement: must use settlement-valid vertices
        - build_city: must use city-valid vertices (existing settlements)
        """
        import random
        all_vertices = self.game_env.game.game_board.vertices
        all_edges = self.game_env.game.game_board.edges

        if action_name == 'place_settlement':
            # Initial placement - must use valid vertex
            # Get valid vertices for initial settlement (no structure, not adjacent to other settlements)
            valid = [v for v in all_vertices if v.structure is None and
                     not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                # Try network's choice first if it's valid
                if vertex_idx is not None and 0 <= vertex_idx < len(all_vertices):
                    chosen = all_vertices[vertex_idx]
                    if chosen in valid:
                        return {'vertex': chosen}
                # Fallback: use strategic placement for CITY BUILDING
                # Prioritize ORE + WHEAT tiles since cities need 3 ore + 2 wheat
                def pip_count(number):
                    """Convert dice number to pip count (probability dots)."""
                    if number is None:
                        return 0
                    return max(0, 6 - abs(7 - number))

                def resource_weight(resource):
                    """Weight resources for balanced play (cities + settlements + roads).

                    10 VP strategy needs:
                    - Cities: 3 ore + 2 wheat (most VP efficient)
                    - Settlements: wood + brick + sheep + wheat
                    - Roads: wood + brick (for longest road + expansion)
                    - Dev cards: ore + wheat + sheep (largest army + VP cards)
                    """
                    if resource == 'ore':
                        return 1.5  # Cities + dev cards
                    elif resource == 'wheat':
                        return 1.5  # Used in everything except roads
                    elif resource == 'brick':
                        return 1.3  # Settlements + roads (expansion)
                    elif resource == 'wood':
                        return 1.3  # Settlements + roads (expansion)
                    elif resource == 'sheep':
                        return 1.0  # Settlements + dev cards
                    else:
                        return 0.0  # Desert

                def vertex_score(v):
                    """Score vertex by weighted pip count (ore/wheat prioritized)."""
                    score = 0
                    for t in v.adjacent_tiles:
                        if hasattr(t, 'number') and t.number and hasattr(t, 'resource'):
                            pips = pip_count(t.number)
                            weight = resource_weight(t.resource)
                            score += pips * weight
                    return score

                # Sort by strategic score (ore/wheat access with good probability)
                valid.sort(key=vertex_score, reverse=True)
                return {'vertex': valid[0]}
            return None

        elif action_name == 'build_settlement':
            # FIXED: Only use settlement-valid vertices
            valid = getattr(self, '_settlement_vertices', [])
            if valid:
                # Try network's choice first if it's valid
                if vertex_idx is not None and 0 <= vertex_idx < len(all_vertices):
                    chosen = all_vertices[vertex_idx]
                    if chosen in valid:
                        return {'vertex': chosen}
                # Fallback: pick best valid vertex (prefer network's preference direction)
                return {'vertex': random.choice(valid)}
            return None

        elif action_name == 'build_city':
            # FIXED: Only use city-valid vertices (existing settlements)
            valid = getattr(self, '_city_vertices', [])
            if valid:
                if vertex_idx is not None and 0 <= vertex_idx < len(all_vertices):
                    chosen = all_vertices[vertex_idx]
                    if chosen in valid:
                        return {'vertex': chosen}
                return {'vertex': random.choice(valid)}
            return None

        elif action_name in ['place_road', 'build_road']:
            # Get valid edges for roads
            if action_name == 'place_road' and self.game_env.game.waiting_for_road:
                last_settlement = self.game_env.game.last_settlement_vertex
                if last_settlement:
                    valid = [e for e in all_edges
                             if e.structure is None and
                             (e.vertex1 == last_settlement or e.vertex2 == last_settlement)]
                else:
                    valid = []
            else:
                valid = self.game_env.game.get_buildable_edges()

            if valid:
                if edge_idx is not None and 0 <= edge_idx < len(all_edges):
                    chosen = all_edges[edge_idx]
                    if chosen in valid:
                        return {'edge': chosen}
                return {'edge': random.choice(valid)}
            return None

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
        # 3rd settlement: +15, 4th: +20, 5th: +25 (scaled up to compete with city bonus ~20+)
        if num_settlements > 2:
            extra_settlements = num_settlements - 2
            settlement_bonus = sum(15.0 + 5.0 * i for i in range(extra_settlements))
            potential += settlement_bonus
        # Small bonus for having settlements (encourages not losing them all to cities too fast)
        potential += num_settlements * 2.0
        # PENALTY: being stuck with zero settlements and <10VP is a dead end
        # (can't build cities without settlements to upgrade)
        num_cities = min(len(player.cities), 4)
        if num_settlements == 0 and (num_settlements + num_cities * 2) < 8:
            potential -= 20.0

        # ========== SETTLEMENT READINESS BONUS ==========
        # Reward for being close to building a settlement (wood, brick, sheep, wheat)
        wood_count = player.resources.get(ResourceType.WOOD, 0)
        brick_count = player.resources.get(ResourceType.BRICK, 0)
        sheep_count = player.resources.get(ResourceType.SHEEP, 0)
        wheat_count_settle = player.resources.get(ResourceType.WHEAT, 0)
        wood_ok = min(wood_count, 1)
        brick_ok = min(brick_count, 1)
        sheep_ok = min(sheep_count, 1)
        wheat_ok = min(wheat_count_settle, 1)
        settlement_readiness = (wood_ok + brick_ok + sheep_ok + wheat_ok) / 4.0

        # URGENCY: at 0 settlements with 2+ cities, agent MUST expand
        # Triple the signal to make the agent trade ore→wood/brick
        if num_settlements == 0 and num_cities >= 2:
            potential += settlement_readiness * 9.0   # Up to +9 (3x normal)
        else:
            potential += settlement_readiness * 3.0   # Normal: up to +3

        # ========== CITY BUILDING INCENTIVE (CAPPED AT 4) ==========
        # MASSIVE bonus for cities - this is the KEY to winning
        # Max 4 cities per player in Catan
        num_cities = min(len(player.cities), 4)  # Cap at 4 (Catan max)
        # CRITICAL: City bonus must EXCEED the settlement potential lost when
        # upgrading. Steeper scaling makes 3rd/4th cities especially valuable,
        # reinforcing the pipeline: road → settlement → city.
        # First city: +25, Second: +55, Third: +90, Fourth: +130
        city_bonus = sum(25.0 + 5.0 * i for i in range(num_cities))
        potential += city_bonus

        # ========== CITY READINESS BONUS ==========
        # Reward for collecting city resources. Kept SMALL so the agent is
        # incentivized to SPEND resources (build city, +20 PBRS) rather than
        # hoard them (old +10 readiness was causing resource-hoarding behaviour).
        ore_count = player.resources.get(ResourceType.ORE, 0)
        wheat_count = player.resources.get(ResourceType.WHEAT, 0)
        ore_progress = min(ore_count / 3.0, 1.0)
        wheat_progress = min(wheat_count / 2.0, 1.0)
        # Only give readiness bonus if we have a settlement to upgrade
        if len(player.settlements) > 0:
            if self.num_players == 2:
                # 1v1: Small readiness bonus (+3 max) so net PBRS for building
                # a city is clearly positive: +20 city - 3 readiness = +17 PBRS.
                potential += ore_progress * 2.0 + wheat_progress * 1.0
            else:
                city_readiness = ore_progress * wheat_progress  # 0 to 1
                potential += city_readiness * 5.0

        # ========== RESOURCE DIVERSITY POTENTIAL ==========
        # In 1v1, producing all 5 resources is mandatory (no player trading).
        # This pulls the agent toward settlements that diversify production.
        produced_resources = set()
        for s in player.settlements:
            for tile in s.position.adjacent_tiles:
                if tile.resource and tile.resource != 'desert':
                    produced_resources.add(tile.resource)
        for c in player.cities:
            for tile in c.position.adjacent_tiles:
                if tile.resource and tile.resource != 'desert':
                    produced_resources.add(tile.resource)
        diversity = len(produced_resources)
        if self.num_players == 2:
            if diversity >= 5:
                potential += 10.0   # All 5 resources: massive 1v1 advantage
            elif diversity >= 4:
                potential += 5.0
            elif diversity >= 3:
                potential += 2.0
            if diversity <= 2:
                potential -= 5.0    # Only 2 resources: crippling in 1v1
        else:
            potential += diversity * 1.5

        # ========== PORT ACCESS POTENTIAL ==========
        # Ports are critical in 1v1 (bank-only trading).
        # A 2:1 port for your most-produced resource = converts excess into anything.
        player_ports = self.game_env.game.game_board.get_player_ports(player)
        if self.num_players == 2:
            for port in player_ports:
                if port.port_type == PortType.GENERIC:
                    potential += 5.0   # 3:1 generic: only efficient trade option in 1v1
                else:
                    potential += 8.0   # 2:1 specialized: massive efficiency in 1v1
                    # Extra bonus if port matches a resource we produce heavily (city tiles)
                    port_resource = port.port_type.value.split('_')[0]  # e.g. "ore" from "ore_2:1"
                    for c in player.cities:
                        for tile in c.position.adjacent_tiles:
                            if tile.resource == port_resource:
                                potential += 3.0  # Port synergizes with city production
                                break
                        else:
                            continue
                        break

        # ========== ROAD VALUE (EXPANSION-ACCESS BASED) ==========
        # Value roads by what settlement spots they ACCESS, not raw count.
        # Count road endpoints adjacent to unoccupied, distance-rule-valid
        # vertices with reasonable pip scores (>= 4 pips).
        reachable_spots = 0
        seen_vertices = set()
        for road in player.roads:
            for vertex in [road.position.vertex1, road.position.vertex2]:
                vid = id(vertex)
                if vid in seen_vertices:
                    continue
                seen_vertices.add(vid)
                if vertex.structure is not None:
                    continue
                # Check distance rule: no adjacent vertex has a structure
                too_close = any(adj.structure is not None for adj in vertex.adjacent_vertices)
                if too_close:
                    continue
                # Check pip quality of this potential settlement spot
                pips = sum(
                    pip_map.get(t.number, 0)
                    for t in vertex.adjacent_tiles
                    if hasattr(t, 'number') and t.number
                    and hasattr(t, 'resource') and t.resource and t.resource != 'desert'
                )
                if pips >= 4:
                    reachable_spots += 1
        # Cap at 3 — beyond 3 reachable spots, more roads have diminishing returns
        reachable_spots = min(reachable_spots, 3)
        potential += reachable_spots * 1.0

        # Longest road bonus
        if player.has_longest_road:
            potential += 3.0
        elif len(player.roads) >= 5:
            potential += 0.5  # contending for longest road

        # ========== STRATEGIC ASSET POTENTIAL ==========
        if player.has_longest_road: potential += 2.0
        if player.has_largest_army: potential += 2.0
        potential += player.development_cards.get(DevelopmentCardType.VICTORY_POINT, 0) * 1.5

        # ========== DEVELOPMENT CARD BONUS (CAPPED) ==========
        # Reduced: data shows 3-4 dev cards is optimal, 4-8 makes no difference.
        # Old weights were pulling agent toward knight-heavy strategy.
        knights = min(player.development_cards.get(DevelopmentCardType.KNIGHT, 0), 5)  # was 10
        potential += knights * 0.3  # was 0.5
        total_dev = min(sum(player.development_cards.values()), 5)  # was 15
        potential += total_dev * 0.2  # was 0.3

        # ========== OPPONENT THREAT POTENTIAL ==========
        for opp in self.game_env.game.players:
            if opp != player:
                opp_vp = opp.calculate_victory_points()
                my_vp = player.calculate_victory_points()

                if self.num_players == 2:
                    # 1v1: Progressive pressure based on opponent VP
                    if opp_vp >= 8:
                        penalty = min((opp_vp - 7) * 5.0, 25.0)
                    elif opp_vp >= 6:
                        penalty = (opp_vp - 5) * 2.0
                    else:
                        penalty = 0.0
                    potential -= penalty

                    # VP lead/deficit: reward being ahead, penalize falling behind
                    vp_diff = my_vp - opp_vp
                    potential += vp_diff * 1.0  # +1 per VP lead, -1 per VP deficit
                else:
                    # 4-player: Standard penalty
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

        # VP State Bonus - REMOVED
        # Was +0.1 per VP per step, which created a constant reward signal
        # that drowned out building rewards and poisoned the value function.
        # PBRS + VP change reward already cover VP incentives properly.
        reward_breakdown['vp_state_bonus'] = 0.0

        # ========== INITIAL PLACEMENT QUALITY REWARD ==========
        # Train the network to pick GOOD settlement locations during initial placement
        # This teaches the agent to understand the map and make strategic decisions
        if is_initial and action_name == 'place_settlement' and step_info.get('success'):
            # Find the vertex that was just placed
            player = self.game_env.game.players[self.player_id]
            if player.settlements:
                last_settlement = player.settlements[-1]
                vertex = last_settlement.position

                # Calculate placement quality score
                def pip_count(number):
                    if number is None:
                        return 0
                    return max(0, 6 - abs(7 - number))

                # Score based on production probability AND resource diversity
                total_pips = 0
                resources_accessed = set()
                ore_pips = 0
                wheat_pips = 0

                for tile in vertex.adjacent_tiles:
                    if hasattr(tile, 'number') and tile.number and hasattr(tile, 'resource'):
                        pips = pip_count(tile.number)
                        total_pips += pips
                        resources_accessed.add(tile.resource)
                        if tile.resource == 'mountain':  # ore
                            ore_pips += pips
                        elif tile.resource == 'field':  # wheat
                            wheat_pips += pips

                # Reward components:
                # 1. Total production (pip count) - max ~15 pips for 3 tiles
                pip_reward = total_pips * 0.3  # Up to ~4.5

                # 2. Resource diversity bonus (access to different resources)
                diversity_reward = len(resources_accessed) * 1.0  # Up to 3.0

                # 3. Ore/wheat access (critical for cities)
                city_resource_reward = (ore_pips + wheat_pips) * 0.2  # Up to ~3.0

                # 4. Expansion potential - score reachable future vertices (2 roads away)
                expansion_score = 0
                FUTURE_DISCOUNT = 0.25
                frontier = {vertex}
                visited = {vertex}
                game_board = self.game_env.game.game_board

                for road_step in range(2):
                    discount = FUTURE_DISCOUNT ** (road_step + 1)
                    next_frontier = set()
                    for v in frontier:
                        # Find connected edges
                        for edge in game_board.edges:
                            if edge.vertex1 == v or edge.vertex2 == v:
                                other = edge.vertex1 if edge.vertex2 == v else edge.vertex2
                                if other in visited:
                                    continue
                                visited.add(other)
                                next_frontier.add(other)
                                if other.structure is not None:
                                    continue
                                too_close = any(adj.structure is not None for adj in other.adjacent_vertices)
                                if too_close:
                                    continue
                                future_pip = sum(
                                    pip_count(tile.number)
                                    for tile in other.adjacent_tiles
                                    if hasattr(tile, 'number') and tile.number and
                                    hasattr(tile, 'resource') and tile.resource and tile.resource != 'desert'
                                )
                                if future_pip >= 3:
                                    expansion_score += future_pip * discount
                    frontier = next_frontier

                expansion_reward = min(expansion_score, 8.0) * 0.2  # Scale to ~0-1.6 range

                # 5. Gap-filling reward for 2nd settlement
                gap_reward = 0.0
                if len(player.settlements) == 2:
                    # Check what resources first settlement covers
                    first_settlement = player.settlements[0]
                    first_resources = set()
                    for tile in first_settlement.position.adjacent_tiles:
                        if hasattr(tile, 'resource') and tile.resource and tile.resource != 'desert':
                            first_resources.add(tile.resource)
                    # Bonus for covering new resources with second settlement
                    new_resources = resources_accessed - first_resources
                    gap_reward = len(new_resources) * 1.5

                placement_reward = pip_reward + diversity_reward + city_resource_reward + expansion_reward + gap_reward
                reward += placement_reward
                reward_breakdown['placement_quality'] = placement_reward

        # ========== CITY BUILDING BONUS (CRITICAL - 1v1 focused) ==========
        # Cities are THE key to winning - make this reward DOMINANT
        if step_info.get('built_city') or (action_name == 'build_city' and vp_diff > 0):
            if self.num_players == 2:
                # 1v1: +40 makes city building the dominant per-step signal.
                # Raised from 15 (which was too small vs old -10 inaction penalty).
                # Now that inaction penalty is removed, city_bonus must drive the behavior.
                city_bonus = 40.0
            else:
                # 4-player: Standard bonus with phase multiplier
                turn = self._turn_count
                if turn < 15:
                    phase_multiplier = 1.5
                elif turn < 40:
                    phase_multiplier = 2.0
                else:
                    phase_multiplier = 1.5
                city_bonus = 15.0 * phase_multiplier
            reward += city_bonus
            reward_breakdown['city_bonus'] = city_bonus

        # ========== SETTLEMENT BUILDING ==========
        # Settlements rewarded through VP reward (+3.0 per VP). No additional bonus
        # in the base env. Wrappers (PBRS/Simplified) add their own positional
        # quality shaping. NOTE: This only affects the base env reward path.
        # PBRSFixedRewardWrapper discards this reward entirely;
        # SimplifiedRewardWrapper computes its own settlement rewards separately.

        # ========== ROAD BUILDING ==========
        # Roads get ZERO direct reward. Value comes from PBRS potential change
        # (expansion access) and from enabling settlements which give VP.
        # This prevents road addiction that dominated previous training runs.

        # ========== DEV CARD PURCHASE BONUS ==========
        # Buying dev cards is a valid strategy, give small reward
        if action_name == 'buy_dev_card' and step_info.get('success', False):
            reward += 3.0
            reward_breakdown['dev_card_bonus'] = 3.0

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

        # ========== INACTION PENALTY ==========
        # NOTE: 1v1 city inaction penalty REMOVED - it caused a death spiral where
        # the value function learned "having city resources = terrible state" and the
        # agent traded away resources to escape the penalty rather than building cities.
        # City building is now incentivized purely via city_bonus (+40) and PBRS (+20 per city).
        if action_name == 'end_turn':
            legal_actions = old_obs.get('legal_actions', [])
            if self.num_players == 2:
                pass  # No inaction penalty in 1v1 - rely on city_bonus + PBRS instead
            else:
                # 4-player: Penalize not building anything
                build_actions = {'build_settlement', 'build_city', 'build_road', 'buy_dev_card'}
                if any(action in legal_actions for action in build_actions):
                    if 'build_city' in legal_actions:
                        inaction_penalty = -15.0
                    elif 'build_settlement' in legal_actions:
                        inaction_penalty = -5.0
                    else:
                        inaction_penalty = -1.0
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
                if self.num_players == 2:
                    # 1v1: Win must dominate all per-step rewards.
                    # 4 cities * 15 + 8 settlement VP * 3 = 84 max per-step → win_bonus > this
                    win_bonus = 200.0  # Increased from 50: was designed for 3VP, wrong for 10VP
                else:
                    # 4-player: Win bonus scales with VP
                    final_vp = new_obs['my_victory_points']
                    win_bonus = 15.0 + final_vp * 1.0
                reward += win_bonus
                reward_breakdown['win_bonus'] = win_bonus
            else:
                # Loss penalty - make it significant so agent learns losing is BAD
                if self.num_players == 2:
                    # 1v1: Scaled with new win_bonus
                    loss_penalty = -75.0
                else:
                    # 4-player: Standard penalty
                    loss_penalty = -15.0
                reward += loss_penalty
                reward_breakdown['loss_penalty'] = loss_penalty

        if debug:
            self._last_reward_breakdown = reward_breakdown

        # Clip reward to prevent extreme values that destabilize training
        # Range expanded to preserve win signal (200) without clipping
        reward = np.clip(reward, -200.0, 200.0)

        return reward

    def render(self):
        pass

    def close(self):
        pass
