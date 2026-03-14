"""
Curriculum Training V3 - STABLE VERSION

Fixes entropy collapse with:
1. Smooth entropy penalty (quadratic, not step function)
2. Adaptive learning rate based on entropy health
3. Trust region via KL divergence constraint
4. Gradient clipping with entropy-aware scaling
5. Performance-based curriculum transitions (not just game count)
6. Separate entropy tracking for all heads
7. Buffer prioritization for recent experiences
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time
import random
import threading
import queue
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force unbuffered stdout so training progress appears immediately
if not sys.stdout.line_buffering:
    sys.stdout.reconfigure(line_buffering=True)

from catan_env_pytorch import CatanEnv
from simplified_reward_wrapper import SimplifiedRewardWrapper
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
from network_wrapper import NetworkWrapper
from self_play_pool import SelfPlayPool
from game_system import ResourceType
from catanatron_opponent import play_weighted_random_turn
from rule_based_ai import score_vertex, score_edge


# Resource order matches catan_env_pytorch.py trade_give/get index mapping
_RESOURCE_ORDER = [ResourceType.WOOD, ResourceType.BRICK, ResourceType.WHEAT,
                   ResourceType.SHEEP, ResourceType.ORE]


def _bc_vertex_prefs(game, vertex_mask, player, temperature=2.0):
    """Soft-label vertex preferences using Strong AI scoring. Returns (54,) prob array."""
    all_vertices = game.game_board.vertices
    existing = player.settlements[0].position if len(player.settlements) == 1 else None
    scores = np.full(54, -1e9, dtype=np.float32)
    for i, v in enumerate(all_vertices[:54]):
        if vertex_mask[i] > 0:
            try:
                scores[i] = score_vertex(v, game_board=game.game_board,
                                         consider_diversity=True, consider_expansion=True,
                                         existing_settlement=existing)
            except Exception:
                scores[i] = 0.0
    valid = vertex_mask[:54] > 0
    if not valid.any():
        return np.ones(54, dtype=np.float32) / 54
    scores[~valid] = -1e9
    scores -= scores[valid].max()  # numerical stability
    exp_s = np.exp(scores / temperature)
    exp_s[~valid] = 0.0
    return exp_s / (exp_s.sum() + 1e-8)


def _bc_edge_prefs(game, edge_mask, player, temperature=2.0):
    """Soft-label edge preferences using Strong AI scoring. Returns (72,) prob array."""
    all_edges = game.game_board.edges
    scores = np.full(72, -1e9, dtype=np.float32)
    for i, e in enumerate(all_edges[:72]):
        if edge_mask[i] > 0:
            try:
                scores[i] = float(score_edge(e, player, game))
            except Exception:
                scores[i] = 0.0
    valid = edge_mask[:72] > 0
    if not valid.any():
        return np.ones(72, dtype=np.float32) / 72
    scores[~valid] = -1e9
    scores -= scores[valid].max()
    exp_s = np.exp(scores / temperature)
    exp_s[~valid] = 0.0
    return exp_s / (exp_s.sum() + 1e-8)


def _bc_trade_get_idx(player_resources):
    """Index (0-4) of the resource the Strong AI would most want to receive in a trade."""
    res = player_resources
    # Priority 1: finish a city
    if res.get(ResourceType.WHEAT, 0) >= 2 and res.get(ResourceType.ORE, 0) < 3:
        return 4  # ORE
    if res.get(ResourceType.ORE, 0) >= 3 and res.get(ResourceType.WHEAT, 0) < 2:
        return 2  # WHEAT
    # Priority 2: finish a settlement
    for idx, rt in enumerate(_RESOURCE_ORDER):
        if rt in (ResourceType.WOOD, ResourceType.BRICK,
                  ResourceType.WHEAT, ResourceType.SHEEP):
            if res.get(rt, 0) < 1:
                return idx
    # Fallback: whichever of wood/brick we have less of
    return 0 if res.get(ResourceType.WOOD, 0) < res.get(ResourceType.BRICK, 0) else 1


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def _pip_count(number):
    """Convert dice number to pip count (probability dots). 6 and 8 are best (5 pips)."""
    if number is None:
        return 0
    return max(0, 6 - abs(7 - number))


DICE_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def _aqs_pip_count(player):
    """Sum pip values from all hexes adjacent to player's settlements and cities."""
    pips = 0
    seen_tiles = set()
    for structure in player.settlements + player.cities:
        for tile in structure.position.adjacent_tiles:
            if id(tile) not in seen_tiles and tile.number and tile.resource and tile.resource != 'desert':
                seen_tiles.add(id(tile))
                pips += DICE_PIPS.get(tile.number, 0)
    return pips


def _aqs_resource_diversity(player):
    """Count distinct resource types produced by player's settlements/cities."""
    resources = set()
    for structure in player.settlements + player.cities:
        for tile in structure.position.adjacent_tiles:
            if tile.resource and tile.resource != 'desert':
                rt = tile.get_resource_type()
                if rt:
                    resources.add(rt)
    return len(resources)


def _aqs_port_access(player, game_board):
    """Count how many ports the player has access to."""
    return len(game_board.get_player_ports(player))


def _aqs_resources_spent(player):
    """Estimate total resources spent from structures built."""
    settlements = len(player.settlements)
    cities = len(player.cities)
    roads = len(player.roads)
    dev_cards = sum(player.development_cards.values()) + player.knights_played
    paid_settlements = max(0, settlements + cities - 2)
    paid_roads = max(0, roads - 2)
    return (paid_settlements * 4) + (cities * 5) + (paid_roads * 2) + (dev_cards * 3)


def play_passive_turn(game, player_id):
    """Passive opponent - NEVER builds after initial placement.

    This is the easiest possible opponent:
    - Completes initial placement (required)
    - Then only rolls dice and ends turn
    - Never builds settlements, cities, roads, or dev cards

    Use this for the very first training phase so agent can learn basics.
    """
    player = game.players[player_id]

    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            if game.last_settlement_vertex:
                edges = game.game_board.edges
                valid = [e for e in edges
                        if e.structure is None and
                        (e.vertex1 == game.last_settlement_vertex or
                         e.vertex2 == game.last_settlement_vertex)]
                if valid:
                    # Random road for passive (doesn't matter, never builds)
                    game.try_place_initial_road(random.choice(valid), player)
        else:
            vertices = game.game_board.vertices
            valid = [v for v in vertices if v.structure is None and
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                # Truly random for passive - keeps opponent weak
                game.try_place_initial_settlement(random.choice(valid), player)
        return True

    # Just roll and end - never build
    # FIX: Must roll AND end turn in same call, not return early
    if game.can_roll_dice():
        game.roll_dice()
        # Don't return here - fall through to end_turn

    if game.can_end_turn():
        game.end_turn()
        return True
    return True


def play_truly_random_turn(game, player_id):
    """Truly random opponent - picks random legal actions with low build probability.

    This is much weaker than play_random_turn because:
    - Only 15% chance to build when possible (vs 85%)
    - Doesn't prioritize any action type
    - Often just ends turn even when builds are available

    Use this for 1v1 initial training to give learning agent a fighting chance.
    """
    player = game.players[player_id]

    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            if game.last_settlement_vertex:
                edges = game.game_board.edges
                valid = [e for e in edges
                        if e.structure is None and
                        (e.vertex1 == game.last_settlement_vertex or
                         e.vertex2 == game.last_settlement_vertex)]
                if valid:
                    # Random road selection (keep truly_random weak)
                    game.try_place_initial_road(random.choice(valid), player)
        else:
            vertices = game.game_board.vertices
            valid = [v for v in vertices if v.structure is None and
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                # Actually truly random - no scoring, just pick any valid spot
                game.try_place_initial_settlement(random.choice(valid), player)
        return True

    if game.can_roll_dice():
        game.roll_dice()
        # FIX: Don't return - fall through to build/end logic

    # Only 15% chance to even try building (reduced from 30%)
    if game.can_trade_or_build() and random.random() < 0.15:
        actions = []
        res = player.resources

        if (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
            res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1):
            v = game.get_buildable_vertices_for_settlements()
            if v: actions.append(('sett', v))

        if res[ResourceType.WHEAT] >= 2 and res[ResourceType.ORE] >= 3:
            v = game.get_buildable_vertices_for_cities()
            if v: actions.append(('city', v))

        if res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1:
            e = game.get_buildable_edges()
            if e: actions.append(('road', e))

        if (res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1 and
            res[ResourceType.ORE] >= 1 and not game.dev_deck.is_empty()):
            actions.append(('dev', None))

        if actions:
            action_type, locs = random.choice(actions)
            if action_type == 'sett':
                player.try_build_settlement(random.choice(locs))
            elif action_type == 'city':
                player.try_build_city(random.choice(locs))
            elif action_type == 'road':
                player.try_build_road(random.choice(locs))
            elif action_type == 'dev':
                game.try_buy_development_card(player)
            # FIX: Don't return - fall through to end turn

    if game.can_end_turn():
        game.end_turn()
        return True
    return True


def play_random_turn(game, player_id):
    """Rule-based 'random' opponent - actually fairly intelligent.

    Builds things 85% of the time when possible. Despite the name,
    this is NOT truly random - it's a competent baseline AI.
    Relabeled as 'baseline' in curriculum descriptions.
    """
    player = game.players[player_id]

    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            if game.last_settlement_vertex:
                last_v = game.last_settlement_vertex
                edges = game.game_board.edges
                valid = [e for e in edges
                        if e.structure is None and
                        (e.vertex1 == last_v or e.vertex2 == last_v)]
                if valid:
                    # Smart road placement: pick edge leading to best expansion vertex
                    def score_initial_road(edge):
                        other = edge.vertex1 if edge.vertex2 == last_v else edge.vertex2
                        return score_vertex(other, consider_diversity=True,
                                            consider_expansion=True,
                                            game_board=game.game_board)
                    valid.sort(key=score_initial_road, reverse=True)
                    game.try_place_initial_road(valid[0], player)
        else:
            vertices = game.game_board.vertices
            valid = [v for v in vertices if v.structure is None and
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                # Detect if this is 2nd placement
                existing_settlement = None
                if len(player.settlements) == 1:
                    existing_settlement = player.settlements[0].position

                # Full scoring: pip + diversity + expansion + gap-filling
                scored = [(v, score_vertex(
                    v,
                    game_board=game.game_board,
                    consider_diversity=True,
                    consider_expansion=True,
                    existing_settlement=existing_settlement
                )) for v in valid]
                scored.sort(key=lambda x: x[1], reverse=True)
                game.try_place_initial_settlement(scored[0][0], player)
        return True

    if game.can_roll_dice():
        game.roll_dice()
        # FIX: Don't return - fall through to build/end logic

    if game.can_trade_or_build():
        actions = []
        res = player.resources

        if (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
            res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1):
            v = game.get_buildable_vertices_for_settlements()
            if v: actions.append(('sett', v))

        if res[ResourceType.WHEAT] >= 2 and res[ResourceType.ORE] >= 3:
            v = game.get_buildable_vertices_for_cities()
            if v: actions.append(('city', v))

        if res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1:
            e = game.get_buildable_edges()
            if e: actions.append(('road', e))

        if (res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1 and
            res[ResourceType.ORE] >= 1 and not game.dev_deck.is_empty()):
            actions.append(('dev', None))

        if actions and random.random() < 0.85:
            action_type, locs = random.choice(actions)
            if action_type == 'sett':
                player.try_build_settlement(random.choice(locs))
            elif action_type == 'city':
                player.try_build_city(random.choice(locs))
            elif action_type == 'road':
                player.try_build_road(random.choice(locs))
            elif action_type == 'dev':
                game.try_buy_development_card(player)
            # FIX: Don't return - fall through to end turn

    if game.can_end_turn():
        game.end_turn()
        return True
    return True


def play_opponent_turn(game, player_id, mix_prob, primary_ai='medium', secondary_ai=None):
    """Play opponent with mix of AI difficulties

    Args:
        game: The game instance
        player_id: Which player to play
        mix_prob: Probability of primary AI (0.0-1.0)
        primary_ai: Primary AI difficulty ('truly_random', 'random', 'very_weak', 'weak', 'medium', 'strong')
        secondary_ai: Secondary AI difficulty (if None, uses primary_ai vs truly_random for 1v1)

    AI Difficulty Scale (easiest to hardest):
        truly_random: 30% build chance, no strategy (easiest, for 1v1 warmup)
        random: 85% build chance, random choices (baseline)
        very_weak: Rule-based, picks from top 5 positions
        weak: Rule-based, picks from top 3 positions
        medium: Rule-based, picks from top 3 with some strategy
        strong: Rule-based, always picks best position
    """
    from rule_based_ai import play_rule_based_turn

    class MinimalEnv:
        def __init__(self, g):
            self.game_env = type('obj', (object,), {'game': g})()

    # Decide which AI to use based on mix_prob
    if random.random() < mix_prob:
        chosen_ai = primary_ai
    else:
        chosen_ai = secondary_ai if secondary_ai else 'truly_random'

    # Play with chosen AI
    if chosen_ai == 'passive':
        return play_passive_turn(game, player_id)
    elif chosen_ai == 'truly_random':
        return play_truly_random_turn(game, player_id)
    elif chosen_ai == 'weighted_random':
        return play_weighted_random_turn(game, player_id)
    elif chosen_ai == 'random':
        return play_random_turn(game, player_id)
    elif chosen_ai in ('city_rusher', 'road_blocker', 'dev_card_spammer',
                        'balanced_aggressor', 'port_specialist'):
        from specialist_ai import play_specialist_turn
        return play_specialist_turn(game, player_id, strategy=chosen_ai)
    else:
        return play_rule_based_turn(MinimalEnv(game), player_id, difficulty=chosen_ai)


class PrioritizedReplayBuffer:
    """Replay buffer with recency bias and priority sampling (thread-safe)"""

    def __init__(self, max_size=300000, recency_bias=0.7):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.recency_bias = recency_bias  # Probability of sampling from recent half
        self._lock = threading.Lock()  # Thread safety for parallel game execution

    def add(self, obs, action_probs, vertex_probs, edge_probs, reward, old_log_prob=None,
            vertex_idx=None, edge_idx=None):
        with self._lock:
            self.buffer.append({
                'obs': obs,
                'probs': action_probs,
                'vertex_probs': vertex_probs,
                'edge_probs': edge_probs,
                'vertex_idx': vertex_idx,
                'edge_idx': edge_idx,
                'reward': reward,
                'old_log_prob': old_log_prob
            })

    def sample(self, batch_size):
        with self._lock:
            n = len(self.buffer)
            if n < batch_size:
                indices = np.arange(n)
            else:
                # Sample with recency bias
                recent_size = min(n // 2, batch_size * 2)
                recent_count = int(batch_size * self.recency_bias)
                old_count = batch_size - recent_count

                recent_indices = np.random.choice(
                    range(n - recent_size, n),
                    min(recent_count, recent_size),
                    replace=False
                )
                old_indices = np.random.choice(
                    range(max(0, n - recent_size)),
                    min(old_count, max(1, n - recent_size)),
                    replace=False
                )
                indices = np.concatenate([recent_indices, old_indices])

            return {
                'observations': np.array([self.buffer[i]['obs'] for i in indices]),
                'action_probs': np.array([self.buffer[i]['probs'] for i in indices]),
                'vertex_probs': np.array([self.buffer[i]['vertex_probs'] for i in indices]),
                'edge_probs': np.array([self.buffer[i]['edge_probs'] for i in indices]),
                'action_idx': np.array([self.buffer[i].get('action_idx', 0) for i in indices]),
                'vertex_idx': np.array([self.buffer[i]['vertex_idx'] for i in indices]),
                'edge_idx': np.array([self.buffer[i]['edge_idx'] for i in indices]),
                'rewards': np.array([self.buffer[i]['reward'] for i in indices]),
                # Support both old format (single log_prob) and new format (per-head log probs)
                'old_action_log_probs': np.array([self.buffer[i].get('old_action_log_prob',
                                                   self.buffer[i].get('old_log_prob', 0.0)) for i in indices]),
                'old_vertex_log_probs': np.array([self.buffer[i].get('old_vertex_log_prob', 0.0) for i in indices]),
                'old_edge_log_probs': np.array([self.buffer[i].get('old_edge_log_prob', 0.0) for i in indices]),
                'trade_give_idx': np.array([self.buffer[i].get('trade_give_idx', 0) for i in indices]),
                'trade_get_idx': np.array([self.buffer[i].get('trade_get_idx', 0) for i in indices]),
                'old_trade_give_log_probs': np.array([self.buffer[i].get('old_trade_give_log_prob', 0.0) for i in indices]),
                'old_trade_get_log_probs': np.array([self.buffer[i].get('old_trade_get_log_prob', 0.0) for i in indices]),
                'action_masks': np.array([self.buffer[i].get('action_mask', np.ones(14)) for i in indices]),
                'vertex_masks': np.array([self.buffer[i].get('vertex_mask', np.ones(54)) for i in indices]),
                'edge_masks': np.array([self.buffer[i].get('edge_mask', np.ones(72)) for i in indices]),
                'bc_vertex_probs': np.array([self.buffer[i].get('bc_vertex_probs', np.ones(54) / 54) for i in indices]),
                'bc_edge_probs': np.array([self.buffer[i].get('bc_edge_probs', np.ones(72) / 72) for i in indices]),
                'bc_trade_get_idx': np.array([self.buffer[i].get('bc_trade_get_idx', 0) for i in indices]),
                'needs_vertex': np.array([self.buffer[i].get('needs_vertex', 0.0) for i in indices]),
                'needs_edge': np.array([self.buffer[i].get('needs_edge', 0.0) for i in indices]),
                'needs_trade': np.array([self.buffer[i].get('needs_trade', 0.0) for i in indices]),
            }

    def add_batch(self, experiences):
        """Add multiple experiences at once (more efficient for parallel games)"""
        with self._lock:
            for exp in experiences:
                self.buffer.append(exp)

    def clear_old(self, keep_fraction=0.3):
        """Clear old experiences, keeping recent fraction"""
        with self._lock:
            keep_count = int(len(self.buffer) * keep_fraction)
            while len(self.buffer) > keep_count:
                self.buffer.popleft()

    def __len__(self):
        with self._lock:
            return len(self.buffer)


class CentralInferenceServer:
    """Batches inference requests from parallel game threads into single GPU forward passes.

    Each game thread calls infer(obs) and blocks. The server collects all pending
    requests, runs one batched forward pass, then unblocks all callers.
    This converts N batch-size-1 GPU calls into one batch-size-N call, giving the
    GPU real work to do and eliminating GIL serialization overhead.
    """

    def __init__(self, network, device, max_batch_wait_ms=2.0):
        self.network = network
        self.device = device
        self.use_amp = (device.type == 'cuda')
        self._max_batch_wait = max_batch_wait_ms / 1000.0
        self._request_queue = queue.Queue()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._request_queue.put(None)  # unblock the inference thread
        if self._thread:
            self._thread.join(timeout=2.0)

    def infer(self, obs):
        """Submit obs dict, block until result returned. Returns (ap, vp, ep, tgp, tgetp, sv)."""
        event = threading.Event()
        holder = [None]
        self._request_queue.put((obs, event, holder))
        event.wait()
        result = holder[0]
        if isinstance(result, Exception):
            raise result
        return result

    def _inference_loop(self):
        while self._running:
            # Block waiting for the first request
            try:
                first = self._request_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if first is None:
                break

            batch = [first]
            # Drain remaining queued requests (up to max_batch_wait)
            deadline = time.monotonic() + self._max_batch_wait
            while time.monotonic() < deadline:
                try:
                    item = self._request_queue.get_nowait()
                    if item is None:
                        self._running = False
                        break
                    batch.append(item)
                except queue.Empty:
                    break

            try:
                obs_list = [item[0] for item in batch]
                observations = torch.from_numpy(
                    np.stack([o['observation'] for o in obs_list])).float().to(self.device)
                action_masks = torch.from_numpy(
                    np.stack([o['action_mask'] for o in obs_list])).float().to(self.device)
                vertex_masks = torch.from_numpy(
                    np.stack([o['vertex_mask'] for o in obs_list])).float().to(self.device)
                edge_masks = torch.from_numpy(
                    np.stack([o['edge_mask'] for o in obs_list])).float().to(self.device)

                ctx = torch.amp.autocast('cuda') if self.use_amp else contextlib.nullcontext()
                with torch.inference_mode(), ctx:
                    ap, vp, ep, tgp, tgetp, sv = self.network(
                        observations, action_masks, vertex_masks, edge_masks)

                aps = ap.cpu().numpy()
                vps = vp.cpu().numpy()
                eps = ep.cpu().numpy()
                tgps = tgp.cpu().numpy()
                tgetps = tgetp.cpu().numpy()
                svs = sv.cpu().numpy()

                for i, (_, event, holder) in enumerate(batch):
                    holder[0] = (aps[i], vps[i], eps[i], tgps[i], tgetps[i], svs[i])
                    event.set()

            except Exception as e:
                # Unblock all waiting threads before propagating (prevents infinite hang)
                print(f"[CentralInferenceServer] Error in inference loop: {e}")
                for _, event, holder in batch:
                    if holder[0] is None:
                        holder[0] = e  # signal error to caller
                        event.set()
                self._running = False
                raise


class CurriculumTrainerV3:
    """Stable curriculum trainer with entropy collapse prevention

    Supports 1v1 training mode (num_players=2) for faster initial learning,
    then transfer to 4-player mode.
    """

    def __init__(self, model_path=None, learning_rate=5e-4, batch_size=None, reward_mode='pbrs_fixed',
                 lr_decay=1.0, value_weight=0.5, entropy_decay=1.0, num_parallel_games=8,
                 buffer_size=200000, num_players=4, bc_coef=0.1, self_play_pool=None,
                 specialist_mix_rate=0.0, use_mcts=False, mcts_simulations=30, mcts_c_puct=1.5,
                 hidden_dim=256):
        self.device = get_device()
        self.reward_mode = reward_mode
        self.lr_decay = lr_decay  # Learning rate decay per 1000 games
        self.value_weight = value_weight  # Weight for value loss (reduced from 0.25; returns are now normalized so value loss ~1.0)
        self.entropy_decay = entropy_decay  # Entropy coefficient decay per 1000 games
        self.num_players = num_players  # 2 for 1v1, 4 for standard
        self.bc_coef = bc_coef          # Behavioral cloning coefficient (annealed to 0 over training)
        self.self_play_pool = self_play_pool  # SelfPlayPool or None
        self.specialist_mix_rate = specialist_mix_rate  # Prob of specialist opponent per turn during self-play
        self.use_mcts = use_mcts        # Use MCTS for action selection during data collection
        self.mcts_simulations = mcts_simulations  # MCTS simulations per move
        self.mcts_c_puct = mcts_c_puct  # MCTS exploration constant

        # Auto-tune settings for 1v1 mode (faster games = can run more in parallel)
        if self.num_players == 2:
            # 1v1 games are ~4x faster, run many more in parallel
            if num_parallel_games == 8:  # Only override if using default
                num_parallel_games = 32  # 4x parallelism for 1v1
            # Smaller buffer for 1v1 - ensures policy trains on fresh data
            if buffer_size == 200000:  # Only override if using default
                buffer_size = 50000  # Much smaller for 1v1 - less variance, fresher data

        # MCTS is ~15x slower per move (15 sims each needing NN eval).
        # Cap parallelism to avoid overwhelming the inference server.
        if self.use_mcts and num_parallel_games > 8:
            print(f"  MCTS mode: reducing parallel games {num_parallel_games} → 8 "
                  f"(each move does {mcts_simulations} sims)")
            num_parallel_games = 8

        # Store the (possibly auto-tuned) values
        self.num_parallel_games = num_parallel_games
        self.buffer_size = buffer_size

        if batch_size is None:
            batch_size = 1024 if self.device.type == 'cuda' else 256
        self.batch_size = batch_size

        self.network_wrapper = NetworkWrapper(model_path=model_path, device=str(self.device),
                                                hidden_dim=hidden_dim)
        self.network = self.network_wrapper.policy

        # torch.compile: fuses kernels for faster inference (~20-40% on CUDA).
        # We do a warmup forward pass immediately to surface Triton/backend errors
        # before training starts (compile() itself doesn't raise, first call does).
        if self.device.type == 'cuda':
            try:
                compiled = torch.compile(self.network, mode='reduce-overhead')
                # Warmup: tiny dummy batch to detect Triton missing / backend errors now
                _d = self.device
                from network_gpu import TOTAL_OBS_DIM
                with torch.inference_mode():
                    compiled(
                        torch.zeros(1, TOTAL_OBS_DIM, device=_d),
                        torch.zeros(1, 14, device=_d),
                        torch.zeros(1, 54, device=_d),
                        torch.zeros(1, 72, device=_d),
                    )
                self.network = compiled
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile unavailable ({type(e).__name__}), skipping")
                self.network = self.network_wrapper.policy

        # Central inference server (started/stopped around play_games_parallel)
        self._inference_server = None

        self.base_lr = learning_rate
        self.optimizer = torch.optim.Adam(self.network_wrapper.policy.parameters(), lr=learning_rate)

        # Replay buffer with recency bias (size configurable for large batch training)
        self.replay_buffer = PrioritizedReplayBuffer(max_size=buffer_size, recency_bias=0.7)

        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        self.games_played = 0
        self._games_lock = threading.Lock()  # Thread safety for games_played counter

        # ENTROPY PARAMETERS - SIMPLIFIED (no more adaptive chaos)
        # Fixed coefficient, let --entropy-decay handle reduction over time
        self.target_entropy = 1.0   # Only used for logging status
        self.entropy_coef = 0.05    # Increased from 0.025: matches train_gpu.py intent.
                                    # Previous "doubling" that collapsed was 0.05→0.10;
                                    # this is 0.025→0.05 (restoring original stable value).
                                    # Rollback to 0.035 if entropy drops below 0.3 within 5k games.
        self.min_entropy_coef = 0.04
        self.max_entropy_coef = 0.15

        # NO adaptive parameters - they caused cascading failures
        self.entropy_history = deque(maxlen=100)
        self.policy_loss_history = deque(maxlen=50)
        self.current_entropy_coef = self.entropy_coef

        # Trust region
        self.max_kl = 0.02  # Maximum KL divergence per update
        self.adaptive_kl_coef = 1.0

        # Gradient clipping - FIXED, no adaptation
        self.base_grad_norm = 0.5
        self.current_grad_norm = 0.5
        self._last_actual_grad_norm = 0.0  # actual pre-clip norm, updated each train step

        # Curriculum tracking (thread-safe)
        self.phase_wins = deque(maxlen=100)
        self.phase_vps = deque(maxlen=100)
        self._phase_lock = threading.Lock()

        # AQS tracking (protected by _phase_lock)
        from agent_quality_score import AgentQualityEvaluator
        self.aqs_evaluator = AgentQualityEvaluator(window_size=200)

        # DIAGNOSTIC: Track city building attempts
        self.diag_could_build_city = 0
        self.diag_took_build_city = 0
        self.diag_city_built = 0
        self._diag_lock = threading.Lock()

        # DIAGNOSTIC: Track settlement expansion and win signal
        # If avg_settlements_at_end stays at 2.0, agent never builds new settlements
        # (VP cap at 4 = only upgrading 2 initial settlements to cities)
        self.diag_settlements_at_end = deque(maxlen=100)
        self.diag_wins_last_log = 0  # How many times win_bonus fired since last log
        self.diag_truncated_last_log = 0  # How many games timed out (hit max_moves)

        print(f"Batch size: {self.batch_size}")
        print(f"Buffer size: {self.buffer_size}")
        print(f"Batch/Buffer ratio: {self.batch_size/self.buffer_size*100:.1f}%")
        print(f"Parallel games: {self.num_parallel_games}")
        print(f"Base LR: {self.base_lr}")
        print(f"LR decay: {self.lr_decay} per 1000 games")
        print(f"Value weight: {self.value_weight}")
        print(f"Reward mode: {self.reward_mode}")
        print(f"Target entropy: {self.target_entropy}")
        print(f"Base entropy coef: {self.entropy_coef}")
        print(f"Entropy decay: {self.entropy_decay} per 1000 games")
        print(f"Max KL divergence: {self.max_kl}")
        print(f"BC coef (imitation): {self.bc_coef}")

        # MCTS for improved action selection during data collection
        self._mcts = None
        if self.use_mcts:
            from mcts import MCTS
            # MCTS uses the NetworkWrapper for priors + value estimation
            self._mcts = MCTS(
                policy_network=self.network_wrapper,
                num_simulations=self.mcts_simulations,
                c_puct=self.mcts_c_puct,
            )
            print(f"MCTS enabled: {self.mcts_simulations} sims, c_puct={self.mcts_c_puct}")

    def _compute_smooth_entropy_penalty(self, entropy):
        """Smooth quadratic penalty instead of hard floor"""
        # Quadratic penalty below target, small bonus above
        # penalty = coef * max(0, target - entropy)^2
        # This creates smooth gradients near the threshold

        diff = self.target_entropy - entropy

        if diff > 0:
            # Below target: quadratic penalty (increases smoothly)
            penalty = diff ** 2
        else:
            # Above target: small bonus (encourage some exploration)
            penalty = -0.1 * min(abs(diff), 0.5)

        return penalty

    def _adapt_hyperparameters(self, entropy, policy_loss):
        """Track entropy for logging only - NO adaptive changes.

        The old adaptive system caused cascading failures:
        - Low entropy → reduced grad norm → agent can't learn → entropy stays low
        - LR adaptation pushed to 1e-3 which was too aggressive
        Now we use fixed coefficients and let --entropy-decay handle reduction.
        """
        self.entropy_history.append(entropy)
        self.policy_loss_history.append(abs(policy_loss))
        # No adaptation - all parameters stay fixed

    def play_game(self, mix_prob=1.0, primary_ai='random', secondary_ai=None, victory_points_to_win=10):
        """Play game and collect experiences

        Args:
            mix_prob: Probability of primary_ai (0.0-1.0)
            primary_ai: Primary AI difficulty ('random', 'very_weak', 'weak', 'medium', 'strong')
                         Special: 'curriculum_mix' selects per-game from self_play/strong/medium (50/25/25)
            secondary_ai: Secondary AI difficulty (if None, uses random for non-primary)
            victory_points_to_win: VP needed to win (default 10)
        """
        # Curriculum mixing: pick a single opponent type per game
        if primary_ai == 'curriculum_mix':
            roll = random.random()
            if roll < 0.50:
                primary_ai = 'self_play'
            elif roll < 0.75:
                primary_ai = 'strong'
            else:
                primary_ai = 'medium'
            # When using rule-based opponents, no secondary needed
            if primary_ai != 'self_play':
                secondary_ai = None
                mix_prob = 1.0
        if self.reward_mode == 'pbrs_fixed':
            env = PBRSFixedRewardWrapper(player_id=0, victory_points_to_win=victory_points_to_win, num_players=self.num_players)
        else:
            env = SimplifiedRewardWrapper(player_id=0, reward_mode=self.reward_mode, victory_points_to_win=victory_points_to_win, num_players=self.num_players)
        obs, _ = env.reset()

        episode_rewards = []
        episode_obs = []
        episode_action_probs = []
        episode_vertex_probs = []
        episode_edge_probs = []
        episode_action_idx = []
        episode_vertex_idx = []
        episode_edge_idx = []
        episode_trade_give_idx = []
        episode_trade_get_idx = []
        episode_log_probs = []
        # CRITICAL FIX: Store masks for proper PPO training
        episode_action_masks = []
        episode_vertex_masks = []
        episode_edge_masks = []
        # Behavioral cloning targets: Strong AI's preferred vertex/edge/trade
        episode_bc_vertex_probs = []
        episode_bc_edge_probs = []
        episode_bc_trade_get_idx = []
        episode_needs_vertex = []
        episode_needs_edge = []
        episode_needs_trade = []

        done = False
        moves = 0
        max_moves = 3000   # raised from 800: model does many trades/turn → 800 = only ~80 real turns, games always timed out → no win signal → plateau
        bank_trade_count = 0
        robber_placement_count = 0

        while not done and moves < max_moves:
            game = env.game_env.game
            current = game.get_current_player()
            current_id = game.players.index(current)

            if current_id == 0:
                moves += 1

                # Skip expensive MCTS for trivial forced actions (roll_dice, end_turn, wait, do_nothing)
                # These have only 1 legal action — MCTS adds N sims of overhead for zero benefit
                _action_mask = obs['action_mask']
                _n_legal = int(_action_mask.sum())
                _use_mcts_this_step = (self.use_mcts and self._mcts is not None and _n_legal > 1)

                if _use_mcts_this_step:
                    action, action_probs, vertex_probs, edge_probs, log_prob = self._get_action_mcts(obs, env)
                else:
                    action, action_probs, vertex_probs, edge_probs, log_prob = self._get_action(obs)
                action_id, vertex_id, edge_id, trade_give, trade_get = action

                episode_obs.append(obs['observation'].copy())
                episode_action_probs.append(action_probs)
                episode_vertex_probs.append(vertex_probs)
                episode_edge_probs.append(edge_probs)
                episode_action_idx.append(action_id)
                episode_vertex_idx.append(vertex_id)
                episode_edge_idx.append(edge_id)
                episode_trade_give_idx.append(trade_give)
                episode_trade_get_idx.append(trade_get)
                episode_log_probs.append(log_prob)
                # CRITICAL FIX: Store masks for PPO training
                episode_action_masks.append(obs['action_mask'].copy())
                episode_vertex_masks.append(obs['vertex_mask'].copy())
                episode_edge_masks.append(obs['edge_mask'].copy())

                # Behavioral cloning targets from Strong AI heuristic
                # action_id: 1=place_settlement, 2=place_road, 3=build_settlement,
                #             4=build_city, 5=build_road, 9=trade_with_bank
                _needs_v = action_id in (1, 3, 4)
                _needs_e = action_id in (2, 5)
                _needs_t = action_id == 9
                if action_id == 9:
                    bank_trade_count += 1
                if action_id == 11:
                    robber_placement_count += 1
                episode_needs_vertex.append(float(_needs_v))
                episode_needs_edge.append(float(_needs_e))
                episode_needs_trade.append(float(_needs_t))
                if _needs_v and self.bc_coef > 0:
                    _game = env.game_env.game
                    _player = _game.players[0]
                    episode_bc_vertex_probs.append(
                        _bc_vertex_prefs(_game, obs['vertex_mask'], _player))
                else:
                    episode_bc_vertex_probs.append(np.ones(54, dtype=np.float32) / 54)
                if _needs_e and self.bc_coef > 0:
                    _game = env.game_env.game
                    _player = _game.players[0]
                    episode_bc_edge_probs.append(
                        _bc_edge_prefs(_game, obs['edge_mask'], _player))
                else:
                    episode_bc_edge_probs.append(np.ones(72, dtype=np.float32) / 72)
                if _needs_t and self.bc_coef > 0:
                    _player = env.game_env.game.players[0]
                    episode_bc_trade_get_idx.append(
                        _bc_trade_get_idx(_player.resources))
                else:
                    episode_bc_trade_get_idx.append(0)

                # DIAGNOSTIC: Track city building opportunities and actions
                # Action 4 = build_city
                action_mask = obs.get('action_mask', None)
                could_build_city = action_mask is not None and len(action_mask) > 4 and action_mask[4] == 1
                took_build_city = action_id == 4

                next_obs, reward, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id,
                    trade_give_idx=trade_give, trade_get_idx=trade_get
                )

                # Track diagnostics
                with self._diag_lock:
                    if could_build_city:
                        self.diag_could_build_city += 1
                    if took_build_city:
                        self.diag_took_build_city += 1
                    if info.get('built_city') or (took_build_city and info.get('success', False)):
                        self.diag_city_built += 1

                episode_rewards.append(reward)
                obs = next_obs
                done = terminated or truncated
            else:
                # Self-play: use a past-checkpoint network as opponent
                # (with optional specialist mixing to prevent blind spots)
                sp_used = False
                if primary_ai == 'self_play' and self.self_play_pool is not None:
                    if self.specialist_mix_rate > 0 and random.random() < self.specialist_mix_rate:
                        specialist = random.choice([
                            'city_rusher', 'road_blocker', 'dev_card_spammer',
                            'balanced_aggressor', 'port_specialist',
                        ])
                        from specialist_ai import play_specialist_turn
                        play_specialist_turn(game, current_id, strategy=specialist)
                        sp_used = True
                    else:
                        sp_used = self.self_play_pool.play_turn(env, current_id)
                if not sp_used:
                    success = play_opponent_turn(game, current_id, mix_prob,
                                                 'strong' if primary_ai == 'self_play' else primary_ai,
                                                 secondary_ai)
                    if not success and game.can_end_turn():
                        game.end_turn()

                # FIX: Handle auto-discards after opponent rolls 7
                # This was being skipped because opponent doesn't go through env.step()
                if game.waiting_for_discards:
                    env.game_env._handle_automatic_discards()

                # FIX: Refresh observation after opponent turn so agent sees current state
                obs = env._get_obs()

                winner = game.check_victory_conditions()
                if winner is not None:
                    done = True

        # Calculate returns
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        # Clip range must be wide enough so city_bonus+win returns are NOT clipped:
        # city_bonus(40) + win_bonus(200)*gamma ≈ 238 → old clip of 200 crushed this to 200,
        # making city-building steps indistinguishable from any other step in a winning episode.
        # New upper bound of 300 lets the city step's ~238 return stand above the ~200 baseline.
        returns = np.clip(returns, -200, 300)

        # Collect experiences as list of dicts
        experiences = []
        for i in range(len(episode_obs)):
            log_probs = episode_log_probs[i]
            action_log_prob = log_probs[0]
            vertex_log_prob = log_probs[1]
            edge_log_prob = log_probs[2]
            trade_give_log_prob = log_probs[3] if len(log_probs) > 3 else 0.0
            trade_get_log_prob = log_probs[4] if len(log_probs) > 4 else 0.0
            experiences.append({
                'obs': episode_obs[i],
                'probs': episode_action_probs[i],
                'vertex_probs': episode_vertex_probs[i],
                'edge_probs': episode_edge_probs[i],
                'vertex_idx': episode_vertex_idx[i],
                'edge_idx': episode_edge_idx[i],
                'action_idx': episode_action_idx[i],
                'trade_give_idx': episode_trade_give_idx[i],
                'trade_get_idx': episode_trade_get_idx[i],
                'reward': returns[i],
                'old_action_log_prob': action_log_prob,
                'old_vertex_log_prob': vertex_log_prob,
                'old_edge_log_prob': edge_log_prob,
                'old_trade_give_log_prob': trade_give_log_prob,
                'old_trade_get_log_prob': trade_get_log_prob,
                'action_mask': episode_action_masks[i],
                'vertex_mask': episode_vertex_masks[i],
                'edge_mask': episode_edge_masks[i],
                'bc_vertex_probs': episode_bc_vertex_probs[i],
                'bc_edge_probs': episode_bc_edge_probs[i],
                'bc_trade_get_idx': episode_bc_trade_get_idx[i],
                'needs_vertex': episode_needs_vertex[i],
                'needs_edge': episode_needs_edge[i],
                'needs_trade': episode_needs_trade[i],
            })

        # Add to buffer (thread-safe)
        self.replay_buffer.add_batch(experiences)

        # Thread-safe counter update
        with self._games_lock:
            self.games_played += 1

        my_vp = env.game_env.game.players[0].calculate_victory_points()
        winner = env.game_env.game.check_victory_conditions()
        winner_id = game.players.index(winner) if winner else None

        # Thread-safe curriculum tracking
        with self._phase_lock:
            self.phase_wins.append(1 if winner_id == 0 else 0)
            self.phase_vps.append(my_vp)
            # AQS tracking
            _p0 = env.game_env.game.players[0]
            self.aqs_evaluator.record_game({
                'won': winner_id == 0,
                'agent_vp': my_vp,
                'opponent_vp': env.game_env.game.players[1].calculate_victory_points(),
                'agent_turns': moves,
                'settlements_built': len(_p0.settlements) + len(_p0.cities),
                'cities_built': len(_p0.cities),
                'roads_built': len(_p0.roads),
                'has_longest_road': _p0.has_longest_road,
                'has_largest_army': _p0.has_largest_army,
                'knights_played': _p0.knights_played,
                'opponent_type': primary_ai,
                'dev_cards_bought': sum(_p0.development_cards.values()) + _p0.knights_played,
                'dev_cards_played': _p0.knights_played,
                'total_resources_spent': _aqs_resources_spent(_p0),
                'bank_trades_made': bank_trade_count,
                'robber_placements': robber_placement_count,
                'pip_count': _aqs_pip_count(_p0),
                'resource_diversity': _aqs_resource_diversity(_p0),
                'port_access_count': _aqs_port_access(_p0, env.game_env.game.game_board),
            })

        # Track settlement count at game end (diagnose expansion failure)
        # If this stays at 2.0, agent never builds new settlements beyond initial placements
        with self._diag_lock:
            player0 = env.game_env.game.players[0]
            settlements_count = len(player0.settlements) if hasattr(player0, 'settlements') else 0
            self.diag_settlements_at_end.append(settlements_count)
            if winner_id == 0:
                self.diag_wins_last_log += 1
            if winner_id is None:  # No winner = game timed out (hit max_moves)
                self.diag_truncated_last_log += 1

        if self.self_play_pool is not None:
            self.self_play_pool.notify_game_done()

        return winner_id, my_vp, sum(episode_rewards)

    def _mcts_infer_via_server(self, obs):
        """Bridge: route MCTS neural network calls through CentralInferenceServer.

        MCTS expects fn(obs) -> (policy_dict, value_float).
        CentralInferenceServer.infer(obs) -> (ap, vp, ep, tgp, tgetp) numpy arrays.
        """
        ap, vp, ep, tgp, tgetp, sv = self._inference_server.infer(obs)
        policy = {
            'action': ap,
            'vertex': vp,
            'edge': ep,
        }
        value = float(np.tanh(sv.flatten()[0]))
        return policy, value

    def play_games_parallel(self, num_games, mix_prob=1.0, primary_ai='random',
                            secondary_ai=None, victory_points_to_win=10):
        """Play multiple games in parallel using ThreadPoolExecutor.

        Returns:
            List of (winner_id, vp, total_reward) tuples
        """
        results = []

        # Start centralized batched inference: all game threads share one GPU forward pass
        server = CentralInferenceServer(self.network, self.device)
        self._inference_server = server
        server.start()

        # Wire MCTS to use the batched inference server instead of individual GPU calls
        if self._mcts is not None:
            self._mcts.set_infer_fn(self._mcts_infer_via_server)

        try:
            with ThreadPoolExecutor(max_workers=num_games) as executor:
                futures = [
                    executor.submit(
                        self.play_game, mix_prob, primary_ai, secondary_ai, victory_points_to_win
                    )
                    for _ in range(num_games)
                ]

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"  Warning: Game failed with error: {e}")
                        results.append((None, 0, 0))
        finally:
            # Disconnect MCTS from server before stopping
            if self._mcts is not None:
                self._mcts.set_infer_fn(None)
            server.stop()
            self._inference_server = None

        return results

    def _get_action(self, obs):
        """Get action from network with entropy-aware exploration

        Returns:
            action: (action_id, vertex_id, edge_id, trade_give, trade_get)
            action_probs: probability distribution over actions
            vertex_probs: probability distribution over vertices
            edge_probs: probability distribution over edges
            log_prob: log probability of chosen action
        """
        if self._inference_server is not None:
            # Batched path: submit to central server and block until result ready
            ap, vp, ep, tgp, tgetp, _sv = self._inference_server.infer(obs)
        else:
            # Fallback single-step path (used outside play_games_parallel)
            ctx = torch.amp.autocast('cuda') if self.use_amp else contextlib.nullcontext()
            with torch.inference_mode(), ctx:
                observation = torch.from_numpy(obs['observation']).float().unsqueeze(0).to(self.device)
                action_mask = torch.from_numpy(obs['action_mask']).float().unsqueeze(0).to(self.device)
                vertex_mask = torch.from_numpy(obs['vertex_mask']).float().unsqueeze(0).to(self.device)
                edge_mask = torch.from_numpy(obs['edge_mask']).float().unsqueeze(0).to(self.device)

                action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, _ = self.network(
                    observation, action_mask, vertex_mask, edge_mask)

                ap = action_probs.cpu().numpy()[0]
                vp = vertex_probs.cpu().numpy()[0]
                ep = edge_probs.cpu().numpy()[0]
                tgp = trade_give_probs.cpu().numpy()[0]
                tgetp = trade_get_probs.cpu().numpy()[0]

        def safe_probs(probs):
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            return probs

        ap = safe_probs(ap)
        vp = safe_probs(vp)
        ep = safe_probs(ep)
        tgp = safe_probs(tgp)
        tgetp = safe_probs(tgetp)

        # Adaptive exploration based on recent entropy health
        if len(self.entropy_history) > 10:
            recent_entropy = np.mean(list(self.entropy_history)[-10:])
            if recent_entropy < 0.3:
                # Entropy collapsing, add temperature
                temperature = 1.5
                ap = np.power(ap, 1/temperature)
                ap = ap / ap.sum()
                vp = np.power(vp, 1/temperature)
                vp = vp / vp.sum()
                ep = np.power(ep, 1/temperature)
                ep = ep / ep.sum()

        action_id = np.random.choice(len(ap), p=ap)
        vertex_id = np.random.choice(len(vp), p=vp)
        edge_id = np.random.choice(len(ep), p=ep)
        trade_give = np.random.choice(len(tgp), p=tgp)
        trade_get = np.random.choice(len(tgetp), p=tgetp)

        # Store log probs for ALL heads (needed for proper PPO)
        action_log_prob = np.log(ap[action_id] + 1e-8)
        vertex_log_prob = np.log(vp[vertex_id] + 1e-8)
        edge_log_prob = np.log(ep[edge_id] + 1e-8)
        trade_give_log_prob = np.log(tgp[trade_give] + 1e-8)
        trade_get_log_prob = np.log(tgetp[trade_get] + 1e-8)

        return (action_id, vertex_id, edge_id, trade_give, trade_get), ap, vp, ep, (action_log_prob, vertex_log_prob, edge_log_prob, trade_give_log_prob, trade_get_log_prob)

    def _get_action_mcts(self, obs, env):
        """Get action using MCTS search for higher-quality decisions.

        Uses MCTS to select action + location, falls back to network for trades.
        Returns same format as _get_action so play_game works unchanged.
        """
        from game_state import GameState

        # Create a temporary GameState from the current env for MCTS to search
        # The env is a reward wrapper; its .env is the CatanEnv
        underlying_env = env.env if hasattr(env, 'env') else env
        temp_state = GameState(env=underlying_env)

        # MCTS search (uses copies internally, doesn't modify temp_state)
        best_action, mcts_probs = self._mcts.search(temp_state, temperature=1.0)
        action_id, vertex_id, edge_id = best_action

        # Get network's own probabilities (needed for PPO log_probs and entropy)
        if self._inference_server is not None:
            ap, vp, ep, tgp, tgetp, _sv = self._inference_server.infer(obs)
        else:
            ctx = torch.amp.autocast('cuda') if self.use_amp else contextlib.nullcontext()
            with torch.inference_mode(), ctx:
                observation = torch.from_numpy(obs['observation']).float().unsqueeze(0).to(self.device)
                action_mask = torch.from_numpy(obs['action_mask']).float().unsqueeze(0).to(self.device)
                vertex_mask = torch.from_numpy(obs['vertex_mask']).float().unsqueeze(0).to(self.device)
                edge_mask = torch.from_numpy(obs['edge_mask']).float().unsqueeze(0).to(self.device)

                action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, _ = self.network(
                    observation, action_mask, vertex_mask, edge_mask)

                ap = action_probs.cpu().numpy()[0]
                vp = vertex_probs.cpu().numpy()[0]
                ep = edge_probs.cpu().numpy()[0]
                tgp = trade_give_probs.cpu().numpy()[0]
                tgetp = trade_get_probs.cpu().numpy()[0]

        def safe_probs(probs):
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            return probs

        ap = safe_probs(ap)
        vp = safe_probs(vp)
        ep = safe_probs(ep)
        tgp = safe_probs(tgp)
        tgetp = safe_probs(tgetp)

        # Trade actions: MCTS doesn't specialize in trades, use network
        trade_give = np.random.choice(len(tgp), p=tgp)
        trade_get = np.random.choice(len(tgetp), p=tgetp)

        # Log probs from the NETWORK's perspective (for PPO importance sampling)
        action_log_prob = np.log(ap[action_id] + 1e-8)
        vertex_log_prob = np.log(vp[vertex_id] + 1e-8)
        edge_log_prob = np.log(ep[edge_id] + 1e-8)
        trade_give_log_prob = np.log(tgp[trade_give] + 1e-8)
        trade_get_log_prob = np.log(tgetp[trade_get] + 1e-8)

        return (action_id, vertex_id, edge_id, trade_give, trade_get), ap, vp, ep, (action_log_prob, vertex_log_prob, edge_log_prob, trade_give_log_prob, trade_get_log_prob)

    def train_step(self):
        """Single training step with proper PPO for ALL heads"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        obs = torch.FloatTensor(batch['observations']).to(self.device)
        target_action_probs = torch.FloatTensor(batch['action_probs']).to(self.device)
        target_vertex_probs = torch.FloatTensor(batch['vertex_probs']).to(self.device)
        target_edge_probs = torch.FloatTensor(batch['edge_probs']).to(self.device)
        action_idx = torch.LongTensor(batch['action_idx']).to(self.device)
        vertex_idx = torch.LongTensor(batch['vertex_idx']).to(self.device)
        edge_idx = torch.LongTensor(batch['edge_idx']).to(self.device)
        returns = torch.FloatTensor(batch['rewards']).to(self.device)
        # Old log probs for all heads (for proper PPO)
        old_action_log_probs = torch.FloatTensor(batch['old_action_log_probs']).to(self.device)
        old_vertex_log_probs = torch.FloatTensor(batch['old_vertex_log_probs']).to(self.device)
        old_edge_log_probs = torch.FloatTensor(batch['old_edge_log_probs']).to(self.device)
        # Trade indices and log probs
        trade_give_idx = torch.LongTensor(batch['trade_give_idx']).to(self.device)
        trade_get_idx = torch.LongTensor(batch['trade_get_idx']).to(self.device)
        old_trade_give_log_probs = torch.FloatTensor(batch['old_trade_give_log_probs']).to(self.device)
        old_trade_get_log_probs = torch.FloatTensor(batch['old_trade_get_log_probs']).to(self.device)
        # CRITICAL FIX: Get masks for proper PPO training
        action_masks = torch.FloatTensor(batch['action_masks']).to(self.device)
        vertex_masks = torch.FloatTensor(batch['vertex_masks']).to(self.device)
        edge_masks = torch.FloatTensor(batch['edge_masks']).to(self.device)

        # Behavioral cloning tensors (may be absent in old buffer data)
        batch_extra = {}
        if 'bc_vertex_probs' in batch and self.bc_coef > 0:
            batch_extra['bc_vertex_probs'] = torch.FloatTensor(batch['bc_vertex_probs']).to(self.device)
            batch_extra['bc_edge_probs'] = torch.FloatTensor(batch['bc_edge_probs']).to(self.device)
            batch_extra['bc_trade_get_idx'] = torch.LongTensor(batch['bc_trade_get_idx']).to(self.device)
            batch_extra['needs_vertex'] = torch.FloatTensor(batch['needs_vertex']).to(self.device)
            batch_extra['needs_edge'] = torch.FloatTensor(batch['needs_edge']).to(self.device)
            batch_extra['needs_trade'] = torch.FloatTensor(batch['needs_trade']).to(self.device)

        self.network.train()

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                loss_dict = self._compute_loss(
                    obs, target_action_probs, target_vertex_probs, target_edge_probs,
                    action_idx, vertex_idx, edge_idx, returns,
                    old_action_log_probs, old_vertex_log_probs, old_edge_log_probs,
                    action_masks, vertex_masks, edge_masks,
                    trade_give_idx, trade_get_idx,
                    old_trade_give_log_probs, old_trade_get_log_probs,
                    batch_extra=batch_extra
                )
        else:
            loss_dict = self._compute_loss(
                obs, target_action_probs, target_vertex_probs, target_edge_probs,
                action_idx, vertex_idx, edge_idx, returns,
                old_action_log_probs, old_vertex_log_probs, old_edge_log_probs,
                action_masks, vertex_masks, edge_masks,
                trade_give_idx, trade_get_idx,
                old_trade_give_log_probs, old_trade_get_log_probs,
                batch_extra=batch_extra
            )

        loss = loss_dict['total_loss']

        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            actual_norm = torch.nn.utils.clip_grad_norm_(self.network_wrapper.policy.parameters(), self.current_grad_norm)
            self._last_actual_grad_norm = actual_norm.item()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            actual_norm = torch.nn.utils.clip_grad_norm_(self.network_wrapper.policy.parameters(), self.current_grad_norm)
            self._last_actual_grad_norm = actual_norm.item()
            self.optimizer.step()

        # Adapt hyperparameters based on this step's metrics
        self._adapt_hyperparameters(loss_dict['entropy'], loss_dict['policy_loss'])

        return {
            'policy': loss_dict['policy_loss'],
            'vertex_policy': loss_dict['vertex_policy_loss'],
            'edge_policy': loss_dict['edge_policy_loss'],
            'value': loss_dict['value_loss'],
            'entropy': loss_dict['entropy'],
            'entropy_penalty': loss_dict['entropy_penalty'],
            'kl': loss_dict.get('kl', 0.0),
            'entropy_coef': self.current_entropy_coef,
            'grad_norm': self._last_actual_grad_norm,  # actual pre-clip norm, not clip limit
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def _compute_loss(self, obs, target_action_probs, target_vertex_probs, target_edge_probs,
                       action_idx, vertex_idx, edge_idx, returns,
                       old_action_log_probs, old_vertex_log_probs, old_edge_log_probs,
                       action_masks=None, vertex_masks=None, edge_masks=None,
                       trade_give_idx=None, trade_get_idx=None,
                       old_trade_give_log_probs=None, old_trade_get_log_probs=None,
                       batch_extra=None):
        """Compute loss with PROPER PPO for ALL heads including trade heads."""
        if batch_extra is None:
            batch_extra = {}
        # CRITICAL FIX: Pass masks to forward() so probabilities match what was used during gameplay
        action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, value = self.network(
            obs, action_masks, vertex_masks, edge_masks
        )
        value = value.squeeze()

        # Compute advantages (shared across all policy heads)
        advantages = returns - value.detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_ratio = 0.3  # Increased from 0.2 to allow larger policy updates

        # ========== ACTION POLICY LOSS (PPO with proper log prob) ==========
        action_log_probs_all = torch.log(action_probs + 1e-8)
        # FIXED: Use gather with actual action index, not weighted sum
        action_chosen_log_probs = action_log_probs_all.gather(1, action_idx.unsqueeze(1)).squeeze(1)

        # PPO clipping for action head
        action_ratio = torch.exp(action_chosen_log_probs - old_action_log_probs)
        action_surr1 = action_ratio * advantages
        action_surr2 = torch.clamp(action_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        action_policy_loss = -torch.min(action_surr1, action_surr2).mean()

        # ========== VERTEX POLICY LOSS (PPO - FIXED!) ==========
        vertex_log_probs_all = torch.log(vertex_probs + 1e-8)
        vertex_chosen_log_probs = vertex_log_probs_all.gather(1, vertex_idx.unsqueeze(1)).squeeze(1)

        # FIXED: PPO clipping for vertex head (was missing!)
        vertex_ratio = torch.exp(vertex_chosen_log_probs - old_vertex_log_probs)
        vertex_surr1 = vertex_ratio * advantages
        vertex_surr2 = torch.clamp(vertex_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        vertex_policy_loss = -torch.min(vertex_surr1, vertex_surr2).mean()

        # ========== EDGE POLICY LOSS (PPO - FIXED!) ==========
        edge_log_probs_all = torch.log(edge_probs + 1e-8)
        edge_chosen_log_probs = edge_log_probs_all.gather(1, edge_idx.unsqueeze(1)).squeeze(1)

        # FIXED: PPO clipping for edge head (was missing!)
        edge_ratio = torch.exp(edge_chosen_log_probs - old_edge_log_probs)
        edge_surr1 = edge_ratio * advantages
        edge_surr2 = torch.clamp(edge_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        edge_policy_loss = -torch.min(edge_surr1, edge_surr2).mean()

        # ========== ENTROPY (for all heads) ==========
        action_entropy = -(action_probs * action_log_probs_all).sum(dim=1).mean()
        vertex_entropy = -(vertex_probs * vertex_log_probs_all).sum(dim=1).mean()
        edge_entropy = -(edge_probs * edge_log_probs_all).sum(dim=1).mean()

        # Combined entropy (weighted average)
        total_entropy = 0.5 * action_entropy + 0.25 * vertex_entropy + 0.25 * edge_entropy

        # Smooth entropy penalty
        entropy_penalty = self._compute_smooth_entropy_penalty(total_entropy.item())
        entropy_penalty_tensor = torch.tensor(entropy_penalty, device=self.device)

        # Value loss: normalize returns to keep MSE ~O(1) so it doesn't swamp policy gradient.
        # Raw returns range -200..300 → MSE 500-2000 which crushes policy signal even with small value_weight.
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8
        value_loss = F.mse_loss((value - returns_mean) / returns_std,
                                 (returns - returns_mean) / returns_std)

        # Approximate KL divergence for monitoring (action head)
        kl_div = 0.5 * ((action_ratio - 1) ** 2).mean()

        # ========== TRADE POLICY LOSSES (PPO) ==========
        trade_give_policy_loss = torch.tensor(0.0, device=obs.device)
        trade_get_policy_loss = torch.tensor(0.0, device=obs.device)
        if trade_give_idx is not None and old_trade_give_log_probs is not None:
            trade_give_log_probs_all = torch.log(trade_give_probs + 1e-8)
            trade_give_chosen = trade_give_log_probs_all.gather(1, trade_give_idx.unsqueeze(1)).squeeze(1)
            tg_ratio = torch.exp(trade_give_chosen - old_trade_give_log_probs)
            tg_surr1 = tg_ratio * advantages
            tg_surr2 = torch.clamp(tg_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            trade_give_policy_loss = -torch.min(tg_surr1, tg_surr2).mean()

            trade_get_log_probs_all = torch.log(trade_get_probs + 1e-8)
            trade_get_chosen = trade_get_log_probs_all.gather(1, trade_get_idx.unsqueeze(1)).squeeze(1)
            tget_ratio = torch.exp(trade_get_chosen - old_trade_get_log_probs)
            tget_surr1 = tget_ratio * advantages
            tget_surr2 = torch.clamp(tget_ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            trade_get_policy_loss = -torch.min(tget_surr1, tget_surr2).mean()

        # ========== BEHAVIORAL CLONING AUXILIARY LOSS ==========
        # Guides vertex/edge/trade heads toward Strong AI's heuristic choices.
        # Applied only on steps where each sub-head was actually used.
        # bc_coef is annealed to 0 over the first 50% of training.
        bc_loss = torch.tensor(0.0, device=obs.device)
        if self.bc_coef > 0 and 'bc_vertex_probs' in batch_extra:
            bc_vertex_probs = batch_extra['bc_vertex_probs']   # (B, 54)
            bc_edge_probs = batch_extra['bc_edge_probs']       # (B, 72)
            bc_trade_get_idx = batch_extra['bc_trade_get_idx'] # (B,)
            needs_v = batch_extra['needs_vertex']              # (B,)
            needs_e = batch_extra['needs_edge']                # (B,)
            needs_t = batch_extra['needs_trade']               # (B,)

            # Vertex: cross-entropy loss against Strong's soft labels
            v_log = torch.log(vertex_probs + 1e-8)  # (B, 54)
            v_ce = -(bc_vertex_probs * v_log).sum(dim=1)  # (B,)
            n_v = needs_v.sum() + 1e-8
            bc_vertex_loss = (v_ce * needs_v).sum() / n_v

            # Edge: cross-entropy loss against Strong's soft labels
            e_log = torch.log(edge_probs + 1e-8)  # (B, 72)
            e_ce = -(bc_edge_probs * e_log).sum(dim=1)  # (B,)
            n_e = needs_e.sum() + 1e-8
            bc_edge_loss = (e_ce * needs_e).sum() / n_e

            # Trade-get: NLL loss against Strong's one-hot resource choice
            tg_log = torch.log(trade_get_probs + 1e-8)  # (B, 5)
            tg_nll = -tg_log.gather(1, bc_trade_get_idx.unsqueeze(1)).squeeze(1)  # (B,)
            n_t = needs_t.sum() + 1e-8
            bc_trade_loss = (tg_nll * needs_t).sum() / n_t

            bc_loss = self.bc_coef * (bc_vertex_loss + bc_edge_loss + 0.3 * bc_trade_loss)

        # ========== COMBINED LOSS ==========
        # All heads now have proper PPO including trade heads!
        total_loss = (
            action_policy_loss
            + 0.5 * vertex_policy_loss
            + 0.5 * edge_policy_loss
            + 0.3 * trade_give_policy_loss
            + 0.3 * trade_get_policy_loss
            - self.current_entropy_coef * total_entropy
            + self.current_entropy_coef * entropy_penalty_tensor
            + self.value_weight * value_loss
            + self.adaptive_kl_coef * torch.clamp(kl_div - self.max_kl, min=0.0)
            + bc_loss
        )

        return {
            'total_loss': total_loss,
            'policy_loss': action_policy_loss.item(),
            'vertex_policy_loss': vertex_policy_loss.item(),
            'edge_policy_loss': edge_policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': total_entropy.item(),
            'action_entropy': action_entropy.item(),
            'vertex_entropy': vertex_entropy.item(),
            'edge_entropy': edge_entropy.item(),
            'entropy_penalty': entropy_penalty,
            'kl': kl_div.item(),
            'bc_loss': bc_loss.item() if isinstance(bc_loss, torch.Tensor) else float(bc_loss),
        }

    def should_advance_curriculum(self, mix_prob, primary_ai, secondary_ai, vp_threshold):
        """Check if agent has mastered current difficulty level based on VP AND win rate"""
        if len(self.phase_wins) < 50:
            return False

        recent_vp = np.mean(list(self.phase_vps)[-50:])
        recent_wr = np.mean(list(self.phase_wins)[-50:])

        # Minimum win rate required by difficulty (must actually win, not just get VP)
        # Different thresholds for 1v1 (50% baseline) vs 4-player (25% baseline)
        if self.num_players == 2:
            # 1v1 mode: 50% theoretical baseline
            # passive = never builds, agent should win 95%+ just by building anything
            # truly_random = 15% build chance, agent should dominate
            # random = 85% build chance, harder baseline
            #
            # NOTE: Win rate thresholds are LOWERED for longer games (8VP, 10VP)
            # because higher VP games have more variance and opponent gets more chances.
            # The key insight: if agent is IMPROVING VP (e.g., 3.5 -> 4.0 -> 4.5),
            # it's learning even if WR is low. Let it advance to see if it can improve.
            min_wr_by_difficulty = {
                # All phases at 10VP now - thresholds reflect realistic 10VP win rates.
                # vs passive (never builds): agent should comfortably win 60%+
                'passive': 0.60,
                # vs truly_random (15% build rate): expect ~50-60% WR (opponent barely builds)
                'truly_random': 0.50,
                # vs weighted_random: expect ~25-35% WR
                'weighted_random': 0.22,
                # vs rule-based random (85% build rate): expect ~20-25% WR
                'random': 0.20,
                'very_weak': 0.28,
                'weak': 0.20,
                'medium': 0.12,
                'strong': 0.15,
                'curriculum_mix': 0.15,  # Mixed opponents: same threshold as strong
            }
        else:
            # 4-player mode: 25% baseline, lower thresholds
            min_wr_by_difficulty = {
                'random': 0.08,      # 8% WR vs Random
                'very_weak': 0.05,   # 5% WR vs VeryWeak
                'weak': 0.03,        # 3% WR vs Weak
                'medium': 0.01,      # 1% WR vs Medium
                'strong': 0.005,     # 0.5% WR vs Strong
            }
        min_wr = min_wr_by_difficulty.get(primary_ai, 0.05)

        # Must meet BOTH VP threshold AND minimum win rate
        return recent_vp >= vp_threshold and recent_wr >= min_wr

    def train(self, total_games=10000, save_path='models/curriculum_v3_stable',
              train_frequency=5, train_steps=15, min_games_per_phase=1000, start_phase=0):
        """Curriculum training with adaptive phase transitions

        Args:
            start_phase: Phase index to start from (use --list-phases to see all phases)
        """
        os.makedirs('models', exist_ok=True)

        # Phases: (primary_ai, secondary_ai, mix_prob, vp_to_win, vp_threshold, phase_name)
        # mix_prob = probability of primary_ai (1.0 = always primary, 0.5 = 50/50 mix)
        #
        # IMPROVED CURRICULUM with GRADUAL OPPONENT MIXING:
        # - Pure difficulty jumps cause 0% win rate = no learning signal
        # - Mix in harder opponents gradually so agent still wins sometimes
        # - Each opponent independently rolls primary vs secondary AI

        if self.num_players == 2:
            # === 1v1 OPTIMIZED CURRICULUM (v3) ===
            # Key insight: Agent needs EASY wins first to learn that building = good
            # - Start with PASSIVE opponent (never builds) = guaranteed win if agent builds anything
            # - Very low VP targets (3 VP = just build 1 settlement)
            # - Gradual difficulty increase only after agent learns to build
            phases = [
                # ALL phases at 10VP - a 3VP game is fundamentally different from a 10VP game.
                # Low-VP phases teach wrong strategies (1 city wins!) that must be unlearned.
                # Progression is purely by opponent difficulty.
                #
                # VP threshold formula for 10VP games (rough): avg_vp ≈ WR*5 + 5
                # (winner gets 10, loser averages ~5 at game end)
                #
                # Phase 0a: Warmup at 6VP — agent needs just 4 more VP beyond 2 initial.
                # Even an untrained agent can win here, providing clean win/loss signal.
                # Expect 70%+ WR quickly. Threshold 5.0 = clear mastery before moving on.
                ('passive', None, 1.0, 6, 5.0, "1v1 Passive 6VP"),
                # Phase 0b: Full 10VP vs passive. Agent now knows "building = winning".
                # Expect 80%+ WR → avg_vp ~9. Threshold 7.5 ensures real learning.
                ('passive', None, 1.0, 10, 7.5, "1v1 Passive"),
                # Phase 1: Mix in truly_random (15% build chance)
                # Expect 55-65% WR → avg_vp ~7.5-8.3. Raised thresholds to ensure mastery.
                ('truly_random', 'passive', 0.5, 10, 7.0, "1v1 TrulyRandom/Passive"),
                ('truly_random', None, 1.0, 10, 7.0, "1v1 TrulyRandom"),
                # Phase 2: WeightedRandom bridge (catanatron-style weighted actions)
                ('weighted_random', 'truly_random', 0.5, 10, 5.5, "1v1 WeightedRandom/TrulyRandom"),
                ('weighted_random', None, 1.0, 10, 5.5, "1v1 WeightedRandom"),
                # Phase 3: Opponent difficulty progression (skip rule-based random - weaker than weighted_random)
                ('very_weak', 'weighted_random', 0.5, 10, 6.2, "1v1 VeryWeak/WeightedRandom"),
                ('weak', 'very_weak', 0.5, 10, 5.5, "1v1 Weak/VeryWeak"),
                ('medium', 'weak', 0.5, 10, 5.0, "1v1 Medium/Weak"),
                ('strong', 'medium', 0.5, 10, 5.5, "1v1 Strong/Medium"),
                ('strong', None, 1.0, 10, 5.5, "1v1 Strong"),
                # Phase 4: Specialist diversity — expose agent to distinct strategies
                ('city_rusher', 'strong', 0.5, 10, 5.5, "1v1 CityRusher/Strong"),
                ('road_blocker', 'strong', 0.5, 10, 5.5, "1v1 RoadBlocker/Strong"),
                ('dev_card_spammer', 'strong', 0.5, 10, 5.5, "1v1 DevCardSpammer/Strong"),
                ('balanced_aggressor', 'strong', 0.5, 10, 5.5, "1v1 BalancedAggressor/Strong"),
                ('port_specialist', 'strong', 0.5, 10, 5.5, "1v1 PortSpecialist/Strong"),
                # Phase 5: Curriculum mix — 50% self-play + 25% strong + 25% medium
                # Ensures continuous learning signal even when self-play WR plateaus
                ('curriculum_mix', 'strong', 0.5, 10, 5.5, "1v1 CurriculumMix/Strong"),
                ('curriculum_mix', None, 1.0, 10, 999, "1v1 CurriculumMix FINAL"),
            ]
        else:
            # === STANDARD 4-PLAYER CURRICULUM ===
            phases = [
                # === PHASE 1: Learn game length progression vs Random ===
                ('random', None, 1.0, 4, 2.8, "Random 4VP"),
                ('random', None, 1.0, 5, 3.2, "Random 5VP"),
                ('random', None, 1.0, 6, 3.6, "Random 6VP"),
                ('random', None, 1.0, 7, 4.0, "Random 7VP"),
                ('random', None, 1.0, 8, 4.4, "Random 8VP"),
                ('random', None, 1.0, 9, 4.8, "Random 9VP"),
                ('random', None, 1.0, 10, 5.2, "Random 10VP"),
                # === PHASE 2: Gradual introduction of harder opponents ===
                # Mix opponents so agent still wins some games (learning signal!)
                ('very_weak', 'random', 0.5, 10, 3.5, "VeryWeak/Random Mix"),
                ('very_weak', None, 1.0, 10, 3.0, "VeryWeak 10VP"),
                ('weak', 'very_weak', 0.5, 10, 3.0, "Weak/VeryWeak Mix"),
                ('weak', None, 1.0, 10, 2.8, "Weak 10VP"),
                ('medium', 'weak', 0.5, 10, 2.8, "Medium/Weak Mix"),
                ('medium', None, 1.0, 10, 2.5, "Medium 10VP"),
                ('strong', 'medium', 0.5, 10, 2.5, "Strong/Medium Mix"),
                ('strong', 'medium', 0.5, 10, 999, "Strong/Medium FINAL"),  # Keep mix for learning signal
            ]

        print("\n" + "=" * 70)
        print("CURRICULUM TRAINING V3 - STABLE VERSION (PARALLEL)")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total games: {total_games}")
        print(f"Batch size: {self.batch_size}")
        print(f"Parallel games: {self.num_parallel_games}")
        print("STABILITY FEATURES:")
        print("  1. Smooth quadratic entropy penalty (not 200x step!)")
        print("  2. Adaptive entropy coefficient (0.05-0.5)")
        print("  3. Adaptive learning rate and gradient clipping")
        print("  4. PPO clipping for policy updates")
        print("  5. Multi-head entropy tracking (action+vertex+edge)")
        print("  6. Performance-based curriculum advancement")
        print("  7. Recency-biased experience replay")
        print("  8. PARALLEL game execution for speed")
        print("=" * 70 + "\n")

        # Validate and set starting phase
        if start_phase < 0 or start_phase >= len(phases):
            print(f"Warning: start_phase {start_phase} out of range, using 0")
            start_phase = 0
        if start_phase > 0:
            print(f"  ► Starting from phase {start_phase}: {phases[start_phase][5]}")
            # Boost entropy when jumping into a harder phase mid-training.
            # Without this the coef stays at the base 0.05 and collapses quickly.
            self.current_entropy_coef = self.max_entropy_coef
            print(f"  ► Entropy coef boosted to {self.current_entropy_coef:.3f} for non-zero start phase")

        current_phase = start_phase
        phase_game_count = 0
        total_wins = 0
        start_time = time.time()
        game_num = 0
        last_log_game = 0
        last_save_game = 0

        # Per-phase linear entropy decay
        phase_start_entropy = self.current_entropy_coef
        # Scale decay window with total budget: at least 2000 games, at most 20% of total.
        # Old hardcoded 1500 caused entropy to hit minimum after 1.5% of a 100k-game run,
        # leaving 98.5k games with zero exploration (root cause of the plateau).
        entropy_decay_games = max(2000, total_games // 5)

        # Stagnation detection
        stagnation_wr_history = []  # list of (phase_game_count, recent_wr)
        last_stagnation_check = 0
        stagnation_lr_cuts = 0      # resets each phase; max 4 cuts per phase

        first_batch = True
        while game_num < total_games:
            primary_ai, secondary_ai, mix_prob, vp_to_win, vp_threshold, phase_name = phases[current_phase]

            # Determine how many games to play in this batch
            games_remaining = total_games - game_num
            batch_size = min(self.num_parallel_games, games_remaining)

            if first_batch:
                mcts_note = f" (MCTS {self.mcts_simulations} sims/move)" if self.use_mcts else ""
                print(f"  ▶ Starting first batch: {batch_size} parallel games{mcts_note}...")
                first_batch = False

            # Play games in parallel
            results = self.play_games_parallel(
                num_games=batch_size,
                mix_prob=mix_prob,
                primary_ai=primary_ai,
                secondary_ai=secondary_ai,
                victory_points_to_win=vp_to_win
            )

            # Process results
            batch_wins = sum(1 for winner_id, _, _ in results if winner_id == 0)
            total_wins += batch_wins
            game_num += batch_size
            phase_game_count += batch_size

            # Train after each batch
            if len(self.replay_buffer) >= self.batch_size:
                # Scale train steps with batch size, but cap at 1 epoch through the buffer.
                # Without the cap, large parallel_games (e.g. 64) produces 192 gradient steps
                # on a buffer of only ~6K experiences = 65+ epochs, causing value overfitting.
                # Overfit value → advantages ≈ 0 → policy gradient dies.
                scaled_steps = max(1, train_steps * batch_size // train_frequency)
                one_epoch = max(1, len(self.replay_buffer) // self.batch_size)
                effective_train_steps = min(scaled_steps, one_epoch)
                losses = [self.train_step() for _ in range(effective_train_steps)]
                losses = [l for l in losses if l]

                # Log every 10 games worth of progress
                if losses and game_num - last_log_game >= 10:
                    last_log_game = game_num
                    avg_p = np.mean([l['policy'] for l in losses])
                    avg_v = np.mean([l['value'] for l in losses])
                    avg_e = np.mean([l['entropy'] for l in losses])
                    curr_ec = losses[-1]['entropy_coef']
                    curr_gn = losses[-1]['grad_norm']
                    curr_lr = losses[-1]['lr']

                    elapsed = time.time() - start_time
                    speed = game_num / elapsed * 60

                    with self._phase_lock:
                        recent_wr = np.mean(list(self.phase_wins)[-50:]) * 100 if len(self.phase_wins) >= 10 else 0
                        recent_vp = np.mean(list(self.phase_vps)[-50:]) if len(self.phase_vps) >= 10 else 0

                    # Status indicators
                    entropy_status = "✓" if 0.6 <= avg_e <= 1.8 else "⚠️" if avg_e < 0.3 else "↑"

                    print(f"  [{phase_name}] Game {game_num:5d}/{total_games} | "
                          f"WR: {recent_wr:5.1f}% | VP: {recent_vp:.1f} | "
                          f"Speed: {speed:.1f} g/min")
                    print(f"    └─ policy={avg_p:+.4f}, value={avg_v:.4f}, "
                          f"entropy={avg_e:.3f}{entropy_status}, "
                          f"ec={curr_ec:.3f}, gn={curr_gn:.2f}, lr={curr_lr:.1e}")

                    # DIAGNOSTIC: City building stats
                    with self._diag_lock:
                        if self.diag_could_build_city > 0:
                            take_rate = self.diag_took_build_city / self.diag_could_build_city * 100
                            success_rate = self.diag_city_built / max(1, self.diag_took_build_city) * 100
                            print(f"    └─ CITY: agent_turns_could_build={self.diag_could_build_city}, "
                                  f"taken={self.diag_took_build_city} ({take_rate:.0f}%), "
                                  f"built={self.diag_city_built} ({success_rate:.0f}% success)")
                        # Reset counters
                        self.diag_could_build_city = 0
                        self.diag_took_build_city = 0
                        self.diag_city_built = 0

                    # DIAGNOSTIC: Settlement expansion and win signal
                    # avg_at_end=2.0 → agent never expands past initial settlements → VP cap at 4
                    # win_bonus_fired=0 → win signal never reaches policy
                    if self.diag_settlements_at_end:
                        avg_setts = np.mean(list(self.diag_settlements_at_end)[-50:])
                        wins_since_log = self.diag_wins_last_log
                        truncated_since_log = self.diag_truncated_last_log
                        self.diag_wins_last_log = 0
                        self.diag_truncated_last_log = 0
                        print(f"    └─ SETTLE: avg_at_end={avg_setts:.1f} | "
                              f"win_bonus_fired={wins_since_log} | "
                              f"timeouts={truncated_since_log} (last ~10 games)")

                    # AQS score
                    if len(self.aqs_evaluator.results) >= 20:
                        aqs = self.aqs_evaluator.calculate_aqs()
                        print(f"    \u2514\u2500 AQS: {aqs:.0f}/1000")

            # Linear entropy decay within phase: high at start, min after entropy_decay_games
            progress = min(1.0, phase_game_count / entropy_decay_games)
            target_ec = phase_start_entropy + (self.min_entropy_coef - phase_start_entropy) * progress
            self.current_entropy_coef = max(self.min_entropy_coef, target_ec)

            # Stagnation detection: if WR fails to improve by 5% over 2000 games, cut LR by 50%
            # Uses 4 readings (each 500 games apart) to smooth out WR noise (±6-8% variance).
            # Disabled in the final phase: self-play WR naturally oscillates around 50% regardless
            # of true skill improvement, so cutting LR here only hurts learning.
            is_final_phase = (current_phase == len(phases) - 1)
            if (not is_final_phase and
                    phase_game_count - last_stagnation_check >= 500 and
                    len(self.phase_wins) >= 50 and stagnation_lr_cuts < 4):
                with self._phase_lock:
                    current_wr = np.mean(list(self.phase_wins)[-50:])
                stagnation_wr_history.append((phase_game_count, current_wr))
                last_stagnation_check = phase_game_count
                if len(stagnation_wr_history) >= 4:
                    window_wrs = [wr for _, wr in stagnation_wr_history[-4:]]
                    baseline_wr = window_wrs[0]
                    best_recent_wr = max(window_wrs[1:])
                    if best_recent_wr - baseline_wr < 0.05:
                        stagnation_lr_cuts += 1
                        for pg in self.optimizer.param_groups:
                            pg['lr'] = max(1e-5, pg['lr'] * 0.5)
                        new_lr = self.optimizer.param_groups[0]['lr']
                        print(f"\n  ⚡ STAGNATION detected: best WR {best_recent_wr*100:.1f}% vs "
                              f"baseline {baseline_wr*100:.1f}% over 2000 games. "
                              f"LR cut to {new_lr:.2e} (cut {stagnation_lr_cuts}/4)")
                        # Keep only the most recent reading so the window slides forward
                        stagnation_wr_history = [stagnation_wr_history[-1]]

            # Check for curriculum advancement
            force_advance = (phase_game_count >= min_games_per_phase * 5 and
                             current_phase < len(phases) - 1)
            if force_advance:
                with self._phase_lock:
                    recent_wr = np.mean(list(self.phase_wins)[-50:]) * 100 if len(self.phase_wins) >= 10 else 0
                    recent_vp = np.mean(list(self.phase_vps)[-50:]) if len(self.phase_vps) >= 10 else 0
                print(f"\n  ⚠ FORCED ADVANCE from {phase_name} after {phase_game_count} games "
                      f"(stuck: WR={recent_wr:.1f}%, VP={recent_vp:.1f})")
            if (force_advance or
                (phase_game_count >= min_games_per_phase and
                 current_phase < len(phases) - 1 and
                 self.should_advance_curriculum(mix_prob, primary_ai, secondary_ai, vp_threshold))):

                with self._phase_lock:
                    print(f"\n  ★ ADVANCING from {phase_name} to {phases[current_phase + 1][5]}")
                    print(f"    Games in phase: {phase_game_count}")
                    print(f"    Recent WR: {np.mean(list(self.phase_wins)[-50:])*100:.1f}%")
                    print(f"    Recent VP: {np.mean(list(self.phase_vps)[-50:]):.1f}\n")

                # Clear old experiences but keep enough to prevent catastrophic forgetting
                # 50% retention to reduce forgetting on phase transitions
                self.replay_buffer.clear_old(keep_fraction=0.5)

                # Boost entropy for new phase - explore new strategies against harder opponent
                # Use moderate boost (not max) to avoid destabilizing learned policy too much
                moderate_boost = min(self.current_entropy_coef * 2.0, self.max_entropy_coef)
                self.current_entropy_coef = max(moderate_boost, 0.05)  # At least 0.05, at most max
                self.entropy_history.clear()

                current_phase += 1
                phase_game_count = 0
                with self._phase_lock:
                    self.phase_wins.clear()
                    self.phase_vps.clear()
                # Reset per-phase entropy schedule and stagnation tracking
                phase_start_entropy = self.current_entropy_coef
                stagnation_wr_history = []
                last_stagnation_check = 0
                stagnation_lr_cuts = 0

                # Save checkpoint
                self.save(f"{save_path}_phase{current_phase}.pt")

            # Periodic saves, LR decay, and entropy decay (every 1000 games)
            if game_num // 1000 > last_save_game // 1000:
                last_save_game = game_num
                ckpt_path = f"{save_path}_game{game_num}.pt"
                self.save(ckpt_path)
                # Register new checkpoint into self-play pool
                if self.self_play_pool is not None:
                    self.self_play_pool.add_checkpoint(ckpt_path)
                # Apply LR decay
                if self.lr_decay < 1.0:
                    for pg in self.optimizer.param_groups:
                        old_lr = pg['lr']
                        pg['lr'] = max(1e-5, pg['lr'] * self.lr_decay)
                        if pg['lr'] != old_lr:
                            print(f"    LR decay: {old_lr:.2e} → {pg['lr']:.2e}")
                # Apply entropy decay (reduce exploration over time)
                if self.entropy_decay < 1.0:
                    old_ec = self.current_entropy_coef
                    # Decay entropy coef but keep above minimum
                    self.current_entropy_coef = max(self.min_entropy_coef,
                                                     self.current_entropy_coef * self.entropy_decay)
                    if self.current_entropy_coef != old_ec:
                        print(f"    Entropy decay: {old_ec:.3f} → {self.current_entropy_coef:.3f}")
                # Anneal BC coef to 0 over the first 50% of total training
                if self.bc_coef > 0:
                    bc_half = total_games * 0.5
                    self.bc_coef = max(0.0, self.bc_coef * (1.0 - (1000 / bc_half)))

        self.save(f"{save_path}_final.pt")

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Total games: {total_games}")
        print(f"Overall win rate: {total_wins/total_games*100:.1f}%")
        print(f"Final phase: {phases[current_phase][3]}")
        print(f"Time: {total_time/60:.1f} min")
        print("=" * 70)

    def save(self, path):
        torch.save({
            'model_state_dict': self.network_wrapper.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_played': self.games_played,
            'entropy_coef': self.current_entropy_coef,
            'grad_norm': self.current_grad_norm,
            'hidden_dim': self.network_wrapper.policy.hidden_dim,
        }, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-games', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--train-frequency', type=int, default=5)
    parser.add_argument('--train-steps', type=int, default=15)
    parser.add_argument('--min-games-per-phase', type=int, default=2000,
                        help='Minimum games before curriculum can advance')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to existing model to continue training')
    parser.add_argument('--reward-mode', type=str, default='pbrs_fixed',
                        choices=['sparse', 'vp_only', 'simplified', 'pbrs_fixed'],
                        help='Reward mode: sparse, vp_only, simplified, or pbrs_fixed (default)')
    parser.add_argument('--lr-decay', type=float, default=1.0,
                        help='Learning rate decay multiplier per 1000 games (default: 1.0 = no decay, try 0.95 for fine-tuning)')
    parser.add_argument('--value-weight', type=float, default=0.5,
                        help='Weight for value loss (default: 0.5; raised from 0.1 to fix plateau where value head was not converging)')
    parser.add_argument('--entropy-decay', type=float, default=1.0,
                        help='Entropy coefficient decay per 1000 games (default: 1.0 = no decay, try 0.95 to reduce exploration over time)')
    parser.add_argument('--parallel-games', type=int, default=8,
                        help='Number of games to run in parallel (default: 8)')
    parser.add_argument('--buffer-size', type=int, default=200000,
                        help='Replay buffer size (default: 200000). Increase for large batch sizes.')
    parser.add_argument('--start-phase', type=int, default=0,
                        help='Phase index to start from (default: 0). Use --list-phases to see all phases.')
    parser.add_argument('--list-phases', action='store_true',
                        help='List all curriculum phases and exit')
    parser.add_argument('--num-players', type=int, default=4, choices=[2, 3, 4],
                        help='Number of players: 2 for 1v1 training, 4 for standard (default: 4)')
    parser.add_argument('--bc-coef', type=float, default=0.1,
                        help='Behavioral cloning coefficient: guides vertex/edge/trade heads toward '
                             'Strong AI choices. Annealed to 0 over first 50%% of training. (default: 0.1)')
    parser.add_argument('--save-path', type=str, default='models/curriculum_v3_stable',
                        help='Path prefix for saving models (default: models/curriculum_v3_stable)')
    parser.add_argument('--self-play-pool', type=str, default=None,
                        help='Glob pattern for past checkpoints to use as self-play opponents, '
                             'e.g. "models/v3_overnight2_game*.pt".  When provided, self-play '
                             'phases are appended automatically to the curriculum.')
    parser.add_argument('--self-play-pool-size', type=int, default=8,
                        help='Number of past-checkpoint networks to keep loaded for self-play (default: 8)')
    parser.add_argument('--specialist-mix-rate', type=float, default=0.0,
                        help='During self-play, probability each opponent turn uses a random specialist AI (0.0-1.0). '
                             'Prevents blind spots against strategies like dev-card spamming or balanced aggression.')
    parser.add_argument('--mcts', action='store_true',
                        help='Use MCTS for action selection during data collection (slower but higher quality)')
    parser.add_argument('--mcts-sims', type=int, default=30,
                        help='Number of MCTS simulations per move (default: 30)')
    parser.add_argument('--mcts-cpuct', type=float, default=1.5,
                        help='MCTS exploration constant (default: 1.5)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Network backbone hidden dimension (default: 256, use 512 for ~3x capacity)')
    args = parser.parse_args()

    # List phases if requested
    if args.list_phases:
        # Format: (primary_ai, secondary_ai, mix_prob, vp_to_win, vp_threshold, name)
        if args.num_players == 2:
            # MUST match the actual phases used in train() method!
            phases = [
                ('passive', None, 1.0, 6, 5.0, "1v1 Passive 6VP"),
                ('passive', None, 1.0, 10, 7.5, "1v1 Passive"),
                ('truly_random', 'passive', 0.5, 10, 6.5, "1v1 TrulyRandom/Passive"),
                ('truly_random', None, 1.0, 10, 6.0, "1v1 TrulyRandom"),
                ('weighted_random', 'truly_random', 0.5, 10, 5.5, "1v1 WeightedRandom/TrulyRandom"),
                ('weighted_random', None, 1.0, 10, 5.5, "1v1 WeightedRandom"),
                ('very_weak', 'weighted_random', 0.5, 10, 6.2, "1v1 VeryWeak/WeightedRandom"),
                ('weak', 'very_weak', 0.5, 10, 5.5, "1v1 Weak/VeryWeak"),
                ('medium', 'weak', 0.5, 10, 5.0, "1v1 Medium/Weak"),
                ('strong', 'medium', 0.5, 10, 5.5, "1v1 Strong/Medium"),
                ('strong', None, 1.0, 10, 5.5, "1v1 Strong"),
                # Specialist diversity — expose agent to distinct strategies
                ('city_rusher', 'strong', 0.5, 10, 5.5, "1v1 CityRusher/Strong"),
                ('road_blocker', 'strong', 0.5, 10, 5.5, "1v1 RoadBlocker/Strong"),
                ('dev_card_spammer', 'strong', 0.5, 10, 5.5, "1v1 DevCardSpammer/Strong"),
                ('balanced_aggressor', 'strong', 0.5, 10, 5.5, "1v1 BalancedAggressor/Strong"),
                ('port_specialist', 'strong', 0.5, 10, 5.5, "1v1 PortSpecialist/Strong"),
                ('curriculum_mix', 'strong', 0.5, 10, 5.5, "1v1 CurriculumMix/Strong"),
                ('curriculum_mix', None, 1.0, 10, 999, "1v1 CurriculumMix FINAL"),
            ]
            print("\n1v1 OPTIMIZED Curriculum Phases (v5 - 10VP throughout):")
            print("-" * 70)
            print("  Phase 0:    Passive opponent (never builds)")
            print("  Phase 1-2:  TrulyRandom (15% build rate)")
            print("  Phase 3-4:  WeightedRandom bridge")
            print("  Phase 5-10: Difficulty progression (VeryWeak → Strong)")
            print("  Phase 11-15: Specialist diversity (CityRusher, RoadBlocker, etc.)")
            print("  Phase 16-17: Curriculum mix (50% self-play / 25% strong / 25% medium)")
            print("-" * 70)
        else:
            phases = [
                ('random', None, 1.0, 4, 2.8, "Random 4VP"),
                ('random', None, 1.0, 5, 3.2, "Random 5VP"),
                ('random', None, 1.0, 6, 3.6, "Random 6VP"),
                ('random', None, 1.0, 7, 4.0, "Random 7VP"),
                ('random', None, 1.0, 8, 4.4, "Random 8VP"),
                ('random', None, 1.0, 9, 4.8, "Random 9VP"),
                ('random', None, 1.0, 10, 5.2, "Random 10VP"),
                ('very_weak', 'random', 0.5, 10, 3.5, "VeryWeak/Random Mix"),
                ('very_weak', None, 1.0, 10, 3.0, "VeryWeak 10VP"),
                ('weak', 'very_weak', 0.5, 10, 3.0, "Weak/VeryWeak Mix"),
                ('weak', None, 1.0, 10, 2.8, "Weak 10VP"),
                ('medium', 'weak', 0.5, 10, 2.8, "Medium/Weak Mix"),
                ('medium', None, 1.0, 10, 2.5, "Medium 10VP"),
                ('strong', 'medium', 0.5, 10, 2.5, "Strong/Medium Mix"),
                ('strong', 'medium', 0.5, 10, 999, "Strong/Medium FINAL"),
            ]
            print("\n4-Player Curriculum Phases (WITH OPPONENT MIXING):")
            print("-" * 70)
            print("  Phase 1: Learn game length (4VP -> 10VP) vs Random")
            print("  Phase 2: Gradual opponent mixing (still win some games = learning signal)")
            print("-" * 70)
        for i, (primary, secondary, mix, vp_win, vp_thresh, name) in enumerate(phases):
            mix_str = f"{int(mix*100)}% {primary}" if secondary is None else f"{int(mix*100)}% {primary} / {int((1-mix)*100)}% {secondary}"
            print(f"  {i:2d}: {name:22s} VP:{vp_win:2d} thresh:{vp_thresh:4.1f} ({mix_str})")
        print("-" * 70)
        print("\nUse --start-phase N to start from phase N")
        print("Use --num-players 2 to see 1v1 phases")
        exit(0)

    # Print training mode
    if args.num_players == 2:
        # Show auto-tuned values (must match logic in __init__ lines 393-399)
        auto_parallel = 32 if args.parallel_games == 8 else args.parallel_games
        # MCTS caps parallelism (must match logic in __init__)
        if args.mcts and auto_parallel > 8:
            auto_parallel = 8
        auto_buffer = 50000 if args.buffer_size == 200000 else args.buffer_size
        print("\n" + "=" * 60)
        print("1v1 TRAINING MODE (Colonist-style)")
        print("=" * 60)
        print("  - 2 players (you vs 1 opponent)")
        print("  - Clearer learning signal than 4-player")
        print("  - Optimized curriculum: 10 phases (vs 15 in 4-player)")
        print("  AUTO-TUNED SETTINGS:")
        print(f"    - Parallel games: {auto_parallel} (vs 8 default)")
        print(f"    - Buffer size: {auto_buffer:,} (vs 200,000 default)")
        print("=" * 60 + "\n")

    # Build self-play pool if requested
    sp_pool = None
    if args.self_play_pool:
        sp_pool = SelfPlayPool(
            checkpoint_glob=args.self_play_pool,
            pool_size=args.self_play_pool_size,
            device='cpu',   # keep GPU free for training
        )

    trainer = CurriculumTrainerV3(
        model_path=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reward_mode=args.reward_mode,
        lr_decay=args.lr_decay,
        value_weight=args.value_weight,
        entropy_decay=args.entropy_decay,
        num_parallel_games=args.parallel_games,
        buffer_size=args.buffer_size,
        num_players=args.num_players,
        bc_coef=args.bc_coef,
        self_play_pool=sp_pool,
        specialist_mix_rate=args.specialist_mix_rate,
        use_mcts=args.mcts,
        mcts_simulations=args.mcts_sims,
        mcts_c_puct=args.mcts_cpuct,
        hidden_dim=args.hidden_dim,
    )
    trainer.train(
        total_games=args.total_games,
        train_frequency=args.train_frequency,
        train_steps=args.train_steps,
        min_games_per_phase=args.min_games_per_phase,
        start_phase=args.start_phase,
        save_path=args.save_path
    )
