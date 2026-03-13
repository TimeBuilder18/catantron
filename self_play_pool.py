"""
Self-Play Opponent Pool for Catan AI

Maintains a rotating pool of past-checkpoint networks and uses them as
opponents during training.  Replaces or supplements scripted AIs once
the agent is strong enough to benefit from playing itself.

Thread-safe: multiple parallel game threads each get their own opponent
env via threading.local(), while the loaded network weights are shared
read-only across threads.

Usage (in training loop):
    pool = SelfPlayPool("models/v3_overnight2_game*.pt", pool_size=8)

    # Pass pool into play_game so opponent turns use it:
    trainer.play_game(..., primary_ai='self_play', self_play_pool=pool)
"""

import glob
import random
import threading
import torch
import numpy as np

from catan_env_pytorch import CatanEnv


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight CatanEnv subclass that can share an existing game_env
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Legacy observation adapter (575 → 427 features) for old checkpoints
# ──────────────────────────────────────────────────────────────────────────────

# Old observation layout constants (427 features)
_OLD_NON_TILE_DIM = 46
_OLD_TILE_FEATURE_DIM = 3
_OLD_NUM_TILES = 19
_OLD_TOTAL_OBS_DIM = 427

# Maps one-hot position → old ordinal value.
# One-hot order: forest=0, hill=1, field=2, mountain=3, pasture=4, desert=5
# Old ordinal:   forest=1, hill=2,  field=3, mountain=4, pasture=5, desert=0
_ONEHOT_TO_ORDINAL = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])


def _convert_obs_575_to_427(obs_575):
    """Convert a batch of new 575-feature observations to old 427-feature format.

    Args:
        obs_575: (batch, 575) tensor in the new observation layout

    Returns:
        obs_427: (batch, 427) tensor in the old observation layout
    """
    batch = obs_575.shape[0]
    device = obs_575.device

    # Non-tile features [0:46] — unchanged between old and new
    non_tile = obs_575[:, :46]

    # Tile features: new [61:251] (19×10) → old (19×3) ordinal encoding
    tile_block = obs_575[:, 61:251].view(batch, 19, 10)
    onehot = tile_block[:, :, :6]                              # (batch, 19, 6)
    ordinal = (onehot * _ONEHOT_TO_ORDINAL.to(device)).sum(-1, keepdim=True)  # (batch, 19, 1)
    number = tile_block[:, :, 6:7] * 12.0                     # denormalize
    has_robber = tile_block[:, :, 8:9]                         # position 8 in new layout
    old_tiles = torch.cat([ordinal, number, has_robber], dim=-1)  # (batch, 19, 3)
    old_tiles_flat = old_tiles.view(batch, 57)

    # Port features: new [251:269] → old [103:121] — same format
    ports = obs_575[:, 251:269]

    # Positional features: new [269:575] → old [121:427] — same format
    positional = obs_575[:, 269:575]

    return torch.cat([non_tile, old_tiles_flat, ports, positional], dim=-1)


class _LegacyPolicyWrapper:
    """Wraps an old-format CatanPolicy to accept new 575-feature observations.

    Converts observations from 575→427 before forwarding to the legacy network.
    Exposes the same interface as CatanPolicy for use in play_turn().
    """

    def __init__(self, policy):
        self.policy = policy

    def get_action_and_value(self, obs, action_mask, vertex_mask=None, edge_mask=None):
        obs_427 = _convert_obs_575_to_427(obs)
        return self.policy.get_action_and_value(obs_427, action_mask, vertex_mask, edge_mask)

    def eval(self):
        self.policy.eval()
        return self


class _SelfPlayEnv(CatanEnv):
    """CatanEnv that can be wired to an external game_env (shared game state)."""

    def wire_to_game(self, game_env):
        """
        Point this env at `game_env` so _get_obs() sees the live game state.
        Must be called before every opponent turn because the underlying game
        object is always the one owned by the main training env.
        """
        self.game_env = game_env
        # Invalidate all caches that depend on the game object
        self._port_access_cache = None
        self._port_access_cache_turn = -1
        self._settlement_vertices = None
        self._city_vertices = None


# ──────────────────────────────────────────────────────────────────────────────
# Self-play pool
# ──────────────────────────────────────────────────────────────────────────────

class SelfPlayPool:
    """
    Pool of past-checkpoint networks used as self-play opponents.

    Checkpoint sampling strategy
    ─────────────────────────────
    With 200 checkpoints spanning the whole training history, we want:
      • Diversity  — include some older, weaker versions so the agent
                     doesn't overfit to its own current play-style.
      • Recency    — weight toward recent checkpoints so the opponent is
                     roughly at the agent's current skill level.

    Default: 70 % of pool slots are drawn from the most recent half of
    checkpoints, 30 % from the older half.  The pool is refreshed every
    `refresh_every` games so new checkpoints (saved during the run) are
    automatically discovered and incorporated.

    Args:
        checkpoint_glob  : glob pattern to find .pt files,
                           e.g. "models/v3_overnight2_game*.pt"
        pool_size        : number of networks kept loaded simultaneously
                           (higher = more diverse but more RAM / VRAM)
        device           : 'cpu' recommended — save GPU memory for training
        recency_bias     : fraction of pool slots drawn from recent half
        refresh_every    : resample + reload pool every N games
    """

    def __init__(self, checkpoint_glob: str, pool_size: int = 8,
                 device: str = 'cpu', recency_bias: float = 0.7,
                 refresh_every: int = 500):
        self.checkpoint_glob = checkpoint_glob
        self.pool_size = pool_size
        self.device = torch.device(device)
        self.recency_bias = recency_bias
        self.refresh_every = refresh_every

        self._lock = threading.Lock()           # guard pool state
        self._thread_local = threading.local()  # per-thread opponent envs

        self._networks: dict = {}       # path -> CatanPolicy (loaded, eval mode)
        self._active_paths: list = []   # currently active subset
        self._games_since_refresh = 0

        all_paths = self._discover()
        if not all_paths:
            raise ValueError(f"[SelfPlayPool] No checkpoints found: {checkpoint_glob}")
        self.all_paths = all_paths
        print(f"[SelfPlayPool] Found {len(all_paths)} checkpoints — "
              f"pool_size={pool_size}, recency_bias={recency_bias:.0%}")
        self._refresh_pool(self.all_paths)

    # ── discovery & loading ──────────────────────────────────────────────────

    def _discover(self) -> list:
        return sorted(glob.glob(self.checkpoint_glob))

    def _sample_paths(self, all_paths: list) -> list:
        """Return a list of `pool_size` paths with recency bias."""
        n = len(all_paths)
        k = min(self.pool_size, n)
        mid = n // 2
        recent = all_paths[mid:]
        older  = all_paths[:mid]

        k_recent = max(1, round(k * self.recency_bias))
        k_older  = k - k_recent

        selected = []
        if recent:
            selected += random.sample(recent, min(k_recent, len(recent)))
        if older and k_older > 0:
            selected += random.sample(older, min(k_older, len(older)))
        return selected

    def _load_policy(self, path: str):
        from network_gpu import CatanPolicy, TILE_FEATURE_DIM
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        hidden_dim = ckpt.get('hidden_dim', None)
        if hidden_dim is None:
            # Infer from weight shapes for old checkpoints without metadata
            fc1_w = ckpt['model_state_dict'].get('fc1.weight')
            if fc1_w is not None:
                hidden_dim = fc1_w.shape[0]
            else:
                hidden_dim = 256

        # Check observation layout compatibility
        tile_embed_w = ckpt['model_state_dict'].get('tile_encoder.tile_embedding.weight')
        if tile_embed_w is not None and tile_embed_w.shape[1] != TILE_FEATURE_DIM:
            # Old checkpoint — load with legacy architecture + adapter
            return self._load_legacy_policy(ckpt, hidden_dim,
                                            tile_dim=tile_embed_w.shape[1])

        policy = CatanPolicy(device=self.device, hidden_dim=hidden_dim)
        policy.load_state_dict(ckpt['model_state_dict'])
        policy.eval()
        return policy

    def _load_legacy_policy(self, ckpt, hidden_dim, tile_dim):
        """Load an old-format checkpoint and wrap it with an observation adapter."""
        from network_gpu import CatanPolicy
        # Build a CatanPolicy with old architecture dimensions
        # We need to temporarily construct a policy that matches the old weights.
        # The old layout: NON_TILE_DIM=46, TILE_FEATURE_DIM=3, TILE_EMBED_DIM=32,
        #   PORT_START_IDX=103, PORT_END_IDX=121, POSITIONAL_START_IDX=121, TOTAL_OBS_DIM=427
        import network_gpu as ng
        saved = (ng.NON_TILE_DIM, ng.TILE_START_IDX, ng.TILE_END_IDX, ng.TILE_FEATURE_DIM,
                 ng.PORT_START_IDX, ng.PORT_END_IDX, ng.POSITIONAL_START_IDX,
                 ng.TOTAL_OBS_DIM, ng.TILE_EMBED_DIM)
        try:
            ng.NON_TILE_DIM = 46
            ng.TILE_START_IDX = 46
            ng.TILE_END_IDX = 103
            ng.TILE_FEATURE_DIM = tile_dim  # typically 3
            ng.PORT_START_IDX = 103
            ng.PORT_END_IDX = 121
            ng.POSITIONAL_START_IDX = 121
            ng.TOTAL_OBS_DIM = 427
            ng.TILE_EMBED_DIM = 32
            policy = CatanPolicy(device=self.device, hidden_dim=hidden_dim)
            # TileAttentionEncoder default params are bound at import time,
            # so module patching doesn't affect them — replace explicitly
            from network_gpu import TileAttentionEncoder
            policy.tile_encoder = TileAttentionEncoder(
                tile_feature_dim=tile_dim, embed_dim=32, output_dim=128
            ).to(self.device)
            policy.load_state_dict(ckpt['model_state_dict'])
            policy.eval()
        finally:
            (ng.NON_TILE_DIM, ng.TILE_START_IDX, ng.TILE_END_IDX, ng.TILE_FEATURE_DIM,
             ng.PORT_START_IDX, ng.PORT_END_IDX, ng.POSITIONAL_START_IDX,
             ng.TOTAL_OBS_DIM, ng.TILE_EMBED_DIM) = saved

        print(f"[SelfPlayPool] Loaded legacy checkpoint (tile_dim={tile_dim}) with adapter")
        return _LegacyPolicyWrapper(policy)

    def _refresh_pool(self, all_paths: list):
        selected = self._sample_paths(all_paths)

        # Evict networks no longer selected
        keep = set(selected)
        for path in list(self._networks):
            if path not in keep:
                del self._networks[path]

        # Load new ones
        loaded = 0
        for path in selected:
            if path not in self._networks:
                try:
                    self._networks[path] = self._load_policy(path)
                    loaded += 1
                except Exception as e:
                    print(f"[SelfPlayPool] Warning: could not load {path}: {e}")

        self._active_paths = [p for p in selected if p in self._networks]
        self._games_since_refresh = 0

        n = len(all_paths)
        mid = n // 2
        n_recent = sum(1 for p in self._active_paths if p in all_paths[mid:])
        n_older  = len(self._active_paths) - n_recent
        if loaded > 0:
            print(f"[SelfPlayPool] Refreshed: {len(self._active_paths)} networks "
                  f"({n_recent} recent, {n_older} older); loaded {loaded} new")

    # ── public API ───────────────────────────────────────────────────────────

    def add_checkpoint(self, path: str):
        """Register a newly saved checkpoint so it can enter the pool on next refresh.

        This is called by the training loop after saving a periodic checkpoint,
        so self-play opponents include the agent's own recent checkpoints — not
        just the initial seed pool.
        """
        with self._lock:
            if path not in self.all_paths:
                self.all_paths.append(path)
                self.all_paths.sort()
                print(f"[SelfPlayPool] Added checkpoint to pool: {path}  "
                      f"(total candidates: {len(self.all_paths)})")

    def notify_game_done(self):
        """Call after each training game.  Triggers pool refresh periodically."""
        with self._lock:
            self._games_since_refresh += 1
            if self._games_since_refresh >= self.refresh_every:
                # Re-discover checkpoints (new ones may have been saved)
                all_paths = self._discover()
                if all_paths:
                    self.all_paths = all_paths
                self._refresh_pool(self.all_paths)

    def pick_network(self):
        """Return a random (path, policy) from the active pool."""
        with self._lock:
            if not self._active_paths:
                return None, None
            path = random.choice(self._active_paths)
            return path, self._networks[path]

    def play_turn(self, main_env, opponent_player_id: int) -> bool:
        """
        Play a complete turn for `opponent_player_id` using a sampled network.

        `main_env` is the training env (CatanEnv or any wrapper that exposes
        `.game_env`).  The shared game state is read directly — no copy is made.

        Returns True when the turn is done (or on any error/fallback).
        The caller should fall back to a scripted AI if this returns False.
        """
        _, policy = self.pick_network()
        if policy is None:
            return False  # pool empty — caller falls back to scripted AI

        # Resolve game_env through wrapper layers (SimplifiedRewardWrapper etc.)
        base = main_env
        while not hasattr(base, 'game_env'):
            base = base.env

        game_env = base.game_env
        game     = game_env.game
        vp_to_win   = getattr(base, 'victory_points_to_win', 10)
        num_players = getattr(base, 'num_players', 2)

        # Get (or create) a thread-local opponent env for this player slot
        opp_env = self._get_thread_opp_env(opponent_player_id, vp_to_win, num_players)
        opp_env.wire_to_game(game_env)

        # Drive the opponent's turn until it ends or the game is over
        max_steps = 200
        for _ in range(max_steps):
            if game.check_victory_conditions() is not None:
                return True
            current_player = game.get_current_player()
            if game.players[opponent_player_id] != current_player:
                return True   # turn finished

            obs = opp_env._get_obs()

            obs_t = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)
            am_t  = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(self.device)
            vm_t  = torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(self.device)
            em_t  = torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(self.device)

            with torch.no_grad():
                result = policy.get_action_and_value(obs_t, am_t, vm_t, em_t)
            # get_action_and_value returns: action, vertex, edge, tg, tk, lp_a, lp_v, lp_e, lp_tg, lp_tk, value, entropy
            action_id = int(result[0].item())
            vertex_id = int(result[1].item())
            edge_id   = int(result[2].item())
            tg        = int(result[3].item())
            tk        = int(result[4].item())

            _, _, terminated, truncated, info = opp_env.step(
                action_id, vertex_id, edge_id, tg, tk
            )
            if terminated or truncated or info.get('illegal_action'):
                return True

        return True   # hit step limit

    # ── internals ────────────────────────────────────────────────────────────

    def _get_thread_opp_env(self, player_id: int,
                             vp_to_win: int, num_players: int) -> '_SelfPlayEnv':
        """Return a thread-local _SelfPlayEnv for the given player slot."""
        if not hasattr(self._thread_local, 'opp_envs'):
            self._thread_local.opp_envs = {}
        key = (player_id, vp_to_win, num_players)
        if key not in self._thread_local.opp_envs:
            self._thread_local.opp_envs[key] = _SelfPlayEnv(
                player_id=player_id,
                victory_points_to_win=vp_to_win,
                num_players=num_players,
            )
        return self._thread_local.opp_envs[key]
