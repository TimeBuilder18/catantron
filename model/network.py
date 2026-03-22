"""
Neural network for our 1v1 Catan AI agent.

Uses a multi-head architecture: one shared backbone feeds into separate output
heads for action type, vertex placement, edge placement, and trading. The idea
is that "what to do" and "where to do it" are related but distinct decisions —
a single head couldn't really handle all of them well. We tried a single flat
output early on and the agent kept trying to build roads in the ocean.

We also tried transformers for the full architecture but feedforward works
better for Catan since each decision is basically independent — there's no
sequence to attend to, unlike chess where move order matters. The one exception
is the tile encoder, where attention over the 19 hex tiles actually helps
because tiles have spatial relationships (adjacent resources, ports, etc).

Architecture overview:
  observation (575 features)
      |
  split into tile features + player/game context
      |                          |
  TileAttentionEncoder     Linear encoder
      |                          |
      +---- concat (256) -------+
                 |
         FC backbone (3 layers)
                 |
     +-----+-----+------+------+-----+
     |     |     |      |      |     |
  action vertex edge  trade  trade value
  (14)   (54)  (72)  give(5) get(5) (1)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Observation layout constants (for 575-feature observation)
# Base features (269):
#   [0:46]    Non-tile features (game state, resources, structures, dev cards, stats, opponents)
#   [46:61]   Strategic features (production pips, turn, road lengths, VP gap, affordability)
#   [61:251]  Tile features (19 tiles × 10 = 190)
#   [251:269] Port features (9 ports × 2 = 18)
# Positional features (306):
#   [269:323] My settlement positions (54)
#   [323:377] My city positions (54)
#   [377:431] Opponent building positions (54)
#   [431:503] My road positions (72)
#   [503:575] Opponent road positions (72)

NON_TILE_DIM = 61          # 46 base + 15 strategic features
TILE_START_IDX = 61
TILE_END_IDX = 251         # 61 + 19*10
NUM_TILES = 19
TILE_FEATURE_DIM = 10      # one-hot(6) + number + pips + robber + building_strength
PORT_START_IDX = 251
PORT_END_IDX = 269
POSITIONAL_START_IDX = 269
TOTAL_OBS_DIM = 575

# Tile attention hyperparameters
# 48 worked better than 32 — with 10 features per tile (up from 6 in our
# old obs format), 32 was too compressed and the agent missed resource combos
TILE_EMBED_DIM = 48
# 4 heads lets different heads specialize (e.g. one for resource clusters,
# one for number distributions). 2 was too few, 8 didn't help and was slower.
NUM_ATTENTION_HEADS = 4


class TileAttentionEncoder(nn.Module):
    """
    Processes the 19 Catan tiles through multi-head self-attention.
    Each tile attends to all other tiles to learn spatial board context.

    This is the one place where attention actually helps — tiles have real
    spatial relationships (a wheat next to an ore matters more than either
    alone). A plain MLP over flattened tiles couldn't learn that a 6-wheat
    adjacent to an 8-ore is a great settlement spot. Attention lets the
    model figure out which tile combos matter.

    Based on the settlers-rl architecture (Charlesworth, 2021).
    """

    def __init__(self, tile_feature_dim=TILE_FEATURE_DIM, embed_dim=TILE_EMBED_DIM,
                 num_heads=NUM_ATTENTION_HEADS, num_tiles=NUM_TILES, output_dim=128):
        super(TileAttentionEncoder, self).__init__()

        self.num_tiles = num_tiles
        self.embed_dim = embed_dim

        # Embed each tile's raw features to embed_dim
        self.tile_embedding = nn.Linear(tile_feature_dim, embed_dim)

        # Multi-head self-attention: tiles attend to each other
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.attn_ln = nn.LayerNorm(embed_dim)

        # Per-tile feedforward after attention
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.ffn_ln = nn.LayerNorm(embed_dim)

        # Flatten all 19 tile embeddings and squeeze down to 128
        # This bottleneck forces the model to learn a compact board summary
        # instead of memorizing individual tile positions
        self.compress = nn.Linear(num_tiles * embed_dim, output_dim)
        self.compress_ln = nn.LayerNorm(output_dim)

    def forward(self, tile_features):
        """
        Args:
            tile_features: (batch, num_tiles, tile_feature_dim) = (batch, 19, 10)

        Returns:
            tile_context: (batch, 128) compressed tile representation
        """
        # Embed tiles: (batch, 19, 10) -> (batch, 19, 48)
        x = self.tile_embedding(tile_features)

        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        x = self.attn_ln(x + attn_output)  # Residual + LayerNorm

        # Per-tile FFN with residual connection
        ffn_output = F.relu(self.ffn(x))
        x = self.ffn_ln(x + ffn_output)  # Residual + LayerNorm

        # Flatten: (batch, 19, 48) -> (batch, 912)
        x = x.view(x.size(0), -1)

        # Compress to output_dim: (batch, 912) -> (batch, output_dim)
        tile_context = F.relu(self.compress_ln(self.compress(x)))

        return tile_context


class CatanPolicy(nn.Module):
    """
    The main policy network for our Catan agent. Takes in the full game
    observation (575 features) and outputs probability distributions over
    all the different decisions the agent can make.

    Multi-head design: instead of one giant output, we have separate heads
    for action type, vertex placement, edge placement, and trading. This
    way the network can learn "I should build a settlement" independently
    from "vertex 23 is a good spot." Early versions with a single flat
    output were basically untrainable — the action space is just too big
    for one softmax to handle (~150 possible moves per turn).

    The value head is standard PPO — estimates how good the current board
    state is so we can compute advantages for the policy gradient.
    """

    def __init__(self, device=None, hidden_dim=256):
        super(CatanPolicy, self).__init__()

        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.hidden_dim = hidden_dim  # 256 default, 512 for the "big" model variant
        # Both encoders output 128 each (256 total). We keep this fixed so we
        # can transfer encoder weights between models with different backbone
        # widths — the encoder learns board understanding which is expensive
        # to retrain, but the backbone retrains fast.
        encoder_dim = 128

        # Store obs layout as instance attrs so legacy policies (created with
        # patched module constants) retain their own layout in forward().
        self._non_tile_dim = NON_TILE_DIM
        self._tile_start = TILE_START_IDX
        self._tile_end = TILE_END_IDX
        self._num_tiles = NUM_TILES
        self._tile_feat_dim = TILE_FEATURE_DIM
        self._port_start = PORT_START_IDX
        self._port_end = PORT_END_IDX
        self._pos_start = POSITIONAL_START_IDX

        print(f" Using device: {self.device}")

        # === Tile Attention Encoder ===
        # Pass dims explicitly so they're evaluated at call time (not bound
        # at class definition like TileAttentionEncoder's default params).
        self.tile_encoder = TileAttentionEncoder(
            tile_feature_dim=TILE_FEATURE_DIM,
            embed_dim=TILE_EMBED_DIM,
            output_dim=encoder_dim,
        )

        # === Player/Game Context Encoder ===
        # Non-tile features: game state + strategic (61) + ports (18) + positional (306) = 385
        player_context_dim = NON_TILE_DIM + (PORT_END_IDX - PORT_START_IDX) + (TOTAL_OBS_DIM - POSITIONAL_START_IDX)
        self.player_embed = nn.Linear(player_context_dim, encoder_dim)
        self.player_ln = nn.LayerNorm(encoder_dim)

        # === Projection from encoder output (256) to backbone width ===
        concat_dim = encoder_dim * 2  # 256 always
        if hidden_dim != concat_dim:
            self.projection = nn.Linear(concat_dim, hidden_dim)
            self.projection_ln = nn.LayerNorm(hidden_dim)
        else:
            self.projection = None

        # === Shared backbone ===
        # 3 FC layers with LayerNorm. We tried BatchNorm first but it was
        # unstable with PPO's small effective batch sizes during rollouts.
        # LayerNorm doesn't depend on batch stats so it's more stable.
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # === Output Heads ===
        # Each head handles a different part of the hierarchical decision.
        # The action head picks WHAT to do (build, trade, end turn, etc.)
        # then the relevant location/trade head picks the details.
        self.policy_head = nn.Linear(hidden_dim, 14)          # 14 action types (build_settlement, build_road, etc.)
        self.location_head_vertex = nn.Linear(hidden_dim, 54) # 54 vertices — for settlements and cities
        self.location_head_edge = nn.Linear(hidden_dim, 72)   # 72 edges — for roads
        self.trade_give_head = nn.Linear(hidden_dim, 5)       # which of 5 resources to give
        self.trade_get_head = nn.Linear(hidden_dim, 5)        # which of 5 resources to get
        # Value head — PPO's critic. Single scalar estimate of how likely
        # we are to win from this state. Shares the backbone with the policy
        # heads so it learns board understanding for free.
        self.value_head = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    def forward(self, obs, action_mask=None, vertex_mask=None, edge_mask=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        # === Split observation into tile and non-tile features ===
        # Uses instance attrs so legacy policies slice their 427-dim obs correctly.
        game_features = obs[:, :self._non_tile_dim]

        tile_features_flat = obs[:, self._tile_start:self._tile_end]
        tile_features = tile_features_flat.view(-1, self._num_tiles, self._tile_feat_dim)

        port_features = obs[:, self._port_start:self._port_end]

        positional_features = obs[:, self._pos_start:]

        # === Process through encoders ===
        # Tile attention: (batch, 19, 10) -> (batch, 128)
        tile_context = self.tile_encoder(tile_features)

        # Player context: concatenate non-tile features and embed
        # (batch, 61) + (batch, 18) + (batch, 306) = (batch, 385) -> (batch, 128)
        player_features = torch.cat([game_features, port_features, positional_features], dim=-1)
        player_context = F.relu(self.player_ln(self.player_embed(player_features)))

        # === Combine and process ===
        # (batch, 128) + (batch, 128) = (batch, 256)
        x = torch.cat([tile_context, player_context], dim=-1)

        # Project to backbone width if needed (256 → hidden_dim)
        if self.projection is not None:
            x = F.relu(self.projection_ln(self.projection(x)))

        # FC backbone
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        # === Action head with masking ===
        # Masking is critical — without it the agent tries illegal moves
        # constantly. We set illegal action logits to -inf before softmax
        # so they get probability ~0. The all-zeros check prevents NaN
        # (shouldn't happen in normal play, but does during edge cases in testing).
        action_logits = self.policy_head(x)
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.FloatTensor(action_mask)
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            action_mask = action_mask.to(self.device)

            # Check if mask is all zeros (would cause NaN in softmax)
            if action_mask.sum() == 0:
                action_probs = torch.ones_like(action_logits) / action_logits.shape[-1]
            else:
                action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))
                action_probs = F.softmax(action_logits, dim=-1)
        else:
            action_probs = F.softmax(action_logits, dim=-1)

        # === Vertex head with masking ===
        vertex_logits = self.location_head_vertex(x)
        if vertex_mask is not None:
            if isinstance(vertex_mask, np.ndarray):
                vertex_mask = torch.FloatTensor(vertex_mask)
            if len(vertex_mask.shape) == 1:
                vertex_mask = vertex_mask.unsqueeze(0)
            vertex_mask = vertex_mask.to(self.device)

            if vertex_mask.sum() == 0:
                vertex_probs = torch.ones_like(vertex_logits) / vertex_logits.shape[-1]
            else:
                vertex_logits = vertex_logits.masked_fill(vertex_mask == 0, float('-inf'))
                vertex_probs = F.softmax(vertex_logits, dim=-1)
        else:
            vertex_probs = F.softmax(vertex_logits, dim=-1)

        # === Edge head with masking ===
        edge_logits = self.location_head_edge(x)
        if edge_mask is not None:
            if isinstance(edge_mask, np.ndarray):
                edge_mask = torch.FloatTensor(edge_mask)
            if len(edge_mask.shape) == 1:
                edge_mask = edge_mask.unsqueeze(0)
            edge_mask = edge_mask.to(self.device)

            if edge_mask.sum() == 0:
                edge_probs = torch.ones_like(edge_logits) / edge_logits.shape[-1]
            else:
                edge_logits = edge_logits.masked_fill(edge_mask == 0, float('-inf'))
                edge_probs = F.softmax(edge_logits, dim=-1)
        else:
            edge_probs = F.softmax(edge_logits, dim=-1)

        # === Trade heads ===
        # In 1v1 Catan there's no player-to-player trading, only bank/port
        # trades. We still output these for compatibility with the full game.
        trade_give_logits = self.trade_give_head(x)
        trade_give_probs = F.softmax(trade_give_logits, dim=-1)

        trade_get_logits = self.trade_get_head(x)
        trade_get_probs = F.softmax(trade_get_logits, dim=-1)

        # === Value head ===
        state_value = self.value_head(x)

        return action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, state_value

    def get_action_and_value(self, obs, action_mask, vertex_mask=None, edge_mask=None):
        """Sample actions from all heads and return log probs + value for PPO.

        This is the main entry point during training rollouts. We sample from
        the categorical distributions (not argmax) so the agent explores.
        The log probs get stored in the experience buffer and used later
        to compute the PPO surrogate loss.
        """
        action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, value = self.forward(
            obs, action_mask, vertex_mask, edge_mask
        )

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        action_entropy = action_dist.entropy()

        vertex_dist = torch.distributions.Categorical(vertex_probs)
        vertex = vertex_dist.sample()
        vertex_log_prob = vertex_dist.log_prob(vertex)

        edge_dist = torch.distributions.Categorical(edge_probs)
        edge = edge_dist.sample()
        edge_log_prob = edge_dist.log_prob(edge)

        trade_give_dist = torch.distributions.Categorical(trade_give_probs)
        trade_give = trade_give_dist.sample()
        trade_give_log_prob = trade_give_dist.log_prob(trade_give)

        trade_get_dist = torch.distributions.Categorical(trade_get_probs)
        trade_get = trade_get_dist.sample()
        trade_get_log_prob = trade_get_dist.log_prob(trade_get)

        return (action, vertex, edge, trade_give, trade_get,
                action_log_prob, vertex_log_prob, edge_log_prob,
                trade_give_log_prob, trade_get_log_prob, value, action_entropy)

    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'device': str(self.device),
            'hidden_dim': self.hidden_dim,
        }, path)

    def load(self, path, device=None):
        checkpoint = torch.load(path, map_location=device if device else self.device)
        self.load_state_dict(checkpoint['model_state_dict'])


if __name__ == "__main__":
    print("Testing upgraded CatanPolicy with TileAttentionEncoder...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = CatanPolicy(device=device)

    batch_size = 8
    obs = torch.randn(batch_size, TOTAL_OBS_DIM).to(net.device)
    action_mask = torch.ones(batch_size, 14).to(net.device)
    vertex_mask = torch.ones(batch_size, 54).to(net.device)
    edge_mask = torch.ones(batch_size, 72).to(net.device)

    out = net.forward(obs, action_mask, vertex_mask, edge_mask)
    action_probs, vertex_probs, edge_probs, tg, tk, value = out

    assert action_probs.shape == (batch_size, 14), f"Bad action shape: {action_probs.shape}"
    assert vertex_probs.shape == (batch_size, 54), f"Bad vertex shape: {vertex_probs.shape}"
    assert edge_probs.shape == (batch_size, 72), f"Bad edge shape: {edge_probs.shape}"
    assert value.shape == (batch_size, 1), f"Bad value shape: {value.shape}"

    # Check probs sum to 1
    assert torch.allclose(action_probs.sum(dim=-1), torch.ones(batch_size).to(net.device), atol=1e-5)

    # Check gradient flow
    loss = value.mean()
    loss.backward()
    print("Gradient flows through tile attention encoder")

    # Count parameters
    total = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total:,}")
    print("All assertions passed!")
