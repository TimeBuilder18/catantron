"""
Neural Network wrapper for MCTS

Connects your existing CatanPolicy to the MCTS algorithm.
"""

import sys
import torch
import numpy as np

# Compatibility: old checkpoints reference 'network_gpu' which was renamed to 'model.network'
from model import network as _network_module
sys.modules['network_gpu'] = _network_module

from model.network import CatanPolicy


class NetworkWrapper:
    """
    Wraps CatanPolicy to provide the interface MCTS needs:
    - evaluate(observation) -> (policy, value)
    """

    def __init__(self, model_path=None, device=None, hidden_dim=256):
        """
        Args:
            model_path: Path to saved model (optional)
            device: 'cuda', 'cpu', or None for auto
            hidden_dim: Backbone hidden dimension (256=original, 512=larger)
        """
        if model_path:
            ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
            # Detect saved hidden_dim: explicit key > infer from fc1 > fallback 256
            saved_dim = ckpt.get('hidden_dim', None)
            if saved_dim is None:
                sd = ckpt.get('model_state_dict', {})
                if 'fc1.weight' in sd:
                    saved_dim = sd['fc1.weight'].shape[0]
                else:
                    saved_dim = 256

            self.policy = CatanPolicy(device=device, hidden_dim=hidden_dim)

            # Check if checkpoint has matching layer shapes (obs format + hidden_dim)
            needs_transfer = (saved_dim != hidden_dim)
            if not needs_transfer:
                # Same hidden_dim, but obs format may differ (427 vs 575)
                old_sd = ckpt['model_state_dict']
                new_sd = self.policy.state_dict()
                for name in old_sd:
                    if name in new_sd and old_sd[name].shape != new_sd[name].shape:
                        needs_transfer = True
                        break

            if needs_transfer:
                reason = f"hidden_dim={saved_dim}→{hidden_dim}" if saved_dim != hidden_dim else "obs format mismatch"
                print(f"  Transferring weights ({reason})")
                self._transfer_weights(ckpt['model_state_dict'], saved_dim, hidden_dim)
                print(f"✅ Transferred compatible weights from {model_path}")
            else:
                self.policy.load(model_path)
                print(f"✅ Loaded model from {model_path}")
        else:
            self.policy = CatanPolicy(device=device, hidden_dim=hidden_dim)

        self.policy.eval()  # Set to evaluation mode

    def _transfer_weights(self, old_state_dict, old_dim, new_dim):
        """Transfer encoder weights from a smaller network into a larger one.

        Only transfers encoder weights (tile_encoder.*, player_embed.*, player_ln.*)
        which have identical shapes regardless of hidden_dim. These encode board
        understanding — the hardest part to learn. The backbone (fc layers, output
        heads, projection) starts fresh and retrains quickly.
        """
        new_state = self.policy.state_dict()
        transferred = 0
        skipped = 0
        for name, old_param in old_state_dict.items():
            if name not in new_state:
                continue
            new_param = new_state[name]
            if old_param.shape == new_param.shape:
                # Same shape — safe to copy (encoders, and any unchanged layers)
                new_state[name] = old_param
                transferred += 1
            else:
                # Different shape — skip (backbone/heads changed due to hidden_dim)
                skipped += 1
        self.policy.load_state_dict(new_state)
        print(f"  Transferred {transferred} params, skipped {skipped} (different shape, fresh init)")

    def evaluate(self, obs):
        """
        Evaluate a game state

        Args:
            obs: Observation dict from GameState.get_observation()

        Returns:
            policy: dict with 'action', 'vertex', 'edge' numpy arrays
            value: float in [-1, 1] estimating win probability
        """
        with torch.no_grad():
            # Get observation tensor
            observation = torch.FloatTensor(obs['observation'])
            action_mask = torch.FloatTensor(obs['action_mask'])
            vertex_mask = torch.FloatTensor(obs['vertex_mask'])
            edge_mask = torch.FloatTensor(obs['edge_mask'])

            # Forward pass through network
            # Returns: action_probs, vertex_probs, edge_probs, trade_give, trade_get, value
            action_probs, vertex_probs, edge_probs, _, _, state_value = self.policy.forward(
                observation,
                action_mask,
                vertex_mask,
                edge_mask
            )

            # Convert to numpy
            policy = {
                'action': action_probs.cpu().numpy().flatten(),
                'vertex': vertex_probs.cpu().numpy().flatten(),
                'edge': edge_probs.cpu().numpy().flatten(),
            }
            value = np.tanh(state_value.cpu().numpy().flatten()[0])

            return policy, value


