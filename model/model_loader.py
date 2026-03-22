"""
Neural Network wrapper for MCTS.

Handles loading saved model checkpoints and providing the evaluate() interface
that MCTS needs. The tricky part is backwards compatibility — we've changed the
observation format (427 -> 575 features) and backbone width (256 -> 512) over
the course of this project, so this wrapper handles detecting old checkpoints
and transferring whatever weights still match. That way we don't lose weeks of
training every time we change the architecture.
"""

import sys
import torch
import numpy as np

# Compatibility shim: our old checkpoints were saved when the module was called
# 'network_gpu' (before we reorganized into model/). torch.load tries to
# reimport the original module path, so we trick it by aliasing the new
# module name into sys.modules under the old name. Without this, loading
# any pre-refactor checkpoint crashes with ModuleNotFoundError.
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
            # Figure out what hidden_dim the saved model used. Newer checkpoints
            # store it explicitly; older ones don't, so we peek at the fc1 weight
            # shape to infer it. If all else fails, assume 256 (original default).
            saved_dim = ckpt.get('hidden_dim', None)
            if saved_dim is None:
                sd = ckpt.get('model_state_dict', {})
                if 'fc1.weight' in sd:
                    saved_dim = sd['fc1.weight'].shape[0]
                else:
                    saved_dim = 256

            self.policy = CatanPolicy(device=device, hidden_dim=hidden_dim)

            # Check if checkpoint has matching layer shapes (obs format + hidden_dim).
            # Two things can cause a mismatch: (1) we changed hidden_dim (256->512),
            # or (2) we changed the observation format (427->575 features) which
            # changes the input layer shapes. Either way we need partial weight transfer.
            needs_transfer = (saved_dim != hidden_dim)
            if not needs_transfer:
                # Same hidden_dim — but still check layer-by-layer in case the
                # obs format changed (different input dims for encoders)
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
        """Transfer encoder weights from a smaller/older network into the new one.

        Only transfers weights where shapes match exactly — basically the encoder
        layers (tile_encoder.*, player_embed.*, player_ln.*). These encode board
        understanding which took the longest to train (~2 days), so preserving
        them is a huge time saver. The backbone and output heads get random init
        and retrain in a few hours.

        We do this instead of strict load_state_dict because that would just
        crash on any shape mismatch.
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
            # Squash value to [-1, 1] range — MCTS expects this for
            # backpropagation through the search tree
            value = np.tanh(state_value.cpu().numpy().flatten()[0])

            return policy, value


