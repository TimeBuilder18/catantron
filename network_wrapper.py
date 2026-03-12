"""
Neural Network wrapper for MCTS

Connects your existing CatanPolicy to the MCTS algorithm.
"""

import torch
import numpy as np
from network_gpu import CatanPolicy


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
            saved_dim = ckpt.get('hidden_dim', 256)  # Old checkpoints default to 256

            if saved_dim != hidden_dim:
                # Different architecture — transfer weights from smaller to larger network
                print(f"  Transferring weights: checkpoint hidden_dim={saved_dim} → new hidden_dim={hidden_dim}")
                self.policy = CatanPolicy(device=device, hidden_dim=hidden_dim)
                self._transfer_weights(ckpt['model_state_dict'], saved_dim, hidden_dim)
                print(f"✅ Transferred weights from {model_path} (expanded {saved_dim}→{hidden_dim})")
            else:
                # Same architecture — direct load
                self.policy = CatanPolicy(device=device, hidden_dim=hidden_dim)
                self.policy.load(model_path)
                print(f"✅ Loaded model from {model_path}")
        else:
            self.policy = CatanPolicy(device=device, hidden_dim=hidden_dim)

        self.policy.eval()  # Set to evaluation mode

    def _transfer_weights(self, old_state_dict, old_dim, new_dim):
        """Transfer weights from a smaller network into a larger one.

        For each parameter, copies the old weights into the top-left corner
        of the new (larger) parameter tensor. The remaining capacity is left
        at its random initialization. This preserves all learned knowledge
        while adding new capacity for the network to grow into.
        """
        new_state = self.policy.state_dict()
        transferred = 0
        for name, old_param in old_state_dict.items():
            if name not in new_state:
                continue
            new_param = new_state[name]
            if old_param.shape == new_param.shape:
                # Same shape — direct copy
                new_state[name] = old_param
                transferred += 1
            else:
                # Different shape — copy old weights into top-left corner
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_param.shape, new_param.shape))
                new_state[name][slices] = old_param[slices]
                transferred += 1
        self.policy.load_state_dict(new_state)
        print(f"  Transferred {transferred}/{len(old_state_dict)} parameter tensors")

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


# Quick test
if __name__ == "__main__":
    from game_state import GameState

    print("Testing NetworkWrapper...")

    # Create wrapper (no saved model - uses random weights)
    network = NetworkWrapper(model_path=None)
    print("✅ Created network wrapper")

    # Create game state
    state = GameState()
    obs = state.get_observation()
    print("✅ Got observation")

    # Evaluate
    policy, value = network.evaluate(obs)

    print(f"✅ Network evaluation:")
    print(f"   Policy shape: {policy.shape}")
    print(f"   Policy sum: {policy.sum():.3f} (should be ~1.0)")
    print(f"   Top 3 action probs: {sorted(policy, reverse=True)[:3]}")
    print(f"   Value: {value:.3f}")

    print("\n✅ NetworkWrapper test passed!")