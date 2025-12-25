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

    def __init__(self, model_path=None, device=None):
        """
        Args:
            model_path: Path to saved model (optional)
            device: 'cuda', 'cpu', or None for auto
        """
        self.policy = CatanPolicy(device=device)

        if model_path:
            self.policy.load(model_path)
            print(f"✅ Loaded model from {model_path}")

        self.policy.eval()  # Set to evaluation mode

    def evaluate(self, obs):
        """
        Evaluate a game state

        Args:
            obs: Observation dict from GameState.get_observation()

        Returns:
            policy: numpy array of action probabilities (11 actions)
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
            result = self.policy.forward(
                observation,
                action_mask,
                vertex_mask,
                edge_mask
            )

            # Unpack results - your network returns 6 values
            action_probs = result[0]
            state_value = result[-1]  # Last one is value

            # Convert to numpy
            policy = action_probs.cpu().numpy().flatten()
            value = state_value.cpu().numpy().flatten()[0]

            # Normalize value to [-1, 1] range (tanh-like)
            value = np.tanh(value)

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