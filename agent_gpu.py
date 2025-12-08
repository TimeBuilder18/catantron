import random
import numpy as np
import torch
from network_gpu import CatanPolicy

class CatanAgent:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = torch.device(device)
        self.policy = CatanPolicy(device=device)
        self.policy.eval()

    def choose_action(self, obs, action_mask, vertex_mask=None, edge_mask=None, is_training=False):
        """
        Choose action and all hierarchical parameters using the policy.
        Handles both training and evaluation.
        """
        observation = torch.tensor(obs['observation'], dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device)

        if vertex_mask is None:
            vertex_mask = np.ones(54, dtype=np.float32)
        if edge_mask is None:
            edge_mask = np.ones(72, dtype=np.float32)

        vertex_mask_tensor = torch.tensor(vertex_mask, dtype=torch.float32, device=self.device)
        edge_mask_tensor = torch.tensor(edge_mask, dtype=torch.float32, device=self.device)

        if is_training:
            # Keep gradients for training
            return self.policy.get_action_and_value(
                observation, mask_tensor, vertex_mask_tensor, edge_mask_tensor
            )
        else:
            # No gradients for evaluation
            with torch.no_grad():
                actions = self.policy.get_action_and_value(
                    observation, mask_tensor, vertex_mask_tensor, edge_mask_tensor
                )
            # Convert single-element tensors to Python scalars
            return tuple(a.item() if isinstance(a, torch.Tensor) and a.numel() == 1 else a for a in actions)


class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.vertices = []
        self.edges = []
        self.trade_gives = []
        self.trade_gets = []
        self.rewards = []
        self.action_log_probs = []
        self.vertex_log_probs = []
        self.edge_log_probs = []
        self.trade_give_log_probs = []
        self.trade_get_log_probs = []
        self.values = []
        self.dones = []
        self.action_masks = []
        self.vertex_masks = []
        self.edge_masks = []

    def store(self, state, action, vertex, edge, trade_give, trade_get, reward,
              action_log_prob, vertex_log_prob, edge_log_prob,
              trade_give_log_prob, trade_get_log_prob, value, done,
              action_mask, vertex_mask, edge_mask):
        self.states.append(state)
        self.actions.append(action)
        self.vertices.append(vertex)
        self.edges.append(edge)
        self.trade_gives.append(trade_give)
        self.trade_gets.append(trade_get)
        self.rewards.append(reward)
        self.action_log_probs.append(action_log_prob)
        self.vertex_log_probs.append(vertex_log_prob)
        self.edge_log_probs.append(edge_log_prob)
        self.trade_give_log_probs.append(trade_give_log_prob)
        self.trade_get_log_probs.append(trade_get_log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.vertex_masks.append(vertex_mask)
        self.edge_masks.append(edge_mask)

    def get(self, device=None):
        device = device or torch.device('cpu')
        return {
            'states': torch.tensor(np.array(self.states), dtype=torch.float32, device=device),
            'actions': torch.tensor(self.actions, dtype=torch.long, device=device),
            'vertices': torch.tensor(self.vertices, dtype=torch.long, device=device),
            'edges': torch.tensor(self.edges, dtype=torch.long, device=device),
            'trade_gives': torch.tensor(self.trade_gives, dtype=torch.long, device=device),
            'trade_gets': torch.tensor(self.trade_gets, dtype=torch.long, device=device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32, device=device),
            'action_log_probs': torch.tensor(self.action_log_probs, dtype=torch.float32, device=device),
            'vertex_log_probs': torch.tensor(self.vertex_log_probs, dtype=torch.float32, device=device),
            'edge_log_probs': torch.tensor(self.edge_log_probs, dtype=torch.float32, device=device),
            'trade_give_log_probs': torch.tensor(self.trade_give_log_probs, dtype=torch.float32, device=device),
            'trade_get_log_probs': torch.tensor(self.trade_get_log_probs, dtype=torch.float32, device=device),
            'values': torch.tensor(self.values, dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=device),
            'action_masks': torch.tensor(np.array(self.action_masks), dtype=torch.float32, device=device),
            'vertex_masks': torch.tensor(np.array(self.vertex_masks), dtype=torch.float32, device=device),
            'edge_masks': torch.tensor(np.array(self.edge_masks), dtype=torch.float32, device=device)
        }

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)
