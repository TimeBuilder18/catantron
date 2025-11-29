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
        This is the single, unified method for both training and evaluation.
        """
        observation = torch.FloatTensor(obs['observation'])
        mask_tensor = torch.FloatTensor(action_mask)

        if vertex_mask is None:
            vertex_mask = np.ones(54, dtype=np.float32)
        if edge_mask is None:
            edge_mask = np.ones(72, dtype=np.float32)

        vertex_mask_tensor = torch.FloatTensor(vertex_mask)
        edge_mask_tensor = torch.FloatTensor(edge_mask)

        if is_training:
            # Keep gradients for training
            (action, vertex, edge, trade_give, trade_get,
             action_log_prob, vertex_log_prob, edge_log_prob,
             trade_give_log_prob, trade_get_log_prob, value, entropy) = \
                self.policy.get_action_and_value(
                    observation,
                    mask_tensor,
                    vertex_mask_tensor,
                    edge_mask_tensor
                )
            return (action, vertex, edge, trade_give, trade_get,
                    action_log_prob, vertex_log_prob, edge_log_prob,
                    trade_give_log_prob, trade_get_log_prob, value, entropy)
        else:
            # No gradients needed for evaluation
            with torch.no_grad():
                (action, vertex, edge, trade_give, trade_get,
                 action_log_prob, vertex_log_prob, edge_log_prob,
                 trade_give_log_prob, trade_get_log_prob, value, entropy) = \
                    self.policy.get_action_and_value(
                        observation,
                        mask_tensor,
                        vertex_mask_tensor,
                        edge_mask_tensor
                    )
            return (action.item(), vertex.item(), edge.item(), trade_give.item(), trade_get.item(),
                    action_log_prob.item(), vertex_log_prob.item(), edge_log_prob.item(),
                    trade_give_log_prob.item(), trade_get_log_prob.item(),
                    value.item())


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

    def get(self):
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'vertices': torch.LongTensor(self.vertices),
            'edges': torch.LongTensor(self.edges),
            'trade_gives': torch.LongTensor(self.trade_gives),
            'trade_gets': torch.LongTensor(self.trade_gets),
            'rewards': torch.FloatTensor(self.rewards),
            'action_log_probs': torch.FloatTensor(self.action_log_probs),
            'vertex_log_probs': torch.FloatTensor(self.vertex_log_probs),
            'edge_log_probs': torch.FloatTensor(self.edge_log_probs),
            'trade_give_log_probs': torch.FloatTensor(self.trade_give_log_probs),
            'trade_get_log_probs': torch.FloatTensor(self.trade_get_log_probs),
            'values': torch.FloatTensor(self.values),
            'dones': torch.FloatTensor(self.dones),
            'action_masks': torch.FloatTensor(np.array(self.action_masks)),
            'vertex_masks': torch.FloatTensor(np.array(self.vertex_masks)),
            'edge_masks': torch.FloatTensor(np.array(self.edge_masks))
        }

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.vertices.clear()
        self.edges.clear()
        self.trade_gives.clear()
        self.trade_gets.clear()
        self.rewards.clear()
        self.action_log_probs.clear()
        self.vertex_log_probs.clear()
        self.edge_log_probs.clear()
        self.trade_give_log_probs.clear()
        self.trade_get_log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()
        self.vertex_masks.clear()
        self.edge_masks.clear()

    def __len__(self):
        return len(self.states)
