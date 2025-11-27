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

    def choose_action(self, obs, action_mask, vertex_mask=None, edge_mask=None):
        """
        Choose action AND location using the hierarchical policy

        Args:
            obs: Observation dict with 'observation' key
            action_mask: [9] mask for valid actions
            vertex_mask: [54] mask for valid vertices (optional)
            edge_mask: [72] mask for valid edges (optional)

        Returns:
            action: Chosen action index
            vertex: Chosen vertex index
            edge: Chosen edge index
            action_log_prob: Log prob of action
            vertex_log_prob: Log prob of vertex
            edge_log_prob: Log prob of edge
            value: State value
        """
        observation = torch.FloatTensor(obs['observation'])
        mask_tensor = torch.FloatTensor(action_mask)

        # Create default masks if not provided
        if vertex_mask is None:
            vertex_mask = np.ones(54, dtype=np.float32)
        if edge_mask is None:
            edge_mask = np.ones(72, dtype=np.float32)

        vertex_mask_tensor = torch.FloatTensor(vertex_mask)
        edge_mask_tensor = torch.FloatTensor(edge_mask)

        with torch.no_grad():  # Don't compute gradients during play
            action, vertex, edge, action_log_prob, vertex_log_prob, edge_log_prob, value, entropy = \
                self.policy.get_action_and_value(
                    observation,
                    mask_tensor,
                    vertex_mask_tensor,
                    edge_mask_tensor
                )

        # Return 7 things (was 3 before)
        return (action.item(), vertex.item(), edge.item(),
                action_log_prob.item(), vertex_log_prob.item(), edge_log_prob.item(),
                value.item())

    def choose_action_training(self, obs, action_mask):
        """Choose action during training (WITH gradients for PPO update)"""
        observation = torch.FloatTensor(obs['observation']).to(self.policy.device)
        mask_tensor = torch.FloatTensor(action_mask).to(self.policy.device)

        # No torch.no_grad() here - we WANT gradients for training
        action, log_prob, value, entropy = self.policy.get_action_and_value(
            observation,
            mask_tensor
        )

        return action, log_prob, value, entropy


class ExperienceBuffer:
    """Stores experience from games for PPO learning"""

    def __init__(self):
        # Store experiences with hierarchical actions
        self.states = []
        self.actions = []
        self.vertices = []  # NEW: Store vertex choices
        self.edges = []  # NEW: Store edge choices
        self.rewards = []
        self.log_probs = []
        self.action_log_probs = []  # NEW: Separate action log prob
        self.vertex_log_probs = []  # NEW: Vertex log prob
        self.edge_log_probs = []  # NEW: Edge log prob
        self.values = []
        self.dones = []
        self.action_masks = []
        self.vertex_masks = []  # NEW: Store vertex masks
        self.edge_masks = []  # NEW: Store edge masks

    def store(self, state, action, vertex, edge, reward,
              action_log_prob, vertex_log_prob, edge_log_prob, value, done,
              action_mask, vertex_mask, edge_mask):
        """
        Store one step of experience with hierarchical action

        Args:
            state: Observation array
            action: Chosen action index
            vertex: Chosen vertex index
            edge: Chosen edge index
            reward: Reward received
            action_log_prob: Log prob of action
            vertex_log_prob: Log prob of vertex
            edge_log_prob: Log prob of edge
            value: State value estimate
            done: Episode done flag
            action_mask: Valid actions mask
            vertex_mask: Valid vertices mask
            edge_mask: Valid edges mask
        """
        self.states.append(state)
        self.actions.append(action)
        self.vertices.append(vertex)
        self.edges.append(edge)
        self.rewards.append(reward)
        self.action_log_probs.append(action_log_prob)
        self.vertex_log_probs.append(vertex_log_prob)
        self.edge_log_probs.append(edge_log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.vertex_masks.append(vertex_mask)
        self.edge_masks.append(edge_mask)

    def get(self):
        """
        Get all stored experiences as tensors

        Returns:
            Dictionary of tensors for training
        """
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'vertices': torch.LongTensor(self.vertices),  # NEW
            'edges': torch.LongTensor(self.edges),  # NEW
            'rewards': torch.FloatTensor(self.rewards),
            'action_log_probs': torch.FloatTensor(self.action_log_probs),  # NEW: Separated
            'vertex_log_probs': torch.FloatTensor(self.vertex_log_probs),  # NEW
            'edge_log_probs': torch.FloatTensor(self.edge_log_probs),  # NEW
            'values': torch.FloatTensor(self.values),
            'dones': torch.FloatTensor(self.dones),
            'action_masks': torch.FloatTensor(np.array(self.action_masks)),
            'vertex_masks': torch.FloatTensor(np.array(self.vertex_masks)),  # NEW
            'edge_masks': torch.FloatTensor(np.array(self.edge_masks))  # NEW
        }

    def clear(self):
        """Clear all stored experiences"""
        self.states.clear()
        self.actions.clear()
        self.vertices.clear()  # NEW
        self.edges.clear()  # NEW
        self.rewards.clear()
        self.action_log_probs.clear()  # NEW
        self.vertex_log_probs.clear()  # NEW
        self.edge_log_probs.clear()  # NEW
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()
        self.vertex_masks.clear()  # NEW
        self.edge_masks.clear()

    def __len__(self):
        return len(self.states)
