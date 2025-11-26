import random
import numpy as np
import torch
from network_gpu import CatanPolicy

class CatanAgent:
    def __init__(self, device=None):
        self.policy = CatanPolicy(device=device)
        self.policy.eval()

    def choose_action(self, obs, action_mask):
        """Choose action during gameplay (no gradients)"""
        observation = obs['observation']
        
        # Convert to tensors and move to device
        obs_tensor = torch.FloatTensor(observation).to(self.policy.device)
        mask_tensor = torch.FloatTensor(action_mask).to(self.policy.device)
        
        with torch.no_grad():  # Don't compute gradients during play
            action, log_prob, value, entropy = self.policy.get_action_and_value(
                obs_tensor,
                mask_tensor
            )
        
        # Return as Python scalars (move back to CPU)
        return action.item(), log_prob.item(), value.item()

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
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.action_masks = []

    def store(self, state, action, reward, log_prob, value, done, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def get(self, device=None):
        """Get all experiences as tensors, optionally moved to specific device"""
        data = {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'rewards': torch.FloatTensor(self.rewards),
            'log_probs': torch.FloatTensor(self.log_probs),
            'values': torch.FloatTensor(self.values),
            'dones': torch.FloatTensor(self.dones),
            'action_masks': torch.FloatTensor(np.array(self.action_masks))
        }
        
        # Move to device if specified
        if device is not None:
            data = {k: v.to(device) for k, v in data.items()}
        
        return data

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        self.action_masks.clear()

    def __len__(self):
        return len(self.states)
