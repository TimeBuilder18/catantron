import random
import numpy as np
import torch
from network import CatanPolicy

class CatanAgent:
    def __init__(self):
        self.policy = CatanPolicy()
        self.policy.eval()

    def choose_action(self, obs, action_mask):
        observation = obs['observation']
        obs_tensor = torch.FloatTensor(observation)
        mask_tensor = torch.FloatTensor(action_mask)
        with torch.no_grad():  # Don't compute gradients during play
            action, log_prob, value, entropy = self.policy.get_action_and_value(
                obs_tensor,
                mask_tensor
            )
        return action.item(), log_prob.item(), value.item()

    def choose_action_training(self, obs, action_mask):
        observation = torch.FloatTensor(obs['observation'])
        mask_tensor = torch.FloatTensor(action_mask)

        # No torch.no_grad() here - we WANT gradients
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

    def store(self, state, action, reward, log_prob, value, done,action_masks):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_masks)

    def get(self):
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'rewards': torch.FloatTensor(self.rewards),
            'log_probs': torch.FloatTensor(self.log_probs),
            'values': torch.FloatTensor(self.values),
            'dones': torch.FloatTensor(self.dones),
            'action_masks': torch.FloatTensor(np.array(self.action_masks))
        }

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
