import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CatanPolicy(nn.Module):
    def __init__(self):
        super(CatanPolicy, self).__init__()
        self.fc1 = nn.Linear(121, 256)  # Input is 120-dim
        self.fc2 = nn.Linear(256, 256)  # Add more layers
        self.fc3 = nn.Linear(256, 256)
        self.policy_head = nn.Linear(256, 9)
        self.value_head = nn.Linear(256, 1)

    def forward(self, obs, action_mask=None):
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        action_logits = self.policy_head(x)

        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.FloatTensor(action_mask)
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            # Move to same device as logits
            action_mask = action_mask.to(action_logits.device)

            # Apply masking: set invalid actions to -inf BEFORE softmax
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

            # Check for all -inf (no valid actions)
            if torch.all(torch.isinf(action_logits)):
                print(f"WARNING: All actions masked! mask={action_mask}")
                # Fallback: allow all actions
                action_logits = self.policy_head(x)

        action_probs = F.softmax(action_logits, dim=-1)

        # Check for NaN
        if torch.isnan(action_probs).any():
            print(f"ERROR: NaN in action_probs! logits={action_logits}, mask={action_mask}")
            # Fallback to uniform distribution over valid actions
            if action_mask is not None:
                action_probs = action_mask.float() / action_mask.sum()
            else:
                action_probs = torch.ones_like(action_logits) / action_logits.shape[-1]

        state_value = self.value_head(x)

        return action_probs, state_value

    def get_action_and_value(self, obs, action_mask):
        action_probs, value = self.forward(obs, action_mask)

        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, value, entropy
