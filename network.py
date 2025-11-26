import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CatanPolicy(nn.Module):
    def __init__(self, device=None):
        super(CatanPolicy, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fc1 = nn.Linear(121, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.policy_head = nn.Linear(256, 9)
        self.value_head = nn.Linear(256, 1)
        self.to(self.device)

    def forward(self, obs, action_mask=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        action_logits = self.policy_head(x)
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.FloatTensor(action_mask)
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            action_mask = action_mask.to(self.device)
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value

    def get_action_and_value(self, obs, action_mask):
        action_probs, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])