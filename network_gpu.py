import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CatanPolicy(nn.Module):
    def __init__(self, device=None):
        super(CatanPolicy, self).__init__()
        
        # Auto-detect GPU or use specified device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        #print(f"🎮 CatanPolicy using device: {self.device}")
        if self.device.type == 'cuda':
            pass  # GPU info suppressed during training
            #print(f"   GPU: {torch.cuda.get_device_name(0)}")
            #print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Network architecture (121-dim input for updated observation space)
        self.fc1 = nn.Linear(121, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.policy_head = nn.Linear(256, 9)
        self.value_head = nn.Linear(256, 1)
        
        # Move entire model to device
        self.to(self.device)

    def forward(self, obs, action_mask=None):
        # Ensure inputs are on correct device
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)
        
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # Forward pass
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        action_logits = self.policy_head(x)
        
        # Handle action masking
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.FloatTensor(action_mask)
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            # Move mask to same device as logits
            action_mask = action_mask.to(self.device)
            # Apply masking
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
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'device': str(self.device)
        }, path)
        #print(f"💾 Model saved to {path}")
    
    def load(self, path, device=None):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=device if device else self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        #print(f"📂 Model loaded from {path}")
        #print(f"   Original device: {checkpoint.get('device', 'unknown')}")
        #print(f"   Current device: {self.device}")
