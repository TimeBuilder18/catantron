import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CatanPolicy(nn.Module):
    def __init__(self, device=None):
        super(CatanPolicy, self).__init__()
        
        # Auto-detect GPU or use specified device
        # Auto-detect best device: CUDA (PC) > MPS (Mac M1/M2) > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')  # Apple Silicon GPU!
        else:
            self.device = torch.device('cpu')

        print(f" Using device: {self.device}")
        
        #print(f"🎮 CatanPolicy using device: {self.device}")
        if self.device.type == 'cuda':
            pass  # GPU info suppressed during training
            #print(f"   GPU: {torch.cuda.get_device_name(0)}")
            #print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Network architecture (121-dim input for updated observation space)
        self.fc1 = nn.Linear(121, 768)
        self.fc2 = nn.Linear(768, 768)
        self.fc3 = nn.Linear(768, 512)
        self.fc4 = nn.Linear(512, 256)
        self.policy_head = nn.Linear(256, 9)  # WHEN to act (9 actions)
        self.location_head_vertex = nn.Linear(256, 54)  # WHERE for settlements/cities (54 vertices)
        self.location_head_edge = nn.Linear(256, 72)  # WHERE for roads (72 edges)
        self.value_head = nn.Linear(256, 1)

        # Move entire model to device
        self.to(self.device)

    def forward(self, obs, action_mask=None, vertex_mask=None, edge_mask=None):
        """
        Forward pass - now outputs actions AND locations

        Args:
            obs: State observation [batch, 121]
            action_mask: Which actions are legal [batch, 9]
            vertex_mask: Which vertices are legal [batch, 54]
            edge_mask: Which edges are legal [batch, 72]

        Returns:
            action_probs: [batch, 9] - probability for each action
            vertex_probs: [batch, 54] - probability for each vertex
            edge_probs: [batch, 72] - probability for each edge
            state_value: [batch, 1] - value of this state
        """
        # Ensure tensor and correct device
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        # Forward through shared layers
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))  # [batch, 256] shared features

        # === ACTION HEAD ===
        action_logits = self.policy_head(x)  # [batch, 9]

        # Apply action mask (mark illegal actions as -inf)
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.FloatTensor(action_mask)
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            action_mask = action_mask.to(self.device)
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))

        action_probs = F.softmax(action_logits, dim=-1)  # [batch, 9]

        # === VERTEX HEAD (for settlements/cities) ===
        vertex_logits = self.location_head_vertex(x)  # [batch, 54]

        # Apply vertex mask
        if vertex_mask is not None:
            if isinstance(vertex_mask, np.ndarray):
                vertex_mask = torch.FloatTensor(vertex_mask)
            if len(vertex_mask.shape) == 1:
                vertex_mask = vertex_mask.unsqueeze(0)
            vertex_mask = vertex_mask.to(self.device)
            vertex_logits = vertex_logits.masked_fill(vertex_mask == 0, float('-inf'))

        vertex_probs = F.softmax(vertex_logits, dim=-1)  # [batch, 54]

        # === EDGE HEAD (for roads) ===
        edge_logits = self.location_head_edge(x)  # [batch, 72]

        # Apply edge mask
        if edge_mask is not None:
            if isinstance(edge_mask, np.ndarray):
                edge_mask = torch.FloatTensor(edge_mask)
            if len(edge_mask.shape) == 1:
                edge_mask = edge_mask.unsqueeze(0)
            edge_mask = edge_mask.to(self.device)
            edge_logits = edge_logits.masked_fill(edge_mask == 0, float('-inf'))

        edge_probs = F.softmax(edge_logits, dim=-1)  # [batch, 72]

        # === VALUE HEAD ===
        state_value = self.value_head(x)  # [batch, 1]

        # Return 4 things now (was 2 before)
        return action_probs, vertex_probs, edge_probs, state_value

    def get_action_and_value(self, obs, action_mask, vertex_mask=None, edge_mask=None):
        """
        Sample action + location and return log probs

        Returns:
            action: Chosen action index [0-8]
            vertex: Chosen vertex index [0-53]
            edge: Chosen edge index [0-71]
            action_log_prob: Log probability of action
            vertex_log_prob: Log probability of vertex
            edge_log_prob: Log probability of edge
            value: State value
            entropy: Policy entropy
        """
        # Get probabilities from network
        action_probs, vertex_probs, edge_probs, value = self.forward(
            obs, action_mask, vertex_mask, edge_mask
        )

        # Sample ACTION
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        action_entropy = action_dist.entropy()

        # Sample VERTEX location
        vertex_dist = torch.distributions.Categorical(vertex_probs)
        vertex = vertex_dist.sample()
        vertex_log_prob = vertex_dist.log_prob(vertex)

        # Sample EDGE location
        edge_dist = torch.distributions.Categorical(edge_probs)
        edge = edge_dist.sample()
        edge_log_prob = edge_dist.log_prob(edge)

        # Return 8 things now (was 4 before)
        return action, vertex, edge, action_log_prob, vertex_log_prob, edge_log_prob, value, action_entropy
    
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
