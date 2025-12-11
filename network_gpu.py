import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CatanPolicy(nn.Module):
    def __init__(self, device=None):
        super(CatanPolicy, self).__init__()
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f" Using device: {self.device}")

        self.fc1 = nn.Linear(121, 768)
        self.fc2 = nn.Linear(768, 768)
        self.fc3 = nn.Linear(768, 512)
        self.fc4 = nn.Linear(512, 256)

        # Add layer normalization for training stability
        self.ln1 = nn.LayerNorm(768)
        self.ln2 = nn.LayerNorm(768)
        self.ln3 = nn.LayerNorm(512)
        self.ln4 = nn.LayerNorm(256)

        self.policy_head = nn.Linear(256, 11)  # 10 -> 11 actions
        self.location_head_vertex = nn.Linear(256, 54)
        self.location_head_edge = nn.Linear(256, 72)
        self.trade_give_head = nn.Linear(256, 5) # 5 resources to give
        self.trade_get_head = nn.Linear(256, 5)  # 5 resources to get
        self.value_head = nn.Linear(256, 1)

        self.to(self.device)

    def forward(self, obs, action_mask=None, vertex_mask=None, edge_mask=None):
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs)
        obs = obs.to(self.device)

        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)

        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        x = F.relu(self.ln4(self.fc4(x)))

        action_logits = self.policy_head(x)
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.FloatTensor(action_mask)
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            action_mask = action_mask.to(self.device)
            action_logits = action_logits.masked_fill(action_mask == 0, float('-inf'))
        action_probs = F.softmax(action_logits, dim=-1)

        vertex_logits = self.location_head_vertex(x)
        if vertex_mask is not None:
            if isinstance(vertex_mask, np.ndarray):
                vertex_mask = torch.FloatTensor(vertex_mask)
            if len(vertex_mask.shape) == 1:
                vertex_mask = vertex_mask.unsqueeze(0)
            vertex_mask = vertex_mask.to(self.device)
            vertex_logits = vertex_logits.masked_fill(vertex_mask == 0, float('-inf'))
        vertex_probs = F.softmax(vertex_logits, dim=-1)

        edge_logits = self.location_head_edge(x)
        if edge_mask is not None:
            if isinstance(edge_mask, np.ndarray):
                edge_mask = torch.FloatTensor(edge_mask)
            if len(edge_mask.shape) == 1:
                edge_mask = edge_mask.unsqueeze(0)
            edge_mask = edge_mask.to(self.device)
            edge_logits = edge_logits.masked_fill(edge_mask == 0, float('-inf'))
        edge_probs = F.softmax(edge_logits, dim=-1)

        trade_give_logits = self.trade_give_head(x)
        trade_give_probs = F.softmax(trade_give_logits, dim=-1)

        trade_get_logits = self.trade_get_head(x)
        trade_get_probs = F.softmax(trade_get_logits, dim=-1)

        state_value = self.value_head(x)

        return action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, state_value

    def get_action_and_value(self, obs, action_mask, vertex_mask=None, edge_mask=None):
        action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, value = self.forward(
            obs, action_mask, vertex_mask, edge_mask
        )

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        action_entropy = action_dist.entropy()

        vertex_dist = torch.distributions.Categorical(vertex_probs)
        vertex = vertex_dist.sample()
        vertex_log_prob = vertex_dist.log_prob(vertex)

        edge_dist = torch.distributions.Categorical(edge_probs)
        edge = edge_dist.sample()
        edge_log_prob = edge_dist.log_prob(edge)

        trade_give_dist = torch.distributions.Categorical(trade_give_probs)
        trade_give = trade_give_dist.sample()
        trade_give_log_prob = trade_give_dist.log_prob(trade_give)

        trade_get_dist = torch.distributions.Categorical(trade_get_probs)
        trade_get = trade_get_dist.sample()
        trade_get_log_prob = trade_get_dist.log_prob(trade_get)

        return (action, vertex, edge, trade_give, trade_get,
                action_log_prob, vertex_log_prob, edge_log_prob,
                trade_give_log_prob, trade_get_log_prob, value, action_entropy)
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'device': str(self.device)
        }, path)
    
    def load(self, path, device=None):
        checkpoint = torch.load(path, map_location=device if device else self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
