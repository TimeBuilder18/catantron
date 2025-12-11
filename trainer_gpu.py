import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network_gpu import CatanPolicy

class PPOTrainer:
    def __init__(
            self,
            policy,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            n_epochs=10,
            batch_size=256
    ):
        self.policy = policy
        self.device = policy.device
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        returns = []
        
        advantage = 0.0
        next_value = 0.0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                advantage = 0.0
            
            td_error = rewards[t] + self.gamma * next_value - values[t]
            advantage = td_error + self.gamma * self.gae_lambda * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
            next_value = values[t]
        
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)

    def update_policy(self, experience_buffer):
        experience = experience_buffer.get()

        states = experience['states'].to(self.policy.device)
        actions = experience['actions'].to(self.policy.device)
        vertices = experience['vertices'].to(self.policy.device)
        edges = experience['edges'].to(self.policy.device)
        trade_gives = experience['trade_gives'].to(self.policy.device)
        trade_gets = experience['trade_gets'].to(self.policy.device)
        old_action_log_probs = experience['action_log_probs'].to(self.policy.device)
        old_vertex_log_probs = experience['vertex_log_probs'].to(self.policy.device)
        old_edge_log_probs = experience['edge_log_probs'].to(self.policy.device)
        old_trade_give_log_probs = experience['trade_give_log_probs'].to(self.policy.device)
        old_trade_get_log_probs = experience['trade_get_log_probs'].to(self.policy.device)
        values = experience['values'].to(self.policy.device)
        rewards = experience['rewards'].to(self.policy.device)
        dones = experience['dones'].to(self.policy.device)
        action_masks = experience['action_masks'].to(self.policy.device)
        vertex_masks = experience['vertex_masks'].to(self.policy.device)
        edge_masks = experience['edge_masks'].to(self.policy.device)

        advantages, returns = self.compute_advantages(
            rewards.cpu().numpy(),
            values.cpu().numpy(),
            dones.cpu().numpy()
        )

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for epoch in range(self.n_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_vertices = vertices[batch_indices]
                batch_edges = edges[batch_indices]
                batch_trade_gives = trade_gives[batch_indices]
                batch_trade_gets = trade_gets[batch_indices]
                batch_old_action_log_probs = old_action_log_probs[batch_indices]
                batch_old_vertex_log_probs = old_vertex_log_probs[batch_indices]
                batch_old_edge_log_probs = old_edge_log_probs[batch_indices]
                batch_old_trade_give_log_probs = old_trade_give_log_probs[batch_indices]
                batch_old_trade_get_log_probs = old_trade_get_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_action_masks = action_masks[batch_indices]
                batch_vertex_masks = vertex_masks[batch_indices]
                batch_edge_masks = edge_masks[batch_indices]

                action_probs, vertex_probs, edge_probs, trade_give_probs, trade_get_probs, state_values = self.policy.forward(
                    batch_states,
                    batch_action_masks,
                    batch_vertex_masks,
                    batch_edge_masks
                )

                action_dist = torch.distributions.Categorical(action_probs)
                new_action_log_probs = action_dist.log_prob(batch_actions)
                action_entropy = action_dist.entropy().mean()

                vertex_dist = torch.distributions.Categorical(vertex_probs)
                new_vertex_log_probs = vertex_dist.log_prob(batch_vertices)

                edge_dist = torch.distributions.Categorical(edge_probs)
                new_edge_log_probs = edge_dist.log_prob(batch_edges)

                trade_give_dist = torch.distributions.Categorical(trade_give_probs)
                new_trade_give_log_probs = trade_give_dist.log_prob(batch_trade_gives)

                trade_get_dist = torch.distributions.Categorical(trade_get_probs)
                new_trade_get_log_probs = trade_get_dist.log_prob(batch_trade_gets)

                needs_vertex = (batch_actions == 1) | (batch_actions == 3) | (batch_actions == 4)
                needs_edge = (batch_actions == 2) | (batch_actions == 5)
                needs_trade = batch_actions == 9

                new_log_probs = new_action_log_probs.clone()
                new_log_probs += new_vertex_log_probs * needs_vertex.float()
                new_log_probs += new_edge_log_probs * needs_edge.float()
                new_log_probs += new_trade_give_log_probs * needs_trade.float()
                new_log_probs += new_trade_get_log_probs * needs_trade.float()

                old_log_probs = batch_old_action_log_probs.clone()
                old_log_probs += batch_old_vertex_log_probs * needs_vertex.float()
                old_log_probs += batch_old_edge_log_probs * needs_edge.float()
                old_log_probs += batch_old_trade_give_log_probs * needs_trade.float()
                old_log_probs += batch_old_trade_get_log_probs * needs_trade.float()

                entropy = action_entropy

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss to prevent extreme values
                values_pred = state_values.squeeze()
                batch_returns_clipped = torch.clamp(batch_returns, -100, 100)  # Prevent extreme values
                value_loss = nn.MSELoss()(values_pred, batch_returns_clipped)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
