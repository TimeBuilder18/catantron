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
            batch_size=256  # ← Increased from 64 for GPU
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
        
        #print(f"📊 PPO Trainer initialized:")
        #print(f"   Device: {self.device}")
        #print(f"   Batch size: {batch_size}")
        #print(f"   Learning rate: {learning_rate}")

    def compute_advantages(self, rewards, values, dones):
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        advantages = []
        returns = []
        
        # Start from the end and work backwards
        advantage = 0.0
        next_value = 0.0  # Value after terminal state is 0
        
        # Go through experience in REVERSE order
        for t in reversed(range(len(rewards))):
            # If episode ended, next value is 0
            if dones[t]:
                next_value = 0.0
                advantage = 0.0
            
            # Calculate TD error: "How much better/worse than expected?"
            td_error = rewards[t] + self.gamma * next_value - values[t]
            
            # Calculate advantage using GAE formula
            advantage = td_error + self.gamma * self.gae_lambda * advantage
            
            # Store advantage for this step
            advantages.insert(0, advantage)
            
            # Calculate return: advantage + value
            returns.insert(0, advantage + values[t])
            
            # Update next_value for next iteration (going backwards)
            next_value = values[t]
        
        # Convert to numpy arrays
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)
        
        # Normalize advantages (helps training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors and move to device
        return torch.FloatTensor(advantages).to(self.device), torch.FloatTensor(returns).to(self.device)

    def update_policy(self, experience_buffer):
        """Update policy using PPO algorithm"""
        # Get experience data and move to device
        experience = experience_buffer.get(device=self.device)
        
        states = experience['states']
        actions = experience['actions']
        old_log_probs = experience['log_probs']
        values = experience['values']
        rewards = experience['rewards']
        dones = experience['dones']
        action_masks = experience['action_masks']

        # Compute advantages (on CPU for numpy operations, then move to GPU)
        advantages, returns = self.compute_advantages(
            rewards.cpu().numpy(),
            values.cpu().numpy(),
            dones.cpu().numpy()
        )
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # Multiple epochs of training
        for epoch in range(self.n_epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(len(states))

            # Train on mini-batches
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                # Get batch data (already on device)
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = action_masks[batch_indices]

                # Evaluate actions with current policy
                action_probs, state_values = self.policy.forward(batch_states, batch_masks)
                dist = torch.distributions.Categorical(action_probs)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(state_values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # Return average metrics
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates
        }
