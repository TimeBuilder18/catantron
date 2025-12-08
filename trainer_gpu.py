import torch

class PPOTrainer:
    def __init__(
        self,
        model,
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.90,
        clip_epsilon=0.27,
        entropy_coef=0.08,
        value_coef=0.5,
        max_grad_norm=0.7,
        device=None
    ):
        self.device = torch.device(device) if device is not None else model.device
        self.model = model.to(self.device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def update_policy(self, buffer):
        """Run one PPO update with data from buffer"""
        data = buffer.get()
        states = data['states'].to(self.device)
        actions = data['actions'].to(self.device)
        values = data['values'].to(self.device)
        old_log_probs = data['action_log_probs'].to(self.device)
        rewards = data['rewards'].to(self.device)
        dones = data['dones'].to(self.device)

        # Compute advantages
        returns = []
        advs = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1-dones[i]) - values[i] if i+1 < len(values) else rewards[i] - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[i])

        advs = torch.FloatTensor(advs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Compute new log probs and value predictions
        action_log_probs, values_pred, entropy = self.model.evaluate_actions(states, actions)
        ratios = torch.exp(action_log_probs - old_log_probs)

        surr1 = ratios * advs
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advs
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (returns - values_pred).pow(2).mean()
        entropy_loss = -entropy.mean()

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'entropy_loss': entropy_loss.item()}
