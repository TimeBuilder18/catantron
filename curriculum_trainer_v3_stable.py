"""
Curriculum Training V3 - STABLE VERSION

Fixes entropy collapse with:
1. Smooth entropy penalty (quadratic, not step function)
2. Adaptive learning rate based on entropy health
3. Trust region via KL divergence constraint
4. Gradient clipping with entropy-aware scaling
5. Performance-based curriculum transitions (not just game count)
6. Separate entropy tracking for all heads
7. Buffer prioritization for recent experiences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time
import random

from catan_env_pytorch import CatanEnv
from simplified_reward_wrapper import SimplifiedRewardWrapper
from pbrs_fixed_reward_wrapper import PBRSFixedRewardWrapper
from network_wrapper import NetworkWrapper
from game_system import ResourceType


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def play_random_turn(game, player_id):
    """Random opponent"""
    player = game.players[player_id]

    if game.is_initial_placement_phase():
        if game.waiting_for_road:
            if game.last_settlement_vertex:
                edges = game.game_board.edges
                valid = [e for e in edges
                        if e.structure is None and
                        (e.vertex1 == game.last_settlement_vertex or
                         e.vertex2 == game.last_settlement_vertex)]
                if valid:
                    game.try_place_initial_road(random.choice(valid), player)
        else:
            vertices = game.game_board.vertices
            valid = [v for v in vertices if v.structure is None and
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                game.try_place_initial_settlement(random.choice(valid), player)
        return True

    if game.can_roll_dice():
        game.roll_dice()
        return True

    if game.can_trade_or_build():
        actions = []
        res = player.resources

        if (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
            res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1):
            v = game.get_buildable_vertices_for_settlements()
            if v: actions.append(('sett', v))

        if res[ResourceType.WHEAT] >= 2 and res[ResourceType.ORE] >= 3:
            v = game.get_buildable_vertices_for_cities()
            if v: actions.append(('city', v))

        if res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1:
            e = game.get_buildable_edges()
            if e: actions.append(('road', e))

        if (res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1 and
            res[ResourceType.ORE] >= 1 and not game.dev_deck.is_empty()):
            actions.append(('dev', None))

        if actions and random.random() < 0.85:
            action_type, locs = random.choice(actions)
            if action_type == 'sett':
                player.try_build_settlement(random.choice(locs))
            elif action_type == 'city':
                player.try_build_city(random.choice(locs))
            elif action_type == 'road':
                player.try_build_road(random.choice(locs))
            elif action_type == 'dev':
                game.try_buy_development_card(player)
            return True

    if game.can_end_turn():
        game.end_turn()
        return True
    return True


def play_opponent_turn(game, player_id, random_prob, ai_difficulty='medium'):
    """Play opponent with mix of random/rule-based AI

    Args:
        game: The game instance
        player_id: Which player to play
        random_prob: Probability of playing randomly (0.0-1.0)
        ai_difficulty: 'weak', 'medium', or 'strong' for rule-based AI
    """
    if random.random() < random_prob:
        return play_random_turn(game, player_id)
    else:
        from rule_based_ai import play_rule_based_turn
        class MinimalEnv:
            def __init__(self, g):
                self.game_env = type('obj', (object,), {'game': g})()
        return play_rule_based_turn(MinimalEnv(game), player_id, difficulty=ai_difficulty)


class PrioritizedReplayBuffer:
    """Replay buffer with recency bias and priority sampling"""

    def __init__(self, max_size=300000, recency_bias=0.7):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.recency_bias = recency_bias  # Probability of sampling from recent half

    def add(self, obs, action_probs, reward, old_log_prob=None):
        self.buffer.append({
            'obs': obs,
            'probs': action_probs,
            'reward': reward,
            'old_log_prob': old_log_prob
        })

    def sample(self, batch_size):
        n = len(self.buffer)
        if n < batch_size:
            indices = np.arange(n)
        else:
            # Sample with recency bias
            recent_size = min(n // 2, batch_size * 2)
            recent_count = int(batch_size * self.recency_bias)
            old_count = batch_size - recent_count

            recent_indices = np.random.choice(
                range(n - recent_size, n),
                min(recent_count, recent_size),
                replace=False
            )
            old_indices = np.random.choice(
                range(max(0, n - recent_size)),
                min(old_count, max(1, n - recent_size)),
                replace=False
            )
            indices = np.concatenate([recent_indices, old_indices])

        return {
            'observations': np.array([self.buffer[i]['obs'] for i in indices]),
            'action_probs': np.array([self.buffer[i]['probs'] for i in indices]),
            'rewards': np.array([self.buffer[i]['reward'] for i in indices]),
            'old_log_probs': np.array([self.buffer[i].get('old_log_prob', 0.0) for i in indices])
        }

    def clear_old(self, keep_fraction=0.3):
        """Clear old experiences, keeping recent fraction"""
        keep_count = int(len(self.buffer) * keep_fraction)
        while len(self.buffer) > keep_count:
            self.buffer.popleft()

    def __len__(self):
        return len(self.buffer)


class CurriculumTrainerV3:
    """Stable curriculum trainer with entropy collapse prevention"""

    def __init__(self, model_path=None, learning_rate=5e-4, batch_size=None, reward_mode='vp_only',
                 lr_decay=1.0, value_weight=0.5):
        self.device = get_device()
        self.reward_mode = reward_mode
        self.lr_decay = lr_decay  # Learning rate decay per 1000 games
        self.value_weight = value_weight  # Weight for value loss (default increased from 0.1)

        if batch_size is None:
            batch_size = 1024 if self.device.type == 'cuda' else 256
        self.batch_size = batch_size

        self.network_wrapper = NetworkWrapper(model_path=model_path, device=str(self.device))
        self.network = self.network_wrapper.policy

        self.base_lr = learning_rate
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        # Smaller buffer with recency bias
        self.replay_buffer = PrioritizedReplayBuffer(max_size=200000, recency_bias=0.7)

        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        self.games_played = 0

        # ENTROPY STABILITY PARAMETERS
        self.target_entropy = 1.8  # Target entropy (healthy range: 1.5-2.2)
        self.entropy_coef = 0.15   # Base entropy coefficient
        self.min_entropy_coef = 0.05
        self.max_entropy_coef = 0.5

        # Adaptive parameters
        self.entropy_history = deque(maxlen=100)
        self.policy_loss_history = deque(maxlen=50)
        self.current_entropy_coef = self.entropy_coef

        # Trust region
        self.max_kl = 0.02  # Maximum KL divergence per update
        self.adaptive_kl_coef = 1.0

        # Gradient clipping
        self.base_grad_norm = 0.5
        self.current_grad_norm = 0.5

        # Curriculum tracking
        self.phase_wins = deque(maxlen=100)
        self.phase_vps = deque(maxlen=100)

        print(f"Batch size: {self.batch_size}")
        print(f"Base LR: {self.base_lr}")
        print(f"LR decay: {self.lr_decay} per 1000 games")
        print(f"Value weight: {self.value_weight}")
        print(f"Reward mode: {self.reward_mode}")
        print(f"Target entropy: {self.target_entropy}")
        print(f"Base entropy coef: {self.entropy_coef}")
        print(f"Max KL divergence: {self.max_kl}")

    def _compute_smooth_entropy_penalty(self, entropy):
        """Smooth quadratic penalty instead of hard floor"""
        # Quadratic penalty below target, small bonus above
        # penalty = coef * max(0, target - entropy)^2
        # This creates smooth gradients near the threshold

        diff = self.target_entropy - entropy

        if diff > 0:
            # Below target: quadratic penalty (increases smoothly)
            penalty = diff ** 2
        else:
            # Above target: small bonus (encourage some exploration)
            penalty = -0.1 * min(abs(diff), 0.5)

        return penalty

    def _adapt_hyperparameters(self, entropy, policy_loss):
        """Dynamically adjust hyperparameters based on training health"""
        self.entropy_history.append(entropy)
        self.policy_loss_history.append(abs(policy_loss))

        if len(self.entropy_history) < 10:
            return

        avg_entropy = np.mean(list(self.entropy_history)[-20:])
        entropy_trend = np.mean(list(self.entropy_history)[-10:]) - np.mean(list(self.entropy_history)[-20:-10])
        avg_policy_loss = np.mean(list(self.policy_loss_history)[-10:])

        # ENTROPY COEFFICIENT ADAPTATION
        if avg_entropy < 1.0:
            # Critical: entropy very low, boost coefficient significantly
            self.current_entropy_coef = min(self.max_entropy_coef, self.current_entropy_coef * 1.3)
            self.current_grad_norm = max(0.1, self.current_grad_norm * 0.7)
        elif avg_entropy < 1.4:
            # Warning: entropy dropping, increase coefficient
            self.current_entropy_coef = min(self.max_entropy_coef, self.current_entropy_coef * 1.1)
            self.current_grad_norm = max(0.2, self.current_grad_norm * 0.9)
        elif avg_entropy > 2.2:
            # Too exploratory, reduce coefficient
            self.current_entropy_coef = max(self.min_entropy_coef, self.current_entropy_coef * 0.95)
            self.current_grad_norm = min(1.0, self.current_grad_norm * 1.1)
        elif 1.5 <= avg_entropy <= 2.0:
            # Healthy range, slowly return to base
            self.current_entropy_coef = 0.9 * self.current_entropy_coef + 0.1 * self.entropy_coef
            self.current_grad_norm = 0.9 * self.current_grad_norm + 0.1 * self.base_grad_norm

        # LEARNING RATE ADAPTATION based on policy loss stability
        # Only reduce LR for instability; don't increase if using lr_decay (fine-tuning mode)
        if avg_policy_loss > 5.0:
            # Training unstable, reduce LR
            for pg in self.optimizer.param_groups:
                pg['lr'] = max(1e-5, pg['lr'] * 0.8)
        elif avg_policy_loss < 0.5 and avg_entropy > 1.4 and self.lr_decay >= 1.0:
            # Training stable and healthy, can increase LR (only if not in fine-tuning mode)
            for pg in self.optimizer.param_groups:
                pg['lr'] = min(self.base_lr * 2, pg['lr'] * 1.02)

    def play_game(self, opponent_random_prob=1.0, ai_difficulty='medium'):
        """Play game and collect experiences

        Args:
            opponent_random_prob: Probability opponents play randomly
            ai_difficulty: Difficulty of rule-based AI ('weak', 'medium', 'strong')
        """
        if self.reward_mode == 'pbrs_fixed':
            env = PBRSFixedRewardWrapper(player_id=0)
        else:
            env = SimplifiedRewardWrapper(player_id=0, reward_mode=self.reward_mode)
        obs, _ = env.reset()

        episode_rewards = []
        episode_obs = []
        episode_probs = []
        episode_log_probs = []

        done = False
        moves = 0
        max_moves = 500

        while not done and moves < max_moves:
            game = env.game_env.game
            current = game.get_current_player()
            current_id = game.players.index(current)

            if current_id == 0:
                moves += 1
                action, probs, log_prob = self._get_action(obs)
                action_id, vertex_id, edge_id = action

                episode_obs.append(obs['observation'].copy())
                episode_probs.append(probs)
                episode_log_probs.append(log_prob)

                next_obs, reward, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id,
                    trade_give_idx=0, trade_get_idx=0
                )

                episode_rewards.append(reward)
                obs = next_obs
                done = terminated or truncated
            else:
                success = play_opponent_turn(game, current_id, opponent_random_prob, ai_difficulty)
                if not success and game.can_end_turn():
                    game.end_turn()

                winner = game.check_victory_conditions()
                if winner is not None:
                    done = True

        # Calculate returns
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = np.clip(returns, -200, 200)

        # Add to buffer with log probs
        for obs_t, probs_t, ret_t, log_p in zip(episode_obs, episode_probs, returns, episode_log_probs):
            self.replay_buffer.add(obs_t, probs_t, ret_t, log_p)

        self.games_played += 1

        my_vp = env.game_env.game.players[0].calculate_victory_points()
        winner = env.game_env.game.check_victory_conditions()
        winner_id = game.players.index(winner) if winner else None

        # Track for curriculum
        self.phase_wins.append(1 if winner_id == 0 else 0)
        self.phase_vps.append(my_vp)

        return winner_id, my_vp, sum(episode_rewards)

    def _get_action(self, obs):
        """Get action from network with entropy-aware exploration"""
        with torch.no_grad():
            observation = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)
            action_mask = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(self.device)
            vertex_mask = torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(self.device)
            edge_mask = torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(self.device)

            action_probs, vertex_probs, edge_probs, _, _, _ = self.network.forward(
                observation, action_mask, vertex_mask, edge_mask
            )

            ap = action_probs.cpu().numpy()[0]
            vp = vertex_probs.cpu().numpy()[0]
            ep = edge_probs.cpu().numpy()[0]

        def safe_probs(probs):
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            if probs.sum() <= 0:
                probs = np.ones_like(probs) / len(probs)
            else:
                probs = probs / probs.sum()
            return probs

        ap = safe_probs(ap)
        vp = safe_probs(vp)
        ep = safe_probs(ep)

        # Adaptive exploration based on recent entropy health
        if len(self.entropy_history) > 10:
            recent_entropy = np.mean(list(self.entropy_history)[-10:])
            if recent_entropy < 1.2:
                # Entropy collapsing, add temperature
                temperature = 1.5
                ap = np.power(ap, 1/temperature)
                ap = ap / ap.sum()

        action_id = np.random.choice(len(ap), p=ap)
        vertex_id = np.random.choice(len(vp), p=vp)
        edge_id = np.random.choice(len(ep), p=ep)

        log_prob = np.log(ap[action_id] + 1e-8)

        return (action_id, vertex_id, edge_id), ap, log_prob

    def train_step(self):
        """Single training step with entropy stability"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        obs = torch.FloatTensor(batch['observations']).to(self.device)
        target_probs = torch.FloatTensor(batch['action_probs']).to(self.device)
        returns = torch.FloatTensor(batch['rewards']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)

        self.network.train()

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                loss_dict = self._compute_loss(obs, target_probs, returns, old_log_probs)
        else:
            loss_dict = self._compute_loss(obs, target_probs, returns, old_log_probs)

        loss = loss_dict['total_loss']

        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.current_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.current_grad_norm)
            self.optimizer.step()

        # Adapt hyperparameters based on this step's metrics
        self._adapt_hyperparameters(loss_dict['entropy'], loss_dict['policy_loss'])

        return {
            'policy': loss_dict['policy_loss'],
            'value': loss_dict['value_loss'],
            'entropy': loss_dict['entropy'],
            'entropy_penalty': loss_dict['entropy_penalty'],
            'kl': loss_dict.get('kl', 0.0),
            'entropy_coef': self.current_entropy_coef,
            'grad_norm': self.current_grad_norm,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def _compute_loss(self, obs, target_probs, returns, old_log_probs):
        """Compute loss with all entropy stability measures"""
        action_probs, vertex_probs, edge_probs, _, _, value = self.network.forward(obs)
        value = value.squeeze()

        # Policy loss with PPO-style clipping
        log_probs = torch.log(action_probs + 1e-8)
        action_log_probs = (log_probs * target_probs).sum(dim=1)

        # Compute advantages
        advantages = returns - value.detach()
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clipping
        ratio = torch.exp(action_log_probs - old_log_probs)
        clip_ratio = 0.2
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute entropy for ALL heads (not just action)
        action_entropy = -(action_probs * log_probs).sum(dim=1).mean()
        vertex_log_probs = torch.log(vertex_probs + 1e-8)
        vertex_entropy = -(vertex_probs * vertex_log_probs).sum(dim=1).mean()
        edge_log_probs = torch.log(edge_probs + 1e-8)
        edge_entropy = -(edge_probs * edge_log_probs).sum(dim=1).mean()

        # Combined entropy (weighted average)
        total_entropy = 0.6 * action_entropy + 0.2 * vertex_entropy + 0.2 * edge_entropy

        # Smooth entropy penalty
        entropy_penalty = self._compute_smooth_entropy_penalty(total_entropy.item())
        entropy_penalty_tensor = torch.tensor(entropy_penalty, device=self.device)

        # Value loss (reduced weight)
        value_loss = F.mse_loss(value, returns)

        # Approximate KL divergence for monitoring
        kl_div = 0.5 * ((ratio - 1) ** 2).mean()

        # Combined loss
        # - policy_loss: minimize
        # - entropy: maximize (so subtract)
        # - entropy_penalty: smooth penalty for low entropy
        # - value_loss: minimize (increased weight for better critic learning)
        total_loss = (
            policy_loss
            - self.current_entropy_coef * total_entropy
            + 5.0 * entropy_penalty_tensor  # Smooth penalty (not 200x!)
            + self.value_weight * value_loss  # Configurable value weight (default 0.5)
            + self.adaptive_kl_coef * torch.clamp(kl_div - self.max_kl, min=0.0)
        )

        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': total_entropy.item(),
            'action_entropy': action_entropy.item(),
            'vertex_entropy': vertex_entropy.item(),
            'edge_entropy': edge_entropy.item(),
            'entropy_penalty': entropy_penalty,
            'kl': kl_div.item()
        }

    def should_advance_curriculum(self, current_random_prob, ai_difficulty='medium'):
        """Check if agent has mastered current difficulty level"""
        if len(self.phase_wins) < 50:
            return False

        recent_wr = np.mean(list(self.phase_wins)[-50:])
        recent_vp = np.mean(list(self.phase_vps)[-50:])

        # Advancement criteria based on difficulty
        # Early phases: OR logic (easier to advance, build confidence)
        # Later phases: AND logic (must demonstrate real competence)
        if current_random_prob >= 0.85:
            # Very easy (100-90% random): OR logic - just need to show some progress
            return recent_wr > 0.10 or recent_vp > 3.5
        elif current_random_prob >= 0.65:
            # Easy (80-70% random): OR logic
            return recent_wr > 0.08 or recent_vp > 3.2
        elif current_random_prob >= 0.45:
            # Medium random (60-50%): AND logic - must show both WR and VP
            return recent_wr > 0.05 and recent_vp > 3.0
        elif current_random_prob >= 0.25:
            # Low random (40-25%): AND logic - stricter
            return recent_wr > 0.05 and recent_vp > 3.2
        elif ai_difficulty == 'medium':
            # 100% Medium AI: AND logic - must prove competence before facing Strong
            # Raised threshold: need >5% WR AND >3.5 VP to advance to Strong
            return recent_wr > 0.05 and recent_vp > 3.5
        else:
            # Strong AI: never auto-advance (final phase)
            return False

    def train(self, total_games=10000, save_path='models/curriculum_v3_stable',
              train_frequency=5, train_steps=15, min_games_per_phase=1000):
        """Curriculum training with adaptive phase transitions"""
        os.makedirs('models', exist_ok=True)

        # Phases: (random_prob, ai_difficulty, phase_name)
        # GRADUAL curriculum: 10% increments to prevent distribution shift collapse
        phases = [
            (1.0, 'weak', "100% Random"),
            (0.90, 'weak', "90% Random + 10% Weak AI"),
            (0.80, 'weak', "80% Random + 20% Weak AI"),
            (0.70, 'weak', "70% Random + 30% Weak AI"),
            (0.60, 'weak', "60% Random + 40% Weak AI"),
            (0.50, 'weak', "50% Random + 50% Weak AI"),
            (0.40, 'medium', "40% Random + 60% Medium AI"),
            (0.30, 'medium', "30% Random + 70% Medium AI"),
            (0.20, 'medium', "20% Random + 80% Medium AI"),
            (0.10, 'medium', "10% Random + 90% Medium AI"),
            (0.0, 'medium', "100% Medium AI"),
            (0.0, 'strong', "100% Strong AI"),
        ]

        print("\n" + "=" * 70)
        print("CURRICULUM TRAINING V3 - STABLE VERSION")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total games: {total_games}")
        print(f"Batch size: {self.batch_size}")
        print("STABILITY FEATURES:")
        print("  1. Smooth quadratic entropy penalty (not 200x step!)")
        print("  2. Adaptive entropy coefficient (0.05-0.5)")
        print("  3. Adaptive learning rate and gradient clipping")
        print("  4. PPO clipping for policy updates")
        print("  5. Multi-head entropy tracking (action+vertex+edge)")
        print("  6. Performance-based curriculum advancement")
        print("  7. Recency-biased experience replay")
        print("=" * 70 + "\n")

        current_phase = 0
        phase_game_count = 0
        total_wins = 0
        start_time = time.time()

        for game_num in range(1, total_games + 1):
            random_prob, ai_difficulty, phase_name = phases[current_phase]

            # Play game
            winner_id, vp, total_reward = self.play_game(random_prob, ai_difficulty)
            phase_game_count += 1

            if winner_id == 0:
                total_wins += 1

            # Train
            if game_num % train_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
                losses = [self.train_step() for _ in range(train_steps)]
                losses = [l for l in losses if l]

                if losses and game_num % 10 == 0:
                    avg_p = np.mean([l['policy'] for l in losses])
                    avg_v = np.mean([l['value'] for l in losses])
                    avg_e = np.mean([l['entropy'] for l in losses])
                    avg_ep = np.mean([l['entropy_penalty'] for l in losses])
                    curr_ec = losses[-1]['entropy_coef']
                    curr_gn = losses[-1]['grad_norm']
                    curr_lr = losses[-1]['lr']

                    elapsed = time.time() - start_time
                    speed = game_num / elapsed * 60
                    recent_wr = np.mean(list(self.phase_wins)[-50:]) * 100 if len(self.phase_wins) >= 10 else 0
                    recent_vp = np.mean(list(self.phase_vps)[-50:]) if len(self.phase_vps) >= 10 else 0

                    # Status indicators
                    entropy_status = "✓" if 1.4 <= avg_e <= 2.2 else "⚠️" if avg_e < 1.0 else "↑"

                    print(f"  [{phase_name}] Game {game_num:5d}/{total_games} | "
                          f"WR: {recent_wr:5.1f}% | VP: {recent_vp:.1f} | "
                          f"Speed: {speed:.1f} g/min")
                    print(f"    └─ policy={avg_p:+.4f}, value={avg_v:.4f}, "
                          f"entropy={avg_e:.3f}{entropy_status}, "
                          f"ec={curr_ec:.3f}, gn={curr_gn:.2f}, lr={curr_lr:.1e}")

            # Check for curriculum advancement
            if (phase_game_count >= min_games_per_phase and
                current_phase < len(phases) - 1 and
                self.should_advance_curriculum(random_prob, ai_difficulty)):

                print(f"\n  ★ ADVANCING from {phase_name} to {phases[current_phase + 1][2]}")
                print(f"    Games in phase: {phase_game_count}")
                print(f"    Recent WR: {np.mean(list(self.phase_wins)[-50:])*100:.1f}%")
                print(f"    Recent VP: {np.mean(list(self.phase_vps)[-50:]):.1f}\n")

                # Clear most old experiences to reduce distribution mismatch
                # Keep only 25% to allow faster adaptation to new opponent mix
                self.replay_buffer.clear_old(keep_fraction=0.25)

                current_phase += 1
                phase_game_count = 0
                self.phase_wins.clear()
                self.phase_vps.clear()

                # Save checkpoint
                self.save(f"{save_path}_phase{current_phase}.pt")

            # Periodic saves and LR decay
            if game_num % 1000 == 0:
                self.save(f"{save_path}_game{game_num}.pt")
                # Apply LR decay
                if self.lr_decay < 1.0:
                    for pg in self.optimizer.param_groups:
                        old_lr = pg['lr']
                        pg['lr'] = max(1e-5, pg['lr'] * self.lr_decay)
                        if pg['lr'] != old_lr:
                            print(f"    LR decay: {old_lr:.2e} → {pg['lr']:.2e}")

        self.save(f"{save_path}_final.pt")

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Total games: {total_games}")
        print(f"Overall win rate: {total_wins/total_games*100:.1f}%")
        print(f"Final phase: {phases[current_phase][2]}")
        print(f"Time: {total_time/60:.1f} min")
        print("=" * 70)

    def save(self, path):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_played': self.games_played,
            'entropy_coef': self.current_entropy_coef,
            'grad_norm': self.current_grad_norm,
        }, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-games', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--train-frequency', type=int, default=5)
    parser.add_argument('--train-steps', type=int, default=15)
    parser.add_argument('--min-games-per-phase', type=int, default=1000,
                        help='Minimum games before curriculum can advance')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to existing model to continue training')
    parser.add_argument('--reward-mode', type=str, default='vp_only',
                        choices=['sparse', 'vp_only', 'simplified', 'pbrs_fixed'],
                        help='Reward mode: sparse, vp_only (default), simplified, or pbrs_fixed')
    parser.add_argument('--lr-decay', type=float, default=1.0,
                        help='Learning rate decay multiplier per 1000 games (default: 1.0 = no decay, try 0.9 for fine-tuning)')
    parser.add_argument('--value-weight', type=float, default=0.5,
                        help='Weight for value loss (default: 0.5, increased from 0.1 for better critic learning)')
    args = parser.parse_args()

    trainer = CurriculumTrainerV3(
        model_path=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        reward_mode=args.reward_mode,
        lr_decay=args.lr_decay,
        value_weight=args.value_weight
    )
    trainer.train(
        total_games=args.total_games,
        train_frequency=args.train_frequency,
        train_steps=args.train_steps,
        min_games_per_phase=args.min_games_per_phase
    )
