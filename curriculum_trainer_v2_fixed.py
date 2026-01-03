"""
Curriculum Training using Full CatanEnv - FIXED VERSION

Fixes:
1. Removed return normalization that destroyed learning signal
2. Increased entropy coefficient from 0.01 to 0.1
3. Reduced value loss weight from 0.5 to 0.1
4. Fixed random opponent to actually play randomly
5. Increased training frequency and steps
6. Added adaptive entropy scaling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    """Weak opponent - random legal moves (FIXED: actually plays randomly now)"""
    player = game.players[player_id]

    # Initial placement
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
                    return True
        else:
            vertices = game.game_board.vertices
            valid = [v for v in vertices if v.structure is None and
                    not any(adj.structure for adj in v.adjacent_vertices)]
            if valid:
                game.try_place_initial_settlement(random.choice(valid), player)
                return True
        return False

    # Normal play
    if game.can_roll_dice():
        game.roll_dice()
        return True

    if game.can_trade_or_build():
        actions = []
        res = player.resources

        can_settlement = (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1 and
                         res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1)
        if can_settlement:
            v = game.get_buildable_vertices_for_settlements()
            if v: actions.append(('sett', v))

        can_city = (res[ResourceType.WHEAT] >= 2 and res[ResourceType.ORE] >= 3)
        if can_city:
            v = game.get_buildable_vertices_for_cities()
            if v: actions.append(('city', v))

        can_road = (res[ResourceType.WOOD] >= 1 and res[ResourceType.BRICK] >= 1)
        if can_road:
            e = game.get_buildable_edges()
            if e: actions.append(('road', e))

        can_dev = (res[ResourceType.WHEAT] >= 1 and res[ResourceType.SHEEP] >= 1 and
                  res[ResourceType.ORE] >= 1)
        if can_dev and not game.dev_deck.is_empty():
            actions.append(('dev', None))

        # FIX: Changed from 0.6 to 0.9 - now actually plays most of the time
        if actions and random.random() < 0.9:
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
    return False


def play_opponent_turn(game, player_id, random_prob):
    """Play opponent with mix of random/smart"""
    if random.random() < random_prob:
        return play_random_turn(game, player_id)
    else:
        from rule_based_ai import play_rule_based_turn
        # rule_based_ai expects env wrapper, create minimal one
        class MinimalEnv:
            def __init__(self, g):
                self.game_env = type('obj', (object,), {'game': g})()
        return play_rule_based_turn(MinimalEnv(game), player_id)


class ReplayBuffer:
    def __init__(self, max_size=500000):
        self.buffer = deque(maxlen=max_size)

    def add(self, obs, action_probs, reward):
        self.buffer.append({'obs': obs, 'probs': action_probs, 'reward': reward})

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return {
            'observations': np.array([self.buffer[i]['obs'] for i in indices]),
            'action_probs': np.array([self.buffer[i]['probs'] for i in indices]),
            'rewards': np.array([self.buffer[i]['reward'] for i in indices])
        }

    def __len__(self):
        return len(self.buffer)


class CurriculumTrainerV2:
    """Uses full CatanEnv with your existing reward system - FIXED"""

    def __init__(self, model_path=None, learning_rate=1e-3, batch_size=None, epsilon=0.1, reward_mode='vp_only'):
        self.device = get_device()

        if batch_size is None:
            batch_size = 2048 if self.device.type == 'cuda' else 256
        self.batch_size = batch_size

        self.network_wrapper = NetworkWrapper(model_path=model_path, device=str(self.device))
        self.network = self.network_wrapper.policy

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()

        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        self.games_played = 0
        self.entropy_coef = 0.3  # FIX: Increased to 0.3 (was 0.15, still collapsed!)
        self.epsilon = epsilon  # Epsilon-greedy exploration
        self.min_entropy = 0.5  # Minimum entropy threshold
        self.reward_mode = reward_mode  # Simplified reward mode
        print(f"Batch size: {self.batch_size}")
        print(f"Reward mode: {self.reward_mode}")
        print(f"Entropy coefficient: {self.entropy_coef} (constant)")
        print(f"Epsilon-greedy: {self.epsilon}")
        print(f"Minimum entropy threshold: {self.min_entropy}")

    def play_game(self, opponent_random_prob=1.0):
        """Play game using SimplifiedRewardWrapper or PBRSFixedRewardWrapper"""
        # Choose wrapper based on reward_mode
        if self.reward_mode == 'pbrs_fixed':
            env = PBRSFixedRewardWrapper(player_id=0)
        else:
            env = SimplifiedRewardWrapper(player_id=0, reward_mode=self.reward_mode)
        obs, _ = env.reset()

        episode_rewards = []
        episode_obs = []
        episode_probs = []

        done = False
        moves = 0
        max_moves = 500

        while not done and moves < max_moves:
            game = env.game_env.game
            current = game.get_current_player()
            current_id = game.players.index(current)

            if current_id == 0:
                # Our agent's turn - use network
                moves += 1

                action, probs = self._get_action(obs)
                action_id, vertex_id, edge_id = action

                # Store experience
                episode_obs.append(obs['observation'].copy())
                episode_probs.append(probs)

                # Step environment - THIS USES YOUR FULL REWARD SYSTEM
                next_obs, reward, terminated, truncated, info = env.step(
                    action_id, vertex_id, edge_id,
                    trade_give_idx=0, trade_get_idx=0  # Default trade params
                )

                episode_rewards.append(reward)
                obs = next_obs
                done = terminated or truncated
            else:
                # Opponent's turn
                success = play_opponent_turn(game, current_id, opponent_random_prob)
                if not success and game.can_end_turn():
                    game.end_turn()

                # Check if game ended
                winner = game.check_victory_conditions()
                if winner is not None:
                    done = True

        # Calculate returns (discounted sum of rewards)
        gamma = 0.99
        returns = []
        G = 0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # FIX: Scale returns to reasonable range for training
        # The issue: returns can be -1500 to +200, causing huge losses
        # Solution: Divide by a constant to bring into [-10, 10] range
        # This preserves relative magnitudes while keeping gradients stable
        returns = np.array(returns)
        returns = returns / 100.0  # Scale down by 100x

        # Clip extreme outliers just in case
        returns = np.clip(returns, -50, 50)

        # Add to buffer
        for obs_t, probs_t, ret_t in zip(episode_obs, episode_probs, returns):
            self.replay_buffer.add(obs_t, probs_t, ret_t)

        self.games_played += 1

        # Get final stats
        my_vp = env.game_env.game.players[0].calculate_victory_points()
        winner = env.game_env.game.check_victory_conditions()
        winner_id = game.players.index(winner) if winner else None

        return winner_id, my_vp, sum(episode_rewards)

    def _get_action(self, obs):
        """Get action from network with epsilon-greedy exploration"""
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

        # Fix NaN/invalid probabilities
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

        # Epsilon-greedy: random action with probability epsilon
        if np.random.random() < self.epsilon:
            # Random action (uniform over valid actions)
            action_id = np.random.choice(len(ap))
            vertex_id = np.random.choice(len(vp))
            edge_id = np.random.choice(len(ep))
        else:
            # Sample from network's distribution
            action_id = np.random.choice(len(ap), p=ap)
            vertex_id = np.random.choice(len(vp), p=vp)
            edge_id = np.random.choice(len(ep), p=ep)

        return (action_id, vertex_id, edge_id), ap

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        obs = torch.FloatTensor(batch['observations']).to(self.device)
        target_probs = torch.FloatTensor(batch['action_probs']).to(self.device)
        returns = torch.FloatTensor(batch['rewards']).to(self.device)

        self.network.train()

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                action_probs, _, _, _, _, value = self.network.forward(obs)
                value = value.squeeze()

                # Policy gradient loss with baseline
                # Compute log probability of actions that were taken
                log_probs = torch.log(action_probs + 1e-8)
                action_log_probs = (log_probs * target_probs).sum(dim=1)

                # Advantage = return - baseline (value prediction)
                advantages = returns - value.detach()

                # Policy loss: maximize log_prob * advantage
                policy_loss = -(action_log_probs * advantages).mean()

                # Entropy bonus for exploration
                entropy = -(action_probs * log_probs).sum(dim=1).mean()

                # Value loss
                value_loss = F.mse_loss(value, returns)

                # Entropy floor penalty: punish if entropy too low
                entropy_penalty = torch.clamp(self.min_entropy - entropy, min=0.0)

                # FIX: Adjusted loss weights
                # - Entropy coef: 0.3 (increased from 0.15)
                # - Entropy floor penalty: prevents collapse below 0.5
                loss = policy_loss + 0.1 * value_loss - self.entropy_coef * entropy + 2.0 * entropy_penalty

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            action_probs, _, _, _, _, value = self.network.forward(obs)
            value = value.squeeze()

            # Policy gradient loss with baseline
            log_probs = torch.log(action_probs + 1e-8)
            action_log_probs = (log_probs * target_probs).sum(dim=1)

            # Advantage = return - baseline
            advantages = returns - value.detach()

            # Policy loss: maximize log_prob * advantage
            policy_loss = -(action_log_probs * advantages).mean()

            # Entropy bonus for exploration
            entropy = -(action_probs * log_probs).sum(dim=1).mean()

            # Value loss
            value_loss = F.mse_loss(value, returns)

            # Entropy floor penalty: punish if entropy too low
            entropy_penalty = torch.clamp(self.min_entropy - entropy, min=0.0)

            # FIX: Adjusted loss weights
            # - Entropy coef: 0.3 (increased from 0.15)
            # - Entropy floor penalty: prevents collapse below 0.5
            loss = policy_loss + 0.1 * value_loss - self.entropy_coef * entropy + 2.0 * entropy_penalty

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()

        return {
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'entropy': entropy.item(),
            'entropy_penalty': entropy_penalty.item()
        }

    def train(self, games_per_phase=1000, parallel_games=8, save_path='models/curriculum_v2_fixed'):
        """Curriculum training with phases"""
        os.makedirs('models', exist_ok=True)

        phases = [
            (1.0, "Random opponents"),
            (0.75, "75% random"),
            (0.5, "50% random"),
            (0.25, "25% random"),
            (0.0, "Full strength"),
        ]

        print("\n" + "=" * 70)
        print("CURRICULUM TRAINING V2 - FIXED VERSION")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Games per phase: {games_per_phase}")
        print(f"Batch size: {self.batch_size}")
        print("FIXES:")
        print("  1. SIMPLIFIED REWARDS (vp_only mode - no complex PBRS!)")
        print("  2. Scaled returns by /100 (prevents gradient explosion)")
        print("  3. STRONG entropy coefficient 0.3 (2x increase!)")
        print("  4. Entropy FLOOR penalty (prevents collapse below 0.5)")
        print("  5. Epsilon-greedy exploration (10% random actions)")
        print("  6. Reduced value loss weight 0.5 -> 0.1")
        print("  7. Fixed random opponent to play 90% of the time")
        print("  8. Increased training frequency (4x more steps)")
        print("\nREWARD SYSTEM:")
        print(f"  Mode: {self.reward_mode}")
        if self.reward_mode == 'pbrs_fixed':
            print("  • VP gain: +10 per VP")
            print("  • Win bonus: +100 (strong terminal reward)")
            print("  • PBRS: Scaled potential (±5 max)")
            print("  • PBRS is small guide, base rewards dominate")
        else:
            print("  • VP gain: +10 per VP")
            print("  • Win bonus: +50")
            print("  • No PBRS decay (no -1000 drift!)")
        print("=" * 70 + "\n")

        total_wins = 0
        total_games = 0
        start_time = time.time()

        for phase_idx, (random_prob, phase_name) in enumerate(phases):
            print(f"\n{'='*70}")
            print(f"PHASE {phase_idx + 1}: {phase_name}")
            print(f"{'='*70}")

            phase_wins = 0
            phase_vp = 0
            phase_rewards = 0
            recent_wins = deque(maxlen=50)
            recent_vp = deque(maxlen=50)

            for game_num in range(1, games_per_phase + 1):
                # Play game
                winner_id, vp, total_reward = self.play_game(random_prob)

                total_games += 1
                phase_vp += vp
                phase_rewards += total_reward
                recent_vp.append(vp)

                if winner_id == 0:
                    phase_wins += 1
                    total_wins += 1
                    recent_wins.append(1)
                else:
                    recent_wins.append(0)

                # FIX: Train more frequently (every 5 games instead of 10)
                if game_num % 5 == 0:
                    elapsed = time.time() - start_time
                    speed = total_games / elapsed * 60
                    recent_wr = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
                    avg_vp = sum(recent_vp) / len(recent_vp) if recent_vp else 0
                    avg_reward = phase_rewards / game_num

                    # FIX: Increased training steps from 10 to 20
                    if len(self.replay_buffer) >= self.batch_size:
                        losses = [self.train_step() for _ in range(20)]
                        losses = [l for l in losses if l]
                        if losses:
                            avg_p = np.mean([l['policy'] for l in losses])
                            avg_v = np.mean([l['value'] for l in losses])
                            avg_e = np.mean([l['entropy'] for l in losses])
                            avg_ep = np.mean([l['entropy_penalty'] for l in losses])

                            # Only print every 10 games to reduce spam
                            if game_num % 10 == 0:
                                print(f"  Game {game_num:3d}/{games_per_phase} | "
                                      f"WR: {recent_wr:5.1f}% | "
                                      f"VP: {avg_vp:.1f} | "
                                      f"Reward: {avg_reward:.1f} | "
                                      f"Speed: {speed:.1f} g/min")
                                # Show warning if entropy penalty is active
                                penalty_warn = " ⚠️" if avg_ep > 0.01 else ""
                                print(f"    └─ Train: policy={avg_p:.4f}, value={avg_v:.4f}, entropy={avg_e:.4f}{penalty_warn}")

            # Phase summary
            phase_wr = phase_wins / games_per_phase * 100
            phase_avg_vp = phase_vp / games_per_phase
            print(f"\n  Phase {phase_idx + 1} Complete: WR={phase_wr:.1f}%, Avg VP={phase_avg_vp:.1f}")
            self.save(f"{save_path}_phase{phase_idx + 1}.pt")

        self.save(f"{save_path}_final.pt")

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Total games: {total_games}")
        print(f"Overall win rate: {total_wins/total_games*100:.1f}%")
        print(f"Time: {total_time/60:.1f} min")
        print("=" * 70)

    def save(self, path):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_played': self.games_played,
        }, path)
        print(f"  Saved: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-per-phase', type=int, default=1000)
    parser.add_argument('--parallel', type=int, default=1)  # Sequential for now
    args = parser.parse_args()

    trainer = CurriculumTrainerV2()
    trainer.train(games_per_phase=args.games_per_phase)
