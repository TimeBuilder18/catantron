"""
Fast AlphaZero Training for Catan

Key optimizations:
1. Very few MCTS simulations (5-10) - network guides most decisions
2. Use network directly for early training, add MCTS later
3. Parallel game execution
4. Smaller, faster training loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from game_state import GameState
from network_wrapper import NetworkWrapper


def get_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class ReplayBuffer:
    def __init__(self, max_size=500000):
        self.buffer = deque(maxlen=max_size)

    def add_batch(self, examples):
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)

        observations = []
        action_probs = []
        values = []

        for idx in indices:
            example = self.buffer[idx]
            observations.append(example['observation'])
            action_probs.append(example['action_probs'])
            values.append(example['value'])

        return {
            'observations': np.array(observations),
            'action_probs': np.array(action_probs),
            'values': np.array(values)
        }

    def __len__(self):
        return len(self.buffer)


class FastTrainer:
    """
    Fast training using network policy directly (no MCTS during early training)
    Then gradually add MCTS as network improves
    """

    def __init__(self, model_path=None, learning_rate=1e-3, batch_size=None):
        self.device = get_device()

        if batch_size is None:
            batch_size = 2048 if self.device.type == 'cuda' else 256
        self.batch_size = batch_size

        # Create network
        self.network_wrapper = NetworkWrapper(model_path=model_path, device=str(self.device))
        self.network = self.network_wrapper.policy

        # Training setup
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()

        # Mixed precision for CUDA
        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')

        self.games_played = 0
        self.training_steps = 0

        print(f"Batch size: {self.batch_size}")

    def play_game_fast(self, use_mcts=False, mcts_sims=5):
        """
        Play one game using network directly (fast) or with light MCTS
        """
        from rule_based_ai import play_rule_based_turn

        state = GameState()
        examples = []
        move_count = 0
        max_moves = 500

        while not state.is_terminal() and move_count < max_moves:
            current_player = state.get_current_player()

            if current_player == 0:
                move_count += 1
                obs = state.get_observation()

                if use_mcts:
                    # Light MCTS
                    from mcts import MCTS
                    mcts = MCTS(self.network_wrapper, num_simulations=mcts_sims, c_puct=1.0)
                    best_action, action_probs_dict = mcts.search(state)
                    action_probs = self._dict_to_array(action_probs_dict)
                else:
                    # Direct network policy (much faster)
                    best_action, action_probs = self._get_network_action(state, obs)

                examples.append({
                    'observation': obs['observation'].copy(),
                    'action_probs': action_probs,
                })

                action_id, vertex_id, edge_id = best_action
                state.apply_action(action_id, vertex_id, edge_id)
            else:
                success = play_rule_based_turn(state.env, current_player)
                if not success:
                    if state.env.game_env.game.can_end_turn():
                        state.env.game_env.game.end_turn()

        # Get outcome
        winner = state.get_winner()
        value = 1.0 if winner == 0 else (-1.0 if winner is not None else 0.0)

        # Add value to examples
        training_examples = [
            {'observation': ex['observation'], 'action_probs': ex['action_probs'], 'value': value}
            for ex in examples
        ]

        self.replay_buffer.add_batch(training_examples)
        self.games_played += 1

        return winner, len(examples), [state.get_victory_points(i) for i in range(4)]

    def _get_network_action(self, state, obs):
        """Get action directly from network (no MCTS)"""
        with torch.no_grad():
            observation = torch.FloatTensor(obs['observation']).unsqueeze(0).to(self.device)
            action_mask = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(self.device)
            vertex_mask = torch.FloatTensor(obs['vertex_mask']).unsqueeze(0).to(self.device)
            edge_mask = torch.FloatTensor(obs['edge_mask']).unsqueeze(0).to(self.device)

            action_probs, vertex_probs, edge_probs, _, _, _ = self.network.forward(
                observation, action_mask, vertex_mask, edge_mask
            )

            # Sample action
            action_probs_np = action_probs.cpu().numpy()[0]
            vertex_probs_np = vertex_probs.cpu().numpy()[0]
            edge_probs_np = edge_probs.cpu().numpy()[0]

            # Get legal actions
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                return (7, 0, 0), np.zeros(11)  # end_turn fallback

            # Filter to legal and sample
            action_weights = {}
            for (action_id, vertex_id, edge_id) in legal_actions:
                weight = action_probs_np[action_id]
                if action_id in [1, 3, 4]:  # vertex actions
                    weight *= vertex_probs_np[vertex_id]
                elif action_id in [2, 5]:  # edge actions
                    weight *= edge_probs_np[edge_id]
                action_weights[(action_id, vertex_id, edge_id)] = weight

            # Normalize and sample
            actions = list(action_weights.keys())
            weights = np.array(list(action_weights.values()))
            weights = weights / (weights.sum() + 1e-8)

            # Add exploration noise
            weights = 0.75 * weights + 0.25 * np.ones_like(weights) / len(weights)
            weights = weights / weights.sum()

            idx = np.random.choice(len(actions), p=weights)
            chosen_action = actions[idx]

            # Create action probs array (just action type distribution)
            return chosen_action, action_probs_np

    def _dict_to_array(self, action_probs_dict):
        """Convert MCTS action probs dict to array"""
        probs = np.zeros(11, dtype=np.float32)
        for (action_id, _, _), prob in action_probs_dict.items():
            probs[action_id] += prob
        if probs.sum() > 0:
            probs /= probs.sum()
        return probs

    def train_step(self):
        """One training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        observations = torch.FloatTensor(batch['observations']).to(self.device)
        target_probs = torch.FloatTensor(batch['action_probs']).to(self.device)
        target_values = torch.FloatTensor(batch['values']).to(self.device)

        self.network.train()

        if self.use_amp:
            with torch.amp.autocast('cuda'):
                action_probs, _, _, _, _, value = self.network.forward(
                    observations, action_mask=None, vertex_mask=None, edge_mask=None
                )
                value = value.squeeze()

                policy_loss = -torch.sum(target_probs * torch.log(action_probs + 1e-8), dim=1).mean()
                value_loss = F.mse_loss(value, target_values)
                total_loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            action_probs, _, _, _, _, value = self.network.forward(
                observations, action_mask=None, vertex_mask=None, edge_mask=None
            )
            value = value.squeeze()

            policy_loss = -torch.sum(target_probs * torch.log(action_probs + 1e-8), dim=1).mean()
            value_loss = F.mse_loss(value, target_values)
            total_loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.training_steps += 1

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }

    def train(self, num_games=1000, parallel_games=8, use_mcts_after=500,
              mcts_sims=5, save_path='models/fast_alphazero'):
        """
        Training loop:
        - First phase: Network only (fast)
        - Second phase: Add light MCTS (slower but better)
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'models', exist_ok=True)

        print("\n" + "=" * 70)
        print("FAST ALPHAZERO TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total games: {num_games}")
        print(f"Parallel games: {parallel_games}")
        print(f"Use MCTS after game: {use_mcts_after}")
        print(f"MCTS simulations (when used): {mcts_sims}")
        print(f"Batch size: {self.batch_size}")
        print("=" * 70 + "\n")

        wins = 0
        total_vp = 0
        start_time = time.time()
        recent_wins = deque(maxlen=100)
        recent_vp = deque(maxlen=100)

        game_num = 0

        while game_num < num_games:
            batch_start = time.time()

            # Decide if using MCTS
            use_mcts = game_num >= use_mcts_after

            # Play parallel games
            games_to_play = min(parallel_games, num_games - game_num)

            with ThreadPoolExecutor(max_workers=games_to_play) as executor:
                futures = [
                    executor.submit(self.play_game_fast, use_mcts, mcts_sims)
                    for _ in range(games_to_play)
                ]

                for future in as_completed(futures):
                    winner, num_examples, vps = future.result()
                    game_num += 1

                    my_vp = vps[0]
                    total_vp += my_vp
                    recent_vp.append(my_vp)

                    if winner == 0:
                        wins += 1
                        recent_wins.append(1)
                    else:
                        recent_wins.append(0)

            batch_time = time.time() - batch_start

            # Progress
            elapsed = time.time() - start_time
            games_per_min = game_num / elapsed * 60
            win_rate = wins / game_num * 100
            recent_wr = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
            avg_vp = sum(recent_vp) / len(recent_vp) if recent_vp else 0

            mode = "MCTS" if use_mcts else "Network"
            print(f"Game {game_num:4d}/{num_games} | "
                  f"WR: {win_rate:5.1f}% | "
                  f"VP: {avg_vp:.1f} | "
                  f"Buffer: {len(self.replay_buffer):6d} | "
                  f"Speed: {games_per_min:.1f} g/min | "
                  f"[{mode}]")

            # Training
            if len(self.replay_buffer) >= self.batch_size:
                losses = []
                for _ in range(10):
                    loss = self.train_step()
                    if loss:
                        losses.append(loss)

                if losses:
                    avg_policy = np.mean([l['policy_loss'] for l in losses])
                    avg_value = np.mean([l['value_loss'] for l in losses])
                    print(f"  └─ Training: policy={avg_policy:.4f}, value={avg_value:.4f}")

            # Save
            if game_num % 100 == 0:
                self.save(f"{save_path}_game_{game_num}.pt")
                print(f"  └─ Saved checkpoint")

        self.save(f"{save_path}_final.pt")

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print(f"Games: {num_games} | Win rate: {wins / num_games * 100:.1f}% | Avg VP: {total_vp / num_games:.1f}")
        print(f"Time: {total_time / 60:.1f} min | Speed: {num_games / total_time * 60:.1f} games/min")
        print("=" * 70)

    def save(self, path):
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_played': self.games_played,
        }, path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=500)
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--mcts-after', type=int, default=300, help='Start using MCTS after N games')
    parser.add_argument('--mcts-sims', type=int, default=5)
    args = parser.parse_args()

    trainer = FastTrainer()
    trainer.train(
        num_games=args.games,
        parallel_games=args.parallel,
        use_mcts_after=args.mcts_after,
        mcts_sims=args.mcts_sims
    )