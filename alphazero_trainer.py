"""
AlphaZero Training Loop for Catan

Auto-detects best device (CUDA > MPS > CPU) and optimizes accordingly.

The training cycle:
1. Self-play: MCTS agent plays games, collecting training data
2. Train: Network learns to predict MCTS policy and game outcomes
3. Repeat: Better network → better MCTS → better data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time

from game_state import GameState
from mcts import MCTS
from network_wrapper import NetworkWrapper


def get_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class ReplayBuffer:
    """Stores training examples from self-play games"""

    def __init__(self, max_size=500000):
        self.buffer = deque(maxlen=max_size)

    def add(self, observation, action_probs, value):
        self.buffer.append({
            'observation': observation,
            'action_probs': action_probs,
            'value': value
        })

    def add_batch(self, examples):
        """Add multiple examples at once"""
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)

        observations = []
        action_targets = []
        vertex_targets = []
        edge_targets = []
        values = []

        for idx in indices:
            example = self.buffer[idx]
            observations.append(example['observation'])
            action_targets.append(example['action_target'])
            vertex_targets.append(example['vertex_target'])
            edge_targets.append(example['edge_target'])
            values.append(example['value'])

        return {
            'observations': np.array(observations),
            'action_targets': np.array(action_targets),
            'vertex_targets': np.array(vertex_targets),
            'edge_targets': np.array(edge_targets),
            'values': np.array(values)
        }

    def __len__(self):
        return len(self.buffer)


class AlphaZeroTrainer:
    """
    AlphaZero-style trainer for Catan
    Auto-optimizes for available hardware
    """

    def __init__(self,
                 model_path=None,
                 num_simulations=50,
                 c_puct=1.0,
                 learning_rate=1e-3,
                 batch_size=None,  # Auto-set based on device
                 buffer_size=500000):

        # Auto-detect device
        self.device = get_device()

        # Auto-set batch size based on device
        if batch_size is None:
            if self.device.type == 'cuda':
                # RTX 2080 Super has 8GB - can handle large batches
                batch_size = 1024
            elif self.device.type == 'mps':
                batch_size = 256
            else:
                batch_size = 64
        self.batch_size = batch_size

        # Create network wrapper (this creates the network)
        self.network_wrapper = NetworkWrapper(model_path=model_path, device=str(self.device))
        self.network = self.network_wrapper.policy

        # Create MCTS
        self.mcts = MCTS(
            policy_network=self.network_wrapper,
            num_simulations=num_simulations,
            c_puct=c_puct
        )
        self.num_simulations = num_simulations

        # Training setup
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)

        # Mixed precision for CUDA
        self.use_amp = (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training (FP16)")

        # Stats
        self.games_played = 0
        self.training_steps = 0

        print(f"Batch size: {self.batch_size}")
        print(f"MCTS simulations: {num_simulations}")

    def self_play_game(self, verbose=False):
        """Play one game using MCTS, collecting training data"""
        from rule_based_ai import play_rule_based_turn

        state = GameState()
        examples = []
        move_count = 0
        max_moves = 500

        # Use higher temperature early for exploration, lower late for exploitation
        explore_moves = 30  # First 30 moves use temperature=1.0

        while not state.is_terminal() and move_count < max_moves:
            current_player = state.get_current_player()

            if current_player == 0:
                move_count += 1
                obs = state.get_observation()

                # Temperature: explore early, exploit late
                temp = 1.0 if move_count <= explore_moves else 0.3

                # MCTS search
                best_action, action_probs_dict = self.mcts.search(state, temperature=temp)

                # Convert to separate target arrays for action, vertex, edge heads
                action_target, vertex_target, edge_target = self._action_probs_to_targets(action_probs_dict)

                examples.append({
                    'observation': obs['observation'].copy(),
                    'action_target': action_target,
                    'vertex_target': vertex_target,
                    'edge_target': edge_target,
                    'player': current_player
                })

                action_id, vertex_id, edge_id = best_action
                reward, done = state.apply_action(action_id, vertex_id, edge_id)
            else:
                success = play_rule_based_turn(state.env, current_player)
                if not success:
                    if state.env.game_env.game.can_end_turn():
                        state.env.game_env.game.end_turn()

        # Determine outcome from player 0's perspective
        winner = state.get_winner()
        final_value = 1.0 if winner == 0 else (-1.0 if winner is not None else 0.0)

        # Create training examples with discounted values
        gamma = 0.99
        training_examples = []
        num_examples = len(examples)

        for i, ex in enumerate(examples):
            steps_to_end = num_examples - i - 1
            discounted_value = final_value * (gamma ** steps_to_end)

            training_examples.append({
                'observation': ex['observation'],
                'action_target': ex['action_target'],
                'vertex_target': ex['vertex_target'],
                'edge_target': ex['edge_target'],
                'value': discounted_value
            })

        self.replay_buffer.add_batch(training_examples)
        self.games_played += 1

        if verbose:
            vps = [state.get_victory_points(i) for i in range(4)]
            print(f"Game {self.games_played}: Winner={winner}, VPs={vps}, Examples={len(examples)}")

        return winner, len(examples)

    def _action_probs_to_targets(self, action_probs_dict):
        """
        Convert MCTS visit-count probabilities into separate training targets
        for the action, vertex, and edge heads.

        This preserves location information so the network learns WHERE to build,
        not just WHAT to build.
        """
        action_target = np.zeros(14, dtype=np.float32)
        vertex_target = np.zeros(54, dtype=np.float32)
        edge_target = np.zeros(72, dtype=np.float32)

        for (action_id, vertex_id, edge_id), prob in action_probs_dict.items():
            # Accumulate action type probabilities
            if action_id < 14:
                action_target[action_id] += prob

            # Accumulate vertex probabilities for vertex-based actions
            if action_id in (1, 3, 4) and vertex_id < 54:
                vertex_target[vertex_id] += prob

            # Accumulate edge probabilities for edge-based actions
            if action_id in (2, 5) and edge_id < 72:
                edge_target[edge_id] += prob

        # Normalize each target (they may not sum to 1 individually)
        if action_target.sum() > 0:
            action_target /= action_target.sum()
        if vertex_target.sum() > 0:
            vertex_target /= vertex_target.sum()
        if edge_target.sum() > 0:
            edge_target /= edge_target.sum()

        return action_target, vertex_target, edge_target

    def _compute_loss(self, batch):
        """Compute combined loss for action, vertex, edge, and value heads."""
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        target_actions = torch.FloatTensor(batch['action_targets']).to(self.device)
        target_vertices = torch.FloatTensor(batch['vertex_targets']).to(self.device)
        target_edges = torch.FloatTensor(batch['edge_targets']).to(self.device)
        target_values = torch.FloatTensor(batch['values']).to(self.device)

        # Forward pass (no masks during training — targets already encode valid actions)
        action_probs, vertex_probs, edge_probs, _, _, value = self.network.forward(
            observations, action_mask=None, vertex_mask=None, edge_mask=None
        )
        value = value.squeeze()

        # Cross-entropy losses for each head
        action_loss = -torch.sum(target_actions * torch.log(action_probs + 1e-8), dim=1).mean()
        value_loss = F.mse_loss(value, target_values)

        # Vertex/edge losses only on examples that have location targets
        # (skip examples where target is all zeros = no location action was taken)
        vertex_mask = target_vertices.sum(dim=1) > 0
        edge_mask = target_edges.sum(dim=1) > 0

        vertex_loss = torch.tensor(0.0, device=self.device)
        edge_loss = torch.tensor(0.0, device=self.device)

        if vertex_mask.any():
            vertex_loss = -torch.sum(
                target_vertices[vertex_mask] * torch.log(vertex_probs[vertex_mask] + 1e-8), dim=1
            ).mean()

        if edge_mask.any():
            edge_loss = -torch.sum(
                target_edges[edge_mask] * torch.log(edge_probs[edge_mask] + 1e-8), dim=1
            ).mean()

        total_loss = action_loss + value_loss + vertex_loss + edge_loss

        return total_loss, {
            'action_loss': action_loss.item(),
            'vertex_loss': vertex_loss.item(),
            'edge_loss': edge_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }

    def train_step(self):
        """One training step with optional mixed precision"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        self.network.train()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                total_loss, loss_dict = self._compute_loss(batch)
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss, loss_dict = self._compute_loss(batch)
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.scheduler.step()
        self.training_steps += 1

        return loss_dict

    def train_batch(self, num_steps=10):
        """Multiple training steps"""
        losses = []
        for _ in range(num_steps):
            loss = self.train_step()
            if loss:
                losses.append(loss)
        return losses

    def train(self,
              num_games=100,
              games_per_training=10,
              training_steps_per_batch=20,
              save_frequency=50,
              save_path='models/alphazero'):
        """Main training loop"""

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'models', exist_ok=True)

        print("\n" + "=" * 70)
        print("ALPHAZERO TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total games: {num_games}")
        print(f"Games per training batch: {games_per_training}")
        print(f"Training steps per batch: {training_steps_per_batch}")
        print(f"MCTS simulations: {self.num_simulations}")
        print(f"Batch size: {self.batch_size}")
        print("=" * 70 + "\n")

        wins = 0
        total_examples = 0
        start_time = time.time()
        recent_wins = deque(maxlen=100)

        for game_num in range(1, num_games + 1):
            game_start = time.time()

            # Self-play
            winner, num_examples = self.self_play_game(verbose=False)
            total_examples += num_examples

            if winner == 0:
                wins += 1
                recent_wins.append(1)
            else:
                recent_wins.append(0)

            game_time = time.time() - game_start

            # Progress update
            if game_num % 5 == 0:
                elapsed = time.time() - start_time
                games_per_min = game_num / elapsed * 60
                win_rate = wins / game_num * 100
                recent_wr = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0

                print(f"Game {game_num:4d}/{num_games} | "
                      f"WR: {win_rate:5.1f}% (recent: {recent_wr:5.1f}%) | "
                      f"Buffer: {len(self.replay_buffer):6d} | "
                      f"Speed: {games_per_min:.1f} g/min | "
                      f"Last: {game_time:.1f}s")

            # Training phase
            if game_num % games_per_training == 0 and len(self.replay_buffer) >= self.batch_size:
                train_start = time.time()
                losses = self.train_batch(training_steps_per_batch)
                train_time = time.time() - train_start

                if losses:
                    avg_action = np.mean([l['action_loss'] for l in losses])
                    avg_vertex = np.mean([l['vertex_loss'] for l in losses])
                    avg_edge = np.mean([l['edge_loss'] for l in losses])
                    avg_value = np.mean([l['value_loss'] for l in losses])
                    print(f"  └─ Training: action={avg_action:.4f}, vertex={avg_vertex:.4f}, "
                          f"edge={avg_edge:.4f}, value={avg_value:.4f} ({train_time:.1f}s)")

            # Save checkpoint
            if game_num % save_frequency == 0:
                checkpoint_path = f"{save_path}_game_{game_num}.pt"
                self.save(checkpoint_path)
                print(f"  └─ Saved: {checkpoint_path}")

        # Final save
        final_path = f"{save_path}_final.pt"
        self.save(final_path)

        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total games: {num_games}")
        print(f"Final win rate: {wins/num_games*100:.1f}%")
        print(f"Total examples: {total_examples}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average speed: {num_games/total_time*60:.1f} games/min")
        print(f"Model saved to: {final_path}")
        print("=" * 70)

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_played': self.games_played,
            'training_steps': self.training_steps,
            'device': str(self.device)
        }, path)

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.games_played = checkpoint.get('games_played', 0)
        self.training_steps = checkpoint.get('training_steps', 0)
        print(f"Loaded checkpoint from {path}")


# Quick test
if __name__ == "__main__":
    print("Testing AlphaZero Trainer...")
    print("=" * 70)

    trainer = AlphaZeroTrainer(num_simulations=15)

    print("\n--- Playing 3 test games ---")
    for i in range(3):
        winner, examples = trainer.self_play_game(verbose=True)

    print(f"\nBuffer size: {len(trainer.replay_buffer)}")
    print("✅ AlphaZero Trainer test passed!")