"""
AlphaZero Training with Curriculum Learning

Improvements over base AlphaZero trainer:
1. Curriculum learning: Random → Mixed → Rule-based → Self-play
2. Increased MCTS simulations (100 default)
3. True self-play mode
4. Progressive difficulty ramp
5. Better training metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time
import random

from game_state import GameState
from mcts import MCTS
from network_wrapper import NetworkWrapper
from game_system import ResourceType


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


class OpponentPlayer:
    """Wrapper for different opponent types"""

    def __init__(self, opponent_type='random', randomness=1.0, mcts=None):
        """
        opponent_type: 'random', 'rule_based', 'mcts'
        randomness: 0.0 (smart) to 1.0 (random) for mixed strategies
        mcts: MCTS object for self-play
        """
        self.opponent_type = opponent_type
        self.randomness = randomness
        self.mcts = mcts

    def play_turn(self, state, player_id):
        """Play one turn for the opponent"""
        game = state.env.game_env.game
        player = game.players[player_id]

        # Initial placement phase
        if game.is_initial_placement_phase():
            return self._play_initial_placement(game, player)

        # Decide strategy based on opponent type
        if self.opponent_type == 'random' or random.random() < self.randomness:
            return self._play_random(game, player)
        elif self.opponent_type == 'rule_based':
            return self._play_rule_based(state, player_id)
        elif self.opponent_type == 'mcts' and self.mcts:
            return self._play_mcts(state)
        else:
            return self._play_random(game, player)

    def _play_initial_placement(self, game, player):
        """Handle initial placement phase"""
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

    def _play_random(self, game, player):
        """Fully random play"""
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

            if actions and random.random() < 0.7:  # 70% chance to build
                action_type, locs = random.choice(actions)
                if action_type == 'sett':
                    game.try_build_settlement(random.choice(locs), player)
                elif action_type == 'city':
                    player.try_build_city(random.choice(locs))
                elif action_type == 'road':
                    game.try_build_road(random.choice(locs), player)
                return True

        if game.can_end_turn():
            game.end_turn()
            return True

        return False

    def _play_rule_based(self, state, player_id):
        """Smart rule-based play"""
        from rule_based_ai import play_rule_based_turn
        return play_rule_based_turn(state.env, player_id)

    def _play_mcts(self, state):
        """MCTS-based play (self-play)"""
        best_action, _ = self.mcts.search(state)
        action_id, vertex_id, edge_id = best_action
        reward, done = state.apply_action(action_id, vertex_id, edge_id)
        return True


class ReplayBuffer:
    """Stores training examples from self-play games"""

    def __init__(self, max_size=500000):
        self.buffer = deque(maxlen=max_size)

    def add_batch(self, examples):
        """Add multiple examples at once"""
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


class AlphaZeroCurriculumTrainer:
    """
    AlphaZero trainer with curriculum learning
    Progressively increases opponent difficulty
    """

    def __init__(self,
                 model_path=None,
                 num_simulations=100,  # Increased from 50
                 c_puct=1.0,
                 learning_rate=1e-3,
                 batch_size=None,
                 buffer_size=500000):

        # Auto-detect device
        self.device = get_device()

        # Auto-set batch size based on device
        if batch_size is None:
            if self.device.type == 'cuda':
                batch_size = 8192  # Increased from 3072 - A100 can handle it!
            elif self.device.type == 'mps':
                batch_size = 256
            else:
                batch_size = 64
        self.batch_size = batch_size

        # Create network wrapper
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

        # Curriculum state
        self.current_phase = 0
        self.phase_games = 0

        print(f"Batch size: {self.batch_size}")
        print(f"MCTS simulations: {num_simulations}")

    def get_curriculum_opponent(self):
        """
        Get opponent based on curriculum phase
        Phase 0: 90% random (0-200 games)
        Phase 1: 75% random (200-400 games)
        Phase 2: 50% random (400-600 games)
        Phase 3: 25% random (600-800 games)
        Phase 4: Rule-based (800-1000 games)
        Phase 5+: Self-play (1000+ games)
        """
        if self.games_played < 200:
            phase = 0
            opponent = OpponentPlayer(opponent_type='random', randomness=0.9)
            phase_name = "Random (90%)"
        elif self.games_played < 400:
            phase = 1
            opponent = OpponentPlayer(opponent_type='random', randomness=0.75)
            phase_name = "Random (75%)"
        elif self.games_played < 600:
            phase = 2
            opponent = OpponentPlayer(opponent_type='random', randomness=0.5)
            phase_name = "Mixed (50%)"
        elif self.games_played < 800:
            phase = 3
            opponent = OpponentPlayer(opponent_type='random', randomness=0.25)
            phase_name = "Smart (75%)"
        elif self.games_played < 1000:
            phase = 4
            opponent = OpponentPlayer(opponent_type='rule_based', randomness=0.0)
            phase_name = "Rule-based"
        else:
            phase = 5
            opponent = OpponentPlayer(opponent_type='mcts', randomness=0.0, mcts=self.mcts)
            phase_name = "Self-play"

        # Track phase transitions
        if phase != self.current_phase:
            self.current_phase = phase
            self.phase_games = 0
            print(f"\n{'='*70}")
            print(f"CURRICULUM PHASE {phase}: {phase_name}")
            print(f"{'='*70}\n")

        self.phase_games += 1
        return opponent, phase_name

    def self_play_game(self, verbose=False):
        """Play one game using MCTS, collecting training data"""
        state = GameState()
        examples = []
        move_count = 0
        max_moves = 500

        # Get curriculum opponent
        opponent, phase_name = self.get_curriculum_opponent()

        while not state.is_terminal() and move_count < max_moves:
            current_player = state.get_current_player()

            if current_player == 0:
                # Player 0: MCTS
                move_count += 1
                obs = state.get_observation()

                # MCTS search
                best_action, action_probs_dict = self.mcts.search(state)
                action_probs_array = self._action_probs_to_array(action_probs_dict)

                examples.append({
                    'observation': obs['observation'].copy(),
                    'action_probs': action_probs_array,
                    'player': current_player
                })

                action_id, vertex_id, edge_id = best_action
                reward, done = state.apply_action(action_id, vertex_id, edge_id)
            else:
                # Opponents: Use curriculum opponent
                success = opponent.play_turn(state, current_player)
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
                'action_probs': ex['action_probs'],
                'value': discounted_value
            })

        self.replay_buffer.add_batch(training_examples)
        self.games_played += 1

        if verbose:
            vps = [state.get_victory_points(i) for i in range(4)]
            print(f"Game {self.games_played}: Winner={winner}, VPs={vps}, Examples={len(examples)}, Opponent={phase_name}")

        return winner, len(examples), phase_name

    def _action_probs_to_array(self, action_probs_dict):
        """Convert MCTS action probabilities to fixed-size array"""
        probs = np.zeros(11, dtype=np.float32)
        for (action_id, vertex_id, edge_id), prob in action_probs_dict.items():
            probs[action_id] += prob
        if probs.sum() > 0:
            probs = probs / probs.sum()
        return probs

    def train_step(self):
        """One training step with optional mixed precision"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        observations = torch.FloatTensor(batch['observations']).to(self.device)
        target_probs = torch.FloatTensor(batch['action_probs']).to(self.device)
        target_values = torch.FloatTensor(batch['values']).to(self.device)

        self.network.train()

        if self.use_amp:
            # Mixed precision training (CUDA)
            with torch.cuda.amp.autocast():
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
            # Regular training (MPS/CPU)
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

        self.scheduler.step()
        self.training_steps += 1

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }

    def train_batch(self, num_steps=10):
        """Multiple training steps"""
        losses = []
        for _ in range(num_steps):
            loss = self.train_step()
            if loss:
                losses.append(loss)
        return losses

    def train(self,
              num_games=2000,
              games_per_training=5,  # Train more frequently
              training_steps_per_batch=50,  # More training steps per batch
              save_frequency=100,
              save_path='models/alphazero_curriculum'):
        """Main training loop with curriculum"""

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'models', exist_ok=True)

        print("\n" + "=" * 70)
        print("ALPHAZERO CURRICULUM TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Total games: {num_games}")
        print(f"Games per training batch: {games_per_training}")
        print(f"Training steps per batch: {training_steps_per_batch}")
        print(f"MCTS simulations: {self.num_simulations}")
        print(f"Batch size: {self.batch_size}")
        print("\nCurriculum Phases:")
        print("  Phase 0 (0-200):     90% Random")
        print("  Phase 1 (200-400):   75% Random")
        print("  Phase 2 (400-600):   50% Random")
        print("  Phase 3 (600-800):   25% Random")
        print("  Phase 4 (800-1000):  Rule-based AI")
        print("  Phase 5 (1000+):     Self-play")
        print("=" * 70 + "\n")

        wins = 0
        total_examples = 0
        start_time = time.time()
        recent_wins = deque(maxlen=100)
        phase_wins = {i: 0 for i in range(6)}
        phase_games = {i: 0 for i in range(6)}

        for game_num in range(1, num_games + 1):
            game_start = time.time()

            # Self-play
            winner, num_examples, phase_name = self.self_play_game(verbose=False)
            total_examples += num_examples

            # Track wins by phase
            phase = self.current_phase
            phase_games[phase] = phase_games.get(phase, 0) + 1
            if winner == 0:
                wins += 1
                recent_wins.append(1)
                phase_wins[phase] = phase_wins.get(phase, 0) + 1
            else:
                recent_wins.append(0)

            game_time = time.time() - game_start

            # Progress update
            if game_num % 5 == 0:
                elapsed = time.time() - start_time
                games_per_min = game_num / elapsed * 60
                win_rate = wins / game_num * 100
                recent_wr = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
                phase_wr = phase_wins[phase] / phase_games[phase] * 100 if phase_games[phase] > 0 else 0

                print(f"Game {game_num:4d}/{num_games} | Phase: {phase} ({phase_name:12s}) | "
                      f"WR: {win_rate:5.1f}% (recent: {recent_wr:5.1f}%, phase: {phase_wr:5.1f}%) | "
                      f"Buffer: {len(self.replay_buffer):6d} | "
                      f"Speed: {games_per_min:.1f} g/min")

            # Training phase
            if game_num % games_per_training == 0 and len(self.replay_buffer) >= self.batch_size:
                train_start = time.time()
                losses = self.train_batch(training_steps_per_batch)
                train_time = time.time() - train_start

                if losses:
                    avg_policy = np.mean([l['policy_loss'] for l in losses])
                    avg_value = np.mean([l['value_loss'] for l in losses])
                    print(f"  └─ Training: policy={avg_policy:.4f}, value={avg_value:.4f} ({train_time:.1f}s)")

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
        print("\nWin rates by phase:")
        for phase in range(6):
            if phase_games[phase] > 0:
                wr = phase_wins[phase] / phase_games[phase] * 100
                print(f"  Phase {phase}: {wr:.1f}% ({phase_wins[phase]}/{phase_games[phase]})")
        print(f"Model saved to: {final_path}")
        print("=" * 70)

    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'games_played': self.games_played,
            'training_steps': self.training_steps,
            'current_phase': self.current_phase,
            'device': str(self.device)
        }, path)

    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.games_played = checkpoint.get('games_played', 0)
        self.training_steps = checkpoint.get('training_steps', 0)
        self.current_phase = checkpoint.get('current_phase', 0)
        print(f"Loaded checkpoint from {path}")
        print(f"  Games played: {self.games_played}")
        print(f"  Current phase: {self.current_phase}")


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AlphaZero Curriculum Training for Catan')
    parser.add_argument('--num-games', type=int, default=2000, help='Total games to play')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations per move')
    parser.add_argument('--batch-size', type=int, default=None, help='Training batch size (auto if None)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save-freq', type=int, default=100, help='Save checkpoint every N games')
    parser.add_argument('--model-path', type=str, default=None, help='Path to load existing model')
    parser.add_argument('--save-path', type=str, default='models/alphazero_curriculum',
                        help='Path to save models')

    args = parser.parse_args()

    print("Starting AlphaZero Curriculum Training...")
    print("=" * 70)

    trainer = AlphaZeroCurriculumTrainer(
        model_path=args.model_path,
        num_simulations=args.simulations,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

    trainer.train(
        num_games=args.num_games,
        games_per_training=5,  # Train more frequently with GPU power
        training_steps_per_batch=50,  # More steps per batch
        save_frequency=args.save_freq,
        save_path=args.save_path
    )
