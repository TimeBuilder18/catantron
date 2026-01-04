"""
Imitation Learning + PPO Fine-Tuning Trainer

TWO-PHASE APPROACH:
Phase 1: Imitation Learning (Behavioral Cloning)
  - Watch rule-based AI play games
  - Train network to predict expert actions
  - Result: Immediate competent baseline

Phase 2: PPO Fine-Tuning
  - Use pre-trained model
  - Train against progressively harder opponents
  - Result: Surpass the teacher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import os
import time
from catan_env_pytorch import CatanEnv
from network_wrapper import NetworkWrapper
from rule_based_ai import play_rule_based_turn
from curriculum_trainer_v2_fixed import play_opponent_turn, get_device


class ImitationLearningBuffer:
    """Stores expert demonstrations for supervised learning"""
    
    def __init__(self):
        self.demonstrations = []
    
    def add_demo(self, observation, action_id, vertex_id=None, edge_id=None):
        """Add expert demonstration"""
        self.demonstrations.append({
            'obs': observation.copy(),
            'action_id': action_id,
            'vertex_id': vertex_id if vertex_id is not None else 0,
            'edge_id': edge_id if edge_id is not None else 0
        })
    
    def sample(self, batch_size):
        """Sample random batch"""
        indices = np.random.choice(len(self.demonstrations), 
                                   min(batch_size, len(self.demonstrations)), 
                                   replace=False)
        
        batch = {
            'observations': np.array([self.demonstrations[i]['obs'] for i in indices]),
            'action_ids': np.array([self.demonstrations[i]['action_id'] for i in indices]),
            'vertex_ids': np.array([self.demonstrations[i]['vertex_id'] for i in indices]),
            'edge_ids': np.array([self.demonstrations[i]['edge_id'] for i in indices])
        }
        return batch
    
    def __len__(self):
        return len(self.demonstrations)


class ImitationPPOTrainer:
    """Two-phase trainer: Imitation → PPO"""
    
    def __init__(self, batch_size=4096, learning_rate=1e-3):
        self.device = get_device()
        self.batch_size = batch_size
        
        # Create network
        self.network_wrapper = NetworkWrapper(device=str(self.device))
        self.network = self.network_wrapper.policy
        
        # Optimizer for both phases
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Buffers
        self.demo_buffer = ImitationLearningBuffer()  # For imitation
        
        print(f"Device: {self.device}")
        print(f"Batch size: {batch_size}")
    
    
    # ==================== PHASE 1: IMITATION LEARNING ====================
    
    def collect_demonstrations(self, num_games=1000):
        """
        PHASE 1A: Collect expert demonstrations
        - Play games with rule-based AI
        - Record all (state, action) pairs
        """
        print("\n" + "="*70)
        print("PHASE 1A: COLLECTING EXPERT DEMONSTRATIONS")
        print("="*70)
        print(f"Playing {num_games} games with rule-based AI...")
        print()
        
        env = CatanEnv(player_id=0)
        demos_collected = 0
        start_time = time.time()
        
        for game_num in range(1, num_games + 1):
            obs, _ = env.reset()
            done = False
            move_count = 0
            max_moves = 500
            
            while not done and move_count < max_moves:
                game = env.game_env.game
                current_player_id = game.players.index(game.get_current_player())
                
                if current_player_id == 0:
                    # Get expert action from rule-based AI
                    action_id, vertex_id, edge_id = self._get_expert_action(env, obs)
                    
                    if action_id is not None:
                        # Store demonstration
                        self.demo_buffer.add_demo(
                            obs['observation'],
                            action_id,
                            vertex_id,
                            edge_id
                        )
                        demos_collected += 1
                        
                        # Execute action
                        obs, reward, terminated, truncated, info = env.step(
                            action_id, vertex_id, edge_id,
                            trade_give_idx=0, trade_get_idx=0
                        )
                        done = terminated or truncated
                    else:
                        break
                    
                    move_count += 1
                else:
                    # Other players: random opponents
                    play_opponent_turn(game, current_player_id, random_prob=1.0)
                    
                    # Check if game ended
                    if game.check_victory_conditions() is not None:
                        done = True
            
            # Progress update
            if game_num % 100 == 0:
                elapsed = time.time() - start_time
                speed = game_num / elapsed * 60
                print(f"  Game {game_num:4d}/{num_games} | "
                      f"Demos: {demos_collected:6d} | "
                      f"Speed: {speed:.1f} g/min")
        
        total_time = time.time() - start_time
        print(f"\n  Complete! Collected {demos_collected:,} demonstrations")
        print(f"  Time: {total_time/60:.1f} minutes")
        print("="*70)
        
        return demos_collected
    
    def _get_expert_action(self, env, obs):
        """
        Get action from rule-based AI logic

        Uses same priority as rule_based_ai.py:
        1. Build city (if affordable) → action_id=4
        2. Build settlement (if affordable) → action_id=3
        3. Build road (if affordable) → action_id=5
        4. Buy dev card (if affordable) → action_id=6
        5. Trade (if beneficial) → action_id=9
        6. End turn → action_id=7
        """
        from game_system import ResourceType
        import random

        game = env.game_env.game
        player = game.players[env.player_id]
        resources = player.resources

        action_mask = obs['action_mask']
        vertex_mask = obs.get('vertex_mask', [])
        edge_mask = obs.get('edge_mask', [])

        # Priority 1: Build city (2 wheat + 3 ore) = 2 VP!
        if action_mask[4] == 1:  # build_city is valid
            if (resources[ResourceType.WHEAT] >= 2 and
                resources[ResourceType.ORE] >= 3):
                valid_vertices = [i for i, m in enumerate(vertex_mask) if m == 1]
                if valid_vertices:
                    return 4, random.choice(valid_vertices), 0

        # Priority 2: Build settlement (1 wood + 1 brick + 1 wheat + 1 sheep) = 1 VP
        if action_mask[3] == 1:  # build_settlement is valid
            if (resources[ResourceType.WOOD] >= 1 and
                resources[ResourceType.BRICK] >= 1 and
                resources[ResourceType.WHEAT] >= 1 and
                resources[ResourceType.SHEEP] >= 1):
                valid_vertices = [i for i, m in enumerate(vertex_mask) if m == 1]
                if valid_vertices:
                    return 3, random.choice(valid_vertices), 0

        # Priority 3: Build road (1 wood + 1 brick) = Progress
        if action_mask[5] == 1:  # build_road is valid
            if (resources[ResourceType.WOOD] >= 1 and
                resources[ResourceType.BRICK] >= 1):
                valid_edges = [i for i, m in enumerate(edge_mask) if m == 1]
                if valid_edges:
                    return 5, 0, random.choice(valid_edges)

        # Priority 4: Buy development card (1 wheat + 1 sheep + 1 ore)
        if action_mask[6] == 1:  # buy_dev_card is valid
            if (resources[ResourceType.WHEAT] >= 1 and
                resources[ResourceType.SHEEP] >= 1 and
                resources[ResourceType.ORE] >= 1):
                return 6, 0, 0

        # Priority 5: Trade if we have 4+ of one resource
        if action_mask[9] == 1:  # trade_with_bank is valid
            for resource_type, amount in resources.items():
                if amount >= 4:
                    # Simplified trade: give excess resource, get what we need most
                    return 9, 0, 0  # Note: trade indices handled separately

        # Priority 6: End turn if nothing else to do
        if action_mask[7] == 1:  # end_turn is valid
            return 7, 0, 0

        # Fallback: Roll dice or wait
        if action_mask[0] == 1:  # roll_dice
            return 0, 0, 0
        if action_mask[8] == 1:  # wait
            return 8, 0, 0

        # Last resort: return first valid action
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        if valid_actions:
            return valid_actions[0], 0, 0

        return None, None, None
    
    def train_imitation(self, num_epochs=10, steps_per_epoch=100):
        """
        PHASE 1B: Train network via supervised learning
        - Network learns to predict expert actions
        - Uses cross-entropy loss
        """
        print("\n" + "="*70)
        print("PHASE 1B: IMITATION LEARNING (SUPERVISED)")
        print("="*70)
        print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps each")
        print()
        
        for epoch in range(1, num_epochs + 1):
            epoch_losses = []
            epoch_accuracies = []
            
            for step in range(steps_per_epoch):
                loss, accuracy = self._imitation_train_step()
                if loss is not None:
                    epoch_losses.append(loss)
                    epoch_accuracies.append(accuracy)
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                avg_acc = np.mean(epoch_accuracies)
                print(f"  Epoch {epoch:2d}/{num_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Accuracy: {avg_acc:5.1f}%")
        
        print("\n  Imitation learning complete!")
        print("  Network now imitates rule-based AI")
        print("="*70)
    
    def _imitation_train_step(self):
        """Single imitation learning training step"""
        if len(self.demo_buffer) < self.batch_size:
            return None, None
        
        # Sample batch
        batch = self.demo_buffer.sample(self.batch_size)
        
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        target_actions = torch.LongTensor(batch['action_ids']).to(self.device)
        
        self.network.train()
        
        # Forward pass
        action_probs, _, _, _, _, _ = self.network.forward(observations)
        
        # Cross-entropy loss (classification)
        loss = F.cross_entropy(torch.log(action_probs + 1e-8), target_actions)
        
        # Calculate accuracy
        predicted_actions = torch.argmax(action_probs, dim=1)
        accuracy = (predicted_actions == target_actions).float().mean().item() * 100
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item(), accuracy
    
    
    # ==================== PHASE 2: PPO FINE-TUNING ====================
    
    def finetune_with_ppo(self, games_per_phase=2000):
        """
        PHASE 2: PPO Fine-Tuning
        - Start from imitation-learned model
        - Train against progressively harder opponents
        - Use PPO to surpass the teacher
        """
        print("\n" + "="*70)
        print("PHASE 2: PPO FINE-TUNING")
        print("="*70)
        print("Now training with PPO to surpass rule-based AI...")
        print()
        
        # Import and use existing PPO trainer
        from curriculum_trainer_v2_fixed import CurriculumTrainerV2
        
        # Save current model
        temp_path = 'models/imitation_pretrained.pt'
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': self.network.state_dict(),
        }, temp_path)
        
        # Create PPO trainer with pre-trained model
        ppo_trainer = CurriculumTrainerV2(
            model_path=temp_path,
            batch_size=self.batch_size,
            learning_rate=1e-4,  # Lower LR for fine-tuning
            reward_mode='pbrs_fixed'
        )
        
        # Train with curriculum
        ppo_trainer.train(
            games_per_phase=games_per_phase,
            train_frequency=10,  # Conservative
            train_steps=10
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
    
    def save(self, path):
        """Save model"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"  Saved: {path}")


# ==================== MAIN TRAINING PIPELINE ====================

def train_full_pipeline(demo_games=1000, imitation_epochs=10, ppo_games_per_phase=2000, batch_size=8192):
    """
    Complete two-phase training pipeline

    Args:
        demo_games: Number of games to collect demonstrations
        imitation_epochs: Epochs of supervised learning
        ppo_games_per_phase: Games per PPO curriculum phase
        batch_size: Training batch size (higher = more GPU usage)
    """
    print("\n" + "="*70)
    print("IMITATION LEARNING + PPO TRAINING PIPELINE")
    print("="*70)
    print("This will:")
    print("  1. Collect expert demonstrations from rule-based AI")
    print("  2. Train network via supervised learning (imitation)")
    print("  3. Fine-tune with PPO against harder opponents")
    print("="*70 + "\n")
    
    # Create trainer
    trainer = ImitationPPOTrainer(batch_size=batch_size, learning_rate=1e-3)
    
    # Phase 1A: Collect demonstrations
    demos = trainer.collect_demonstrations(num_games=demo_games)
    
    # Phase 1B: Imitation learning
    trainer.train_imitation(num_epochs=imitation_epochs, steps_per_epoch=100)
    
    # Save imitation model
    trainer.save('models/imitation_only.pt')
    
    # Phase 2: PPO fine-tuning
    trainer.finetune_with_ppo(games_per_phase=ppo_games_per_phase)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo-games', type=int, default=1000,
                        help='Number of demonstration games to collect')
    parser.add_argument('--imitation-epochs', type=int, default=10,
                        help='Epochs of supervised learning')
    parser.add_argument('--ppo-games', type=int, default=2000,
                        help='Games per phase in PPO fine-tuning')
    parser.add_argument('--batch-size', type=int, default=8192,
                        help='Training batch size (8192/16384/24576 for RTX 2080 Super)')
    args = parser.parse_args()

    train_full_pipeline(
        demo_games=args.demo_games,
        imitation_epochs=args.imitation_epochs,
        ppo_games_per_phase=args.ppo_games,
        batch_size=args.batch_size
    )
